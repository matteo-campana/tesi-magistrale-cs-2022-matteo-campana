# SPDX-License-Identifier: AGPL-3.0-or-later
"""Supervised training of the segmentation network with Keras ``fit`` and MLflow tracking."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import keras
import mlflow
import numpy as np

from rsseg import progress
from rsseg.config import ExperimentConfig, ModelConfig, TrainConfig
from rsseg.control import PauseRequested
from rsseg.data.datasets import segmentation_loaders
from rsseg.eval.report import evaluate_tiles, report_metrics, write_report
from rsseg.models.backbones import NO_WEIGHT_DECAY
from rsseg.models.segmentor import build_segmentor
from rsseg.runstate import resume_dir
from rsseg.runtime import setup_runtime
from rsseg.tracking import GPUMemoryLogger, MLflowMetricsLogger, mlflow_run
from rsseg.train import resume as resume_mod
from rsseg.train.callbacks import (
    CodebookMonitor,
    PredictionLogger,
    ProgressReporter,
    RichProgressBar,
    TopKCheckpoint,
)
from rsseg.train.losses import MaskedSparseCrossEntropy
from rsseg.train.metrics import segmentation_metrics

log = logging.getLogger(__name__)


def make_run_dir(cfg: ExperimentConfig, kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(cfg.output_dir) / f"{kind}-{cfg.name}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_config(cfg: ExperimentConfig, require_stats: bool = True) -> ModelConfig:
    """``cfg.model`` with the input channels implied by ``cfg.data``: with ``data.ndsm`` the
    model takes 4 channels and normalises the nDSM with the training statistics of ``stats.json``.
    """
    m = cfg.model
    if not cfg.data.ndsm:
        if m.in_channels != 3:
            raise ValueError("model.in_channels > 3 requires data.ndsm=true")
        return m
    mean, std = m.extra_channel_mean, m.extra_channel_std
    stats_path = Path(cfg.data.root) / "stats.json"
    if (mean is None or std is None) and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if "ndsm_mean" in stats:
            mean, std = [stats["ndsm_mean"]], [stats["ndsm_std"]]
    if (mean is None or std is None) and require_stats:
        raise FileNotFoundError(
            f"no nDSM statistics in {stats_path}: run `rsseg prepare` with data.ndsm=true"
        )
    return dataclasses.replace(m, in_channels=4, extra_channel_mean=mean, extra_channel_std=std)


def lr_at_epoch(cfg: TrainConfig, epoch: int) -> float:
    if cfg.scheduler == "step":
        return cfg.lr * cfg.gamma ** (epoch // cfg.step_size)
    if cfg.scheduler == "cosine":
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (
            1 + math.cos(math.pi * epoch / max(cfg.epochs, 1))
        )
    if cfg.scheduler == "constant":
        return cfg.lr
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


def make_optimizer(cfg: TrainConfig) -> keras.optimizers.Optimizer:
    kwargs = {"learning_rate": cfg.lr}
    if cfg.grad_accum_steps > 1:
        kwargs["gradient_accumulation_steps"] = cfg.grad_accum_steps
    if cfg.optimizer == "adamw":
        opt = keras.optimizers.AdamW(weight_decay=cfg.weight_decay, **kwargs)
        opt.exclude_from_weight_decay(
            var_names=["bias", "gamma", "beta", "codebook", *NO_WEIGHT_DECAY]
        )
        return opt
    if cfg.optimizer == "adam":
        return keras.optimizers.Adam(**kwargs)
    if cfg.optimizer == "sgd":
        return keras.optimizers.SGD(momentum=0.9, nesterov=True, **kwargs)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def train_segmentation(cfg: ExperimentConfig, resume: str | Path | None = None) -> dict:
    """Train, then evaluate on whole tiles. ``resume`` continues a paused run (its directory);
    ``cfg`` must then be the configuration saved in it (``<run>/config.yaml``)."""
    state = resume_mod.load(resume) if resume else None
    cfg = dataclasses.replace(cfg, model=model_config(cfg))  # recorded in the run config
    precision = setup_runtime(cfg.train.precision, cfg.train.seed)
    run_dir = Path(resume) if resume else make_run_dir(cfg, "seg")
    progress.report(run_dir=str(run_dir), kind="seg", name=cfg.name)
    train_loader, val_loader = segmentation_loaders(cfg.data, cfg.train.batch_size, cfg.train.seed)
    stream = train_loader.sampler
    steps = cfg.train.steps_per_epoch or stream.batches_per_pass

    # A resumed model gets every variable from the checkpoint: skip the pre-trained weights.
    model = build_segmentor(dataclasses.replace(cfg.model, weights="none") if resume else cfg.model)
    model.compile(
        optimizer=make_optimizer(cfg.train),
        loss=MaskedSparseCrossEntropy(cfg.model.num_classes, class_weights=cfg.train.class_weights),
        metrics=segmentation_metrics(cfg.model.num_classes),
    )

    ref_images, ref_masks = next(iter(val_loader))
    ref_images = np.asarray(ref_images[:4])
    ref_masks = np.asarray(ref_masks[:4])
    best_path = run_dir / "best.weights.h5"

    top_k = TopKCheckpoint(run_dir / "checkpoints", best_path, k=cfg.train.keep_top_k)
    metrics_logger = MLflowMetricsLogger()
    reporter = ProgressReporter("train", "val_miou", cfg.train.early_stopping_patience,
                                epochs=cfg.train.epochs, steps=steps)  # fmt: skip
    tracked = {"top_k": top_k, "mlflow": metrics_logger, "progress": reporter}
    callbacks = [
        keras.callbacks.LearningRateScheduler(lambda epoch: lr_at_epoch(cfg.train, epoch)),
        top_k,
        CodebookMonitor(),
        GPUMemoryLogger(),
        PredictionLogger(
            ref_images, ref_masks, run_dir / "predictions", every=cfg.train.log_images_every
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=bool(resume)),
        metrics_logger,
        RichProgressBar(keys=("loss", "miou"), epochs=cfg.train.epochs),
        reporter,
    ]
    if cfg.train.early_stopping_patience:
        early = keras.callbacks.EarlyStopping(
            monitor="val_miou", mode="max", patience=cfg.train.early_stopping_patience
        )
        callbacks.insert(0, early)
        tracked["early_stopping"] = early
    checkpoint = resume_mod.ResumeCheckpoint(
        run_dir, "seg", stream, cfg.train.epochs, steps, cfg.train.checkpoint_every_min,
        tracked, state if state and state["phase"] == "train" else None,
    )  # fmt: skip
    callbacks.append(checkpoint)  # last: see ResumeCheckpoint
    if state and state["phase"] != "train":  # paused during the evaluation
        checkpoint.active_before = state.get("active_s", 0.0)
        progress.report(active_before_s=checkpoint.active_before)
    eval_state = resume_dir(run_dir) / "evaluation.json"

    with mlflow_run(cfg, "seg", run_dir, state["run_id"] if state else None) as run:
        checkpoint.run_id = run.info.run_id
        if not resume:
            mlflow.log_params(
                {"runtime.precision": precision, "model.params": model.count_params()}
            )
            (run_dir / "model_summary.txt").write_text(_summary(model), encoding="utf-8")
            mlflow.log_artifact(str(run_dir / "model_summary.txt"))

        if state is None or state["phase"] == "train":
            resume_mod.fit_resumable(
                model, train_loader, cfg.train.epochs, steps, callbacks, checkpoint, reporter,
                validation_data=val_loader, validation_steps=cfg.train.validation_steps,
                shuffle=False,  # the stream sampler already shuffles
                verbose=0,  # progress is shown by RichProgressBar
            )  # fmt: skip
            history = reporter.history
            val = [h["val_miou"] for h in history if "val_miou" in h]
            trained = {"best_val_miou": max(val) if val else float("nan"),
                       "epochs_trained": len(history)}  # fmt: skip
            mlflow.log_metric("best_val_miou", trained["best_val_miou"])
            if best_path.exists():
                mlflow.log_artifact(str(best_path), artifact_path="model")
                mlflow.log_artifact(
                    str(run_dir / "checkpoints" / "topk.json"), artifact_path="model"
                )
            mlflow.log_metric("epochs_trained", trained["epochs_trained"])
        else:
            trained = state["trained"]
            model.optimizer.build(model.trainable_variables)  # best.weights.h5 includes it
        if best_path.exists():
            model.load_weights(str(best_path))
        result: dict = {
            "run_id": run.info.run_id,
            "run_dir": str(run_dir),
            "best_val_miou": trained["best_val_miou"],
            "weights": str(best_path),
        }

        if cfg.eval.split:
            try:
                report = evaluate_tiles(
                    model, cfg.data.root, cfg.eval, out_dir=run_dir, state_path=eval_state
                )
            except PauseRequested:
                resume_mod.save_phase(run_dir, {
                    "kind": "seg", "phase": "evaluation", "run_id": run.info.run_id,
                    "trained": trained, "epoch": cfg.train.epochs, "batch": 0,
                    "epochs": cfg.train.epochs, "active_s": checkpoint.active_seconds(),
                })  # fmt: skip
                raise
            report["name"] = cfg.name
            json_path, md_path = write_report(report, run_dir, f"{cfg.name} — {cfg.eval.split}")
            mlflow.log_metrics(report_metrics(report, cfg.eval.split))
            mlflow.log_artifact(str(json_path), artifact_path="evaluation")
            mlflow.log_artifact(str(md_path), artifact_path="evaluation")
            if (run_dir / "predictions").exists():
                mlflow.log_artifacts(str(run_dir / "predictions"), artifact_path="predictions")
            result["test"] = report["all_classes"]
            if "eroded" in report:
                result["test_eroded"] = report["eroded"]["all_classes"]
        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    resume_mod.discard(run_dir)
    return result


def _summary(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lambda s, **_: lines.append(s), expand_nested=False)
    return "\n".join(lines)


def evaluate_segmentation(cfg: ExperimentConfig, weights: str) -> dict:
    """Evaluate saved weights on whole tiles of ``cfg.eval.split`` and log the report to MLflow."""
    cfg = dataclasses.replace(cfg, model=model_config(cfg))
    setup_runtime(cfg.train.precision, cfg.train.seed)
    run_dir = make_run_dir(cfg, "eval")
    # The encoder weights come from the checkpoint: skip downloading ImageNet weights.
    model = build_segmentor(dataclasses.replace(cfg.model, weights="none"))
    model.load_weights(weights)
    progress.report(run_dir=str(run_dir), kind="eval", name=cfg.name)
    with mlflow_run(cfg, "eval", run_dir) as run:
        mlflow.log_param("eval.weights", weights)
        report = evaluate_tiles(model, cfg.data.root, cfg.eval, out_dir=run_dir)
        report["name"] = cfg.name
        json_path, md_path = write_report(report, run_dir, f"{cfg.name} — {cfg.eval.split}")
        mlflow.log_metrics(report_metrics(report, cfg.eval.split))
        mlflow.log_artifact(str(json_path), artifact_path="evaluation")
        mlflow.log_artifact(str(md_path), artifact_path="evaluation")
        if (run_dir / "predictions").exists():
            mlflow.log_artifacts(str(run_dir / "predictions"), artifact_path="predictions")
    result = {"run_id": run.info.run_id, "run_dir": str(run_dir), **report["all_classes"]}
    if "eroded" in report:
        result["eroded"] = report["eroded"]["all_classes"]
    return result
