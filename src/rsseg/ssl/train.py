# SPDX-License-Identifier: AGPL-3.0-or-later
"""DINO pre-training entry point: fit, monitor collapse, export the student encoder."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import keras
import mlflow
import numpy as np

from rsseg import progress
from rsseg.config import ExperimentConfig
from rsseg.data.datasets import MultiCropDataset, read_crops, training_loader
from rsseg.data.sources import get_source
from rsseg.data.transforms import MultiCropAugment
from rsseg.data.unlabeled import read_pool_meta
from rsseg.models.backbones import NO_WEIGHT_DECAY
from rsseg.runtime import setup_runtime
from rsseg.ssl.dino import DINOModel
from rsseg.tracking import GPUMemoryLogger, MLflowMetricsLogger, mlflow_run
from rsseg.train import resume as resume_mod
from rsseg.train.callbacks import ProgressReporter, RichProgressBar
from rsseg.train.seg_train import make_run_dir

log = logging.getLogger(__name__)


class CollapseMonitor(keras.callbacks.Callback):
    """Log teacher-output entropy on a fixed batch: it must stay well above 0."""

    def __init__(self, global_views: np.ndarray):
        super().__init__()
        self.global_views = global_views

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["teacher_entropy"] = self.model.teacher_entropy(self.global_views)


def pool_rows(cfg: ExperimentConfig) -> list[dict]:
    """Crops of the unlabelled pools ``ssl.extra_pools``, with absolute image paths.

    A pool from the city of the benchmark (e.g. the Vaihingen frames with ``data.dataset:
    vaihingen``) also covers its test areas: it is refused unless ``ssl.allow_same_city_pool``.
    """
    city = get_source(cfg.data.dataset).city
    rows: list[dict] = []
    for pool in cfg.ssl.extra_pools:
        meta = read_pool_meta(pool)
        if meta["city"] == city and not cfg.ssl.allow_same_city_pool:
            raise ValueError(
                f"pool {pool} comes from {city}, the city of the evaluation data: its images "
                "cover the test areas. Set ssl.allow_same_city_pool=true to use it anyway."
            )
        root = Path(pool).resolve()
        rows += [{**r, "image": str(root / r["image"])} for r in read_crops(pool, "unlabeled")]
    return rows


def pretrain_dino(cfg: ExperimentConfig, resume: str | Path | None = None) -> dict:
    """DINO pre-training of the encoder. ``resume`` continues a paused run (its directory)."""
    s = cfg.ssl
    state = resume_mod.load(resume) if resume else None
    precision = setup_runtime(s.precision, s.seed)
    run_dir = Path(resume) if resume else make_run_dir(cfg, "dino")
    progress.report(run_dir=str(Path(run_dir).resolve()), kind="dino", name=cfg.name)

    # Only training tiles are used (no labels are read): val/test stay untouched.
    rows = read_crops(cfg.data.root, "train") + pool_rows(cfg)
    log.info("DINO pre-training on %d crops (%d pools)", len(rows), len(s.extra_pools))
    augment = MultiCropAugment(
        s.global_crop_size, s.local_crop_size, s.local_crops_number,
        tuple(s.global_crops_scale), tuple(s.local_crops_scale), aerial=s.aerial_augment,
    )  # fmt: skip
    loader = training_loader(
        MultiCropDataset(cfg.data.root, rows, augment), s.batch_size, cfg.data.num_workers, s.seed
    )
    stream = loader.sampler
    steps = s.steps_per_epoch or stream.batches_per_pass

    model = DINOModel(
        dataclasses.replace(s, weights="none") if resume else s, steps_per_epoch=steps
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0, weight_decay=s.weight_decay, global_clipnorm=s.clip_grad or None
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", *NO_WEIGHT_DECAY])
    model.compile(optimizer=optimizer)

    ref_globals = np.asarray(next(iter(loader))[0][:8])
    export_path = s.export_path or str(run_dir / "backbone.weights.h5")

    metrics_logger = MLflowMetricsLogger()
    reporter = ProgressReporter("pretrain", monitor=None, epochs=s.epochs, steps=steps)
    checkpoint = resume_mod.ResumeCheckpoint(
        run_dir, "dino", stream, s.epochs, steps, s.checkpoint_every_min,
        {"mlflow": metrics_logger, "progress": reporter}, state,
    )  # fmt: skip
    callbacks = [
        CollapseMonitor(ref_globals),
        GPUMemoryLogger(),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=bool(resume)),
        metrics_logger,
        RichProgressBar(keys=("loss", "lr", "teacher_temp"), epochs=s.epochs),
        reporter,
        checkpoint,  # last: see ResumeCheckpoint
    ]

    with mlflow_run(cfg, "dino", run_dir, state["run_id"] if state else None) as run:
        checkpoint.run_id = run.info.run_id
        if not resume:
            mlflow.log_params(
                {
                    "runtime.precision": precision,
                    "runtime.ssl_steps_per_epoch": steps,  # ssl.steps_per_epoch (may be null) is in the config
                    "model.params": model.student.count_params(),
                }
            )
        resume_mod.fit_resumable(
            model, loader, s.epochs, steps, callbacks, checkpoint, reporter,
            shuffle=False,  # the stream sampler already shuffles
            verbose=0,  # progress is shown by RichProgressBar
        )  # fmt: skip
        model.student_backbone.save_weights(export_path)
        mlflow.log_artifact(export_path, artifact_path="backbone")
        result = {
            "run_id": run.info.run_id,
            "run_dir": str(run_dir),
            "backbone_weights": export_path,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    resume_mod.discard(run_dir)
    log.info(
        "student backbone exported to %s — use it with model.weights=%s", export_path, export_path
    )
    return result
