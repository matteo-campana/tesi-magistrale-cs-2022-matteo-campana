# SPDX-License-Identifier: AGPL-3.0-or-later
"""DINO pre-training entry point: fit, monitor collapse, export the student encoder."""

from __future__ import annotations

import json
import logging

import keras
import mlflow
import numpy as np

from rsseg.config import ExperimentConfig
from rsseg.data.datasets import MultiCropDataset, make_loader, read_crops
from rsseg.data.transforms import MultiCropAugment
from rsseg.models.backbones import NO_WEIGHT_DECAY
from rsseg.runtime import setup_runtime
from rsseg.ssl.dino import DINOModel
from rsseg.tracking import GPUMemoryLogger, MLflowMetricsLogger, mlflow_run
from rsseg.train.callbacks import RichProgressBar
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


def pretrain_dino(cfg: ExperimentConfig) -> dict:
    s = cfg.ssl
    precision = setup_runtime(s.precision, s.seed)
    run_dir = make_run_dir(cfg, "dino")

    # Only training tiles are used (no labels are read): val/test stay untouched.
    rows = read_crops(cfg.data.root, "train")
    augment = MultiCropAugment(
        s.global_crop_size, s.local_crop_size, s.local_crops_number,
        tuple(s.global_crops_scale), tuple(s.local_crops_scale), aerial=s.aerial_augment,
    )  # fmt: skip
    loader = make_loader(
        MultiCropDataset(cfg.data.root, rows, augment),
        s.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(f"batch_size {s.batch_size} larger than the {len(rows)} training crops")
    steps = s.steps_per_epoch or len(loader)

    model = DINOModel(s, steps_per_epoch=steps)
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0, weight_decay=s.weight_decay, global_clipnorm=s.clip_grad or None
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", *NO_WEIGHT_DECAY])
    model.compile(optimizer=optimizer)

    ref_globals = np.asarray(next(iter(loader))[0][:8])
    export_path = s.export_path or str(run_dir / "backbone.weights.h5")

    with mlflow_run(cfg, "dino", run_dir) as run:
        mlflow.log_params(
            {
                "runtime.precision": precision,
                "ssl.steps_per_epoch": steps,
                "model.params": model.student.count_params(),
            }
        )
        model.fit(
            loader,
            epochs=s.epochs,
            steps_per_epoch=s.steps_per_epoch,
            callbacks=[
                CollapseMonitor(ref_globals),
                GPUMemoryLogger(),
                keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
                MLflowMetricsLogger(),
                RichProgressBar(keys=("loss", "lr", "teacher_temp")),
            ],
            shuffle=False,  # the DataLoader already shuffles
            verbose=0,  # progress is shown by RichProgressBar
        )
        model.student_backbone.save_weights(export_path)
        mlflow.log_artifact(export_path, artifact_path="backbone")
        result = {
            "run_id": run.info.run_id,
            "run_dir": str(run_dir),
            "backbone_weights": export_path,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(
        "student backbone exported to %s — use it with model.weights=%s", export_path, export_path
    )
    return result
