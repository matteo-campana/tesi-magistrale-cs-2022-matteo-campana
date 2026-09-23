# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLflow integration: run lifecycle, parameters, environment info and Keras callbacks."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import keras
import mlflow
import numpy as np

from rsseg.config import ExperimentConfig, save_config

log = logging.getLogger(__name__)
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict[str, str]:
    import torch

    info = {
        "python": platform.python_version(),
        "keras": keras.__version__,
        "torch": torch.__version__,
        "keras_backend": keras.backend.backend(),
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = f"{props.total_memory / 2**30:.1f}"
    if commit := _git_commit():
        info["git_commit"] = commit
    return info


@contextmanager
def mlflow_run(cfg: ExperimentConfig, kind: str, run_dir: Path):
    """Start an MLflow run, log config/params/environment, yield the active run."""
    uri = cfg.tracking.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.tracking.experiment)
    run_name = cfg.tracking.run_name or f"{kind}-{cfg.name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "kind": kind,
                **cfg.tracking.tags,
                **{f"env.{k}": v for k, v in environment_info().items()},
            }
        )
        params = {
            k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in cfg.flat().items()
        }
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        save_config(cfg, run_dir / "config.yaml")
        mlflow.log_artifact(str(run_dir / "config.yaml"))
        log.info("MLflow run %s (%s) — tracking URI %s", run.info.run_id, run_name, uri)
        yield run


def _to_float(value: Any) -> float | None:
    try:
        f = float(np.asarray(keras.ops.convert_to_numpy(value)))
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class MLflowMetricsLogger(keras.callbacks.Callback):
    """Log every epoch's logs (train and ``val_*``) plus learning rate to the active MLflow run."""

    def __init__(self, log_every_n_steps: int = 0):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step = 0

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        if self.log_every_n_steps and self._step % self.log_every_n_steps == 0 and logs:
            metrics = {
                f"step/{k}": v
                for k, v in ((k, _to_float(v)) for k, v in logs.items())
                if v is not None
            }
            mlflow.log_metrics(metrics, step=self._step)

    def on_epoch_end(self, epoch, logs=None):
        metrics = {
            k: v for k, v in ((k, _to_float(v)) for k, v in (logs or {}).items()) if v is not None
        }
        lr = _to_float(self.model.optimizer.learning_rate)
        if lr is not None:
            metrics["lr"] = lr
        mlflow.log_metrics(metrics, step=epoch)


class GPUMemoryLogger(keras.callbacks.Callback):
    """Record peak CUDA memory per epoch — useful to size batches on 8 GB cards."""

    def on_epoch_begin(self, epoch, logs=None):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, epoch, logs=None):
        import torch

        if torch.cuda.is_available() and logs is not None:
            logs["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
