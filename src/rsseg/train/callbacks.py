# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training callbacks: codebook health and qualitative predictions (both logged to MLflow) and
a rich progress bar that replaces the Keras one.

Place these callbacks *before* :class:`rsseg.tracking.MLflowMetricsLogger`: they add entries to
``logs`` that the logger then records.
"""

from __future__ import annotations

from pathlib import Path

import keras
import mlflow
import numpy as np

from rsseg.console import err_console, format_value, make_progress
from rsseg.eval.visualize import save_panels
from rsseg.models.quantizer import find_quantizers


class CodebookMonitor(keras.callbacks.Callback):
    """Adds ``<vq layer>_perplexity`` and ``<vq layer>_usage`` (last training batch) to epoch logs."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for q in find_quantizers(self.model):
            logs[f"{q.name}_perplexity"] = float(keras.ops.convert_to_numpy(q.perplexity))
            logs[f"{q.name}_usage"] = float(keras.ops.convert_to_numpy(q.usage))


class PredictionLogger(keras.callbacks.Callback):
    """Every ``every`` epochs, predict a fixed reference batch and log a PNG panel to MLflow."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, out_dir: str | Path, every: int = 10):
        super().__init__()
        self.images = images
        self.masks = masks
        self.out_dir = Path(out_dir)
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if self.every <= 0 or (epoch + 1) % self.every:
            return
        logits = keras.ops.convert_to_numpy(self.model(self.images, training=False))
        preds = logits.argmax(-1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"epoch_{epoch + 1:03d}.png"
        save_panels(path, self.images, self.masks, preds)
        if mlflow.active_run():
            mlflow.log_artifact(str(path), artifact_path="predictions")


class RichProgressBar(keras.callbacks.Callback):
    """Progress bar per epoch (training and validation) plus a one-line summary of the metrics.

    Use it with ``fit(..., verbose=0)``. ``keys`` limits the metrics shown in the bar (the
    summary line shows all numeric entries of ``logs``).
    """

    def __init__(self, keys: tuple[str, ...] = ("loss", "miou")):
        super().__init__()
        self.keys = keys
        self.progress = None
        self.task = None
        self.epochs = 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs") or 0
        self.progress = make_progress(transient=True)
        self.progress.start()

    def on_train_end(self, logs=None):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None

    def _start(self, description: str, total: int | None):
        if self.task is not None:
            self.progress.remove_task(self.task)
        self.task = self.progress.add_task(description, total=total, info="")

    def on_epoch_begin(self, epoch, logs=None):
        self._start(f"epoch {epoch + 1}/{self.epochs}", self.params.get("steps"))

    def on_train_batch_end(self, batch, logs=None):
        info = "  ".join(
            f"{k} {format_value(float(logs[k]))}" for k in self.keys if logs and k in logs
        )
        self.progress.update(self.task, completed=batch + 1, info=info)

    def on_test_begin(self, logs=None):
        if self.progress is not None:  # validation inside fit()
            self._start("  validation", None)

    def on_test_batch_end(self, batch, logs=None):
        if self.progress is not None:
            self.progress.update(self.task, completed=batch + 1)

    def on_epoch_end(self, epoch, logs=None):
        if self.task is not None:
            self.progress.remove_task(self.task)
            self.task = None
        parts = []
        for k, v in (logs or {}).items():
            try:
                parts.append(f"[key]{k}[/] {format_value(float(v))}")
            except (TypeError, ValueError):
                continue
        err_console.log(f"[title]epoch {epoch + 1}/{self.epochs}[/]  " + "  ".join(parts))
