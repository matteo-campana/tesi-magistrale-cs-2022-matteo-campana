# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training callbacks: codebook health and qualitative predictions, both logged to MLflow.

Place these callbacks *before* :class:`rsseg.tracking.MLflowMetricsLogger`: they add entries to
``logs`` that the logger then records.
"""

from __future__ import annotations

from pathlib import Path

import keras
import mlflow
import numpy as np

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
