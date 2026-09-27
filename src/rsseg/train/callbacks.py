# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training callbacks: top-k checkpoints, codebook health and qualitative predictions (both
logged to MLflow), a rich progress bar that replaces the Keras one and the progress shared with
``rsseg campaign``.

Place these callbacks *before* :class:`rsseg.tracking.MLflowMetricsLogger`: they add entries to
``logs`` that the logger then records.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path

import keras
import mlflow
import numpy as np

from rsseg.console import err_console, format_value, make_progress
from rsseg.eval.visualize import save_panels
from rsseg.models.quantizer import find_quantizers
from rsseg.progress import report

log = logging.getLogger(__name__)


class TopKCheckpoint(keras.callbacks.Callback):
    """Keep the weights of the ``k`` best epochs by ``monitor`` (higher is better).

    Checkpoints are ``<dir>/epoch_<NNN>_<monitor>_<value>.weights.h5``; when a better epoch
    arrives, the worst of the kept ones is deleted. ``best_path`` is always a copy of the best
    checkpoint, and ``<dir>/topk.json`` lists the kept epochs from best to worst. Epochs whose
    ``monitor`` is missing or NaN are skipped.
    """

    def __init__(self, directory: str | Path, best_path: str | Path, k: int = 5,
                 monitor: str = "val_miou"):  # fmt: skip
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.directory = Path(directory)
        self.best_path = Path(best_path)
        self.k = k
        self.monitor = monitor
        self.kept: list[tuple[float, int, Path]] = []  # (value, epoch, path), best first

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None or math.isnan(float(value)):
            log.warning("TopKCheckpoint: %s not available at epoch %d", self.monitor, epoch + 1)
            return
        value = float(value)
        if len(self.kept) == self.k and value <= self.kept[-1][0]:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"epoch_{epoch + 1:03d}_{self.monitor}_{value:.4f}.weights.h5"
        self.model.save_weights(str(path))
        self.kept.append((value, epoch + 1, path))
        self.kept.sort(key=lambda t: (-t[0], t[1]))  # ties: the earlier epoch ranks first
        for _, _, dropped in self.kept[self.k :]:
            dropped.unlink(missing_ok=True)
        self.kept = self.kept[: self.k]
        if self.kept[0][2] == path:
            shutil.copyfile(path, self.best_path)
        (self.directory / "topk.json").write_text(
            json.dumps(
                [{"epoch": e, self.monitor: v, "path": p.name} for v, e, p in self.kept], indent=2
            ),
            encoding="utf-8",
        )


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
        from rsseg.eval.sliding_window import infer

        logits = infer(self.model, self.images)
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


class ProgressReporter(keras.callbacks.Callback):
    """Share epoch, batch and ``monitor`` with ``rsseg campaign`` (see :mod:`rsseg.progress`).

    ``patience`` mirrors the early stopping: the campaign shows how many epochs without an
    improvement of ``monitor`` are left before training stops.
    """

    def __init__(self, phase: str = "train", monitor: str | None = "val_miou",
                 patience: int | None = None, every: int = 10):  # fmt: skip
        super().__init__()
        self.phase, self.monitor, self.patience, self.every = phase, monitor, patience, every
        self.best = -math.inf
        self.best_epoch = 0
        self.durations: list[float] = []
        self.t_epoch = 0.0

    def on_train_begin(self, logs=None):
        report(phase=self.phase, stage="train", epoch=0, epochs=self.params.get("epochs"),
               batch=0, steps=self.params.get("steps"), patience=self.patience)  # fmt: skip

    def on_epoch_begin(self, epoch, logs=None):
        self.t_epoch = time.time()
        report(stage="train", epoch=epoch + 1, batch=0)

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every == 0:
            report(batch=batch + 1)

    def on_test_begin(self, logs=None):
        report(stage="validation")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.durations.append(time.time() - self.t_epoch)
        fields = {"epoch_s": sum(self.durations) / len(self.durations),
                  "loss": _number(logs.get("loss"))}  # fmt: skip
        value = _number(logs.get(self.monitor)) if self.monitor else None
        if value is not None:
            if value > self.best:  # same rule as keras.callbacks.EarlyStopping (min_delta 0)
                self.best, self.best_epoch = value, epoch + 1
            fields |= {"monitor": self.monitor, "value": value, "best": self.best,
                       "best_epoch": self.best_epoch, "since_best": epoch + 1 - self.best_epoch}  # fmt: skip
        report(**fields)


def _number(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v
