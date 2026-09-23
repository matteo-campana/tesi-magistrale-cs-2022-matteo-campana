# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segmentation metrics computed from a confusion matrix accumulated over the whole split.

Averaging per-batch scores (as the 2022 code did) over-weights small or easy batches; every
score here derives from one global confusion matrix (rows = ground truth, cols = prediction).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

from rsseg.data.classes import CLASS_NAMES, CLUTTER, IGNORE_INDEX


def confusion_matrix_np(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_index: int = IGNORE_INDEX
) -> np.ndarray:
    t = y_true.reshape(-1).astype(np.int64)
    p = y_pred.reshape(-1).astype(np.int64)
    valid = (t != ignore_index) & (t < num_classes)
    return np.bincount(t[valid] * num_classes + p[valid], minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def scores_from_confusion(
    cm: np.ndarray, class_names: tuple[str, ...] = CLASS_NAMES, exclude: tuple[int, ...] = ()
) -> dict:
    """OA, per-class IoU / F1 / precision / recall, and their means (optionally excluding classes)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(1)
    pred = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (gt + pred - tp)
        f1 = 2 * tp / (gt + pred)
        recall = tp / gt
        precision = tp / pred
    present = gt > 0
    keep = present & ~np.isin(np.arange(len(tp)), exclude)
    total = cm.sum()
    return {
        "oa": float(tp.sum() / total) if total else float("nan"),
        "miou": float(np.nanmean(iou[keep])) if keep.any() else float("nan"),
        "mf1": float(np.nanmean(f1[keep])) if keep.any() else float("nan"),
        "per_class": {
            name: {
                "iou": float(iou[i]),
                "f1": float(f1[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "support": int(gt[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


class _ConfusionMatrixMetric(keras.metrics.Metric):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        exclude: tuple[int, ...] = (),
        name: str = "cm",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.exclude = tuple(exclude)
        self.cm = self.add_variable(
            shape=(num_classes, num_classes), initializer="zeros", dtype="float64", name="confusion"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        k = self.num_classes
        t = ops.reshape(ops.cast(y_true, "int32"), (-1,))
        p = ops.reshape(ops.cast(ops.argmax(y_pred, axis=-1), "int32"), (-1,))
        valid = ops.logical_and(t != self.ignore_index, t < k)
        idx = ops.where(valid, t * k + p, k * k)
        counts = ops.bincount(idx, minlength=k * k + 1)[: k * k]
        self.cm.assign_add(ops.reshape(ops.cast(counts, "float64"), (k, k)))

    def reset_state(self):
        self.cm.assign(ops.zeros_like(self.cm))

    def _parts(self):
        cm = ops.cast(self.cm, "float64")
        tp = ops.diagonal(cm)
        return tp, ops.sum(cm, axis=1), ops.sum(cm, axis=0)

    def _class_mask(self, gt):
        mask = gt > 0
        if self.exclude:
            excluded = np.isin(np.arange(self.num_classes), self.exclude)
            mask = ops.logical_and(mask, ops.convert_to_tensor(~excluded))
        return ops.cast(mask, "float64")

    def _masked_mean(self, values, gt):
        mask = self._class_mask(gt)
        values = ops.where(mask > 0, values, 0.0)
        return ops.sum(values) / ops.maximum(ops.sum(mask), 1.0)

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "exclude": self.exclude,
        }


class MeanIoU(_ConfusionMatrixMetric):
    def __init__(self, num_classes: int, name: str = "miou", **kwargs):
        super().__init__(num_classes, name=name, **kwargs)

    def result(self):
        tp, gt, pred = self._parts()
        return self._masked_mean(tp / ops.maximum(gt + pred - tp, 1.0), gt)


class MeanF1(_ConfusionMatrixMetric):
    def __init__(self, num_classes: int, name: str = "mf1", **kwargs):
        super().__init__(num_classes, name=name, **kwargs)

    def result(self):
        tp, gt, pred = self._parts()
        return self._masked_mean(2 * tp / ops.maximum(gt + pred, 1.0), gt)


class OverallAccuracy(_ConfusionMatrixMetric):
    def __init__(self, num_classes: int, name: str = "oa", **kwargs):
        super().__init__(num_classes, name=name, **kwargs)

    def result(self):
        tp, gt, _ = self._parts()
        return ops.sum(tp) / ops.maximum(ops.sum(gt), 1.0)


def segmentation_metrics(num_classes: int) -> list[keras.metrics.Metric]:
    return [
        MeanIoU(num_classes),
        MeanF1(num_classes),
        OverallAccuracy(num_classes),
        MeanIoU(num_classes, exclude=(CLUTTER,), name="miou_no_clutter"),
    ]
