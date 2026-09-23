# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segmentation loss: cross-entropy that ignores only unlabelled pixels.

The thesis code used ``ignore_index=0``, which silently removed the *clutter* class from
training (``docs/REVISIONE_CODICE_ORIGINALE.md`` S1). Here the ignored value is 255, never a class.
"""

from __future__ import annotations

import keras
from keras import ops

from rsseg.data.classes import IGNORE_INDEX


@keras.saving.register_keras_serializable(package="rsseg")
class MaskedSparseCrossEntropy(keras.losses.Loss):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        class_weights: list[float] | None = None,
        name: str = "masked_ce",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = list(class_weights) if class_weights is not None else None

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        valid = ops.logical_and(y_true != self.ignore_index, y_true < self.num_classes)
        labels = ops.where(valid, y_true, 0)
        ce = keras.losses.sparse_categorical_crossentropy(
            labels, ops.cast(y_pred, "float32"), from_logits=True
        )
        weights = ops.cast(valid, "float32")
        if self.class_weights is not None:
            weights = weights * ops.take(
                ops.convert_to_tensor(self.class_weights, "float32"), labels
            )
        return ops.sum(ce * weights) / ops.maximum(ops.sum(weights), 1.0)

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "class_weights": self.class_weights,
        }
