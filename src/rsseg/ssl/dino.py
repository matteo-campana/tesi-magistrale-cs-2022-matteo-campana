# SPDX-License-Identifier: AGPL-3.0-or-later
"""DINO self-distillation (Caron et al., 2021) for the segmentation encoder — thesis ch. 3.

Faithful to the reference method, unlike the 2022 code (``REVISIONE_CODICE_ORIGINALE.md`` D1–D6):

* the loss acts on a **projection head** over globally pooled encoder features, not on the
  dense decoder output;
* softmax and cross-entropy are computed **per sample** (no mixing across the batch);
* the teacher starts as an exact copy of the student and follows it by EMA with a cosine
  momentum schedule 0.996 → 1;
* teacher temperature warm-up, cosine learning-rate and weight-decay schedules, gradient clipping;
* multi-crop: 2 global views (teacher + student) and N local views (student only).

The training step is written for the PyTorch backend of Keras 3 (``loss.backward()``), following
the Keras guide "Customizing what happens in fit() with PyTorch".
"""

from __future__ import annotations

import math

import keras
import numpy as np
from keras import layers, ops

from rsseg.config import SSLConfig
from rsseg.models.backbones import build_backbone


class WeightNormDense(layers.Layer):
    """Last DINO layer: linear map with column-normalised weights (``norm_last_layer=True``)."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="kernel",
        )

    def call(self, x):
        kernel = ops.cast(self.kernel, "float32")
        kernel = kernel / ops.maximum(
            ops.sqrt(ops.sum(ops.square(kernel), axis=0, keepdims=True)), 1e-12
        )
        return ops.matmul(ops.cast(x, "float32"), kernel)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_dino_head(in_dim: int, cfg: SSLConfig, name: str) -> keras.Model:
    inputs = keras.Input((in_dim,))
    x = layers.Dense(cfg.hidden_dim, activation="gelu", name="mlp1")(inputs)
    x = layers.Dense(cfg.hidden_dim, activation="gelu", name="mlp2")(x)
    x = layers.Dense(cfg.bottleneck_dim, dtype="float32", name="bottleneck")(x)
    x = layers.UnitNormalization(axis=-1, dtype="float32", name="l2norm")(x)
    out = WeightNormDense(cfg.out_dim, dtype="float32", name="prototypes")(x)
    return keras.Model(inputs, out, name=name)


def build_dino_network(cfg: SSLConfig, name: str) -> tuple[keras.Model, keras.Model]:
    """Return ``(network, backbone)``: image → prototype logits, and the encoder inside it."""
    backbone = build_backbone(cfg.arch, cfg.weights, cfg.drop_path)
    inputs = keras.Input((None, None, 3), name="image")
    features = backbone(inputs)[-1]
    pooled = layers.GlobalAveragePooling2D(name="gap")(features)
    head = build_dino_head(features.shape[-1], cfg, name=f"{name}_head")
    return keras.Model(inputs, head(pooled), name=name), backbone


def dino_loss(
    student_logits,
    teacher_logits,
    center,
    student_temp: float,
    teacher_temp: float,
    n_views: int,
    n_global: int = 2,
):
    """Cross-entropy between teacher and student distributions over all pairs of different views.

    ``student_logits``: ``(n_views * B, K)``, ``teacher_logits``: ``(n_global * B, K)``, both
    view-major (all samples of view 0, then view 1, ...). Returns a scalar averaged over the
    batch and over the ``n_global * (n_views - 1)`` view pairs.
    """
    k = ops.shape(student_logits)[-1]
    student = ops.reshape(
        ops.log_softmax(ops.cast(student_logits, "float32") / student_temp, axis=-1),
        (n_views, -1, k),
    )
    teacher = ops.softmax((ops.cast(teacher_logits, "float32") - center) / teacher_temp, axis=-1)
    teacher = ops.reshape(ops.stop_gradient(teacher), (n_global, -1, k))
    total = 0.0
    terms = 0
    for iq in range(n_global):
        for v in range(n_views):
            if v == iq:
                continue
            total = total + ops.mean(ops.sum(-teacher[iq] * student[v], axis=-1))
            terms += 1
    return total / terms


def cosine_schedule(
    start: float, end: float, total_steps: int, warmup_steps: int = 0, warmup_start: float = 0.0
) -> np.ndarray:
    total_steps = max(total_steps, 1)
    warm = np.linspace(warmup_start, start, warmup_steps) if warmup_steps > 0 else np.array([])
    steps = np.arange(max(total_steps - len(warm), 0))
    cos = end + 0.5 * (start - end) * (1 + np.cos(np.pi * steps / max(len(steps), 1)))
    return np.concatenate([warm, cos]).astype(np.float64)


class DINOModel(keras.Model):
    """Student/teacher pair trained with ``fit`` on ``(global_views, local_views)`` batches."""

    def __init__(self, cfg: SSLConfig, steps_per_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.steps_per_epoch = max(steps_per_epoch, 1)
        self.student, self.student_backbone = build_dino_network(cfg, "student")
        self.teacher, self.teacher_backbone = build_dino_network(cfg, "teacher")
        self.teacher.set_weights(self.student.get_weights())
        self.teacher.trainable = False
        self.center = self.add_weight(
            shape=(1, cfg.out_dim),
            initializer="zeros",
            trainable=False,
            name="center",
            dtype="float32",
        )

        total = cfg.epochs * self.steps_per_epoch
        self.lr_schedule = cosine_schedule(
            cfg.lr * cfg.batch_size / 256.0,
            cfg.min_lr,
            total,
            warmup_steps=cfg.warmup_epochs * self.steps_per_epoch,
        )
        self.wd_schedule = cosine_schedule(cfg.weight_decay, cfg.weight_decay_end, total)
        self.momentum_schedule = cosine_schedule(cfg.momentum_teacher, 1.0, total)
        warm = min(cfg.warmup_teacher_temp_epochs, cfg.epochs)
        self.teacher_temp_schedule = np.concatenate(
            [
                np.linspace(cfg.warmup_teacher_temp, cfg.teacher_temp, warm),
                np.full(max(cfg.epochs - warm, 0), cfg.teacher_temp),
            ]
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.built = True

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _schedules(self, step: int) -> tuple[float, float, float, float]:
        i = min(step, len(self.lr_schedule) - 1)
        epoch = min(step // self.steps_per_epoch, len(self.teacher_temp_schedule) - 1)
        return (
            float(self.lr_schedule[i]),
            float(self.wd_schedule[i]),
            float(self.momentum_schedule[i]),
            float(self.teacher_temp_schedule[epoch]),
        )

    def train_step(self, data):
        import torch

        global_views, local_views = data[0], data[1]
        batch = global_views.shape[0]
        n_global, n_local = global_views.shape[1], local_views.shape[1]
        g = ops.reshape(
            ops.transpose(global_views, (1, 0, 2, 3, 4)),
            (n_global * batch, *global_views.shape[2:]),
        )

        step = int(ops.convert_to_numpy(self.optimizer.iterations))
        lr, wd, momentum, teacher_temp = self._schedules(step)
        # With mixed_float16 Keras wraps the optimizer in a LossScaleOptimizer: schedule the inner one.
        inner = getattr(self.optimizer, "inner_optimizer", self.optimizer)
        inner.learning_rate.assign(lr)
        inner.weight_decay = wd

        with torch.no_grad():
            teacher_out = ops.cast(self.teacher(g, training=False), "float32")

        self.zero_grad()
        student_out = [self.student(g, training=True)]
        if n_local:
            loc = ops.reshape(
                ops.transpose(local_views, (1, 0, 2, 3, 4)),
                (n_local * batch, *local_views.shape[2:]),
            )
            student_out.append(self.student(loc, training=True))
        student_out = ops.concatenate([ops.cast(s, "float32") for s in student_out], axis=0)

        loss = dino_loss(
            student_out,
            teacher_out,
            self.center,
            self.cfg.student_temp,
            teacher_temp,
            n_views=n_global + n_local,
            n_global=n_global,
        )
        loss_value = float(ops.convert_to_numpy(ops.stop_gradient(loss)))
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"DINO loss is {loss_value} at step {step}")
        self.optimizer.scale_loss(loss).backward()

        variables = self.student.trainable_weights
        grads = [v.value.grad for v in variables]
        with torch.no_grad():
            self.optimizer.apply(grads, variables)
            # Teacher EMA (parameters and BatchNorm statistics).
            t_vals = [t.value for t in self.teacher.weights]
            s_vals = [s.value for s in self.student.weights]
            torch._foreach_mul_(t_vals, momentum)
            torch._foreach_add_(t_vals, s_vals, alpha=1.0 - momentum)
            batch_center = ops.mean(teacher_out, axis=0, keepdims=True)
            m = self.cfg.center_momentum
            self.center.assign(self.center * m + batch_center * (1.0 - m))

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "lr": lr,
            "wd": wd,
            "momentum": momentum,
            "teacher_temp": teacher_temp,
        }

    def teacher_entropy(self, global_views) -> float:
        """Mean entropy of teacher distributions — near 0 signals collapse to one prototype."""
        b = global_views.shape[0]
        g = ops.reshape(global_views, (b * global_views.shape[1], *global_views.shape[2:]))
        probs = ops.softmax(
            (ops.cast(self.teacher(g, training=False), "float32") - self.center)
            / self.cfg.teacher_temp,
            axis=-1,
        )
        return float(
            ops.convert_to_numpy(ops.mean(-ops.sum(probs * ops.log(probs + 1e-12), axis=-1)))
        )
