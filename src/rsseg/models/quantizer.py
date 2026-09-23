# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vector quantization layer (van den Oord et al., 2017) — thesis §2.4 and §4.2.1.

``L_vq = ||sg[z_e] - e||² + β ||z_e - sg[e]||²``. The gradient reaches the encoder through the
straight-through estimator ``z_q = z_e + sg[e - z_e]``.

Differences from the 2022 code (see ``docs/REVISIONE_CODICE_ORIGINALE.md`` V2, V5):

* with ``ema_decay > 0`` the codebook is updated by exponential moving averages kept in
  **non-trainable variables** (never re-created as new parameters);
* the codebook is initialised from encoder outputs of the first training batch (a uniform
  ±1/K init is far from post-ReLU features and makes every vector pick the same code);
* dead codes (EMA assignments per batch below ``dead_code_threshold``) are re-initialised from
  encoder outputs;
* ``perplexity`` and ``usage`` are exposed as variables for monitoring.

Computation is always float32. The loss is registered with ``add_loss`` and summed by Keras
with the segmentation loss.
"""

from __future__ import annotations

import keras
from keras import layers, ops


class VectorQuantizer(layers.Layer):
    def __init__(
        self,
        num_codes: int = 512,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        loss_weight: float = 1.0,
        restart_dead_codes: bool = True,
        dead_code_threshold: float = 0.1,
        epsilon: float = 1e-5,
        seed: int | None = None,
        **kwargs,
    ):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.num_codes = num_codes
        self.beta = beta
        self.ema_decay = ema_decay
        self.loss_weight = loss_weight
        self.restart_dead_codes = restart_dead_codes
        self.dead_code_threshold = dead_code_threshold
        self.epsilon = epsilon
        self.seed_generator = keras.random.SeedGenerator(seed)

    @property
    def use_ema(self) -> bool:
        return self.ema_decay > 0

    def build(self, input_shape):
        dim = input_shape[-1]
        self.code_dim = dim
        init = keras.initializers.RandomUniform(-1.0 / self.num_codes, 1.0 / self.num_codes)
        self.codebook = self.add_weight(
            shape=(self.num_codes, dim),
            initializer=init,
            trainable=not self.use_ema,
            name="codebook",
        )
        if self.use_ema:
            self.ema_cluster_size = self.add_weight(
                shape=(self.num_codes,),
                initializer="ones",
                trainable=False,
                name="ema_cluster_size",
            )
            self.ema_embed_sum = self.add_weight(
                shape=(self.num_codes, dim),
                initializer="zeros",
                trainable=False,
                name="ema_embed_sum",
            )
            self.ema_embed_sum.assign(self.codebook)
        self.initialized = self.add_weight(
            shape=(), initializer="zeros", trainable=False, name="initialized"
        )
        self.perplexity = self.add_weight(
            shape=(), initializer="zeros", trainable=False, name="perplexity"
        )
        self.usage = self.add_weight(shape=(), initializer="zeros", trainable=False, name="usage")

    def lookup(self, flat):
        """Return ``(indices, one_hot, quantized)`` for ``flat`` of shape ``(M, D)``."""
        codebook = ops.convert_to_tensor(self.codebook)
        distances = (
            ops.sum(ops.square(flat), axis=1, keepdims=True)
            - 2.0 * ops.matmul(flat, ops.transpose(codebook))
            + ops.sum(ops.square(codebook), axis=1)[None, :]
        )
        indices = ops.argmin(distances, axis=1)
        one_hot = ops.one_hot(indices, self.num_codes, dtype="float32")
        return indices, one_hot, ops.matmul(one_hot, codebook)

    def _sample_rows(self, flat, n: int):
        idx = keras.random.randint((n,), 0, ops.shape(flat)[0], seed=self.seed_generator)
        return ops.take(ops.stop_gradient(flat), idx, axis=0)

    def _init_from_data(self, flat):
        codes = self._sample_rows(flat, self.num_codes)
        codes = codes + 1e-3 * keras.random.normal(ops.shape(codes), seed=self.seed_generator)
        self.codebook.assign(codes)
        if self.use_ema:
            self.ema_embed_sum.assign(codes)
            self.ema_cluster_size.assign(ops.ones((self.num_codes,)))
        self.initialized.assign(1.0)

    def _ema_update(self, flat, one_hot):
        flat = ops.stop_gradient(flat)
        decay = self.ema_decay
        counts = ops.sum(one_hot, axis=0)
        cluster_size = self.ema_cluster_size * decay + (1.0 - decay) * counts
        embed_sum = self.ema_embed_sum * decay + (1.0 - decay) * ops.matmul(
            ops.transpose(one_hot), flat
        )
        if self.restart_dead_codes:
            dead = cluster_size < self.dead_code_threshold
            samples = self._sample_rows(flat, self.num_codes)
            embed_sum = ops.where(dead[:, None], samples, embed_sum)
            cluster_size = ops.where(dead, ops.ones_like(cluster_size), cluster_size)
        total = ops.sum(cluster_size)
        smoothed = (cluster_size + self.epsilon) / (total + self.num_codes * self.epsilon) * total
        self.ema_cluster_size.assign(cluster_size)
        self.ema_embed_sum.assign(embed_sum)
        self.codebook.assign(embed_sum / smoothed[:, None])

    def call(self, inputs, training=None):
        z = ops.cast(inputs, "float32")
        flat = ops.reshape(z, (-1, self.code_dim))
        if training and float(ops.convert_to_numpy(self.initialized)) == 0.0:
            self._init_from_data(flat)
        _, one_hot, quantized = self.lookup(flat)
        quantized = ops.reshape(quantized, ops.shape(z))

        commitment = ops.mean(ops.square(ops.stop_gradient(quantized) - z))
        if self.use_ema:
            loss = self.beta * commitment
        else:
            loss = ops.mean(ops.square(quantized - ops.stop_gradient(z))) + self.beta * commitment

        if training:
            self.add_loss(self.loss_weight * loss)
            probs = ops.mean(one_hot, axis=0)
            self.perplexity.assign(ops.exp(-ops.sum(probs * ops.log(probs + 1e-10))))
            self.usage.assign(ops.mean(ops.cast(probs > 0, "float32")))
            if self.use_ema:
                self._ema_update(flat, one_hot)
        return z + ops.stop_gradient(quantized - z)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            "num_codes": self.num_codes,
            "beta": self.beta,
            "ema_decay": self.ema_decay,
            "loss_weight": self.loss_weight,
            "restart_dead_codes": self.restart_dead_codes,
            "dead_code_threshold": self.dead_code_threshold,
            "epsilon": self.epsilon,
        }


def find_quantizers(model: keras.Model) -> list[VectorQuantizer]:
    found = []
    for layer in model._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, VectorQuantizer):
            found.append(layer)
    return found
