# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attention blocks of MAResU-Net (Li et al., 2022), channels-last.

* :class:`LinearPAM` — position attention with the *Linear Attention Mechanism* (Li et al., 2020):
  ``softmax(q·k)`` is replaced by its first-order Taylor expansion ``1 + q̂·k̂`` on L2-normalised
  queries and keys, so the cost is O(N) in the number of pixels (thesis §2.5.3)::

      D(Q,K,V)_i = (Σ_j v_j + q̂_iᵀ Σ_j k̂_j v_jᵀ) / (N + q̂_iᵀ Σ_j k̂_j)

* :class:`ChannelAttention` — dot-product attention over channels (C ≪ N, so C×C is cheap).

The attention maths runs in float32 even under mixed precision.
"""

from __future__ import annotations

import keras
from keras import layers, ops


def _l2_normalize(x, axis: int = -1, eps: float = 1e-12):
    return x / ops.maximum(ops.sqrt(ops.sum(ops.square(x), axis=axis, keepdims=True)), eps)


class LinearPAM(layers.Layer):
    def __init__(self, scale: int = 8, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.eps = eps

    def build(self, input_shape):
        channels = input_shape[-1]
        self.query_conv = layers.Conv2D(
            max(channels // self.scale, 1), 1, dtype=self.dtype_policy, name="query"
        )
        self.key_conv = layers.Conv2D(
            max(channels // self.scale, 1), 1, dtype=self.dtype_policy, name="key"
        )
        self.value_conv = layers.Conv2D(channels, 1, dtype=self.dtype_policy, name="value")
        for conv in (self.query_conv, self.key_conv, self.value_conv):
            conv.build(input_shape)
        self.gamma = self.add_weight(shape=(1,), initializer="zeros", name="gamma")

    def call(self, x):
        shape = ops.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        n = h * w
        q = _l2_normalize(ops.reshape(ops.cast(self.query_conv(x), "float32"), (b, n, -1)))
        k = _l2_normalize(ops.reshape(ops.cast(self.key_conv(x), "float32"), (b, n, -1)))
        v = ops.reshape(ops.cast(self.value_conv(x), "float32"), (b, n, c))

        denom = ops.cast(n, "float32") + ops.einsum("bnm,bm->bn", q, ops.sum(k, axis=1) + self.eps)
        value_sum = ops.sum(v, axis=1, keepdims=True)  # (b, 1, c)
        kv = ops.einsum("bnm,bnc->bmc", k, v)  # (b, m, c)
        out = (value_sum + ops.einsum("bnm,bmc->bnc", q, kv)) / denom[..., None]
        out = ops.reshape(out, (b, h, w, c))
        gamma = ops.cast(self.gamma, "float32")
        return ops.cast(ops.cast(x, "float32") + gamma * out, x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "scale": self.scale, "eps": self.eps}


class ChannelAttention(layers.Layer):
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,), initializer="zeros", name="gamma")

    def call(self, x):
        shape = ops.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        feats = ops.reshape(ops.cast(x, "float32"), (b, h * w, c))
        energy = ops.einsum("bnc,bnd->bcd", feats, feats)
        energy = ops.max(energy, axis=-1, keepdims=True) - energy
        attention = ops.softmax(energy, axis=-1)
        out = ops.reshape(ops.einsum("bcd,bnd->bnc", attention, feats), (b, h, w, c))
        gamma = ops.cast(self.gamma, "float32")
        return ops.cast(ops.cast(x, "float32") + gamma * out, x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def conv_relu(x, filters: int, kernel: int, name: str):
    return layers.Conv2D(filters, kernel, padding="same", activation="relu", name=name)(x)


def pam_cam_block(x, out_channels: int, dropout: float, name: str):
    """MAResU-Net attention block: conv → (PAM ‖ CAM) → sum → 1×1 conv."""
    channels = x.shape[-1]
    x = conv_relu(x, channels, 3, f"{name}_conv1")
    p = LinearPAM(name=f"{name}_pam")(x)
    p = conv_relu(
        layers.SpatialDropout2D(dropout, name=f"{name}_drop_p")(p), channels, 1, f"{name}_conv_p"
    )
    c = ChannelAttention(name=f"{name}_cam")(x)
    c = conv_relu(
        layers.SpatialDropout2D(dropout, name=f"{name}_drop_c")(c), channels, 1, f"{name}_conv_c"
    )
    y = layers.Add(name=f"{name}_sum")([p, c])
    y = layers.SpatialDropout2D(dropout, name=f"{name}_drop_out")(y)
    return conv_relu(y, out_channels, 1, f"{name}_conv_out")


def linear_pam_reference(x, wq, bq, wk, bk, wv, bv, gamma, eps=1e-6):
    """NumPy transcription of the original PyTorch PAM (channels-first maths), for tests."""
    import numpy as np

    b, h, w, c = x.shape
    n = h * w
    flat = x.reshape(b, n, c)
    q = (flat @ wq + bq).transpose(0, 2, 1)  # (b, cq, n)
    k = (flat @ wk + bk).transpose(0, 2, 1)
    v = (flat @ wv + bv).transpose(0, 2, 1)  # (b, c, n)
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).transpose(0, 2, 1)  # (b, n, cq)
    k = k / np.linalg.norm(k, axis=1, keepdims=True)  # (b, cq, n)
    tailor = 1 / (n + np.einsum("bnc,bc->bn", q, k.sum(-1) + eps))
    value_sum = np.broadcast_to(v.sum(-1, keepdims=True), v.shape)
    matrix = np.einsum("bmn,bcn->bmc", k, v)
    matrix_sum = value_sum + np.einsum("bnm,bmc->bcn", q, matrix)
    weight_value = np.einsum("bcn,bn->bcn", matrix_sum, tailor)
    return x + gamma * weight_value.transpose(0, 2, 1).reshape(b, h, w, c)


__all__ = ["LinearPAM", "ChannelAttention", "pam_cam_block", "linear_pam_reference", "keras"]
