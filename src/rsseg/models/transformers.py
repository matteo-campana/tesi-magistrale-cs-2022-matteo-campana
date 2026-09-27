# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vision Transformer encoders in Keras 3 returning four feature levels ``[C1, C2, C3, C4]``
(strides 4, 8, 16, 32), interchangeable with the ResNet encoders of :mod:`backbones`.

* **Swin Transformer** (Liu et al., 2021) — hierarchical: the four stages already have strides
  4–32. Each stage output gets a LayerNorm (the last one is the ImageNet ``norm``) as in the
  segmentation variants of Swin. Numerically identical to ``torchvision.models.swin_*``.
* **ViT** (Dosovitskiy et al., 2021) — plain, single stride 16. Four evenly spaced blocks are
  tapped and turned into a pyramid with the "simple feature pyramid" of BEiT/ViTDet:
  2× transposed conv twice (stride 4), once (stride 8), identity (16), 2×2 max-pool (32).
  Position embeddings are bicubically interpolated to the input grid, so any size that is a
  multiple of 32 works. The encoder is identical to ``torchvision.models.vit_*``.

Input: channels-last RGB in ``[0, 255]`` (optionally followed by extra channels such as
the nDSM); ImageNet normalisation is the first layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers, ops

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SwinSpec:
    embed_dim: int
    depths: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    window_size: int = 7
    patch_size: int = 4
    mlp_ratio: float = 4.0


@dataclass(frozen=True)
class ViTSpec:
    dim: int
    depth: int
    num_heads: int
    patch_size: int = 16
    mlp_dim: int | None = None
    image_size: int = 224
    """Resolution of the stored position embeddings (interpolated to the actual input)."""

    @property
    def taps(self) -> tuple[int, ...]:
        """Blocks (0-based) whose outputs form the four feature levels."""
        q = self.depth // 4
        return (q - 1, 2 * q - 1, 3 * q - 1, self.depth - 1)


SWIN_ARCHS: dict[str, SwinSpec] = {
    "swin_t": SwinSpec(96, (2, 2, 6, 2), (3, 6, 12, 24)),
    "swin_s": SwinSpec(96, (2, 2, 18, 2), (3, 6, 12, 24)),
    "swin_b": SwinSpec(128, (2, 2, 18, 2), (4, 8, 16, 32)),
}
VIT_ARCHS: dict[str, ViTSpec] = {
    "vit_s_16": ViTSpec(384, 12, 6),  # ViT-S/16 as in DeiT/DINO
    "vit_b_16": ViTSpec(768, 12, 12),
    "vit_l_16": ViTSpec(1024, 24, 16),
}


def swin_feature_channels(spec: SwinSpec) -> list[int]:
    return [spec.embed_dim * 2**i for i in range(4)]


def vit_feature_channels(spec: ViTSpec) -> list[int]:
    return [spec.dim // 4, spec.dim // 2, spec.dim, spec.dim]


def _layer_norm(name: str, epsilon: float) -> layers.LayerNormalization:
    return layers.LayerNormalization(epsilon=epsilon, name=name)


def _gelu(x):
    return ops.gelu(x, approximate=False)  # exact GELU, as nn.GELU()


class DropPath(layers.Layer):
    """Stochastic depth ("row" mode): drop the whole residual branch of a sample."""

    def __init__(self, rate: float = 0.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed) if self.rate > 0 else None

    def call(self, x, training=None):
        if not training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        mask = ops.floor(keep + keras.random.uniform(shape, seed=self.seed_generator))
        return x * ops.cast(mask / keep, x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "seed": self.seed}


class _MLP(layers.Layer):
    def __init__(self, dim: int, hidden: int, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden, name="fc1")
        self.fc2 = layers.Dense(dim, name="fc2")

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build((*input_shape[:-1], self.fc1.units))

    def call(self, x):
        return self.fc2(_gelu(self.fc1(x)))

    def compute_output_shape(self, input_shape):
        return input_shape


# --------------------------------------------------------------------------------------------
# Swin Transformer
# --------------------------------------------------------------------------------------------


def relative_position_index(window: int) -> np.ndarray:
    coords = np.stack(np.meshgrid(np.arange(window), np.arange(window), indexing="ij")).reshape(
        2, -1
    )
    rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (window - 1)
    return (rel[..., 0] * (2 * window - 1) + rel[..., 1]).reshape(-1)


def shift_attention_mask(
    pad_h: int, pad_w: int, window: int, shift_h: int, shift_w: int
) -> np.ndarray:
    """``(num_windows, N, N)`` additive mask: -100 between tokens of different regions.

    The slices reproduce torchvision exactly, including a zero shift along one axis.
    """
    img = np.zeros((pad_h, pad_w), np.float32)
    count = 0
    for hs in (slice(0, -window), slice(-window, -shift_h), slice(-shift_h, None)):
        for ws in (slice(0, -window), slice(-window, -shift_w), slice(-shift_w, None)):
            img[hs, ws] = count
            count += 1
    win = img.reshape(pad_h // window, window, pad_w // window, window).transpose(0, 2, 1, 3)
    win = win.reshape(-1, window * window)
    diff = win[:, None, :] - win[:, :, None]
    return np.where(diff != 0, -100.0, 0.0).astype(np.float32)


class SwinBlock(layers.Layer):
    """(Shifted-)window multi-head self-attention block with relative position bias.

    Operates on ``(B, H, W, C)``; ``H``/``W`` are padded to multiples of the window, and the
    shift is disabled when a single window covers the whole (padded) map, as in torchvision.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim, self.num_heads = dim, num_heads
        self.window_size, self.shift_size = window_size, shift_size
        self.mlp_ratio, self.drop_path_rate = mlp_ratio, drop_path
        self.norm1 = _layer_norm("norm1", 1e-5)
        self.qkv = layers.Dense(3 * dim, name="qkv")
        self.proj = layers.Dense(dim, name="proj")
        self.norm2 = _layer_norm("norm2", 1e-5)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), name="mlp")
        self.drop_path = DropPath(drop_path, name="drop_path")
        self._rel_index = relative_position_index(window_size)
        self._masks: dict[tuple[int, int], np.ndarray] = {}

    def build(self, input_shape):
        tokens = (None, self.dim)
        for layer in (self.norm1, self.norm2, self.mlp, self.drop_path):
            layer.build(tokens)
        self.qkv.build(tokens)
        self.proj.build(tokens)
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size - 1) ** 2, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="relative_position_bias_table",
        )

    def _attention(self, x):
        b, h, w, c = ops.shape(x)
        ws, heads = self.window_size, self.num_heads
        pad_b, pad_r = (ws - h % ws) % ws, (ws - w % ws) % ws
        x = ops.pad(x, [[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
        ph, pw = h + pad_b, w + pad_r
        shift_h = self.shift_size if ws < ph else 0
        shift_w = self.shift_size if ws < pw else 0
        if shift_h or shift_w:
            x = ops.roll(x, shift=(-shift_h, -shift_w), axis=(1, 2))

        nh, nw, n = ph // ws, pw // ws, ws * ws
        x = ops.reshape(x, (b, nh, ws, nw, ws, c))
        x = ops.reshape(ops.transpose(x, (0, 1, 3, 2, 4, 5)), (b * nh * nw, n, c))

        qkv = ops.reshape(self.qkv(x), (-1, n, 3, heads, c // heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * (c // heads) ** -0.5, qkv[1], qkv[2]
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B*nW, heads, N, N)

        bias = ops.take(self.relative_position_bias_table, self._rel_index, axis=0)
        bias = ops.transpose(ops.reshape(bias, (n, n, heads)), (2, 0, 1))
        attn = attn + ops.cast(bias, attn.dtype)[None]
        if shift_h or shift_w:
            key = (ph, pw)
            if key not in self._masks:
                self._masks[key] = shift_attention_mask(ph, pw, ws, shift_h, shift_w)
            mask = ops.convert_to_tensor(self._masks[key], dtype=attn.dtype)
            attn = ops.reshape(attn, (b, nh * nw, heads, n, n)) + mask[None, :, None]
            attn = ops.reshape(attn, (-1, heads, n, n))
        attn = ops.softmax(attn, axis=-1)

        x = ops.reshape(ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3)), (-1, n, c))
        x = self.proj(x)

        x = ops.reshape(x, (b, nh, nw, ws, ws, c))
        x = ops.reshape(ops.transpose(x, (0, 1, 3, 2, 4, 5)), (b, ph, pw, c))
        if shift_h or shift_w:
            x = ops.roll(x, shift=(shift_h, shift_w), axis=(1, 2))
        return x[:, :h, :w, :]

    def call(self, x, training=None):
        x = x + self.drop_path(self._attention(self.norm1(x)), training=training)
        return x + self.drop_path(self.mlp(self.norm2(x)), training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "drop_path": self.drop_path_rate,
        }


class PatchMerging(layers.Layer):
    """2×2 neighbourhood concat ``[x00, x10, x01, x11]`` → LayerNorm → linear ``4C → 2C``."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = _layer_norm("norm", 1e-5)
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")

    def build(self, input_shape):
        self.norm.build((None, 4 * self.dim))
        self.reduction.build((None, 4 * self.dim))

    def call(self, x):
        b, h, w, c = ops.shape(x)
        x = ops.pad(x, [[0, 0], [0, h % 2], [0, w % 2], [0, 0]])
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        x = ops.reshape(x, (b, h2, 2, w2, 2, c))
        # (row parity a, column parity b) flattened b-major gives x00, x10, x01, x11
        x = ops.reshape(ops.transpose(x, (0, 1, 3, 4, 2, 5)), (b, h2, w2, 4 * c))
        return self.reduction(self.norm(x))

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        half = lambda s: None if s is None else (s + 1) // 2  # noqa: E731
        return (b, half(h), half(w), 2 * self.dim)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


def input_normalization(
    in_channels: int = 3,
    extra_mean: list[float] | None = None,
    extra_std: list[float] | None = None,
) -> layers.Normalization:
    """ImageNet normalisation of the first three channels; channels after the third (e.g. the
    nDSM) use ``extra_mean``/``extra_std`` (default 127.5 ± 127.5, i.e. ``[0, 255] → [-1, 1]``)."""
    n_extra = in_channels - 3
    if n_extra < 0:
        raise ValueError(f"in_channels must be >= 3, got {in_channels}")
    extra_mean = list(extra_mean or [127.5] * n_extra)
    extra_std = list(extra_std or [127.5] * n_extra)
    if len(extra_mean) != n_extra or len(extra_std) != n_extra:
        raise ValueError(f"expected {n_extra} extra channel statistics")
    return layers.Normalization(
        mean=[m * 255.0 for m in IMAGENET_MEAN] + extra_mean,
        variance=[(s * 255.0) ** 2 for s in IMAGENET_STD] + [s**2 for s in extra_std],
        name="input_norm",
    )


def _normalized_input(in_channels: int = 3, extra_mean=None, extra_std=None):
    inputs = keras.Input((None, None, in_channels), name="image")
    return inputs, input_normalization(in_channels, extra_mean, extra_std)(inputs)


def build_swin(
    spec: SwinSpec,
    drop_path: float = 0.0,
    name: str = "swin_backbone",
    in_channels: int = 3,
    extra_mean: list[float] | None = None,
    extra_std: list[float] | None = None,
) -> keras.Model:
    inputs, x = _normalized_input(in_channels, extra_mean, extra_std)
    p = spec.patch_size
    x = layers.Conv2D(spec.embed_dim, p, strides=p, name="patch_embed")(x)
    x = _layer_norm("patch_norm", 1e-5)(x)

    total, block_id = sum(spec.depths), 0
    features = []
    for i, (depth, heads) in enumerate(zip(spec.depths, spec.num_heads, strict=True)):
        dim = spec.embed_dim * 2**i
        for j in range(depth):
            x = SwinBlock(
                dim,
                heads,
                spec.window_size,
                shift_size=0 if j % 2 == 0 else spec.window_size // 2,
                mlp_ratio=spec.mlp_ratio,
                drop_path=drop_path * block_id / max(total - 1, 1),
                name=f"stage{i + 1}_block{j}",
            )(x)
            block_id += 1
        features.append(_layer_norm(f"out_norm{i + 1}", 1e-5)(x))
        if i < 3:
            x = PatchMerging(dim, name=f"merge{i + 1}")(x)
    return keras.Model(inputs, features, name=name)


# --------------------------------------------------------------------------------------------
# Vision Transformer
# --------------------------------------------------------------------------------------------


class ViTEmbedding(layers.Layer):
    """Patch grid ``(B, h, w, D)`` → tokens ``(B, 1 + h·w, D)`` with class token and position
    embeddings (bicubically resized when the grid differs from the pre-training one)."""

    def __init__(self, dim: int, grid: int, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.grid = dim, grid

    def build(self, input_shape):
        init = keras.initializers.TruncatedNormal(stddev=0.02)
        self.class_token = self.add_weight(
            shape=(1, 1, self.dim), initializer="zeros", name="class_token"
        )
        self.pos_embedding = self.add_weight(
            shape=(1, 1 + self.grid**2, self.dim), initializer=init, name="pos_embedding"
        )

    def call(self, x):
        b, h, w, d = ops.shape(x)
        pos = ops.cast(self.pos_embedding, x.dtype)
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
        if (h, w) != (self.grid, self.grid):
            patch_pos = ops.reshape(patch_pos, (1, self.grid, self.grid, d))
            patch_pos = ops.image.resize(
                ops.cast(patch_pos, "float32"), (h, w), interpolation="bicubic", antialias=False
            )
            patch_pos = ops.cast(ops.reshape(patch_pos, (1, h * w, d)), x.dtype)
        tokens = ops.reshape(x, (b, h * w, d)) + patch_pos
        cls = ops.broadcast_to(ops.cast(self.class_token, x.dtype) + cls_pos, (b, 1, d))
        return ops.concatenate([cls, tokens], axis=1)

    def compute_output_shape(self, input_shape):
        b, h, w, d = input_shape
        return (b, None if h is None or w is None else 1 + h * w, d)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "grid": self.grid}


class ViTBlock(layers.Layer):
    """Pre-norm transformer block: ``x + MSA(LN(x))``, then ``x + MLP(LN(x))``."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads, self.mlp_dim, self.drop_path_rate = (
            dim, num_heads, mlp_dim, drop_path,
        )  # fmt: skip
        self.ln_1 = _layer_norm("ln_1", 1e-6)
        self.qkv = layers.Dense(3 * dim, name="qkv")
        self.proj = layers.Dense(dim, name="proj")
        self.ln_2 = _layer_norm("ln_2", 1e-6)
        self.mlp = _MLP(dim, mlp_dim, name="mlp")
        self.drop_path = DropPath(drop_path, name="drop_path")

    def build(self, input_shape):
        tokens = (None, self.dim)
        for layer in (self.ln_1, self.qkv, self.proj, self.ln_2, self.mlp, self.drop_path):
            layer.build(tokens)

    def _attention(self, x):
        b, n, d = ops.shape(x)
        qkv = ops.reshape(self.qkv(x), (b, n, 3, self.num_heads, d // self.num_heads))
        # (B, N, heads, head_dim) layout; scaled_dot_product_attention (flash when available)
        out = ops.dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(ops.reshape(out, (b, n, d)))

    def call(self, x, training=None):
        x = x + self.drop_path(self._attention(self.ln_1(x)), training=training)
        return x + self.drop_path(self.mlp(self.ln_2(x)), training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "drop_path": self.drop_path_rate,
        }


class TokensToMap(layers.Layer):
    """Drop the class token and fold ``(B, 1 + h·w, D)`` back to ``(B, h, w, D)``."""

    def call(self, inputs):
        tokens, grid = inputs
        _, h, w, _ = ops.shape(grid)
        return ops.reshape(tokens[:, 1:], (-1, h, w, ops.shape(tokens)[-1]))

    def compute_output_shape(self, input_shape):
        tokens_shape, grid_shape = input_shape
        return (*grid_shape[:3], tokens_shape[-1])


def build_vit(
    spec: ViTSpec,
    drop_path: float = 0.0,
    name: str = "vit_backbone",
    in_channels: int = 3,
    extra_mean: list[float] | None = None,
    extra_std: list[float] | None = None,
) -> keras.Model:
    inputs, x = _normalized_input(in_channels, extra_mean, extra_std)
    grid = layers.Conv2D(spec.dim, spec.patch_size, strides=spec.patch_size, name="patch_embed")(x)
    x = ViTEmbedding(spec.dim, spec.image_size // spec.patch_size, name="embedding")(grid)

    taps = []
    for i in range(spec.depth):
        x = ViTBlock(
            spec.dim,
            spec.num_heads,
            spec.mlp_dim or 4 * spec.dim,
            drop_path=drop_path * i / max(spec.depth - 1, 1),
            name=f"block{i}",
        )(x)
        if i in spec.taps:
            taps.append(TokensToMap(name=f"tokens_to_map{len(taps) + 1}")([x, grid]))
    # out_norm4 is the pre-trained final LayerNorm of the encoder
    taps = [_layer_norm(f"out_norm{i + 1}", 1e-6)(t) for i, t in enumerate(taps)]

    c1, c2, c3, c4 = vit_feature_channels(spec)
    f1 = layers.Conv2DTranspose(c2, 2, strides=2, name="fpn1_up1")(taps[0])
    f1 = layers.Activation(_gelu, name="fpn1_gelu")(layers.BatchNormalization(name="fpn1_bn")(f1))
    f1 = layers.Conv2DTranspose(c1, 2, strides=2, name="fpn1_up2")(f1)
    f2 = layers.Conv2DTranspose(c2, 2, strides=2, name="fpn2_up")(taps[1])
    f3 = taps[2]
    f4 = layers.MaxPooling2D(2, name="fpn4_pool")(taps[3])
    return keras.Model(inputs, [f1, f2, f3, f4], name=name)
