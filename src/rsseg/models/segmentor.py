# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder–decoder segmentation network (MAResU-Net family) built from interchangeable parts.

::

    image ─► Backbone ─► [C1, C2, C3, C4]
                          │    │    │    └─► skip(C4) ─► Neck (identity | VQ) ─┐
                          │    │    └──────► skip(C3) ──────────────── + ◄─ Dec4
                          │    └───────────► skip(C2) ──────────────── + ◄─ Dec3
                          └────────────────► skip(C1) ──────────────── + ◄─ Dec2
                                                                           Dec1 ─► Head ─► logits

``skip = lam`` applies the MAResU-Net PAM/CAM attention block, ``plain`` passes features through
(U-Net/LinkNet-style). ``neck = vq`` quantizes the bottleneck; ``vq.on_skips`` also quantizes
the skip connections so that the decoder cannot bypass the codebook.

Input height and width must be multiples of 32. Logits are float32 even under mixed precision.
"""

from __future__ import annotations

import keras
from keras import layers

from rsseg.config import ModelConfig
from rsseg.models.attention import pam_cam_block
from rsseg.models.backbones import build_backbone, feature_channels
from rsseg.models.quantizer import VectorQuantizer


def _conv_bn_relu(x, filters: int, kernel: int, name: str):
    x = layers.Conv2D(filters, kernel, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def decoder_block(x, out_channels: int, name: str):
    """1×1 reduce → 3×3 transposed conv (×2) → 1×1 expand, each with BN + ReLU."""
    mid = max(x.shape[-1] // 4, 1)
    x = _conv_bn_relu(x, mid, 1, f"{name}_reduce")
    x = layers.Conv2DTranspose(
        mid, 3, strides=2, padding="same", use_bias=False, name=f"{name}_up"
    )(x)
    x = layers.ReLU(name=f"{name}_up_relu")(layers.BatchNormalization(name=f"{name}_up_bn")(x))
    return _conv_bn_relu(x, out_channels, 1, f"{name}_expand")


def _quantize(x, cfg: ModelConfig, name: str, seed: int):
    vq = cfg.vq
    channels = x.shape[-1]
    code_dim = vq.code_dim or channels
    if code_dim != channels:
        x = layers.Conv2D(code_dim, 1, name=f"{name}_pre")(x)
    x = VectorQuantizer(
        num_codes=vq.num_codes,
        beta=vq.beta,
        ema_decay=vq.ema_decay,
        loss_weight=vq.loss_weight,
        restart_dead_codes=vq.restart_dead_codes,
        dead_code_threshold=vq.dead_code_threshold,
        seed=seed,
        name=name,
    )(x)
    if code_dim != channels:
        x = layers.Conv2D(channels, 1, name=f"{name}_post")(x)
    return x


def build_segmentor(
    cfg: ModelConfig, name: str | None = None, backbone: keras.Model | None = None
) -> keras.Model:
    if cfg.skip not in ("lam", "plain"):
        raise ValueError(f"model.skip must be 'lam' or 'plain', got {cfg.skip!r}")
    if cfg.neck not in ("identity", "vq"):
        raise ValueError(f"model.neck must be 'identity' or 'vq', got {cfg.neck!r}")

    backbone = backbone or build_backbone(cfg.arch, cfg.weights)
    filters = feature_channels(cfg.arch)

    inputs = keras.Input((None, None, 3), name="image")
    c1, c2, c3, c4 = backbone(inputs)

    def skip(x, level: int):
        if cfg.skip == "lam":
            x = pam_cam_block(x, filters[level], cfg.dropout, name=f"attention{level + 1}")
        if cfg.neck == "vq" and cfg.vq.on_skips and level < 3:
            x = _quantize(x, cfg, f"vq_skip{level + 1}", seed=level)
        return x

    s1, s2, s3, s4 = skip(c1, 0), skip(c2, 1), skip(c3, 2), skip(c4, 3)
    bottleneck = _quantize(s4, cfg, "vq", seed=3) if cfg.neck == "vq" else s4

    d4 = layers.Add(name="skip3")([decoder_block(bottleneck, filters[2], "decoder4"), s3])
    d3 = layers.Add(name="skip2")([decoder_block(d4, filters[1], "decoder3"), s2])
    d2 = layers.Add(name="skip1")([decoder_block(d3, filters[0], "decoder2"), s1])
    d1 = decoder_block(d2, filters[0], "decoder1")

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu", name="head_up")(
        d1
    )
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="head_conv")(x)
    logits = layers.Conv2D(cfg.num_classes, 3, padding="same", dtype="float32", name="logits")(x)

    default_name = f"{'maresunet' if cfg.skip == 'lam' else 'unet'}{'_vq' if cfg.neck == 'vq' else ''}_{cfg.arch}"
    return keras.Model(inputs, logits, name=name or default_name)
