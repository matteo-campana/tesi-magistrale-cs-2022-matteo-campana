# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoders in Keras 3 returning the four feature levels ``[C1, C2, C3, C4]`` (strides 4, 8,
16, 32): ResNet-18/34/50 (defined here) and the Vision Transformers of
:mod:`rsseg.models.transformers` (Swin-T/S/B, ViT-S/B/L). :func:`build_backbone` and
:func:`feature_channels` accept any name of :data:`BACKBONES`.

``keras.applications`` offers no ResNet-18/34, the backbones used in the thesis, so the
networks are defined here with layer names mirroring torchvision (``layer1.0.conv1`` →
``layer1_0_conv1``) and ImageNet weights are ported from torchvision
(:mod:`rsseg.models.torchvision_weights`). Padding is explicit (``ZeroPadding2D`` + ``valid``)
so that outputs match PyTorch exactly, including strided layers.

Input: channels-last RGB in ``[0, 255]``; ImageNet normalisation is the first layer.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers

from rsseg.models.transformers import (
    IMAGENET_MEAN,  # noqa: F401  (re-exported: preprocessing constants of the encoders)
    IMAGENET_STD,  # noqa: F401
    SWIN_ARCHS,
    VIT_ARCHS,
    build_swin,
    build_vit,
    input_normalization,
    swin_feature_channels,
    vit_feature_channels,
)

ARCHS: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "resnet18": ("basic", (2, 2, 2, 2)),
    "resnet34": ("basic", (3, 4, 6, 3)),
    "resnet50": ("bottleneck", (3, 4, 6, 3)),
}
# Backbones and their ImageNet weights in torchvision (None: no ImageNet weights available,
# use ``weights: none`` or a checkpoint of ``rsseg pretrain``).
BACKBONES: dict[str, dict[str, str | None]] = {
    "resnet18": {"family": "ResNet", "imagenet": "ResNet18_Weights.IMAGENET1K_V1"},
    "resnet34": {"family": "ResNet", "imagenet": "ResNet34_Weights.IMAGENET1K_V1"},
    "resnet50": {"family": "ResNet", "imagenet": "ResNet50_Weights.IMAGENET1K_V2"},
    "swin_t": {"family": "Swin Transformer", "imagenet": "Swin_T_Weights.IMAGENET1K_V1"},
    "swin_s": {"family": "Swin Transformer", "imagenet": "Swin_S_Weights.IMAGENET1K_V1"},
    "swin_b": {"family": "Swin Transformer", "imagenet": "Swin_B_Weights.IMAGENET1K_V1"},
    "vit_s_16": {"family": "ViT", "imagenet": None},
    "vit_b_16": {"family": "ViT", "imagenet": "ViT_B_16_Weights.IMAGENET1K_V1"},
    "vit_l_16": {"family": "ViT", "imagenet": "ViT_L_16_Weights.IMAGENET1K_V1"},
}
# Transformer parameters that must not be decayed by AdamW (besides biases and norms).
NO_WEIGHT_DECAY = ["relative_position_bias_table", "pos_embedding", "class_token"]
BN_EPS = 1e-5
BN_MOMENTUM = 0.9  # = 1 - torch momentum (0.1)


def _check_arch(arch: str) -> None:
    if arch not in BACKBONES:
        raise ValueError(f"unknown arch {arch!r}, expected one of {list(BACKBONES)}")


def feature_channels(arch: str) -> list[int]:
    _check_arch(arch)
    if arch in SWIN_ARCHS:
        return swin_feature_channels(SWIN_ARCHS[arch])
    if arch in VIT_ARCHS:
        return vit_feature_channels(VIT_ARCHS[arch])
    block, _ = ARCHS[arch]
    expansion = 4 if block == "bottleneck" else 1
    return [64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]


def _conv(x, filters: int, kernel: int, stride: int, name: str):
    pad = kernel // 2
    if pad:
        x = layers.ZeroPadding2D(pad, name=f"{name}_pad")(x)
    return layers.Conv2D(
        filters, kernel, strides=stride, padding="valid", use_bias=False, name=name
    )(x)


def _bn(x, name: str):
    return layers.BatchNormalization(epsilon=BN_EPS, momentum=BN_MOMENTUM, name=name)(x)


def _basic_block(x, filters: int, stride: int, name: str):
    shortcut = x
    y = layers.ReLU(name=f"{name}_relu1")(
        _bn(_conv(x, filters, 3, stride, f"{name}_conv1"), f"{name}_bn1")
    )
    y = _bn(_conv(y, filters, 3, 1, f"{name}_conv2"), f"{name}_bn2")
    if stride != 1 or x.shape[-1] != filters:
        shortcut = _bn(_conv(x, filters, 1, stride, f"{name}_downsample_0"), f"{name}_downsample_1")
    return layers.ReLU(name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, shortcut]))


def _bottleneck_block(x, filters: int, stride: int, name: str):
    out = filters * 4
    shortcut = x
    y = layers.ReLU(name=f"{name}_relu1")(
        _bn(_conv(x, filters, 1, 1, f"{name}_conv1"), f"{name}_bn1")
    )
    y = layers.ReLU(name=f"{name}_relu2")(
        _bn(_conv(y, filters, 3, stride, f"{name}_conv2"), f"{name}_bn2")
    )
    y = _bn(_conv(y, out, 1, 1, f"{name}_conv3"), f"{name}_bn3")
    if stride != 1 or x.shape[-1] != out:
        shortcut = _bn(_conv(x, out, 1, stride, f"{name}_downsample_0"), f"{name}_downsample_1")
    return layers.ReLU(name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, shortcut]))


def build_resnet(
    arch: str = "resnet18",
    name: str | None = None,
    in_channels: int = 3,
    extra_mean: list[float] | None = None,
    extra_std: list[float] | None = None,
) -> keras.Model:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {list(ARCHS)}")
    block_type, depths = ARCHS[arch]
    block = _basic_block if block_type == "basic" else _bottleneck_block

    inputs = keras.Input((None, None, in_channels), name="image")
    x = input_normalization(in_channels, extra_mean, extra_std)(inputs)
    x = layers.ReLU(name="relu")(_bn(_conv(x, 64, 7, 2, "conv1"), "bn1"))
    x = layers.ZeroPadding2D(1, name="maxpool_pad")(x)  # safe: inputs are >= 0 after ReLU
    x = layers.MaxPooling2D(3, strides=2, name="maxpool")(x)

    features = []
    for i, (filters, depth) in enumerate(zip((64, 128, 256, 512), depths, strict=True), start=1):
        for j in range(depth):
            stride = 2 if (j == 0 and i > 1) else 1
            x = block(x, filters, stride, f"layer{i}_{j}")
        features.append(x)
    return keras.Model(inputs, features, name=name or f"{arch}_backbone")


# First layer of each encoder family: the only one whose shape depends on the input channels.
STEM_LAYERS = ("conv1", "patch_embed")
EXTRA_CHANNEL_INITS = ("mean", "zeros")


def _build_encoder(
    arch: str, drop_path: float, in_channels: int = 3, extra_mean=None, extra_std=None
) -> keras.Model:
    kw = {"in_channels": in_channels, "extra_mean": extra_mean, "extra_std": extra_std}
    if arch in SWIN_ARCHS:
        return build_swin(SWIN_ARCHS[arch], drop_path, name=f"{arch}_backbone", **kw)
    if arch in VIT_ARCHS:
        return build_vit(VIT_ARCHS[arch], drop_path, name=f"{arch}_backbone", **kw)
    return build_resnet(arch, **kw)


def inflate_input_channels(source: keras.Model, target: keras.Model, init: str = "mean") -> None:
    """Copy the weights of a 3-channel encoder into the same encoder with more input channels.

    The stem kernel keeps its RGB slice; the slices of the extra channels are the mean of the
    RGB slices (``mean``) or zero (``zeros``: the network computes exactly what the 3-channel one
    did, and learns to use the extra channels during fine-tuning). ``input_norm`` keeps the
    statistics of ``target``.
    """
    if init not in EXTRA_CHANNEL_INITS:
        raise ValueError(f"extra_channel_init must be one of {EXTRA_CHANNEL_INITS}, got {init!r}")
    for layer in source.layers:
        if not layer.weights or layer.name == "input_norm":
            continue
        weights = layer.get_weights()
        if layer.name in STEM_LAYERS:
            kernel = weights[0]  # (kh, kw, 3, out)
            n_extra = target.get_layer(layer.name).get_weights()[0].shape[2] - kernel.shape[2]
            if init == "mean":
                fill = kernel.mean(axis=2, keepdims=True)
            else:
                fill = np.zeros_like(kernel[:, :, :1])
            weights[0] = np.concatenate([kernel] + [fill] * n_extra, axis=2)
        target.get_layer(layer.name).set_weights(weights)


def build_backbone(
    arch: str = "resnet18",
    weights: str = "imagenet",
    drop_path: float = 0.0,
    in_channels: int = 3,
    extra_mean: list[float] | None = None,
    extra_std: list[float] | None = None,
    extra_channel_init: str = "mean",
) -> keras.Model:
    """Build an encoder and load ``weights``: ``imagenet``, ``none`` or a ``.weights.h5`` path.

    ``drop_path`` is the maximum stochastic-depth rate of transformer blocks (linearly increasing
    with depth); it is ignored by ResNets. With ``in_channels > 3`` the weights (always of a
    3-channel encoder: ImageNet or ``rsseg pretrain``) are loaded first and then inflated by
    :func:`inflate_input_channels`.
    """
    _check_arch(arch)
    no_weights = weights in (None, "", "none")
    if in_channels != 3 and no_weights:
        return _build_encoder(arch, drop_path, in_channels, extra_mean, extra_std)
    model = _build_encoder(arch, drop_path)
    if no_weights:
        return model
    if weights == "imagenet":
        from rsseg.models.torchvision_weights import load_imagenet_weights

        load_imagenet_weights(model, arch)
    else:
        model.load_weights(weights)
    if in_channels == 3:
        return model
    wide = _build_encoder(arch, drop_path, in_channels, extra_mean, extra_std)
    inflate_input_channels(model, wide, extra_channel_init)
    return wide
