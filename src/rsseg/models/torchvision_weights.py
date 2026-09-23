# SPDX-License-Identifier: AGPL-3.0-or-later
"""Port torchvision ``state_dict`` tensors (ResNet, Swin, ViT) into the Keras encoders of
:mod:`backbones` and :mod:`transformers`."""

from __future__ import annotations

from collections.abc import Mapping

import keras
import numpy as np


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def port_resnet_state_dict(model: keras.Model, state_dict: Mapping[str, object]) -> None:
    """Copy conv/BN parameters by name; raises if any layer or tensor is left unmatched."""
    used: set[str] = set()
    for layer in model.layers:
        prefix = layer.name.replace("_", ".")
        if isinstance(layer, keras.layers.Conv2D):
            key = f"{prefix}.weight"
            w = _np(state_dict[key]).transpose(2, 3, 1, 0)  # (out, in, kh, kw) -> (kh, kw, in, out)
            layer.set_weights([w])
            used.add(key)
        elif isinstance(layer, keras.layers.BatchNormalization):
            keys = [f"{prefix}.{k}" for k in ("weight", "bias", "running_mean", "running_var")]
            layer.set_weights([_np(state_dict[k]) for k in keys])
            used.update(keys)
    expected = {
        k for k in state_dict if not k.startswith("fc.") and not k.endswith("num_batches_tracked")
    }
    missing = expected - used
    if missing:
        raise KeyError(f"torchvision tensors not ported: {sorted(missing)[:10]}")


class _Porter:
    """Copy tensors by key and check at the end that every expected key was consumed."""

    def __init__(self, state_dict: Mapping[str, object]):
        self.state_dict = state_dict
        self.used: set[str] = set()

    def get(self, key: str) -> np.ndarray:
        self.used.add(key)
        return _np(self.state_dict[key])

    def dense(self, layer, prefix: str) -> None:
        w = [self.get(f"{prefix}.weight").T]
        if layer.use_bias:
            w.append(self.get(f"{prefix}.bias"))
        layer.set_weights(w)

    def norm(self, layer, prefix: str) -> None:
        layer.set_weights([self.get(f"{prefix}.weight"), self.get(f"{prefix}.bias")])

    def conv(self, layer, prefix: str) -> None:
        w = self.get(f"{prefix}.weight").transpose(2, 3, 1, 0)
        layer.set_weights([w, self.get(f"{prefix}.bias")] if layer.use_bias else [w])

    def check(self, ignore: tuple[str, ...]) -> None:
        missing = {k for k in self.state_dict if not k.startswith(ignore)} - self.used
        if missing:
            raise KeyError(f"torchvision tensors not ported: {sorted(missing)[:10]}")


def port_swin_state_dict(model: keras.Model, state_dict: Mapping[str, object]) -> None:
    """``torchvision.models.SwinTransformer`` → :func:`rsseg.models.transformers.build_swin`.

    ``features`` alternates stages and patch merging: ``features.0`` is the patch embedding,
    stage ``i`` (0-based) is ``features.{2i+1}`` and its merging layer ``features.{2i+2}``.
    """
    p = _Porter(state_dict)
    p.conv(model.get_layer("patch_embed"), "features.0.0")
    p.norm(model.get_layer("patch_norm"), "features.0.2")
    for i in range(4):
        j = 0
        while f"features.{2 * i + 1}.{j}.norm1.weight" in state_dict:
            block, pre = model.get_layer(f"stage{i + 1}_block{j}"), f"features.{2 * i + 1}.{j}"
            p.norm(block.norm1, f"{pre}.norm1")
            p.dense(block.qkv, f"{pre}.attn.qkv")
            p.dense(block.proj, f"{pre}.attn.proj")
            block.relative_position_bias_table.assign(
                p.get(f"{pre}.attn.relative_position_bias_table")
            )
            p.norm(block.norm2, f"{pre}.norm2")
            p.dense(block.mlp.fc1, f"{pre}.mlp.0")
            p.dense(block.mlp.fc2, f"{pre}.mlp.3")
            j += 1
        if i < 3:
            merge = model.get_layer(f"merge{i + 1}")
            p.norm(merge.norm, f"features.{2 * i + 2}.norm")
            p.dense(merge.reduction, f"features.{2 * i + 2}.reduction")
    p.norm(model.get_layer("out_norm4"), "norm")
    p.check(ignore=("head.",) + tuple(k for k in state_dict if "relative_position_index" in k))


def port_vit_state_dict(model: keras.Model, state_dict: Mapping[str, object]) -> None:
    """``torchvision.models.VisionTransformer`` → :func:`rsseg.models.transformers.build_vit`."""
    p = _Porter(state_dict)
    p.conv(model.get_layer("patch_embed"), "conv_proj")
    emb = model.get_layer("embedding")
    emb.class_token.assign(p.get("class_token"))
    emb.pos_embedding.assign(p.get("encoder.pos_embedding"))
    i = 0
    while f"encoder.layers.encoder_layer_{i}.ln_1.weight" in state_dict:
        block, pre = model.get_layer(f"block{i}"), f"encoder.layers.encoder_layer_{i}"
        p.norm(block.ln_1, f"{pre}.ln_1")
        # nn.MultiheadAttention packs q/k/v (heads-major inside each) in ``in_proj_*``
        block.qkv.set_weights(
            [
                p.get(f"{pre}.self_attention.in_proj_weight").T,
                p.get(f"{pre}.self_attention.in_proj_bias"),
            ]
        )
        p.dense(block.proj, f"{pre}.self_attention.out_proj")
        p.norm(block.ln_2, f"{pre}.ln_2")
        p.dense(block.mlp.fc1, f"{pre}.mlp.0")
        p.dense(block.mlp.fc2, f"{pre}.mlp.3")
        i += 1
    p.norm(model.get_layer("out_norm4"), "encoder.ln")
    p.check(ignore=("heads.",))


def load_imagenet_weights(model: keras.Model, arch: str) -> None:
    import torchvision

    from rsseg.models.backbones import BACKBONES

    name = BACKBONES[arch]["imagenet"]
    if name is None:
        raise ValueError(
            f"no ImageNet weights for {arch!r} in torchvision: use weights=none or a "
            "backbone pre-trained with `rsseg pretrain`"
        )
    net = getattr(torchvision.models, arch)(weights=torchvision.models.get_weight(name))
    family = BACKBONES[arch]["family"]
    if family == "Swin Transformer":
        port_swin_state_dict(model, net.state_dict())
    elif family == "ViT":
        port_vit_state_dict(model, net.state_dict())
    else:
        port_resnet_state_dict(model, net.state_dict())
