# SPDX-License-Identifier: AGPL-3.0-or-later
"""Port torchvision ResNet ``state_dict`` tensors into the Keras encoders of :mod:`backbones`."""

from __future__ import annotations

from collections.abc import Mapping

import keras
import numpy as np

_TORCHVISION_WEIGHTS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V2",
}


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


def load_imagenet_weights(model: keras.Model, arch: str) -> None:
    import torchvision

    weights = torchvision.models.get_weight(_TORCHVISION_WEIGHTS[arch])
    net = getattr(torchvision.models, arch)(weights=weights)
    port_resnet_state_dict(model, net.state_dict())
