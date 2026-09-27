# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whole-tile evaluation with overlapping sliding windows.

Tiles are evaluated whole (6000×6000 for Potsdam) instead of pre-cut crops: every pixel is
scored exactly once, and predictions near crop borders get context from neighbouring windows.
Softmax probabilities of overlapping windows are blended with a separable Hann-like weight.
Memory: the probability buffer lives on the CPU (6000² × 6 × 4 B ≈ 0.9 GB), only one batch of
windows is on the GPU at a time, and the forward pass runs under ``torch.no_grad()``: calling a
Keras model directly on the torch backend otherwise records the autograd graph, which for
ResNet-34 at 8 × 512² does not fit in 8 GB.
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
import torch

from rsseg.data.prepare import crop_positions


def _window_weight(size: int) -> np.ndarray:
    ramp = np.hanning(size + 2)[1:-1].astype(np.float32)
    w = np.outer(ramp, ramp)
    return np.maximum(w, 1e-3)


def infer(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """Logits of ``batch`` as NumPy, without building the autograd graph."""
    with torch.no_grad():
        return keras.ops.convert_to_numpy(model(batch, training=False))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _predict_batch(model: keras.Model, batch: np.ndarray, tta: bool) -> np.ndarray:
    probs = _softmax(infer(model, batch))
    if tta:
        flipped = _softmax(infer(model, batch[:, :, ::-1].copy()))
        probs = 0.5 * (probs + flipped[:, :, ::-1])
    return probs


def predict_tile(
    model: keras.Model,
    image: np.ndarray,
    num_classes: int,
    window: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    tta: bool = False,
    confidence: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the ``(H, W)`` argmax prediction for an ``(H, W, C)`` image (RGB, or RGB + nDSM).

    With ``confidence`` also return the ``(H, W)`` float32 probability of the predicted class.
    ``on_progress(done, total)`` is called after every batch of windows.
    """
    if window % 32:
        raise ValueError("window must be a multiple of 32")
    h, w = image.shape[:2]
    ph, pw = max(h, window), max(w, window)
    if (ph, pw) != (h, w):
        image = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
    probs = np.zeros((ph, pw, num_classes), dtype=np.float32)
    weights = np.zeros((ph, pw, 1), dtype=np.float32)
    win_w = _window_weight(window)[..., None]

    coords = [
        (y, x)
        for y in crop_positions(ph, window, stride)
        for x in crop_positions(pw, window, stride)
    ]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        batch = np.stack([image[y : y + window, x : x + window] for y, x in chunk]).astype(
            np.float32
        )
        out = _predict_batch(model, batch, tta)
        for (y, x), p in zip(chunk, out, strict=True):
            probs[y : y + window, x : x + window] += p * win_w
            weights[y : y + window, x : x + window] += win_w
        if on_progress is not None:
            on_progress(min(i + batch_size, len(coords)), len(coords))
    probs /= weights
    pred = probs.argmax(-1)[:h, :w].astype(np.uint8)
    if confidence:
        return pred, probs.max(-1)[:h, :w]
    return pred
