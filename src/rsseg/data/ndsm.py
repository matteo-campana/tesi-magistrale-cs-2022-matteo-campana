# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalised DSM (nDSM, height above ground) used as an extra input channel.

Encoding shared by every source: ``uint8`` in **decimetres**, heights above 25.5 m saturate at
255. It is the encoding of the official Potsdam nDSM (``1_DSM_normalisation``, produced by ISPRS
with lastools or an "own approach"; no manual check, see the readme in that folder).

Vaihingen ships only the absolute DSM, so :func:`compute_ndsm` estimates the ground with a
morphological opening: objects narrower than ``window_m`` (buildings, trees, cars) are removed,
the remaining surface is the terrain. It is an approximation — large buildings wider than the
window keep part of their height in the ground estimate, and DSM outliers below the ground lower
it locally — but it is simple, deterministic and applicable to both datasets
(``data.ndsm_source: computed``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

MAX_HEIGHT_M = 25.5
DM_PER_M = 10.0


def load_official_ndsm(path) -> np.ndarray:
    """Official Potsdam nDSM JPG → ``(H, W)`` uint8 (decimetres)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def load_dsm(path) -> np.ndarray:
    """Absolute DSM (float TIFF, metres) → ``(H, W)`` float32."""
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.float32)


def encode_heights(heights_m: np.ndarray) -> np.ndarray:
    return np.round(np.clip(heights_m, 0, MAX_HEIGHT_M) * DM_PER_M).astype(np.uint8)


def _filter1d(a: np.ndarray, size: int, axis: int, op) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (size // 2, size // 2)
    view = sliding_window_view(np.pad(a, pad, mode="edge"), size, axis=axis)
    return op(view, axis=-1)


def _filter2d(a: np.ndarray, size: int, op) -> np.ndarray:
    """Separable ``size × size`` min/max/mean filter with edge padding."""
    return _filter1d(_filter1d(a, size, 0, op), size, 1, op)


def estimate_ground(
    dsm: np.ndarray, gsd: float, window_m: float = 30.0, cell_m: float = 1.0
) -> np.ndarray:
    """Terrain height (metres) under ``dsm``: grey opening on a coarse grid, smoothed, upsampled."""
    h, w = dsm.shape
    f = max(1, round(cell_m / gsd))
    hc, wc = math.ceil(h / f), math.ceil(w / f)
    padded = np.pad(dsm, ((0, hc * f - h), (0, wc * f - w)), mode="edge")
    # the ground is the lowest point of each cell
    coarse = padded.reshape(hc, f, wc, f).min(axis=(1, 3)).astype(np.float32)
    k = max(3, round(window_m / (f * gsd)) | 1)
    ground = _filter2d(_filter2d(coarse, k, np.min), k, np.max)  # opening
    ground = _filter2d(ground, max(3, (k // 2) | 1), np.mean)
    up = Image.fromarray(ground.astype(np.float32)).resize(
        (wc * f, hc * f), Image.Resampling.BILINEAR
    )
    return np.asarray(up, dtype=np.float32)[:h, :w]


def compute_ndsm(
    dsm: np.ndarray, gsd: float, window_m: float = 30.0, cell_m: float = 1.0
) -> np.ndarray:
    """Absolute DSM (metres) → nDSM ``uint8`` in decimetres (see the module docstring)."""
    dsm = np.asarray(dsm, dtype=np.float32)
    return encode_heights(dsm - estimate_ground(dsm, gsd, window_m, cell_m))
