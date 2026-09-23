# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single source of truth for the ISPRS Potsdam classes, colours and label encoding.

The class order follows ISPRS/MMSegmentation (clutter last). The thesis code used a
different order with clutter as index 0 — see :data:`LEGACY_TO_INDEX`.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES: tuple[str, ...] = (
    "impervious_surfaces",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
)
NUM_CLASSES = len(CLASS_NAMES)
CLUTTER = CLASS_NAMES.index("clutter")
IGNORE_INDEX = 255

PALETTE = np.array(
    [
        [255, 255, 255],  # impervious surfaces
        [0, 0, 255],  # building
        [0, 255, 255],  # low vegetation
        [0, 255, 0],  # tree
        [255, 255, 0],  # car
        [255, 0, 0],  # clutter / background
    ],
    dtype=np.uint8,
)

# Legacy (thesis 2022) index -> current index.
# Legacy order: 0 clutter, 1 impervious, 2 car, 3 building, 4 low vegetation, 5 tree.
LEGACY_TO_INDEX = {0: 5, 1: 0, 2: 4, 3: 1, 4: 2, 5: 3}

_CODES = PALETTE[:, 0].astype(np.int64) << 16 | PALETTE[:, 1].astype(np.int64) << 8 | PALETTE[:, 2]
_ORDER = np.argsort(_CODES)
_SORTED_CODES = _CODES[_ORDER]


def rgb_to_index(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, 3)`` uint8 colour mask to ``(H, W)`` class indices.

    Colours not in :data:`PALETTE` (e.g. anti-aliased borders) become :data:`IGNORE_INDEX`.
    Vectorised: a sorted lookup over packed 24-bit colour codes.
    """
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"expected (H, W, 3) array, got {rgb.shape}")
    rgb = rgb[..., :3].astype(np.int32)  # 24-bit codes fit in int32 (half the memory of int64)
    codes = rgb[..., 0] << 16 | rgb[..., 1] << 8 | rgb[..., 2]
    pos = np.clip(np.searchsorted(_SORTED_CODES, codes), 0, len(_SORTED_CODES) - 1)
    found = _SORTED_CODES[pos] == codes
    return np.where(found, _ORDER[pos], IGNORE_INDEX).astype(np.uint8)


def index_to_rgb(index: np.ndarray) -> np.ndarray:
    """Convert ``(H, W)`` class indices to an ``(H, W, 3)`` colour mask (ignored pixels black)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:NUM_CLASSES] = PALETTE
    return lut[index.astype(np.uint8)]


def legacy_to_index(legacy: np.ndarray) -> np.ndarray:
    """Remap a mask encoded with the legacy (thesis) class order to the current order."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for old, new in LEGACY_TO_INDEX.items():
        lut[old] = new
    return lut[legacy.astype(np.uint8)]
