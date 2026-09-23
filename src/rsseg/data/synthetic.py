# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic Potsdam-like tiles, used by the tests and to try the whole pipeline without the dataset.

Files are written with the same names as the real dataset (``top_potsdam_<id>_RGB.tif`` and
``top_potsdam_<id>_label.tif``), so ``rsseg prepare`` treats them exactly like real tiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rsseg.data.classes import NUM_CLASSES, PALETTE
from rsseg.data.splits import ALL_TILES

# A distinct mean colour per class, so the task is learnable.
_CLASS_COLOURS = np.array(
    [[128, 128, 128], [180, 90, 60], [120, 200, 90], [30, 110, 40], [220, 40, 40], [60, 60, 200]],
    dtype=np.float32,
)


def make_tile(size: int, rng: np.random.Generator, n_shapes: int = 24) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_shapes):
        c = int(rng.integers(0, NUM_CLASSES))
        h, w = rng.integers(size // 16, size // 3, size=2)
        y, x = rng.integers(0, size - h), rng.integers(0, size - w)
        labels[y : y + h, x : x + w] = c
    image = _CLASS_COLOURS[labels] + rng.normal(0, 12, size=(size, size, 3))
    return np.clip(image, 0, 255).astype(np.uint8), PALETTE[labels]


def make_synthetic_potsdam(out_dir: str | Path, size: int = 256, tiles: tuple[str, ...] = ALL_TILES, seed: int = 0) -> Path:
    out = Path(out_dir)
    (out / "2_Ortho_RGB").mkdir(parents=True, exist_ok=True)
    (out / "5_Labels_all").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for tile in tiles:
        image, label = make_tile(size, rng)
        Image.fromarray(image).save(out / "2_Ortho_RGB" / f"top_potsdam_{tile}_RGB.tif")
        Image.fromarray(label).save(out / "5_Labels_all" / f"top_potsdam_{tile}_label.tif")
    return out
