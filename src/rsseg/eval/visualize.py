# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qualitative figures in the style of thesis figures 4.3–4.4: TOP | GT | prediction | errors."""

from __future__ import annotations

import numpy as np
from PIL import Image

from rsseg.data.classes import IGNORE_INDEX, index_to_rgb


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[(gt == pred)] = (40, 40, 40)
    out[(gt != pred) & (gt != IGNORE_INDEX)] = (255, 0, 255)
    return out


def panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, gap: int = 4) -> np.ndarray:
    """One row ``[image | gt | pred | errors]`` as uint8 RGB."""
    parts = [
        np.clip(image, 0, 255).astype(np.uint8),
        index_to_rgb(gt),
        index_to_rgb(pred),
        error_map(gt, pred),
    ]
    h = image.shape[0]
    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    row = []
    for p in parts:
        row += [p, spacer]
    return np.concatenate(row[:-1], axis=1)


def save_panels(path, images, gts, preds) -> None:
    rows = [panel(i, g, p) for i, g, p in zip(images, gts, preds, strict=True)]
    gap = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked = []
    for r in rows:
        stacked += [r, gap]
    Image.fromarray(np.concatenate(stacked[:-1], axis=0)).save(path)
