# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic ISPRS-like tiles, used by the tests and to try the whole pipeline without the dataset.

Files are written with the same names and folders as the real datasets
(:mod:`rsseg.data.sources`), so ``rsseg prepare`` treats them exactly like real tiles:
``top_potsdam_<id>_RGB.tif`` / ``_label.tif`` (plus eroded labels, official nDSM and DSM on
request) for Potsdam, ``top_mosaic_09cm_area<n>.tif`` and the matching ground truth and DSM for
Vaihingen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rsseg.data.classes import NUM_CLASSES, PALETTE
from rsseg.data.splits import ALL_TILES, VAIHINGEN_ALL

# A distinct mean colour per class, so the task is learnable.
_CLASS_COLOURS = np.array(
    [[128, 128, 128], [180, 90, 60], [120, 200, 90], [30, 110, 40], [220, 40, 40], [60, 60, 200]],
    dtype=np.float32,
)
# Height above ground (metres) per class: building and tree stand out, as in a real nDSM.
_CLASS_HEIGHTS = np.array([0.0, 12.0, 0.2, 7.0, 1.5, 1.0], dtype=np.float32)


def make_tile(
    size: int, rng: np.random.Generator, n_shapes: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    labels = make_labels(size, rng, n_shapes)
    image = _CLASS_COLOURS[labels] + rng.normal(0, 12, size=(size, size, 3))
    return np.clip(image, 0, 255).astype(np.uint8), PALETTE[labels]


def make_labels(size: int, rng: np.random.Generator, n_shapes: int = 24) -> np.ndarray:
    labels = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_shapes):
        c = int(rng.integers(0, NUM_CLASSES))
        h, w = rng.integers(size // 16, size // 3, size=2)
        y, x = rng.integers(0, size - h), rng.integers(0, size - w)
        labels[y : y + h, x : x + w] = c
    return labels


def erode_label(rgb: np.ndarray, radius: int = 1) -> np.ndarray:
    """Black out pixels within ``radius`` of a class boundary, like the ISPRS eroded labels."""
    codes = rgb.astype(np.int32) @ np.array([1 << 16, 1 << 8, 1], dtype=np.int32)
    padded = np.pad(codes, radius, mode="edge")
    h, w = codes.shape
    boundary = np.zeros((h, w), dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            boundary |= padded[dy : dy + h, dx : dx + w] != codes
    out = rgb.copy()
    out[boundary] = 0
    return out


def heights_for(rgb_label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Height above ground (metres) of a colour label."""
    codes = rgb_label.astype(np.int32) @ np.array([1 << 16, 1 << 8, 1], dtype=np.int32)
    pal = PALETTE.astype(np.int32) @ np.array([1 << 16, 1 << 8, 1], dtype=np.int32)
    idx = np.argmax(codes[..., None] == pal, axis=-1)
    return _CLASS_HEIGHTS[idx] + rng.normal(0, 0.1, size=idx.shape).astype(np.float32)


def _ground(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return 100.0 + 0.002 * x + 0.001 * y  # gentle slope: the nDSM must remove it


def make_synthetic_potsdam(
    out_dir: str | Path,
    size: int = 256,
    tiles: tuple[str, ...] = ALL_TILES,
    seed: int = 0,
    extras: bool = False,
) -> Path:
    """With ``extras`` also write eroded labels, the official-style nDSM (JPG, decimetres) and
    the absolute DSM (float TIFF, zero-padded ids as in the real dataset)."""
    out = Path(out_dir)
    dirs = ["2_Ortho_RGB", "5_Labels_all"]
    if extras:
        dirs += ["5_Labels_all_noBoundary", "1_DSM", "1_DSM_normalisation"]
    for d in dirs:
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for tile in tiles:
        image, label = make_tile(size, rng)
        Image.fromarray(image).save(out / "2_Ortho_RGB" / f"top_potsdam_{tile}_RGB.tif")
        Image.fromarray(label).save(out / "5_Labels_all" / f"top_potsdam_{tile}_label.tif")
        if not extras:
            continue
        Image.fromarray(erode_label(label)).save(
            out / "5_Labels_all_noBoundary" / f"top_potsdam_{tile}_label_noBoundary.tif"
        )
        heights = heights_for(label, rng)
        padded = "_".join(f"{int(p):02d}" for p in tile.split("_"))
        Image.fromarray(np.round(np.clip(heights, 0, 25.5) * 10).astype(np.uint8)).save(
            out / "1_DSM_normalisation" / f"dsm_potsdam_{padded}_normalized_lastools.jpg",
            quality=95,
        )
        Image.fromarray(_ground(size) + heights).save(out / "1_DSM" / f"dsm_potsdam_{padded}.tif")
    return out


def make_synthetic_vaihingen(
    out_dir: str | Path, size: int = 256, tiles: tuple[str, ...] = VAIHINGEN_ALL, seed: int = 0
) -> Path:
    """Vaihingen layout: IRRG tops, full and eroded ground truth, absolute DSM (no nDSM)."""
    out = Path(out_dir)
    base = out / "ISPRS_semantic_labeling_Vaihingen"
    dirs = {
        "top": base / "top",
        "dsm": base / "dsm",
        "gt": out / "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE",
        "eroded": out / "ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for area in tiles:
        # areas have different sizes in the real dataset
        h = size + int(rng.integers(0, size // 4))
        image, label = make_tile(max(h, size), rng)
        image, label = image[:h, :size], label[:h, :size]
        name = f"top_mosaic_09cm_area{area}"
        Image.fromarray(image).save(dirs["top"] / f"{name}.tif")
        Image.fromarray(label).save(dirs["gt"] / f"{name}.tif")
        Image.fromarray(erode_label(label)).save(dirs["eroded"] / f"{name}_noBoundary.tif")
        dsm = _ground(max(h, size))[:h, :size] + heights_for(label, rng)
        Image.fromarray(dsm.astype(np.float32)).save(
            dirs["dsm"] / f"dsm_09cm_matching_area{area}.tif"
        )
    return out
