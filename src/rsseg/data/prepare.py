# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn raw ISPRS Potsdam tiles into training crops, full-tile masks and manifests.

Order of operations (the order matters to avoid leakage):

1. assign every *tile* to a split (:func:`rsseg.data.splits.get_split`);
2. convert colour labels to class indices with a vectorised lookup;
3. ``train``/``val`` tiles are cut into crops (overlap allowed only for ``train``);
4. ``val``/``test`` tiles are also registered whole for sliding-window evaluation;
5. per-channel statistics are computed on the training tiles only.

Output layout under ``cfg.root``::

    crops/<split>/<tile>_<y>_<x>.png        RGB crop
    crops/<split>/<tile>_<y>_<x>_mask.png   class-index crop (uint8)
    tiles/<tile>_mask.png                   class-index mask of val/test tiles
    crops.csv   tiles.csv   stats.json
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rsseg.config import DataConfig
from rsseg.data.classes import IGNORE_INDEX, NUM_CLASSES, rgb_to_index
from rsseg.data.splits import get_split

Image.MAX_IMAGE_PIXELS = None  # Potsdam tiles are 6000x6000
log = logging.getLogger(__name__)

CROP_FIELDS = ["split", "tile", "y", "x", "image", "mask"] + [f"n{c}" for c in range(NUM_CLASSES)]
TILE_FIELDS = ["split", "tile", "image", "mask"]


@dataclass(frozen=True)
class TileFiles:
    tile: str
    image: Path
    label: Path


def find_tiles(raw_dir: str | Path, image_pattern: str, label_pattern: str) -> dict[str, TileFiles]:
    """Match RGB images and colour labels by tile id (searched recursively)."""
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    img_re, lab_re = re.compile(image_pattern), re.compile(label_pattern)
    for path in sorted(Path(raw_dir).rglob("*")):
        if not path.is_file():
            continue
        if m := img_re.search(path.name):
            images[m.group(1)] = path
        elif m := lab_re.search(path.name):
            labels[m.group(1)] = path
    missing_labels = sorted(set(images) - set(labels))
    if missing_labels:
        log.warning("tiles without labels (ignored): %s", missing_labels)
    return {t: TileFiles(t, images[t], labels[t]) for t in sorted(set(images) & set(labels))}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im)


def crop_positions(length: int, size: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, length)``; the last crop is aligned to the border."""
    if length <= size:
        return [0]
    pos = list(range(0, length - size + 1, stride))
    if pos[-1] != length - size:
        pos.append(length - size)
    return pos


def _process_tile(args: tuple[TileFiles, str, Path, int, int, bool]) -> tuple[list[dict], dict | None, np.ndarray | None]:
    files, split, root, size, stride, make_crops = args
    image = load_rgb(files.image)
    mask = rgb_to_index(load_rgb(files.label))
    if image.shape[:2] != mask.shape:
        raise ValueError(f"tile {files.tile}: image {image.shape} and label {mask.shape} differ")

    rows: list[dict] = []
    if make_crops:
        out_dir = root / "crops" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = mask.shape
        for y in crop_positions(h, size, stride):
            for x in crop_positions(w, size, stride):
                m = mask[y : y + size, x : x + size]
                if (m == IGNORE_INDEX).all():
                    continue
                stem = f"{files.tile}_{y}_{x}"
                img_path = out_dir / f"{stem}.png"
                mask_path = out_dir / f"{stem}_mask.png"
                Image.fromarray(image[y : y + size, x : x + size]).save(img_path)
                Image.fromarray(m).save(mask_path)
                hist = np.bincount(m[m != IGNORE_INDEX].ravel(), minlength=NUM_CLASSES)
                rows.append(
                    {"split": split, "tile": files.tile, "y": y, "x": x,
                     "image": img_path.relative_to(root).as_posix(),
                     "mask": mask_path.relative_to(root).as_posix(),
                     **{f"n{c}": int(hist[c]) for c in range(NUM_CLASSES)}}
                )  # fmt: skip

    tile_row = None
    if split in ("val", "test"):
        tiles_dir = root / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        mask_path = tiles_dir / f"{files.tile}_mask.png"
        Image.fromarray(mask).save(mask_path)
        tile_row = {"split": split, "tile": files.tile, "image": str(files.image.resolve()),
                    "mask": mask_path.relative_to(root).as_posix()}  # fmt: skip

    stats = None
    if split == "train":
        pix = image.reshape(-1, 3).astype(np.float64)
        stats = np.stack([pix.sum(0), (pix**2).sum(0), np.full(3, len(pix), dtype=np.float64)])
    return rows, tile_row, stats


def prepare_dataset(cfg: DataConfig) -> dict[str, int]:
    """Run the full preparation; returns the number of crops per split."""
    if not cfg.raw_dir:
        raise ValueError("data.raw_dir is required for `rsseg prepare`")
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    tiles = find_tiles(cfg.raw_dir, cfg.image_pattern, cfg.label_pattern)
    if not tiles:
        raise FileNotFoundError(f"no Potsdam tiles found in {cfg.raw_dir}")
    split = get_split(cfg.protocol, cfg.val_tiles, cfg.exclude_7_10)

    jobs: list[tuple[TileFiles, str, Path, int, int, bool]] = []
    # A tile may appear in both val and test (thesis2022): crops are produced once for val.
    seen: set[tuple[str, str]] = set()
    for name in ("train", "val", "test"):
        missing = [t for t in split[name] if t not in tiles]
        if missing:
            log.warning("%s tiles not found in raw_dir: %s", name, missing)
        for t in split[name]:
            if t in tiles and (name, t) not in seen:
                seen.add((name, t))
                stride = cfg.train_stride if name == "train" else cfg.crop_size
                jobs.append((tiles[t], name, root, cfg.crop_size, stride, name in ("train", "val")))

    crop_rows: list[dict] = []
    tile_rows: list[dict] = []
    acc = np.zeros((3, 3), dtype=np.float64)
    for rows, tile_row, stats in _map(jobs, cfg.prepare_workers):
        crop_rows.extend(rows)
        if tile_row:
            tile_rows.append(tile_row)
        if stats is not None:
            acc += stats

    _write_csv(root / "crops.csv", CROP_FIELDS, crop_rows)
    _write_csv(root / "tiles.csv", TILE_FIELDS, tile_rows)
    if acc[2, 0] > 0:
        mean = acc[0] / acc[2]
        std = np.sqrt(np.maximum(acc[1] / acc[2] - mean**2, 0.0))
        (root / "stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
    (root / "split.json").write_text(json.dumps({"protocol": cfg.protocol, **split}, indent=2))

    counts = {s: sum(r["split"] == s for r in crop_rows) for s in ("train", "val")}
    counts["test_tiles"] = sum(r["split"] == "test" for r in tile_rows)
    return counts


def _map(jobs: list, workers: int) -> Iterable:
    if workers <= 1:
        yield from map(_process_tile, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(_process_tile, jobs)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
