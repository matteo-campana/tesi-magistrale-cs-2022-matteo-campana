# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn raw ISPRS tiles (Potsdam, Vaihingen) into training crops, full-tile masks and manifests.

Order of operations (the order matters to avoid leakage):

1. assign every *tile* to a split (:func:`rsseg.data.splits.get_split`);
2. convert colour labels to class indices with a vectorised lookup, rejecting tiles whose
   label is not an ISPRS colour mask (``data.max_offpalette``);
3. ``train``/``val`` tiles are cut into crops (overlap allowed only for ``train``);
4. ``val``/``test`` tiles are also registered whole for sliding-window evaluation, with both the
   full and the eroded (no boundary) masks;
5. per-channel statistics are computed on the training tiles only.

Output layout under ``cfg.root``::

    crops/<split>/<tile>_<y>_<x>.png        RGB crop
    crops/<split>/<tile>_<y>_<x>_mask.png   class-index crop (uint8)
    crops/<split>/<tile>_<y>_<x>_ndsm.png   nDSM crop in decimetres (only with data.ndsm)
    tiles/<tile>_mask.png                   class-index mask of val/test tiles
    tiles/<tile>_mask_eroded.png            same, eroded labels (boundaries ignored)
    tiles/<tile>_ndsm.png                   nDSM of val/test tiles (only with data.ndsm)
    crops.csv   tiles.csv   stats.json   split.json
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rsseg import control, progress
from rsseg.config import DataConfig
from rsseg.console import track
from rsseg.data.classes import IGNORE_INDEX, NUM_CLASSES, rgb_to_index
from rsseg.data.ndsm import compute_ndsm, load_dsm, load_official_ndsm
from rsseg.data.sources import DatasetSource, TileFiles, get_source
from rsseg.data.splits import get_split

Image.MAX_IMAGE_PIXELS = None  # Potsdam tiles are 6000x6000
log = logging.getLogger(__name__)

CROP_FIELDS = ["split", "tile", "y", "x", "image", "mask", "ndsm"] + [
    f"n{c}" for c in range(NUM_CLASSES)
]
TILE_FIELDS = ["split", "tile", "image", "mask", "mask_eroded", "ndsm"]
SIZE_TOLERANCE = 2
"""Rasters smaller/larger than the image by at most this many pixels are padded/cropped
(the Potsdam DSM and nDSM of tile 3_13 are 5999 pixels wide)."""

__all__ = ["TileFiles", "crop_positions", "find_tiles", "prepare_dataset", "read_csv"]


def find_tiles(cfg: DataConfig) -> dict[str, TileFiles]:
    """Files of every tile of ``cfg.dataset`` under ``cfg.raw_dir``."""
    if not cfg.raw_dir:
        raise ValueError("data.raw_dir is required for `rsseg prepare`")
    return get_source(cfg.dataset).find(
        cfg.raw_dir, cfg.ndsm_variant, cfg.image_pattern, cfg.label_pattern
    )


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


def fit_to(a: np.ndarray, shape: tuple[int, int], what: str) -> np.ndarray:
    """Pad (edge) or crop ``a`` to ``shape`` if it differs by at most :data:`SIZE_TOLERANCE`."""
    dh, dw = shape[0] - a.shape[0], shape[1] - a.shape[1]
    if (dh, dw) == (0, 0):
        return a
    if max(abs(dh), abs(dw)) > SIZE_TOLERANCE:
        raise ValueError(f"{what}: size {a.shape[:2]} differs from the image {shape}")
    log.warning("%s: size %s padded/cropped to %s", what, a.shape[:2], shape)
    pad = [(0, max(dh, 0)), (0, max(dw, 0))] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad, mode="edge")[: shape[0], : shape[1]]


@dataclass(frozen=True)
class TileJob:
    files: TileFiles
    split: str
    root: Path
    size: int
    stride: int
    make_crops: bool
    train_labels: str
    max_offpalette: float
    ndsm: str | None
    """``official``, ``computed`` or ``None`` (no nDSM)."""
    gsd: float


def label_to_index(path: Path, tile: str, max_offpalette: float | None) -> np.ndarray:
    mask = rgb_to_index(load_rgb(path))
    off = float((mask == IGNORE_INDEX).mean())
    if max_offpalette is not None and off > max_offpalette:
        raise ValueError(
            f"tile {tile}: {off:.2%} of the label pixels are not ISPRS colours ({path}). "
            "The file is probably broken: use another copy of the label or raise "
            "data.max_offpalette."
        )
    return mask


def load_ndsm(files: TileFiles, kind: str, gsd: float) -> np.ndarray:
    if kind == "official":
        if files.ndsm is None:
            raise FileNotFoundError(f"tile {files.tile}: no official nDSM found")
        return load_official_ndsm(files.ndsm)
    if files.dsm is None:
        raise FileNotFoundError(f"tile {files.tile}: no DSM found to compute the nDSM")
    return compute_ndsm(load_dsm(files.dsm), gsd)


def _process_tile(job: TileJob) -> tuple[list[dict], dict | None, np.ndarray | None]:
    files, split, root, size = job.files, job.split, job.root, job.size
    image = load_rgb(files.image)
    shape = image.shape[:2]
    mask = fit_to(label_to_index(files.label, files.tile, job.max_offpalette), shape,
                  f"tile {files.tile} label")  # fmt: skip
    eroded = None
    if files.label_eroded is not None:
        # eroded labels have black (ignored) boundaries: no off-palette check
        eroded = fit_to(label_to_index(files.label_eroded, files.tile, None), shape,
                        f"tile {files.tile} eroded label")  # fmt: skip
    elif job.train_labels == "eroded":
        raise FileNotFoundError(f"tile {files.tile}: data.train_labels=eroded but no eroded label")
    ndsm = None
    if job.ndsm:
        ndsm = fit_to(load_ndsm(files, job.ndsm, job.gsd), shape, f"tile {files.tile} nDSM")
    crop_mask = eroded if job.train_labels == "eroded" else mask

    rows: list[dict] = []
    if job.make_crops:
        out_dir = root / "crops" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = shape
        for y in crop_positions(h, size, job.stride):
            for x in crop_positions(w, size, job.stride):
                m = crop_mask[y : y + size, x : x + size]
                if (m == IGNORE_INDEX).all():
                    continue
                stem = f"{files.tile}_{y}_{x}"
                img_path = out_dir / f"{stem}.png"
                mask_path = out_dir / f"{stem}_mask.png"
                Image.fromarray(image[y : y + size, x : x + size]).save(img_path)
                Image.fromarray(m).save(mask_path)
                ndsm_rel = ""
                if ndsm is not None:
                    ndsm_path = out_dir / f"{stem}_ndsm.png"
                    Image.fromarray(ndsm[y : y + size, x : x + size]).save(ndsm_path)
                    ndsm_rel = ndsm_path.relative_to(root).as_posix()
                hist = np.bincount(m[m != IGNORE_INDEX].ravel(), minlength=NUM_CLASSES)
                rows.append(
                    {"split": split, "tile": files.tile, "y": y, "x": x,
                     "image": img_path.relative_to(root).as_posix(),
                     "mask": mask_path.relative_to(root).as_posix(), "ndsm": ndsm_rel,
                     **{f"n{c}": int(hist[c]) for c in range(NUM_CLASSES)}}
                )  # fmt: skip

    tile_row = None
    if split in ("val", "test"):
        tiles_dir = root / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        def save(arr: np.ndarray | None, suffix: str) -> str:
            if arr is None:
                return ""
            path = tiles_dir / f"{files.tile}_{suffix}.png"
            Image.fromarray(arr).save(path)
            return path.relative_to(root).as_posix()

        tile_row = {"split": split, "tile": files.tile, "image": str(files.image.resolve()),
                    "mask": save(mask, "mask"), "mask_eroded": save(eroded, "mask_eroded"),
                    "ndsm": save(ndsm, "ndsm")}  # fmt: skip

    stats = None
    if split == "train":
        pix = image.reshape(-1, 3).astype(np.float64)
        if ndsm is not None:
            pix = np.concatenate([pix, ndsm.reshape(-1, 1).astype(np.float64)], axis=1)
        stats = np.stack([pix.sum(0), (pix**2).sum(0), np.full(pix.shape[1], len(pix), float)])
    return rows, tile_row, stats


def resolve_ndsm_source(cfg: DataConfig, source: DatasetSource) -> str | None:
    if not cfg.ndsm:
        return None
    kind = cfg.ndsm_source
    if kind == "auto":
        kind = "official" if source.ndsm else "computed"
    if kind not in ("official", "computed"):
        raise ValueError(f"data.ndsm_source must be auto, official or computed, got {kind!r}")
    if kind == "official" and not source.ndsm:
        raise ValueError(f"{source.name} has no official nDSM: use data.ndsm_source=computed")
    return kind


def _plan(cfg: DataConfig) -> tuple[dict[str, TileFiles], dict[str, list[str]], list[TileJob]]:
    if cfg.train_labels not in ("full", "eroded"):
        raise ValueError(f"data.train_labels must be full or eroded, got {cfg.train_labels!r}")
    source = get_source(cfg.dataset)
    tiles = find_tiles(cfg)
    if not tiles:
        raise FileNotFoundError(f"no {cfg.dataset} tiles found in {cfg.raw_dir}")
    split = get_split(cfg.protocol, cfg.val_tiles, cfg.exclude_7_10, cfg.dataset)
    ndsm = resolve_ndsm_source(cfg, source)
    root = Path(cfg.root)

    jobs: list[TileJob] = []
    # A tile may appear in both val and test (thesis2022): crops are produced once for val.
    seen: set[tuple[str, str]] = set()
    for name in ("train", "val", "test"):
        missing = [t for t in split[name] if t not in tiles]
        if missing:
            log.warning("%s tiles not found in raw_dir: %s", name, missing)
        for t in split[name]:
            if t in tiles and (name, t) not in seen:
                seen.add((name, t))
                jobs.append(TileJob(
                    tiles[t], name, root, cfg.crop_size,
                    cfg.train_stride if name == "train" else cfg.crop_size,
                    name in ("train", "val"), cfg.train_labels, cfg.max_offpalette, ndsm,
                    source.gsd,
                ))  # fmt: skip
    return tiles, split, jobs


def prepare_dataset(cfg: DataConfig) -> dict[str, int]:
    """Run the full preparation; returns the number of crops per split."""
    _, split, jobs = _plan(cfg)
    ndsm = jobs[0].ndsm if jobs else None
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)

    crop_rows: list[dict] = []
    tile_rows: list[dict] = []
    acc = None
    tiles_done = progress.tracked(_map(_process_tile, jobs, cfg.prepare_workers), len(jobs),
                                  phase="prepare", stage="tiles")  # fmt: skip
    for rows, tile_row, stats in track(tiles_done, "preparing tiles", total=len(jobs)):
        control.check(pausable=False)
        crop_rows.extend(rows)
        if tile_row:
            tile_rows.append(tile_row)
        if stats is not None:
            acc = stats if acc is None else acc + stats

    _write_csv(root / "crops.csv", CROP_FIELDS, crop_rows)
    _write_csv(root / "tiles.csv", TILE_FIELDS, tile_rows)
    if acc is not None and acc[2, 0] > 0:
        mean = acc[0] / acc[2]
        std = np.sqrt(np.maximum(acc[1] / acc[2] - mean**2, 0.0))
        stats_json = {"mean": mean[:3].tolist(), "std": std[:3].tolist()}
        if len(mean) > 3:
            stats_json |= {"ndsm_mean": float(mean[3]), "ndsm_std": float(std[3])}
        (root / "stats.json").write_text(json.dumps(stats_json, indent=2), encoding="utf-8")
    (root / "split.json").write_text(
        json.dumps({"dataset": cfg.dataset, "protocol": cfg.protocol,
                    "train_labels": cfg.train_labels, "crop_size": cfg.crop_size,
                    "train_stride": cfg.train_stride, "ndsm": ndsm,
                    "ndsm_variant": cfg.ndsm_variant if ndsm == "official" else None,
                    **split}, indent=2),
        encoding="utf-8",
    )  # fmt: skip

    counts = {s: sum(r["split"] == s for r in crop_rows) for s in ("train", "val")}
    counts["test_tiles"] = sum(r["split"] == "test" for r in tile_rows)
    return counts


def _inspect_tile(job: TileJob) -> dict:
    f = job.files

    def size(p: Path | None) -> tuple[int, int] | None:
        if p is None:
            return None
        with Image.open(p) as im:
            return im.size[::-1]

    def where(p: Path | None) -> str:
        return p.parent.name if p is not None else "—"

    shape = size(f.image)
    notes = []
    for role, p in (("label", f.label), ("eroded", f.label_eroded), ("ndsm", f.ndsm),
                    ("dsm", f.dsm)):  # fmt: skip
        s = size(p)
        if s is not None and s != shape:
            ok = max(abs(s[0] - shape[0]), abs(s[1] - shape[1])) <= SIZE_TOLERANCE
            notes.append(f"{role} {s[1]}×{s[0]} ({'pad' if ok else 'ERROR'})")
    needed = {"official": ("nDSM", f.ndsm), "computed": ("DSM", f.dsm)}.get(job.ndsm)
    if needed is not None and needed[1] is None:
        notes.append(f"{needed[0]} missing (ERROR)")
    mask = rgb_to_index(load_rgb(f.label))
    off = float((mask == IGNORE_INDEX).mean())
    if off > job.max_offpalette:
        notes.append(f"label off-palette {off:.2%} (ERROR)")
    ndsm = {"official": where(f.ndsm), "computed": f"computed from {where(f.dsm)}"}
    return {"tile": f.tile, "split": job.split, "size": f"{shape[1]}×{shape[0]}",
            "label": where(f.label), "eroded": where(f.label_eroded),
            "ndsm": ndsm.get(job.ndsm, "—"), "off_palette": off,
            "notes": "; ".join(notes)}  # fmt: skip


def inspect_dataset(cfg: DataConfig) -> dict:
    """Dry run of :func:`prepare_dataset`: files chosen for every tile, split and anomalies."""
    tiles, split, jobs = _plan(cfg)
    inspected = progress.tracked(_map(_inspect_tile, jobs, cfg.prepare_workers), len(jobs),
                                 phase="prepare", stage="inspection")  # fmt: skip
    rows = list(track(inspected, "inspecting tiles", total=len(jobs)))
    in_split = {t for s in split.values() for t in s}
    return {"dataset": cfg.dataset, "raw_dir": cfg.raw_dir, "tiles_found": len(tiles),
            "unused_tiles": sorted(set(tiles) - in_split), "tiles": rows,
            "errors": sum("ERROR" in r["notes"] for r in rows)}  # fmt: skip


def _map(fn, jobs: list, workers: int) -> Iterable:
    if workers <= 1:
        yield from map(fn, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(fn, jobs)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
