# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unlabelled crops from the raw ISPRS aerial frames, as an extra pool for DINO pre-training.

Sources (``data/download/<city>/Images``):

* ``toronto``: 13 UltraCam-D frames 11500×7500, RGB 8 bit, ~15 cm;
* ``vaihingen_frames``: 20 Intergraph DMC frames 7680×13824, colour-infrared (IR-R-G, the
  colours of the Vaihingen TOP), **16 bit per channel**, ~8 cm. Pillow reads 16-bit RGB by
  keeping the high byte (values ≈ 0–15), so frames are read with ``tifffile`` and mapped to
  8 bit with a per-channel percentile stretch computed on an overview of the frame.

Frames are perspective images, not orthophotos: they are fine for self-supervised pre-training,
not for evaluation. Every pool records its city in ``source.json``: the frames of Vaihingen also
cover the Vaihingen *test* areas, so :func:`rsseg.ssl.train.pretrain_dino` refuses to mix a pool
with the city of the benchmark unless ``ssl.allow_same_city_pool`` is set.

Output layout under ``out``::

    crops/<frame>_<y>_<x>.png   previews/<frame>.png   crops.csv (split = unlabeled)   source.json
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rsseg import control, progress
from rsseg.console import track
from rsseg.data.prepare import _write_csv, crop_positions

log = logging.getLogger(__name__)

UNLABELED_FIELDS = ["split", "tile", "y", "x", "image"]


@dataclass(frozen=True)
class UnlabeledSource:
    name: str
    city: str
    gsd: float
    pattern: str
    """Regex on the path relative to the raw directory; group 1 is the frame id."""


UNLABELED_SOURCES: dict[str, UnlabeledSource] = {
    s.name: s
    for s in (
        UnlabeledSource("toronto", "toronto", 0.15, r"(?:^|/)(\d{5})\.tif$"),
        UnlabeledSource("vaihingen_frames", "vaihingen", 0.08, r"(?:^|/)(\d{8})\.tif$"),
    )
}


def read_frame(path: str | Path, overview_level: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution frame and one of its pyramid levels (``(H, W, 3)``, native dtype)."""
    import tifffile

    with tifffile.TiffFile(path) as tf:
        full = tf.pages[0].asarray()
        if len(tf.pages) > 1:
            overview = tf.pages[min(overview_level, len(tf.pages) - 1)].asarray()
        else:
            step = 2**overview_level
            overview = full[::step, ::step]
    return full[..., :3], overview[..., :3]


def stretch_lut(overview: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray | None:
    """Per-channel look-up tables ``(3, 65536)`` mapping ``[p_low, p_high]`` to ``[0, 255]``
    (``None`` for 8-bit data, which is kept as is)."""
    if overview.dtype == np.uint8:
        return None
    if overview.dtype != np.uint16:
        raise ValueError(f"unsupported frame dtype {overview.dtype}")
    values = np.arange(65536, dtype=np.float32)
    luts = []
    for c in range(3):
        lo, hi = np.percentile(overview[..., c], [low, high])
        luts.append(np.clip((values - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).round())
    return np.stack(luts).astype(np.uint8)


def to_uint8(a: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    if lut is None:
        return a
    return np.stack([lut[c][a[..., c]] for c in range(3)], axis=-1)


def prepare_unlabeled(
    source: str,
    raw_dir: str | Path,
    out: str | Path,
    crop_size: int = 512,
    max_per_frame: int = 0,
    min_std: float = 5.0,
    seed: int = 0,
) -> dict:
    """Cut non-overlapping crops from every frame; ``max_per_frame > 0`` samples at most that
    many crops per frame. Nearly uniform crops (std below ``min_std``) are discarded."""
    if source not in UNLABELED_SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {list(UNLABELED_SOURCES)}")
    src = UNLABELED_SOURCES[source]
    raw, out = Path(raw_dir), Path(out)
    rx = re.compile(src.pattern)
    frames = sorted(
        (m.group(1), p)
        for p in raw.rglob("*.tif")
        if (m := rx.search(p.relative_to(raw).as_posix()))
    )
    if not frames:
        raise FileNotFoundError(f"no {source} frames found in {raw}")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict] = []
    skipped = 0
    for frame, path in track(progress.tracked(frames, phase="prepare", stage="frames"),
                             f"cutting {source} frames", total=len(frames)):  # fmt: skip
        control.check(pausable=False)
        full, overview = read_frame(path)
        lut = stretch_lut(overview)
        Image.fromarray(to_uint8(overview, lut)).save(out / "previews" / f"{frame}.png")
        h, w = full.shape[:2]
        coords = [(y, x) for y in crop_positions(h, crop_size, crop_size)
                  for x in crop_positions(w, crop_size, crop_size)]  # fmt: skip
        if max_per_frame > 0 and len(coords) > max_per_frame:
            coords = sorted(rng.sample(coords, max_per_frame))
        for y, x in coords:
            crop = to_uint8(full[y : y + crop_size, x : x + crop_size], lut)
            if crop.std() < min_std:
                skipped += 1
                continue
            path_out = out / "crops" / f"{frame}_{y}_{x}.png"
            Image.fromarray(crop).save(path_out)
            rows.append({"split": "unlabeled", "tile": frame, "y": y, "x": x,
                         "image": path_out.relative_to(out).as_posix()})  # fmt: skip
    _write_csv(out / "crops.csv", UNLABELED_FIELDS, rows)
    meta = {"source": source, "city": src.city, "gsd": src.gsd, "raw_dir": str(raw),
            "frames": [f for f, _ in frames], "crop_size": crop_size,
            "max_per_frame": max_per_frame, "crops": len(rows), "skipped_uniform": skipped}  # fmt: skip
    (out / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_pool_meta(root: str | Path) -> dict:
    path = Path(root) / "source.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: prepare the pool with `rsseg prepare-unlabeled`"
        )
    return json.loads(path.read_text(encoding="utf-8"))
