# SPDX-License-Identifier: AGPL-3.0-or-later
"""File layout of the ISPRS datasets as distributed: which file is the image, the label, the
eroded label, the nDSM and the DSM of each tile.

Every role is a list of regular expressions matched against the path relative to ``raw_dir``
(``/`` separators), **highest priority first**. The priority matters for Potsdam, where some
files of ``5_Labels_all`` are broken while their copy in ``5_Labels_for_participants`` is not:

* ``top_potsdam_4_12_label.tif`` (and its ``noBoundary`` version) contains no ISPRS colour at all;
* ``top_potsdam_6_7_label.tif`` has 0.7% of the car pixels coloured ``(252, 255, 0)``.

The capture groups of a regex form the tile id; numbers are normalised (``02_10`` → ``2_10``)
because the Potsdam DSM files use zero-padded ids.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileFiles:
    tile: str
    image: Path
    label: Path
    label_eroded: Path | None = None
    ndsm: Path | None = None
    """Official nDSM (8 bit, decimetres above ground)."""
    dsm: Path | None = None
    """Absolute DSM (float, metres), used to compute an nDSM."""


@dataclass(frozen=True)
class DatasetSource:
    name: str
    city: str
    gsd: float
    """Ground sampling distance in metres."""
    image: tuple[str, ...]
    label: tuple[str, ...]
    label_eroded: tuple[str, ...] = ()
    ndsm: tuple[str, ...] = ()
    """May contain ``{variant}`` (see ``data.ndsm_variant``)."""
    dsm: tuple[str, ...] = ()

    def find(
        self,
        raw_dir: str | Path,
        ndsm_variant: str = "lastools",
        image_pattern: str | None = None,
        label_pattern: str | None = None,
    ) -> dict[str, TileFiles]:
        """Match the files of every tile under ``raw_dir`` (searched recursively)."""
        raw = Path(raw_dir)
        if not raw.is_dir():
            raise FileNotFoundError(f"raw_dir {raw} does not exist")
        paths = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file())
        roles = {
            "image": (image_pattern,) if image_pattern else self.image,
            "label": (label_pattern,) if label_pattern else self.label,
            "label_eroded": self.label_eroded,
            "ndsm": tuple(p.format(variant=re.escape(ndsm_variant)) for p in self.ndsm),
            "dsm": self.dsm,
        }
        found = {role: _match(paths, patterns, role) for role, patterns in roles.items()}
        images, labels = found["image"], found["label"]
        missing = sorted(set(images) - set(labels), key=tile_sort_key)
        if missing:
            log.warning("tiles without labels (ignored): %s", missing)
        tiles = {}
        for t in sorted(set(images) & set(labels), key=tile_sort_key):
            opt = {r: raw / found[r][t] for r in ("label_eroded", "ndsm", "dsm") if t in found[r]}
            tiles[t] = TileFiles(t, raw / images[t], raw / labels[t], **opt)
        return tiles


def normalise_tile_id(groups: tuple[str, ...]) -> str:
    return "_".join(str(int(p)) if p.isdigit() else p for g in groups for p in g.split("_"))


def tile_sort_key(tile: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in tile.split("_"))


def _match(paths: list[str], patterns: tuple[str, ...], role: str) -> dict[str, str]:
    """Tile id → path of the highest-priority pattern that matches it."""
    out: dict[str, str] = {}
    for pattern in patterns:
        rx = re.compile(pattern)
        level: dict[str, str] = {}
        for path in paths:
            m = rx.search(path)
            if not m:
                continue
            tile = normalise_tile_id(m.groups())
            if tile in out:  # already resolved by a higher-priority pattern
                continue
            if tile in level:
                log.warning("%s of tile %s: %s and %s both match, using the first", role, tile,
                            level[tile], path)  # fmt: skip
                continue
            level[tile] = path
        for tile, path in level.items():
            out.setdefault(tile, path)
    return out


_P = r"(?:^|/)"
POTSDAM = DatasetSource(
    name="potsdam",
    city="potsdam",
    gsd=0.05,
    image=(rf"{_P}top_potsdam_(\d+_\d+)_RGB\.tif$",),
    label=(
        r"5_Labels_for_participants/(?:.+/)?top_potsdam_(\d+_\d+)_label\.tif$",
        rf"{_P}top_potsdam_(\d+_\d+)_label\.tif$",
    ),
    label_eroded=(
        r"5_Labels_for_participants_no_Boundary/(?:.+/)?top_potsdam_(\d+_\d+)_label_noBoundary\.tif$",
        rf"{_P}top_potsdam_(\d+_\d+)_label_noBoundary\.tif$",
    ),
    ndsm=(rf"{_P}dsm_potsdam_(\d+_\d+)_normalized_{{variant}}\.jpg$",),
    dsm=(rf"{_P}dsm_potsdam_(\d+_\d+)\.tif$",),
)
VAIHINGEN = DatasetSource(
    name="vaihingen",
    city="vaihingen",
    gsd=0.09,
    # The ground truth files have the same names as the images: the folder tells them apart.
    image=(rf"{_P}top/top_mosaic_09cm_area(\d+)\.tif$",),
    label=(
        r"ground_truth_COMPLETE/top_mosaic_09cm_area(\d+)\.tif$",
        r"gts_for_participants/top_mosaic_09cm_area(\d+)\.tif$",
    ),
    label_eroded=(
        r"eroded_COMPLETE/top_mosaic_09cm_area(\d+)_noBoundary\.tif$",
        r"eroded_for_participants/top_mosaic_09cm_area(\d+)_noBoundary\.tif$",
    ),
    dsm=(rf"{_P}dsm/dsm_09cm_matching_area(\d+)\.tif$",),
)
SOURCES: dict[str, DatasetSource] = {s.name: s for s in (POTSDAM, VAIHINGEN)}


def get_source(dataset: str) -> DatasetSource:
    if dataset not in SOURCES:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(SOURCES)}")
    return SOURCES[dataset]
