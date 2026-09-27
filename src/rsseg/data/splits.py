# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tile-level train/val/test splits of ISPRS Potsdam and Vaihingen.

Splits are always defined on whole tiles, so no crop can leak between splits.
"""

from __future__ import annotations

# Split commonly used in the literature (e.g. MMSegmentation): 24 labelled training tiles
# and the 14 tiles of the original contest test set.
STANDARD_TRAIN: tuple[str, ...] = (
    "2_10", "2_11", "2_12", "3_10", "3_11", "3_12", "4_10", "4_11", "4_12",
    "5_10", "5_11", "5_12", "6_7", "6_8", "6_9", "6_10", "6_11", "6_12",
    "7_7", "7_8", "7_9", "7_10", "7_11", "7_12",
)  # fmt: skip
STANDARD_TEST: tuple[str, ...] = (
    "2_13", "2_14", "3_13", "3_14", "4_13", "4_14", "4_15",
    "5_13", "5_14", "5_15", "6_13", "6_14", "6_15", "7_13",
)  # fmt: skip
# Training tiles held out entirely for validation (model selection).
STANDARD_VAL_DEFAULT: tuple[str, ...] = ("4_12", "7_11")

# Thesis (2022, §4.1.2): these ten areas were removed as "too homogeneous".
THESIS2022_REMOVED: tuple[str, ...] = (
    "2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "6_7", "6_13", "7_13", "7_9",
)  # fmt: skip
# The thesis does not list its six validation areas; this choice is fixed here and documented.
THESIS2022_VAL_DEFAULT: tuple[str, ...] = ("3_14", "4_15", "5_13", "6_15", "7_11", "4_11")

ALL_TILES: tuple[str, ...] = STANDARD_TRAIN + STANDARD_TEST

# Vaihingen: the 16 areas released with labels to the contest participants are the training set,
# the other 17 the test set (same split as MMSegmentation).
VAIHINGEN_TRAIN: tuple[str, ...] = (
    "1", "3", "5", "7", "11", "13", "15", "17", "21", "23", "26", "28", "30", "32", "34", "37",
)  # fmt: skip
VAIHINGEN_TEST: tuple[str, ...] = (
    "2", "4", "6", "8", "10", "12", "14", "16", "20", "22", "24", "27", "29", "31", "33", "35", "38",
)  # fmt: skip
# Area 26 holds most of the (rare) training clutter, so it is not used for validation.
VAIHINGEN_VAL_DEFAULT: tuple[str, ...] = ("11", "30")
VAIHINGEN_ALL: tuple[str, ...] = VAIHINGEN_TRAIN + VAIHINGEN_TEST

PROTOCOLS = ("standard", "thesis2022")
DATASETS = ("potsdam", "vaihingen")


def _holdout(train: tuple[str, ...], val_tiles, default: tuple[str, ...]) -> tuple[list, list]:
    val = list(val_tiles or default)
    unknown = set(val) - set(train)
    if unknown:
        raise ValueError(f"validation tiles must come from the training tiles: {sorted(unknown)}")
    return [t for t in train if t not in val], val


def get_split(
    protocol: str = "standard",
    val_tiles: list[str] | tuple[str, ...] | None = None,
    exclude_7_10: bool = False,
    dataset: str = "potsdam",
) -> dict[str, list[str]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` tile ids.

    ``thesis2022`` (Potsdam only) has no separate test set (the thesis does not describe one):
    its ``test`` split equals ``val``, evaluated on whole tiles.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")

    if dataset == "vaihingen":
        if protocol != "standard":
            raise ValueError("Vaihingen supports only the 'standard' protocol")
        train, val = _holdout(VAIHINGEN_TRAIN, val_tiles, VAIHINGEN_VAL_DEFAULT)
        return {"train": train, "val": val, "test": list(VAIHINGEN_TEST)}

    if protocol == "standard":
        train, val = _holdout(STANDARD_TRAIN, val_tiles, STANDARD_VAL_DEFAULT)
        test = list(STANDARD_TEST)
    else:
        remaining = [t for t in ALL_TILES if t not in THESIS2022_REMOVED]
        val = list(val_tiles or THESIS2022_VAL_DEFAULT)
        unknown = set(val) - set(remaining)
        if unknown:
            raise ValueError(f"unknown or removed validation tiles: {sorted(unknown)}")
        train = [t for t in remaining if t not in val]
        test = list(val)

    if exclude_7_10:
        train = [t for t in train if t != "7_10"]
        val = [t for t in val if t != "7_10"]
    return {"train": train, "val": val, "test": test}
