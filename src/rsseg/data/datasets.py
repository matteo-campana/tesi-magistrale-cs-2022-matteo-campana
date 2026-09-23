# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch datasets feeding Keras ``fit`` (Keras 3 accepts ``torch.utils.data.DataLoader``).

Images are returned channels-last as float32 in ``[0, 255]``: normalisation happens inside
the model, so inference code never has to remember it.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rsseg.config import DataConfig
from rsseg.data.prepare import read_csv
from rsseg.data.transforms import JointAugment, MultiCropAugment


def read_crops(root: str | Path, split: str) -> list[dict]:
    rows = [r for r in read_csv(Path(root) / "crops.csv") if r["split"] == split]
    if not rows:
        raise FileNotFoundError(
            f"no '{split}' crops in {root}/crops.csv — run `rsseg prepare` first"
        )
    return rows


def read_tiles(root: str | Path, split: str) -> list[dict]:
    return [r for r in read_csv(Path(root) / "tiles.csv") if r["split"] == split]


def subsample_per_tile(rows: list[dict], fraction: float, seed: int = 0) -> list[dict]:
    """Keep ``fraction`` of the crops of each tile (at least one), deterministically."""
    if not 0 < fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1]")
    if fraction == 1:
        return rows
    rng = random.Random(seed)
    by_tile: dict[str, list[dict]] = {}
    for r in rows:
        by_tile.setdefault(r["tile"], []).append(r)
    kept: list[dict] = []
    for tile in sorted(by_tile):
        items = by_tile[tile]
        kept.extend(rng.sample(items, max(1, round(len(items) * fraction))))
    return kept


class CropDataset(Dataset):
    """Pre-computed crops for supervised training/validation."""

    def __init__(self, root: str | Path, rows: list[dict], augment: JointAugment | None = None):
        self.root = Path(root)
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[idx]
        with Image.open(self.root / row["image"]) as im:
            image = np.asarray(im.convert("RGB"))
        with Image.open(self.root / row["mask"]) as im:
            mask = np.asarray(im)
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(
            mask, dtype=np.int64
        )


class MultiCropDataset(Dataset):
    """Unlabelled crops for DINO: returns ``(global_views, local_views)`` stacked per sample."""

    def __init__(self, root: str | Path, rows: list[dict], augment: MultiCropAugment):
        self.root = Path(root)
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        with Image.open(self.root / self.rows[idx]["image"]) as im:
            image = im.convert("RGB")
            image.load()
        return self.augment(image)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
    )


def segmentation_loaders(
    cfg: DataConfig, batch_size: int, seed: int = 0
) -> tuple[DataLoader, DataLoader]:
    train_rows = subsample_per_tile(read_crops(cfg.root, "train"), cfg.label_fraction, seed)
    val_rows = read_crops(cfg.root, "val")
    train_ds = CropDataset(
        cfg.root, train_rows, JointAugment(cfg.crop_size) if cfg.augment else None
    )
    val_ds = CropDataset(cfg.root, val_rows, None)
    return (
        make_loader(
            train_ds, batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True
        ),
        make_loader(val_ds, batch_size, shuffle=False, num_workers=cfg.num_workers),
    )
