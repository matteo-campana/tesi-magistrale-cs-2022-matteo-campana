# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch datasets feeding Keras ``fit`` (Keras 3 accepts ``torch.utils.data.DataLoader``).

Images are returned channels-last as float32 in ``[0, 255]``: normalisation happens inside
the model, so inference code never has to remember it.

Training batches come from a :class:`StreamSampler`: an endless stream of ``(index, seed)``
pairs addressed by position. The order of the crops and their augmentation depend only on the
training seed and on how many samples were consumed, so a run paused mid-epoch and resumed
(:mod:`rsseg.train.resume`) sees exactly the batches of an uninterrupted run.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from rsseg.config import DataConfig
from rsseg.data.prepare import read_csv
from rsseg.data.transforms import JointAugment, MultiCropAugment, seeded


def read_crops(root: str | Path, split: str) -> list[dict]:
    rows = [r for r in read_csv(Path(root) / "crops.csv") if r["split"] == split]
    if not rows:
        raise FileNotFoundError(
            f"no '{split}' crops in {root}/crops.csv — run `rsseg prepare` first"
        )
    return rows


def read_tiles(root: str | Path, split: str) -> list[dict]:
    return [r for r in read_csv(Path(root) / "tiles.csv") if r["split"] == split]


def load_image(path: str | Path, ndsm_path: str | Path | None = None) -> np.ndarray:
    """``(H, W, 3)`` uint8 image, or ``(H, W, 4)`` with the nDSM as last channel."""
    with Image.open(path) as im:
        image = np.asarray(im.convert("RGB"))
    if ndsm_path is None:
        return image
    with Image.open(ndsm_path) as im:
        ndsm = np.asarray(im)
    return np.concatenate([image, ndsm[..., None]], axis=-1)


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
    """Pre-computed crops for supervised training/validation.

    With ``ndsm=True`` the nDSM crop (decimetres, ``uint8``) is appended as a 4th channel.
    """

    def __init__(
        self,
        root: str | Path,
        rows: list[dict],
        augment: JointAugment | None = None,
        ndsm: bool = False,
    ):
        self.root = Path(root)
        self.rows = rows
        self.augment = augment
        self.ndsm = ndsm
        if ndsm and not all(r.get("ndsm") for r in rows):
            raise FileNotFoundError(
                f"crops in {root} have no nDSM: run `rsseg prepare` with data.ndsm=true"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        idx, seed = idx if isinstance(idx, tuple) else (idx, None)
        row = self.rows[idx]
        image = load_image(self.root / row["image"], self.root / row["ndsm"] if self.ndsm else None)
        with Image.open(self.root / row["mask"]) as im:
            mask = np.asarray(im)
        if self.augment is not None:
            with _maybe_seeded(seed):
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

    def __getitem__(self, idx: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        idx, seed = idx if isinstance(idx, tuple) else (idx, None)
        with Image.open(self.root / self.rows[idx]["image"]) as im:
            image = im.convert("RGB")
            image.load()
        with _maybe_seeded(seed):
            return self.augment(image)


def _maybe_seeded(seed: int | None):
    return seeded(seed) if seed is not None else contextlib.nullcontext()


class StreamSampler(Sampler):
    """Endless stream of ``(index, sample seed)`` for training, addressed by position.

    The stream is a sequence of *passes*. Pass ``p`` is a permutation of the dataset drawn from
    ``(seed, p)`` — repeated and truncated to ``pass_length`` samples when an epoch must be
    longer than the dataset (low-label runs) — and cut to whole batches. Every sample also gets
    a seed for its augmentation, derived from ``(seed, position)``.

    :attr:`position` counts the samples *consumed* by training (advanced after every batch by
    :class:`rsseg.train.resume.ResumeCheckpoint`, restored on resume): a new iterator starts
    there, so batches prefetched by a dropped iterator, or lost to a pause, are produced again
    identically.

    ``len()`` is effectively infinite, so Keras never restarts or truncates the iterator: the
    length of an epoch is given to ``fit`` as ``steps_per_epoch`` (:attr:`batches_per_pass`
    for a full pass).
    """

    ENDLESS = 2**62

    def __init__(self, n: int, batch_size: int, seed: int = 0, pass_length: int | None = None):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.pass_length = (pass_length or n) // batch_size * batch_size
        if self.pass_length == 0:
            raise ValueError(f"batch_size {batch_size} larger than the {n} training samples")
        self.batches_per_pass = self.pass_length // batch_size
        self.position = 0
        self._cache: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.ENDLESS

    def pass_order(self, p: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != p:
            rng = np.random.default_rng([self.seed, p])
            reps = -(-self.pass_length // self.n)
            order = np.concatenate([rng.permutation(self.n) for _ in range(reps)])
            self._cache = (p, order[: self.pass_length])
        return self._cache[1]

    def sample_seed(self, position: int) -> int:
        return int(np.random.SeedSequence([self.seed, position]).generate_state(1)[0])

    def __iter__(self) -> Iterator[tuple[int, int]]:
        g = self.position
        while True:
            p, offset = divmod(g, self.pass_length)
            order = self.pass_order(p)
            for i in range(offset, self.pass_length):
                yield int(order[i]), self.sample_seed(p * self.pass_length + i)
            g = (p + 1) * self.pass_length


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    stream: StreamSampler | None = None,
) -> DataLoader:
    """Training loader over a :class:`StreamSampler` (whole batches only), or, without
    ``stream``, a loader in dataset order (validation)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=stream,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=stream is not None,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
    )


def training_loader(dataset: Dataset, batch_size: int, num_workers: int, seed: int = 0,
                    pass_length: int | None = None) -> DataLoader:  # fmt: skip
    """Loader of a training stream; the sampler is ``loader.sampler`` (a :class:`StreamSampler`).

    ``pass_length`` draws that many samples per epoch, cycling the dataset in a new random
    order at every pass."""
    stream = StreamSampler(len(dataset), batch_size, seed, pass_length)
    return make_loader(dataset, batch_size, num_workers, stream)


def segmentation_loaders(
    cfg: DataConfig, batch_size: int, seed: int = 0
) -> tuple[DataLoader, DataLoader]:
    all_rows = read_crops(cfg.root, "train")
    train_rows = subsample_per_tile(all_rows, cfg.label_fraction, seed)
    pass_length = len(all_rows) if cfg.full_length_epochs and cfg.label_fraction < 1 else None
    val_rows = read_crops(cfg.root, "val")
    train_ds = CropDataset(
        cfg.root, train_rows, JointAugment(cfg.crop_size) if cfg.augment else None, cfg.ndsm
    )
    val_ds = CropDataset(cfg.root, val_rows, None, cfg.ndsm)
    return (
        training_loader(train_ds, batch_size, cfg.num_workers, seed, pass_length),
        make_loader(val_ds, batch_size, cfg.num_workers),
    )
