# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time estimates of campaign steps: priors measured on an RTX 4060, refined by measurements.

A training step costs ``startup + epochs x epoch + evaluation``, where

* ``epoch`` = (training megapixels + ``VAL_COST`` x validation megapixels) x seconds per
  megapixel of the encoder;
* ``epochs`` = ``train.epochs``, or with early stopping the mean number of epochs of the runs
  already finished (``PRIOR_EPOCHS`` before the first one), at most ``train.epochs``;
* ``evaluation`` = test megapixels x seconds per megapixel of sliding-window inference.

A DINO pre-training costs ``epochs x images per epoch x seconds per image``. The priors below
(``docs/ESECUZIONE_E_TEMPI.md``) are replaced by what the steps measure: :meth:`Estimator.measure`
turns the last ``progress.json`` of a finished step into the ``timing`` stored in its
``outcome.json``, and :meth:`Estimator.observe` uses the running step to update the estimate of
the steps still to come.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rsseg.config import ExperimentConfig

STARTUP_S = 100.0
"""Imports, DataLoader workers, model and MLflow before the first batch (Windows)."""
BENCHMARK_S = 40.0
VAL_COST = 0.5
"""A validation image costs about half a training image (no backward pass)."""
PRIOR_EPOCHS = 60.0
"""Epochs of a run with early stopping (patience 20, lr ×0.4 every 15 epochs) before a
measurement exists."""
PRIOR_EVAL_S_PER_MPX = 0.85
"""Sliding-window evaluation (window 512, stride 256): ~0.5 min per 6000² tile."""
MPX_512 = 512 * 512 / 1e6
PRIOR_TRAIN_S_PER_MPX = {arch: s / MPX_512 for arch, s in {
    "resnet18": 0.044, "resnet34": 0.050, "resnet50": 0.080, "swin_t": 0.073,
    "vit_s_16": 0.067, "vit_b_16": 0.46,
}.items()}  # fmt: skip
"""Seconds per training megapixel (from seconds per 512² image, bf16)."""
PRIOR_DINO_S_PER_IMAGE = {"resnet18": 0.0135, "resnet34": 0.022, "vit_s_16": 0.0146}


@dataclass
class TrainCost:
    startup: float
    epochs: float
    epoch_s: float
    eval_s: float

    @property
    def total(self) -> float:
        return self.startup + self.epochs * self.epoch_s + self.eval_s


@dataclass
class DataStats:
    n_train: int
    n_val: int
    crop_mpx: float
    test_mpx: list[float]
    """Megapixels of every test tile, in ``tiles.csv`` order."""


def _counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


class Estimator:
    """Expected duration of campaign steps (seconds), calibrated on the steps already run."""

    def __init__(self, store: Path | None = None):
        self._data: dict[str, DataStats | None] = {}
        self._pools: dict[str, int] = {}
        self.train_s_per_mpx: dict[str, list[float]] = {}
        self.dino_s_per_image: dict[str, list[float]] = {}
        self.epochs: list[float] = []
        self.eval_s_per_mpx: list[float] = []
        self.benchmark_s: list[float] = []
        self.live: dict[str, float] = {}
        """Rates measured by the running step (not yet finished), by ``kind:arch``."""
        if store is not None:
            for f in sorted(store.glob("*/*/outcome.json")):
                o = json.loads(f.read_text(encoding="utf-8"))
                if o.get("state") == "done":
                    self.add(f.parent.parent.name, o.get("timing") or {}, o.get("duration_s"))

    # ------------------------------------------------------------------ measurements

    def add(self, folder: str, timing: dict, duration_s: float | None = None) -> None:
        """Record the ``timing`` of a finished step (``folder``: runs, backbones, benchmarks)."""
        arch = timing.get("arch")
        if folder == "benchmarks" and duration_s:
            self.benchmark_s.append(duration_s)
        if not arch:
            return
        if timing.get("train_s_per_mpx"):
            self.train_s_per_mpx.setdefault(arch, []).append(timing["train_s_per_mpx"])
        if timing.get("dino_s_per_image"):
            self.dino_s_per_image.setdefault(arch, []).append(timing["dino_s_per_image"])
        if timing.get("early_stopping") and timing.get("epochs"):
            self.epochs.append(timing["epochs"])
        if timing.get("eval_s_per_mpx"):
            self.eval_s_per_mpx.append(timing["eval_s_per_mpx"])

    def measure(self, kind: str, cfg: ExperimentConfig | None, p: dict | None) -> dict:
        """The ``timing`` of a finished step, from its last progress."""
        if cfg is None or p is None:
            return {}
        arch = cfg.ssl.arch if kind == "pretrain" else cfg.model.arch
        timing: dict = {"arch": arch}
        if kind == "train" and p.get("epoch_s"):
            timing["train_s_per_mpx"] = p["epoch_s"] / self._epoch_mpx(cfg)
            timing["epochs"] = p.get("epochs_done") or p.get("epoch")
            timing["early_stopping"] = bool(cfg.train.early_stopping_patience)
        if kind == "train" and p.get("phase") == "evaluation" and p.get("tile_s"):
            stats = self.data(cfg)
            if stats and stats.test_mpx:
                timing["eval_s_per_mpx"] = p["tile_s"] / (sum(stats.test_mpx) / len(stats.test_mpx))
        if kind == "pretrain" and p.get("epoch_s"):
            timing["dino_s_per_image"] = p["epoch_s"] / self._dino_images(cfg)
        return timing

    def observe(self, kind: str, cfg: ExperimentConfig | None, p: dict | None) -> None:
        """Use the progress of the running step for the steps that follow it."""
        timing = self.measure(kind, cfg, p)
        for key in ("train_s_per_mpx", "dino_s_per_image"):
            if key in timing:
                self.live[f"{key}:{timing['arch']}"] = timing[key]

    # ---------------------------------------------------------------------- estimates

    def data(self, cfg: ExperimentConfig) -> DataStats | None:
        root = Path(cfg.data.root)
        key = root.as_posix()
        if key not in self._data:
            self._data[key] = None
            try:
                counts = _counts(root / "crops.csv")
                split = json.loads((root / "split.json").read_text(encoding="utf-8"))
                crop = split.get("crop_size", cfg.data.crop_size)
                test = []
                from PIL import Image

                with (root / "tiles.csv").open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["split"] == (cfg.eval.split or "test"):
                            with Image.open(row["image"]) as im:
                                test.append(im.size[0] * im.size[1] / 1e6)
                self._data[key] = DataStats(counts.get("train", 0), counts.get("val", 0),
                                            crop * crop / 1e6, test)  # fmt: skip
            except (OSError, ValueError, KeyError):
                pass  # not prepared yet: no estimate
        return self._data[key]

    def _epoch_mpx(self, cfg: ExperimentConfig) -> float:
        stats = self.data(cfg)
        if stats is None:
            return float("nan")
        t, bs = cfg.train, cfg.train.batch_size
        labelled = stats.n_train
        if cfg.data.label_fraction < 1 and not cfg.data.full_length_epochs:
            labelled = round(stats.n_train * cfg.data.label_fraction)
        train = t.steps_per_epoch * bs if t.steps_per_epoch else labelled // bs * bs
        val = t.validation_steps * bs if t.validation_steps else stats.n_val
        return (train + VAL_COST * min(val, stats.n_val)) * stats.crop_mpx

    def _pool_crops(self, path: str) -> int:
        if path not in self._pools:
            try:
                self._pools[path] = _counts(Path(path) / "crops.csv").get("unlabeled", 0)
            except OSError:
                self._pools[path] = 0
        return self._pools[path]

    def _dino_images(self, cfg: ExperimentConfig) -> float:
        s = cfg.ssl
        if s.steps_per_epoch:
            return s.steps_per_epoch * s.batch_size
        stats = self.data(cfg)
        n = (stats.n_train if stats else 0) + sum(self._pool_crops(p) for p in s.extra_pools)
        return n // s.batch_size * s.batch_size or float("nan")

    def _rate(self, table: dict[str, list[float]], key: str, arch: str, prior: dict,
              default: float) -> float:  # fmt: skip
        if f"{key}:{arch}" in self.live:
            return self.live[f"{key}:{arch}"]
        measured = table.get(arch)
        return sum(measured) / len(measured) if measured else prior.get(arch, default)

    def expected_epochs(self, cfg: ExperimentConfig) -> float:
        t = cfg.train
        if t.steps_per_epoch and t.epochs <= 2:  # quick runs
            return float(t.epochs)
        if not t.early_stopping_patience:
            return float(t.epochs)
        mean = sum(self.epochs) / len(self.epochs) if self.epochs else PRIOR_EPOCHS
        return min(float(t.epochs), max(mean, t.early_stopping_patience + 1.0))

    def eval_seconds(self, cfg: ExperimentConfig) -> float:
        stats = self.data(cfg)
        if not cfg.eval.split:
            return 0.0
        if stats is None:
            return float("nan")
        tiles = stats.test_mpx[: cfg.eval.max_tiles] if cfg.eval.max_tiles else stats.test_mpx
        rate = (sum(self.eval_s_per_mpx) / len(self.eval_s_per_mpx) if self.eval_s_per_mpx
                else PRIOR_EVAL_S_PER_MPX)  # fmt: skip
        return sum(tiles) * rate

    def train_cost(self, cfg: ExperimentConfig) -> TrainCost:
        rate = self._rate(self.train_s_per_mpx, "train_s_per_mpx", cfg.model.arch,
                          PRIOR_TRAIN_S_PER_MPX, PRIOR_TRAIN_S_PER_MPX["resnet34"])  # fmt: skip
        return TrainCost(STARTUP_S, self.expected_epochs(cfg), self._epoch_mpx(cfg) * rate,
                         self.eval_seconds(cfg))  # fmt: skip

    def dino_epoch_s(self, cfg: ExperimentConfig) -> float:
        rate = self._rate(self.dino_s_per_image, "dino_s_per_image", cfg.ssl.arch,
                          PRIOR_DINO_S_PER_IMAGE, 0.02)  # fmt: skip
        return self._dino_images(cfg) * rate

    def seconds(self, kind: str, cfg: ExperimentConfig | None) -> float:
        """Expected duration of a step; ``nan`` when it cannot be estimated yet."""
        if kind == "benchmark":
            return (
                sum(self.benchmark_s) / len(self.benchmark_s) if self.benchmark_s else BENCHMARK_S
            )
        if cfg is None:
            return float("nan")
        if kind == "pretrain":
            return STARTUP_S + cfg.ssl.epochs * self.dino_epoch_s(cfg)
        return self.train_cost(cfg).total
