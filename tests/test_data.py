# SPDX-License-Identifier: AGPL-3.0-or-later
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from rsseg.data.classes import (
    IGNORE_INDEX,
    NUM_CLASSES,
    PALETTE,
    index_to_rgb,
    legacy_to_index,
    rgb_to_index,
)
from rsseg.data.datasets import (
    CropDataset,
    StreamSampler,
    read_crops,
    read_tiles,
    segmentation_loaders,
    subsample_per_tile,
)
from rsseg.data.prepare import crop_positions, load_mask
from rsseg.data.splits import STANDARD_TEST, STANDARD_TRAIN, THESIS2022_REMOVED, get_split
from rsseg.data.transforms import JointAugment, MultiCropAugment


def test_rgb_index_roundtrip_and_unknown_colours():
    idx = np.random.default_rng(0).integers(0, NUM_CLASSES, (7, 9)).astype(np.uint8)
    rgb = index_to_rgb(idx)
    assert (rgb_to_index(rgb) == idx).all()
    rgb[0, 0] = (12, 34, 56)
    assert rgb_to_index(rgb)[0, 0] == IGNORE_INDEX


def test_palette_order_and_legacy_mapping():
    assert tuple(PALETTE[5]) == (255, 0, 0)  # clutter is last, never index 0
    legacy = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8)
    # legacy: clutter, impervious, car, building, low veg, tree
    assert legacy_to_index(legacy).tolist() == [[5, 0, 4, 1, 2, 3]]


def test_standard_split_is_disjoint_and_complete():
    s = get_split("standard")
    assert len(STANDARD_TRAIN) == 24 and len(STANDARD_TEST) == 14
    assert not set(s["train"]) & set(s["val"])
    assert not (set(s["train"]) | set(s["val"])) & set(s["test"])
    assert sorted(s["train"] + s["val"]) == sorted(STANDARD_TRAIN)


def test_thesis2022_split():
    s = get_split("thesis2022")
    assert len(s["train"]) == 22 and len(s["val"]) == 6
    assert not set(THESIS2022_REMOVED) & set(s["train"] + s["val"])
    assert s["test"] == s["val"]


def test_split_rejects_bad_validation_tiles():
    with pytest.raises(ValueError):
        get_split("standard", val_tiles=["2_13"])  # a test tile


def test_crop_positions_cover_borders():
    assert crop_positions(1000, 512, 512) == [0, 488]
    assert crop_positions(512, 512, 256) == [0]
    assert crop_positions(100, 512, 256) == [0]


def test_prepare_outputs(synthetic_dataset):
    cfg, counts = synthetic_dataset
    train, val = read_crops(cfg.root, "train"), read_crops(cfg.root, "val")
    assert counts["train"] == len(train) > 0 and counts["val"] == len(val) > 0
    assert not {r["tile"] for r in train} & {r["tile"] for r in val}
    tiles = read_tiles(cfg.root, "test")
    assert {t["tile"] for t in tiles} == set(STANDARD_TEST)
    mask = load_mask(Path(cfg.root) / tiles[0]["mask"])
    assert mask.shape == (128, 128) and set(np.unique(mask)) <= set(range(NUM_CLASSES)) | {
        IGNORE_INDEX
    }
    stats = json.loads((Path(cfg.root) / "stats.json").read_text(encoding="utf-8"))
    assert len(stats["mean"]) == 3


def test_crop_dataset_and_augment_keep_alignment(synthetic_dataset):
    cfg, _ = synthetic_dataset
    rows = read_crops(cfg.root, "train")
    ds = CropDataset(cfg.root, rows, JointAugment())
    image, mask = ds[0]
    assert image.shape == (64, 64, 3) and image.dtype == np.float32 and mask.shape == (64, 64)

    # A mask derived from the image must stay consistent after joint augmentation.
    aug = JointAugment()
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3)).astype(np.uint8)
    for _ in range(10):
        a, m = aug(img, img[..., 0].copy())
        assert (a[..., 0] == m).all()


def test_subsample_per_tile_keeps_every_tile():
    rows = [{"tile": t, "i": i} for t in ("a", "b") for i in range(10)]
    kept = subsample_per_tile(rows, 0.1, seed=0)
    assert {r["tile"] for r in kept} == {"a", "b"} and len(kept) == 2
    assert subsample_per_tile(rows, 0.1, seed=0) == kept


def test_low_label_epochs_keep_the_full_length(synthetic_dataset):
    cfg, _ = synthetic_dataset
    n_all = len(read_crops(cfg.root, "train"))
    full, _ = segmentation_loaders(cfg, batch_size=8)
    low = dataclasses.replace(cfg, label_fraction=0.25)
    cycled, _ = segmentation_loaders(low, batch_size=8)
    assert full.sampler.batches_per_pass == cycled.sampler.batches_per_pass == n_all // 8
    one_pass, _ = segmentation_loaders(dataclasses.replace(low, full_length_epochs=False), 8)
    labelled = len(one_pass.dataset)
    assert labelled < n_all and one_pass.sampler.batches_per_pass == labelled // 8
    # the cycled epoch only uses the labelled crops, each about n_all / labelled times
    order = cycled.sampler.pass_order(0)
    assert len(order) == n_all // 8 * 8 and set(order) == set(range(labelled))


def test_stream_sampler_is_addressed_by_position():
    s = StreamSampler(10, batch_size=3, seed=5)
    assert s.pass_length == 9 and s.batches_per_pass == 3
    stream = list(itertools.islice(iter(s), 30))
    assert len({i for i, _ in stream[:9]}) == 9 and stream[:9] != stream[9:18]
    # a new iterator starts at the consumed position: same indices and augmentation seeds
    s.position = 13
    assert list(itertools.islice(iter(s), 17)) == stream[13:]
    # the stream depends only on the seed
    assert list(itertools.islice(iter(StreamSampler(10, 3, seed=5)), 30)) == stream
    assert list(itertools.islice(iter(StreamSampler(10, 3, seed=6)), 30)) != stream


def test_seeded_augmentation_is_reproducible_and_restores_the_rng():
    import random

    from rsseg.data.transforms import seeded

    img = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    aug = JointAugment(8)
    random.seed(1)
    before = random.random()
    random.seed(1)
    with seeded(42):
        a, _ = aug(img, img[..., 0])
    assert random.random() == before  # the global generator is untouched
    with seeded(42):
        b, _ = aug(img, img[..., 0])
    np.testing.assert_array_equal(a, b)


def test_multicrop_shapes():
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (128, 128, 3)).astype(np.uint8))
    for aerial in (False, True):
        g, loc = MultiCropAugment(64, 32, 3, aerial=aerial)(img)
        assert g.shape == (2, 64, 64, 3) and loc.shape == (3, 32, 32, 3)
