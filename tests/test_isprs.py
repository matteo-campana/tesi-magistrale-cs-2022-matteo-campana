# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-dataset layouts: label priority and checks, Vaihingen, nDSM, eroded labels, DINO pools."""

import json
import shutil
from pathlib import Path

import keras
import numpy as np
import pytest
from PIL import Image

from rsseg.config import DataConfig, ExperimentConfig, load_config
from rsseg.data.datasets import CropDataset, read_crops, read_tiles
from rsseg.data.ndsm import compute_ndsm
from rsseg.data.prepare import find_tiles, fit_to, inspect_dataset, load_mask, prepare_dataset
from rsseg.data.splits import VAIHINGEN_TEST, VAIHINGEN_TRAIN, get_split
from rsseg.data.synthetic import make_synthetic_potsdam, make_synthetic_vaihingen
from rsseg.eval.report import aggregate_reports, evaluate_tiles, report_markdown

SIZE = 96


def _data_cfg(raw: Path, root: Path, **kw) -> DataConfig:
    return DataConfig(raw_dir=str(raw), root=str(root), crop_size=64, train_stride=64,
                      num_workers=0, prepare_workers=1, **kw)  # fmt: skip


@pytest.fixture(scope="module")
def potsdam_raw(tmp_path_factory) -> Path:
    """Potsdam layout with extras and the two real label defects (4_12 broken, 6_7 off-colour)
    in ``5_Labels_all``; the good copies are in ``5_Labels_for_participants``."""
    raw = make_synthetic_potsdam(tmp_path_factory.mktemp("potsdam_raw"), size=SIZE, extras=True)
    part = raw / "5_Labels_for_participants" / "5_Labels_for_participants"
    part.mkdir(parents=True)
    for tile in ("4_12", "6_7"):
        shutil.copy(raw / "5_Labels_all" / f"top_potsdam_{tile}_label.tif", part)
    rng = np.random.default_rng(0)
    broken = rng.integers(0, 255, (SIZE, SIZE, 3)).astype(np.uint8)
    Image.fromarray(broken).save(raw / "5_Labels_all" / "top_potsdam_4_12_label.tif")
    with Image.open(raw / "5_Labels_all" / "top_potsdam_6_7_label.tif") as im:
        label = np.asarray(im).copy()
    label[: SIZE // 4] = (252, 255, 0)
    Image.fromarray(label).save(raw / "5_Labels_all" / "top_potsdam_6_7_label.tif")
    # the DSM and nDSM of 3_13 are one pixel narrower, as in the real dataset
    for name in ("1_DSM/dsm_potsdam_03_13.tif",
                 "1_DSM_normalisation/dsm_potsdam_03_13_normalized_lastools.jpg"):  # fmt: skip
        with Image.open(raw / name) as im:
            im.crop((0, 0, SIZE - 1, SIZE)).save(raw / name)
    return raw


@pytest.fixture(scope="module")
def potsdam_ndsm(potsdam_raw, tmp_path_factory) -> DataConfig:
    cfg = _data_cfg(potsdam_raw, tmp_path_factory.mktemp("potsdam_ndsm"), ndsm=True)
    prepare_dataset(cfg)
    return cfg


def test_label_priority_prefers_participants_copy(potsdam_raw):
    tiles = find_tiles(_data_cfg(potsdam_raw, potsdam_raw))
    assert len(tiles) == 38
    assert tiles["4_12"].label.parent.name == "5_Labels_for_participants"
    assert tiles["6_7"].label.parent.name == "5_Labels_for_participants"
    assert tiles["2_13"].label.parent.name == "5_Labels_all"
    # zero-padded DSM ids are matched to the tile ids
    assert tiles["2_10"].dsm.name == "dsm_potsdam_02_10.tif"
    assert tiles["2_10"].ndsm.name == "dsm_potsdam_02_10_normalized_lastools.jpg"
    assert tiles["2_10"].label_eroded.name == "top_potsdam_2_10_label_noBoundary.tif"


def test_broken_label_is_rejected(potsdam_raw, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(potsdam_raw, raw, ignore=shutil.ignore_patterns("5_Labels_for_participants*"))
    cfg = _data_cfg(raw, tmp_path / "out")
    with pytest.raises(ValueError, match="not ISPRS colours"):
        prepare_dataset(cfg)
    report = inspect_dataset(cfg)
    bad = {r["tile"] for r in report["tiles"] if "ERROR" in r["notes"]}
    assert bad == {"4_12", "6_7"} and report["errors"] == 2


def test_dry_run_reports_sources_and_padding(potsdam_raw):
    report = inspect_dataset(_data_cfg(potsdam_raw, potsdam_raw, ndsm=True))
    rows = {r["tile"]: r for r in report["tiles"]}
    assert report["errors"] == 0 and report["tiles_found"] == 38
    assert rows["4_12"]["label"] == "5_Labels_for_participants"
    assert "pad" in rows["3_13"]["notes"] and "ERROR" not in rows["3_13"]["notes"]


def test_empty_ndsm_file_counts_as_missing(potsdam_raw, tmp_path):
    """11 official ``ownapproach`` nDSMs are 0-byte files in the ISPRS archive."""
    raw = tmp_path / "raw"
    shutil.copytree(potsdam_raw, raw)
    empty = find_tiles(_data_cfg(raw, raw))["4_10"].ndsm
    empty.write_bytes(b"")
    assert find_tiles(_data_cfg(raw, raw))["4_10"].ndsm is None
    report = inspect_dataset(_data_cfg(raw, raw, ndsm=True))
    bad = {r["tile"]: r["notes"] for r in report["tiles"] if "ERROR" in r["notes"]}
    assert bad == {"4_10": "nDSM missing (ERROR)"} and report["errors"] == 1


def test_fit_to_pads_small_mismatches_only():
    a = np.arange(90, dtype=np.float32).reshape(10, 9)
    out = fit_to(a, (10, 10), "x")
    assert out.shape == (10, 10) and (out[:, 9] == a[:, 8]).all()
    assert fit_to(a, (9, 9), "x").shape == (9, 9)
    with pytest.raises(ValueError, match="differs"):
        fit_to(a, (10, 20), "x")


def test_prepare_with_ndsm_and_eroded_labels(potsdam_ndsm):
    cfg = potsdam_ndsm
    rows = read_crops(cfg.root, "train")
    assert all(r["ndsm"] for r in rows)
    image, mask = CropDataset(cfg.root, rows, ndsm=True)[0]
    assert image.shape == (64, 64, 4) and mask.shape == (64, 64)
    tiles = read_tiles(cfg.root, "test")
    assert all(t["mask_eroded"] and t["ndsm"] for t in tiles)
    t = next(t for t in tiles if t["tile"] == "3_13")  # padded nDSM
    assert load_mask(Path(cfg.root) / t["ndsm"]).shape == (SIZE, SIZE)
    eroded = load_mask(Path(cfg.root) / t["mask_eroded"])
    assert (eroded == 255).any() and (eroded != 255).any()
    stats = json.loads((Path(cfg.root) / "stats.json").read_text(encoding="utf-8"))
    assert len(stats["mean"]) == 3 and stats["ndsm_std"] > 0
    with pytest.raises(FileNotFoundError, match="nDSM"):
        CropDataset(cfg.root, [{**rows[0], "ndsm": ""}], ndsm=True)


def test_vaihingen_split():
    s = get_split(dataset="vaihingen")
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (14, 2, 17)
    assert sorted(s["train"] + s["val"]) == sorted(VAIHINGEN_TRAIN)
    assert not set(VAIHINGEN_TRAIN) & set(VAIHINGEN_TEST)
    with pytest.raises(ValueError, match="standard"):
        get_split("thesis2022", dataset="vaihingen")
    with pytest.raises(ValueError, match="training tiles"):
        get_split(val_tiles=["2"], dataset="vaihingen")


def test_prepare_vaihingen_with_computed_ndsm(tmp_path):
    raw = make_synthetic_vaihingen(tmp_path / "raw", size=SIZE)
    # a stray DSM outside the dsm/ folder (as in the real download) must be ignored
    shutil.copy(raw / "ISPRS_semantic_labeling_Vaihingen/dsm/dsm_09cm_matching_area11.tif", raw)
    cfg = _data_cfg(raw, tmp_path / "out", dataset="vaihingen", ndsm=True)
    counts = prepare_dataset(cfg)
    assert counts["test_tiles"] == 17 and counts["train"] > 0 and counts["val"] > 0
    split = json.loads((tmp_path / "out" / "split.json").read_text(encoding="utf-8"))
    assert split["dataset"] == "vaihingen" and split["ndsm"] == "computed"
    with pytest.raises(ValueError, match="no official nDSM"):
        prepare_dataset(_data_cfg(raw, tmp_path / "x", dataset="vaihingen", ndsm=True,
                                  ndsm_source="official"))  # fmt: skip


def test_compute_ndsm_removes_slope_and_keeps_objects():
    y, x = np.mgrid[0:600, 0:800].astype(np.float32)
    dsm = 250 + 0.002 * x + 0.001 * y  # gentle terrain, 0.1 m pixels
    dsm[200:400, 300:500] += 10.0  # a 20 m × 20 m, 10 m high building
    ndsm = compute_ndsm(dsm, gsd=0.1)
    assert ndsm.dtype == np.uint8
    assert abs(float(np.median(ndsm[220:380, 320:480])) - 100) <= 10  # decimetres
    assert np.percentile(ndsm[:150], 95) <= 5


def _pixel_model(channels: int) -> keras.Model:
    inputs = keras.Input((None, None, channels))
    return keras.Model(inputs, keras.layers.Conv2D(6, 1)(inputs))


def test_evaluation_reports_full_and_eroded(potsdam_ndsm, tmp_path):
    ecfg = ExperimentConfig().eval
    ecfg.window, ecfg.stride, ecfg.max_tiles, ecfg.save_predictions = 64, 32, 2, False
    report = evaluate_tiles(_pixel_model(4), potsdam_ndsm.root, ecfg)
    assert {"all_classes", "without_clutter"} <= set(report["eroded"])
    # boundaries are ignored in the eroded labels: fewer scored pixels
    assert np.sum(report["eroded"]["confusion_matrix"]) < np.sum(report["confusion_matrix"])
    assert "(eroded)" in report_markdown(report)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "report.json").write_text(json.dumps({"name": "r", **report}), "utf-8")
    assert "| r | 1 |" in aggregate_reports([tmp_path / "a" / "report.json"])
    assert "without_clutter" in evaluate_tiles(_pixel_model(3), potsdam_ndsm.root, ecfg)


def _write_frame(path: Path, rng: np.random.Generator) -> None:
    """16-bit tiled frame with an overview pyramid, like the Vaihingen DMC images."""
    import tifffile

    full = rng.integers(0, 16, (320, 256, 3)) * 256 + rng.integers(0, 4095, (320, 256, 3))
    full = (full // 16).astype(np.uint16)  # 12-bit data in 16-bit samples
    with tifffile.TiffWriter(path) as tw:
        for level in range(4):
            tw.write(full[:: 2**level, :: 2**level], photometric="rgb", tile=(64, 64))


def test_unlabeled_pool_and_city_guard(tmp_path, potsdam_ndsm):
    from rsseg.data.unlabeled import prepare_unlabeled
    from rsseg.ssl.train import pool_rows

    rng = np.random.default_rng(0)
    raw = tmp_path / "Images"
    raw.mkdir()
    for frame in ("10030060", "10030061"):
        _write_frame(raw / f"{frame}.tif", rng)
    (raw / "notes.tif").write_bytes(b"")  # not a frame name: ignored
    meta = prepare_unlabeled("vaihingen_frames", raw, tmp_path / "pool", crop_size=64)
    assert meta["city"] == "vaihingen" and meta["crops"] == 2 * 5 * 4
    crop = np.asarray(
        Image.open(tmp_path / "pool" / read_crops(tmp_path / "pool", "unlabeled")[0]["image"])
    )
    assert crop.dtype == np.uint8 and crop.max() > 200  # stretched, not the high byte only
    assert (tmp_path / "pool" / "previews" / "10030060.png").exists()

    pool = (tmp_path / "pool").as_posix()
    cfg = load_config(None, [f"data.root={potsdam_ndsm.root}", f"ssl.extra_pools=['{pool}']"])
    rows = pool_rows(cfg)
    assert len(rows) == 40 and Path(rows[0]["image"]).is_absolute()
    cfg.data.dataset = "vaihingen"
    with pytest.raises(ValueError, match="test areas"):
        pool_rows(cfg)
    cfg.ssl.allow_same_city_pool = True
    assert len(pool_rows(cfg)) == 40


def test_train_with_ndsm_end_to_end(potsdam_ndsm, tmp_path):
    from rsseg.train.seg_train import model_config, train_segmentation

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = load_config("configs/smoke.yaml", [
        f"output_dir={tmp_path / 'runs'}", f"tracking.tracking_uri={uri}",
        f"data.root={potsdam_ndsm.root}", "data.ndsm=true", "data.crop_size=64",
        "train.epochs=1", "train.steps_per_epoch=2", "train.validation_steps=1",
        "eval.window=64", "eval.stride=32", "eval.max_tiles=1",
    ])  # fmt: skip
    m = model_config(cfg)
    assert m.in_channels == 4 and m.extra_channel_std[0] > 0
    result = train_segmentation(cfg)
    run_dir = Path(result["run_dir"])
    assert "eroded" in json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    saved = load_config(run_dir / "config.yaml")
    assert saved.model.in_channels == 4 and saved.model.extra_channel_mean == m.extra_channel_mean
