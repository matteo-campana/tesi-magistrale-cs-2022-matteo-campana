# SPDX-License-Identifier: AGPL-3.0-or-later
"""``rsseg predict``: outputs on the tiles of a split (with scores) and on loose images."""

import json

import numpy as np
from PIL import Image

from rsseg import progress
from rsseg.cli import main
from rsseg.config import load_config, save_config
from rsseg.data.datasets import read_tiles


def _run_dir(tmp_path, data_cfg):
    from rsseg.models.segmentor import build_segmentor

    cfg = load_config("configs/smoke.yaml", [f"data.root={data_cfg.root}", "eval.window=64",
                                             "eval.stride=32"])  # fmt: skip
    run = tmp_path / "seg-smoke"
    save_config(cfg, run / "config.yaml")
    build_segmentor(cfg.model).save_weights(str(run / "best.weights.h5"))
    return run


def test_predict_split_and_images(synthetic_dataset, tmp_path, monkeypatch):
    data_cfg, _ = synthetic_dataset
    run = _run_dir(tmp_path, data_cfg)
    prog = tmp_path / "progress.json"
    monkeypatch.setenv(progress.ENV, str(prog))
    progress.reset()

    out = tmp_path / "pred"
    assert main(["--json", "predict", "--run", str(run), "--split", "test", "--out", str(out)]) == 0
    summary = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    tiles = read_tiles(data_cfg.root, "test")
    assert len(summary["images"]) == len(tiles) and 0 <= summary["scores"]["miou"] <= 1
    first = summary["images"][0]
    pred = np.asarray(Image.open(first["pred"]))
    assert pred.shape == tuple(first["size"]) and pred.max() < 6
    assert (
        Image.open(first["color"]).mode == "RGB" and (out / f"{first['name']}_errors.png").exists()
    )
    assert abs(sum(first["class_fractions"].values()) - 1) < 1e-6
    p = progress.read(prog)
    assert p["state"] == "done" and p["done"] == p["total"] == len(tiles)
    assert p["windows"] and p["window"] == p["windows"]

    # loose images (a folder), without labels
    folder = tmp_path / "images"
    folder.mkdir()
    Image.open(tiles[0]["image"]).save(folder / "a.png")
    out2 = tmp_path / "pred2"
    assert main(["predict", "--run", str(run), str(folder), "--out", str(out2), "--tta"]) == 0
    summary2 = json.loads((out2 / "predictions.json").read_text(encoding="utf-8"))
    assert [i["name"] for i in summary2["images"]] == ["a"] and "scores" not in summary2
    assert (out2 / "a_confidence.png").exists()
    progress.reset()
