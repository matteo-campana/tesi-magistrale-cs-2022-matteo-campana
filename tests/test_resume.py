# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pause and resume: a paused and resumed run must equal an uninterrupted one."""

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from rsseg import control, progress
from rsseg.config import load_config
from rsseg.runstate import (
    environment_changes,
    find_resumable,
    is_resumable,
    read_state,
    unsafe_overrides,
)


def _cfg(tmp_path, data_cfg, out: str, extra=()):
    return load_config("configs/smoke.yaml", [
        f"output_dir={tmp_path / out}", f"tracking.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
        f"data.root={data_cfg.root}", "data.crop_size=64", "train.epochs=3", "train.steps_per_epoch=3",
        "train.validation_steps=1", "train.early_stopping_patience=null", "train.log_images_every=0",
        "train.checkpoint_every_min=0", "eval.window=64", "eval.stride=32", "eval.max_tiles=2",
        "eval.save_predictions=false", *extra,
    ])  # fmt: skip


class PauseAt:
    """Fake :func:`rsseg.control.pending`: ask a pause at the ``n``-th call from ``where``."""

    def __init__(self, where: str, n: int):
        self.where, self.n, self.calls = where, n, 0

    def __call__(self):
        caller = sys._getframe(1).f_code.co_name
        if caller == self.where:
            self.calls += 1
            if self.calls == self.n:
                return "pause"
        return None


def _history(run_dir: Path) -> list[dict]:
    with (run_dir / "history.csv").open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items() if v} for row in csv.DictReader(f)]


def _weights(path: Path) -> list[np.ndarray]:
    arrays = []
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda _, obj: arrays.append(obj[()]) if isinstance(obj, h5py.Dataset) else None
        )
    return arrays


def test_paused_and_resumed_training_equals_uninterrupted(synthetic_dataset, tmp_path, monkeypatch):
    from rsseg.train.seg_train import train_segmentation

    data_cfg, _ = synthetic_dataset
    straight = train_segmentation(_cfg(tmp_path, data_cfg, "straight"))

    # pause in the middle of epoch 2 (batch 5 of 9)
    monkeypatch.setattr(control, "pending", PauseAt("on_train_batch_end", 5))
    with pytest.raises(control.PauseRequested) as exc:
        train_segmentation(_cfg(tmp_path, data_cfg, "paused"))
    run_dir = Path(exc.value.run_dir)
    state = read_state(run_dir)
    assert (state["phase"], state["epoch"], state["batch"], state["reason"]) == (
        "train",
        1,
        2,
        "pause",
    )
    assert is_resumable(run_dir) and find_resumable(tmp_path / "paused") == [run_dir]

    # resume, then pause again during the evaluation (before the second tile)
    monkeypatch.setattr(control, "pending", PauseAt("check", 2))
    cfg = load_config(run_dir / "config.yaml")
    with pytest.raises(control.PauseRequested):
        train_segmentation(cfg, resume=run_dir)
    assert read_state(run_dir)["phase"] == "evaluation"
    assert (run_dir / "resume" / "evaluation.json").exists()

    monkeypatch.setattr(control, "pending", lambda: None)
    resumed = train_segmentation(cfg, resume=run_dir)
    assert not (run_dir / "resume").exists()  # a finished run keeps no resume checkpoint

    a, b = _history(Path(straight["run_dir"])), _history(run_dir)
    assert len(a) == len(b) == 3
    for ra, rb in zip(a, b, strict=True):
        assert ra.keys() == rb.keys()
        np.testing.assert_allclose([ra[k] for k in ra], [rb[k] for k in ra], rtol=1e-6, atol=1e-7)
    assert resumed["best_val_miou"] == pytest.approx(straight["best_val_miou"], abs=1e-7)
    assert resumed["test"]["miou"] == pytest.approx(straight["test"]["miou"], abs=1e-7)
    for wa, wb in zip(_weights(Path(straight["weights"])), _weights(Path(resumed["weights"])),
                      strict=True):  # fmt: skip
        np.testing.assert_allclose(wa, wb, rtol=1e-6, atol=1e-7)

    import mlflow

    run = mlflow.get_run(resumed["run_id"])
    assert run.data.tags["rsseg.resumes"] == "2" and run.data.tags["rsseg.state"] == "done"
    assert "test_miou" in run.data.metrics


def test_paused_and_resumed_dino_equals_uninterrupted(synthetic_dataset, tmp_path, monkeypatch):
    from rsseg.ssl.train import pretrain_dino

    data_cfg, _ = synthetic_dataset
    extra = ["ssl.epochs=2", "ssl.steps_per_epoch=3", "ssl.global_crop_size=64",
             "ssl.local_crop_size=32", "ssl.checkpoint_every_min=0"]  # fmt: skip
    a = pretrain_dino(
        _cfg(tmp_path, data_cfg, "a", [*extra, f"ssl.export_path={tmp_path / 'a.weights.h5'}"])
    )
    monkeypatch.setattr(control, "pending", PauseAt("on_train_batch_end", 4))
    cfg = _cfg(tmp_path, data_cfg, "b", [*extra, f"ssl.export_path={tmp_path / 'b.weights.h5'}"])
    with pytest.raises(control.PauseRequested) as exc:
        pretrain_dino(cfg)
    monkeypatch.setattr(control, "pending", lambda: None)
    run_dir = Path(exc.value.run_dir)
    b = pretrain_dino(load_config(run_dir / "config.yaml"), resume=run_dir)
    for wa, wb in zip(_weights(Path(a["backbone_weights"])), _weights(Path(b["backbone_weights"])),
                      strict=True):  # fmt: skip
        np.testing.assert_allclose(wa, wb, rtol=1e-6, atol=1e-7)


def test_stop_request_leaves_no_resume_checkpoint(synthetic_dataset, tmp_path, monkeypatch):
    from rsseg.cli import main

    data_cfg, _ = synthetic_dataset
    ctrl, prog = tmp_path / "control.json", tmp_path / "progress.json"
    monkeypatch.setenv(control.ENV, str(ctrl))
    monkeypatch.setenv(progress.ENV, str(prog))
    progress.reset()
    control.request(ctrl, "stop")
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg = _cfg(tmp_path, data_cfg, "stopped")
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    assert main(["train", "-c", str(cfg_path)]) == control.EXIT_STOPPED
    p = progress.read(prog)
    assert p["state"] == "stopped" and p["run_dir"]
    assert not (Path(p["run_dir"]) / "resume").exists()
    progress.reset()


def test_runstate_helpers():
    assert unsafe_overrides(["eval.tta=true", "data.num_workers=2"]) == []
    assert unsafe_overrides(["train.lr=0.1", "model.arch=resnet34"]) == ["train.lr", "model.arch"]
    state = {"env": {"torch": "2.0", "git_commit": "a" * 40, "gpu": "x"}}
    changes = environment_changes(state, {"torch": "2.1", "git_commit": "b" * 40})
    assert changes == ["torch: 2.0 → 2.1", "git_commit: aaaaaaaa → bbbbbbbb"]


def test_control_and_progress_files(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    monkeypatch.setenv(control.ENV, str(path))
    assert control.pending() is None
    control.request(path, "pause")
    assert control.pending() == "pause"
    with pytest.raises(control.PauseRequested):
        control.check()
    with pytest.raises(control.StopRequested):
        control.check(pausable=False)
    control.clear()
    assert control.pending() is None
    with pytest.raises(ValueError):
        control.request(path, "explode")

    prog = tmp_path / "p.json"
    monkeypatch.setenv(progress.ENV, str(prog))
    progress.reset()
    assert list(progress.tracked([1, 2, 3], phase="prepare")) == [1, 2, 3]
    p = progress.read(prog)
    assert (p["done"], p["total"], p["phase"], p["state"]) == (3, 3, "prepare", "running")
    progress.finish("done")
    assert progress.read(prog)["state"] == "done"
    assert json.loads(prog.read_text(encoding="utf-8"))["pid"] > 0
    progress.reset()
