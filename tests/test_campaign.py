# SPDX-License-Identifier: AGPL-3.0-or-later
"""Campaigns: expansion, resume, failures, prerequisites, progress (fake runner) and one real run."""

import contextlib
import json
from pathlib import Path

import pytest
import yaml

from rsseg.campaign import (
    QUICK_TRAIN,
    STALE_S,
    CampaignError,
    Layout,
    Options,
    Running,
    Step,
    command,
    describe_progress,
    load_campaign,
    plan_campaign,
    run_campaign,
    select,
    subprocess_runner,
)
from rsseg.cli import main
from rsseg.config import load_config
from rsseg.data.synthetic import make_synthetic_potsdam
from rsseg.progress import ENV, FILE_NAME, read
from rsseg.train.callbacks import ProgressReporter

SMOKE = "configs/smoke.yaml"
SCORES = {"oa": 0.9, "mf1": 0.8, "miou": 0.7}


def _small(raw: Path, root: Path, tmp: Path) -> list[str]:
    """Overrides that make configs/smoke.yaml tiny and point it at synthetic data."""
    return [f"data.raw_dir={raw.as_posix()}", f"data.root={root.as_posix()}", "data.crop_size=64",
            "data.train_stride=64", "eval.window=64", "eval.stride=32", "eval.max_tiles=1",
            "train.epochs=1", "train.steps_per_epoch=2", "train.validation_steps=1",
            "ssl.global_crop_size=64", "ssl.local_crop_size=32",
            f"tracking.tracking_uri=sqlite:///{(tmp / 'mlflow.db').as_posix()}"]  # fmt: skip


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:
    return make_synthetic_potsdam(tmp_path_factory.mktemp("campaign_raw"), size=128)


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class FakeRunner:
    """Creates the files a real step would create; ``fail`` lists step names that fail."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls: list[str] = []

    def __call__(self, cmd, log_path, stdout=None):
        kind = cmd[4]
        sets = dict(a.split("=", 1) for a in cmd[cmd.index("--set") + 1 :] if "=" in a)
        name = sets.get("name", Path(log_path).parent.name)
        self.calls.append(f"{kind}:{name}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake\n", encoding="utf-8")
        if name in self.fail:
            return 1
        if kind == "pretrain":
            Path(sets["ssl.export_path"]).write_bytes(b"weights")
        elif kind == "train":
            run = Path(sets["output_dir"]) / f"seg-{name}-20260101-000000"
            run.mkdir(parents=True)
            report = {"name": name, "all_classes": SCORES, "without_clutter": SCORES}
            (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
        else:
            stdout.write_text('noise\n{"mode": "seg", "params_millions": 16.2}\n', "utf-8")
        return 0


def _campaign(tmp_path, raw, **extra) -> Path:
    small = _small(raw, tmp_path / "data", tmp_path)
    data = {
        "name": "demo",
        "seeds": [0, 1],
        "pretrain": {
            "dino_a": {"config": SMOKE, "set": small},
            "dino_bad": {"config": SMOKE, "set": small},
        },
        "runs": [
            {"config": SMOKE, "name": "base", "set": small},
            {"config": SMOKE, "name": "lf{data.label_fraction}", "set": small,
             "sweep": {"data.label_fraction": [0.5, 1.0]}},
            {"config": SMOKE, "name": "ft", "backbone": "dino_a", "set": small},
            {"config": SMOKE, "name": "ft_bad", "backbone": "dino_bad", "set": small},
        ],
        "benchmarks": [{"config": SMOKE, "name": "bench", "set": small}],
        **extra,
    }  # fmt: skip
    return _write(tmp_path / "demo.yaml", data)


def _opts(tmp_path, **kw) -> Options:
    return Options(root=str(tmp_path / "campaigns"), **kw)


def _counter(keys):
    return {k: 1000 for k in keys}


def test_load_expands_sweeps_seeds_and_backbones(tmp_path, raw):
    c = load_campaign(_campaign(tmp_path, raw))
    ids = [s.id for s in c.runs]
    assert len(ids) == 10 and ids[:5] == [
        "base__s0",
        "lf0.5__s0",
        "lf1.0__s0",
        "ft__s0",
        "ft_bad__s0",
    ]
    assert set(c.pretrains) == {"dino_a", "dino_bad"} and c.shared_seed_pretrains == [
        "dino_a",
        "dino_bad",
    ]
    assert next(s for s in c.runs if s.id == "lf0.5__s1").cfg.data.label_fraction == 0.5
    assert load_campaign(tmp_path / "demo.yaml", seeds=[3]).runs[0].id == "base__s3"
    only = select(c, ["ft__*"])
    assert [s.id for s in only.runs] == ["ft__s0", "ft__s1"] and set(only.pretrains) == {"dino_a"}


def test_per_seed_pretrain_and_include_dedup(tmp_path, raw):
    small = _small(raw, tmp_path / "data", tmp_path)
    _write(tmp_path / "a.yaml", {"name": "a", "seeds": [0, 1],
                                  "pretrain": {"d": {"config": SMOKE, "set": small, "per_seed": True}},
                                  "runs": [{"config": SMOKE, "name": "ft", "backbone": "d", "set": small}]})  # fmt: skip
    _write(tmp_path / "b.yaml", {"name": "b", "seeds": [0, 1], "include": ["a.yaml"],
                                  "runs": [{"config": SMOKE, "name": "ft", "backbone": "d", "set": small}],
                                  "pretrain": {"d": {"config": SMOKE, "set": small, "per_seed": True}}})  # fmt: skip
    c = load_campaign(tmp_path / "b.yaml")
    assert sorted(c.pretrains) == ["d_s0", "d_s1"] and [s.backbone for s in c.runs] == [
        "d_s0",
        "d_s1",
    ]
    assert "ssl.seed=1" in c.pretrains["d_s1"].overrides and c.shared_seed_pretrains == []


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"runs": [{"config": SMOKE, "name": "x"}, {"config": SMOKE, "name": "x", "set": ["train.lr=0.1"]}]},
         "defined twice"),
        ({"runs": [{"config": SMOKE, "backbone": "nope"}]}, "unknown backbone"),
        ({"runs": [{"config": SMOKE, "name": "x{train.lr}"}]}, "not a sweep key"),
        ({"runs": [{"config": SMOKE, "set": ["train.not_a_key=1"]}]}, "not_a_key"),
        ({"runs": [{"config": "configs/missing.yaml"}]}, "not found"),
        ({"typo": 1}, "unknown keys"),
    ],
)  # fmt: skip
def test_invalid_campaigns(tmp_path, bad, match):
    with pytest.raises(CampaignError, match=match):
        load_campaign(_write(tmp_path / "c.yaml", {"name": "c", **bad}))


def test_run_prepares_data_continues_after_failures_and_resumes(tmp_path, raw):
    c = load_campaign(_campaign(tmp_path, raw))
    runner = FakeRunner(fail={"lf0.5", "dino_bad"})
    summary = run_campaign(c, _opts(tmp_path), runner, _counter)
    # data prepared once by the campaign itself
    assert (tmp_path / "data" / "crops.csv").exists()
    split = json.loads((tmp_path / "data" / "split.json").read_text(encoding="utf-8"))
    assert split["crop_size"] == 64
    assert summary["counts"] == {"done": 8, "failed": 3, "blocked": 2}
    assert summary["steps"]["ft_bad__s0"]["reason"] == "dino_bad failed"
    assert runner.calls[:2] == ["pretrain:dino_a", "pretrain:dino_bad"]
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "| base | 2 | 0.9000 ± 0.0000" in report and "bench" in report
    assert "Failed or blocked" in report and "single seed" in report
    status = json.loads((tmp_path / "campaigns" / "demo" / "status.json").read_text("utf-8"))
    assert status["counts"] == summary["counts"]

    # resume: only the failed and blocked steps run again
    runner2 = FakeRunner()
    summary2 = run_campaign(c, _opts(tmp_path), runner2, _counter)
    assert sorted(runner2.calls) == sorted(["pretrain:dino_bad", "train:lf0.5", "train:lf0.5",
                                            "train:ft_bad", "train:ft_bad"])  # fmt: skip
    assert summary2["counts"] == {"done": 13}


def test_stop_on_error_and_quick_root(tmp_path, raw):
    c = load_campaign(_campaign(tmp_path, raw))
    with pytest.raises(CampaignError, match="dino_bad failed"):
        run_campaign(
            c, _opts(tmp_path, stop_on_error=True), FakeRunner(fail={"dino_bad"}), _counter
        )
    quick = Layout(Path(str(tmp_path / "campaigns") + "-quick"))
    cmd = command(quick, c.runs[0], c.name, quick=True)
    assert all(q in cmd for q in QUICK_TRAIN)
    assert any(a.startswith(f"output_dir={quick.root.as_posix()}/store/runs/base__s0") for a in cmd)
    plan = plan_campaign(c, _opts(tmp_path, quick=True), _counter)
    assert plan["root"].endswith("campaigns-quick")


def test_prerequisite_checks(tmp_path, raw):
    c = load_campaign(_campaign(tmp_path, raw))
    run_campaign(c, _opts(tmp_path), FakeRunner(), _counter)  # prepares data with crop 64
    # same root, other crop size: refused instead of overwritten
    other = [*_small(raw, tmp_path / "data", tmp_path), "data.crop_size=32"]
    bad = _write(tmp_path / "bad.yaml", {"name": "bad", "runs": [{"config": SMOKE, "set": other}]})
    with pytest.raises(CampaignError, match="crop_size"):
        plan_campaign(load_campaign(bad), _opts(tmp_path), _counter)
    # a different run under an existing id
    spec = tmp_path / "campaigns" / "store" / "runs" / "base__s0" / "spec.json"
    spec.write_text(json.dumps({"other": True}), encoding="utf-8")
    with pytest.raises(CampaignError, match="different settings"):
        plan_campaign(c, _opts(tmp_path), _counter)
    spec.unlink()
    # not enough disk
    fresh = load_campaign(_campaign(tmp_path, raw), seeds=[7])
    with pytest.raises(CampaignError, match="GB are free"):
        run_campaign(fresh, _opts(tmp_path), FakeRunner(), lambda keys: {k: 10**15 for k in keys})


def test_progress_reporter_follows_early_stopping(tmp_path, monkeypatch):
    path = tmp_path / FILE_NAME
    cb = ProgressReporter("train", "val_miou", patience=3, every=2)
    cb.set_params({"epochs": 10, "steps": 4})
    cb.on_train_begin()
    assert not path.exists()  # no campaign asked for it
    monkeypatch.setenv(ENV, str(path))
    for epoch, miou in enumerate([0.5, 0.6, 0.55, 0.58]):
        cb.on_epoch_begin(epoch)
        for batch in range(4):
            cb.on_train_batch_end(batch, {"loss": 1.0})
        cb.on_test_begin()
        cb.on_epoch_end(epoch, {"loss": 1.0, "val_miou": miou})
    p = read(path)
    assert p["epoch"] == 4 and p["batch"] == 4 and p["stage"] == "validation"
    assert (p["best"], p["best_epoch"], p["since_best"]) == (0.6, 2, 2)
    assert p["epoch_s"] >= 0 and p["updated"] > 0


def _running(tmp_path, after=("train", "train")) -> Running:
    step = Step("train", "base__s0", "base", SMOKE, [], seed=0)
    return Running(step, 2, 5, 1000.0, tmp_path / FILE_NAME, tmp_path / "log.txt", list(after),
                   {"pretrain": 600.0, "train": 3 * 3600.0, "benchmark": None})  # fmt: skip


def test_describe_progress(tmp_path):
    r = _running(tmp_path)
    assert describe_progress(r, None, 1060.0)[1].startswith("starting")
    train = {"phase": "train", "stage": "train", "epoch": 12, "epochs": 100, "batch": 198,
             "steps": 396, "monitor": "val_miou", "value": 0.712, "best": 0.718, "best_epoch": 10,
             "since_best": 1, "patience": 20, "epoch_s": 168.0, "updated": 1990.0}  # fmt: skip
    lines = describe_progress(r, train, 2000.0)
    assert lines[0] == "[2/5] train base__s0 — running for 16m40s"
    assert lines[1] == ("epoch 12/100 · batch 198/396 · val_miou 0.7120 (best 0.7180 at "
                        "epoch 10)")  # fmt: skip
    assert lines[2] == "2m48s per epoch · early stop after 19 more epochs without improvement"
    # 19 epochs + half of the current one if it stops early, 88.5 epochs otherwise
    assert lines[3] == "this step: ~54m36s–4h07m left + test evaluation"
    assert lines[4] == "then 2 more steps: ~6h00m" and lines[5].startswith("log: ")
    evaluation = {"phase": "evaluation", "stage": "test", "tile": 5, "tiles": 14, "tile_s": 30.0,
                  "updated": 1000.0}  # fmt: skip
    lines = describe_progress(_running(tmp_path, ["benchmark"]), evaluation, 1000.0 + STALE_S + 60)
    assert lines[1] == "test evaluation · tile 5/14 · 0m30s per tile"
    assert lines[2] == "this step: ~5m00s left"
    assert lines[3].startswith("no progress for 16m00s")
    assert "n/a" in lines[4]


def test_run_watches_every_step_that_runs(tmp_path, raw):
    c = load_campaign(_campaign(tmp_path, raw))
    seen = []

    @contextlib.contextmanager
    def watch(r):
        assert not r.progress_path.exists()  # a stale file of a previous attempt is removed
        seen.append((r.step.id, r.index, r.total, len(r.after)))
        yield

    stale = tmp_path / "campaigns" / "store" / "backbones" / "dino_a" / FILE_NAME
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    run_campaign(c, _opts(tmp_path), FakeRunner(fail={"dino_bad"}), _counter, watch=watch)
    assert seen[0] == ("dino_a", 1, 13, 12) and seen[-1] == ("bench", 13, 13, 0)
    # the two fine-tunings of the failed backbone are blocked: they never run
    assert len(seen) == 11 and not {"ft_bad__s0", "ft_bad__s1"} & {s[0] for s in seen}


def test_report_out_creates_the_folder(tmp_path, capsys):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "report.json").write_text(
        json.dumps({"name": "r", "all_classes": SCORES, "without_clutter": SCORES}), "utf-8"
    )
    out = tmp_path / "new" / "dir" / "table.md"
    assert main(["report", str(tmp_path / "r"), "--out", str(out), "--json"]) == 0
    assert out.exists()


def test_repository_campaigns_load():
    for path in Path("configs/campaigns").glob("*.yaml"):
        c = load_campaign(path)
        assert c.runs or c.benchmarks, path
    all_ = load_campaign("configs/campaigns/all.yaml")
    ids = [s.id for s in all_.runs]
    assert len(ids) == len(set(ids)) and "maresunet_r18_imagenet__s0" in ids
    assert (
        load_config("configs/experiments/maresunet_r18_imagenet.yaml").name
        == "maresunet_r18_imagenet"
    )


def test_campaign_end_to_end_with_real_subprocesses(tmp_path, raw):
    small = _small(raw, tmp_path / "data", tmp_path)
    path = _write(tmp_path / "e2e.yaml", {
        "name": "e2e", "seeds": [0],
        "pretrain": {"dino": {"config": SMOKE, "set": small}},
        "runs": [{"config": SMOKE, "name": "scratch", "set": small},
                 {"config": SMOKE, "name": "from_dino", "backbone": "dino", "set": small}],
    })  # fmt: skip
    c = load_campaign(path)
    summary = run_campaign(c, _opts(tmp_path), subprocess_runner)
    assert summary["counts"] == {"done": 3}, summary
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "| scratch | 1 |" in report and "| from_dino | 1 |" in report
    log = (tmp_path / "campaigns" / "store" / "runs" / "from_dino__s0" / "log.txt").read_text(
        "utf-8"
    )
    assert "model.weights=" in log and "backbone.weights.h5" in log
    # every step shared its progress with the campaign
    store = tmp_path / "campaigns" / "store"
    assert read(store / "backbones" / "dino" / FILE_NAME)["phase"] == "pretrain"
    p = read(store / "runs" / "scratch__s0" / FILE_NAME)
    assert p["phase"] == "evaluation" and p["tile"] == p["tiles"] == 1

    def never(*_):
        raise AssertionError("a completed step was run again")

    assert run_campaign(c, _opts(tmp_path), never)["counts"] == {"done": 3}
