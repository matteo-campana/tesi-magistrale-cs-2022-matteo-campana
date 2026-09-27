# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qt-free core of the application: jobs, times, GPU, tooltips, runs."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from rsseg import control, progress
from rsseg.gui.core import gpu, help, runs, timing
from rsseg.gui.core.jobs import Job, JobStore, pid_alive


def _wait(store: JobStore, job: Job, timeout: float = 120) -> None:
    t0 = time.time()
    while job.state in ("queued", "running", "pausing", "stopping"):
        assert time.time() - t0 < timeout, store.log_tail(job)
        store.tick()
        time.sleep(0.2)


def test_core_does_not_import_torch_keras_or_qt():
    code = ("import sys, rsseg.gui.core.jobs, rsseg.gui.core.timing, rsseg.gui.core.gpu, "
            "rsseg.gui.core.help, rsseg.gui.core.runs, rsseg.eta, rsseg.campaign\n"
            "rsseg.gui.core.help.config_fields(); rsseg.gui.core.help.catalog()\n"
            "print([m for m in ('torch', 'keras', 'mlflow', 'PySide6') if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_job_runs_detached_and_is_persisted(tmp_path):
    store = JobStore(tmp_path)
    job = store.add("synth", ["--out", str(tmp_path / "raw"), "--size", "64"], "Dati sintetici")
    assert not job.uses_gpu and store.gpu_busy() is None
    store.tick()
    assert job.state == "running" and job.pid
    _wait(store, job)
    assert job.state == "done" and job.exit_code == 0, store.log_tail(job)
    p = store.progress(job)
    assert p["state"] == "done" and p["command"] == "synth"
    assert any((tmp_path / "raw").rglob("*.tif"))
    again = JobStore(tmp_path)  # a new session reads the record back
    assert again.jobs[job.id].state == "done" and again.jobs[job.id].title == "Dati sintetici"


def test_gpu_jobs_are_queued_one_at_a_time(tmp_path):
    store = JobStore(tmp_path)
    a = store.add("train", ["-c", "missing.yaml"], "A")
    b = store.add("train", ["-c", "missing.yaml"], "B")
    c = store.add("synth", ["--out", str(tmp_path / "s"), "--size", "64"], "C")
    store.tick()
    assert (a.state, b.state, c.state) == ("running", "queued", "running")
    store.request(b, "pause")  # a queued job is just dropped
    assert b.state == "stopped" and not b.resumable
    _wait(store, a)
    assert a.state == "failed" and a.error  # missing config: the error is reported
    _wait(store, c)


def test_interrupted_job_is_detected_after_a_restart(tmp_path):
    store = JobStore(tmp_path)
    job = store.add("train", ["-c", "x.yaml"], "T")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    job.state, job.pid, job.started = "running", proc.pid, time.time()
    store.folder(job).mkdir(parents=True)
    run = tmp_path / "runs" / "seg-x-20260101-000000"
    (run / "resume").mkdir(parents=True)
    (run / "resume" / "state.json").write_text('{"phase": "evaluation"}', encoding="utf-8")
    store.progress_path(job).write_text(json.dumps({"state": "running", "run_dir": str(run)}),
                                        encoding="utf-8")  # fmt: skip
    store.save()
    fresh = JobStore(tmp_path)
    fresh.refresh()
    j = fresh.jobs[job.id]
    assert j.state == "interrupted" and j.run_dir == str(run) and j.resumable
    fresh.resume(j)
    assert j.state == "queued" and j.args == ["--resume", str(run)]
    assert not pid_alive(proc.pid)


def test_request_writes_the_control_file(tmp_path):
    store = JobStore(tmp_path)
    job = store.add("train", [], "T", state="running")
    store.request(job, "pause")
    assert job.state == "pausing"
    assert json.loads(store.control_path(job).read_text(encoding="utf-8"))["action"] == "pause"
    prep = store.add("prepare", [], "P", state="running")
    store.request(prep, "pause")  # not pausable: becomes a stop
    assert prep.state == "stopping"


def test_describe_training_progress():
    job = Job("1", "train", [], "T", state="running", started=1000.0)
    now = 1000.0 + 600
    p = {"started": 1000.0, "active_before_s": 300.0, "updated": now, "phase": "train",
         "stage": "train", "epoch": 5, "epochs": 50, "batch": 50, "steps": 100, "epoch_s": 100.0,
         "patience": 10, "best_epoch": 3, "best": 0.5, "monitor": "val_miou", "since_best": 1,
         "history": [{"epoch": 1, "loss": 1.0}]}  # fmt: skip
    v = timing.describe(job, p, now)
    assert v.elapsed_s == pytest.approx(900)  # 300 before the resume + 600 now
    assert [b.label for b in v.bars] == ["Epoca", "Batch"]
    assert v.bars[0].fraction == pytest.approx(4 / 50) and v.bars[1].fraction == 0.5
    # early stopping: at best 13 epochs in total, at most 50
    assert v.remaining_s == pytest.approx((13 - 4.5) * 100)
    assert v.remaining_max_s == pytest.approx((50 - 4.5) * 100)
    assert v.finish_at == pytest.approx(now + v.remaining_s)
    assert [s for _, s in v.phases] == ["done", "active", "todo", "todo"]
    assert "early stopping: 1/10" in v.notes[0] and v.history

    p_eval = {**p, "phase": "evaluation", "stage": "test", "tile": 2, "tiles": 4, "tile_s": 30.0}
    v = timing.describe(job, p_eval, now)
    assert v.bars[0].label == "Tile" and v.remaining_s == pytest.approx(90)
    assert [s for _, s in v.phases][:3] == ["done", "done", "active"]

    stale = timing.describe(job, {**p, "updated": now - 3600}, now)
    assert stale.stale and "nessun avanzamento" in stale.notes[-1]
    done = Job("2", "train", [], "T", state="done")
    assert timing.describe(done, {**p, "state": "done"}, now).total_fraction == 1.0
    queued = Job("3", "train", [], "T", estimate_s=120.0)
    assert timing.describe(queued, None, now).remaining_s == 120.0


def test_describe_counts_and_prediction():
    job = Job("1", "prepare", [], "P", state="running")
    v = timing.describe(job, {"started": 0.0, "phase": "prepare", "stage": "tiles", "done": 5,
                              "total": 20}, 50.0)  # fmt: skip
    assert v.bars[0].label == "Tile" and v.remaining_s == pytest.approx(150)
    job = Job("2", "predict", [], "I", state="running")
    v = timing.describe(job, {"started": 0.0, "phase": "predict", "done": 1, "total": 3,
                              "window": 5, "windows": 10}, 30.0)  # fmt: skip
    assert v.remaining_s == pytest.approx(30 / 1.5 * 1.5)


def test_fmt_duration():
    assert timing.fmt_duration(None) == "—"
    assert timing.fmt_duration(42) == "42 s"
    assert timing.fmt_duration(3 * 60 + 5) == "3 min 05 s"
    assert timing.fmt_duration(2 * 3600 + 5 * 60) == "2 h 05 min"
    assert timing.fmt_range(600, 3600) == "10 min 00 s – 1 h 00 min"


def test_gpu_parsing_status_and_preflight():
    text = "0, NVIDIA GeForce RTX 4060, 87, 6144, 8188, 71, 105.5, 115.0, 2595, [N/A], 560.94\n"
    samples, driver = gpu.parse_smi(text, now=1.0)
    s = samples[0]
    assert driver == "560.94" and s.name == "NVIDIA GeForce RTX 4060" and s.util == 87
    assert s.mem_used_gb == pytest.approx(6.0) and s.fan_pct is None
    assert gpu.status([], None)[0] == "absent"
    level, msg = gpu.status(samples, {"torch": "2.5.0+cpu", "cuda_available": False})
    assert level == "warning" and "senza CUDA" in msg and "cu128" in msg
    assert (
        gpu.status(samples, {"cuda_available": True, "cuda_version": "12.8", "bf16": True})[0]
        == "ready"
    )
    s.processes = [gpu.GPUProcess(1, "chrome", 1.2), gpu.GPUProcess(2, "python", 3.0)]
    warn = gpu.preflight(s, needed_gb=3.0, own_pids={2}, gpu_job_running=False)
    assert "chrome" in warn[0] and "servono" in warn[1]
    assert gpu.preflight(None, None, set(), False)[0].startswith("nessuna GPU")
    full = gpu.GPUSample(0, 0, "x", 100, 7.9, 8.0, temp_c=90)
    assert len(gpu.alerts(full)) == 2


def test_gpu_log_roundtrip(tmp_path):
    log = gpu.GPULog(tmp_path / "gpu_log.csv")
    log.append(gpu.GPUSample(1.0, 0, "x", 50, 2.0, 8.0, temp_c=60))
    log.append(gpu.GPUSample(2.0, 0, "x", None, 2.5, 8.0))
    rows = log.read()
    assert [r["util"] for r in rows] == [50.0, None] and rows[1]["mem_used_gb"] == 2.5


def test_every_setting_has_an_italian_tooltip():
    assert help.missing_config_help() == []
    fields = {f.key: f for f in help.config_fields()}
    assert (
        fields["train.lr"].kind == "float"
        and fields["train.steps_per_epoch"].kind == "optional-int"
    )
    assert fields["model.vq.num_codes"].doc == ""  # no docstring: the Italian text is needed
    tip = help.config_tooltip(fields["train.precision"], source="configs/base.yaml")
    assert "mixed_bfloat16" in tip and "Predefinito" in tip and "configs/base.yaml" in tip
    for section in ("options", "metrics", "gpu"):
        assert help.catalog()[section]


def test_runs_are_found_and_described(tmp_path):
    run = tmp_path / "runs" / "seg-demo-20260101-120000"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"model": {"arch": "resnet34",
                                                               "weights": "imagenet"}}), "utf-8")  # fmt: skip
    (run / "best.weights.h5").write_bytes(b"x")
    (run / "checkpoints" / "e.weights.h5").write_bytes(b"x")
    (run / "checkpoints" / "topk.json").write_text(
        json.dumps([{"epoch": 3, "val_miou": 0.6, "path": "e.weights.h5"}]), "utf-8"
    )
    (run / "history.csv").write_text("epoch,loss,val_miou\n0,1.5,0.4\n1,1.2,0.6\n", "utf-8")
    (run / "report.json").write_text(json.dumps({"all_classes": {"miou": 0.55}}), "utf-8")
    (tmp_path / "runs" / "not-a-run").mkdir()
    found = runs.find_runs([tmp_path / "runs"])
    assert len(found) == 1
    r = found[0]
    assert (r.kind, r.name, r.arch, r.pretrain, r.channels, r.test_miou) == (
        "seg",
        "demo",
        "resnet34",
        "imagenet",
        3,
        0.55,
    )
    assert [lbl for lbl, _ in r.checkpoints()][1].startswith("epoca 3")
    assert r.history()[1]["val_miou"] == 0.6 and r.state == "incomplete"


def test_control_and_progress_env_names_are_stable():
    # the application and the commands meet through these names
    assert (progress.ENV, control.ENV) == ("RSSEG_PROGRESS_FILE", "RSSEG_CONTROL_FILE")
    assert Path(help.HELP_FILE).exists()
