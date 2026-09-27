# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Qt application, headless (``QT_QPA_PLATFORM=offscreen``), and a real pause/resume of a
training job across an application restart."""

import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from rsseg.config import load_config
from rsseg.gui.core.jobs import JobStore
from rsseg.runstate import read_state

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path, synthetic_dataset):
    """A project folder with the repository configs and a tiny smoke configuration."""
    shutil.copytree("configs", tmp_path / "configs")
    data_cfg, _ = synthetic_dataset
    cfg = load_config("configs/smoke.yaml", [
        f"data.root={data_cfg.root}", "data.crop_size=64", "train.epochs=3", "train.steps_per_epoch=3",
        "train.validation_steps=1", "train.checkpoint_every_min=0", "train.log_images_every=0",
        "eval.window=64", "eval.stride=32", "eval.max_tiles=1",
        f"tracking.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}", "output_dir=runs",
    ])  # fmt: skip
    (tmp_path / "configs" / "tiny.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    return tmp_path


def _wait(store, job, states, timeout=300):
    t0 = time.time()
    while job.state not in states:
        assert time.time() - t0 < timeout, store.log_tail(job)
        store.tick()
        time.sleep(0.3)


def test_training_job_pauses_and_resumes_after_an_app_restart(project):
    store = JobStore(project)
    job = store.add("train", ["-c", "configs/tiny.yaml"], "Training tiny")
    store.tick()
    assert job.state == "running"
    store.request(job, "pause")  # reaches the job while it starts: it pauses at the first batch
    _wait(store, job, ("paused", "failed", "done", "interrupted"))
    assert job.state == "paused", store.log_tail(job)
    assert job.exit_code == 10 and job.resumable
    state = read_state(job.run_dir)
    assert state["phase"] == "train" and state["batch"] >= 1

    restarted = JobStore(project)  # the application is closed and opened again
    restarted.refresh()
    again = restarted.jobs[job.id]
    assert again.state == "paused" and again.resumable
    restarted.resume(again)
    restarted.tick()
    assert again.state == "running" and again.sessions == 2
    _wait(restarted, again, ("done", "failed", "paused", "interrupted"))
    assert again.state == "done", restarted.log_tail(again)
    run = Path(again.run_dir)
    assert (run / "report.json").exists() and not (run / "resume").exists()
    p = restarted.progress(again)
    assert p["active_before_s"] > 0 and len(p["history"]) == 3


def test_main_window_builds_every_page(qapp, project):
    from rsseg.gui.app import PAGES, MainWindow
    from rsseg.gui.context import AppContext

    ctx = AppContext(project, gpu=False, tick_ms=100000)
    window = MainWindow(ctx)
    window.show()
    for key, _, _ in PAGES:
        window.show_page(key)
        qapp.processEvents()
        assert window.stack.currentWidget() is window.pages[key]
    ctx.tick()
    window._gpu([])
    assert "non rilevata" in window.gpu_status.text.text()
    assert window.grab().width() > 0
    ctx.shutdown()
    window.close()


def test_config_page_tooltips_and_launch(qapp, project):
    from rsseg.gui.context import AppContext
    from rsseg.gui.pages.config import ConfigPage

    ctx = AppContext(project, gpu=False, tick_ms=100000)
    page = ConfigPage(ctx)
    page.load(project / "configs" / "smoke.yaml")
    epochs = page.editors["train.epochs"]
    assert "Numero massimo di epoche" in epochs.widget.toolTip()
    assert "configs/smoke.yaml" in epochs.widget.toolTip()  # where the value comes from
    assert "configs/base.yaml" in page.editors["train.optimizer"].widget.toolTip()
    assert all(ed.widget.toolTip() for ed in page.editors.values())  # every setting has one
    assert page.overrides() == []
    epochs.widget.setValue(5)
    page.editors["train.lr"].widget.setText("1e-4")
    assert page.overrides() == ["train.epochs=5", "train.lr=0.0001"]
    assert "font-weight: bold" in epochs.label.styleSheet()
    assert "Batch effettivo" in page.summary.text()
    page.train = page.train  # launch without the GPU dialog (no GPU here)
    page._submit("train", "Training smoke", gpu_check=False)
    job = next(j for j in ctx.store.jobs.values() if j.title == "Training smoke")
    assert job.args == ["-c", "configs/smoke.yaml", "--set", "train.epochs=5", "train.lr=0.0001"]
    ctx.store.request(job, "stop")
    ctx.shutdown()


def test_job_progress_widget_shows_times_and_buttons(qapp):
    from rsseg.gui.core.jobs import Job
    from rsseg.gui.core.timing import describe
    from rsseg.gui.widgets import JobProgressWidget

    job = Job("1", "train", [], "T", state="running", started=0.0)
    now = 1000.0
    p = {"started": 0.0, "updated": now, "phase": "train", "epoch": 2, "epochs": 10, "batch": 5,
         "steps": 10, "epoch_s": 100.0}  # fmt: skip
    w = JobProgressWidget()
    w.set_view(describe(job, p, now), job)
    assert "Trascorso" in w.times.text() and "Rimanente" in w.times.text()
    assert not w.pause_btn.isHidden() and w.resume_btn.isHidden()
    bars = [b for _, b in w.bars if not b.isHidden()]
    assert len(bars) == 3 and bars[0].value() == 100  # epoch 1 of 10 done


def test_inference_page_shows_and_compares_predictions(qapp, project, synthetic_dataset):
    from rsseg.cli import main
    from rsseg.config import save_config
    from rsseg.gui.context import AppContext
    from rsseg.gui.pages.inference import InferencePage
    from rsseg.models.segmentor import build_segmentor

    data_cfg, _ = synthetic_dataset
    run = project / "runs" / "seg-demo-20260101-000000"
    cfg = load_config(project / "configs" / "tiny.yaml")
    save_config(cfg, run / "config.yaml")
    build_segmentor(cfg.model).save_weights(str(run / "best.weights.h5"))
    out = project / "runs" / "predictions" / "a"
    assert main(["predict", "--run", str(run), "--split", "test", "--out", str(out)]) == 0
    shutil.copytree(out, project / "runs" / "predictions" / "b")

    ctx = AppContext(project, gpu=False, tick_ms=100000)
    page = InferencePage(ctx)
    page.show()
    assert page.model.count() == 1 and page.checkpoint.count() == 1
    assert page.window.value() == 64  # default from the run configuration
    page.refresh_results(select=out)
    assert page.images.count() >= 1 and page.arrays["pred"] is not None
    page._hover(3, 4)
    assert "predetta" in page.pixel.text() and "vera" in page.pixel.text()
    page.compare.setCurrentIndex(page.compare.findText("b"))
    assert page.arrays["other"] is not None
    assert "Differenze" in page.synced.labels[3].text()
    page.legend.boxes[0].setChecked(False)  # hiding a class redraws without errors
    page.layout_mode.setCurrentIndex(1)
    assert "Classi predette" in page.hist.plotItem.titleLabel.text
    ctx.shutdown()


def test_results_page_lists_and_compares_runs(qapp, project):
    import json

    from rsseg.gui.context import AppContext
    from rsseg.gui.pages.results import ResultsPage, aggregate

    scores = {"oa": 0.8, "mf1": 0.7, "miou": 0.6,
              "per_class": {n: {"iou": 0.5, "f1": 0.6, "precision": 0.7, "recall": 0.8,
                                "support": 10} for n in ("building", "tree")}}  # fmt: skip
    for seed, miou in ((0, 0.6), (1, 0.62)):
        run = project / "runs" / f"seg-exp-2026010{seed + 1}-000000"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text("model: {arch: resnet18, weights: imagenet}\n", "utf-8")
        rep = {"all_classes": {**scores, "miou": miou}, "without_clutter": scores, "tiles": ["t"],
               "confusion_matrix": [[5] * 6] * 6}  # fmt: skip
        (run / "report.json").write_text(json.dumps(rep), "utf-8")
    ctx = AppContext(project, gpu=False, tick_ms=100000)
    page = ResultsPage(ctx)
    page.show()
    assert page.table.rowCount() == 2
    page.table.selectAll()
    rows = aggregate(page._selected())
    assert rows[0]["n"] == 2 and rows[0]["miou"][0] == pytest.approx(0.61)
    assert page.compare_table.rowCount() == 1
    assert "0.6100 ± 0.0100" in page.compare_table.item(0, 4).text()
    ctx.shutdown()


def test_closing_with_active_jobs_asks_what_to_do(qapp, project, monkeypatch):
    from rsseg.gui import app as app_module
    from rsseg.gui.context import AppContext

    ctx = AppContext(project, gpu=False, tick_ms=100000)
    window = app_module.MainWindow(ctx)
    job = ctx.store.add("train", ["-c", "configs/tiny.yaml"], "T")  # queued = active

    def choose(choice):
        def exec_(self):
            self.choice = choice
            return choice != 0

        return exec_

    monkeypatch.setattr(app_module.CloseDialog, "exec", choose(0))  # Annulla
    assert not window.close() and job.state == "queued"
    monkeypatch.setattr(app_module.CloseDialog, "exec", choose(app_module.CloseDialog.PAUSE))
    monkeypatch.setattr(app_module.WaitDialog, "exec", lambda self: 1)
    assert window.close()
    assert job.state == "stopped"  # a queued job is dropped; a running one would pause
