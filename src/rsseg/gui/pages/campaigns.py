# SPDX-License-Identifier: AGPL-3.0-or-later
"""Campaigns: plan (dry run) with disk and time estimates, run/resume, pause, step states,
final report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui import actions
from rsseg.gui.context import AppContext
from rsseg.gui.core.jobs import Job
from rsseg.gui.core.timing import fmt_duration
from rsseg.gui.widgets import JobProgressWidget, action_button, section_title, tip

STEP_STATES = {"done": "completato", "failed": "fallito", "blocked": "bloccato",
               "running": "in esecuzione", "paused": "in pausa", "stopped": "interrotto",
               "pending": "da eseguire"}  # fmt: skip


def list_campaigns(root: Path) -> list[str]:
    base = root / "configs" / "campaigns"
    return (
        sorted(p.relative_to(root).as_posix() for p in base.glob("*.yaml")) if base.exists() else []
    )


class CampaignsPage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.job_id: str | None = None
        self._last = 0.0

        form_box = QGroupBox("Campagna")
        f = QFormLayout(form_box)
        self.file = QComboBox()
        self.file.addItems(list_campaigns(ctx.root))
        self.file.currentTextChanged.connect(self._describe)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.quick = tip(QCheckBox("Prova rapida (--quick)"), "options", "campaign.quick")
        self.only = tip(QLineEdit(), "options", "campaign.only")
        self.only.setPlaceholderText("tutti (es. maresunet_* unet_*)")
        self.seeds = tip(QLineEdit(), "options", "campaign.seeds")
        self.seeds.setPlaceholderText("quelli della campagna (es. 0,1,2)")
        self.stop_on_error = tip(
            QCheckBox("Ferma al primo errore"), "options", "campaign.stop_on_error"
        )
        self.clean_failed = tip(QCheckBox("Pulisci i tentativi falliti"), "options",
                                "campaign.clean_failed")  # fmt: skip
        self.ignore_disk = QCheckBox("Ignora il controllo dello spazio su disco")
        self.ignore_disk.setToolTip("Avvia anche se la stima dello spazio necessario supera quello "
                                    "libero")  # fmt: skip
        f.addRow("File:", self.file)
        f.addRow(self.description)
        f.addRow("Solo:", self.only)
        f.addRow("Seed:", self.seeds)
        for w in (self.quick, self.stop_on_error, self.clean_failed, self.ignore_disk):
            f.addRow(w)
        row = QHBoxLayout()
        plan = tip(QPushButton("Piano (dry-run)"), "options", "campaign.dry_run")
        plan.clicked.connect(lambda: self._run(dry=True))
        start = action_button("Avvia / riprendi", "play", "primary")
        start.setToolTip("Esegue i passi non ancora completati; un passo in pausa riparte dal suo "
                         "checkpoint. I passi girano uno alla volta sulla GPU.")  # fmt: skip
        start.clicked.connect(lambda: self._run(dry=False))
        row.addWidget(plan)
        row.addWidget(start)
        f.addRow(row)

        self.progress = JobProgressWidget()
        self.progress.pause_clicked.connect(lambda: self._request("pause"))
        self.progress.stop_clicked.connect(lambda: self._request("stop"))
        self.progress.resume_clicked.connect(self._resume)

        self.totals = QLabel()
        self.plan = QTableWidget(0, 4)
        self.plan.setHorizontalHeaderLabels(["Tipo", "Passo", "Stato", "Stima"])
        self.plan.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.plan.setEditTriggers(QTableWidget.NoEditTriggers)
        self.plan.horizontalHeaderItem(3).setToolTip("Durata stimata del passo (tempi misurati "
                                                     "sulla RTX 4060, ricalibrati sui passi finiti)")  # fmt: skip
        self.status = QTableWidget(0, 4)
        self.status.setHorizontalHeaderLabels(["Passo", "Tipo", "Stato", "Durata"])
        self.status.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.status.setEditTriggers(QTableWidget.NoEditTriggers)
        self.report = QTextBrowser()
        tabs = QTabWidget()
        plan_w = QWidget()
        pv = QVBoxLayout(plan_w)
        pv.addWidget(self.totals)
        pv.addWidget(self.plan)
        tabs.addTab(plan_w, "Piano")
        tabs.addTab(self.status, "Stato dei passi")
        tabs.addTab(self.report, "Report")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(form_box)
        lv.addStretch(1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.progress)
        rv.addWidget(tabs, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 800])
        v = QVBoxLayout(self)
        v.addWidget(section_title("Campagne di esperimenti"))
        v.addWidget(split, 1)
        ctx.jobs_changed.connect(self.refresh)
        ctx.job_finished.connect(self._on_finished)
        self._describe(self.file.currentText())

    def _describe(self, path: str) -> None:
        try:
            data = yaml.safe_load((self.ctx.root / path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        n, inc = len(data.get("runs") or []), data.get("include") or []
        what = f"{n} run definiti" + (f", include {', '.join(inc)}" if inc else "")
        self.description.setText(f"<i>{data.get('description', '')}</i><br>{what} · seed "
                                 f"{data.get('seeds', [0])}")  # fmt: skip
        self._load_status()

    def _args(self, dry: bool) -> list[str]:
        args = [self.file.currentText()]
        if dry:
            args.append("--dry-run")
        if self.quick.isChecked():
            args.append("--quick")
        if self.only.text().strip():
            args += ["--only", *self.only.text().split()]
        if self.seeds.text().strip():
            args += ["--seeds", self.seeds.text().strip()]
        for box, flag in ((self.stop_on_error, "--stop-on-error"), (self.clean_failed, "--clean-failed"),
                          (self.ignore_disk, "--ignore-disk")):  # fmt: skip
            if box.isChecked() and not dry:
                args.append(flag)
        return args

    def _run(self, dry: bool) -> None:
        name = Path(self.file.currentText()).stem
        if not dry and not actions.confirm_gpu_start(self, self.ctx, None, f"la campagna {name}"):
            return
        job = self.ctx.submit("campaign", self._args(dry),
                              f"{'Piano' if dry else 'Campagna'} {name}"
                              + (" (rapida)" if self.quick.isChecked() else ""),
                              meta={"show": "plan"} if dry else {"campaign": name})  # fmt: skip
        if not dry:
            self.job_id = job.id

    def _job(self) -> Job | None:
        if self.job_id and self.job_id in self.ctx.store.jobs:
            return self.ctx.store.jobs[self.job_id]
        name = Path(self.file.currentText()).stem
        for job in self.ctx.store.ordered():
            if job.command == "campaign" and job.meta.get("campaign") == name:
                self.job_id = job.id
                return job
        return None

    def _request(self, action: str) -> None:
        job = self._job()
        if job:
            self.ctx.store.request(job, action)
            self.ctx.tick()

    def _resume(self) -> None:
        job = self._job()
        if job:
            actions.resume_job(self, self.ctx, job)

    def refresh(self) -> None:
        job = self._job()
        self.progress.set_view(self.ctx.view(job) if job else None, job, self.ctx.sample)
        if job and job.active and time.time() - self._last > 3:
            self._load_status()

    def _on_finished(self, job: Job) -> None:
        if job.command != "campaign":
            return
        result = self.ctx.store.result(job)
        if job.meta.get("show") == "plan" and isinstance(result, dict) and "steps" in result:
            self.show_plan(result)
        self._load_status()

    def show_plan(self, plan: dict) -> None:
        rows = ([("pool", p["pool"], p["action"], None) for p in plan.get("pools", [])]
                + [("dati", d["root"], d["action"], None) for d in plan.get("data", [])]
                + [(s["kind"], s["id"], s["state"], s.get("estimate_s")) for s in plan["steps"]])  # fmt: skip
        self.plan.setRowCount(len(rows))
        for r, (kind, name, state, est) in enumerate(rows):
            for c, text in enumerate([kind, name, STEP_STATES.get(state, state),
                                      "" if state == "done" else fmt_duration(est)]):  # fmt: skip
                self.plan.setItem(r, c, QTableWidgetItem(text))
        self.totals.setText(
            f"Da eseguire <b>{plan['pending']}</b>, completati {plan['done']} · disco necessario "
            f"~{plan['disk_needed_gb']} GB (liberi {plan['disk_free_gb']} GB) · tempo stimato "
            f"<b>{plan['eta']}</b>")  # fmt: skip

    def _campaign_dir(self) -> Path:
        name = Path(self.file.currentText()).stem
        try:
            data = yaml.safe_load(
                (self.ctx.root / self.file.currentText()).read_text(encoding="utf-8")
            )
            name = data.get("name", name)
        except (OSError, yaml.YAMLError, AttributeError):
            pass
        root = "runs/campaigns-quick" if self.quick.isChecked() else "runs/campaigns"
        return self.ctx.root / root / name

    def _load_status(self) -> None:
        self._last = time.time()
        d = self._campaign_dir()
        try:
            status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {"steps": {}}
        steps = status.get("steps", {})
        self.status.setRowCount(len(steps))
        for r, (sid, s) in enumerate(steps.items()):
            cells = [sid, s.get("kind", ""), STEP_STATES.get(s.get("state"), s.get("state", "")),
                     fmt_duration(s.get("duration_s")) if s.get("duration_s") else ""]  # fmt: skip
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if s.get("reason"):
                    item.setToolTip(s["reason"])
                self.status.setItem(r, c, item)
        report = d / "report.md"
        self.report.setMarkdown(report.read_text(encoding="utf-8") if report.exists()
                                else "_Il report compare alla fine della campagna._")  # fmt: skip
