# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dashboard: GPU, active jobs, runs to resume, prepared data, latest results."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui import actions, theme
from rsseg.gui.context import AppContext
from rsseg.gui.core import gpu as gpucore
from rsseg.gui.core.runs import find_runs
from rsseg.gui.core.timing import fmt_duration
from rsseg.gui.widgets import LEVEL_TONES, JobProgressWidget, action_button, gpu_line, section_title
from rsseg.runstate import find_resumable, read_state


def prepared_datasets(root: Path) -> list[tuple[str, dict[str, int]]]:
    """Data roots under ``root/data`` (and their crop counts per split)."""
    out = []
    base = root / "data"
    if not base.exists():
        return out
    for crops in sorted(base.rglob("crops.csv")):
        counts: dict[str, int] = {}
        try:
            with crops.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    counts[row["split"]] = counts.get(row["split"], 0) + 1
        except (OSError, KeyError):
            continue
        out.append((crops.parent.relative_to(root).as_posix(), counts))
    return out


class DashboardPage(QWidget):
    open_page = Signal(str)
    open_job = Signal(str)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.setSpacing(theme.Space.MD)
        title = section_title(f"Progetto: {ctx.root.name}")
        title.setToolTip(str(ctx.root))
        v.addWidget(title)

        self.gpu_box = QGroupBox("GPU")
        gv = QHBoxLayout(self.gpu_box)
        self.gpu_label = QLabel("Rilevamento…")
        self.gpu_label.setWordWrap(True)
        gv.addWidget(self.gpu_label, 1)
        go = QPushButton("Dettagli GPU")
        go.clicked.connect(lambda: self.open_page.emit("gpu"))
        gv.addWidget(go)
        v.addWidget(self.gpu_box)

        self.active_box = QGroupBox("Job attivi")
        self.active_layout = QVBoxLayout(self.active_box)
        self.active_widgets: dict[str, JobProgressWidget] = {}
        self.no_active = QLabel("<i>Nessun job in esecuzione.</i>")
        self.active_layout.addWidget(self.no_active)
        v.addWidget(self.active_box)

        self.resume_box = QGroupBox("Da riprendere")
        self.resume_box.setToolTip("Run in pausa o terminati in modo anomalo con un checkpoint di "
                                   "ripresa: ripartono esattamente da dove si erano fermati.")  # fmt: skip
        self.resume_layout = QGridLayout(self.resume_box)
        self.resume_layout.setColumnStretch(0, 1)
        self.resume_layout.setVerticalSpacing(theme.Space.LG)
        v.addWidget(self.resume_box)

        row = QHBoxLayout()
        self.data_box = QGroupBox("Dati preparati")
        self.data_label = QLabel()
        self.data_label.setWordWrap(True)
        QVBoxLayout(self.data_box).addWidget(self.data_label)
        self.results_box = QGroupBox("Ultimi risultati")
        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        QVBoxLayout(self.results_box).addWidget(self.results_label)
        row.addWidget(self.data_box)
        row.addWidget(self.results_box)
        v.addLayout(row)
        quick = QHBoxLayout()
        for text, page in (("Nuovo training…", "config"), ("Prepara dati…", "data"),
                           ("Campagne…", "campaigns"), ("Inferenza…", "inference")):  # fmt: skip
            b = QPushButton(text)
            if page == "config":
                b.setObjectName("primary")
            b.clicked.connect(lambda _=False, p=page: self.open_page.emit(p))
            quick.addWidget(b)
        quick.addStretch(1)
        v.addLayout(quick)
        v.addStretch(1)

        ctx.jobs_changed.connect(self.refresh_jobs)
        ctx.gpu_sampled.connect(lambda _: self.refresh_gpu())
        ctx.torch_checked.connect(lambda _: self.refresh_gpu())
        ctx.job_finished.connect(lambda _: self.refresh_slow())
        self._last_slow = 0.0
        self.refresh_slow()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.refresh_slow()

    def refresh_gpu(self) -> None:
        level, message = gpucore.status(self.ctx.samples, self.ctx.torch_info)
        line = gpu_line(self.ctx.sample) if self.ctx.sample else ""
        dot = f"<span style='color:{theme.fill(LEVEL_TONES[level])}; font-size:16px'>●</span>"
        self.gpu_label.setText(f"{dot} {message.splitlines()[0]}<br>"
                               f"{theme.span(line, 'neutral')}")  # fmt: skip

    def retheme(self) -> None:
        self.refresh_gpu()
        self.refresh_slow()

    def refresh_jobs(self) -> None:
        active = [j for j in self.ctx.store.ordered() if j.active]
        for jid in list(self.active_widgets):
            if jid not in {j.id for j in active}:
                w = self.active_widgets.pop(jid)
                w.setParent(None)  # leaves the layout now, deleted later
                w.deleteLater()
        for job in active:
            w = self.active_widgets.get(job.id)
            if w is None:
                w = JobProgressWidget()
                w.pause_clicked.connect(
                    lambda j=job: (self.ctx.store.request(j, "pause"), self.ctx.tick())
                )
                w.stop_clicked.connect(
                    lambda j=job: (self.ctx.store.request(j, "stop"), self.ctx.tick())
                )
                self.active_layout.addWidget(w)
                self.active_widgets[job.id] = w
            w.set_view(self.ctx.view(job), job, self.ctx.sample)
        self.no_active.setVisible(not active)
        if time.time() - self._last_slow > 30:
            self.refresh_slow()

    def refresh_slow(self) -> None:
        """Parts that scan the disk (every 30 s, and when a job ends)."""
        self._last_slow = time.time()
        while self.resume_layout.count():
            widget = self.resume_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        rows = 0
        known = set()
        for job in self.ctx.store.ordered():
            if job.resumable:
                known.add(str(Path(job.run_dir).resolve()) if job.run_dir else job.id)
                self._resume_row(rows, job.title, self._job_detail(job),
                                 lambda _=False, j=job: actions.resume_job(self, self.ctx, j))  # fmt: skip
                rows += 1
        for run in find_resumable(self.ctx.root / "runs"):
            if str(run.resolve()) in known:
                continue
            s = read_state(run) or {}
            detail = self._state_detail(s)
            self._resume_row(rows, run.name, detail,
                             lambda _=False, r=run: actions.resume_run(self, self.ctx, r))  # fmt: skip
            rows += 1
        if not rows:
            self.resume_layout.addWidget(QLabel("<i>Nessun run in pausa.</i>"), 0, 0)

        data = prepared_datasets(self.ctx.root)
        self.data_label.setText("<br>".join(
            f"<b>{name}</b>: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + " crop"
            for name, counts in data) or "<i>Nessun dato preparato in data/ — vai a «Dati».</i>")  # fmt: skip
        runs = [
            r for r in find_runs([self.ctx.root / "runs"], ("seg",)) if r.test_miou is not None
        ][:6]
        self.results_label.setText("<br>".join(
            f"<b>{r.name}</b> ({r.arch}, {r.pretrain}) — mIoU test {r.test_miou:.4f}" for r in runs)
            or "<i>Nessun run valutato.</i>")  # fmt: skip

    def _job_detail(self, job) -> str:
        if job.command == "campaign":
            return "campagna: riparte dai passi non completati"
        return self._state_detail(read_state(job.run_dir) or {}) if job.run_dir else ""

    @staticmethod
    def _state_detail(s: dict) -> str:
        if s.get("phase") == "evaluation":
            where = "valutazione finale"
        else:
            where = (
                f"epoca {s.get('epoch', 0) + 1}/{s.get('epochs', '?')}, batch {s.get('batch', 0)}"
            )
        when = time.strftime("%d/%m %H:%M", time.localtime(s["saved"])) if s.get("saved") else "?"
        reason = "pausa" if s.get("reason") == "pause" else "checkpoint automatico"
        return f"{where} · {reason} del {when} · lavoro svolto {fmt_duration(s.get('active_s'))}"

    def _resume_row(self, row: int, title: str, detail: str, callback) -> None:
        label = QLabel(f"<b>{title}</b><br>{theme.span(detail, 'neutral')}")
        label.setTextFormat(Qt.RichText)
        button = action_button("Riprendi", "play")
        button.clicked.connect(callback)
        self.resume_layout.addWidget(label, row, 0)
        self.resume_layout.addWidget(button, row, 1)
