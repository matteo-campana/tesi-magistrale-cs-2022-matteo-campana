# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job queue: progress, curves, GPU use and log of every command launched from the app."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui import actions, theme
from rsseg.gui.context import AppContext
from rsseg.gui.core.gpu import GPULog
from rsseg.gui.core.jobs import Job
from rsseg.gui.core.timing import fmt_duration, fmt_range
from rsseg.gui.widgets import (
    STATE_TONES,
    JobProgressWidget,
    action_button,
    plot_series,
    plot_widget,
    section_title,
    set_plot_title,
)

COLUMNS = ["Stato", "Job", "Avanzamento", "Trascorso", "Rimanente", "Creato"]


class JobsPage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.selected: str | None = None
        self._log_time = 0.0

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(section_title("Job"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)  # every column fits its values
        header.setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.horizontalHeaderItem(2).setToolTip("Avanzamento complessivo del job")
        self.table.horizontalHeaderItem(3).setToolTip("Tempo di lavoro, pause escluse")
        self.table.horizontalHeaderItem(4).setToolTip("Stima del tempo che manca alla fine")
        lv.addWidget(self.table)
        buttons = QHBoxLayout()
        self.remove_btn = QPushButton("Rimuovi dalla lista")
        self.remove_btn.setObjectName("quiet")
        self.remove_btn.setToolTip("Toglie il job dalla lista (i file del run restano)")
        self.remove_btn.clicked.connect(self._remove)
        self.kill_btn = QPushButton("Termina forzatamente")
        self.kill_btn.setObjectName("danger")
        self.kill_btn.setToolTip("Da usare solo se il job non risponde a Stop: termina il processo "
                                 "senza salvare nulla")  # fmt: skip
        self.kill_btn.clicked.connect(self._kill)
        self.folder_btn = action_button("Apri cartella", "folder")
        self.folder_btn.setToolTip("Apre la cartella del run (o del job)")
        self.folder_btn.clicked.connect(self._open_folder)
        buttons.addWidget(self.remove_btn)
        buttons.addWidget(self.kill_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.folder_btn)
        lv.addLayout(buttons)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.progress = JobProgressWidget()
        self.progress.pause_clicked.connect(lambda: self._request("pause"))
        self.progress.stop_clicked.connect(lambda: self._request("stop"))
        self.progress.resume_clicked.connect(self._resume)
        rv.addWidget(self.progress)
        self.tabs = QTabWidget()
        self.loss_plot = plot_widget("Loss", "loss", "epoca")
        self.metric_plot = plot_widget("mIoU", "", "epoca")
        curves = QWidget()
        cv = QHBoxLayout(curves)
        cv.addWidget(self.loss_plot)
        cv.addWidget(self.metric_plot)
        self.tabs.addTab(curves, "Curve")
        self.gpu_plot = plot_widget("Uso della GPU durante il job", "", "minuti")
        self.tabs.addTab(self.gpu_plot, "GPU")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.log, "Log")
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.tabs.addTab(self.result, "Risultato")
        self.command = QLabel()
        self.command.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command.setWordWrap(True)
        self.command.setObjectName("mono")
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.command)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([720, 680])
        layout = QVBoxLayout(self)
        layout.addWidget(split)
        ctx.jobs_changed.connect(self.refresh)
        self.refresh()

    # ---------------------------------------------------------------------- table
    def refresh(self) -> None:
        jobs = self.ctx.store.ordered()
        now = time.time()
        if self.table.rowCount() != len(jobs):
            self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            view = self.ctx.view(job, now)
            frac = view.total_fraction
            cells = [
                view.state_label,
                job.title,
                "" if frac is None else f"{frac:.0%}",
                fmt_duration(view.elapsed_s) if job.state != "queued" else "",
                fmt_range(view.remaining_s, view.remaining_max_s) if job.active else "",
                time.strftime("%d/%m %H:%M", time.localtime(job.created)),
            ]
            for col, text in enumerate(cells):
                item = self.table.item(row, col)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, col, item)
                item.setText(text)
                item.setData(Qt.UserRole, job.id)
            fg, bg = theme.tone_colors(theme.palette(), STATE_TONES.get(job.state, "neutral"))
            self.table.item(row, 0).setForeground(QColor(fg))
            self.table.item(row, 0).setBackground(QColor(bg))
            self.table.item(row, 1).setToolTip(" ".join(self.ctx.store.command_line(job)[2:]))
        if self.selected is None and jobs:
            self.select(jobs[0].id)
        self._show_selected()

    def retheme(self) -> None:
        self.refresh()

    def select(self, job_id: str) -> None:
        self.selected = job_id
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == job_id:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
        self._log_time = 0
        self._show_selected()

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
            self._log_time = 0
            self._show_selected()

    def current(self) -> Job | None:
        return self.ctx.store.jobs.get(self.selected) if self.selected else None

    # --------------------------------------------------------------------- detail
    def _show_selected(self) -> None:
        job = self.current()
        if job is None:
            self.progress.set_view(None)
            return
        view = self.ctx.view(job)
        self.progress.set_view(view, job, self.ctx.sample)
        self.kill_btn.setEnabled(job.state in ("running", "pausing", "stopping"))
        self.remove_btn.setEnabled(not job.active or job.state == "queued")
        self.command.setText("$ rsseg " + " ".join(self.ctx.store.command_line(job)[4:]))
        if not self.isVisible():
            return
        self._curves(job, view.history)
        now = time.time()
        if now - self._log_time > 2:
            self._log_time = now
            text = self.ctx.store.log_tail(job, 400)
            if text != self.log.toPlainText():
                self.log.setPlainText(text)
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self._gpu(job)
            result = self.ctx.store.result(job)
            self.result.setPlainText(json.dumps(result, indent=2, ensure_ascii=False)
                                     if result is not None else "")  # fmt: skip

    def _curves(self, job: Job, history: list[dict]) -> None:
        if not history and job.run_dir:
            from rsseg.gui.core.runs import load_run

            run = load_run(job.run_dir)
            history = (
                [{**h, "epoch": (h.get("epoch") or 0) + 1} for h in run.history()] if run else []
            )
        p = self.ctx.store.progress(job) or {}
        markers = [p["resumed_from_epoch"]] if p.get("resumed_from_epoch") else []
        x = [h.get("epoch") for h in history]
        loss = {k: (x, [h.get(k) for h in history]) for k in ("loss", "val_loss")
                if any(k in h for h in history)}  # fmt: skip
        plot_series(self.loss_plot, {("training" if k == "loss" else "validazione"): v
                                     for k, v in loss.items()}, markers)  # fmt: skip
        keys = [k for k in ("miou", "val_miou", "teacher_entropy") if any(k in h for h in history)]
        names = {"miou": "mIoU training", "val_miou": "mIoU validazione",
                 "teacher_entropy": "entropia teacher"}  # fmt: skip
        plot_series(self.metric_plot, {names[k]: (x, [h.get(k) for h in history]) for k in keys},
                    markers)  # fmt: skip
        set_plot_title(self.metric_plot,
                       "Entropia del teacher" if keys == ["teacher_entropy"] else "mIoU")  # fmt: skip

    def _gpu(self, job: Job) -> None:
        rows = GPULog(self.ctx.store.folder(job) / "gpu_log.csv").read()
        if not rows:
            self.gpu_plot.clear()
            return
        t0 = rows[0]["time"]
        x = [(r["time"] - t0) / 60 for r in rows]
        plot_series(self.gpu_plot, {
            "utilizzo %": (x, [r["util"] for r in rows]),
            "VRAM % ": (x, [100 * r["mem_used_gb"] / r["mem_total_gb"] if r["mem_used_gb"] and
                            r["mem_total_gb"] else None for r in rows]),
            "temperatura °C": (x, [r["temp_c"] for r in rows]),
        })  # fmt: skip

    # -------------------------------------------------------------------- actions
    def _request(self, action: str) -> None:
        job = self.current()
        if job is not None:
            self.ctx.store.request(job, action)
            self.ctx.tick()

    def _resume(self) -> None:
        job = self.current()
        if job is not None:
            actions.resume_job(self, self.ctx, job)

    def _remove(self) -> None:
        job = self.current()
        if job is None:
            return
        try:
            self.ctx.store.remove(job)
        except ValueError as e:
            QMessageBox.information(self, "Job attivo", str(e))
            return
        self.selected = None
        self.refresh()

    def _kill(self) -> None:
        job = self.current()
        if job is None:
            return
        if QMessageBox.question(self, "Termina forzatamente",
                                f"Terminare «{job.title}» senza salvare?") == QMessageBox.Yes:  # fmt: skip
            self.ctx.store.kill(job)

    def _open_folder(self) -> None:
        job = self.current()
        if job is not None:
            actions.open_folder(job.run_dir or self.ctx.store.folder(job))
