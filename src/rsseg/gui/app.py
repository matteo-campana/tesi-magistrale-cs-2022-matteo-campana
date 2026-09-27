# SPDX-License-Identifier: AGPL-3.0-or-later
"""Main window: navigation, status bar (GPU, jobs), close dialog, reattach at start."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui.context import AppContext
from rsseg.gui.core import gpu as gpucore
from rsseg.gui.core.timing import fmt_clock
from rsseg.gui.pages.campaigns import CampaignsPage
from rsseg.gui.pages.config import ConfigPage
from rsseg.gui.pages.dashboard import DashboardPage
from rsseg.gui.pages.data import DataPage
from rsseg.gui.pages.gpu import GPUPage
from rsseg.gui.pages.inference import InferencePage
from rsseg.gui.pages.jobs import JobsPage
from rsseg.gui.pages.results import ResultsPage
from rsseg.gui.widgets import GPUStatusWidget

PAGES = [
    ("dashboard", "Panoramica", "Stato della GPU, job attivi, run da riprendere, ultimi risultati"),
    (
        "config",
        "Configurazione",
        "Imposta un esperimento e avvia training, pre-training, benchmark",
    ),
    ("jobs", "Training e job", "Avanzamento, curve, uso della GPU e log di ogni job"),
    ("campaigns", "Campagne", "Gruppi di esperimenti riprendibili, con piano e tempi stimati"),
    ("data", "Dati", "Preparazione dei dataset e visualizzazione dei crop"),
    ("results", "Risultati", "Metriche per classe, confronti tra run, esportazione"),
    (
        "inference",
        "Inferenza",
        "Segmenta immagini con i modelli addestrati e confronta i risultati",
    ),
    ("gpu", "GPU", "Rilevamento, utilizzo, memoria, temperatura e processi della GPU"),
]

HELP_HTML = """
<h2>rsseg — guida rapida</h2>
<ol>
<li><b>Dati</b>: prepara Potsdam/Vaihingen (prima «Verifica», poi «Prepara») o genera tile sintetiche
per provare.</li>
<li><b>Configurazione</b>: scegli un esperimento, modifica i valori (passa il mouse su ogni campo per
il suo significato), verifica la memoria con «Entra in 8 GB?», poi «Avvia training».</li>
<li><b>Training e job</b>: segui fasi, barre, tempo trascorso e rimanente, curve e GPU. «Pausa» salva
un checkpoint e libera la GPU; «Riprendi» riparte esattamente da lì, anche dopo aver chiuso l'app.</li>
<li><b>Risultati</b> e <b>Inferenza</b>: analizza i modelli e segmenta nuove immagini.</li>
</ol>
<p>Chiudendo l'app con job attivi puoi metterli in pausa oppure lasciarli in esecuzione in
background: alla riapertura l'app si ricollega. Ogni job scrive anche un log e i file
<code>progress.json</code>/<code>control.json</code> in <code>.gui/jobs/</code>.</p>
"""


class CloseDialog(QDialog):
    """What to do with the running jobs when the window closes."""

    PAUSE, BACKGROUND = 1, 2

    def __init__(self, parent, jobs):
        super().__init__(parent)
        self.setWindowTitle("Job in esecuzione")
        v = QVBoxLayout(self)
        names = "".join(f"<li>{j.title}</li>" for j in jobs)
        v.addWidget(QLabel(f"<p>Ci sono job attivi:</p><ul>{names}</ul>"
                           "<p><b>Metti in pausa e chiudi</b>: i training salvano un checkpoint e "
                           "potrai riprenderli in un altro momento; gli altri job vengono "
                           "interrotti.<br><b>Lascia in background</b>: i job continuano senza "
                           "interfaccia; alla riapertura l'app si ricollega.</p>"))  # fmt: skip
        box = QDialogButtonBox()
        self.pause = box.addButton("Metti in pausa e chiudi", QDialogButtonBox.AcceptRole)
        self.background = box.addButton("Lascia in background", QDialogButtonBox.AcceptRole)
        box.addButton("Annulla", QDialogButtonBox.RejectRole)
        self.choice = 0
        self.pause.clicked.connect(lambda: self._done(self.PAUSE))
        self.background.clicked.connect(lambda: self._done(self.BACKGROUND))
        box.rejected.connect(self.reject)
        v.addWidget(box)

    def _done(self, choice: int) -> None:
        self.choice = choice
        self.accept()


class WaitDialog(QDialog):
    """Wait for paused jobs to write their checkpoint and exit."""

    def __init__(self, parent, ctx: AppContext, timeout_s: float = 600):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Pausa in corso")
        v = QVBoxLayout(self)
        self.label = QLabel()
        v.addWidget(self.label)
        bar = QProgressBar()
        bar.setRange(0, 0)
        v.addWidget(bar)
        box = QDialogButtonBox()
        force = box.addButton("Chiudi senza attendere", QDialogButtonBox.RejectRole)
        force.setToolTip("I job continuano a fermarsi da soli anche ad app chiusa")
        force.clicked.connect(self.reject)
        v.addWidget(box)
        self.t0 = time.time()
        self.timeout = timeout_s
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check)
        self.timer.start(500)
        self._check()

    def _check(self) -> None:
        self.ctx.tick()
        waiting = [j for j in self.ctx.store.jobs.values()
                   if j.state in ("running", "pausing", "stopping")]  # fmt: skip
        if not waiting or time.time() - self.t0 > self.timeout:
            self.accept()
            return
        self.label.setText("Attendo che i job salvino il checkpoint e si fermino "
                           "(durante la validazione si fermano a fine epoca):\n"
                           + "\n".join(f"• {j.title}" for j in waiting))  # fmt: skip


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"rsseg — segmentazione di immagini aeree · {ctx.root.name}")
        self.resize(1400, 900)

        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.setIconSize(QSize(1, 1))
        self.nav.setStyleSheet(
            "QListWidget { font-size: 14px; } QListWidget::item { padding: 8px; }"
        )
        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        factories = {"dashboard": DashboardPage, "config": ConfigPage, "jobs": JobsPage,
                     "campaigns": CampaignsPage, "data": DataPage, "results": ResultsPage,
                     "inference": InferencePage, "gpu": GPUPage}  # fmt: skip
        for key, title, tooltip in PAGES:
            page = factories[key](ctx)
            self.pages[key] = page
            self.stack.addWidget(page)
            item = QListWidgetItem(title)
            item.setToolTip(tooltip)
            item.setData(Qt.UserRole, key)
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.nav)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.gpu_status = GPUStatusWidget()
        self.gpu_status.clicked.connect(lambda: self.show_page("gpu"))
        self.jobs_status = QLabel()
        self.jobs_status.setToolTip("Job attivi e ora prevista di fine dell'ultimo job in coda")
        self.statusBar().addPermanentWidget(self.jobs_status)
        self.statusBar().addPermanentWidget(self.gpu_status)
        ctx.status_message.connect(lambda m: self.statusBar().showMessage(m, 8000))
        ctx.gpu_sampled.connect(self._gpu)
        ctx.torch_checked.connect(lambda _: self._gpu(ctx.samples))
        ctx.jobs_changed.connect(self._jobs)
        ctx.job_finished.connect(self._finished)

        self.pages["dashboard"].open_page.connect(self.show_page)
        self.pages["config"].job_started.connect(self._open_job)
        self._menus()
        QTimer.singleShot(300, self._startup_notice)

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = QAction("Esci", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view = self.menuBar().addMenu("&Vai")
        for i, (key, title, _) in enumerate(PAGES):
            action = QAction(title, self)
            action.setShortcut(f"Ctrl+{i + 1}")
            action.triggered.connect(lambda _=False, k=key: self.show_page(k))
            view.addAction(action)
        help_menu = self.menuBar().addMenu("&Aiuto")
        guide = QAction("Guida rapida", self)
        guide.setShortcut(QKeySequence.HelpContents)
        guide.triggered.connect(self._help)
        help_menu.addAction(guide)

    def show_page(self, key: str) -> None:
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.UserRole) == key:
                self.nav.setCurrentRow(i)

    def _open_job(self, job_id: str) -> None:
        self.show_page("jobs")
        self.pages["jobs"].select(job_id)

    def _help(self) -> None:
        d = QDialog(self)
        d.setWindowTitle("Guida rapida")
        v = QVBoxLayout(d)
        t = QTextBrowser()
        t.setHtml(HELP_HTML)
        v.addWidget(t)
        d.resize(640, 420)
        d.exec()

    # ------------------------------------------------------------------ status bar
    def _gpu(self, samples) -> None:
        level, message = gpucore.status(samples, self.ctx.torch_info)
        sample = samples[0] if samples else None
        alerts = gpucore.alerts(sample) if sample else []
        if alerts:
            message += "\n⚠ " + "\n⚠ ".join(alerts)
        self.gpu_status.update_status(level, message, sample)

    def _jobs(self) -> None:
        active = [j for j in self.ctx.store.jobs.values() if j.active]
        if not active:
            self.jobs_status.setText("Nessun job attivo")
            return
        ends = [v.finish_at for v in (self.ctx.view(j) for j in active) if v.finish_at]
        text = f"{len(active)} job attivi"
        if ends:
            text += f" · fine prevista {fmt_clock(max(ends))}"
        self.jobs_status.setText(text)

    def _finished(self, job) -> None:
        labels = {"done": "completato", "failed": "fallito", "paused": "in pausa",
                  "stopped": "interrotto", "interrupted": "terminato in modo anomalo"}  # fmt: skip
        self.statusBar().showMessage(f"«{job.title}»: {labels.get(job.state, job.state)}", 15000)
        QApplication.alert(self)

    def _startup_notice(self) -> None:
        """After a restart: say which jobs are still running and which can be resumed."""
        running = [j for j in self.ctx.store.jobs.values() if j.state in ("running", "pausing")]
        broken = [j for j in self.ctx.store.jobs.values()
                  if j.state == "interrupted" and j.finished and j.finished > time.time() - 7 * 86400
                  and not j.meta.get("notified")]  # fmt: skip
        if not running and not broken:
            return
        lines = []
        if running:
            lines.append("Job ancora in esecuzione (l'app si è ricollegata):")
            lines += [f"• {j.title}" for j in running]
        if broken:
            lines.append("Job terminati in modo anomalo:")
            lines += [f"• {j.title}" + (" — riprendibile dall'ultimo checkpoint" if j.resumable
                                        else "") for j in broken]  # fmt: skip
            for j in broken:
                j.meta["notified"] = True
            self.ctx.store.save()
        box = QMessageBox(QMessageBox.Information, "Job trovati all'avvio", "\n".join(lines),
                          parent=self)  # fmt: skip
        box.setModal(False)  # the jobs keep being followed while the notice is open
        box.show()
        self.startup_box = box

    # ------------------------------------------------------------------------ close
    def closeEvent(self, event):  # noqa: N802 (Qt API)
        active = [j for j in self.ctx.store.jobs.values() if j.active]
        if active:
            dialog = CloseDialog(self, active)
            if not dialog.exec():
                event.ignore()
                return
            if dialog.choice == CloseDialog.PAUSE:
                for job in active:
                    self.ctx.store.request(job, "pause")
                WaitDialog(self, self.ctx).exec()
        self.ctx.shutdown()
        event.accept()


def run_app(root: str | Path | None = None, argv: list[str] | None = None) -> int:
    import sys

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("rsseg")
    app.setStyle("Fusion")
    ctx = AppContext(root or Path.cwd())
    ctx.store.refresh()
    window = MainWindow(ctx)
    window.show()
    return app.exec()
