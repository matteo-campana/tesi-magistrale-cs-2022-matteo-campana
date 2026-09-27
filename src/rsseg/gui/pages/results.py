# SPDX-License-Identifier: AGPL-3.0-or-later
"""Results: every run with its scores per class, confusion matrix, curves, GPU use and
predictions; comparison of runs (mean ± std over seeds) with export for the thesis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui.context import AppContext
from rsseg.gui.core import help as helptext
from rsseg.gui.core.gpu import GPULog
from rsseg.gui.core.runs import RunInfo, find_runs
from rsseg.gui.widgets import (
    CLASS_LABELS_IT,
    ImageView,
    bar_chart,
    class_colors,
    load_array,
    plot_series,
    plot_widget,
    section_title,
)

RUN_COLUMNS = ["Nome", "Tipo", "Encoder", "Pre-training", "Stato", "val mIoU", "test mIoU",
               "test mIoU eroded", "Data"]  # fmt: skip
STATES = {"done": "completato", "paused": "in pausa", "incomplete": "incompleto"}


def _f(v) -> str:
    return "" if v is None else f"{v:.4f}"


def aggregate(runs: list[RunInfo]) -> list[dict]:
    """Mean ± std of the test scores of runs grouped by name (e.g. several seeds)."""
    groups: dict[str, list[RunInfo]] = {}
    for r in runs:
        if r.report:
            groups.setdefault(r.name, []).append(r)
    out = []
    for name, rs in sorted(groups.items()):
        row = {"name": name, "n": len(rs)}
        for key, getter in (
            ("oa", lambda r: r.report["all_classes"]["oa"]),
            ("mf1", lambda r: r.report["all_classes"]["mf1"]),
            ("miou", lambda r: r.report["all_classes"]["miou"]),
            ("miou_no_clutter", lambda r: r.report["without_clutter"]["miou"]),
            ("miou_eroded", lambda r: r.report["eroded"]["all_classes"]["miou"]),
        ):
            try:
                vals = np.array([getter(r) for r in rs], dtype=float)
            except (KeyError, TypeError):
                row[key] = None
                continue
            row[key] = (float(vals.mean()), float(vals.std()))
        out.append(row)
    return out


def _ms(v) -> str:
    if v is None:
        return "—"
    return f"{v[0]:.4f} ± {v[1]:.4f}"


class ResultsPage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.runs: list[RunInfo] = []

        self.table = QTableWidget(0, len(RUN_COLUMNS))
        self.table.setHorizontalHeaderLabels(RUN_COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col, key in ((5, "val_miou"), (6, "miou"), (7, "eroded")):
            self.table.horizontalHeaderItem(col).setToolTip(helptext.tooltip("metrics", key))
        self.table.itemSelectionChanged.connect(self._on_select)
        refresh = QPushButton("Aggiorna")
        refresh.clicked.connect(self.refresh)

        self.tabs = QTabWidget()
        # per class
        per_class = QWidget()
        pv = QVBoxLayout(per_class)
        self.eroded = QCheckBox("Etichette erose")
        self.eroded.setToolTip(helptext.tooltip("metrics", "eroded"))
        self.eroded.toggled.connect(lambda _: self._show_current())
        self.summary = QLabel()
        self.class_table = QTableWidget(0, 6)
        self.class_table.setHorizontalHeaderLabels(["Classe", "IoU", "F1", "Precision", "Recall",
                                                    "Pixel"])  # fmt: skip
        self.class_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.class_plot = plot_widget("IoU per classe", "IoU")
        pv.addWidget(self.eroded)
        pv.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(self.class_table)
        row.addWidget(self.class_plot)
        pv.addLayout(row)
        self.tabs.addTab(per_class, "Per classe")
        self.confusion = QTableWidget(6, 6)
        self.confusion.setEditTriggers(QTableWidget.NoEditTriggers)
        self.confusion.setToolTip("Righe: classe vera; colonne: classe predetta. Percentuali per riga "
                                  "(la diagonale è il recall).")  # fmt: skip
        self.tabs.addTab(self.confusion, "Matrice di confusione")
        curves = QWidget()
        cv = QHBoxLayout(curves)
        self.loss_plot = plot_widget("Loss", "", "epoca")
        self.miou_plot = plot_widget("mIoU", "", "epoca")
        cv.addWidget(self.loss_plot)
        cv.addWidget(self.miou_plot)
        self.tabs.addTab(curves, "Curve")
        self.gpu_plot = plot_widget("GPU durante il run", "", "minuti")
        self.tabs.addTab(self.gpu_plot, "GPU")
        preds = QWidget()
        prv = QHBoxLayout(preds)
        self.pred_list = QListWidget()
        self.pred_list.setMaximumWidth(260)
        self.pred_list.currentTextChanged.connect(self._show_prediction)
        self.pred_view = ImageView()
        prv.addWidget(self.pred_list)
        prv.addWidget(self.pred_view, 1)
        self.tabs.addTab(preds, "Predizioni")
        self.config_text = QPlainTextEdit()
        self.config_text.setReadOnly(True)
        self.tabs.addTab(self.config_text, "Configurazione")
        # comparison
        compare = QWidget()
        cmv = QVBoxLayout(compare)
        cmv.addWidget(QLabel("Seleziona più run nella tabella (Ctrl/Maiusc + clic): i run con lo "
                             "stesso nome (seed diversi) sono raggruppati in media ± deviazione "
                             "standard."))  # fmt: skip
        self.compare_table = QTableWidget(0, 7)
        self.compare_table.setHorizontalHeaderLabels(["Run", "n", "OA", "mF1", "mIoU",
                                                      "mIoU senza clutter", "mIoU eroded"])  # fmt: skip
        self.compare_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        for col, key in ((2, "oa"), (3, "mf1"), (4, "miou"), (5, "without_clutter"), (6, "eroded")):
            self.compare_table.horizontalHeaderItem(col).setToolTip(
                helptext.tooltip("metrics", key)
            )
        self.compare_plot = plot_widget("mIoU di test", "mIoU")
        export = QHBoxLayout()
        md = QPushButton("Esporta Markdown…")
        md.clicked.connect(lambda: self._export("md"))
        cs = QPushButton("Esporta CSV…")
        cs.clicked.connect(lambda: self._export("csv"))
        png = QPushButton("Salva grafico PNG…")
        png.clicked.connect(self._save_plot)
        for b in (md, cs, png):
            export.addWidget(b)
        cmv.addWidget(self.compare_table)
        cmv.addWidget(self.compare_plot)
        cmv.addLayout(export)
        self.tabs.addTab(compare, "Confronto")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.table)
        lv.addWidget(refresh)
        split = QSplitter(Qt.Vertical)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([300, 500])
        v = QVBoxLayout(self)
        v.addWidget(section_title("Risultati"))
        v.addWidget(split, 1)
        ctx.job_finished.connect(lambda _: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        self.runs = find_runs([self.ctx.root / "runs"])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.runs))
        for r, run in enumerate(self.runs):
            eroded = (run.report or {}).get("eroded", {}).get("all_classes", {}).get("miou")
            cells = [run.name, run.kind, run.arch, run.pretrain if run.kind != "dino" else "—",
                     STATES[run.state], _f(run.best_val_miou), _f(run.test_miou), _f(eroded),
                     run.stamp]  # fmt: skip
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, r)
                item.setToolTip(str(run.path))
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)

    def _selected(self) -> list[RunInfo]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.runs[self.table.item(r, 0).data(Qt.UserRole)] for r in rows]

    def _on_select(self) -> None:
        sel = self._selected()
        self.current = sel[0] if sel else None
        self._show_current()
        self._compare(sel)

    # -------------------------------------------------------------------- one run
    def _show_current(self) -> None:
        run = getattr(self, "current", None)
        if run is None:
            return
        report = run.report or {}
        section = report.get("eroded", {}) if self.eroded.isChecked() else report
        allc = section.get("all_classes")
        if allc:
            nc = section.get("without_clutter", {})
            self.summary.setText(f"OA <b>{allc['oa']:.4f}</b> · mF1 <b>{allc['mf1']:.4f}</b> · "
                                 f"mIoU <b>{allc['miou']:.4f}</b> · mIoU senza clutter "
                                 f"<b>{nc.get('miou', float('nan')):.4f}</b> · tile: "
                                 f"{', '.join(report.get('tiles', []))}")  # fmt: skip
            classes = list(allc["per_class"].items())
            self.class_table.setRowCount(len(classes))
            for r, (name, c) in enumerate(classes):
                cells = [CLASS_LABELS_IT.get(name, name), _f(c["iou"]), _f(c["f1"]),
                         _f(c["precision"]), _f(c["recall"]), f"{c['support']:,}".replace(",", ".")]  # fmt: skip
                for col, text in enumerate(cells):
                    self.class_table.setItem(r, col, QTableWidgetItem(text))

            colors = class_colors()[: len(classes)]
            bar_chart(self.class_plot, [CLASS_LABELS_IT.get(n, n).split()[0] for n, _ in classes],
                      [c["iou"] for _, c in classes], colors)  # fmt: skip
            self._confusion(section.get("confusion_matrix"))
        else:
            self.summary.setText("<i>Nessuna valutazione di test per questo run.</i>")
            self.class_table.setRowCount(0)
            self.class_plot.clear()
            self._confusion(None)
        hist = run.history()
        x = [(h.get("epoch") or 0) + 1 for h in hist]
        plot_series(self.loss_plot, {k: (x, [h.get(k) for h in hist]) for k in ("loss", "val_loss")
                                     if any(h.get(k) is not None for h in hist)})  # fmt: skip
        plot_series(self.miou_plot, {k: (x, [h.get(k) for h in hist])
                                     for k in ("miou", "val_miou", "teacher_entropy")
                                     if any(h.get(k) is not None for h in hist)})  # fmt: skip
        rows = GPULog(run.path / "gpu_log.csv").read()
        if rows:
            t0 = rows[0]["time"]
            xs = [(r["time"] - t0) / 60 for r in rows]
            plot_series(self.gpu_plot, {"utilizzo %": (xs, [r["util"] for r in rows]),
                                        "VRAM GB": (xs, [r["mem_used_gb"] for r in rows]),
                                        "°C": (xs, [r["temp_c"] for r in rows])})  # fmt: skip
        else:
            self.gpu_plot.clear()
        self.pred_list.clear()
        pred_dir = run.path / "predictions"
        if pred_dir.exists():
            self.pred_list.addItems(sorted(p.name for p in pred_dir.glob("*.png")))
        try:
            self.config_text.setPlainText((run.path / "config.yaml").read_text(encoding="utf-8"))
        except OSError:
            self.config_text.setPlainText("")

    def _confusion(self, cm) -> None:
        from rsseg.data.classes import CLASS_NAMES

        labels = [CLASS_LABELS_IT[n].split()[0] for n in CLASS_NAMES]
        self.confusion.setHorizontalHeaderLabels(labels)
        self.confusion.setVerticalHeaderLabels(labels)
        if cm is None:
            self.confusion.clearContents()
            return
        cm = np.asarray(cm, dtype=float)
        rows = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                item = QTableWidgetItem(f"{rows[i, j]:.1%}")
                shade = int(255 - 180 * rows[i, j])
                item.setBackground(
                    QColor(shade, shade, 255) if i == j else QColor(255, shade, shade)
                )
                item.setToolTip(f"{int(cm[i, j]):,} pixel".replace(",", "."))
                self.confusion.setItem(i, j, item)

    def _show_prediction(self, name: str) -> None:
        run = getattr(self, "current", None)
        if run is None or not name:
            return
        self.pred_view.set_image(load_array(run.path / "predictions" / name, "RGB"), fit=True)

    # ------------------------------------------------------------------ comparison
    def _compare(self, runs: list[RunInfo]) -> None:
        self.rows = aggregate(runs)
        self.compare_table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            cells = [row["name"], str(row["n"]), _ms(row["oa"]), _ms(row["mf1"]), _ms(row["miou"]),
                     _ms(row["miou_no_clutter"]), _ms(row["miou_eroded"])]  # fmt: skip
            for c, text in enumerate(cells):
                self.compare_table.setItem(r, c, QTableWidgetItem(text))
        bar_chart(self.compare_plot, [row["name"] for row in self.rows],
                  [row["miou"][0] if row["miou"] else 0 for row in self.rows])  # fmt: skip

    def _export(self, kind: str) -> None:
        if not getattr(self, "rows", None):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta", str(self.ctx.root / f"confronto.{kind}")
        )
        if not path:
            return
        head = ["Run", "n", "OA", "mF1", "mIoU", "mIoU senza clutter", "mIoU eroded"]
        body = [[r["name"], str(r["n"]), _ms(r["oa"]), _ms(r["mf1"]), _ms(r["miou"]),
                 _ms(r["miou_no_clutter"]), _ms(r["miou_eroded"])] for r in self.rows]  # fmt: skip
        if kind == "md":
            lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            lines += ["| " + " | ".join(b) + " |" for b in body]
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            import csv

            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([head, *body])
        self.ctx.status_message.emit(f"Tabella esportata in {path}")

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salva grafico", str(self.ctx.root / "miou.png"),
                                              "PNG (*.png)")  # fmt: skip
        if path:
            self.compare_plot.grab().save(path)


def report_json(run: RunInfo) -> str:
    return json.dumps(run.report or {}, indent=2)
