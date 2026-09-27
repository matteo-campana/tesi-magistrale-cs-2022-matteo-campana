# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference with the trained models: choose a run and a checkpoint, segment tiles or any
image, then inspect the result — synchronised views, overlay, class toggles, pixel info,
confidence, errors, comparison of two models — and export figures."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui import actions
from rsseg.gui.context import AppContext
from rsseg.gui.core import help as helptext
from rsseg.gui.core.jobs import Job
from rsseg.gui.core.runs import RunInfo, find_runs
from rsseg.gui.widgets import (
    CLASS_LABELS_IT,
    ClassLegend,
    ImageView,
    JobProgressWidget,
    SyncedViews,
    bar_chart,
    class_colors,
    colorize,
    load_array,
    plot_widget,
    section_title,
    tip,
)


def prediction_folders(root: Path) -> list[Path]:
    base = root / "runs" / "predictions"
    if not base.exists():
        return []
    return sorted((p.parent for p in base.glob("*/predictions.json")),
                  key=lambda p: p.stat().st_mtime, reverse=True)  # fmt: skip


def read_predictions(folder: Path) -> dict:
    try:
        return json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"images": []}


def difference_map(a: np.ndarray, b: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    """RGB: grey where the models agree; with a ground truth green where only B is right, red
    where only A is right, orange where both are wrong differently; without it, magenta."""
    out = np.full((*a.shape, 3), 40, dtype=np.uint8)
    diff = a != b
    if gt is None:
        out[diff] = (255, 0, 255)
        return out
    valid = gt != 255
    out[diff & valid & (b == gt)] = (40, 200, 40)
    out[diff & valid & (a == gt)] = (220, 40, 40)
    out[diff & valid & (a != gt) & (b != gt)] = (240, 150, 0)
    return out


class InferencePage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.runs: list[RunInfo] = []
        self.job_id: str | None = None
        self.current: dict | None = None
        self.arrays: dict[str, np.ndarray | None] = {}

        # ------------------------------------------------------------------ setup panel
        setup = QWidget()
        sv = QVBoxLayout(setup)
        model_box = QGroupBox("1. Modello")
        mf = QFormLayout(model_box)
        self.model = QComboBox()
        self.model.currentIndexChanged.connect(self._on_model)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        self.checkpoint = tip(QComboBox(), "options", "predict.checkpoint")
        refresh = QPushButton("Aggiorna elenco")
        refresh.clicked.connect(self.refresh_models)
        mf.addRow("Run:", self.model)
        mf.addRow(self.model_info)
        mf.addRow("Pesi:", self.checkpoint)
        mf.addRow(refresh)
        sv.addWidget(model_box)

        input_box = QGroupBox("2. Immagini")
        iv = QVBoxLayout(input_box)
        self.mode_split = QRadioButton("Tile di uno split dei dati del run (con ground truth)")
        tip(self.mode_split, "options", "predict.split")
        self.mode_files = QRadioButton("Immagini o cartelle scelte (senza ground truth)")
        self.mode_split.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.mode_split)
        group.addButton(self.mode_files)
        self.split = QComboBox()
        self.split.addItems(["test", "val"])
        self.files = QListWidget()
        self.files.setMaximumHeight(90)
        self.files.setToolTip("Immagini RGB (TIFF, PNG, JPG) o cartelle che le contengono")
        add_files = QPushButton("Aggiungi immagini…")
        add_files.clicked.connect(self._add_files)
        add_dir = QPushButton("Aggiungi cartella…")
        add_dir.clicked.connect(self._add_dir)
        clear = QPushButton("Svuota")
        clear.clicked.connect(self.files.clear)
        self.ndsm_files = QListWidget()
        self.ndsm_files.setMaximumHeight(60)
        self.ndsm_files.setToolTip(
            "Per un modello RGB + nDSM: un nDSM per immagine, nello stesso ordine"
        )
        add_ndsm = QPushButton("Aggiungi nDSM…")
        add_ndsm.clicked.connect(self._add_ndsm)
        iv.addWidget(self.mode_split)
        iv.addWidget(self.split)
        iv.addWidget(self.mode_files)
        iv.addWidget(self.files)
        row = QHBoxLayout()
        for b in (add_files, add_dir, clear):
            row.addWidget(b)
        iv.addLayout(row)
        self.ndsm_label = QLabel("nDSM (solo modelli a 4 canali):")
        iv.addWidget(self.ndsm_label)
        iv.addWidget(self.ndsm_files)
        iv.addWidget(add_ndsm)
        sv.addWidget(input_box)

        params = QGroupBox("3. Parametri")
        pf = QFormLayout(params)
        self.window = tip(QSpinBox(), "options", "predict.window")
        self.window.setRange(32, 4096)
        self.window.setSingleStep(32)
        self.stride = tip(QSpinBox(), "options", "predict.stride")
        self.stride.setRange(16, 4096)
        self.batch = tip(QSpinBox(), "options", "predict.batch_size")
        self.batch.setRange(1, 256)
        self.tta = tip(QCheckBox("Test-time augmentation"), "options", "predict.tta")
        self.out = QLineEdit()
        self.out.setToolTip("Cartella in cui salvare predizioni, confidenza ed errori")
        pf.addRow("Finestra:", self.window)
        pf.addRow("Passo:", self.stride)
        pf.addRow("Batch:", self.batch)
        pf.addRow(self.tta)
        pf.addRow("Uscita:", self.out)
        run = QPushButton("▶ Esegui inferenza")
        run.clicked.connect(self.run)
        pf.addRow(run)
        sv.addWidget(params)
        self.progress = JobProgressWidget()
        self.progress.stop_clicked.connect(self._stop)
        sv.addWidget(self.progress)
        sv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(setup)
        scroll.setMinimumWidth(380)

        # ------------------------------------------------------------------ viewer
        viewer = QWidget()
        vv = QVBoxLayout(viewer)
        top = QHBoxLayout()
        self.results = QComboBox()
        self.results.setToolTip("Inferenze già eseguite (runs/predictions)")
        self.results.currentIndexChanged.connect(self._on_result)
        self.images = QComboBox()
        self.images.currentIndexChanged.connect(self._on_image)
        self.compare = QComboBox()
        self.compare.setToolTip("Confronta con un'altra inferenza sulla stessa immagine: i pannelli "
                                "mostrano i due modelli e la mappa delle differenze")  # fmt: skip
        self.compare.currentIndexChanged.connect(
            lambda _: self._on_image(self.images.currentIndex())
        )
        top.addWidget(QLabel("Risultato:"))
        top.addWidget(self.results, 2)
        top.addWidget(QLabel("Immagine:"))
        top.addWidget(self.images, 1)
        top.addWidget(QLabel("Confronta con:"))
        top.addWidget(self.compare, 2)
        vv.addLayout(top)
        controls = QHBoxLayout()
        self.layout_mode = QComboBox()
        self.layout_mode.addItems(["Pannelli affiancati", "Sovrapposizione"])
        self.layout_mode.setToolTip("Affiancati: immagine, predizione, verità/confidenza ed errori "
                                    "con zoom sincronizzato. Sovrapposizione: predizione sopra "
                                    "l'immagine con opacità regolabile.")  # fmt: skip
        self.layout_mode.currentIndexChanged.connect(lambda i: self.stack.setCurrentIndex(i))
        self.third = QComboBox()
        self.third.addItems(["Ground truth", "Confidenza"])
        self.third.setToolTip(helptext.text("metrics", "confidence"))
        self.third.currentIndexChanged.connect(lambda _: self._render())
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(55)
        self.opacity.setToolTip("Opacità della predizione sopra l'immagine")
        self.opacity.valueChanged.connect(lambda v: self.overlay_view.overlay.setOpacity(v / 100))
        fit = QPushButton("Adatta")
        fit.clicked.connect(self._fit)
        save = QPushButton("Salva vista PNG…")
        save.clicked.connect(self._save_view)
        panel = QPushButton("Esporta pannello…")
        panel.setToolTip("Immagine | predizione | verità | errori affiancati a piena risoluzione, "
                         "pronti per la tesi")  # fmt: skip
        panel.clicked.connect(self._export_panel)
        for w in (self.layout_mode, QLabel("3° pannello:"), self.third, QLabel("Opacità:"),
                  self.opacity, fit, save, panel):  # fmt: skip
            controls.addWidget(w)
        vv.addLayout(controls)
        self.legend = ClassLegend()
        self.legend.changed.connect(self._render)
        vv.addWidget(self.legend)
        self.stack = QStackedWidget()
        self.synced = SyncedViews(["Immagine", "Predizione", "Ground truth", "Errori"])
        self.synced.hovered.connect(self._hover)
        self.overlay_view = ImageView()
        self.overlay_view.hovered.connect(self._hover)
        self.stack.addWidget(self.synced)
        self.stack.addWidget(self.overlay_view)
        vv.addWidget(self.stack, 1)
        self.pixel = QLabel(
            "Passa il mouse sull'immagine per vedere classe e confidenza del pixel."
        )
        vv.addWidget(self.pixel)
        stats = QHBoxLayout()
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextFormat(Qt.RichText)
        self.hist = plot_widget("Classi predette", "% pixel")
        self.hist.setMaximumHeight(200)
        stats.addWidget(self.stats, 1)
        stats.addWidget(self.hist, 1)
        vv.addLayout(stats)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(viewer)
        split.setSizes([400, 1000])
        v = QVBoxLayout(self)
        v.addWidget(section_title("Inferenza con i modelli addestrati"))
        v.addWidget(split, 1)
        ctx.jobs_changed.connect(self._refresh_job)
        ctx.job_finished.connect(self._on_job_finished)
        self.refresh_models()
        self.refresh_results()

    # ------------------------------------------------------------------ models
    def refresh_models(self) -> None:
        self.runs = [r for r in find_runs([self.ctx.root / "runs"], ("seg",)) if r.checkpoints()]
        self.model.blockSignals(True)
        self.model.clear()
        for i, r in enumerate(self.runs):
            miou = f" · test mIoU {r.test_miou:.3f}" if r.test_miou is not None else ""
            state = {"paused": " · in pausa", "incomplete": " · in corso/incompleto"}.get(
                r.state, ""
            )
            self.model.addItem(f"{r.name} ({r.stamp}) — {r.arch}, {r.pretrain}{miou}{state}")
            self.model.setItemData(i, self._run_tooltip(r), Qt.ToolTipRole)
        self.model.blockSignals(False)
        self._on_model(self.model.currentIndex())

    @staticmethod
    def _run_tooltip(r: RunInfo) -> str:
        cfg = r.config
        t = cfg.get("train") or {}
        return (f"<qt><b>{r.name}</b> ({r.stamp})<br>encoder {r.arch}, pre-training {r.pretrain}, "
                f"{r.channels} canali<br>epoche {t.get('epochs')}, batch {t.get('batch_size')}"
                f"<br>val mIoU {r.best_val_miou or float('nan'):.4f} · test mIoU "
                f"{r.test_miou if r.test_miou is not None else float('nan'):.4f}<br>"
                f"dati: {r.data_root()}<br>{r.path}</qt>")  # fmt: skip

    def _run(self) -> RunInfo | None:
        i = self.model.currentIndex()
        return self.runs[i] if 0 <= i < len(self.runs) else None

    def _on_model(self, _index: int) -> None:
        run = self._run()
        self.checkpoint.clear()
        if run is None:
            self.model_info.setText("<i>Nessun modello addestrato in runs/ (servono i pesi "
                                    "best.weights.h5).</i>")  # fmt: skip
            return
        for label, path in run.checkpoints():
            self.checkpoint.addItem(label, str(path))
        self.model_info.setText(self._run_tooltip(run).replace("<qt>", "").replace("</qt>", ""))
        ev = run.config.get("eval") or {}
        self.window.setValue(int(ev.get("window") or 512))
        self.stride.setValue(int(ev.get("stride") or 256))
        self.batch.setValue(int(ev.get("batch_size") or 8))
        self.tta.setChecked(bool(ev.get("tta")))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out.setText(f"runs/predictions/{run.name}-{stamp}")
        four = run.channels == 4
        self.ndsm_label.setVisible(four)
        self.ndsm_files.setVisible(four)

    # ------------------------------------------------------------------ inputs
    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Immagini", str(self.ctx.root),
                                                "Immagini (*.tif *.tiff *.png *.jpg *.jpeg)")  # fmt: skip
        self.files.addItems(paths)
        if paths:
            self.mode_files.setChecked(True)

    def _add_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Cartella di immagini", str(self.ctx.root))
        if path:
            self.files.addItem(path)
            self.mode_files.setChecked(True)

    def _add_ndsm(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "nDSM", str(self.ctx.root))
        self.ndsm_files.addItems(paths)

    def run(self) -> None:
        run = self._run()
        if run is None:
            return
        args = ["--run", str(run.path), "--weights", self.checkpoint.currentData() or "",
                "--out", self.out.text(), "--window", str(self.window.value()),
                "--stride", str(self.stride.value()), "--batch-size", str(self.batch.value())]  # fmt: skip
        if self.tta.isChecked():
            args.append("--tta")
        if self.mode_split.isChecked():
            args += ["--split", self.split.currentText()]
        else:
            items = [self.files.item(i).text() for i in range(self.files.count())]
            if not items:
                self.ctx.status_message.emit("Aggiungi almeno un'immagine")
                return
            args = [*items, *args]
            nd = [self.ndsm_files.item(i).text() for i in range(self.ndsm_files.count())]
            if nd:
                args += ["--ndsm", *nd]
        if not actions.confirm_gpu_start(self, self.ctx, None, "l'inferenza"):
            return
        job = self.ctx.submit("predict", args, f"Inferenza {run.name}",
                              meta={"show": "predictions", "out": self.out.text()})  # fmt: skip
        self.job_id = job.id
        self._on_model(self.model.currentIndex())  # new output folder for the next run

    def _stop(self) -> None:
        job = self.ctx.store.jobs.get(self.job_id or "")
        if job:
            self.ctx.store.request(job, "stop")

    def _refresh_job(self) -> None:
        job = self.ctx.store.jobs.get(self.job_id or "")
        self.progress.set_view(self.ctx.view(job) if job else None, job, self.ctx.sample)

    def _on_job_finished(self, job: Job) -> None:
        if job.command == "train":
            self.refresh_models()
        if job.meta.get("show") == "predictions":
            self.refresh_results(select=self.ctx.root / job.meta["out"])

    # ----------------------------------------------------------------- results
    def refresh_results(self, select: Path | None = None) -> None:
        self.folders = prediction_folders(self.ctx.root)
        for combo, first in ((self.results, None), (self.compare, "— nessun confronto —")):
            combo.blockSignals(True)
            combo.clear()
            if first:
                combo.addItem(first, None)
            for f in self.folders:
                combo.addItem(f.name, str(f))
            combo.blockSignals(False)
        if select is not None:
            idx = self.results.findData(str(select.resolve()))
            if idx < 0:
                idx = self.results.findText(select.name)
            self.results.setCurrentIndex(max(idx, 0))
        self._on_result(self.results.currentIndex())

    def _on_result(self, _index: int) -> None:
        folder = self.results.currentData()
        self.summary = read_predictions(Path(folder)) if folder else {"images": []}
        self.images.blockSignals(True)
        self.images.clear()
        for item in self.summary["images"]:
            self.images.addItem(item["name"])
        self.images.blockSignals(False)
        self._on_image(0)

    def _on_image(self, index: int) -> None:
        images = self.summary.get("images", []) if hasattr(self, "summary") else []
        if not 0 <= index < len(images):
            self.current = None
            return
        item = dict(images[index])
        for key in ("image", "pred", "confidence", "mask", "errors"):
            if item.get(key) and not Path(item[key]).is_absolute():
                item[key] = str(self.ctx.root / item[key])
        self.current = item
        a = self.arrays
        a.clear()
        a["image"] = load_array(item["image"], "RGB")
        a["pred"] = load_array(item["pred"])
        a["conf"] = load_array(item["confidence"]) if Path(item["confidence"]).exists() else None
        a["gt"] = load_array(item["mask"]) if item.get("mask") else None
        a["errors"] = load_array(item["errors"], "RGB") if item.get("errors") else None
        a["other"] = None
        other_dir = self.compare.currentData()
        if other_dir:
            other = next((i for i in read_predictions(Path(other_dir))["images"]
                          if i["name"] == item["name"]), None)  # fmt: skip
            if other is not None:
                a["other"] = load_array(other["pred"])
                self.compare_name = Path(other_dir).name
            else:
                self.ctx.status_message.emit(
                    "L'inferenza di confronto non contiene questa immagine"
                )
        self._render(fit=True)
        self._stats(item)

    def _render(self, fit: bool = False) -> None:
        a = self.arrays
        if not a or a.get("pred") is None:
            return
        enabled = self.legend.enabled()
        pred_rgba = colorize(a["pred"], enabled)
        views, labels = self.synced.views, self.synced.labels
        views[0].set_image(a["image"])
        views[1].set_image(a["image"] * 0 + 30)
        views[1].set_overlay(pred_rgba, 1.0)
        if a.get("other") is not None:
            labels[1].setText("<b>Predizione A (questo risultato)</b>")
            views[2].set_image(a["image"] * 0 + 30)
            views[2].set_overlay(colorize(a["other"], enabled), 1.0)
            labels[2].setText(f"<b>Predizione B ({self.compare_name})</b>")
            views[3].set_image(difference_map(a["pred"], a["other"], a.get("gt")))
            views[3].set_overlay(None)
            labels[3].setText("<b>Differenze</b> (verde: solo B giusto · rosso: solo A giusto · "
                              "arancio: entrambi sbagliati)" if a.get("gt") is not None else
                              "<b>Differenze</b> (magenta: i modelli non concordano)")  # fmt: skip
        else:
            labels[1].setText("<b>Predizione</b>")
            third = self.third.currentText()
            if third == "Ground truth" and a.get("gt") is not None:
                views[2].set_image(a["image"] * 0 + 30)
                views[2].set_overlay(colorize(a["gt"], enabled), 1.0)
                labels[2].setText("<b>Ground truth</b>")
            elif a.get("conf") is not None:
                views[2].set_image(a["conf"])
                views[2].set_overlay(None)
                labels[2].setText("<b>Confidenza</b> (bianco = sicura, nero = incerta)")
            views[3].set_image(a["errors"] if a.get("errors") is not None else a["image"] * 0)
            views[3].set_overlay(None)
            labels[3].setText("<b>Errori</b> (magenta = sbagliato)" if a.get("errors") is not None
                              else "<b>Errori</b> (serve la ground truth)")  # fmt: skip
        self.overlay_view.set_image(a["image"])
        self.overlay_view.set_overlay(pred_rgba, self.opacity.value() / 100)
        if fit:
            self._fit()

    def _fit(self) -> None:
        self.synced.fit()
        self.overlay_view.fit()

    def _hover(self, x: int, y: int) -> None:
        a = self.arrays
        pred = a.get("pred")
        if pred is None or not (0 <= y < pred.shape[0] and 0 <= x < pred.shape[1]):
            return
        from rsseg.data.classes import CLASS_NAMES

        def name(c: int) -> str:
            return CLASS_LABELS_IT[CLASS_NAMES[c]] if c < len(CLASS_NAMES) else "non etichettato"

        parts = [f"({x}, {y})", f"predetta: <b>{name(int(pred[y, x]))}</b>"]
        if a.get("conf") is not None:
            parts.append(f"confidenza {a['conf'][y, x] / 255:.2f}")
        if a.get("gt") is not None:
            ok = a["gt"][y, x] == pred[y, x]
            parts.append(f"vera: <b style='color:{'#1e8449' if ok else '#c0392b'}'>"
                         f"{name(int(a['gt'][y, x]))}</b>")  # fmt: skip
        if a.get("other") is not None:
            parts.append(f"modello B: <b>{name(int(a['other'][y, x]))}</b>")
        self.pixel.setText(" · ".join(parts))

    def _stats(self, item: dict) -> None:
        from rsseg.data.classes import CLASS_NAMES

        lines = [f"<b>{item['name']}</b> · {item['size'][1]}×{item['size'][0]} px · "
                 f"{item['seconds']:.1f} s · confidenza media {item.get('mean_confidence', 0):.3f}"]  # fmt: skip
        s = item.get("scores")
        if s:
            lines.append(f"OA <b>{s['oa']:.4f}</b> · mF1 <b>{s['mf1']:.4f}</b> · mIoU "
                         f"<b>{s['miou']:.4f}</b>")  # fmt: skip
            lines.append(" · ".join(f"{CLASS_LABELS_IT[n].split()[0]} {c['iou']:.3f}"
                                    for n, c in s["per_class"].items()))  # fmt: skip
        overall = self.summary.get("scores")
        if overall:
            lines.append(f"Su tutte le immagini: mIoU <b>{overall['miou']:.4f}</b>")
        self.stats.setText("<br>".join(lines))
        self.stats.setToolTip(helptext.tooltip("metrics", "miou"))
        fr = item.get("class_fractions", {})
        colors = class_colors()[: len(CLASS_NAMES)]
        bar_chart(self.hist, [CLASS_LABELS_IT[n].split()[0] for n in CLASS_NAMES],
                  [100 * fr.get(n, 0) for n in CLASS_NAMES], colors)  # fmt: skip

    # ------------------------------------------------------------------ export
    def _save_view(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salva vista", str(self.ctx.root / "vista.png"),
                                              "PNG (*.png)")  # fmt: skip
        if path:
            self.stack.currentWidget().grab().save(path)
            self.ctx.status_message.emit(f"Vista salvata in {path}")

    def _export_panel(self) -> None:
        a = self.arrays
        if a.get("pred") is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta pannello", str(self.ctx.root / f"{self.current['name']}_pannello.png"),
            "PNG (*.png)")  # fmt: skip
        if not path:
            return
        from PIL import Image

        from rsseg.data.classes import index_to_rgb

        parts = [a["image"][..., :3], index_to_rgb(a["pred"])]
        if a.get("other") is not None:
            parts += [index_to_rgb(a["other"]), difference_map(a["pred"], a["other"], a.get("gt"))]
        else:
            if a.get("gt") is not None:
                parts.append(index_to_rgb(a["gt"]))
            if a.get("errors") is not None:
                parts.append(a["errors"])
        h = parts[0].shape[0]
        gap = np.full((h, 8, 3), 255, dtype=np.uint8)
        row = []
        for p in parts:
            row += [p, gap]
        Image.fromarray(np.concatenate(row[:-1], axis=1)).save(path)
        self.ctx.status_message.emit(f"Pannello esportato in {path}")
