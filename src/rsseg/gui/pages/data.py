# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data: synthetic tiles, dataset preparation (with the dry-run check), unlabelled pools for
DINO, and a viewer of the prepared crops with their masks."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui.context import AppContext
from rsseg.gui.core.jobs import Job
from rsseg.gui.pages.config import list_configs
from rsseg.gui.pages.dashboard import prepared_datasets
from rsseg.gui.widgets import (
    ClassLegend,
    ImageView,
    action_button,
    colorize,
    load_array,
    section_title,
    tip,
)

INSPECTION_COLUMNS = ["tile", "split", "size", "label", "eroded", "ndsm", "off_palette", "notes"]


def _path_row(edit: QLineEdit, parent: QWidget, directory: bool = True) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    b = action_button("Sfoglia…", "folder")
    b.setToolTip("Scegli la cartella" if directory else "Scegli il file")

    def pick():
        path = (QFileDialog.getExistingDirectory(parent, "Cartella", edit.text() or ".") if directory
                else QFileDialog.getOpenFileName(parent, "File", edit.text() or ".")[0])  # fmt: skip
        if path:
            edit.setText(path)

    b.clicked.connect(pick)
    row.addWidget(b)
    return row


class DataPage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        tabs = QTabWidget()
        tabs.addTab(self._prepare_tab(), "Preparazione")
        tabs.addTab(self._unlabeled_tab(), "Non etichettati (DINO)")
        tabs.addTab(self._viewer_tab(), "Visualizza crop")
        v = QVBoxLayout(self)
        v.addWidget(section_title("Dati"))
        v.addWidget(tabs, 1)
        ctx.job_finished.connect(self._on_job_finished)

    # ------------------------------------------------------------------ preparation
    def _prepare_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        synth = QGroupBox("Dati sintetici (per provare tutto senza il dataset)")
        f = QFormLayout(synth)
        self.synth_out = QLineEdit("data/synthetic_raw")
        self.synth_size = QSpinBox()
        self.synth_size.setRange(64, 6000)
        self.synth_size.setValue(256)
        self.synth_size.setToolTip("Lato delle tile sintetiche in pixel")
        f.addRow("Cartella di uscita:", _path_row(self.synth_out, self))
        f.addRow("Lato delle tile:", self.synth_size)
        b = QPushButton("Genera tile sintetiche")
        b.setToolTip("Tile finte con gli stessi nomi di file di Potsdam: poi prepara con "
                     "configs/smoke.yaml")  # fmt: skip
        b.clicked.connect(lambda: self.ctx.submit("synth", ["--out", self.synth_out.text(), "--size",
                                                            str(self.synth_size.value())],
                                                  "Dati sintetici"))  # fmt: skip
        f.addRow(b)
        v.addWidget(synth)

        prep = QGroupBox("Prepara un dataset ISPRS")
        f = QFormLayout(prep)
        self.prep_config = QComboBox()
        self.prep_config.addItems(list_configs(self.ctx.root))
        idx = self.prep_config.findText("configs/base.yaml")
        self.prep_config.setCurrentIndex(max(idx, 0))
        tip(self.prep_config, "options", "config_file")
        self.prep_raw = QLineEdit()
        self.prep_raw.setPlaceholderText("dalla configurazione (data.raw_dir)")
        tip(self.prep_raw, "config", "data.raw_dir")
        self.prep_root = QLineEdit()
        self.prep_root.setPlaceholderText("dalla configurazione (data.root)")
        tip(self.prep_root, "config", "data.root")
        f.addRow("Configurazione:", self.prep_config)
        f.addRow("Dati originali:", _path_row(self.prep_raw, self))
        f.addRow("Cartella preparata:", _path_row(self.prep_root, self))
        row = QHBoxLayout()
        check = tip(QPushButton("1. Verifica (dry-run)"), "options", "prepare.dry_run")
        check.clicked.connect(lambda: self._prepare(dry=True))
        run = QPushButton("2. Prepara")
        run.setObjectName("primary")
        run.setToolTip("Taglia le tile in crop e maschere, calcola statistiche e manifest")
        run.clicked.connect(lambda: self._prepare(dry=False))
        row.addWidget(check)
        row.addWidget(run)
        f.addRow(row)
        v.addWidget(prep)

        self.inspection_title = QLabel("<i>Esegui la verifica per vedere file, split e anomalie "
                                       "di ogni tile.</i>")  # fmt: skip
        v.addWidget(self.inspection_title)
        self.inspection = QTableWidget(0, len(INSPECTION_COLUMNS))
        self.inspection.setHorizontalHeaderLabels(["tile", "split", "dimensione", "etichetta",
                                                   "erosa", "nDSM", "fuori palette", "note"])  # fmt: skip
        self.inspection.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.inspection.setEditTriggers(QTableWidget.NoEditTriggers)
        self.inspection.horizontalHeaderItem(6).setToolTip(
            "Pixel delle etichette con colori fuori dalla palette ISPRS (oltre data.max_offpalette "
            "la tile è rifiutata)")  # fmt: skip
        v.addWidget(self.inspection, 1)
        return w

    def _overrides(self) -> list[str]:
        out = []
        if self.prep_raw.text().strip():
            out.append(f"data.raw_dir={self.prep_raw.text().strip()}")
        if self.prep_root.text().strip():
            out.append(f"data.root={self.prep_root.text().strip()}")
        return out

    def _prepare(self, dry: bool) -> None:
        config, overrides = self.prep_config.currentText(), self._overrides()
        args = [*(["--dry-run"] if dry else []), "-c", config,
                *(["--set", *overrides] if overrides else [])]  # fmt: skip
        title = f"{'Verifica' if dry else 'Preparazione'} dati {Path(config).stem}"
        self.ctx.submit("prepare", args, title, config=config, overrides=overrides,
                        meta={"show": "inspection"} if dry else {})  # fmt: skip

    def _on_job_finished(self, job: Job) -> None:
        if job.meta.get("show") == "inspection":
            result = self.ctx.store.result(job)
            if isinstance(result, dict) and "tiles" in result:
                self.show_inspection(result)
        if job.command in ("prepare", "prepare-unlabeled"):
            self._refresh_roots()

    def show_inspection(self, result: dict) -> None:
        tiles = result["tiles"]
        self.inspection_title.setText(
            f"<b>{result['dataset']}</b> — {result['raw_dir']}: {result['tiles_found']} tile trovate, "
            f"non usate: {result.get('unused_tiles') or 'nessuna'}, errori: <b>{result['errors']}</b>")  # fmt: skip
        self.inspection.setRowCount(len(tiles))
        for r, t in enumerate(tiles):
            for c, key in enumerate(INSPECTION_COLUMNS):
                v = t.get(key, "")
                text = (
                    f"{v:.3%}" if key == "off_palette" and isinstance(v, (int, float)) else str(v)
                )
                item = QTableWidgetItem(text)
                if "ERROR" in str(t.get("notes", "")):
                    item.setForeground(Qt.red)
                self.inspection.setItem(r, c, item)

    # --------------------------------------------------------------------- unlabelled
    def _unlabeled_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.ul_source = QComboBox()
        self.ul_source.addItems(["toronto", "vaihingen_frames"])
        self.ul_source.setToolTip("Origine dei frame aerei grezzi (Toronto è di un'altra città, "
                                  "quindi non copre le aree di test)")  # fmt: skip
        self.ul_raw = QLineEdit("data/download/Toronto/Images")
        self.ul_out = QLineEdit("data/unlabeled/toronto")
        self.ul_out.setToolTip("Cartella del pool: poi aggiungila a ssl.extra_pools")
        self.ul_crop = QSpinBox()
        self.ul_crop.setRange(64, 4096)
        self.ul_crop.setValue(512)
        self.ul_crop.setToolTip("Lato dei crop in pixel")
        self.ul_max = QSpinBox()
        self.ul_max.setRange(0, 100000)
        self.ul_max.setToolTip("Crop massimi per frame (0 = tutti)")
        f.addRow("Sorgente:", self.ul_source)
        f.addRow("Frame grezzi:", _path_row(self.ul_raw, self))
        f.addRow("Uscita:", _path_row(self.ul_out, self))
        f.addRow("Lato crop:", self.ul_crop)
        f.addRow("Massimo per frame:", self.ul_max)
        b = QPushButton("Prepara pool non etichettato")
        b.clicked.connect(lambda: self.ctx.submit("prepare-unlabeled", [
            "--source", self.ul_source.currentText(), "--raw", self.ul_raw.text(), "--out",
            self.ul_out.text(), "--crop", str(self.ul_crop.value()), "--max-per-frame",
            str(self.ul_max.value())], f"Pool {self.ul_source.currentText()}"))  # fmt: skip
        f.addRow(b)
        return w

    # ------------------------------------------------------------------------ viewer
    def _viewer_tab(self) -> QWidget:
        w = QWidget()
        left = QWidget()
        lv = QVBoxLayout(left)
        self.roots = QComboBox()
        self.roots.setToolTip("Cartella di dati preparati (data.root)")
        self.split = QComboBox()
        self.split.addItems(["train", "val"])
        self.crops = QListWidget()
        self.roots.currentTextChanged.connect(lambda _: self._load_crops())
        self.split.currentTextChanged.connect(lambda _: self._load_crops())
        self.crops.currentRowChanged.connect(self._show_crop)
        self.crop_info = QLabel()
        self.crop_info.setWordWrap(True)
        for widget in (QLabel("Dati:"), self.roots, QLabel("Split:"), self.split, self.crops,
                       self.crop_info):  # fmt: skip
            lv.addWidget(widget)
        right = QWidget()
        rv = QVBoxLayout(right)
        self.crop_view = ImageView()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Opacità maschera:"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip("Trasparenza delle etichette sopra l'immagine")
        self.opacity.valueChanged.connect(lambda v: self.crop_view.overlay.setOpacity(v / 100))
        controls.addWidget(self.opacity)
        self.show_ndsm = QPushButton("Mostra nDSM")
        self.show_ndsm.setCheckable(True)
        self.show_ndsm.setToolTip("Mostra l'altezza sul terreno al posto dell'immagine RGB")
        self.show_ndsm.toggled.connect(lambda _: self._show_crop(self.crops.currentRow()))
        controls.addWidget(self.show_ndsm)
        self.legend = ClassLegend()
        self.legend.changed.connect(lambda: self._show_crop(self.crops.currentRow()))
        rv.addLayout(controls)
        rv.addWidget(self.legend)
        rv.addWidget(self.crop_view, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 800])
        QVBoxLayout(w).addWidget(split)
        self.rows: list[dict] = []
        self._refresh_roots()
        return w

    def _refresh_roots(self) -> None:
        current = self.roots.currentText()
        self.roots.blockSignals(True)
        self.roots.clear()
        self.roots.addItems([name for name, _ in prepared_datasets(self.ctx.root)])
        if current:
            self.roots.setCurrentText(current)
        self.roots.blockSignals(False)
        self._load_crops()

    def _load_crops(self) -> None:
        self.crops.clear()
        self.rows = []
        root = self.roots.currentText()
        if not root:
            return
        path = self.ctx.root / root / "crops.csv"
        try:
            with path.open(newline="", encoding="utf-8") as f:
                self.rows = [r for r in csv.DictReader(f) if r["split"] == self.split.currentText()]
        except OSError:
            return
        self.crops.addItems([f"{r['tile']} ({r['y']}, {r['x']})" for r in self.rows[:5000]])
        if self.rows:
            self.crops.setCurrentRow(0)

    def _show_crop(self, index: int) -> None:
        if not 0 <= index < len(self.rows):
            return
        row, root = self.rows[index], self.ctx.root / self.roots.currentText()
        use_ndsm = self.show_ndsm.isChecked() and row.get("ndsm")
        image = load_array(root / (row["ndsm"] if use_ndsm else row["image"]),
                           "L" if use_ndsm else "RGB")  # fmt: skip
        mask = load_array(root / row["mask"])
        self.crop_view.set_image(image, fit=True)
        self.crop_view.set_overlay(
            colorize(mask, self.legend.enabled()), self.opacity.value() / 100
        )
        from rsseg.data.classes import CLASS_NAMES
        from rsseg.gui.widgets import CLASS_LABELS_IT

        counts = [int(row.get(f"n{c}") or 0) for c in range(len(CLASS_NAMES))]
        total = sum(counts) or 1
        self.crop_info.setText("<br>".join(f"{CLASS_LABELS_IT[n]}: {c / total:.1%}"
                                           for n, c in zip(CLASS_NAMES, counts, strict=True)))  # fmt: skip
