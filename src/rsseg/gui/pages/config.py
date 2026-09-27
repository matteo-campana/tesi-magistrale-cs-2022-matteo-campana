# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration editor: a form generated from the dataclasses of :mod:`rsseg.config`, with an
Italian tooltip on every setting, and the buttons that launch training, pre-training, data
preparation and the VRAM benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rsseg.config import load_config
from rsseg.gui import actions
from rsseg.gui.context import AppContext
from rsseg.gui.core import help as helptext
from rsseg.gui.core.timing import fmt_duration
from rsseg.gui.widgets import section_title, tip

SECTIONS = [("general", "Generale"), ("data", "Dati"), ("model", "Modello"),
            ("train", "Training"), ("eval", "Valutazione"), ("ssl", "Pre-training DINO"),
            ("tracking", "MLflow")]  # fmt: skip


def list_configs(root: Path) -> list[str]:
    base = root / "configs"
    if not base.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in base.rglob("*.yaml")
                  if "campaigns" not in p.parts)  # fmt: skip


def _raw_chain(path: Path) -> list[tuple[Path, dict]]:
    """``[(file, raw values)]`` from the most basic ``base:`` to ``path``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    chain = _raw_chain((path.parent / base).resolve()) if base else []
    return [*chain, (path, raw)]


def value_sources(path: Path, root: Path) -> dict[str, str]:
    """Which file sets each dotted key (the last one in the ``base:`` chain wins)."""
    out: dict[str, str] = {}

    def walk(d: dict, prefix: str, name: str) -> None:
        for k, v in d.items():
            if isinstance(v, dict) and k != "tags":
                walk(v, f"{prefix}{k}.", name)
            else:
                out[prefix + k] = name

    for file, raw in _raw_chain(path):
        try:
            name = file.resolve().relative_to(root).as_posix()
        except ValueError:
            name = str(file)
        walk(raw, "", name)
    return out


def get_value(cfg, key: str) -> Any:
    obj = cfg
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def to_override(key: str, value: Any) -> str:
    return f"{key}={yaml.safe_dump(value, default_flow_style=True, width=10**6).strip().removesuffix('...').strip()}"


class FieldEditor(QObject):
    """One setting: label + editor, reading/writing Python values."""

    changed = Signal()
    focused = Signal(object)

    def __init__(self, field: helptext.FieldInfo):
        super().__init__()
        self.field = field
        self.label = QLabel(field.key.split(".")[-1])
        kind = field.kind
        values = (helptext.entry("config", field.key) or {}).get("values") or {}
        choices = [str(k) for k in values if not str(k).startswith("<")]
        if kind == "bool":
            self.widget = QCheckBox()
            self.widget.toggled.connect(lambda *_: self.changed.emit())
        elif kind == "int":
            self.widget = QSpinBox()
            self.widget.setRange(-1, 10**9)
            self.widget.valueChanged.connect(lambda *_: self.changed.emit())
        elif kind == "str" and choices:
            self.widget = QComboBox()
            self.widget.setEditable(True)
            self.widget.addItems(choices)
            self.widget.currentTextChanged.connect(lambda *_: self.changed.emit())
        else:
            self.widget = QLineEdit()
            if kind.startswith("optional"):
                self.widget.setPlaceholderText("vuoto")
            self.widget.textChanged.connect(lambda *_: self.changed.emit())
        self.widget.installEventFilter(self)
        self.file_value: Any = None

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.FocusIn:
            self.focused.emit(self)
        return False

    def set_value(self, value: Any) -> None:
        w = self.widget
        w.blockSignals(True)
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QSpinBox):
            w.setValue(int(value or 0))
        elif isinstance(w, QComboBox):
            w.setCurrentText("" if value is None else str(value))
        else:
            w.setText(self._text(value))
        w.blockSignals(False)

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
        if isinstance(value, float):
            return repr(value)
        return str(value)

    def value(self) -> Any:
        w, kind = self.widget, self.field.kind
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        text = w.currentText() if isinstance(w, QComboBox) else w.text()
        text = text.strip()
        if kind in ("str",):
            return text
        if not text:
            return None if kind.startswith("optional") or kind in ("list", "dict") else text
        if kind == "optional-str":
            return text
        if kind in ("float", "optional-float"):
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"{self.field.key}: serve un numero") from None
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"{self.field.key}: valore non valido ({e.problem})") from e
        if kind in ("float", "optional-float") and isinstance(value, int):
            value = float(value)
        if kind in ("int", "optional-int") and not isinstance(value, int):
            raise ValueError(f"{self.field.key}: serve un numero intero")
        if kind in ("float",) and not isinstance(value, float):
            raise ValueError(f"{self.field.key}: serve un numero")
        if kind == "list" and not isinstance(value, list):
            raise ValueError(f"{self.field.key}: serve una lista, es. [a, b]")
        if kind == "dict" and not isinstance(value, dict):
            raise ValueError(f"{self.field.key}: serve una mappa, es. {{chiave: valore}}")
        return value


class ConfigPage(QWidget):
    job_started = Signal(str)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.path: Path | None = None
        self.sources: dict[str, str] = {}
        self.editors: dict[str, FieldEditor] = {}

        top = QHBoxLayout()
        top.addWidget(tip(QLabel("File di configurazione:"), "options", "config_file"))
        self.files = QComboBox()
        self.files.addItems(list_configs(ctx.root))
        tip(self.files, "options", "config_file")
        self.files.currentTextChanged.connect(lambda t: self.load(ctx.root / t) if t else None)
        top.addWidget(self.files, 1)
        browse = QPushButton("Sfoglia…")
        browse.clicked.connect(self._browse)
        top.addWidget(browse)
        reset = QPushButton("Annulla modifiche")
        reset.setToolTip("Riporta tutti i valori a quelli del file")
        reset.clicked.connect(lambda: self.load(self.path) if self.path else None)
        top.addWidget(reset)

        self.tabs = QTabWidget()
        forms: dict[str, QFormLayout] = {}
        for key, title in SECTIONS:
            page = QWidget()
            forms[key] = QFormLayout(page)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)
        for field in helptext.config_fields():
            ed = FieldEditor(field)
            ed.changed.connect(self._on_changed)
            ed.focused.connect(self._show_help)
            forms[field.section if field.section in forms else "general"].addRow(
                ed.label, ed.widget
            )
            self.editors[field.key] = ed

        self.guide = QTextBrowser()
        self.guide.setMinimumWidth(260)
        self.guide.setHtml("<h3>Guida</h3><p>Passa il mouse su un'impostazione per vederne il "
                           "riepilogo, oppure selezionala per leggere qui la spiegazione.</p>")  # fmt: skip
        split = QSplitter()
        split.addWidget(self.tabs)
        split.addWidget(self.guide)
        split.setSizes([800, 320])

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.overrides_label = QLabel()
        self.overrides_label.setWordWrap(True)
        self.overrides_label.setStyleSheet("color:#555; font-family: monospace;")
        self.overrides_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        buttons = QHBoxLayout()
        spec = [("Entra in 8 GB?", self.benchmark_seg, "options", "benchmark"),
                ("Benchmark DINO", self.benchmark_dino, "options", "benchmark"),
                ("Verifica dati", self.prepare_dry, "options", "prepare.dry_run"),
                ("Prepara dati", self.prepare, None, None),
                ("▶ Avvia training", self.train, None, None),
                ("▶ Avvia pre-training DINO", self.pretrain, None, None),
                ("Valuta pesi…", self.evaluate, None, None),
                ("Salva come…", self.save_as, None, None)]  # fmt: skip
        for text, slot, section, key in spec:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if section:
                tip(b, section, key)
            buttons.addWidget(b)
        tips = {"Prepara dati": "Taglia le tile originali (data.raw_dir) in crop e maschere in "
                                "data.root, con manifest e statistiche.",
                "▶ Avvia training": "Aggiunge alla coda il training con questa configurazione "
                                    "(le modifiche diventano override --set).",
                "▶ Avvia pre-training DINO": "Pre-training self-supervised dell'encoder (ssl.*).",
                "Valuta pesi…": "Valuta un file di pesi sulle tile intere dello split eval.split.",
                "Salva come…": "Salva un nuovo YAML che eredita da questo file (base:) con i "
                               "soli valori modificati."}  # fmt: skip
        for i in range(buttons.count()):
            b = buttons.itemAt(i).widget()
            if b.text() in tips:
                b.setToolTip(tips[b.text()])

        v = QVBoxLayout(self)
        v.addWidget(section_title("Configurazione e avvio"))
        v.addLayout(top)
        v.addWidget(split, 1)
        v.addWidget(self.summary)
        v.addWidget(self.overrides_label)
        v.addLayout(buttons)
        if self.files.count():
            idx = self.files.findText("configs/experiments/maresunet_r18_imagenet.yaml")
            self.files.setCurrentIndex(max(idx, 0))
            self.load(ctx.root / self.files.currentText())

    # -------------------------------------------------------------------------- load
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "File di configurazione", str(self.ctx.root / "configs"),
                                              "YAML (*.yaml *.yml)")  # fmt: skip
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        try:
            cfg = load_config(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Configurazione non valida", f"{path}:\n{e}")
            return
        self.path = Path(path)
        self.sources = value_sources(self.path, self.ctx.root)
        for key, ed in self.editors.items():
            value = get_value(cfg, key)
            ed.file_value = value
            ed.set_value(value)
            source = self.sources.get(key, "predefinito del codice")
            tooltip = helptext.config_tooltip(ed.field, source=source)
            ed.label.setToolTip(tooltip)
            ed.widget.setToolTip(tooltip)
        self._on_changed()

    def _show_help(self, ed: FieldEditor) -> None:
        self.guide.setHtml(ed.widget.toolTip())

    # ---------------------------------------------------------------------- values
    def overrides(self) -> list[str]:
        """``key=value`` for the settings that differ from the file (raises on invalid input)."""
        out = []
        for key, ed in self.editors.items():
            value = ed.value()
            if value != ed.file_value:
                out.append(to_override(key, value))
        return out

    def config_arg(self) -> str:
        assert self.path is not None
        try:
            return self.path.resolve().relative_to(self.ctx.root).as_posix()
        except ValueError:
            return str(self.path)

    def _on_changed(self) -> None:
        errors = []
        for ed in self.editors.values():
            try:
                changed = ed.value() != ed.file_value
            except ValueError as e:
                errors.append(str(e))
                changed = True
            ed.label.setStyleSheet("font-weight: bold; color: #2471a3;" if changed else "")
        try:
            overrides = self.overrides()
        except ValueError:
            overrides = []
        self.overrides_label.setText(("--set " + " ".join(overrides)) if overrides else
                                     "Nessuna modifica rispetto al file.")  # fmt: skip
        if errors:
            self.summary.setText("<span style='color:#c0392b'>" + "<br>".join(errors) + "</span>")
            return
        cfg = self.ctx.config_for(self.config_arg(), overrides) if self.path else None
        if cfg is None:
            self.summary.setText("<span style='color:#c0392b'>Configurazione non valida</span>")
            return
        t = cfg.train
        parts = [f"Batch effettivo <b>{t.batch_size} × {t.grad_accum_steps} = "
                 f"{t.batch_size * t.grad_accum_steps}</b>",
                 f"input <b>{4 if cfg.data.ndsm else 3} canali</b>",
                 f"encoder <b>{cfg.model.arch}</b> ({cfg.model.weights})"]  # fmt: skip
        est = self.ctx.estimate("train", cfg)
        parts.append(f"durata stimata del training <b>{fmt_duration(est)}</b>" if est else
                     "durata stimata: <i>serve «Prepara dati»</i>")  # fmt: skip
        peak = self.ctx.benchmark_peak_gb(self.config_arg(), overrides)
        total = self.ctx.sample.mem_total_gb if self.ctx.sample else None
        if peak is not None:
            ok = total is None or peak < total * 0.95
            parts.append(f"picco VRAM misurato <b style='color:{'#1e8449' if ok else '#c0392b'}'>"
                         f"{peak:.1f} GB</b>" + (f" su {total:.1f}" if total else ""))  # fmt: skip
        else:
            parts.append("picco VRAM: <i>non misurato («Entra in 8 GB?»)</i>")
        if not (Path(cfg.data.root) / "crops.csv").exists():
            parts.append(
                f"<span style='color:#b9770e'>dati non preparati in {cfg.data.root}</span>"
            )
        self.summary.setText(" · ".join(parts))
        self.summary.setToolTip(helptext.text("config", "train.grad_accum_steps"))

    # --------------------------------------------------------------------- launch
    def _collect(self) -> tuple[str, list[str]] | None:
        if self.path is None:
            return None
        try:
            return self.config_arg(), self.overrides()
        except ValueError as e:
            QMessageBox.warning(self, "Valore non valido", str(e))
            return None

    def _args(self, config: str, overrides: list[str]) -> list[str]:
        return ["-c", config, *(["--set", *overrides] if overrides else [])]

    def _submit(self, command: str, title: str, extra: list[str] | None = None,
                meta: dict | None = None, gpu_check: bool = True, mode: str = "seg") -> None:  # fmt: skip
        got = self._collect()
        if got is None:
            return
        config, overrides = got
        if gpu_check:
            peak = self.ctx.benchmark_peak_gb(config, overrides, mode)
            if not actions.confirm_gpu_start(self, self.ctx, peak, title):
                return
        job = self.ctx.submit(command, [*(extra or []), *self._args(config, overrides)], title,
                              config=config, overrides=overrides, meta=meta)  # fmt: skip
        self.job_started.emit(job.id)

    def _name(self) -> str:
        return str(self.editors["name"].value() or Path(self.path).stem)

    def train(self) -> None:
        self._submit("train", f"Training {self._name()}")

    def pretrain(self) -> None:
        self._submit("pretrain", f"Pre-training DINO {self._name()}", mode="dino")

    def benchmark_seg(self) -> None:
        got = self._collect()
        if got:
            self._submit("benchmark", f"Benchmark VRAM {self._name()}", ["--mode", "seg"],
                         meta={"bench_key": self.ctx.bench_key(*got, "seg")}, gpu_check=False)  # fmt: skip

    def benchmark_dino(self) -> None:
        got = self._collect()
        if got:
            self._submit("benchmark", f"Benchmark DINO {self._name()}", ["--mode", "dino"],
                         meta={"bench_key": self.ctx.bench_key(*got, "dino")}, gpu_check=False)  # fmt: skip

    def prepare(self) -> None:
        self._submit("prepare", f"Preparazione dati {self._name()}", gpu_check=False)

    def prepare_dry(self) -> None:
        self._submit("prepare", f"Verifica dati {self._name()}", ["--dry-run"], gpu_check=False)

    def evaluate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pesi da valutare", str(self.ctx.root / "runs"),
                                              "Pesi Keras (*.weights.h5)")  # fmt: skip
        if path:
            self._submit("evaluate", f"Valutazione {Path(path).parent.name}", ["--weights", path])

    def save_as(self) -> None:
        got = self._collect()
        if got is None:
            return
        _, overrides = got
        path, _ = QFileDialog.getSaveFileName(self, "Salva configurazione",
                                              str(self.path.parent / f"{self.path.stem}_custom.yaml"),
                                              "YAML (*.yaml)")  # fmt: skip
        if not path:
            return
        from rsseg.config import parse_overrides

        data = {"base": os.path.relpath(self.path.resolve(), Path(path).resolve().parent),
                **parse_overrides(overrides)}  # fmt: skip
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                              encoding="utf-8")  # fmt: skip
        self.files.blockSignals(True)
        self.files.clear()
        self.files.addItems(list_configs(self.ctx.root))
        self.files.blockSignals(False)
        self.load(Path(path))
        self.ctx.status_message.emit(f"Configurazione salvata in {path}")
