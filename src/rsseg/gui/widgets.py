# SPDX-License-Identifier: AGPL-3.0-or-later
"""Widgets shared by the pages: job progress, GPU status, charts, synchronised image views."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui.core import help as helptext
from rsseg.gui.core.gpu import GPUSample
from rsseg.gui.core.timing import JobView, fmt_clock, fmt_duration, fmt_range

STATE_COLORS = {
    "queued": "#7f8c8d",
    "running": "#2471a3",
    "pausing": "#b9770e",
    "stopping": "#b9770e",
    "done": "#1e8449",
    "failed": "#c0392b",
    "paused": "#b9770e",
    "stopped": "#7f8c8d",
    "interrupted": "#c0392b",
}
LEVEL_COLORS = {"ready": "#1e8449", "warning": "#d4ac0d", "absent": "#c0392b"}


def tip(widget: QWidget, section: str, key: str, **kw) -> QWidget:
    """Attach the Italian tooltip ``section/key`` to ``widget`` (returns it)."""
    widget.setToolTip(helptext.tooltip(section, key, **kw))
    return widget


def badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;border-radius:3px;"
            f"padding:1px 6px'>&nbsp;{text}&nbsp;</span>")  # fmt: skip


def section_title(text: str) -> QLabel:
    label = QLabel(f"<b>{text}</b>")
    label.setStyleSheet("font-size: 14px; margin-top: 6px;")
    return label


# --------------------------------------------------------------------------- progress
class PhaseStrip(QWidget):
    """``● Avvio  ● Training  ○ Valutazione  ○ Report``."""

    COLORS = {"done": "#1e8449", "active": "#2471a3", "todo": "#95a5a6", "error": "#c0392b"}
    SYMBOLS = {"done": "✔", "active": "●", "todo": "○", "error": "✖"}

    def __init__(self):
        super().__init__()
        self.label = QLabel()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setToolTip("Fasi del processo: ✔ completata, ● in corso, ○ da fare, ✖ interrotta")

    def set_phases(self, phases: list[tuple[str, str]]) -> None:
        parts = [f"<span style='color:{self.COLORS[s]}'>{self.SYMBOLS[s]} {name}</span>"
                 for name, s in phases]  # fmt: skip
        self.label.setText("  →  ".join(parts))


class JobProgressWidget(QFrame):
    """Phases, progress bars, elapsed/remaining time and the buttons of one job."""

    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, compact: bool = False):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.compact = compact
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setTextFormat(Qt.RichText)
        head.addWidget(self.title, 1)
        self.pause_btn = tip(QPushButton("⏸ Pausa"), "options", "pause")
        self.resume_btn = tip(QPushButton("▶ Riprendi"), "options", "resume")
        self.stop_btn = tip(QPushButton("⏹ Stop"), "options", "stop")
        for b, sig in ((self.pause_btn, self.pause_clicked), (self.resume_btn, self.resume_clicked),
                       (self.stop_btn, self.stop_clicked)):  # fmt: skip
            b.clicked.connect(lambda *_, s=sig: s.emit())
            head.addWidget(b)
        v.addLayout(head)
        self.phases = PhaseStrip()
        v.addWidget(self.phases)
        self.grid = QGridLayout()
        v.addLayout(self.grid)
        self.bars: list[tuple[QLabel, QProgressBar]] = []
        self.total_label = QLabel("Totale")
        self.total = QProgressBar()
        self.total.setToolTip("Avanzamento complessivo: tempo trascorso / (trascorso + rimanente)")
        self.times = QLabel()
        self.times.setToolTip(
            "Trascorso: solo il tempo di lavoro (le pause sono escluse). Rimanente: stima dai tempi "
            "misurati (epoche, tile, immagini) o, prima delle misure, dai tempi di riferimento; "
            "con early stopping è un intervallo."
        )
        self.gpu = QLabel()
        self.gpu.setToolTip(helptext.text("gpu", "util"))
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #555;")
        v.addWidget(self.times)
        v.addWidget(self.gpu)
        v.addWidget(self.notes)
        self.set_view(None)

    def _bar(self, i: int) -> tuple[QLabel, QProgressBar]:
        while len(self.bars) <= i:
            label, bar = QLabel(), QProgressBar()
            bar.setTextVisible(True)
            bar.setMaximum(1000)
            self.grid.addWidget(label, len(self.bars), 0)
            self.grid.addWidget(bar, len(self.bars), 1)
            self.bars.append((label, bar))
        return self.bars[i]

    def set_view(self, view: JobView | None, job=None, sample: GPUSample | None = None) -> None:
        for label, bar in self.bars:
            label.hide()
            bar.hide()
        if view is None:
            self.title.setText("<i>Nessun job selezionato</i>")
            self.phases.hide()
            self.total_label.hide()
            self.total.hide()
            for w in (self.times, self.gpu, self.notes, self.pause_btn, self.resume_btn,
                      self.stop_btn):  # fmt: skip
                w.hide()
            return
        self.phases.show()
        color = STATE_COLORS.get(view.state, "#555")
        self.title.setText(f"<b>{view.title}</b> {badge(view.state_label, color)}")
        self.phases.set_phases(view.phases)
        rows = [*view.bars, None]
        for i, b in enumerate(rows):
            label, bar = self._bar(i)
            label.show()
            bar.show()
            if b is None:  # total
                label.setText("<b>Totale</b>")
                frac = view.total_fraction
                text = "" if frac is None else f"{frac:.0%}"
            else:
                label.setText(b.label.strip())
                frac, text = b.fraction, b.text
            if frac is None:
                bar.setRange(0, 0 if view.state == "running" else 1000)  # 0..0: busy indicator
                bar.setValue(0)
                bar.setFormat(text)
            else:
                bar.setRange(0, 1000)
                bar.setValue(int(frac * 1000))
                bar.setFormat(f"{text}  ({frac:.0%})" if text and b is not None else text or "%p%")
        running = view.state in ("running", "pausing", "stopping")
        parts = [f"Trascorso <b>{fmt_duration(view.elapsed_s)}</b>"]
        if running:
            parts.append(f"Rimanente <b>{fmt_range(view.remaining_s, view.remaining_max_s)}</b>")
            if view.finish_at:
                end = fmt_clock(view.finish_at)
                if view.remaining_max_s:
                    end += f" – {fmt_clock(time.time() + view.remaining_max_s)}"
                parts.append(f"Fine prevista <b>{end}</b>")
        elif view.state == "queued" and view.remaining_s:
            parts.append(f"Durata stimata <b>{fmt_duration(view.remaining_s)}</b>")
        self.times.setText(" · ".join(parts))
        self.times.show()
        if sample is not None and running and job is not None and job.uses_gpu:
            self.gpu.setText(gpu_line(sample))
            self.gpu.show()
        else:
            self.gpu.hide()
        self.notes.setText("<br>".join(view.notes))
        self.notes.setVisible(bool(view.notes))
        self.notes.setStyleSheet("color: #b9770e;" if view.stale else "color: #555;")
        pausable = job is not None and job.pausable
        self.pause_btn.setVisible(view.state == "running" and pausable)
        self.stop_btn.setVisible(view.state in ("running", "pausing", "queued"))
        self.resume_btn.setVisible(job is not None and job.resumable)


def gpu_line(s: GPUSample) -> str:
    parts = []
    if s.util is not None:
        parts.append(f"GPU {s.util:.0f} %")
    if s.mem_used_gb is not None:
        parts.append(f"VRAM {s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB")
    if s.temp_c is not None:
        parts.append(f"{s.temp_c:.0f} °C")
    if s.power_w is not None:
        parts.append(f"{s.power_w:.0f} W")
    return " · ".join(parts)


# ---------------------------------------------------------------------------- GPU status
class GPUStatusWidget(QWidget):
    """Compact status for the status bar: indicator, name, utilisation, memory, temperature."""

    clicked = Signal()

    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 0, 4, 0)
        self.dot = QLabel("●")
        self.text = QLabel("GPU: rilevamento…")
        self.mem = QProgressBar()
        self.mem.setMaximumWidth(120)
        self.mem.setMaximumHeight(14)
        self.mem.setRange(0, 1000)
        self.mem.setTextVisible(False)
        tip(self.mem, "gpu", "memory")
        for w in (self.dot, self.text, self.mem):
            h.addWidget(w)
        self.level = "warning"

    def mousePressEvent(self, event):  # noqa: N802 (Qt API)
        self.clicked.emit()

    def update_status(self, level: str, message: str, sample: GPUSample | None) -> None:
        self.level = level
        self.dot.setStyleSheet(f"color: {LEVEL_COLORS.get(level, '#999')}; font-size: 16px;")
        if sample is None:
            self.text.setText("GPU non rilevata" if level == "absent" else "GPU")
            self.mem.hide()
        else:
            name = sample.name.replace("NVIDIA ", "").replace("GeForce ", "")
            self.text.setText(f"{name} · {gpu_line(sample)}")
            if sample.mem_fraction is not None:
                self.mem.show()
                self.mem.setValue(int(sample.mem_fraction * 1000))
                color = "#c0392b" if sample.mem_fraction > 0.95 else "#2471a3"
                self.mem.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        self.setToolTip(f"<qt>{message.replace(chr(10), '<br>')}<br><br>"
                        "Clic per aprire la sezione GPU.</qt>")  # fmt: skip


# ------------------------------------------------------------------------------ charts
def plot_widget(title: str = "", left: str = "", bottom: str = ""):
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, background="w", foreground="#333")
    w = pg.PlotWidget(title=title)
    w.showGrid(x=True, y=True, alpha=0.3)
    if left:
        w.setLabel("left", left)
    if bottom:
        w.setLabel("bottom", bottom)
    w.addLegend(offset=(10, 10))
    return w


SERIES_COLORS = ["#2471a3", "#c0392b", "#1e8449", "#b9770e", "#7d3c98", "#17a589", "#566573"]


def plot_series(widget, series: dict[str, tuple[list, list]], markers: list[float] | None = None):
    """Replace the curves of ``widget``: ``{name: (x, y)}``; ``markers`` draw vertical lines."""
    import pyqtgraph as pg

    widget.clear()
    for i, (name, (x, y)) in enumerate(series.items()):
        pts = [(a, b) for a, b in zip(x, y, strict=False) if b is not None and b == b]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        pen = pg.mkPen(SERIES_COLORS[i % len(SERIES_COLORS)], width=2)
        widget.plot(list(xs), list(ys), pen=pen, name=name, symbol="o" if len(xs) < 60 else None,
                    symbolSize=5, symbolBrush=SERIES_COLORS[i % len(SERIES_COLORS)])  # fmt: skip
    for m in markers or []:
        widget.addItem(pg.InfiniteLine(pos=m, angle=90, pen=pg.mkPen("#b9770e", style=Qt.DashLine),
                                       label="ripresa", labelOpts={"position": 0.9}))  # fmt: skip


def bar_chart(widget, labels: list[str], values: list[float], colors: list[str] | None = None):
    import pyqtgraph as pg

    widget.clear()
    xs = list(range(len(values)))
    brushes = [pg.mkBrush(c) for c in (colors or [SERIES_COLORS[0]] * len(values))]
    widget.addItem(pg.BarGraphItem(x=xs, height=[v or 0 for v in values], width=0.7,
                                   brushes=brushes))  # fmt: skip
    widget.getAxis("bottom").setTicks([list(zip(xs, labels, strict=True))])


# ------------------------------------------------------------------------------ images
def to_qimage(array: np.ndarray) -> QImage:
    """QImage (deep copy) of an ``(H, W)`` uint8, ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA array."""
    a = np.ascontiguousarray(array)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    if a.ndim == 2:
        fmt, bpl = QImage.Format_Grayscale8, w
    elif a.shape[2] == 3:
        fmt, bpl = QImage.Format_RGB888, 3 * w
    else:
        fmt, bpl = QImage.Format_RGBA8888, 4 * w
    return QImage(a.data, w, h, bpl, fmt).copy()


def load_array(path: str | Path, mode: str | None = None) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # ISPRS tiles are large on purpose
    with Image.open(path) as im:
        return np.asarray(im.convert(mode) if mode else im)


class ImageView(QGraphicsView):
    """Zoomable, pannable image with an optional overlay; emits the pixel under the mouse."""

    view_changed = Signal()
    hovered = Signal(int, int)

    def __init__(self, title: str = ""):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#2c2c2c"))
        self.base = QGraphicsPixmapItem()
        self.overlay = QGraphicsPixmapItem()
        self.overlay.setZValue(1)
        self.scene().addItem(self.base)
        self.scene().addItem(self.overlay)
        self.title = title
        self._syncing = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizontalScrollBar().valueChanged.connect(lambda _: self._changed())
        self.verticalScrollBar().valueChanged.connect(lambda _: self._changed())

    def set_image(self, array: np.ndarray | None, fit: bool = False) -> None:
        if array is None:
            self.base.setPixmap(QPixmap())
            return
        self.base.setPixmap(QPixmap.fromImage(to_qimage(array)))
        self.setSceneRect(QRectF(self.base.pixmap().rect()))
        if fit:
            self.fit()

    def set_overlay(self, array: np.ndarray | None, opacity: float = 0.5) -> None:
        if array is None:
            self.overlay.setPixmap(QPixmap())
        else:
            self.overlay.setPixmap(QPixmap.fromImage(to_qimage(array)))
        self.overlay.setOpacity(opacity)

    def fit(self) -> None:
        self.fitInView(self.base, Qt.KeepAspectRatio)
        self._changed()

    def wheelEvent(self, event):  # noqa: N802 (Qt API)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._changed()

    def mouseMoveEvent(self, event):  # noqa: N802
        super().mouseMoveEvent(event)
        p = self.mapToScene(event.position().toPoint())
        self.hovered.emit(int(p.x()), int(p.y()))

    def _changed(self) -> None:
        if not self._syncing:
            self.view_changed.emit()

    def center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def apply(self, other: ImageView) -> None:
        """Copy zoom and position of ``other``."""
        self._syncing = True
        self.setTransform(other.transform())
        self.centerOn(other.center())
        self._syncing = False


class SyncedViews(QWidget):
    """Side-by-side image views sharing zoom and position."""

    hovered = Signal(int, int)

    def __init__(self, titles: list[str]):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.views: list[ImageView] = []
        self.labels: list[QLabel] = []
        cols = 2 if len(titles) > 2 else len(titles)
        for i, t in enumerate(titles):
            box = QVBoxLayout()
            label = QLabel(f"<b>{t}</b>")
            view = ImageView(t)
            view.view_changed.connect(lambda v=view: self._sync(v))
            view.hovered.connect(self.hovered.emit)
            box.addWidget(label)
            box.addWidget(view, 1)
            grid.addLayout(box, i // cols, i % cols)
            self.views.append(view)
            self.labels.append(label)

    def _sync(self, source: ImageView) -> None:
        for v in self.views:
            if v is not source:
                v.apply(source)

    def fit(self) -> None:
        if self.views:
            self.views[0].fit()
            self._sync(self.views[0])

    def render_png(self, path: str | Path) -> None:
        self.grab().save(str(path))


# ------------------------------------------------------------------------------ classes
CLASS_LABELS_IT = {
    "impervious_surfaces": "superfici impermeabili",
    "building": "edifici",
    "low_vegetation": "vegetazione bassa",
    "tree": "alberi",
    "car": "auto",
    "clutter": "clutter / sfondo",
}


def class_colors() -> list[str]:
    """Hex colours of the classes for charts (white impervious surfaces drawn light grey)."""
    from rsseg.data.classes import PALETTE

    colors = ["#{:02x}{:02x}{:02x}".format(*(int(x) for x in rgb)) for rgb in PALETTE]
    return [c if c != "#ffffff" else "#bbbbbb" for c in colors]


def colorize(index: np.ndarray, enabled: set[int] | None = None) -> np.ndarray:
    """RGBA image of a class-index map; disabled classes and ignored pixels are transparent."""
    from rsseg.data.classes import NUM_CLASSES, PALETTE

    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:NUM_CLASSES, :3] = PALETTE
    lut[:NUM_CLASSES, 3] = 255
    if enabled is not None:
        for c in range(NUM_CLASSES):
            if c not in enabled:
                lut[c, 3] = 0
    return lut[index]


class ClassLegend(QWidget):
    """Coloured check boxes, one per class: which classes are drawn."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        from PySide6.QtWidgets import QCheckBox

        from rsseg.data.classes import CLASS_NAMES, PALETTE

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.boxes = []
        for i, name in enumerate(CLASS_NAMES):
            r, g, b = (int(x) for x in PALETTE[i])
            box = QCheckBox(CLASS_LABELS_IT[name])
            box.setChecked(True)
            box.setStyleSheet(f"QCheckBox::indicator:checked {{ background: rgb({r},{g},{b}); "
                              "border: 1px solid #333; }")  # fmt: skip
            box.setToolTip(f"Mostra o nascondi la classe «{CLASS_LABELS_IT[name]}» "
                           f"(colore ISPRS rgb({r},{g},{b}))")  # fmt: skip
            box.toggled.connect(lambda *_: self.changed.emit())
            h.addWidget(box)
            self.boxes.append(box)
        h.addStretch(1)

    def enabled(self) -> set[int]:
        return {i for i, b in enumerate(self.boxes) if b.isChecked()}
