# SPDX-License-Identifier: AGPL-3.0-or-later
"""Widgets shared by the pages: job progress, GPU status, charts, synchronised image views.

Colours come from `rsseg.gui.theme` by meaning (a tone such as ``"danger"``), never as hex:
widgets that bake a colour into rich text, a brush or a chart define ``retheme()``, which the
theme calls when the user switches between dark and light.
"""

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

from rsseg.gui import theme
from rsseg.gui.core import help as helptext
from rsseg.gui.core.gpu import GPUSample
from rsseg.gui.core.timing import JobView, fmt_clock, fmt_duration, fmt_range

STATE_TONES = {
    "queued": "neutral",
    "running": "accent",
    "pausing": "warning",
    "stopping": "warning",
    "done": "success",
    "failed": "danger",
    "paused": "warning",
    "stopped": "neutral",
    "interrupted": "danger",
}
LEVEL_TONES = {"ready": "success", "warning": "warning", "absent": "danger"}


def tip(widget: QWidget, section: str, key: str, **kw) -> QWidget:
    """Attach the Italian tooltip ``section/key`` to ``widget`` (returns it)."""
    widget.setToolTip(helptext.tooltip(section, key, **kw))
    return widget


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def action_button(text: str, icon: str, role: str | None = None) -> QPushButton:
    """A button with a drawn icon; ``role`` is ``"primary"``, ``"danger"`` or ``"quiet"``."""
    button = QPushButton(text)
    if role:
        button.setObjectName(role)
    theme.set_action_icon(button, icon)
    return button


def status_dot(level: str) -> QLabel:
    """``●`` in the fill colour of a GPU level (``ready``, ``warning``, ``absent``)."""
    dot = QLabel("●")
    dot.setObjectName("statusDot")
    theme.set_tone(dot, LEVEL_TONES.get(level, "neutral"))
    return dot


class StatusPill(QLabel):
    """A job state as a rounded, tinted label (``in esecuzione``, ``in pausa``, …)."""

    def __init__(self, text: str = "", tone: str = "neutral"):
        super().__init__()
        self.setObjectName("statusPill")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.set(text, tone)

    def set(self, text: str, tone: str) -> None:
        self.setText(text)
        theme.set_tone(self, tone)


# --------------------------------------------------------------------------- progress
class PhaseStrip(QWidget):
    """``✔ Avvio → ● Training → ○ Valutazione → ○ Report``."""

    TONES = {"done": "success", "active": "accent", "todo": "neutral", "error": "danger"}
    SYMBOLS = {"done": "✔", "active": "●", "todo": "○", "error": "✖"}

    def __init__(self):
        super().__init__()
        self.label = QLabel()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.phases: list[tuple[str, str]] = []
        self.setToolTip("Fasi del processo: ✔ completata, ● in corso, ○ da fare, ✖ interrotta")

    def set_phases(self, phases: list[tuple[str, str]]) -> None:
        self.phases = list(phases)
        parts = [theme.span(f"{self.SYMBOLS[s]} {name}", self.TONES[s], bold=s == "active")
                 for name, s in phases]  # fmt: skip
        self.label.setText(theme.span("  →  ", "neutral").join(parts))

    def retheme(self) -> None:
        self.set_phases(self.phases)


class JobProgressWidget(QFrame):
    """Phases, progress bars, elapsed/remaining time and the buttons of one job."""

    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, compact: bool = False):
        super().__init__()
        self.setObjectName("card")
        self.compact = compact
        v = QVBoxLayout(self)
        v.setContentsMargins(theme.Space.XL, theme.Space.LG, theme.Space.XL, theme.Space.LG)
        v.setSpacing(theme.Space.MD)
        head = QHBoxLayout()
        head.setSpacing(theme.Space.MD)
        self.title = QLabel()
        self.title.setTextFormat(Qt.RichText)
        self.pill = StatusPill()
        head.addWidget(self.title)
        head.addWidget(self.pill)
        head.addStretch(1)
        self.pause_btn = tip(action_button("Pausa", "pause"), "options", "pause")
        self.resume_btn = tip(action_button("Riprendi", "play", "primary"), "options", "resume")
        self.stop_btn = tip(action_button("Stop", "stop", "danger"), "options", "stop")
        for b, sig in ((self.pause_btn, self.pause_clicked), (self.resume_btn, self.resume_clicked),
                       (self.stop_btn, self.stop_clicked)):  # fmt: skip
            b.clicked.connect(lambda *_, s=sig: s.emit())
            head.addWidget(b)
        v.addLayout(head)
        self.phases = PhaseStrip()
        v.addWidget(self.phases)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(theme.Space.LG)
        self.grid.setVerticalSpacing(theme.Space.SM)
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
        self.gpu.setObjectName("hint")
        self.gpu.setToolTip(helptext.text("gpu", "util"))
        self.notes = QLabel()
        self.notes.setObjectName("detail")
        self.notes.setWordWrap(True)
        theme.set_tone(self.notes, "neutral")
        v.addWidget(self.times)
        v.addWidget(self.gpu)
        v.addWidget(self.notes)
        self._last: tuple = (None, None, None)
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

    def retheme(self) -> None:
        self.set_view(*self._last)

    def set_view(self, view: JobView | None, job=None, sample: GPUSample | None = None) -> None:
        self._last = (view, job, sample)
        for label, bar in self.bars:
            label.hide()
            bar.hide()
        if view is None:
            self.title.setText("<i>Nessun job selezionato</i>")
            self.phases.hide()
            self.total_label.hide()
            self.total.hide()
            for w in (self.pill, self.times, self.gpu, self.notes, self.pause_btn,
                      self.resume_btn, self.stop_btn):  # fmt: skip
                w.hide()
            return
        self.phases.show()
        self.title.setText(f"<b>{view.title}</b>")
        self.pill.set(view.state_label, STATE_TONES.get(view.state, "neutral"))
        self.pill.show()
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
            theme.set_tone(bar, {"done": "success", "failed": "danger", "interrupted": "danger",
                                 "paused": "warning"}.get(view.state))  # fmt: skip
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
        self.times.setText(theme.span(" · ", "neutral").join(parts))
        self.times.show()
        if sample is not None and running and job is not None and job.uses_gpu:
            self.gpu.setText(gpu_line(sample))
            self.gpu.show()
        else:
            self.gpu.hide()
        self.notes.setText("<br>".join(view.notes))
        self.notes.setVisible(bool(view.notes))
        theme.set_tone(self.notes, "warning" if view.stale else "neutral")
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
        h.setContentsMargins(theme.Space.SM, 0, theme.Space.SM, 0)
        self.dot = status_dot("warning")
        self.text = QLabel("GPU: rilevamento…")
        self.mem = QProgressBar()
        self.mem.setMaximumWidth(120)
        self.mem.setMaximumHeight(10)
        self.mem.setStyleSheet("min-height: 0px;")
        self.mem.setRange(0, 1000)
        self.mem.setTextVisible(False)
        tip(self.mem, "gpu", "memory")
        for w in (self.dot, self.text, self.mem):
            h.addWidget(w)
        self.level = "warning"
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):  # noqa: N802 (Qt API)
        self.clicked.emit()

    def update_status(self, level: str, message: str, sample: GPUSample | None) -> None:
        self.level = level
        theme.set_tone(self.dot, LEVEL_TONES.get(level, "neutral"))
        if sample is None:
            self.text.setText("GPU non rilevata" if level == "absent" else "GPU")
            self.mem.hide()
        else:
            name = sample.name.replace("NVIDIA ", "").replace("GeForce ", "")
            self.text.setText(f"{name} · {gpu_line(sample)}")
            if sample.mem_fraction is not None:
                self.mem.show()
                self.mem.setValue(int(sample.mem_fraction * 1000))
                theme.set_tone(self.mem, "danger" if sample.mem_fraction > 0.95 else None)
        self.setToolTip(f"<qt>{message.replace(chr(10), '<br>')}<br><br>"
                        "Clic per aprire la sezione GPU.</qt>")  # fmt: skip


# ------------------------------------------------------------------------------ charts
def _style_plot(w) -> None:
    """Background, axes, grid, title and legend of a pyqtgraph plot in the current theme."""
    import pyqtgraph as pg

    p = theme.palette()
    w.setBackground(p.surface)
    axis_pen = pg.mkPen(p.border_strong, width=1)
    for name in ("left", "bottom"):
        axis = w.getAxis(name)
        axis.setPen(axis_pen)
        axis.setTextPen(pg.mkPen(p.text_muted))
        axis.setGrid(40 if p.is_dark else 60)  # recessive grid: alpha out of 255
        axis.setZValue(-1000)  # the grid the axis draws goes behind the data, not over it
    w.setTitle(w.property("chartTitle") or None, color=p.text, size="11pt")
    legend = w.getPlotItem().legend
    if legend is not None:
        legend.setLabelTextColor(p.text_muted)
        legend.setBrush(pg.mkBrush(QColor(p.surface)))
        legend.setPen(pg.mkPen(p.border))
    w.setProperty("seriesTheme", p.name)


def plot_widget(title: str = "", left: str = "", bottom: str = ""):
    """A pyqtgraph plot styled by the theme (and restyled when the theme changes)."""
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True)
    w = pg.PlotWidget()
    w.setProperty("chartTitle", title)
    if left:
        w.setLabel("left", left)
    if bottom:
        w.setLabel("bottom", bottom)
    w.addLegend(offset=(10, 10)).setVisible(False)  # shown when there are two series or more
    _style_plot(w)
    w.retheme = lambda: _restyle(w)
    return w


def _restyle(w) -> None:
    _style_plot(w)
    redraw = getattr(w, "redraw", None)
    if callable(redraw):
        redraw()


def save_plot_png(widget, path: str | Path) -> None:
    """Save a chart as PNG in the light theme (for documents), then restore the theme on screen."""
    with theme.using(theme.LIGHT):
        _restyle(widget)
        widget.grab().save(str(path))
    _restyle(widget)


def set_plot_title(widget, title: str) -> None:
    widget.setProperty("chartTitle", title)
    widget.setTitle(title or None, color=theme.palette().text, size="11pt")


def plot_series(widget, series: dict[str, tuple[list, list]], markers: list[float] | None = None):
    """Replace the curves of ``widget``: ``{name: (x, y)}``; ``markers`` draw vertical lines.

    Series keep their colour slot by position (training first, validation second, …), so the
    same quantity has the same colour in every chart.
    """
    import pyqtgraph as pg

    widget.redraw = lambda: plot_series(widget, series, markers)
    widget.clear()
    legend = widget.getPlotItem().legend
    if legend is not None:  # one series: the title names it, a legend box would only repeat it
        legend.setVisible(len(series) > 1)
    colors = theme.series_colors()
    surface = theme.palette().surface
    for i, (name, (x, y)) in enumerate(series.items()):
        pts = [(a, b) for a, b in zip(x, y, strict=False) if b is not None and b == b]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        color = colors[i % len(colors)]
        pen = pg.mkPen(color, width=2)
        widget.plot(list(xs), list(ys), pen=pen, name=name, symbol="o" if len(xs) < 60 else None,
                    symbolSize=7, symbolBrush=color,
                    symbolPen=pg.mkPen(surface, width=2))  # fmt: skip
    muted = theme.palette().text_muted
    for m in markers or []:
        widget.addItem(pg.InfiniteLine(pos=m, angle=90, pen=pg.mkPen(muted, style=Qt.DashLine),
                                       label="ripresa", labelOpts={"position": 0.9,
                                                                   "color": muted}))  # fmt: skip


def bar_chart(widget, labels: list[str], values: list[float], colors: list[str] | None = None):
    """Bars with the colours given (e.g. the class colours) or the first series colour."""
    import pyqtgraph as pg

    widget.redraw = lambda: bar_chart(widget, labels, values, colors)
    widget.clear()
    widget.getAxis("bottom").setGrid(False)  # categories: only the value lines help
    if widget.getPlotItem().legend is not None:
        widget.getPlotItem().legend.setVisible(False)  # the axis labels name the bars
    xs = list(range(len(values)))
    fills = colors or [theme.series_colors()[0]] * len(values)
    brushes = [pg.mkBrush(c) for c in fills]
    widget.addItem(pg.BarGraphItem(x=xs, height=[v or 0 for v in values], width=0.7,
                                   brushes=brushes,
                                   pen=pg.mkPen(theme.palette().surface, width=2)))  # fmt: skip
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
        self.retheme()
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

    def retheme(self) -> None:
        self.setBackgroundBrush(QColor(theme.palette().surface_alt))

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
    """Hex colours of the classes for charts.

    The ISPRS colours are data and do not follow the theme; only white (impervious surfaces)
    is drawn light grey on the light theme, where it would vanish into the chart surface.
    """
    from rsseg.data.classes import PALETTE

    colors = ["#{:02x}{:02x}{:02x}".format(*(int(x) for x in rgb)) for rgb in PALETTE]
    white = "#ffffff" if theme.palette().is_dark else "#bbbbbb"
    return [c if c != "#ffffff" else white for c in colors]


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
        self.rgb = [tuple(int(x) for x in PALETTE[i]) for i in range(len(CLASS_NAMES))]
        for i, name in enumerate(CLASS_NAMES):
            r, g, b = self.rgb[i]
            box = QCheckBox(CLASS_LABELS_IT[name])
            box.setChecked(True)
            box.setToolTip(f"Mostra o nascondi la classe «{CLASS_LABELS_IT[name]}» "
                           f"(colore ISPRS rgb({r},{g},{b}))")  # fmt: skip
            box.toggled.connect(lambda *_: self.changed.emit())
            h.addWidget(box)
            self.boxes.append(box)
        h.addStretch(1)
        self.retheme()

    def retheme(self) -> None:
        """Each indicator filled with its class colour, the tick in black or white for contrast."""
        border = theme.palette().border_strong
        for box, (r, g, b) in zip(self.boxes, self.rgb, strict=True):
            ink = "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#ffffff"
            box.setStyleSheet(
                f"QCheckBox::indicator {{ border: 2px solid rgb({r},{g},{b}); }}"
                f"QCheckBox::indicator:checked {{ background: rgb({r},{g},{b}); "
                f'border: 1px solid {border}; image: url("{theme.checkmark_file(ink)}"); }}'
            )

    def enabled(self) -> set[int]:
        return {i for i, b in enumerate(self.boxes) if b.isChecked()}
