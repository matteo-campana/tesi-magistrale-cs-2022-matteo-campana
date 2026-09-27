# SPDX-License-Identifier: AGPL-3.0-or-later
"""Design tokens and the application stylesheet (dark by default, light on request).

Adapted from the ``theme.py`` of the author's video-downloader, so that the two desktop apps
look alike. Everything visual in the GUI resolves to a name defined here: spacing comes from
`Space`, radii from `Radius`, and every colour from a `Palette` field whose name says what the
colour *means* ("danger") rather than what it looks like ("red"). `DARK` and `LIGHT` answer the
same questions with different values, so no call site has to know which one is in force.

Two rules are worth keeping when extending this module:

1. Colours used as *text* must clear 4.5:1 against the surface they sit on (WCAG AA). The
   ``*_text`` variants exist because the vivid fill colours do not: #22C55E is a fine
   progress-bar fill and an illegible label. ``tests/test_gui_theme.py`` asserts it.
2. Sizes that must track the user's font belong in `em()`, not in a pixel constant. Pixel
   constants here are for padding and borders, which do not need to scale with text.

The class colours of the segmentation maps (`rsseg.data.classes.PALETTE`) are data, not theme:
they never change with the theme, and neither do the exported figures.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget

# ── spacing / radius scale ────────────────────────────────────────────────────────────────
#
# A 4px base step. Using a scale rather than ad-hoc numbers is what produces a visible grouping
# rhythm: XL between sections, MD inside one, XS between a label and the control it names.


class Space:
    XS = 2
    SM = 4
    MD = 8
    LG = 12
    XL = 16
    XXL = 24


class Radius:
    SM = 4
    MD = 6
    LG = 10
    PILL = 999


# ── palette ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Palette:
    """Semantic colours for one colour scheme."""

    name: str

    # Surfaces, from furthest back to furthest forward.
    window: str  # the window background
    surface: str  # cards, panels, list rows
    surface_alt: str  # table headers, inset areas, hover rows
    border: str
    border_strong: str

    # Text.
    text: str
    text_muted: str
    text_disabled: str

    # Accent (the single primary-action colour).
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str  # text drawn *on* accent
    accent_soft: str  # tinted background for selections

    # Status. `*_text` clears 4.5:1 on `surface`; the plain value is a fill.
    success: str
    success_text: str
    success_soft: str
    warning: str
    warning_text: str
    warning_soft: str
    danger: str
    danger_text: str
    danger_soft: str
    info: str
    info_text: str
    info_soft: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT = Palette(
    name="light",
    window="#F4F5F7",
    surface="#FFFFFF",
    surface_alt="#EFF1F4",
    border="#DFE2E7",
    border_strong="#C4C9D1",
    text="#171A1F",
    text_muted="#5C6370",
    text_disabled="#A2A8B3",
    accent="#4F46E5",
    accent_hover="#443CD4",
    accent_pressed="#3A33BC",
    accent_text="#FFFFFF",
    accent_soft="#E8E7FB",
    success="#22C55E",
    success_text="#15803D",
    success_soft="#DCFCE7",
    warning="#F59E0B",
    warning_text="#B45309",
    warning_soft="#FEF3C7",
    danger="#EF4444",
    danger_text="#B91C1C",
    danger_soft="#FEE2E2",
    info="#3B82F6",
    info_text="#1D4ED8",
    info_soft="#DBEAFE",
)

DARK = Palette(
    name="dark",
    window="#14161A",
    surface="#1D2026",
    surface_alt="#262A32",
    border="#2F343D",
    border_strong="#424957",
    text="#E7E9ED",
    text_muted="#9BA2AE",
    text_disabled="#646B77",
    accent="#8E8DF3",
    accent_hover="#A3A2F6",
    accent_pressed="#7A79EC",
    # Dark ink on a bright accent. The reverse (white on a mid indigo) cannot clear 4.5:1 here
    # without an accent so dark it disappears into the surface.
    accent_text="#14161A",
    accent_soft="#1E1E3C",
    success="#22C55E",
    success_text="#4ADE80",
    success_soft="#12331F",
    warning="#F59E0B",
    warning_text="#FBBF24",
    warning_soft="#3A2C0B",
    danger="#EF4444",
    danger_text="#F87171",
    danger_soft="#3B1718",
    info="#3B82F6",
    info_text="#93C5FD",
    info_soft="#152540",
)

PALETTES = {"dark": DARK, "light": LIGHT}
DEFAULT = DARK
"""The app always starts dark unless the user picked the light theme."""

# Tone name -> (text colour attr, soft background attr). Used by the status pills and the
# inline banners so a "warning" looks the same wherever it is.
TONES = {
    "neutral": ("text_muted", "surface_alt"),
    "accent": ("accent", "accent_soft"),
    "success": ("success_text", "success_soft"),
    "warning": ("warning_text", "warning_soft"),
    "danger": ("danger_text", "danger_soft"),
    "info": ("info_text", "info_soft"),
}


def tone_colors(palette: Palette, tone: str) -> tuple[str, str]:
    """(foreground, soft background) for a tone name."""
    fg_attr, bg_attr = TONES.get(tone, TONES["neutral"])
    return getattr(palette, fg_attr), getattr(palette, bg_attr)


# Categorical colours for chart series: the reference palette of the data-viz method, eight hues
# in a fixed order, stepped separately for each surface. Validated with its
# ``validate_palette.js`` against `surface`: lightness band, chroma, colour-blind separation
# of adjacent series (ΔE ≥ 8) and normal vision (ΔE ≥ 19). On white three slots fall below
# 3:1, which is allowed because every chart has a legend. Status colours are not reused here.
SERIES = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}  # fmt: skip

# Tone name -> fill attribute, for marks rather than text: progress chunks, dots, heatmap cells.
FILLS = {
    "neutral": "border_strong",
    "accent": "accent",
    "success": "success",
    "warning": "warning",
    "danger": "danger",
    "info": "info",
}


# ── current palette and the saved choice ──────────────────────────────────────────────────

_current: Palette = DEFAULT
_listeners: list[Callable[[Palette], None]] = []

SETTINGS_KEY = "theme"


def palette() -> Palette:
    """The palette currently in force."""
    return _current


@contextmanager
def using(p: Palette) -> Iterator[Palette]:
    """Draw with `p` for a moment without restyling the app: exported charts are always light,
    whatever the theme on screen, so they can go straight into the thesis."""
    global _current
    previous, _current = _current, p
    try:
        yield p
    finally:
        _current = previous


def saved_palette(settings: QSettings | None = None) -> Palette:
    """The theme chosen last time, dark when nothing (or something unknown) was saved."""
    settings = settings or QSettings("rsseg", "rsseg-gui")
    return PALETTES.get(str(settings.value(SETTINGS_KEY, DEFAULT.name)), DEFAULT)


def save_palette(p: Palette, settings: QSettings | None = None) -> None:
    settings = settings or QSettings("rsseg", "rsseg-gui")
    settings.setValue(SETTINGS_KEY, p.name)


def series_colors() -> tuple[str, ...]:
    """The chart series colours for the current palette, in their fixed order."""
    return SERIES[_current.name]


def fg(tone: str) -> str:
    """Text colour of a tone in the current palette (clears 4.5:1 on `surface`)."""
    return tone_colors(_current, tone)[0]


def fill(tone: str) -> str:
    """Fill colour of a tone in the current palette, for marks that are not text."""
    return getattr(_current, FILLS.get(tone, "border_strong"))


def bar_chunk(p: Palette, tone: str = "accent") -> str:
    """The filled part of a progress bar with text on it.

    A progress bar prints its text across both the filled and the empty part, so the fill is
    the tone's colour softened towards the groove (dark) or white (light) until `text` clears
    4.5:1 on it as well; the vivid fill would leave the percentage unreadable.
    """
    base = getattr(p, FILLS.get(tone, "accent"))
    return mix(base, p.surface_alt, 0.55) if p.is_dark else mix(base, "#FFFFFF", 0.3)


def mix(a: str, b: str, t: float) -> str:
    """``a`` blended towards ``b`` by ``t`` (0…1), as hex."""
    ca, cb = QColor(a), QColor(b)
    t = min(max(t, 0.0), 1.0)
    return QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    ).name()


def span(text: str, tone: str, bold: bool = False) -> str:
    """``text`` as rich text in the colour of ``tone``."""
    tag = "b" if bold else "span"
    return f"<{tag} style='color:{fg(tone)}'>{text}</{tag}>"


def set_tone(widget: QWidget, tone: str | None) -> None:
    """Set the dynamic ``tone`` property that the stylesheet keys on, and re-polish."""
    if widget.property("tone") == tone:
        return
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_action_icon(button: QAbstractButton, name: str) -> None:
    """Give ``button`` the drawn icon ``name``; it is redrawn when the theme changes."""
    button.setProperty("actionIcon", name)
    _paint_action_icon(button)


def _paint_action_icon(button: QAbstractButton) -> None:
    role = button.objectName()
    ink = {"primary": _current.accent_text, "danger": _current.danger_text}.get(role)
    button.setIcon(action_icon(button.property("actionIcon"), ink))


def retheme_widgets() -> None:
    """Refresh what the stylesheet cannot reach, in every open window.

    Widgets and pages with colours baked into rich text, brushes or pyqtgraph items define a
    ``retheme()`` method; buttons with a drawn icon get it redrawn in the new colour.
    """
    app = QApplication.instance()
    if app is None:
        return
    for top in app.topLevelWidgets():
        retheme_tree(top)


def retheme_tree(root: QWidget) -> None:
    """`retheme_widgets` for one widget and its children."""
    for widget in (root, *root.findChildren(QWidget)):
        if isinstance(widget, QAbstractButton) and widget.property("actionIcon"):
            _paint_action_icon(widget)
        retheme = getattr(widget, "retheme", None)
        if callable(retheme):
            retheme()


def grab_light(widget: QWidget) -> QPixmap:
    """An image of `widget` drawn in the light theme, whatever the theme on screen.

    Exported views and charts go into documents (the thesis), so they are always light. The
    light stylesheet is set on the widget only for the grab, which happens before the event loop
    can repaint the screen, and the image is laid on the light window colour because the widget
    itself is transparent.
    """
    own = widget.styleSheet()
    with using(LIGHT):
        widget.setStyleSheet(build_stylesheet(LIGHT) + own)
        retheme_tree(widget)
        shot = widget.grab()
    widget.setStyleSheet(own)
    retheme_tree(widget)
    image = QPixmap(shot.size())
    image.setDevicePixelRatio(shot.devicePixelRatio())
    image.fill(QColor(LIGHT.window))
    painter = QPainter(image)
    painter.drawPixmap(0, 0, shot)
    painter.end()
    return image


def on_palette_changed(callback: Callable[[Palette], None]) -> None:
    """Register a callback for live theme switches.

    Widgets that paint themselves (charts, phase strips, rich-text badges) need this: the
    stylesheet reaches everything else, but a colour baked into a QPainter call, a QBrush or an
    HTML string does not.
    """
    _listeners.append(callback)


def apply_theme(app, p: Palette | None = None) -> Palette:
    """Install the stylesheet of `p` (default: the saved choice) on `app` and return it."""
    global _current
    _current = p if p is not None else saved_palette()
    app.setStyle("Fusion")  # the QSS below assumes Fusion's metrics
    app.setStyleSheet(build_stylesheet(_current))
    retheme_widgets()
    for callback in list(_listeners):
        try:
            callback(_current)
        except RuntimeError:
            # A listener whose widget has been destroyed; drop it.
            _listeners.remove(callback)
    return _current


# ── font-relative sizing ──────────────────────────────────────────────────────────────────


def em(widget: QWidget, characters: float) -> int:
    """Width of `characters` digit-widths in `widget`'s font.

    Fixed pixel widths break at high DPI and at large system font sizes; this is the unit to
    reach for when a control genuinely needs a fixed width.
    """
    return int(QFontMetrics(widget.font()).horizontalAdvance("0") * characters)


def line_height(widget: QWidget, lines: float = 1.0) -> int:
    return int(QFontMetrics(widget.font()).height() * lines)


def icon_size(widget: QWidget) -> QSize:
    side = line_height(widget)
    return QSize(side, side)


# ── application icon ──────────────────────────────────────────────────────────────────────

# Opacity of each cell of the 3×3 grid: a few regions of a segmentation map, not a checkerboard.
_ICON_CELLS = (1.0, 1.0, 0.55, 1.0, 0.55, 0.55, 0.3, 0.3, 0.55)


def app_icon(size: int = 256) -> QIcon:
    """A 3×3 grid of cells (a segmented tile) on a rounded accent square, at several sizes.

    Generated rather than shipped so the icon follows the accent colour and there is no binary
    asset to keep in sync with the palette.
    """
    icon = QIcon()
    for px_size in (16, 24, 32, 48, 64, 128, size):
        icon.addPixmap(_icon_pixmap(px_size))
    return icon


def _icon_pixmap(size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = size / 32.0
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_current.accent))
    painter.drawRoundedRect(QRectF(0, 0, size, size), 7 * unit, 7 * unit)
    ink = QColor(_current.accent_text)
    cell, gap, origin = 6 * unit, 1.5 * unit, 5.75 * unit
    for i, alpha in enumerate(_ICON_CELLS):
        ink.setAlphaF(alpha)
        painter.setBrush(ink)
        row, col = divmod(i, 3)
        x, y = origin + col * (cell + gap), origin + row * (cell + gap)
        painter.drawRoundedRect(QRectF(x, y, cell, cell), 1.2 * unit, 1.2 * unit)
    painter.end()
    return pixmap


# ── action icons ──────────────────────────────────────────────────────────────────────────
#
# Drawn rather than taken from QStyle.standardIcon or a font glyph. Style icons are not a set:
# Fusion draws SP_BrowserReload as a blue circular arrow next to a flat monochrome
# SP_MediaPause, so a row of them reads as clip-art. Font glyphs have the worse problem of not
# being there at all on some systems. These are shapes on one grid in one colour, and they
# follow the palette.

_ICON_GRID = 64.0
ACTION_ICONS = ("play", "pause", "stop", "folder", "retry", "moon", "sun")


def action_icon(name: str, color: str | None = None) -> QIcon:
    """A monochrome icon for an action, in the current palette's colour."""
    ink = QColor(color or _current.text_muted)
    icon = QIcon()
    for size in (16, 20, 24, 32, 48):
        icon.addPixmap(_draw_action(name, size, ink))
    return icon


def _draw_action(name: str, size: int, ink: QColor) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = size / _ICON_GRID
    painter.scale(unit, unit)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink)

    if name == "play":
        painter.drawPolygon(QPolygonF([QPointF(22, 15), QPointF(49, 32), QPointF(22, 49)]))
    elif name == "pause":
        painter.drawRoundedRect(QRectF(21, 16, 8, 32), 2, 2)
        painter.drawRoundedRect(QRectF(35, 16, 8, 32), 2, 2)
    elif name == "stop":
        painter.drawRoundedRect(QRectF(19, 19, 26, 26), 3, 3)
    elif name == "folder":
        painter.drawRoundedRect(QRectF(9, 17, 21, 9), 2, 2)
        painter.drawRoundedRect(QRectF(9, 22, 46, 26), 3, 3)
    elif name == "retry":
        pen = QPen(ink, 7)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Open at the top right; the arrowhead closes the loop there.
        painter.drawArc(QRectF(15, 15, 34, 34), 55 * 16, 290 * 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ink)
        painter.drawPolygon(QPolygonF([QPointF(26, 14), QPointF(46, 12), QPointF(38, 30)]))
    elif name == "moon":
        painter.drawEllipse(QPointF(30, 34), 20, 20)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.drawEllipse(QPointF(42, 24), 17, 17)
    elif name == "sun":
        painter.drawEllipse(QPointF(32, 32), 11, 11)
        painter.translate(32, 32)
        for _ in range(8):
            painter.drawRoundedRect(QRectF(-3, -27, 6, 9), 3, 3)
            painter.rotate(45)
    painter.end()
    return pixmap


# ── generated stylesheet assets ───────────────────────────────────────────────────────────
#
# Styling a checkbox indicator replaces the native drawing wholesale, so the style's own tick
# disappears and a checked box becomes an opaque square. QSS can only take an image from a URL,
# so the tick is painted once per colour and cached on disk.

_asset_cache: dict[str, str] = {}


def checkmark_file(color: str) -> str:
    cached = _asset_cache.get(color)
    if cached and Path(cached).exists():
        return cached
    size = 14
    # QImage, not QPixmap: a pixmap needs a live QGuiApplication and aborts the process without
    # one, which would make the stylesheet impossible to build (or test) before the app exists.
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline([QPointF(3, 7.2), QPointF(5.8, 10), QPointF(11, 4.2)])
    painter.end()
    target = Path(tempfile.gettempdir()) / f"rsseg-check-{color.lstrip('#')}.png"
    image.save(str(target), "PNG")
    path = target.as_posix()
    _asset_cache[color] = path
    return path


def _arrow_file(color: str, direction: str) -> str:
    """A small chevron (``up`` or ``down``) for combo and spin boxes.

    The border-triangle trick of the original stylesheet draws a flat bar with some Qt builds,
    so the arrows are painted once per colour and cached on disk, like the checkmark.
    """
    key = f"{color}-{direction}"
    cached = _asset_cache.get(key)
    if cached and Path(cached).exists():
        return cached
    w, h = 10, 6
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    top, bottom = (1.0, 5.0) if direction == "down" else (5.0, 1.0)
    painter.drawPolyline([QPointF(1, top), QPointF(5, bottom), QPointF(9, top)])
    painter.end()
    target = Path(tempfile.gettempdir()) / f"rsseg-{direction}-{color.lstrip('#')}.png"
    image.save(str(target), "PNG")
    path = target.as_posix()
    _asset_cache[key] = path
    return path


# ── stylesheet ────────────────────────────────────────────────────────────────────────────


def build_stylesheet(p: Palette) -> str:
    """The full application stylesheet for one palette.

    Object names carry the intent that Qt's class names cannot: `#primary` on the one
    call-to-action per view, `#card` on a grouped surface, `#danger` on a destructive button,
    `#nav` on the navigation sidebar, `#mono` on command lines and overrides.
    """
    return _base_stylesheet(p) + _rsseg_stylesheet(p) + _tone_stylesheet(p)


def _tone_stylesheet(p: Palette) -> str:
    """Per-tone rules for the widgets that carry a dynamic `tone` property.

    Generating these from `TONES` keeps the pill and the banner honest: adding a tone to the
    palette gives both of them the new variant for free, and neither can drift from the other.
    """
    blocks = []
    for tone in TONES:
        fg, bg = tone_colors(p, tone)
        blocks.append(f"""
QLabel#statusPill[tone="{tone}"] {{
    color: {fg};
    background: {bg};
    border-radius: {Radius.PILL}px;
    padding: {Space.XS}px {Space.MD}px;
    font-size: 11px;
    font-weight: 600;
}}
QFrame#banner[tone="{tone}"] {{
    background: {bg};
    border: 1px solid {fg};
    border-radius: {Radius.MD}px;
}}
QFrame#banner[tone="{tone}"] QLabel {{ color: {fg}; background: transparent; }}
QLabel#detail[tone="{tone}"] {{ color: {fg}; background: transparent; }}
QLabel#statusDot[tone="{tone}"] {{ color: {getattr(p, FILLS[tone])}; }}
""")
    return "\n".join(blocks)


def _rsseg_stylesheet(p: Palette) -> str:
    """Rules for the widgets rsseg-gui has and video-downloader does not."""
    return f"""
/* ── navigation sidebar ───────────────────────────────────────────────── */
QListWidget#nav {{
    background: {p.surface};
    border: none;
    border-right: 1px solid {p.border};
    border-radius: 0px;
    padding: {Space.MD}px {Space.SM}px;
    font-size: 14px;
}}
QListWidget#nav::item {{
    color: {p.text_muted};
    border: none;
    border-radius: {Radius.MD}px;
    padding: {Space.MD}px {Space.LG}px;
    margin: 1px {Space.XS}px;
}}
QListWidget#nav::item:hover {{ background: {p.surface_alt}; color: {p.text}; }}
QListWidget#nav::item:selected {{
    background: {p.accent_soft};
    color: {p.accent};
    font-weight: 600;
}}

/* ── menu bar ─────────────────────────────────────────────────────────── */
QMenuBar {{
    background: {p.surface};
    border-bottom: 1px solid {p.border};
    padding: {Space.XS}px {Space.SM}px;
}}
QMenuBar::item {{
    background: transparent;
    padding: {Space.SM}px {Space.MD + 2}px;
    border-radius: {Radius.SM}px;
}}
QMenuBar::item:selected {{ background: {p.surface_alt}; }}

/* ── spin boxes, sliders, radio buttons ───────────────────────────────── */
QDoubleSpinBox {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.MD}px;
    padding: {Space.SM + 1}px {Space.MD}px;
}}
QDoubleSpinBox:focus {{ border: 2px solid {p.accent}; padding: {Space.SM}px {Space.MD - 1}px; }}
QSlider::groove:horizontal {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {p.accent};
    border: 2px solid {p.surface};
    width: 14px;
    height: 14px;
    margin: -7px 0px;
    border-radius: 9px;
}}
QRadioButton {{ spacing: {Space.MD}px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {p.border_strong};
    border-radius: 8px;
    background: {p.surface};
}}
QRadioButton::indicator:hover {{ border-color: {p.accent}; }}
QRadioButton::indicator:checked {{
    border: 4px solid {p.accent};
    width: 8px; height: 8px;
    background: {p.accent_text};
}}

/* ── read-only text: logs, help, command lines ────────────────────────── */
QTextBrowser, QPlainTextEdit[readOnly="true"] {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {Radius.LG}px;
}}
QLabel#mono {{
    color: {p.text_muted};
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
}}
QLabel#pageTitle {{
    font-size: 18px;
    font-weight: 600;
}}
QLabel#sectionTitle {{
    font-size: 15px;
    font-weight: 600;
    margin-top: {Space.SM}px;
}}
QLabel#statusDot {{ font-size: 16px; }}

/* ── splitters and scroll areas ───────────────────────────────────────── */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {p.border}; }}
QScrollArea {{ border: none; background: transparent; }}
QGraphicsView {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: {Radius.MD}px;
}}

/* ── status bar extras ────────────────────────────────────────────────── */
QPushButton#themeToggle {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {Radius.MD}px;
    padding: {Space.XS}px {Space.MD}px;
    color: {p.text_muted};
}}
QPushButton#themeToggle:hover {{ background: {p.surface_alt}; color: {p.text}; }}
"""


def _base_stylesheet(p: Palette) -> str:
    """The stylesheet of video-downloader; group boxes, arrows and spin boxes adapted."""
    return f"""
/* ── base ─────────────────────────────────────────────────────────────── */
QWidget {{
    color: {p.text};
    background: transparent;
}}
QMainWindow, QDialog, QMessageBox {{
    background: {p.window};
}}
QToolTip {{
    color: {p.text};
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.SM}px;
    padding: {Space.SM}px {Space.MD}px;
}}

/* ── cards: the grouping primitive ────────────────────────────────────── */
QFrame#card {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {Radius.LG}px;
}}
QLabel#cardTitle {{
    color: {p.text_muted};
    font-weight: 600;
}}
QLabel#sectionLabel {{
    color: {p.text_muted};
}}
QLabel#hint {{
    color: {p.text_muted};
}}
QFrame#separator {{
    background: {p.border};
    border: none;
}}

/* ── buttons ──────────────────────────────────────────────────────────── */
QPushButton {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.MD}px;
    padding: {Space.SM + 1}px {Space.LG}px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {p.surface_alt};
    border-color: {p.border_strong};
}}
QPushButton:pressed {{
    background: {p.border};
}}
QPushButton:disabled {{
    color: {p.text_disabled};
    border-color: {p.border};
    background: {p.surface};
}}
QPushButton:focus {{
    border: 2px solid {p.accent};
    padding: {Space.SM}px {Space.LG - 1}px;
}}

QPushButton#primary {{
    background: {p.accent};
    color: {p.accent_text};
    border: 1px solid {p.accent};
    font-weight: 600;
    padding: {Space.MD - 1}px {Space.XL}px;
}}
QPushButton#primary:hover  {{ background: {p.accent_hover};  border-color: {p.accent_hover}; }}
QPushButton#primary:pressed {{ background: {p.accent_pressed}; border-color: {p.accent_pressed}; }}
QPushButton#primary:disabled {{
    background: {p.surface_alt};
    color: {p.text_disabled};
    border-color: {p.border};
}}

/* Quiet buttons: present but visually subordinate to the primary action. */
QPushButton#quiet {{
    background: transparent;
    border-color: transparent;
    color: {p.text_muted};
}}
QPushButton#quiet:hover  {{ background: {p.surface_alt}; color: {p.text}; }}
QPushButton#quiet:disabled {{ color: {p.text_disabled}; background: transparent; }}

QPushButton#danger {{
    background: transparent;
    border-color: transparent;
    color: {p.danger_text};
}}
QPushButton#danger:hover {{ background: {p.danger_soft}; }}
QPushButton#danger:disabled {{ color: {p.text_disabled}; background: transparent; }}

/* ── text inputs ──────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.MD}px;
    padding: {Space.SM + 1}px {Space.MD}px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QComboBox:focus {{
    border: 2px solid {p.accent};
    padding: {Space.SM}px {Space.MD - 1}px;
}}
QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background: {p.surface_alt};
    color: {p.text_disabled};
}}
QLineEdit[invalid="true"], QTextEdit[invalid="true"] {{
    border: 1px solid {p.danger};
    background: {p.danger_soft};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: url("{_arrow_file(p.text_muted, "down")}");
    width: 10px; height: 6px;
    margin-right: {Space.MD}px;
}}
QComboBox QAbstractItemView {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.MD}px;
    selection-background-color: {p.accent_soft};
    selection-color: {p.text};
    outline: none;
    padding: {Space.SM}px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {p.surface_alt};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {p.border}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{_arrow_file(p.text_muted, "up")}");
    width: 10px; height: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{_arrow_file(p.text_muted, "down")}");
    width: 10px; height: 6px;
}}

/* ── checkboxes ───────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: {Space.MD}px;
    color: {p.text};
}}
QCheckBox:disabled {{ color: {p.text_disabled}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {p.border_strong};
    border-radius: {Radius.SM}px;
    background: {p.surface};
}}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked {{
    background: {p.accent};
    border-color: {p.accent};
    image: url("{checkmark_file(p.accent_text)}");
}}
QCheckBox::indicator:checked:disabled {{
    background: {p.text_disabled};
    border-color: {p.text_disabled};
}}
QCheckBox::indicator:disabled {{ background: {p.surface_alt}; }}

/* ── tabs ─────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    top: 0px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.text_muted};
    border: none;
    border-bottom: 2px solid transparent;
    padding: {Space.MD}px {Space.XL}px;
    margin-right: {Space.SM}px;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{
    color: {p.accent};
    border-bottom: 2px solid {p.accent};
    font-weight: 600;
}}
QTabBar::tab:focus {{ border-bottom: 2px solid {p.accent}; }}

/* ── lists and tables ─────────────────────────────────────────────────── */
QListWidget, QTableWidget, QTreeWidget {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {Radius.LG}px;
    outline: none;
    gridline-color: {p.border};
    alternate-background-color: {p.surface_alt};
}}
QListWidget::item {{
    border-bottom: 1px solid {p.border};
    padding: {Space.SM}px {Space.MD}px;
}}
QListWidget::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}
QListWidget::item:hover {{ background: {p.surface_alt}; }}

QTableWidget::item {{
    padding: {Space.SM + 2}px {Space.MD}px;
    border: none;
}}
QTableWidget::item:selected {{
    background: {p.accent_soft};
    color: {p.text};
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: {p.surface_alt};
    color: {p.text_muted};
    border: none;
    border-bottom: 1px solid {p.border};
    border-right: 1px solid {p.border};
    padding: {Space.MD}px;
    font-weight: 600;
}}
QHeaderView::section:last {{ border-right: none; }}
QHeaderView::section:hover {{ color: {p.text}; }}
QTableCornerButton::section {{
    background: {p.surface_alt};
    border: none;
}}

/* ── progress ─────────────────────────────────────────────────────────── */
QProgressBar {{
    background: {p.surface_alt};
    border: none;
    border-radius: {Radius.SM}px;
    text-align: center;
    color: {p.text};
    min-height: 16px;
}}
QProgressBar::chunk {{
    background: {bar_chunk(p)};
    border-radius: {Radius.SM}px;
}}
QProgressBar[tone="success"]::chunk {{ background: {bar_chunk(p, "success")}; }}
QProgressBar[tone="warning"]::chunk {{ background: {bar_chunk(p, "warning")}; }}
QProgressBar[tone="danger"]::chunk  {{ background: {bar_chunk(p, "danger")}; }}
QProgressBar[tone="muted"]::chunk   {{ background: {bar_chunk(p, "neutral")}; }}

/* ── scrollbars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {p.border_strong};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_disabled}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: {p.border_strong};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p.text_disabled}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── menus ────────────────────────────────────────────────────────────── */
QMenu {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: {Radius.MD}px;
    padding: {Space.SM}px;
}}
QMenu::item {{
    padding: {Space.SM + 2}px {Space.XL}px {Space.SM + 2}px {Space.LG}px;
    border-radius: {Radius.SM}px;
}}
QMenu::item:selected {{ background: {p.accent_soft}; }}
QMenu::separator {{
    height: 1px;
    background: {p.border};
    margin: {Space.SM}px {Space.MD}px;
}}

/* ── group boxes: the cards of the pages ──────────────────────────────── */
QGroupBox {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: {Radius.LG}px;
    margin-top: {Space.XL}px;
    padding: {Space.XL}px {Space.LG}px {Space.LG}px {Space.LG}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {Space.LG}px;
    top: {Space.XS}px;
    padding: 0px {Space.SM}px;
    color: {p.text_muted};
}}
/* The title is bold; what the box contains is not. */
QGroupBox > QWidget, QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QRadioButton,
QGroupBox QPushButton, QGroupBox QLineEdit, QGroupBox QComboBox, QGroupBox QSpinBox,
QGroupBox QDoubleSpinBox, QGroupBox QTableWidget, QGroupBox QListWidget {{
    font-weight: normal;
}}

/* ── status bar ───────────────────────────────────────────────────────── */
QStatusBar {{
    background: {p.surface};
    color: {p.text_muted};
    border-top: 1px solid {p.border};
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {p.text_muted}; }}

/* ── empty-state overlay ──────────────────────────────────────────────── */
QLabel#emptyState {{
    color: {p.text_muted};
    background: transparent;
}}
"""
