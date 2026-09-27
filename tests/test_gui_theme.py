# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guards for the design tokens of the desktop app, and the live switch between the themes.

The contrast checks are the point of this module. Colour choices decay silently: someone nudges
a green to look nicer against a card and, months later, nobody can read a status on a laptop in
daylight. WCAG gives a number, so the number is asserted.
"""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rsseg.gui import theme  # noqa: E402
from rsseg.gui.theme import DARK, LIGHT, TONES  # noqa: E402

PALETTES = [DARK, LIGHT]
IDS = [p.name for p in PALETTES]


def _channel(value: int) -> float:
    srgb = value / 255
    return srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "rsseg-gui.ini"), QSettings.IniFormat)


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_body_text_clears_aa_on_every_surface(palette):
    for surface in (palette.window, palette.surface, palette.surface_alt):
        assert contrast(palette.text, surface) >= 4.5, surface


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_muted_text_clears_aa_on_surfaces(palette):
    for surface in (palette.window, palette.surface, palette.surface_alt):
        assert contrast(palette.text_muted, surface) >= 4.5, surface


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_status_text_clears_aa_on_its_own_tint(palette):
    """A pill draws its tone's text on its tone's tint; that pairing must pass."""
    for tone in TONES:
        foreground, background = theme.tone_colors(palette, tone)
        assert contrast(foreground, background) >= 4.5, f"{palette.name}/{tone}"


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_status_text_clears_aa_on_the_plain_surface(palette):
    """Tone colours are also used as plain text, with no tint behind."""
    for tone in TONES:
        foreground, _ = theme.tone_colors(palette, tone)
        assert contrast(foreground, palette.surface) >= 4.5, f"{palette.name}/{tone}"


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_primary_button_text_clears_aa(palette):
    assert contrast(palette.accent_text, palette.accent) >= 4.5


@pytest.mark.parametrize("palette", PALETTES, ids=IDS)
def test_stylesheet_is_complete(palette):
    qss = theme.build_stylesheet(palette)
    assert "{" in qss and qss.count("{") == qss.count("}")
    assert palette.window in qss and palette.accent in qss
    assert "QListWidget#nav" in qss and 'QLabel#statusPill[tone="danger"]' in qss


def test_the_app_starts_dark_and_remembers_the_light_theme(settings):
    assert theme.saved_palette(settings) is DARK
    settings.setValue(theme.SETTINGS_KEY, "sepia")  # unknown values fall back to dark
    assert theme.saved_palette(settings) is DARK
    theme.save_palette(LIGHT, settings)
    assert theme.saved_palette(settings) is LIGHT


def test_icons_are_drawn_for_both_themes(qapp):
    for palette in PALETTES:
        theme.apply_theme(qapp, palette)
        assert not theme.app_icon().pixmap(32, 32).isNull()
        for name in theme.ACTION_ICONS:
            assert not theme.action_icon(name).pixmap(20, 20).isNull(), name
    theme.apply_theme(qapp, DARK)


def test_main_window_switches_theme_live(qapp, tmp_path, settings):
    from rsseg.gui.app import PAGES, MainWindow
    from rsseg.gui.context import AppContext

    theme.apply_theme(qapp, DARK)
    ctx = AppContext(tmp_path, gpu=False, tick_ms=100000)
    window = MainWindow(ctx, settings)
    window.show()
    assert window.theme_button.text() == "Tema chiaro"
    assert DARK.window in qapp.styleSheet()

    window.toggle_theme()
    assert theme.palette() is LIGHT and LIGHT.window in qapp.styleSheet()
    assert window.theme_button.text() == "Tema scuro"
    assert theme.saved_palette(settings) is LIGHT
    for key, _, _ in PAGES:  # every page still builds and paints in the other theme
        window.show_page(key)
        qapp.processEvents()
        assert window.grab().width() > 0

    window.toggle_theme()
    assert theme.palette() is DARK and theme.saved_palette(settings) is DARK
    ctx.shutdown()
    window.close()
