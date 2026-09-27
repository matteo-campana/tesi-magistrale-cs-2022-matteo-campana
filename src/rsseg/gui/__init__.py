# SPDX-License-Identifier: AGPL-3.0-or-later
"""Desktop application (Qt / PySide6) to prepare data, train, monitor, resume and run inference.

Start it from the project folder with ``rsseg-gui`` (or ``python -m rsseg.gui``). It needs the
optional dependencies: ``pip install -e ".[gui]"``.

The application never imports torch or keras: every heavy command runs as a separate
``rsseg`` process (:mod:`rsseg.gui.core.jobs`), followed through ``progress.json`` and steered
through ``control.json``. ``rsseg.gui.core`` holds the parts without Qt (tested headless).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsseg-gui", description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="project folder (default: current folder)")
    args, qt_args = parser.parse_known_args(argv)
    try:
        from rsseg.gui.app import run_app
    except ImportError as e:  # pragma: no cover - missing optional dependency
        print(f"rsseg-gui needs the GUI dependencies: pip install -e \".[gui]\"\n({e})",
              file=sys.stderr)  # fmt: skip
        return 1
    return run_app(args.root, [sys.argv[0], *qt_args])
