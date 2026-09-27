# SPDX-License-Identifier: AGPL-3.0-or-later
"""Progress of a running command, shared with ``rsseg campaign`` through a small JSON file.

The campaign sets ``RSSEG_PROGRESS_FILE`` for every step it launches; training, DINO and the
evaluation call :func:`report` to update the file, and the campaign reads it (:func:`read`) to
show the current epoch, the metrics and the tile being evaluated. Without the variable (a plain
``rsseg train``) nothing is written. Progress is best effort: a write that fails is skipped.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ENV = "RSSEG_PROGRESS_FILE"
FILE_NAME = "progress.json"

_state: dict = {}


def report(**fields) -> None:
    """Merge ``fields`` into the progress of this process and write it (if a campaign asks)."""
    path = os.environ.get(ENV)
    if not path:
        return
    _state.update(fields)
    _state["updated"] = time.time()
    tmp = Path(path).with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_state), encoding="utf-8")
        os.replace(tmp, path)  # readers never see a half-written file
    except OSError:  # e.g. the reader holds the file open on Windows: next update wins
        pass


def read(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
