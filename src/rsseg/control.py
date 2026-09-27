# SPDX-License-Identifier: AGPL-3.0-or-later
"""Requests sent to a running command (pause, stop) through a small JSON file.

Whoever launches a command (the Qt application, ``rsseg campaign``) sets ``RSSEG_CONTROL_FILE``
and later writes ``{"action": "pause"}`` or ``{"action": "stop"}`` to that file with
:func:`request`. Training, DINO, evaluation and prediction call :func:`pending` between batches
(or tiles): the check is a ``stat`` of the file, cheap enough for every batch. A file is used
instead of a signal because it behaves the same on Windows and Linux.

* **pause**: save a resume checkpoint and exit with :data:`EXIT_PAUSED`; ``--resume RUN_DIR``
  continues exactly where it stopped;
* **stop**: exit with :data:`EXIT_STOPPED` without a resume checkpoint (the top-k checkpoints of
  the finished epochs stay).

This module does not import torch or keras: the GUI uses it too.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ENV = "RSSEG_CONTROL_FILE"
FILE_NAME = "control.json"
ACTIONS = ("pause", "stop")

EXIT_PAUSED = 10
"""Exit code of a command paused on request (it can be resumed with ``--resume``)."""
EXIT_STOPPED = 11
"""Exit code of a command stopped on request."""


class ControlRequest(Exception):  # noqa: N818 (a request, not an error)
    """Raised inside a command to unwind it after a pause or stop request."""

    action = ""

    def __init__(self, message: str = "", run_dir: str | Path | None = None):
        super().__init__(message or self.action)
        self.run_dir = run_dir


class PauseRequested(ControlRequest):
    action = "pause"


class StopRequested(ControlRequest):
    action = "stop"


def request(path: str | Path, action: str) -> None:
    """Ask the command that watches ``path`` to ``pause`` or ``stop`` (atomic write)."""
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"action": action, "time": time.time()}), encoding="utf-8")
    os.replace(tmp, path)


def clear(path: str | Path | None = None) -> None:
    """Remove a request (done by a command when it starts, so a stale file is not obeyed)."""
    path = path or os.environ.get(ENV)
    if path:
        Path(path).unlink(missing_ok=True)


def pending() -> str | None:
    """The action requested to this process (``pause``/``stop``), or ``None``."""
    path = os.environ.get(ENV)
    if not path or not os.path.exists(path):
        return None
    try:
        action = json.loads(Path(path).read_text(encoding="utf-8")).get("action")
    except (OSError, ValueError, AttributeError):
        return None  # half-written by another process: seen at the next check
    return action if action in ACTIONS else None


def check(pausable: bool = True) -> None:
    """Raise :class:`PauseRequested` or :class:`StopRequested` if one was requested.

    Where no resume checkpoint can be written (``pausable=False``, e.g. data preparation or
    prediction) a pause behaves like a stop.
    """
    action = pending()
    if action == "pause" and pausable:
        raise PauseRequested()
    if action in ("pause", "stop"):
        raise StopRequested()
