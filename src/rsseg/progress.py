# SPDX-License-Identifier: AGPL-3.0-or-later
"""Progress of a running command, shared through a small JSON file.

Whoever launches a command (``rsseg campaign``, the Qt application) sets ``RSSEG_PROGRESS_FILE``;
training, DINO, evaluation, data preparation and prediction call :func:`report` to update the
file, and the launcher reads it (:func:`read`) to show phases, progress bars and times. Without
the variable (a plain ``rsseg train``) nothing is written. Progress is best effort: a write that
fails is skipped.

Fields written by every command:

* ``started`` (time of the first report of this process), ``pid``, ``updated``;
* ``state``: ``running``, then ``done``/``failed``/``paused``/``stopped`` (:func:`finish`);
* ``active_before_s``: seconds of work before a resume (the elapsed time excludes pauses);
* ``phase``/``stage`` and the counters of the phase (``epoch``/``epochs``, ``batch``/``steps``,
  ``tile``/``tiles``, ``done``/``total``);
* ``history``: one entry per finished epoch (loss and validation metrics), for the curves.

This module does not import torch or keras: the GUI uses it too.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

ENV = "RSSEG_PROGRESS_FILE"
FILE_NAME = "progress.json"
FINAL_STATES = ("done", "failed", "paused", "stopped")

_state: dict = {}
T = TypeVar("T")


def report(**fields) -> None:
    """Merge ``fields`` into the progress of this process and write it (if a launcher asks)."""
    path = os.environ.get(ENV)
    if not path:
        return
    if not _state:
        _state.update(started=time.time(), pid=os.getpid(), state="running", active_before_s=0.0)
    _state.update(fields)
    _state["updated"] = time.time()
    tmp = Path(path).with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_state, default=str), encoding="utf-8")
        os.replace(tmp, path)  # readers never see a half-written file
    except OSError:  # e.g. the reader holds the file open on Windows: next update wins
        pass


def finish(state: str, **fields) -> None:
    """Record the final state of the command (``done``, ``failed``, ``paused``, ``stopped``)."""
    if state not in FINAL_STATES:
        raise ValueError(f"state must be one of {FINAL_STATES}, got {state!r}")
    report(state=state, finished=time.time(), **fields)


def active_seconds(now: float | None = None) -> float:
    """Seconds of work of this command so far, including the runs before a resume."""
    if not _state:
        return 0.0
    return _state.get("active_before_s", 0.0) + (now or time.time()) - _state["started"]


def tracked(items: Iterable[T], total: int | None = None, **fields) -> Iterator[T]:
    """Yield ``items`` while reporting ``done``/``total`` (and ``fields``) after each one."""
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    report(done=0, total=total, **fields)
    for i, item in enumerate(items):
        yield item
        report(done=i + 1)


def reset() -> None:
    """Forget the progress of this process (tests, or a command started again in-process)."""
    _state.clear()


def read(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
