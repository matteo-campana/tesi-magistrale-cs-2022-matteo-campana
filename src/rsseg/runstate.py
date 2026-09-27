# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resume checkpoints seen from outside the training process (no torch/keras import).

A paused, or periodically checkpointed, run keeps its state in ``<run>/resume/``
(:mod:`rsseg.train.resume` writes it): ``state.json`` describes where the run stopped and in
which environment, ``state.npz`` holds the arrays. The GUI and the CLI read ``state.json``
through this module to list resumable runs, to show epoch, batch and time already spent, and
to warn when the code or the libraries changed since the pause.
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

RESUME_DIR = "resume"
STATE_JSON = "state.json"
STATE_ARRAYS = "state.npz"

SAFE_OVERRIDES = (
    "data.num_workers",
    "data.prepare_workers",
    "eval.",
    "train.checkpoint_every_min",
    "ssl.checkpoint_every_min",
    "tracking.tracking_uri",
)
"""Settings that may change when a run is resumed: they do not alter the training itself."""

VERSIONED = ("rsseg", "keras", "torch", "torchvision", "numpy")


def resume_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / RESUME_DIR


def read_state(run_dir: str | Path) -> dict | None:
    """``state.json`` of the resume checkpoint of ``run_dir`` (``None`` if there is none)."""
    try:
        return json.loads((resume_dir(run_dir) / STATE_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_resumable(run_dir: str | Path) -> bool:
    state = read_state(run_dir)
    if state is None:
        return False
    return state.get("phase") == "evaluation" or (resume_dir(run_dir) / STATE_ARRAYS).exists()


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=cwd
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, str | None]:
    """Versions that matter to resume a run bit for bit (read without importing the packages)."""
    env: dict[str, str | None] = {"python": platform.python_version()}
    for name in VERSIONED:
        try:
            env[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env[name] = None
    env["git_commit"] = git_commit()
    return env


def environment_changes(state: dict, current: dict | None = None) -> list[str]:
    """Human-readable differences between the environment of a checkpoint and the current one."""
    saved = state.get("env") or {}
    current = current or environment()
    changes = []
    for key, old in saved.items():
        if key not in current or key == "gpu":
            continue
        new = current[key]
        if old != new:
            if key == "git_commit":
                old, new = (old or "?")[:8], (new or "?")[:8]
            changes.append(f"{key}: {old} → {new}")
    return changes


def unsafe_overrides(overrides: list[str] | None) -> list[str]:
    """Overrides that would change a resumed run (anything outside :data:`SAFE_OVERRIDES`)."""
    bad = []
    for item in overrides or []:
        key = item.split("=", 1)[0]
        if not any(key == s or (s.endswith(".") and key.startswith(s)) for s in SAFE_OVERRIDES):
            bad.append(key)
    return bad


def find_resumable(root: str | Path) -> list[Path]:
    """Run directories under ``root`` (any depth) with a resume checkpoint, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    runs = [p.parent.parent for p in root.rglob(f"{RESUME_DIR}/{STATE_JSON}")]
    runs = [r for r in runs if is_resumable(r)]
    return sorted(runs, key=lambda r: (resume_dir(r) / STATE_JSON).stat().st_mtime, reverse=True)
