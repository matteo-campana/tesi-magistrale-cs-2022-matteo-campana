# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run folders written by ``rsseg train``/``pretrain``/``evaluate``, read for the results and
inference pages (no torch/keras import).

A run folder is ``<kind>-<name>-<YYYYmmdd-HHMMSS>`` with ``config.yaml`` and, depending on how
far it went, ``history.csv``, ``checkpoints/topk.json``, ``best.weights.h5``, ``result.json``,
``report.json`` and ``resume/`` (a paused run).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rsseg.runstate import is_resumable, read_state

RUN_RE = re.compile(r"^(seg|dino|eval)-(.+)-(\d{8}-\d{6})$")


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class RunInfo:
    path: Path
    kind: str
    name: str
    stamp: str
    config: dict = field(default_factory=dict)
    result: dict | None = None
    report: dict | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.stamp})"

    @property
    def state(self) -> str:
        if self.result is not None:
            return "done"
        if is_resumable(self.path):
            return "paused"
        return "incomplete"

    @property
    def arch(self) -> str:
        section = "ssl" if self.kind == "dino" else "model"
        return (self.config.get(section) or {}).get("arch", "?")

    @property
    def pretrain(self) -> str:
        w = (self.config.get("model") or {}).get("weights", "")
        if w in ("imagenet", "none"):
            return w
        return "DINO" if w else "?"

    @property
    def channels(self) -> int:
        m = self.config.get("model") or {}
        return int(
            m.get("in_channels") or (4 if (self.config.get("data") or {}).get("ndsm") else 3)
        )

    @property
    def test_miou(self) -> float | None:
        if self.report:
            return self.report.get("all_classes", {}).get("miou")
        if self.result and "test" in self.result:
            return self.result["test"].get("miou")
        return None

    @property
    def best_val_miou(self) -> float | None:
        return (self.result or {}).get("best_val_miou")

    @property
    def best_weights(self) -> Path | None:
        p = self.path / "best.weights.h5"
        return p if p.exists() else None

    def checkpoints(self) -> list[tuple[str, Path]]:
        """``(label, path)`` of the weights usable for inference, the best first."""
        out = []
        if self.best_weights:
            out.append(("migliore (best.weights.h5)", self.best_weights))
        for item in _json(self.path / "checkpoints" / "topk.json") or []:
            p = self.path / "checkpoints" / item["path"]
            if p.exists():
                value = next((v for k, v in item.items() if k not in ("epoch", "path")), None)
                out.append((f"epoca {item['epoch']} (val mIoU {value:.4f})", p))
        return out

    def history(self) -> list[dict]:
        try:
            with (self.path / "history.csv").open(encoding="utf-8") as f:
                return [{k: _float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        except OSError:
            return []

    def resume_state(self) -> dict | None:
        return read_state(self.path)

    def data_root(self) -> str | None:
        return (self.config.get("data") or {}).get("root")


def _float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_run(path: str | Path) -> RunInfo | None:
    path = Path(path)
    m = RUN_RE.match(path.name)
    if not m or not (path / "config.yaml").exists():
        return None
    try:
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        cfg = {}
    return RunInfo(path, m.group(1), m.group(2), m.group(3), cfg,
                   _json(path / "result.json"), _json(path / "report.json"))  # fmt: skip


def find_runs(
    roots: list[str | Path], kinds: tuple[str, ...] = ("seg", "dino", "eval")
) -> list[RunInfo]:
    """Run folders under ``roots`` (any depth), newest first."""
    seen, runs = set(), []
    for root in map(Path, roots):
        if not root.exists():
            continue
        for cfg in root.rglob("config.yaml"):
            d = cfg.parent
            if ".gui" in d.parts or d in seen:
                continue
            seen.add(d)
            run = load_run(d)
            if run is not None and run.kind in kinds:
                runs.append(run)
    return sorted(runs, key=lambda r: r.stamp, reverse=True)
