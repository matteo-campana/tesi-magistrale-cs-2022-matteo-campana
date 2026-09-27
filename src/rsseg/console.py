# SPDX-License-Identifier: AGPL-3.0-or-later
"""Terminal output with `rich <https://github.com/Textualize/rich>`_: coloured logging, progress
bars and tables.

Logs and progress bars go to **stderr** (:data:`err_console`), results to **stdout**
(:data:`console`), so ``rsseg ... --json > result.json`` still produces a clean file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

THEME = Theme(
    {
        "key": "bold cyan",
        "value": "white",
        "metric": "bold green",
        "muted": "dim",
        "title": "bold magenta",
    }
)
console = Console(theme=THEME)
err_console = Console(stderr=True, theme=THEME, log_path=False)

T = TypeVar("T")


def setup_logging(verbose: bool = False) -> None:
    """Route the standard ``logging`` module through rich (level, time, coloured tracebacks)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=err_console,
                rich_tracebacks=True,
                show_path=verbose,
                markup=False,
            )
        ],
        force=True,
    )


def make_progress(transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•", style="muted"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}", style="muted"),
        console=err_console,
        transient=transient,
    )


def track(
    items: Iterable[T], description: str, total: int | None = None, transient: bool = False
) -> Iterator[T]:
    """Like :func:`rich.progress.track`, on stderr and with elapsed/remaining time."""
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    with make_progress(transient) as progress:
        task = progress.add_task(description, total=total, info="")
        for item in items:
            yield item
            progress.advance(task)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping) and v:
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if v == 0 or 1e-3 <= abs(v) < 1e4 else f"{v:.3g}"
    return str(v)


def result_table(result: Mapping[str, Any], title: str | None = None) -> Table:
    """Key/value table of a (nested) result dictionary; metrics are highlighted."""
    table = Table(title=title, title_style="title", show_header=True, header_style="bold")
    table.add_column("Key", style="key", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for k, v in _flatten(result).items():
        if isinstance(v, (list, tuple)) and len(v) > 12:
            v = f"<{len(v)} values>"
        style = "metric" if isinstance(v, float) else "value"
        table.add_row(k, f"[{style}]{escape(format_value(v))}[/]")
    return table


def _split_per_class(d: Mapping[str, Any], prefix: str = "") -> tuple[dict, dict[str, dict]]:
    """Separate ``per_class`` sub-dictionaries (rendered as their own tables) from the rest."""
    rest: dict[str, Any] = {}
    tables: dict[str, dict] = {}
    for k, v in d.items():
        if k == "per_class" and isinstance(v, Mapping):
            tables[prefix.rstrip(".") or "per class"] = dict(v)
        elif isinstance(v, Mapping):
            sub, sub_tables = _split_per_class(v, f"{prefix}{k}.")
            rest[k] = sub
            tables.update(sub_tables)
        else:
            rest[k] = v
    return rest, tables


def per_class_table(per_class: Mapping[str, Mapping[str, Any]], title: str) -> Table:
    columns = list(next(iter(per_class.values()), {}))
    table = Table(title=title, title_style="title", header_style="bold")
    table.add_column("class", style="key")
    for c in columns:
        table.add_column(c, justify="right")
    for name, scores in per_class.items():
        table.add_row(name, *(format_value(scores.get(c)) for c in columns))
    return table


def print_result(result: Mapping[str, Any], title: str | None = None) -> None:
    """Print a result dictionary: key/value table plus one table per ``per_class`` block."""
    rest, tables = _split_per_class(result)
    console.print(result_table(rest, title=title))
    for name, per_class in tables.items():
        console.print(per_class_table(per_class, title=f"{name} — per class"))


def config_panel(cfg, command: str) -> Panel:
    """Compact summary of the settings that matter for ``command``."""
    rows: dict[str, Any] = {"name": cfg.name}
    if command == "pretrain":
        s = cfg.ssl
        rows |= {
            "backbone": f"{s.arch} ({s.weights})",
            "data": ", ".join([cfg.data.root, *s.extra_pools]),
            "epochs × batch": f"{s.epochs} × {s.batch_size}",
            "crops": f"2×{s.global_crop_size}² + {s.local_crops_number}×{s.local_crop_size}²",
            "precision": s.precision,
        }
    elif command == "prepare":
        d = cfg.data
        rows |= {"dataset": d.dataset, "raw_dir": d.raw_dir, "root": d.root,
                 "protocol": d.protocol, "crop / stride": f"{d.crop_size} / {d.train_stride}",
                 "labels (train)": d.train_labels,
                 "nDSM": (d.ndsm_source + ("" if d.ndsm_source == "computed" else f" ({d.ndsm_variant})")) if d.ndsm else "no"}  # fmt: skip
    else:
        m, t = cfg.model, cfg.train
        rows |= {
            "backbone": f"{m.arch} ({m.weights})",
            "skip / neck": f"{m.skip} / {m.neck}",
            "data": f"{cfg.data.root} ({cfg.data.dataset}{', + nDSM' if cfg.data.ndsm else ''})",
        }
        if command in ("train", "benchmark"):
            rows |= {
                "epochs × batch": f"{t.epochs} × {t.batch_size} (accum {t.grad_accum_steps})",
                "optimizer": f"{t.optimizer}, lr {t.lr:g}, {t.scheduler}",
                "precision": t.precision,
            }
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="key", no_wrap=True)
    grid.add_column()
    for k, v in rows.items():
        grid.add_row(k, escape(str(v)))
    return Panel(grid, title=f"[title]rsseg {command}[/]", expand=False, border_style="magenta")
