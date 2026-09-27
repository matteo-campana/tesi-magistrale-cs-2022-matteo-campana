# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tooltips of the application: Italian texts (``help_it.yaml``) plus what the code knows.

For a configuration key the tooltip shows what the setting does, the admitted values, the
practical effects, the default and the file the current value comes from, and where to read
more. Keys without an Italian text fall back to the English docstring of the dataclass field
in :mod:`rsseg.config` (a test requires every field to have an Italian text).
"""

from __future__ import annotations

import ast
import dataclasses
import html
import inspect
import typing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rsseg import config as config_module
from rsseg.config import ExperimentConfig

HELP_FILE = Path(__file__).resolve().parent.parent / "help_it.yaml"


@dataclass
class FieldInfo:
    key: str
    """Dotted key, e.g. ``train.lr``."""
    type: Any
    default: Any
    doc: str
    """English docstring of the field (may be empty)."""

    @property
    def section(self) -> str:
        return self.key.split(".")[0] if "." in self.key else "general"

    @property
    def kind(self) -> str:
        """Widget kind: bool, int, float, str, optional-int, optional-float, optional-str, list, dict."""
        t = self.type
        origin = typing.get_origin(t)
        args = [a for a in typing.get_args(t) if a is not type(None)]
        optional = origin in (typing.Union, getattr(__import__("types"), "UnionType", ())) and len(
            typing.get_args(t)
        ) > len(args)
        base = args[0] if optional and args else t
        base_origin = typing.get_origin(base)
        if base_origin is list or base is list:
            return "list"
        if base_origin is dict or base is dict:
            return "dict"
        name = {bool: "bool", int: "int", float: "float", str: "str"}.get(base, "str")
        return f"optional-{name}" if optional else name


@lru_cache(maxsize=1)
def field_docs() -> dict[str, dict[str, str]]:
    """``{ClassName: {field: docstring}}`` parsed from the source of :mod:`rsseg.config`."""
    tree = ast.parse(inspect.getsource(config_module))
    docs: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls: dict[str, str] = {}
        body = node.body
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                nxt = body[i + 1] if i + 1 < len(body) else None
                if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                        and isinstance(nxt.value.value, str)):  # fmt: skip
                    cls[stmt.target.id] = inspect.cleandoc(nxt.value.value)
        docs[node.name] = cls
    return docs


def _default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


@lru_cache(maxsize=1)
def config_fields() -> tuple[FieldInfo, ...]:
    """Every leaf setting of :class:`ExperimentConfig`, in declaration order."""
    docs = field_docs()
    out: list[FieldInfo] = []

    def walk(cls: type, prefix: str) -> None:
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            t = hints[f.name]
            if dataclasses.is_dataclass(t):
                walk(t, f"{prefix}{f.name}.")
            else:
                doc = docs.get(cls.__name__, {}).get(f.name, "")
                out.append(FieldInfo(prefix + f.name, t, _default(f), doc))

    walk(ExperimentConfig, "")
    return tuple(out)


@lru_cache(maxsize=1)
def catalog() -> dict:
    return yaml.safe_load(HELP_FILE.read_text(encoding="utf-8")) or {}


def entry(section: str, key: str) -> dict | None:
    return (catalog().get(section) or {}).get(key)


def _fmt(value: Any) -> str:
    if value is None:
        return "vuoto"
    if isinstance(value, bool):
        return "sì" if value else "no"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip() or "[]"
    return str(value)


def tooltip(section: str, key: str, *, default: Any = None, has_default: bool = False,
            current: Any = None, source: str | None = None, doc_en: str = "",
            extra: str | None = None) -> str:  # fmt: skip
    """Rich-text tooltip (Qt accepts this HTML subset)."""
    e = entry(section, key) or {}
    parts = [f"<b>{html.escape(key)}</b>"]
    text = e.get("text") or doc_en
    if text:
        parts.append(html.escape(text))
    if e.get("values"):
        items = "".join(f"<li><code>{html.escape(str(k))}</code> — {html.escape(str(v))}</li>"
                        for k, v in e["values"].items())  # fmt: skip
        parts.append(f"Valori:<ul style='margin:0'>{items}</ul>")
    if e.get("effects"):
        parts.append(f"<i>Effetti:</i> {html.escape(e['effects'])}")
    if extra:
        parts.append(html.escape(extra))
    if has_default:
        line = f"Predefinito: <code>{html.escape(_fmt(default))}</code>"
        if source:
            line += f" · valore attuale da: {html.escape(source)}"
        parts.append(line)
    if e.get("doc"):
        parts.append(f"Approfondisci: <code>{html.escape(e['doc'])}</code>")
    return "<qt>" + "<br>".join(parts) + "</qt>"


def config_tooltip(field: FieldInfo, current: Any = None, source: str | None = None) -> str:
    extra = None
    if field.key in ("train.batch_size", "train.grad_accum_steps") and isinstance(current, dict):
        extra = (f"Batch effettivo attuale: {current['batch_size']} × {current['grad_accum_steps']}"
                 f" = {current['batch_size'] * current['grad_accum_steps']}")  # fmt: skip
    return tooltip("config", field.key, default=field.default, has_default=True, source=source,
                   doc_en=field.doc, extra=extra)  # fmt: skip


def text(section: str, key: str) -> str:
    """Plain text of an entry (for labels and status tips)."""
    return (entry(section, key) or {}).get("text", "")


def missing_config_help() -> list[str]:
    """Configuration keys without an Italian text (must be empty)."""
    known = catalog().get("config") or {}
    return [f.key for f in config_fields() if f.key not in known or not known[f.key].get("text")]
