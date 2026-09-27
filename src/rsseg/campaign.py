# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment campaigns: run many trainings, DINO pre-trainings and benchmarks from one YAML file.

A campaign (``configs/campaigns/*.yaml``) lists experiment configs, seeds, sweeps and the DINO
backbones and unlabelled pools they need::

    name: rq6_ndsm
    seeds: [0, 1, 2]
    include: [other_campaign.yaml]          # merged (identical runs are shared)
    pools:    {toronto: {source: toronto, raw: data/download/Toronto/Images, out: data/unlabeled/toronto}}
    pretrain: {dino_r18: {config: configs/ssl/dino_r18.yaml, per_seed: false, set: []}}
    runs:
      - config: configs/experiments/maresunet_r18_dino.yaml
        backbone: dino_r18                  # model.weights = the exported backbone
        name: maresunet_r18_dino_lf{data.label_fraction}
        sweep: {data.label_fraction: [0.01, 0.1]}
        set: [train.epochs=100]
    benchmarks:
      - {config: configs/experiments/maresunet_r18_imagenet.yaml, mode: seg, set: [], name: r18}

Execution (:func:`run_campaign`): unlabelled pools and prepared data are created if missing, then
DINO pre-trainings, trainings (seed by seed) and benchmarks run one per **subprocess** (a crash
or an out-of-memory error stops only that step), and a report is written. Every step lives in a
store shared by all campaigns, ``<root>/store/{backbones,runs,benchmarks}/<id>/`` with its
``spec.json``, ``log.txt`` and ``outcome.json``: completed steps are skipped, so an interrupted
campaign resumes where it stopped and a run shared by two campaigns (e.g. the baseline) is
trained once. ``<root>/<campaign>/`` holds ``status.json`` and ``report.md``. ``--quick`` uses a
separate root, so smoke-sized results never mix with real ones. While a step runs it writes
``progress.json`` (:mod:`rsseg.progress`) and :func:`describe_progress` turns it into the lines
the console shows (epoch, metrics, tile under evaluation, time left).
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from rsseg import progress
from rsseg.config import DataConfig, ExperimentConfig, load_config

log = logging.getLogger(__name__)

QUICK_TRAIN = ["train.epochs=1", "train.steps_per_epoch=20", "train.validation_steps=5",
               "eval.max_tiles=1"]  # fmt: skip
QUICK_SSL = ["ssl.epochs=1", "ssl.steps_per_epoch=10"]
BENCHMARK_STEPS, QUICK_BENCHMARK_STEPS = 10, 3
PREPARE_ESTIMATE_GB = 2.0
"""Rough disk need of a data root or an unlabelled pool that still has to be prepared."""
BACKBONE_ESTIMATE_GB = 0.3
OPTIMIZER_SLOTS = {"adamw": 2, "adam": 2, "sgd": 1}
"""Optimizer variables per weight, saved in every checkpoint together with the model."""

Runner = Callable[[list[str], Path, "Path | None"], int]


class CampaignError(Exception):
    """Invalid campaign or unmet prerequisite, detected before any GPU time is spent."""


@dataclass
class Step:
    kind: str
    """``pretrain``, ``train`` or ``benchmark``."""
    id: str
    name: str
    """Experiment name (``cfg.name``): reports group seeds by it."""
    config: str
    overrides: list[str]
    seed: int | None = None
    backbone: str | None = None
    """Id of the pre-training step whose backbone initialises the encoder."""
    mode: str = "seg"
    cfg: ExperimentConfig | None = field(default=None, repr=False, compare=False)

    def spec(self) -> dict:
        return {"kind": self.kind, "config": Path(self.config).as_posix(), "name": self.name,
                "overrides": self.overrides, "seed": self.seed, "backbone": self.backbone,
                "mode": self.mode if self.kind == "benchmark" else None}  # fmt: skip


@dataclass
class Pool:
    key: str
    source: str
    raw: str
    out: str
    max_per_frame: int = 0


@dataclass
class Campaign:
    name: str
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [0])
    pools: dict[str, Pool] = field(default_factory=dict)
    pretrains: dict[str, Step] = field(default_factory=dict)
    runs: list[Step] = field(default_factory=list)
    benchmarks: list[Step] = field(default_factory=list)
    shared_seed_pretrains: list[str] = field(default_factory=list)
    """Pre-trainings run once and shared by every seed (noted in the report)."""


# ------------------------------------------------------------------------------------ loading


def _effective(config: str, overrides: list[str], where: str) -> ExperimentConfig:
    if not Path(config).exists():
        raise CampaignError(f"{where}: config {config} not found (run from the repository root)")
    try:
        return load_config(config, overrides)
    except (KeyError, ValueError, TypeError) as e:
        raise CampaignError(f"{where}: {e}") from e


def _format_name(template: str, values: dict[str, object], where: str) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) not in values:
            raise CampaignError(f"{where}: {{{m.group(1)}}} in the name is not a sweep key")
        return str(values[m.group(1)])

    return re.sub(r"\{([^}]+)\}", sub, template)


def _merge(into: dict, key: str, item, what: str) -> None:
    old = into.get(key)
    if old is None:
        into[key] = item
    elif (old.spec() if isinstance(old, Step) else old) != (
        item.spec() if isinstance(item, Step) else item
    ):
        raise CampaignError(f"{what} {key!r} is defined twice with different settings")


def load_campaign(path: str | Path, seeds: list[int] | None = None) -> Campaign:
    """Parse and expand a campaign file (``seeds`` overrides the seeds of every file)."""
    path = Path(path)
    if not path.exists():
        raise CampaignError(f"campaign {path} not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"name", "description", "seeds", "include", "pools", "pretrain", "runs",
                          "benchmarks"}  # fmt: skip
    if unknown:
        raise CampaignError(f"{path}: unknown keys {sorted(unknown)}")
    c = Campaign(raw.get("name") or path.stem, raw.get("description", ""),
                 list(seeds if seeds is not None else raw.get("seeds", [0])))  # fmt: skip
    pools: dict[str, Pool] = {}
    pretrains: dict[str, Step] = {}
    runs: dict[str, Step] = {}
    benchmarks: dict[str, Step] = {}
    shared: set[str] = set()

    for inc in raw.get("include", []) or []:
        sub = load_campaign(path.parent / inc, seeds)
        for k, p in sub.pools.items():
            _merge(pools, k, p, "pool")
        for k, s in sub.pretrains.items():
            _merge(pretrains, k, s, "pre-training")
        for s in sub.runs:
            _merge(runs, s.id, s, "run")
        for s in sub.benchmarks:
            _merge(benchmarks, s.id, s, "benchmark")
        shared |= set(sub.shared_seed_pretrains)

    for key, p in (raw.get("pools") or {}).items():
        _merge(pools, key, Pool(key, p["source"], p["raw"], p["out"], p.get("max_per_frame", 0)),
               "pool")  # fmt: skip

    pre_defs = raw.get("pretrain") or {}
    for i, entry in enumerate(raw.get("runs") or []):
        where = f"{path.name} runs[{i}]"
        if "config" not in entry:
            raise CampaignError(f"{where}: 'config' is required")
        base_set = list(entry.get("set", []))
        sweep = entry.get("sweep") or {}
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            values = dict(zip(keys, combo, strict=True))
            assigns = [f"{k}={v}" for k, v in values.items()]
            if entry.get("name"):
                name = _format_name(entry["name"], values, where)
            else:
                name = _effective(entry["config"], base_set, where).name
                if values:
                    name += "_" + "_".join(f"{k.split('.')[-1]}{v}" for k, v in values.items())
            for seed in c.seeds:
                backbone = None
                if entry.get("backbone"):
                    key = entry["backbone"]
                    if key not in pre_defs and key not in pretrains:
                        raise CampaignError(f"{where}: unknown backbone {key!r}")
                    if key in pre_defs:
                        backbone = _add_pretrain(pretrains, shared, key, pre_defs[key], seed, path)
                    else:
                        backbone = key
                step = Step("train", f"{name}__s{seed}", name, entry["config"],
                            base_set + assigns, seed, backbone)  # fmt: skip
                step.cfg = _effective(step.config, step.overrides, f"{where} ({step.id})")
                _merge(runs, step.id, step, "run")

    for i, entry in enumerate(raw.get("benchmarks") or []):
        where = f"{path.name} benchmarks[{i}]"
        overrides = list(entry.get("set", []))
        cfg = _effective(entry["config"], overrides, where)
        mode = entry.get("mode", "seg")
        if mode not in ("seg", "dino"):
            raise CampaignError(f"{where}: mode must be seg or dino")
        name = entry.get("name") or f"{cfg.name}_{mode}"
        step = Step("benchmark", name, name, entry["config"], overrides, mode=mode)
        step.cfg = cfg
        _merge(benchmarks, step.id, step, "benchmark")

    c.pools, c.pretrains = pools, pretrains
    c.runs = sorted(runs.values(), key=lambda s: s.seed or 0)  # seed by seed, entry order kept
    c.benchmarks = list(benchmarks.values())
    c.shared_seed_pretrains = sorted(shared)
    return c


def _add_pretrain(pretrains: dict[str, Step], shared: set[str], key: str, entry: dict, seed: int,
                  path: Path) -> str:  # fmt: skip
    where = f"{path.name} pretrain.{key}"
    if "config" not in entry:
        raise CampaignError(f"{where}: 'config' is required")
    overrides = list(entry.get("set", []))
    if entry.get("per_seed", False):
        pid, overrides, pseed = f"{key}_s{seed}", overrides + [f"ssl.seed={seed}"], seed
    else:
        pid, pseed = key, None
        shared.add(key)
    step = Step("pretrain", pid, pid, entry["config"], overrides, pseed)
    step.cfg = _effective(step.config, step.overrides, where)
    _merge(pretrains, pid, step, "pre-training")
    return pid


def select(c: Campaign, only: list[str] | None) -> Campaign:
    """Keep the runs/benchmarks whose id or name matches one of the ``only`` glob patterns, and
    the pre-trainings and pools they need."""
    if not only:
        return c

    def keep(s: Step) -> bool:
        return any(fnmatch.fnmatch(s.id, p) or fnmatch.fnmatch(s.name, p) for p in only)

    runs = [s for s in c.runs if keep(s)]
    needed = {s.backbone for s in runs if s.backbone}
    pretrains = {k: s for k, s in c.pretrains.items() if k in needed}
    pools_needed = {Path(p).as_posix() for s in pretrains.values() for p in s.cfg.ssl.extra_pools}
    pools = {k: p for k, p in c.pools.items() if Path(p.out).as_posix() in pools_needed}
    return replace(c, runs=runs, benchmarks=[s for s in c.benchmarks if keep(s)],
                   pretrains=pretrains, pools=pools)  # fmt: skip


# ------------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Layout:
    root: Path

    def step_dir(self, step: Step) -> Path:
        folder = {"pretrain": "backbones", "train": "runs", "benchmark": "benchmarks"}[step.kind]
        return self.root / "store" / folder / step.id

    def backbone(self, pretrain_id: str) -> Path:
        return self.root / "store" / "backbones" / pretrain_id / "backbone.weights.h5"

    def campaign_dir(self, name: str) -> Path:
        return self.root / name


def completed_report(layout: Layout, step: Step) -> Path | None:
    """Latest ``report.json`` (or ``result.json`` without test evaluation) of a training step."""
    d = layout.step_dir(step)
    target = "report.json" if step.cfg is None or step.cfg.eval.split else "result.json"
    found = sorted(d.glob(f"seg-*/{target}"))
    return found[-1] if found else None


def is_done(layout: Layout, step: Step) -> bool:
    if step.kind == "pretrain":
        return layout.backbone(step.id).exists()
    if step.kind == "benchmark":
        return (layout.step_dir(step) / "result.json").exists()
    return completed_report(layout, step) is not None


def check_spec(layout: Layout, step: Step) -> None:
    path = layout.step_dir(step) / "spec.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != step.spec():
            raise CampaignError(
                f"{step.kind} {step.id!r} already exists in {path.parent} with different settings "
                f"({old} vs {step.spec()}): give the new one another name"
            )


def command(layout: Layout, step: Step, campaign: str, quick: bool) -> list[str]:
    d = layout.step_dir(step).as_posix()
    tags = [f"tracking.tags.campaign={campaign}", f"tracking.tags.run_id={step.id}"]
    if step.kind == "pretrain":
        args = ["pretrain", "-c", step.config, "--set", *step.overrides, f"output_dir={d}",
                f"name={step.name}", f"ssl.export_path={layout.backbone(step.id).as_posix()}",
                *tags, *(QUICK_SSL if quick else [])]  # fmt: skip
    elif step.kind == "train":
        weights = (
            [f"model.weights={layout.backbone(step.backbone).as_posix()}"] if step.backbone else []
        )
        args = ["train", "-c", step.config, "--set", *step.overrides, f"output_dir={d}",
                f"name={step.name}", f"train.seed={step.seed}", f"tracking.tags.seed='{step.seed}'",
                *weights, *tags, *(QUICK_TRAIN if quick else [])]  # fmt: skip
    else:
        steps = QUICK_BENCHMARK_STEPS if quick else BENCHMARK_STEPS
        args = ["benchmark", "-c", step.config, "--mode", step.mode, "--steps", str(steps),
                "--set", *step.overrides]  # fmt: skip
    return [sys.executable, "-m", "rsseg.cli", "--json", *args]


def subprocess_runner(cmd: list[str], log_path: Path, stdout: Path | None = None) -> int:
    """Run ``cmd`` with stderr (and stdout, unless ``stdout`` is given) appended to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           progress.ENV: str(log_path.parent / progress.FILE_NAME)}  # fmt: skip
    with log_path.open("a", encoding="utf-8") as err:
        err.write(f"\n$ {subprocess.list2cmdline(cmd)}\n")
        err.flush()
        if stdout is None:
            return subprocess.run(cmd, stdout=err, stderr=err, env=env, check=False).returncode
        with stdout.open("w", encoding="utf-8") as out:
            return subprocess.run(cmd, stdout=out, stderr=err, env=env, check=False).returncode


def _last_json(text: str) -> dict | None:
    """The JSON object printed last by ``rsseg --json`` (other tools may print before it)."""
    for m in reversed(list(re.finditer(r"^\{", text, flags=re.M))):
        try:
            return json.loads(text[m.start() :])
        except json.JSONDecodeError:
            continue
    return None


# ----------------------------------------------------------------------------- prerequisites


@dataclass
class DataPrep:
    root: str
    cfg: DataConfig
    action: str
    """``ok``, ``prepare`` or ``prepare (add nDSM)``."""


PREPARE_KEYS = ("dataset", "protocol", "val_tiles", "exclude_7_10", "train_labels", "crop_size",
                "train_stride", "image_pattern", "label_pattern")  # fmt: skip


def _ndsm_kind(d: DataConfig) -> tuple[str, str | None] | None:
    from rsseg.data.prepare import resolve_ndsm_source
    from rsseg.data.sources import get_source

    kind = resolve_ndsm_source(d, get_source(d.dataset))
    if kind is None:
        return None
    return kind, d.ndsm_variant if kind == "official" else None


def plan_data(steps: list[Step]) -> list[DataPrep]:
    """One entry per ``data.root`` used by the steps: is it ready, or how to prepare it."""
    from rsseg.data.splits import get_split

    groups: dict[str, list[DataConfig]] = {}
    for s in steps:
        if s.kind in ("train", "pretrain") and s.cfg is not None:
            groups.setdefault(Path(s.cfg.data.root).as_posix(), []).append(s.cfg.data)
    preps = []
    for root, datas in groups.items():
        first = datas[0]
        for d in datas[1:]:
            diff = [k for k in PREPARE_KEYS if getattr(d, k) != getattr(first, k)]
            if diff:
                raise CampaignError(f"runs sharing data.root={root} disagree on {diff}: "
                                    "give them different data roots")  # fmt: skip
        kinds = {_ndsm_kind(d) for d in datas if d.ndsm}
        if len(kinds) > 1:
            raise CampaignError(f"runs sharing data.root={root} use different nDSMs {kinds}: "
                                "give them different data roots")  # fmt: skip
        cfg = replace(first, ndsm=bool(kinds))
        if kinds:
            kind, variant = next(iter(kinds))
            cfg = replace(cfg, ndsm_source=kind, ndsm_variant=variant or first.ndsm_variant)
        preps.append(DataPrep(root, cfg, _root_action(Path(root), cfg, get_split)))
    return preps


def _root_action(root: Path, cfg: DataConfig, get_split) -> str:
    files = ("crops.csv", "tiles.csv", "stats.json", "split.json")
    if not all((root / f).exists() for f in files):
        return "prepare"
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    expected = get_split(cfg.protocol, cfg.val_tiles, cfg.exclude_7_10, cfg.dataset)
    problems = []
    for key, want, default in (("dataset", cfg.dataset, "potsdam"), ("protocol", cfg.protocol, None),
                               ("train_labels", cfg.train_labels, "full"),
                               ("crop_size", cfg.crop_size, cfg.crop_size),
                               ("train_stride", cfg.train_stride, cfg.train_stride)):  # fmt: skip
        if split.get(key, default) != want:
            problems.append(f"{key} {split.get(key, default)!r} ≠ {want!r}")
    problems += [
        f"{k} tiles differ" for k in ("train", "val", "test") if split.get(k) != expected[k]
    ]
    if problems:
        raise CampaignError(f"{root} was prepared with other settings ({'; '.join(problems)}): "
                            "use another data.root or delete the folder")  # fmt: skip
    if not cfg.ndsm:
        return "ok"
    have = split.get("ndsm")
    if not have:
        return "prepare (add nDSM)"
    variant = split.get("ndsm_variant")
    if have != cfg.ndsm_source or (have == "official" and variant not in (None, cfg.ndsm_variant)):
        raise CampaignError(f"{root} holds the {have} nDSM ({variant or '-'}), the runs need "
                            f"{cfg.ndsm_source} ({cfg.ndsm_variant}): use another data.root")  # fmt: skip
    return "ok"


def pool_ready(pool: Pool) -> bool:
    out = Path(pool.out)
    if not ((out / "crops.csv").exists() and (out / "source.json").exists()):
        return False
    meta = json.loads((out / "source.json").read_text(encoding="utf-8"))
    return meta.get("source") == pool.source


def check_pools(c: Campaign) -> None:
    declared = {Path(p.out).as_posix() for p in c.pools.values()}
    for s in c.pretrains.values():
        for p in s.cfg.ssl.extra_pools:
            ready = (Path(p) / "crops.csv").exists() and (Path(p) / "source.json").exists()
            if not ready and Path(p).as_posix() not in declared:
                raise CampaignError(f"pre-training {s.id} needs the pool {p}, which is neither "
                                    "prepared nor declared under 'pools'")  # fmt: skip


# ------------------------------------------------------------------------------ disk and time


def _model_key(cfg: ExperimentConfig) -> str:
    m = replace(cfg.model, weights="none", in_channels=4 if cfg.data.ndsm else 3,
                extra_channel_mean=None, extra_channel_std=None)  # fmt: skip
    return json.dumps(m.__dict__ | {"vq": m.vq.__dict__}, sort_keys=True, default=str)


def count_params_subprocess(model_keys: list[str]) -> dict[str, int]:
    """Parameter counts of the given model configs, built on the CPU in a child process (so
    this process never holds GPU memory)."""
    if not model_keys:
        return {}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1"}
    out = subprocess.run([sys.executable, "-m", "rsseg.campaign", "count-params"],
                         input=json.dumps(model_keys), capture_output=True, text=True, env=env,
                         check=True).stdout  # fmt: skip
    return dict(zip(model_keys, json.loads(out.strip().splitlines()[-1]), strict=True))


def _count_params_main() -> None:
    import rsseg  # noqa: F401  (selects the torch backend)
    from rsseg.config import ModelConfig, _build
    from rsseg.models.segmentor import build_segmentor

    keys = json.loads(sys.stdin.read())
    counts = [build_segmentor(_build(ModelConfig, json.loads(k))).count_params() for k in keys]
    print(json.dumps(counts))


def estimate_disk_gb(pending: list[Step], preps: list[DataPrep], pools: list[Pool],
                     param_counter: Callable[[list[str]], dict[str, int]]) -> float:  # fmt: skip
    trains = [s for s in pending if s.kind == "train"]
    params = param_counter(sorted({_model_key(s.cfg) for s in trains}))
    total = 0.0
    for s in trains:
        t = s.cfg.train
        slots = 1 + OPTIMIZER_SLOTS.get(t.optimizer, 2) + (1 if t.grad_accum_steps > 1 else 0)
        total += params[_model_key(s.cfg)] * 4 * slots * (t.keep_top_k + 1) / 1e9
    total += BACKBONE_ESTIMATE_GB * sum(s.kind == "pretrain" for s in pending)
    total += PREPARE_ESTIMATE_GB * (sum(p.action != "ok" for p in preps) + len(pools))
    return total


def _free_gb(path: Path) -> float:
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _mean_duration(layout: Layout, kind: str) -> float | None:
    folder = {"pretrain": "backbones", "train": "runs", "benchmark": "benchmarks"}[kind]
    durations = []
    for f in (layout.root / "store" / folder).glob("*/outcome.json"):
        o = json.loads(f.read_text(encoding="utf-8"))
        if o.get("state") == "done":
            durations.append(o["duration_s"])
    return sum(durations) / len(durations) if durations else None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds):
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    if h >= 24:
        return f"{h // 24}d{h % 24:02d}h"
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


STALE_S = 900
"""A running step whose progress has not changed for this long is flagged (hang or crash)."""


@dataclass
class Running:
    """The step being run, as shown by the console while it runs."""

    step: Step
    index: int
    """Position among the steps this invocation runs (1-based)."""
    total: int
    started: float
    progress_path: Path
    log_path: Path
    after: list[str]
    """Kinds of the steps still to run after this one."""
    mean_s: dict[str, float | None]
    """Mean duration of the completed steps by kind (store-wide), for the time left."""


def describe_progress(r: Running, p: dict | None, now: float) -> list[str]:
    """Lines describing a running step: phase, epoch, metrics and time left."""
    lines = [f"[{r.index}/{r.total}] {r.step.kind} {r.step.id} — running for "
             f"{fmt_duration(now - r.started)}"]  # fmt: skip
    left: tuple[float, float] | None = None
    if p is None:
        lines.append("starting: imports, data loaders, model…")
    elif p.get("phase") in ("train", "pretrain"):
        epoch, epochs = p.get("epoch") or 0, p.get("epochs") or 0
        s = f"epoch {epoch}/{epochs}"
        if p.get("stage") == "validation":
            s += " · validation"
        elif p.get("steps"):
            s += f" · batch {p.get('batch', 0)}/{p['steps']}"
        if p.get("value") is not None:
            s += (f" · {p['monitor']} {p['value']:.4f} (best {p['best']:.4f} at epoch "
                  f"{p['best_epoch']})")  # fmt: skip
        elif p.get("loss") is not None:
            s += f" · loss {p['loss']:.4f}"
        lines.append(s)
        if p.get("epoch_s"):
            details = [f"{fmt_duration(p['epoch_s'])} per epoch"]
            if p.get("stage") == "validation":
                done = float(epoch)
            else:
                done = epoch - 1 + (p.get("batch", 0) / p["steps"] if p.get("steps") else 0)
            longest = max(epochs - done, 0) * p["epoch_s"]
            shortest = longest
            if p.get("patience") and p.get("since_best") is not None:
                wait = max(p["patience"] - p["since_best"], 0)
                details.append(f"early stop after {wait} more epochs without improvement")
                shortest = min(longest, max(epoch + wait - done, 0) * p["epoch_s"])
            left = (shortest, longest)
            lines.append(" · ".join(details))
    elif p.get("phase") == "evaluation":
        s = f"{p.get('stage', 'test')} evaluation · tile {p.get('tile')}/{p.get('tiles')}"
        if p.get("tile_s"):
            rest = (p["tiles"] - p["tile"] + 1) * p["tile_s"]
            s += f" · {fmt_duration(p['tile_s'])} per tile"
            left = (rest, rest)
        lines.append(s)
    if left is not None:
        if round(left[0]) == round(left[1]):
            span = fmt_duration(left[1])
        else:
            span = f"{fmt_duration(left[0])}–{fmt_duration(left[1])}"
        extra = " + test evaluation" if p.get("phase") == "train" else ""
        lines.append(f"this step: ~{span} left{extra}")
    if p is not None and now - p.get("updated", now) > STALE_S:
        lines.append(f"no progress for {fmt_duration(now - p['updated'])}: check the log")
    if r.after:
        means = [r.mean_s.get(k) for k in r.after]
        if None in means:
            eta = "time n/a until a step of each kind has completed"
        else:
            eta = f"~{fmt_duration(sum(means))}"
        lines.append(f"then {len(r.after)} more steps: {eta}")
    lines.append(f"log: {r.log_path.as_posix()}")
    return lines


# ---------------------------------------------------------------------------------- execution


@dataclass
class Options:
    root: str = "runs/campaigns"
    quick: bool = False
    dry_run: bool = False
    stop_on_error: bool = False
    clean_failed: bool = False
    ignore_disk: bool = False


def layout_for(opts: Options) -> Layout:
    return Layout(Path(opts.root + "-quick" if opts.quick else opts.root))


def plan_campaign(c: Campaign, opts: Options,
                  param_counter: Callable[[list[str]], dict[str, int]] | None = None) -> dict:  # fmt: skip
    """Everything that would happen, without running anything (``--dry-run``)."""
    layout = layout_for(opts)
    steps = [*c.pretrains.values(), *c.runs, *c.benchmarks]
    for s in steps:
        check_spec(layout, s)
    check_pools(c)
    preps = plan_data([*c.pretrains.values(), *c.runs])
    pools = [p for p in c.pools.values() if not pool_ready(p)]
    pending = [s for s in steps if not is_done(layout, s)]
    disk = estimate_disk_gb(pending, preps, pools, param_counter or count_params_subprocess)
    eta = 0.0
    for kind in ("pretrain", "train", "benchmark"):
        n = sum(s.kind == kind for s in pending)
        mean = _mean_duration(layout, kind)
        eta = math.nan if n and mean is None else eta + n * (mean or 0)
    return {
        "campaign": c.name, "root": layout.root.as_posix(), "quick": opts.quick,
        "seeds": c.seeds,
        "pools": [{"pool": p.key, "action": "prepare-unlabeled" if p in pools else "ok"}
                  for p in c.pools.values()],
        "data": [{"root": p.root, "action": p.action, "ndsm": p.cfg.ndsm} for p in preps],
        "steps": [{"kind": s.kind, "id": s.id, "state": "pending" if s in pending else "done"}
                  for s in steps],
        "pending": len(pending), "done": len(steps) - len(pending),
        "disk_needed_gb": round(disk, 1), "disk_free_gb": round(_free_gb(layout.root), 1),
        "eta": fmt_duration(eta) if pending else "0m00s",
    }  # fmt: skip


def run_campaign(c: Campaign, opts: Options, runner: Runner = subprocess_runner,
                 param_counter: Callable[[list[str]], dict[str, int]] | None = None,
                 echo: Callable[[str], None] = log.info, plan: dict | None = None,
                 watch: Callable[[Running], AbstractContextManager] | None = None) -> dict:  # fmt: skip
    """Run (or resume) a campaign; returns the summary written to ``status.json``.

    ``plan`` is the result of :func:`plan_campaign` if already computed. ``watch`` wraps every
    step that runs (the console uses it to show the live progress).
    """
    plan = plan or plan_campaign(c, opts, param_counter)
    if opts.dry_run:
        return plan
    if plan["disk_needed_gb"] > plan["disk_free_gb"] and not opts.ignore_disk:
        raise CampaignError(f"the campaign needs ~{plan['disk_needed_gb']} GB, only "
                            f"{plan['disk_free_gb']} GB are free (--ignore-disk to go on anyway)")  # fmt: skip
    layout = layout_for(opts)
    cdir = layout.campaign_dir(c.name)
    cdir.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}
    t_start = time.time()

    def save() -> dict:
        summary = {"campaign": c.name, "quick": opts.quick, "elapsed_s": round(time.time() - t_start),
                   "counts": _counts(status), "steps": status}  # fmt: skip
        (cdir / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # 1. unlabelled pools and prepared data (a failure here stops the campaign)
    from rsseg.data.prepare import inspect_dataset, prepare_dataset
    from rsseg.data.unlabeled import prepare_unlabeled

    for p in c.pools.values():
        if not pool_ready(p):
            echo(f"pool {p.key}: prepare-unlabeled {p.source} → {p.out}")
            prepare_unlabeled(p.source, p.raw, p.out, max_per_frame=p.max_per_frame)
    for prep in plan_data([*c.pretrains.values(), *c.runs]):
        if prep.action == "ok":
            continue
        echo(f"data {prep.root}: {prep.action}")
        report = inspect_dataset(prep.cfg)
        if report["errors"]:
            bad = [r["tile"] for r in report["tiles"] if "ERROR" in r["notes"]]
            raise CampaignError(f"{prep.root}: `rsseg prepare --dry-run` finds errors in {bad}")
        prepare_dataset(prep.cfg)

    # 2. pre-trainings, trainings (seed by seed), benchmarks
    failed: set[str] = set()
    steps = [*c.pretrains.values(), *c.runs, *c.benchmarks]
    to_run = [s for s in steps if not is_done(layout, s)]
    for step in steps:
        d = layout.step_dir(step)
        entry = {"kind": step.kind, "log": (d / "log.txt").as_posix()}
        if is_done(layout, step):
            status[step.id] = {**entry, "state": "done", "skipped": True}
            continue
        if step.backbone in failed:
            status[step.id] = {**entry, "state": "blocked", "reason": f"{step.backbone} failed"}
            save()
            continue
        d.mkdir(parents=True, exist_ok=True)
        if opts.clean_failed:
            for old in d.glob("*-*-*"):  # timestamped run folders of failed attempts
                shutil.rmtree(old, ignore_errors=True)
        (d / "spec.json").write_text(json.dumps(step.spec(), indent=2), encoding="utf-8")
        index = to_run.index(step) + 1
        echo(f"▶ [{index}/{len(to_run)}] {step.kind} {step.id}  (log: {entry['log']})")
        status[step.id] = {**entry, "state": "running"}
        save()
        t0 = time.time()
        stdout = d / "stdout.json" if step.kind == "benchmark" else None
        (d / progress.FILE_NAME).unlink(missing_ok=True)
        means = {k: _mean_duration(layout, k) for k in ("pretrain", "train", "benchmark")}
        running = Running(step, index, len(to_run), t0, d / progress.FILE_NAME, d / "log.txt",
                          [s.kind for s in to_run[index:]], means)  # fmt: skip
        with watch(running) if watch else nullcontext():
            code = runner(command(layout, step, c.name, opts.quick), d / "log.txt", stdout)
        if code == 0 and step.kind == "benchmark":
            result = _last_json(stdout.read_text(encoding="utf-8")) if stdout.exists() else None
            if result is None:
                code = -1
            else:
                (d / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        ok = code == 0 and is_done(layout, step)
        outcome = {"state": "done" if ok else "failed", "exit_code": code,
                   "duration_s": round(time.time() - t0)}  # fmt: skip
        (d / "outcome.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        status[step.id] = {**entry, **outcome}
        echo(f"{'✓' if ok else '✗'} {step.kind} {step.id}: {outcome['state']} in "
             f"{fmt_duration(outcome['duration_s'])}")  # fmt: skip
        if not ok:
            failed.add(step.id)
            if opts.stop_on_error:
                save()
                raise CampaignError(f"{step.kind} {step.id} failed (log: {entry['log']})")
        save()

    summary = save()
    report = write_campaign_report(c, layout, status)
    summary["report"] = report.as_posix()
    return summary


def _counts(status: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in status.values():
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    return counts


def write_campaign_report(c: Campaign, layout: Layout, status: dict[str, dict]) -> Path:
    from rsseg.eval.report import aggregate_reports

    lines = [f"# Campaign `{c.name}`", ""]
    if c.description:
        lines += [c.description, ""]
    lines += [f"Seeds: {', '.join(map(str, c.seeds))} — steps: "
              + ", ".join(f"{v} {k}" for k, v in sorted(_counts(status).items())), ""]  # fmt: skip
    if c.shared_seed_pretrains:
        lines += [f"DINO pre-training with a single seed, shared by every fine-tuning seed: "
                  f"{', '.join(c.shared_seed_pretrains)}.", ""]  # fmt: skip
    reports = [p for p in (completed_report(layout, s) for s in c.runs) if p is not None]
    if reports:
        lines += ["## Results (test split, mean ± std over seeds)", "", aggregate_reports(reports)]
    bench = [(s, layout.step_dir(s) / "result.json") for s in c.benchmarks]
    bench = [(s, json.loads(p.read_text(encoding="utf-8"))) for s, p in bench if p.exists()]
    if bench:
        lines += ["## Benchmarks", "", "| Name | mode | params (M) | images/s | peak VRAM (GB) |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for s, r in bench:
            lines.append(f"| {s.name} | {r.get('mode')} | {r.get('params_millions')} | "
                         f"{r.get('images_per_second')} | {r.get('peak_gpu_memory_gb')} |")  # fmt: skip
        lines.append("")
    bad = {k: v for k, v in status.items() if v["state"] in ("failed", "blocked")}
    if bad:
        lines += ["## Failed or blocked steps", ""]
        lines += [f"- `{k}`: {v['state']} ({v.get('reason') or 'exit ' + str(v.get('exit_code'))}"
                  f") — log `{v['log']}`" for k, v in bad.items()]  # fmt: skip
    path = layout.campaign_dir(c.name) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__" and sys.argv[1:] == ["count-params"]:
    _count_params_main()
