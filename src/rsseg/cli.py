# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line interface.

::

    rsseg synth     --out data/synthetic_raw               # fake tiles to try the pipeline
    rsseg prepare   -c CONFIG [--set data.raw_dir=...]     # tiles -> crops + manifests
    rsseg prepare   -c CONFIG --dry-run                    # files, splits and anomalies per tile
    rsseg prepare-unlabeled --source toronto --raw DIR --out DIR   # DINO pool from raw frames
    rsseg train     -c CONFIG [--set k=v ...]              # supervised training (+ test evaluation)
    rsseg pretrain  -c CONFIG                              # DINO pre-training of the encoder
    rsseg evaluate  -c CONFIG --weights best.weights.h5    # whole-tile evaluation
    rsseg benchmark -c CONFIG [--mode seg|dino]            # peak GPU memory / speed probe
    rsseg report    RUN_DIRS...                            # mean ± std table over runs
    rsseg campaign  configs/campaigns/X.yaml [--dry-run]   # many experiments, resumable
    rsseg backbones [--params]                             # available encoders (ResNet, Swin, ViT)

Results are printed as tables (``--json`` for machine-readable output on stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import rsseg  # noqa: E402,F401  (selects the torch backend before keras is imported)
from rsseg.config import load_config  # noqa: E402
from rsseg.console import (  # noqa: E402
    config_panel,
    console,
    err_console,
    print_result,
    setup_logging,
)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML configuration file")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. train.epochs=5",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="print results as JSON instead of rich tables"
    )
    # ``--json`` is also accepted after the sub-command (``rsseg train -c X --json``)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    sub = parser.add_subparsers(dest="command", required=True)
    add = lambda name, **kw: sub.add_parser(name, parents=[common], **kw)  # noqa: E731

    p = add("synth", help="generate synthetic Potsdam-like tiles")
    p.add_argument("--out", default="data/synthetic_raw")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    p = add("prepare", help="tile the raw dataset into crops and manifests")
    _add_config_args(p)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="only list the files chosen for every tile, the split and the anomalies",
    )

    p = add("prepare-unlabeled", help="cut unlabelled crops from raw aerial frames (DINO pool)")
    p.add_argument("--source", required=True, choices=["toronto", "vaihingen_frames"])
    p.add_argument("--raw", required=True, help="directory with the frames (e.g. .../Images)")
    p.add_argument("--out", required=True, help="output root, then listed in ssl.extra_pools")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--max-per-frame", type=int, default=0, help="0 = every crop")
    p.add_argument("--seed", type=int, default=0)

    for name, help_ in (
        ("train", "train a segmentation model"),
        ("pretrain", "DINO self-supervised pre-training of the encoder"),
    ):
        _add_config_args(add(name, help=help_))

    p = add("evaluate", help="evaluate weights on whole tiles")
    _add_config_args(p)
    p.add_argument("--weights", required=True)

    p = add("benchmark", help="measure peak GPU memory and speed of a configuration")
    _add_config_args(p)
    p.add_argument("--mode", choices=["seg", "dino"], default="seg")
    p.add_argument("--steps", type=int, default=10)

    p = add("report", help="aggregate report.json files (mean ± std)")
    p.add_argument("paths", nargs="+", help="run directories or report.json files")
    p.add_argument("--out", help="write the Markdown table to this file")

    p = add("campaign", help="run (or resume) a campaign of experiments from a YAML file")
    p.add_argument("path", help="campaign file, e.g. configs/campaigns/rq6_ndsm.yaml")
    p.add_argument("--dry-run", action="store_true", help="show the plan without running anything")
    p.add_argument("--quick", action="store_true",
                   help="tiny runs (1 epoch, 20 steps, 1 test tile) in a separate folder")  # fmt: skip
    p.add_argument("--only", nargs="+", metavar="GLOB", help="only runs/benchmarks matching")
    p.add_argument("--seeds", help="comma-separated seeds, overriding the campaign ones")
    p.add_argument("--root", default="runs/campaigns", help="output root (default runs/campaigns)")
    p.add_argument("--stop-on-error", action="store_true", help="stop at the first failed step")
    p.add_argument("--clean-failed", action="store_true",
                   help="delete the folders of failed attempts before retrying")  # fmt: skip
    p.add_argument("--ignore-disk", action="store_true", help="run even if disk looks too small")

    p = add("backbones", help="list the available encoders")
    p.add_argument(
        "--params", action="store_true", help="also build each encoder and count its parameters"
    )
    return parser


def backbones_table(with_params: bool = False):
    from rich.table import Table

    from rsseg.models.backbones import BACKBONES, build_backbone, feature_channels

    table = Table(title="Backbones", title_style="title", header_style="bold")
    table.add_column("arch", style="key")
    table.add_column("family")
    table.add_column("channels C1–C4")
    table.add_column("ImageNet weights (torchvision)")
    if with_params:
        table.add_column("params", justify="right")
    for arch, info in BACKBONES.items():
        row = [
            arch,
            str(info["family"]),
            " / ".join(map(str, feature_channels(arch))),
            f"✓ {str(info['imagenet']).split('.')[-1]}"
            if info["imagenet"]
            else "[muted]— (none, or DINO via `rsseg pretrain`)[/]",
        ]
        if with_params:
            with err_console.status(f"building {arch}…"):
                row.append(f"{build_backbone(arch, 'none').count_params() / 1e6:.1f} M")
        table.add_row(*row)
    return table


def inspection_table(result: dict):
    from rich.table import Table

    table = Table(
        title=f"{result['dataset']} — {result['raw_dir']}", title_style="title",
        header_style="bold",
    )  # fmt: skip
    for col in ("tile", "split", "size", "label", "eroded", "ndsm", "off-palette", "notes"):
        table.add_column(col, style="key" if col == "tile" else None)
    for r in result["tiles"]:
        notes = f"[red]{r['notes']}[/]" if "ERROR" in r["notes"] else r["notes"]
        table.add_row(r["tile"], r["split"], r["size"], r["label"], r["eroded"], r["ndsm"],
                      f"{r['off_palette']:.3%}", notes)  # fmt: skip
    return table


def campaign_plan_table(plan: dict):
    from rich.table import Table

    table = Table(title=f"campaign {plan['campaign']} — {plan['root']}", title_style="title",
                  header_style="bold")  # fmt: skip
    for col in ("kind", "id", "state"):
        table.add_column(col, style="key" if col == "id" else None)
    for p in plan["pools"]:
        table.add_row("pool", p["pool"], p["action"])
    for d in plan["data"]:
        table.add_row("data", d["root"] + (" (+nDSM)" if d["ndsm"] else ""), d["action"])
    for s in plan["steps"]:
        table.add_row(s["kind"], s["id"], "[muted]done[/]" if s["state"] == "done" else s["state"])
    return table


@contextlib.contextmanager
def live_step_progress(running):
    """Show the progress of a running campaign step, refreshed every second, until it ends."""
    from rich.live import Live
    from rich.text import Text

    from rsseg.campaign import describe_progress
    from rsseg.progress import read

    def render() -> Text:
        first, *rest = describe_progress(running, read(running.progress_path), time.time())
        text = Text(first, style="title")
        for line in rest:
            text.append("\n  " + line, style="muted" if line.startswith("log:") else None)
        return text

    with Live(get_renderable=render, console=err_console, refresh_per_second=1, transient=True):
        yield


def run_campaign_command(args) -> int:
    from rsseg.campaign import (
        CampaignError,
        Options,
        load_campaign,
        plan_campaign,
        run_campaign,
        select,
    )

    opts = Options(root=args.root, quick=args.quick, dry_run=args.dry_run,
                   stop_on_error=args.stop_on_error, clean_failed=args.clean_failed,
                   ignore_disk=args.ignore_disk)  # fmt: skip
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    try:
        campaign = select(load_campaign(args.path, seeds), args.only)
        with err_console.status("planning the campaign…"):
            plan = result = plan_campaign(campaign, opts)
        if not args.json:
            console.print(campaign_plan_table(result))
            console.print(f"pending {result['pending']}, done {result['done']} — disk needed "
                          f"~{result['disk_needed_gb']} GB (free {result['disk_free_gb']} GB) — "
                          f"estimated time {result['eta']}")  # fmt: skip
        if not args.dry_run:
            watch = None if args.json else live_step_progress
            result = run_campaign(campaign, opts, echo=err_console.print, plan=plan, watch=watch)
    except CampaignError as e:
        err_console.print(f"[red]campaign error:[/] {e}")
        return 2
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.dry_run:
        console.print(f"[metric]campaign {campaign.name}[/]: {result['counts']} — report: "
                      f"[key]{result['report']}[/]")  # fmt: skip
    return 0 if args.dry_run or not {"failed", "blocked"} & set(result["counts"]) else 1


def _print_result(result, title: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_result(result, title=title)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if args.command == "synth":
        from rsseg.data.synthetic import make_synthetic_potsdam

        with err_console.status("generating synthetic tiles…"):
            out = make_synthetic_potsdam(args.out, size=args.size, seed=args.seed)
        console.print(f"[metric]✓[/] synthetic tiles written to [key]{out}[/]")
        return 0

    if args.command == "prepare-unlabeled":
        from rsseg.data.unlabeled import prepare_unlabeled

        result = prepare_unlabeled(
            args.source, args.raw, args.out, args.crop, args.max_per_frame, seed=args.seed
        )
        _print_result(result, f"rsseg prepare-unlabeled — {args.source}", args.json)
        return 0

    if args.command == "backbones":
        console.print(backbones_table(args.params))
        return 0

    if args.command == "campaign":
        return run_campaign_command(args)

    if args.command == "report":
        from rsseg.eval.report import aggregate_reports

        files: list[Path] = []
        for p in map(Path, args.paths):
            files += sorted(p.rglob("report.json")) if p.is_dir() else [p]
        table = aggregate_reports(files)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(table, encoding="utf-8")
        if args.json:
            print(table)
        else:
            from rich.markdown import Markdown

            console.print(Markdown(table))
            if args.out:
                console.print(f"[muted]written to {args.out}[/]")
        return 0

    cfg = load_config(args.config, args.set)
    if not args.json:
        err_console.print(config_panel(cfg, args.command))
    if args.command == "prepare" and args.dry_run:
        from rsseg.data.prepare import inspect_dataset

        result = inspect_dataset(cfg.data)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            console.print(inspection_table(result))
            console.print(
                f"tiles found: {result['tiles_found']} — not in any split: "
                f"{result['unused_tiles'] or 'none'} — errors: {result['errors']}"
            )
        return 1 if result["errors"] else 0
    if args.command == "prepare":
        from rsseg.data.prepare import prepare_dataset

        result = prepare_dataset(cfg.data)
    elif args.command == "train":
        from rsseg.train.seg_train import train_segmentation

        result = train_segmentation(cfg)
    elif args.command == "pretrain":
        from rsseg.ssl.train import pretrain_dino

        result = pretrain_dino(cfg)
    elif args.command == "evaluate":
        from rsseg.train.seg_train import evaluate_segmentation

        result = evaluate_segmentation(cfg, args.weights)
    elif args.command == "benchmark":
        from rsseg.eval.benchmark import benchmark

        result = benchmark(cfg, args.mode, steps=args.steps)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print_result(result, f"rsseg {args.command} — {cfg.name}", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
