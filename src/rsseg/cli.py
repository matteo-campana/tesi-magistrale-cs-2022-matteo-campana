# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line interface.

::

    rsseg synth     --out data/synthetic_raw               # fake tiles to try the pipeline
    rsseg prepare   -c CONFIG [--set data.raw_dir=...]     # tiles -> crops + manifests
    rsseg train     -c CONFIG [--set k=v ...]              # supervised training (+ test evaluation)
    rsseg pretrain  -c CONFIG                              # DINO pre-training of the encoder
    rsseg evaluate  -c CONFIG --weights best.weights.h5    # whole-tile evaluation
    rsseg benchmark -c CONFIG [--mode seg|dino]            # peak GPU memory / speed probe
    rsseg report    RUN_DIRS...                            # mean ± std table over runs
    rsseg backbones [--params]                             # available encoders (ResNet, Swin, ViT)

Results are printed as tables (``--json`` for machine-readable output on stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

    for name, help_ in (
        ("prepare", "tile the raw dataset into crops and manifests"),
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

    if args.command == "backbones":
        console.print(backbones_table(args.params))
        return 0

    if args.command == "report":
        from rsseg.eval.report import aggregate_reports

        files: list[Path] = []
        for p in map(Path, args.paths):
            files += sorted(p.rglob("report.json")) if p.is_dir() else [p]
        table = aggregate_reports(files)
        if args.out:
            Path(args.out).write_text(table)
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
