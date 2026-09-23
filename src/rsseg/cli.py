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
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import rsseg  # noqa: E402,F401  (selects the torch backend before keras is imported)
from rsseg.config import load_config  # noqa: E402


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
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate synthetic Potsdam-like tiles")
    p.add_argument("--out", default="data/synthetic_raw")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    for name, help_ in (
        ("prepare", "tile the raw dataset into crops and manifests"),
        ("train", "train a segmentation model"),
        ("pretrain", "DINO self-supervised pre-training of the encoder"),
    ):
        _add_config_args(sub.add_parser(name, help=help_))

    p = sub.add_parser("evaluate", help="evaluate weights on whole tiles")
    _add_config_args(p)
    p.add_argument("--weights", required=True)

    p = sub.add_parser("benchmark", help="measure peak GPU memory and speed of a configuration")
    _add_config_args(p)
    p.add_argument("--mode", choices=["seg", "dino"], default="seg")
    p.add_argument("--steps", type=int, default=10)

    p = sub.add_parser("report", help="aggregate report.json files (mean ± std)")
    p.add_argument("paths", nargs="+", help="run directories or report.json files")
    p.add_argument("--out", help="write the Markdown table to this file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "synth":
        from rsseg.data.synthetic import make_synthetic_potsdam

        out = make_synthetic_potsdam(args.out, size=args.size, seed=args.seed)
        print(f"synthetic tiles written to {out}")
        return 0

    if args.command == "report":
        from rsseg.eval.report import aggregate_reports

        files: list[Path] = []
        for p in map(Path, args.paths):
            files += sorted(p.rglob("report.json")) if p.is_dir() else [p]
        table = aggregate_reports(files)
        if args.out:
            Path(args.out).write_text(table)
        print(table)
        return 0

    cfg = load_config(args.config, args.set)
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
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
