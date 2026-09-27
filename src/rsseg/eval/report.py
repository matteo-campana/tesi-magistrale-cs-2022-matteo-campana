# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tile-level evaluation and Markdown/JSON reports (same layout as thesis tables 4.3–4.8).

Tiles are scored against the full labels and, when available, against the **eroded** labels
(object boundaries ignored), the reference of the ISPRS benchmark and of most papers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from rsseg.config import EvalConfig
from rsseg.console import track
from rsseg.data.classes import CLASS_NAMES, CLUTTER, NUM_CLASSES, index_to_rgb
from rsseg.data.datasets import load_image, read_tiles
from rsseg.data.prepare import load_mask
from rsseg.eval.sliding_window import predict_tile
from rsseg.eval.visualize import error_map
from rsseg.train.metrics import confusion_matrix_np, scores_from_confusion

log = logging.getLogger(__name__)


def _scores(cm: np.ndarray) -> dict:
    return {
        "all_classes": scores_from_confusion(cm),
        "without_clutter": scores_from_confusion(cm, exclude=(CLUTTER,)),
    }


def evaluate_tiles(
    model: keras.Model, data_root: str | Path, cfg: EvalConfig, out_dir: str | Path | None = None
) -> dict:
    """Sliding-window evaluation of whole tiles. A 4-channel model also reads the tile nDSM."""
    tiles = read_tiles(data_root, cfg.split)
    if cfg.max_tiles:
        tiles = tiles[: cfg.max_tiles]
    if not tiles:
        raise FileNotFoundError(f"no '{cfg.split}' tiles in {data_root}/tiles.csv")
    use_ndsm = model.input_shape[-1] == 4
    if use_ndsm and not all(r.get("ndsm") for r in tiles):
        raise FileNotFoundError(
            f"the model takes an nDSM channel but {data_root}/tiles.csv has none: "
            "run `rsseg prepare` with data.ndsm=true"
        )
    eroded = all(r.get("mask_eroded") for r in tiles)
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_eroded = np.zeros_like(total)
    per_tile = {}
    for row in track(tiles, f"evaluating {cfg.split} tiles"):
        image = load_image(row["image"], Path(data_root) / row["ndsm"] if use_ndsm else None)
        gt = load_mask(Path(data_root) / row["mask"])
        pred = predict_tile(
            model, image, NUM_CLASSES, cfg.window, cfg.stride, cfg.batch_size, cfg.tta
        )
        cm = confusion_matrix_np(gt, pred, NUM_CLASSES)
        total += cm
        if eroded:
            total_eroded += confusion_matrix_np(
                load_mask(Path(data_root) / row["mask_eroded"]), pred, NUM_CLASSES
            )
        per_tile[row["tile"]] = scores_from_confusion(cm)["miou"]
        log.info("tile %s: mIoU %.4f", row["tile"], per_tile[row["tile"]])
        if out_dir and cfg.save_predictions:
            pred_dir = Path(out_dir) / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(index_to_rgb(pred)).save(pred_dir / f"{row['tile']}_pred.png")
            Image.fromarray(error_map(gt, pred)).save(pred_dir / f"{row['tile']}_errors.png")
    report = {
        "split": cfg.split,
        "tiles": [r["tile"] for r in tiles],
        **_scores(total),
        "per_tile_miou": per_tile,
        "confusion_matrix": total.tolist(),
    }
    if eroded:
        report["eroded"] = {**_scores(total_eroded), "confusion_matrix": total_eroded.tolist()}
    return report


def report_metrics(report: dict, prefix: str) -> dict[str, float]:
    """Flat ``{prefix}_{oa,mf1,miou}[_no_clutter][_eroded]`` metrics for MLflow."""
    out = {}
    sections = [("", report)] + ([("_eroded", report["eroded"])] if "eroded" in report else [])
    for suffix, sec in sections:
        for key, part in (("", "all_classes"), ("_no_clutter", "without_clutter")):
            for metric in ("oa", "mf1", "miou"):
                out[f"{prefix}_{metric}{key}{suffix}"] = sec[part][metric]
    return out


def report_markdown(report: dict, title: str = "Evaluation") -> str:
    a, nc = report["all_classes"], report["without_clutter"]
    lines = [
        f"# {title}",
        "",
        f"Split: `{report['split']}` — tiles: {', '.join(report['tiles'])}",
        "",
        "| | OA | mF1 | mIoU |",
        "|---|---|---|---|",
        f"| all classes | {a['oa']:.4f} | {a['mf1']:.4f} | {a['miou']:.4f} |",
        f"| without clutter | {nc['oa']:.4f} | {nc['mf1']:.4f} | {nc['miou']:.4f} |",
    ]
    if "eroded" in report:
        ea, en = report["eroded"]["all_classes"], report["eroded"]["without_clutter"]
        lines += [
            f"| all classes (eroded) | {ea['oa']:.4f} | {ea['mf1']:.4f} | {ea['miou']:.4f} |",
            f"| without clutter (eroded) | {en['oa']:.4f} | {en['mf1']:.4f} | {en['miou']:.4f} |",
        ]
    lines += [
        "",
        "| Class | IoU | F1 | Precision | Recall | Pixels |",
        "|---|---|---|---|---|---|",
    ]
    for name in CLASS_NAMES:
        c = a["per_class"][name]
        lines.append(
            f"| {name} | {c['iou']:.4f} | {c['f1']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['support']} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path, title: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "report.json", out / "report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report_markdown(report, title), encoding="utf-8")
    return json_path, md_path


def aggregate_reports(paths: list[Path]) -> str:
    """Mean ± std over several runs (e.g. seeds) of ``report.json`` files, grouped by run name."""
    groups: dict[str, list[dict]] = {}
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        groups.setdefault(data.get("name", p.parent.name), []).append(data)
    lines = [
        "| Run | n | OA | mF1 | mIoU | mIoU (no clutter) | mIoU eroded | mIoU eroded (no clutter) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(groups):
        rs = groups[name]

        def ms(key: str, section: str = "all_classes", eroded: bool = False, rs=rs) -> str:
            if eroded and not all("eroded" in r for r in rs):
                return "—"
            v = np.array([(r["eroded"] if eroded else r)[section][key] for r in rs])
            return f"{v.mean():.4f} ± {v.std():.4f}" if len(v) > 1 else f"{v[0]:.4f}"

        lines.append(
            f"| {name} | {len(rs)} | {ms('oa')} | {ms('mf1')} | {ms('miou')} "
            f"| {ms('miou', 'without_clutter')} | {ms('miou', eroded=True)} "
            f"| {ms('miou', 'without_clutter', eroded=True)} |"
        )
    return "\n".join(lines) + "\n"
