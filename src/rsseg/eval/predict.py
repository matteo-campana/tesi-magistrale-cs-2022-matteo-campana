# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference with a trained model on arbitrary images or on the tiles of a split.

For every image ``<out>/<name>_pred.png`` (class indices), ``<name>_color.png`` (ISPRS palette)
and ``<name>_confidence.png`` (probability of the predicted class, 0–255) are written; with a
ground truth (``--split``) also ``<name>_errors.png`` and the scores. ``<out>/predictions.json``
lists the files, the class fractions, the times and the scores (per image and overall).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rsseg import control, progress
from rsseg.config import ExperimentConfig
from rsseg.data.classes import CLASS_NAMES, NUM_CLASSES, index_to_rgb
from rsseg.data.datasets import load_image, read_tiles
from rsseg.data.prepare import load_mask
from rsseg.eval.sliding_window import predict_tile
from rsseg.eval.visualize import error_map
from rsseg.train.metrics import confusion_matrix_np, scores_from_confusion

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def collect_images(inputs: list[str | Path]) -> list[Path]:
    """Files of ``inputs``: images as given, folders expanded (sorted, not recursive)."""
    files: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            files += sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif item.exists():
            files.append(item)
        else:
            raise FileNotFoundError(item)
    return files


def load_model(cfg: ExperimentConfig, weights: str | Path):
    """Segmentation model of ``cfg`` with ``weights`` (the encoder is not downloaded)."""
    from rsseg.models.segmentor import build_segmentor
    from rsseg.train.seg_train import model_config

    model_cfg = dataclasses.replace(model_config(cfg), weights="none")
    model = build_segmentor(model_cfg)
    model.load_weights(str(weights))
    return model


def predict_images(
    cfg: ExperimentConfig,
    weights: str | Path,
    inputs: list[str | Path],
    out: str | Path,
    split: str | None = None,
    ndsm: list[str | Path] | None = None,
    window: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    tta: bool = False,
) -> dict:
    from rsseg.runtime import setup_runtime

    setup_runtime(cfg.train.precision, cfg.train.seed)
    window, stride = window or cfg.eval.window, stride or cfg.eval.stride
    batch_size = batch_size or cfg.eval.batch_size
    out = Path(out).resolve()  # predictions.json lists absolute paths
    out.mkdir(parents=True, exist_ok=True)
    progress.report(phase="startup", stage="model", run_dir=str(out), kind="predict",
                    name=cfg.name)  # fmt: skip
    model = load_model(cfg, weights)
    channels = model.input_shape[-1]

    items: list[dict] = []
    if split:
        root = Path(cfg.data.root)
        for r in read_tiles(root, split):
            items.append({"name": r["tile"], "image": r["image"],
                          "ndsm": str(root / r["ndsm"]) if r.get("ndsm") else None,
                          "mask": str(root / r["mask"])})  # fmt: skip
    files = collect_images(inputs)
    ndsm = list(ndsm or [])
    if ndsm and len(ndsm) != len(files):
        raise ValueError(f"{len(ndsm)} nDSM images for {len(files)} images")
    for i, f in enumerate(files):
        items.append({"name": f.stem, "image": str(f), "ndsm": str(ndsm[i]) if ndsm else None,
                      "mask": None})  # fmt: skip
    if not items:
        raise ValueError("no images to predict")
    if channels == 4 and not all(it["ndsm"] for it in items):
        raise ValueError("the model takes RGB + nDSM: give --ndsm for every image (or --split "
                         "on data prepared with data.ndsm=true)")  # fmt: skip

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    results = []
    t_start = time.time()
    for n, it in enumerate(items):
        control.check(pausable=False)
        progress.report(phase="predict", stage=it["name"], done=n, total=len(items),
                        window=0, windows=None)  # fmt: skip
        t0 = time.time()
        image = load_image(it["image"], it["ndsm"] if channels == 4 else None)

        def on_windows(done: int, total: int) -> None:
            progress.report(window=done, windows=total)
            control.check(pausable=False)

        pred, conf = predict_tile(model, image, NUM_CLASSES, window, stride, batch_size, tta,
                                  confidence=True, on_progress=on_windows)  # fmt: skip
        base = out / it["name"]
        Image.fromarray(pred).save(f"{base}_pred.png")
        Image.fromarray(index_to_rgb(pred)).save(f"{base}_color.png")
        Image.fromarray((conf * 255).round().astype(np.uint8)).save(f"{base}_confidence.png")
        counts = np.bincount(pred.reshape(-1), minlength=NUM_CLASSES)
        entry = {
            "name": it["name"],
            "image": str(Path(it["image"]).resolve()),
            "pred": f"{base}_pred.png",
            "color": f"{base}_color.png",
            "confidence": f"{base}_confidence.png",
            "size": list(pred.shape),
            "seconds": time.time() - t0,
            "mean_confidence": float(conf.mean()),
            "class_fractions": {
                c: float(counts[i] / counts.sum()) for i, c in enumerate(CLASS_NAMES)
            },  # fmt: skip
        }
        if it["mask"]:
            gt = load_mask(it["mask"])
            cm = confusion_matrix_np(gt, pred, NUM_CLASSES)
            total_cm += cm
            Image.fromarray(error_map(gt, pred)).save(f"{base}_errors.png")
            entry |= {"mask": str(Path(it["mask"]).resolve()), "errors": f"{base}_errors.png",
                      "scores": scores_from_confusion(cm), "confusion_matrix": cm.tolist()}  # fmt: skip
        results.append(entry)
        log.info("%s: %d×%d in %.1f s", it["name"], *pred.shape, entry["seconds"])
        progress.report(done=n + 1)

    summary: dict = {
        "weights": str(weights),
        "config": cfg.name,
        "window": window,
        "stride": stride,
        "tta": tta,
        "seconds": time.time() - t_start,
        "images": results,
    }
    if total_cm.sum():
        summary["scores"] = scores_from_confusion(total_cm)
        summary["confusion_matrix"] = total_cm.tolist()
    (out / "predictions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"out": str(out), "images": len(results), "seconds": summary["seconds"],
            **({"miou": summary["scores"]["miou"]} if "scores" in summary else {})}  # fmt: skip
