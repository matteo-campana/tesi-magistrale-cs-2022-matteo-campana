# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory / speed probe: run a few real training steps on random data with the configured
batch size, crop size and precision, and report peak GPU memory.

Use it before a long run to check that a configuration fits the GPU (e.g. RTX 4060, 8 GB)::

    rsseg benchmark -c configs/experiments/maresunet_r18_imagenet.yaml
    rsseg benchmark -c configs/ssl/dino_r18.yaml --mode dino
"""

from __future__ import annotations

import time

import keras
import numpy as np

from rsseg.config import ExperimentConfig
from rsseg.runtime import setup_runtime


def _cuda():
    import torch

    return torch if torch.cuda.is_available() else None


def _peak_gb() -> float | None:
    torch = _cuda()
    return torch.cuda.max_memory_allocated() / 2**30 if torch else None


def benchmark(cfg: ExperimentConfig, mode: str = "seg", steps: int = 10, warmup: int = 3) -> dict:
    torch = _cuda()
    rng = np.random.default_rng(0)
    if mode == "seg":
        from rsseg.models.segmentor import build_segmentor
        from rsseg.train.losses import MaskedSparseCrossEntropy
        from rsseg.train.seg_train import make_optimizer

        precision = setup_runtime(cfg.train.precision, cfg.train.seed)
        b, s = cfg.train.batch_size, cfg.data.crop_size
        model = build_segmentor(cfg.model)
        model.compile(
            optimizer=make_optimizer(cfg.train),
            loss=MaskedSparseCrossEntropy(cfg.model.num_classes),
        )
        x = rng.uniform(0, 255, (b, s, s, 3)).astype(np.float32)
        y = rng.integers(0, cfg.model.num_classes, (b, s, s)).astype(np.int64)

        def step():
            model.train_on_batch(x, y)

        info = {"batch_size": b, "crop_size": s, "grad_accum_steps": cfg.train.grad_accum_steps}
    elif mode == "dino":
        from rsseg.ssl.dino import DINOModel

        sc = cfg.ssl
        precision = setup_runtime(sc.precision, sc.seed)
        model = DINOModel(sc, steps_per_epoch=100)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0, global_clipnorm=sc.clip_grad or None
            )
        )
        b = sc.batch_size
        g = rng.uniform(0, 255, (b, 2, sc.global_crop_size, sc.global_crop_size, 3)).astype(
            np.float32
        )
        l_ = rng.uniform(
            0, 255, (b, sc.local_crops_number, sc.local_crop_size, sc.local_crop_size, 3)
        ).astype(np.float32)

        def step():
            model.train_on_batch(g, l_)

        info = {
            "batch_size": b,
            "global_crop": sc.global_crop_size,
            "local_crops": sc.local_crops_number,
        }
    else:
        raise ValueError("mode must be 'seg' or 'dino'")

    for _ in range(warmup):
        step()
    if torch:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if torch:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    result = {
        "mode": mode,
        "precision": precision,
        **info,
        "params_millions": round(model.count_params() / 1e6, 2),
        "seconds_per_step": round(dt, 4),
        "images_per_second": round(info["batch_size"] / dt, 2),
        "peak_gpu_memory_gb": round(_peak_gb(), 2) if torch else None,
        "gpu": torch.cuda.get_device_name(0) if torch else "cpu",
    }
    if torch:
        result["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        )
    return result
