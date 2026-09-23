# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-wide runtime setup: precision policy, seeds and CUDA flags."""

from __future__ import annotations

import logging

import keras

log = logging.getLogger(__name__)
PRECISIONS = ("float32", "mixed_bfloat16", "mixed_float16")


def setup_runtime(precision: str = "mixed_bfloat16", seed: int = 0) -> str:
    """Configure Keras/PyTorch and return the precision actually used.

    Mixed precision is only enabled on CUDA: on CPU it is slower and it is switched off.
    ``mixed_bfloat16`` needs compute capability ≥ 8.0 (RTX 30xx/40xx, e.g. RTX 4060).
    """
    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision != "float32" and not torch.cuda.is_available():
        log.warning("CUDA not available: using float32 instead of %s", precision)
        precision = "float32"
    if precision == "mixed_bfloat16" and not torch.cuda.is_bf16_supported():
        log.warning("GPU without bfloat16 support: falling back to mixed_float16")
        precision = "mixed_float16"

    keras.config.set_dtype_policy(precision)
    keras.utils.set_random_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return precision
