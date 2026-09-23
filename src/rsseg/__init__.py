# SPDX-License-Identifier: AGPL-3.0-or-later
"""rsseg — semantic segmentation of aerial images with Keras 3 on the PyTorch backend.

The Keras backend must be chosen before ``keras`` is imported for the first time, so every
entry point imports :mod:`rsseg` first.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

if os.environ["KERAS_BACKEND"] != "torch":  # pragma: no cover - guard for misconfigured shells
    raise RuntimeError(
        f"rsseg requires KERAS_BACKEND=torch, found {os.environ['KERAS_BACKEND']!r}. "
        "Unset the variable or set it to 'torch'."
    )

__version__ = "0.1.0"
