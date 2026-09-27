# SPDX-License-Identifier: AGPL-3.0-or-later
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import keras  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import rsseg  # noqa: E402,F401

# Reference checks use float32 tolerances; TF32 convs on Ampere+ GPUs are ~1e-3 accurate.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@pytest.fixture(autouse=True)
def float32_policy():
    keras.config.set_dtype_policy("float32")
    keras.utils.set_random_seed(0)
    yield


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Synthetic raw tiles prepared with the standard protocol (crop 64)."""
    from rsseg.config import DataConfig
    from rsseg.data.prepare import prepare_dataset
    from rsseg.data.synthetic import make_synthetic_potsdam

    base = tmp_path_factory.mktemp("potsdam")
    raw = make_synthetic_potsdam(base / "raw", size=128)
    cfg = DataConfig(
        raw_dir=str(raw),
        root=str(base / "prepared"),
        crop_size=64,
        train_stride=64,
        num_workers=0,
        prepare_workers=1,
    )
    counts = prepare_dataset(cfg)
    return cfg, counts
