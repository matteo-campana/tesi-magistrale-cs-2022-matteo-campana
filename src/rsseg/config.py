# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment configuration: typed dataclasses loaded from composable YAML files.

A YAML file may declare ``base: other.yaml`` (path relative to the file) to inherit from
another configuration; values are deep-merged. Command-line overrides use dotted keys,
e.g. ``--set train.epochs=5 model.arch=resnet34``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "potsdam"
    """``potsdam`` or ``vaihingen``: file layout (:mod:`rsseg.data.sources`) and splits."""
    raw_dir: str | None = None
    """Directory with the original ISPRS files (searched recursively)."""
    root: str = "data/potsdam"
    """Output directory of ``rsseg prepare`` (crops, tile masks, manifests)."""
    protocol: str = "standard"
    """``standard`` (literature split) or ``thesis2022`` (split used in the thesis, Potsdam only)."""
    val_tiles: list[str] | None = None
    """Override the validation tiles (held out entirely from the training tiles)."""
    exclude_7_10: bool = False
    image_pattern: str | None = None
    """Override the image regex of the dataset source (e.g. ``_IRRG\\.tif$`` for Potsdam IRRG)."""
    label_pattern: str | None = None
    """Override the label regex (and its priority list) of the dataset source."""
    train_labels: str = "full"
    """Labels of the training/validation crops: ``full`` (thesis) or ``eroded`` (no boundaries)."""
    max_offpalette: float = 0.005
    """Maximum fraction of label pixels outside the ISPRS palette before a tile is rejected."""
    ndsm: bool = False
    """Add the normalised DSM (height above ground) as a 4th input channel."""
    ndsm_source: str = "auto"
    """``official`` (Potsdam nDSM JPGs), ``computed`` (from the DSM) or ``auto`` (official if any)."""
    ndsm_variant: str = "lastools"
    """Official Potsdam nDSM variant: ``lastools`` or ``ownapproach``."""
    crop_size: int = 512
    train_stride: int = 512
    """Stride of training crops. Overlap is allowed only inside the training split."""
    label_fraction: float = 1.0
    """Fraction of labelled training crops (sampled per tile) — low-label regime for RQ3."""
    augment: bool = True
    num_workers: int = 4
    prepare_workers: int = 4


@dataclass
class VQConfig:
    num_codes: int = 512
    code_dim: int | None = None
    """Dimension of the codes; if different from the bottleneck channels a 1x1 projection is added."""
    beta: float = 0.25
    ema_decay: float = 0.99
    """EMA decay of the codebook; 0 disables EMA and learns the codebook by gradient."""
    loss_weight: float = 1.0
    restart_dead_codes: bool = True
    dead_code_threshold: float = 0.1
    """Codes whose EMA count of assignments per batch falls below this value are restarted."""
    on_skips: bool = False
    """Also quantize the skip connections (otherwise the decoder can bypass the codebook)."""


@dataclass
class ModelConfig:
    arch: str = "resnet18"
    """Encoder: ``resnet18/34/50``, ``swin_t/s/b`` or ``vit_s_16/vit_b_16/vit_l_16``
    (``rsseg backbones`` lists them)."""
    weights: str = "imagenet"
    """``imagenet``, ``none`` or a path to a ``.weights.h5`` backbone exported by ``rsseg pretrain``."""
    drop_path: float = 0.2
    """Maximum stochastic-depth rate of transformer encoders (ignored by ResNets)."""
    skip: str = "lam"
    """``lam`` (linear PAM + CAM as in MAResU-Net) or ``plain`` (U-Net-like identity skips)."""
    neck: str = "identity"
    """``identity`` or ``vq``."""
    vq: VQConfig = field(default_factory=VQConfig)
    num_classes: int = 6
    dropout: float = 0.1
    in_channels: int = 3
    """Input channels: 3 (RGB/IRRG) or 4 (+ nDSM). Set automatically from ``data.ndsm``."""
    extra_channel_mean: list[float] | None = None
    """Normalisation of the channels after the first three (default: ``stats.json`` of the data)."""
    extra_channel_std: list[float] | None = None
    extra_channel_init: str = "mean"
    """First-layer weights of the extra channels when inflating a 3-channel encoder:
    ``mean`` (mean of the RGB kernels) or ``zeros`` (the pre-trained network is unchanged)."""


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 4
    """Gradient accumulation: effective batch = batch_size * grad_accum_steps (thesis: 32)."""
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "step"
    """``step`` (StepLR 15/0.4 as in the thesis), ``cosine`` or ``constant``."""
    step_size: int = 15
    gamma: float = 0.4
    min_lr: float = 1e-6
    precision: str = "mixed_bfloat16"
    """Keras dtype policy: ``mixed_bfloat16`` (RTX 30xx/40xx), ``mixed_float16`` or ``float32``."""
    seed: int = 0
    class_weights: list[float] | None = None
    steps_per_epoch: int | None = None
    """Limit the number of training batches per epoch (``None`` = full dataset)."""
    validation_steps: int | None = None
    early_stopping_patience: int | None = 20
    """Stop after this many epochs without a better ``val_miou`` (``None`` disables it). Keep it
    above ``step_size``: a learning-rate drop often restarts the improvements."""
    keep_top_k: int = 5
    """Checkpoints kept in ``<run>/checkpoints``: the ``k`` epochs with the highest ``val_miou``."""
    log_images_every: int = 10


@dataclass
class EvalConfig:
    split: str = "test"
    window: int = 512
    stride: int = 256
    batch_size: int = 8
    tta: bool = False
    max_tiles: int | None = None
    save_predictions: bool = True


@dataclass
class SSLConfig:
    arch: str = "resnet18"
    weights: str = "none"
    drop_path: float = 0.1
    """Maximum stochastic-depth rate of transformer encoders (DINO uses 0.1 for ViT)."""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    """Base learning rate, scaled by ``batch_size / 256`` (DINO linear scaling rule)."""
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    momentum_teacher: float = 0.996
    out_dim: int = 16384
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    student_temp: float = 0.1
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    center_momentum: float = 0.9
    global_crop_size: int = 224
    local_crop_size: int = 96
    local_crops_number: int = 6
    global_crops_scale: list[float] = field(default_factory=lambda: [0.14, 1.0])
    local_crops_scale: list[float] = field(default_factory=lambda: [0.05, 0.4])
    aerial_augment: bool = False
    """Milder colour augmentation plus vertical flips / 90° rotations (RQ4)."""
    clip_grad: float = 3.0
    precision: str = "mixed_bfloat16"
    seed: int = 0
    steps_per_epoch: int | None = None
    export_path: str | None = None
    """Where to save the student backbone (default: ``<run dir>/backbone.weights.h5``)."""
    extra_pools: list[str] = field(default_factory=list)
    """Unlabelled crop roots made by ``rsseg prepare-unlabeled``, added to the training crops."""
    allow_same_city_pool: bool = False
    """Allow a pool from the city of ``data.dataset``: its images may cover the test areas."""


@dataclass
class TrackingConfig:
    tracking_uri: str | None = None
    """MLflow tracking URI; ``None`` uses ``MLFLOW_TRACKING_URI`` or ``sqlite:///mlflow.db``."""
    experiment: str = "rsseg"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def flat(self) -> dict[str, Any]:
        """Flattened ``section.key -> value`` view, used for MLflow parameters."""
        return _flatten(self.to_dict())


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    if base is not None:
        raw = _deep_merge(_load_yaml((path.parent / base).resolve()), raw)
    return raw


def _build(cls: type, values: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        t = hints[k]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            kwargs[k] = _build(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Turn ``["train.epochs=5", "model.arch=resnet34"]`` into a nested dict."""
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = result
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return result


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    values: dict[str, Any] = _load_yaml(Path(path)) if path else {}
    values = _deep_merge(values, parse_overrides(overrides))
    return _build(ExperimentConfig, values)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
