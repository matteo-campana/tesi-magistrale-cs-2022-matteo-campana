# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path

import keras
import mlflow
import numpy as np
import pytest
import torch
from keras import ops

from rsseg.config import ExperimentConfig, SSLConfig, load_config
from rsseg.data.classes import IGNORE_INDEX
from rsseg.eval.sliding_window import predict_tile
from rsseg.ssl.dino import DINOModel, cosine_schedule, dino_loss
from rsseg.train.losses import MaskedSparseCrossEntropy
from rsseg.train.metrics import (
    MeanF1,
    MeanIoU,
    OverallAccuracy,
    confusion_matrix_np,
    scores_from_confusion,
)
from rsseg.train.seg_train import lr_at_epoch


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: a\ntrain:\n  epochs: 7\n  lr: 0.5\n", encoding="utf-8"
    )
    (tmp_path / "exp.yaml").write_text(
        "base: base.yaml\nname: b\ntrain:\n  lr: 0.1\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path / "exp.yaml", ["train.batch_size=3", "model.vq.num_codes=64"])
    assert (
        cfg.name,
        cfg.train.epochs,
        cfg.train.lr,
        cfg.train.batch_size,
        cfg.model.vq.num_codes,
    ) == ("b", 7, 0.1, 3, 64)
    with pytest.raises(KeyError):
        load_config(None, ["train.not_a_key=1"])


def test_repository_configs_load():
    for path in Path("configs").rglob("*.yaml"):
        assert isinstance(load_config(path), ExperimentConfig), path


def test_loss_ignores_only_255_and_trains_class_zero():
    logits = np.random.default_rng(0).normal(size=(2, 4, 4, 6)).astype(np.float32)
    labels = np.random.default_rng(1).integers(0, 6, (2, 4, 4))
    loss = MaskedSparseCrossEntropy(6)
    ref = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)
    np.testing.assert_allclose(
        ops.convert_to_numpy(loss(labels, logits)), ops.convert_to_numpy(ref), rtol=1e-5
    )

    ignored = labels.copy()
    ignored[0] = IGNORE_INDEX
    ref_half = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels[1:], logits[1:])
    np.testing.assert_allclose(
        ops.convert_to_numpy(loss(ignored, logits)), ops.convert_to_numpy(ref_half), rtol=1e-5
    )

    # Regression for the thesis bug (ignore_index=0): class 0 must produce a gradient.
    zeros = np.zeros((1, 2, 2), dtype=np.int64)
    t = torch.zeros((1, 2, 2, 6), requires_grad=True)
    loss(zeros, t).backward()
    assert t.grad.abs().sum() > 0


def test_scores_from_confusion():
    cm = np.array([[8, 2], [1, 9]])
    s = scores_from_confusion(cm, class_names=("a", "b"))
    assert s["oa"] == pytest.approx(17 / 20)
    assert s["per_class"]["a"]["iou"] == pytest.approx(8 / 11)
    assert s["miou"] == pytest.approx((8 / 11 + 9 / 12) / 2)
    assert s["mf1"] == pytest.approx((16 / 19 + 18 / 21) / 2)
    assert scores_from_confusion(cm, class_names=("a", "b"), exclude=(1,))["miou"] == pytest.approx(
        8 / 11
    )


def test_keras_metrics_match_numpy_over_batches():
    rng = np.random.default_rng(0)
    metrics = [MeanIoU(6), MeanF1(6), OverallAccuracy(6)]
    cm = np.zeros((6, 6), dtype=np.int64)
    for _ in range(3):
        y = rng.integers(0, 6, (2, 8, 8))
        y[0, 0] = IGNORE_INDEX
        logits = rng.normal(size=(2, 8, 8, 6)).astype(np.float32)
        for m in metrics:
            m.update_state(y, logits)
        cm += confusion_matrix_np(y, logits.argmax(-1), 6)
    s = scores_from_confusion(cm)
    got = [float(ops.convert_to_numpy(m.result())) for m in metrics]
    np.testing.assert_allclose(got, [s["miou"], s["mf1"], s["oa"]], rtol=1e-6)


def test_lr_schedules():
    cfg = ExperimentConfig().train
    assert lr_at_epoch(cfg, 0) == cfg.lr
    assert lr_at_epoch(cfg, 15) == pytest.approx(cfg.lr * 0.4)
    cfg.scheduler = "cosine"
    assert lr_at_epoch(cfg, cfg.epochs) == pytest.approx(cfg.min_lr)


def test_dino_loss_is_per_sample():
    """The loss must be the mean of independent per-sample losses (no mixing across the batch)."""
    rng = np.random.default_rng(0)
    b, k, views = 4, 16, 5
    s = rng.normal(size=(views * b, k)).astype(np.float32)
    t = rng.normal(size=(2 * b, k)).astype(np.float32)
    center = np.zeros((1, k), np.float32)
    full = float(ops.convert_to_numpy(dino_loss(s, t, center, 0.1, 0.04, views)))
    per_sample = []
    for i in range(b):
        si = s.reshape(views, b, k)[:, i]
        ti = t.reshape(2, b, k)[:, i]
        per_sample.append(float(ops.convert_to_numpy(dino_loss(si, ti, center, 0.1, 0.04, views))))
    assert full == pytest.approx(np.mean(per_sample), rel=1e-5)


def test_cosine_schedule_endpoints():
    sched = cosine_schedule(1.0, 0.0, 100, warmup_steps=10)
    assert len(sched) == 100 and sched[0] == 0.0 and sched[10] == pytest.approx(1.0)
    assert sched[-1] == pytest.approx(0.0, abs=1e-3)


def test_dino_train_step_updates_student_and_ema_teacher():
    cfg = SSLConfig(weights="none", epochs=2, batch_size=2, out_dim=32, hidden_dim=16, bottleneck_dim=8,
                    global_crop_size=64, local_crop_size=32, local_crops_number=2, precision="float32")  # fmt: skip
    model = DINOModel(cfg, steps_per_epoch=2)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.0, global_clipnorm=3.0))
    for s, t in zip(model.student.weights, model.teacher.weights, strict=True):
        np.testing.assert_array_equal(ops.convert_to_numpy(s), ops.convert_to_numpy(t))
    student_before = [ops.convert_to_numpy(w).copy() for w in model.student.trainable_weights]
    teacher_before = [ops.convert_to_numpy(w).copy() for w in model.teacher.weights]

    rng = np.random.default_rng(0)
    g = rng.uniform(0, 255, (2, 2, 64, 64, 3)).astype(np.float32)
    loc = rng.uniform(0, 255, (2, 2, 32, 32, 3)).astype(np.float32)
    model.train_on_batch(g, loc)  # step 0: lr is 0 during warm-up
    model.train_on_batch(g, loc)

    assert all(not w.trainable for w in model.teacher.weights)
    changed = [
        not np.allclose(a, ops.convert_to_numpy(w))
        for a, w in zip(student_before, model.student.trainable_weights, strict=True)
    ]
    assert any(changed)
    t_changed = [
        not np.allclose(a, ops.convert_to_numpy(w))
        for a, w in zip(teacher_before, model.teacher.weights, strict=True)
    ]
    assert any(t_changed)
    assert model.teacher_entropy(g) > 0


class _ColourModel(keras.Model):
    """Predicts class = argmax of the RGB channels: a deterministic per-pixel model."""

    def call(self, x, training=False):
        logits = ops.concatenate([x, ops.zeros_like(x)], axis=-1) / 255.0 * 50.0
        return logits


def test_sliding_window_equals_direct_prediction_for_pixelwise_model():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 330, 3)).astype(np.uint8)
    pred = predict_tile(_ColourModel(), image, num_classes=6, window=64, stride=32, batch_size=5)
    assert pred.shape == (200, 330)
    np.testing.assert_array_equal(pred, image.argmax(-1))


def test_end_to_end_train_and_pretrain(synthetic_dataset, tmp_path, monkeypatch):
    from rsseg.ssl.train import pretrain_dino
    from rsseg.train.seg_train import train_segmentation

    data_cfg, _ = synthetic_dataset
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = load_config("configs/smoke.yaml", [
        f"output_dir={tmp_path / 'runs'}", f"tracking.tracking_uri={uri}", f"data.root={data_cfg.root}",
        "data.crop_size=64", "train.epochs=1", "train.steps_per_epoch=2", "train.validation_steps=1",
        "eval.window=64", "eval.stride=32", "eval.max_tiles=1",
        f"ssl.export_path={tmp_path / 'bb.weights.h5'}", "ssl.global_crop_size=64", "ssl.local_crop_size=32",
    ])  # fmt: skip
    result = train_segmentation(cfg)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "report.json").exists() and (run_dir / "best.weights.h5").exists()
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["split"] == "test"

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(result["run_id"])
    assert "val_miou" in run.data.metrics and "test_miou" in run.data.metrics

    pre = pretrain_dino(cfg)
    assert Path(pre["backbone_weights"]).exists()
    cfg.model.weights = pre["backbone_weights"]
    cfg.eval.split = ""
    assert train_segmentation(cfg)["best_val_miou"] >= 0
