# SPDX-License-Identifier: AGPL-3.0-or-later
import keras
import numpy as np
import pytest
import torch
import torchvision
from keras import ops

from rsseg.config import ModelConfig, VQConfig
from rsseg.models.attention import ChannelAttention, LinearPAM, linear_pam_reference
from rsseg.models.backbones import IMAGENET_MEAN, IMAGENET_STD, build_resnet, feature_channels
from rsseg.models.quantizer import VectorQuantizer, find_quantizers
from rsseg.models.segmentor import build_segmentor
from rsseg.models.torchvision_weights import port_resnet_state_dict


def _np(x):
    return ops.convert_to_numpy(x)


@pytest.mark.parametrize("arch", ["resnet18", "resnet34"])
def test_backbone_matches_torchvision(arch):
    net = getattr(torchvision.models, arch)(weights=None).eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    model = build_resnet(arch)
    port_resnet_state_dict(model, net.state_dict())

    x = np.random.default_rng(0).uniform(0, 255, (2, 64, 96, 3)).astype(np.float32)
    xt = (
        (torch.tensor(x) / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    ).permute(0, 3, 1, 2)
    with torch.no_grad():
        f = net.maxpool(net.relu(net.bn1(net.conv1(xt))))
        expected = []
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            f = layer(f)
            expected.append(f.permute(0, 2, 3, 1).numpy())
    for got, exp in zip(model(x, training=False), expected, strict=True):
        np.testing.assert_allclose(_np(got), exp, rtol=1e-3, atol=1e-3 * np.abs(exp).max())


def test_feature_channels():
    assert feature_channels("resnet18") == [64, 128, 256, 512]
    assert feature_channels("resnet50") == [256, 512, 1024, 2048]


def test_linear_pam_matches_reference_formula():
    x = np.random.default_rng(1).normal(size=(2, 5, 6, 16)).astype(np.float32)
    layer = LinearPAM()
    layer(x)
    layer.gamma.assign([0.7])
    wq, bq = [_np(w) for w in layer.query_conv.weights]
    wk, bk = [_np(w) for w in layer.key_conv.weights]
    wv, bv = [_np(w) for w in layer.value_conv.weights]
    expected = linear_pam_reference(
        x.astype(np.float64), wq[0, 0], bq, wk[0, 0], bk, wv[0, 0], bv, 0.7
    )
    np.testing.assert_allclose(_np(layer(x)), expected, rtol=1e-4, atol=1e-5)


def test_linear_pam_equals_first_order_taylor_attention():
    """Independent check of the O(N) formula against the explicit O(N²) weights 1 + q̂·k̂."""
    rng = np.random.default_rng(2)
    q, k, v = rng.normal(size=(10, 3)), rng.normal(size=(10, 3)), rng.normal(size=(10, 4))
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    kn = k / np.linalg.norm(k, axis=1, keepdims=True)
    weights = 1 + qn @ kn.T
    explicit = (weights @ v) / weights.sum(1, keepdims=True)
    linear = (v.sum(0) + qn @ (kn.T @ v)) / (len(q) + qn @ kn.sum(0))[:, None]
    np.testing.assert_allclose(linear, explicit, rtol=1e-10)


def test_channel_attention_reference():
    x = np.random.default_rng(3).normal(size=(1, 4, 4, 8)).astype(np.float32)
    layer = ChannelAttention()
    layer(x)
    layer.gamma.assign([0.5])
    f = x.reshape(1, 16, 8)[0].T.astype(np.float64)  # (c, n)
    energy = f @ f.T
    energy = energy.max(-1, keepdims=True) - energy
    att = np.exp(energy - energy.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    expected = x + 0.5 * (att @ f).T.reshape(1, 4, 4, 8)
    np.testing.assert_allclose(_np(layer(x)), expected, rtol=1e-4, atol=1e-4)


def test_vq_straight_through_gradient_is_identity():
    vq = VectorQuantizer(num_codes=8, ema_decay=0.0)
    x = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(2, 3, 3, 4)), "float32")
    x.requires_grad_(True)  # on the Keras default device (CPU or CUDA)
    out = vq(x, training=False)
    out.sum().backward()
    np.testing.assert_allclose(_np(x.grad), np.ones((2, 3, 3, 4)))
    # the forward output is the nearest code
    _, _, q = vq.lookup(ops.reshape(x.detach(), (-1, 4)))
    np.testing.assert_allclose(_np(out).reshape(-1, 4), _np(q), atol=1e-6)


def test_vq_loss_only_in_training():
    vq = VectorQuantizer(num_codes=8)
    x = np.random.default_rng(0).normal(size=(2, 3, 3, 4)).astype(np.float32)
    vq(x, training=False)
    assert len(vq.losses) == 0
    vq(x, training=True)
    assert len(vq.losses) == 1


def test_vq_ema_learns_clusters_and_uses_codes():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=(8, 4)).astype(np.float32)
    vq = VectorQuantizer(num_codes=16, ema_decay=0.9, seed=0)
    for _ in range(200):
        labels = rng.integers(0, 8, 512)
        x = (centers[labels] + rng.normal(0, 0.1, (512, 4))).astype(np.float32).reshape(8, 8, 8, 4)
        vq(x, training=True)
    codebook = _np(vq.codebook)
    dist = np.linalg.norm(centers[:, None] - codebook[None], axis=-1).min(1)
    assert dist.max() < 0.5, dist  # every cluster has a code close to its centre
    assert float(_np(vq.perplexity)) > 6


def test_segmentor_variants_shapes_and_losses():
    x = np.random.default_rng(0).uniform(0, 255, (1, 64, 96, 3)).astype(np.float32)
    plain = build_segmentor(ModelConfig(weights="none", skip="plain"))
    assert tuple(plain(x).shape) == (1, 64, 96, 6)
    vq = build_segmentor(
        ModelConfig(
            weights="none", neck="vq", vq=VQConfig(num_codes=16, code_dim=32, on_skips=True)
        )
    )
    out = vq(x, training=True)
    assert out.dtype == torch.float32
    assert len(find_quantizers(vq)) == 4 and len(vq.losses) == 4


def test_mixed_precision_logits_are_float32():
    keras.config.set_dtype_policy("mixed_bfloat16")
    model = build_segmentor(ModelConfig(weights="none"))
    out = model(np.zeros((1, 64, 64, 3), np.float32))
    assert out.dtype == torch.float32
