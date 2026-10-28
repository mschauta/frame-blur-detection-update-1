"""The detector's input, shapes and weight-file compatibility; no private data. The published weights are optional."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from blurdetect.model import (BlurDetector, CELL, KERNEL, SCALE_FLOOR, fingerprint, load_detector, model_input,
                               nearest_fill)

ROOT = Path(__file__).resolve().parents[1]
# file: sha256, trunk, channels at stride 16, mask handling
PUBLISHED = {
    "round3_base_p99_ep029.pt": ("9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6", "convnext_base", 512,
                                 "leaky"),
    "round3_small_p99_ep029.pt": ("b715a163f9c6927464d97b7002967c8361c6a0de2b87e3103dc72a7ba733d48b", "convnext_small",
                                  384, "leak-free"),
}


def textured(h=48, w=64, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (1, 3, h, w), dtype=np.uint8))


def test_exact_fingerprint_is_the_float32_formula():
    x = textured()
    lin = x.double() / 255
    lin = torch.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4)
    y = (lin * torch.tensor([0.2126729, 0.7151522, 0.0721750], dtype=torch.float64).view(1, 3, 1, 1)).sum(1, keepdim=True)
    expected = y - torch.nn.functional.conv2d(torch.nn.functional.pad(y, (1, 1, 1, 1), mode="replicate"),
                                              KERNEL.view(1, 1, 3, 3))
    assert torch.allclose(fingerprint(x, exact=True).double(), expected, atol=1e-6)


def test_bf16_rounded_fingerprint_is_deterministic_and_differs_from_exact():
    x = textured()
    rounded = fingerprint(x)
    assert torch.equal(rounded, fingerprint(x))
    assert not torch.equal(rounded, fingerprint(x, exact=True))
    assert (rounded - fingerprint(x, exact=True)).abs().max() < 0.01


@pytest.mark.parametrize("kind, channels", [("p99", 2), ("rgb", 4), ("gray", 4)])
def test_model_input_channels_and_validity_channel(kind, channels):
    x = textured()
    mask = torch.ones((1, 1, 48, 64))
    mask[..., :, :20] = 0
    out = model_input(x, mask, kind, exact=True)
    assert out.shape == (1, channels, 48, 64)
    assert torch.equal(out[:, -1:], mask)
    assert (out[:, :-1][mask.expand(-1, channels - 1, -1, -1) == 0] == 0).all()


def test_flat_image_uses_the_scale_floor():
    x = torch.full((1, 3, 32, 32), 120, dtype=torch.uint8)
    out = model_input(x, None, "p99", exact=True)
    # No edges: only float32 rounding remains (|d| ~ 1e-8); the floor keeps it near zero instead of blowing it up.
    assert torch.isfinite(out).all() and out[:, 0].abs().max() < 0.01
    assert SCALE_FLOOR == 1e-3


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(0)
    return BlurDetector("p99").eval()


def test_forward_shapes_and_cell_validity(small):
    with torch.no_grad():
        out = small(textured(40, 50))
    assert out["local"].shape == (1, 1, 3, 4)                       # ceil(40 / 16), ceil(50 / 16)
    assert torch.isfinite(out["image"]).all()


def test_a_frame_without_a_valid_cell_cannot_be_judged(small):
    with pytest.raises(ValueError, match="no valid"):
        small(textured(10, 10))                                       # one padded cell, under half valid
    with pytest.raises(ValueError, match="no valid"):
        small(textured(32, 32), torch.zeros((1, 1, 32, 32)))


def test_last_half_cell_row_of_1080p_counts():
    # 1080 = 67.5 cells: the padded last row is half valid and counts (>= 0.5) - a documented edge choice.
    valid = torch.nn.functional.pad(torch.ones((1, 1, 1080, CELL)), (0, 0, 0, 8))
    cell_valid = torch.nn.functional.avg_pool2d(valid, CELL, CELL) >= 0.5
    assert cell_valid.shape[-2] == 68 and bool(cell_valid[..., -1, :].all())


def test_old_weight_files_keep_their_meaning(tmp_path, small):
    path = tmp_path / "old.pt"
    torch.save({"state_dict": small.state_dict(), "meta": {"input": "p99"}}, path)    # no arch / fingerprint keys
    model, meta = load_detector(path)
    assert not model.exact and not model.leak_free and model.head[0].in_channels == 384   # Small, bf16, leaky mask
    x = textured(32, 48)
    with torch.no_grad():
        assert torch.equal(model(x)["local"], small(x)["local"])
    base = BlurDetector("p99", "convnext_base", "fp32")
    assert base.exact and base.head[0].in_channels == 512


@pytest.mark.xfail(strict=True, reason="known: the fingerprint of a valid pixel next to a mask is computed from masked "
                   "neighbours, and those pixels enter the p99 scale (README, Limitations); the published weights were "
                   "trained with this behaviour, so it is documented, not changed")
def test_masked_content_does_not_reach_the_valid_region():
    x = torch.full((1, 3, 32, 32), 120, dtype=torch.uint8)
    mask = torch.ones((1, 1, 32, 32))
    mask[..., :16] = 0
    x2 = x.clone()
    x2[..., :16] = 255
    a = model_input(x, mask, "p99", exact=True)
    b = model_input(x2, mask, "p99", exact=True)
    assert torch.allclose(a * mask, b * mask, atol=1e-5)


def test_leak_free_masked_content_does_not_reach_the_valid_region():
    x = textured(32, 32)
    mask = torch.ones((1, 1, 32, 32))
    mask[..., 8:20, :16] = 0
    x2 = x.clone()
    x2[..., 8:20, :16] = textured(32, 32, seed=1)[..., 8:20, :16]
    a = model_input(x, mask, "p99", exact=True, leak_free=True)
    b = model_input(x2, mask, "p99", exact=True, leak_free=True)
    assert torch.equal(a, b)
    assert not torch.equal(model_input(x, mask, "p99", exact=True), model_input(x2, mask, "p99", exact=True))


def test_leak_free_without_a_mask_is_the_leaky_input():
    x = textured()
    assert torch.equal(model_input(x, None, "p99", exact=True, leak_free=True), model_input(x, None, "p99", exact=True))
    ones = torch.ones((1, 1, 48, 64))
    assert torch.equal(model_input(x, ones, "p99", exact=True, leak_free=True), model_input(x, ones, "p99", exact=True))


def test_nearest_fill_takes_the_nearest_valid_colour():
    rgb = np.zeros((5, 7, 3), np.uint8)
    rgb[:, :2] = 10                                    # valid, left
    rgb[:, 5:] = 200                                   # valid, right
    rgb[:, 2:5] = 99                                   # masked
    valid = np.ones((5, 7), np.uint8)
    valid[:, 2:5] = 0
    out = nearest_fill(rgb, valid)
    assert (out[:, 2] == 10).all() and (out[:, 4] == 200).all() and (out[:, 3] != 99).all()
    assert np.array_equal(out[valid == 1], rgb[valid == 1])


@pytest.mark.parametrize("name", PUBLISHED)
def test_published_weight_file(name):
    path = ROOT / "weights" / name
    if not path.is_file():
        pytest.skip(f"{name} is a release asset; put it into weights/")
    sha, arch, width, mask = PUBLISHED[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    model, meta = load_detector(path)
    assert (meta["arch"], meta["fingerprint"], meta["input"], meta["epoch"]) == (arch, "fp32", "p99", 29)
    assert meta.get("mask", "leaky") == mask
    assert set(meta["thresholds"]) == {"r90", "r95", "r98"}
    assert model.exact and model.leak_free == (mask == "leak-free") and model.head[0].in_channels == width
    with torch.no_grad():
        out = model(textured(64, 96))
    assert out["local"].shape == (1, 1, 4, 6) and torch.isfinite(out["image"]).all()
