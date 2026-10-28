"""The blur detector and its three input types, for inference.

A ConvNeXt trunk (Small by default; Base / Large when the weight file says so) up to stride 16, a 1x1 head that gives one logit per 16 px cell, and a masked log-sum-exp over
the valid cells with a learnable sharpness r, normalised by the number n of valid cells: a differentiable soft
maximum, not a logical "any cell" rule. A dominant local logit contributes approximately local - log(n) / r, so both
the local score and the extent of a high-scoring region affect the image decision (METHOD.md, Section 8).

Inputs (METHOD.md, Section 7), always followed by a validity-mask channel:
  rgb   the image, ImageNet-normalised
  gray  the linear-light luminance, sRGB-encoded, copied to 3 channels, ImageNet-normalised
  p99   the edge fingerprint, based on d = Y - K * Y (K = [3 10 3; 10 92 10; 3 10 3] / 144, Y linear luminance)
        with the bfloat16 rounding documented in fingerprint() (or exactly in float32, for weights trained on it);
        divided by the 99.5th percentile of |d| over valid pixels of a 2 x 2 sub-grid (floor 1e-3) and companded as
        sign * sqrt(|.|)

A weight file's meta may name the trunk ("arch": convnext_small / convnext_base / convnext_large), the fingerprint
precision ("fingerprint": "bf16-rounded" / "fp32") and the mask handling ("mask": "leaky" / "leak-free"); without
them it is the published ConvNeXt-Small with the bfloat16-rounded fingerprint and the leaky mask, so the original
weight files load and score exactly as before.

mask "leaky": the fingerprint of a valid pixel next to the mask is computed from masked neighbours, and those pixels
enter the p99 scale (README, Limitations). "leak-free", for weights trained that way: the masked pixels first take the
colour of the nearest valid pixel (as the image border is replicated), and the p99 scale counts only valid pixels
whose 3 x 3 neighbourhood holds no masked pixel, so no masked content reaches the valid region. Without a mask the two
are identical.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL = 16
KERNEL = torch.tensor([[3.0, 10.0, 3.0], [10.0, 92.0, 10.0], [3.0, 10.0, 3.0]], dtype=torch.float64) / 144.0
LUMA = (0.2126729, 0.7151522, 0.0721750)
SCALE_Q, SCALE_FLOOR = 0.995, 1e-3
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WIDTH = {"convnext_small": 384, "convnext_base": 512, "convnext_large": 768}      # channels at stride 16
FINGERPRINTS = ("bf16-rounded", "fp32")
MASKS = ("leaky", "leak-free")


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def luminance(lin: torch.Tensor) -> torch.Tensor:
    return (lin * lin.new_tensor(LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)


def fingerprint(rgb8: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """(B, 3, H, W) uint8 sRGB -> (B, 1, H, W) d = Y - K * Y, edges replicated.

    The published models were trained with this convolution running under bfloat16 autocast on the GPU: its inputs, the
    kernel and its output were rounded to bfloat16, while Y itself stayed float32. The rounding is reproduced here
    explicitly, on any device and with or without autocast, so that the input matches the one the weights were trained
    on. (In smooth regions this rounding is of the order of the fingerprint itself; a model trained on the exact
    float32 fingerprint would need this function without the rounding.) exact=True is that function: float32
    throughout, for weights trained on the exact fingerprint (meta "fingerprint": "fp32")."""
    with torch.autocast(device_type=rgb8.device.type, enabled=False):
        y = luminance(srgb_to_linear(rgb8.float() / 255.0))
        if exact:
            k = KERNEL.to(y.device, torch.float32).view(1, 1, 3, 3)
            return y - F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), k)
        k = KERNEL.to(y.device, torch.float32).to(torch.bfloat16).float().view(1, 1, 3, 3)
        yb = F.pad(y, (1, 1, 1, 1), mode="replicate").to(torch.bfloat16).float()
        return y - F.conv2d(yb, k).to(torch.bfloat16).float()


def _scale(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """99.5th percentile of |d| over the valid pixels of a fixed 2 x 2 sub-grid (every second row and column), floored."""
    a, v = d.abs()[..., ::2, ::2], valid[..., ::2, ::2] > 0
    out = [torch.quantile(a[i][v[i]].float(), SCALE_Q) if v[i].any() else a.new_tensor(0.0) for i in range(a.shape[0])]
    return torch.stack(out).clamp(min=SCALE_FLOOR).view(-1, 1, 1, 1).to(d.dtype)


def nearest_fill(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8, (H, W) {0, 1} -> the image with each masked pixel set to the colour of the nearest valid pixel
    (exact Euclidean distance; the training code's function, so ties resolve the same way). The nearest valid pixel of
    a masked one lies within the mask's bounding box grown by one pixel, so only that box is transformed."""
    from scipy.ndimage import distance_transform_edt
    bad = valid == 0
    if not bad.any() or bad.all():
        return rgb
    ys, xs = np.nonzero(bad)
    y0, y1 = max(0, ys.min() - 2), min(bad.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(bad.shape[1], xs.max() + 3)
    if bad[y0:y1, x0:x1].all():                 # the box is clipped by the image and holds no valid pixel
        y0, y1, x0, x1 = 0, bad.shape[0], 0, bad.shape[1]
    _, (iy, ix) = distance_transform_edt(bad[y0:y1, x0:x1], return_indices=True)
    out = rgb.copy()
    out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1][iy, ix]
    return out


def _leak_free(v: torch.Tensor) -> torch.Tensor:
    """Valid pixels whose 3 x 3 neighbourhood holds no masked pixel (outside the image counts as valid)."""
    near = F.max_pool2d(F.pad((v <= 0).float(), (1, 1, 1, 1)), 3, 1)
    return ((v > 0) & (near == 0)).to(v.dtype)


def _imagenet(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return torch.cat(((x - mean) / std * v, v), 1)


def model_input(rgb8: torch.Tensor, valid: torch.Tensor | None, kind: str, exact: bool = False,
                leak_free: bool = False) -> torch.Tensor:
    """(B, 3, H, W) uint8 + optional (B, 1, H, W) {0, 1} mask -> the model input of type `kind`."""
    v = torch.ones_like(rgb8[:, :1], dtype=torch.float32) if valid is None else valid.float()
    if leak_free and valid is not None:
        filled = [nearest_fill(x.permute(1, 2, 0).cpu().numpy(), m[0].cpu().numpy()) for x, m in zip(rgb8, v)]
        rgb8 = torch.from_numpy(np.stack(filled)).permute(0, 3, 1, 2).to(rgb8.device)
    if kind == "rgb":
        return _imagenet(rgb8.float() / 255.0, v)
    if kind == "gray":
        y = luminance(srgb_to_linear(rgb8.float() / 255.0)).clamp(0, 1)
        y = torch.where(y <= 0.0031308, y * 12.92, 1.055 * y.clamp(min=1e-12) ** (1 / 2.4) - 0.055)
        return _imagenet(y.expand(-1, 3, -1, -1), v)
    if kind == "p99":
        d = fingerprint(rgb8, exact)
        dn = d / _scale(d, _leak_free(v) if leak_free else v)
        return torch.cat((torch.sign(dn) * torch.sqrt(dn.abs()) * v, v), 1)
    raise ValueError(kind)


class BlurDetector(nn.Module):
    def __init__(self, kind: str, arch: str = "convnext_small", fingerprint: str = "bf16-rounded", mask: str = "leaky"):
        super().__init__()
        import torchvision.models as tvm
        if arch not in WIDTH or fingerprint not in FINGERPRINTS or mask not in MASKS:
            raise ValueError(f"unknown trunk {arch!r}, fingerprint precision {fingerprint!r} or mask handling {mask!r}")
        self.kind, self.exact, self.leak_free = kind, fingerprint == "fp32", mask == "leak-free"
        m = getattr(tvm, arch)(weights=None)
        stem = m.features[0][0]
        n_in = 2 if kind == "p99" else 4
        m.features[0][0] = nn.Conv2d(n_in, stem.out_channels, kernel_size=4, stride=4)
        self.trunk = nn.Sequential(*list(m.features.children())[:6])        # stride 16, WIDTH[arch] channels
        self.head = nn.Sequential(nn.Conv2d(WIDTH[arch], 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
        self._r = nn.Parameter(torch.tensor(0.0))

    @property
    def r(self) -> torch.Tensor:
        return F.softplus(self._r) + 0.1

    def forward(self, rgb8: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
        """rgb8 (B, 3, H, W) uint8 -> {"image": logit per image, "local": logit per 16 px cell}."""
        H, W = rgb8.shape[-2:]
        x = model_input(rgb8, valid, self.kind, self.exact, self.leak_free)
        x = F.pad(x, (0, math.ceil(W / CELL) * CELL - W, 0, math.ceil(H / CELL) * CELL - H))
        local = self.head(self.trunk(x)).float()
        cell_valid = F.avg_pool2d(x[:, -1:], CELL, CELL) >= 0.5
        r = self.r
        l = torch.where(cell_valid, local * r, torch.full_like(local, -1e4))
        if not bool(cell_valid.flatten(1).any(1).all()):
            raise ValueError("no valid 16 px cell in the image: too small or fully masked, cannot be judged")
        n = cell_valid.flatten(1).sum(1)
        image = (torch.logsumexp(l.flatten(1), 1) - torch.log(n.float())) / r
        return {"image": image, "local": local}


def load_detector(path, device="cpu") -> tuple[BlurDetector, dict]:
    """A published weight file -> (model in eval mode, its metadata incl. the decision thresholds)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = ck["meta"]
    model = BlurDetector(meta["input"], meta.get("arch", "convnext_small"), meta.get("fingerprint", "bf16-rounded"),
                         meta.get("mask", "leaky"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), meta
