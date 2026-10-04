"""A minimal U-Net from rasterized link channels to a rain field: the interface, not a result.

The model reads ``inputs(time, channel, lat, lon)`` of a ``core.maps.learning`` dataset
and predicts ``target(time, lat, lon)``. Everything here is deliberately small (two
resolution levels, a few thousand weights, a few epochs on a CPU) so that the data flow
is clear; choosing and tuning a real model is the project's work.

Transforms: rain-like channels and the target enter as ``log1p``; the loss is the mean
squared error in that space, over cells that are ``observable`` (near a link) and have a
target value.

    from learned_2d.unet import TinyUNet, fit, predict
    model = TinyUNet(n_in=4)
    history = fit(model, train["x"], train["y"], mask, val["x"], val["y"], epochs=30)
    rain = predict(model, test["x"])            # (n, H, W) in the target's units
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_CHANNELS = (0, 1, 2)          # rain, attenuation, coverage: log1p; length: / 10 km


def transform_inputs(x: np.ndarray) -> torch.Tensor:
    """``(n, C, H, W)`` channels -> network input (log1p of the rain-like ones, length / 10)."""
    x = np.array(x, dtype=np.float32, copy=True)
    for c in range(x.shape[1]):
        x[:, c] = np.log1p(np.clip(x[:, c], 0, None)) if c in LOG_CHANNELS else x[:, c] / 10.0
    return torch.from_numpy(x)


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    """Two-level U-Net with a linear head (log1p space); :func:`predict` clips at 0."""

    def __init__(self, n_in: int = 4, c=(16, 32, 64)):
        super().__init__()
        self.enc1, self.enc2, self.bottom = DoubleConv(n_in, c[0]), DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.up2, self.dec2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), DoubleConv(2 * c[1], c[1])
        self.up1, self.dec1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), DoubleConv(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        ny, nx = x.shape[-2:]
        x = F.pad(x, (0, (-nx) % 4, 0, (-ny) % 4))           # both poolings need a multiple of 4
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottom(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)[:, 0, :ny, :nx]


def _masked_mse(pred, y, w):
    return ((pred - y) ** 2 * w).sum() / w.sum().clamp(min=1.0)


def _targets(y: np.ndarray, mask: np.ndarray):
    ok = np.isfinite(y) & np.asarray(mask, bool)[None]
    return torch.from_numpy(np.log1p(np.clip(np.nan_to_num(y), 0, None)).astype(np.float32)), \
        torch.from_numpy(ok.astype(np.float32))


def fit(model: nn.Module, x_train, y_train, mask, x_val=None, y_val=None, epochs: int = 30,
        batch: int = 16, lr: float = 2e-3, seed: int = 0, augment: bool = True) -> list:
    """Adam on the masked log1p MSE; keeps the weights of the best validation epoch.

    ``augment``: random horizontal / vertical flips of each batch (the rain field has no
    preferred direction; the link layout does, so flips also widen the geometries seen).
    Returns one ``{"epoch", "train", "val"}`` row per epoch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt, (yt, wt) = transform_inputs(x_train), _targets(y_train, mask)
    has_val = x_val is not None and len(x_val)
    if has_val:
        xv, (yv, wv) = transform_inputs(x_val), _targets(y_val, mask)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], (np.inf, None)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(xt))
        losses = []
        for s in range(0, len(order), batch):
            i = torch.from_numpy(order[s:s + batch])
            xb, yb, wb = xt[i], yt[i], wt[i]
            if augment:
                dims = [d for d, flip in ((-1, rng.random() < 0.5), (-2, rng.random() < 0.5)) if flip]
                if dims:
                    xb, yb, wb = xb.flip(dims), yb.flip(dims), wb.flip(dims)
            loss = _masked_mse(model(xb), yb, wb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "train": float(np.mean(losses))}
        if has_val:
            model.eval()
            with torch.no_grad():
                row["val"] = float(_masked_mse(model(xv), yv, wv))
            if row["val"] < best[0]:
                best = (row["val"], {k: v.clone() for k, v in model.state_dict().items()})
        history.append(row)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return history


def predict(model: nn.Module, x: np.ndarray, batch: int = 64) -> np.ndarray:
    """Rain field ``(n, H, W)`` in the target's units."""
    model.eval()
    xt = transform_inputs(x)
    with torch.no_grad():
        out = torch.cat([model(xt[s:s + batch]) for s in range(0, len(xt), batch)])
    return np.expm1(np.clip(out.numpy(), 0, None))
