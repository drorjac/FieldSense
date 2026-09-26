"""
Forecasting rain maps 15-60 minutes ahead from a history of maps.

Part 5 of ``advanced_models_colab_v2.ipynb``:

``windows`` / ``multi_windows``   (lookback frames) -> (target frame[s])
``RainTransformer``               attention over the lookback frames
``GRUForecaster``                 a GRU over the frames. The notebook called
                                  it ``SimpleMamba`` and the paper a
                                  "Mamba-style SSM"; it is a GRU, not a
                                  state-space model (alias kept)
``train``                         Adam + MSE, gradient clipping, best-val state
``train_grid``                    single-step models per horizon + one
                                  multi-horizon model, per input map type
``PODSindy``                      POD (truncated SVD) + SINDy on the modes
``persistence``                   the last observed frame, h steps on
``pysteps_forecasts``             Lucas-Kanade extrapolation of the radar

Alignment, the convention every function here follows: sample ``i`` sees
frames ``[i, i + lookback)`` of its split and is compared with frame
``i + lookback - 1 + h``. The notebook's pySTEPS extrapolated instead from
frame ``i + lookback`` - one frame later, which for h = 1 *is* the frame it
is scored against. ``pysteps_forecasts(..., faithful=True)`` reproduces
that; the default aligns it like everything else.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def windows(maps: np.ndarray, lookback: int, h: int):
    """X (N, lookback, H, W) and Y (N, H, W) for one horizon."""
    clean = np.nan_to_num(maps, nan=0.0).astype(np.float32)
    n = len(clean) - lookback - h + 1
    x = np.stack([clean[i:i + lookback] for i in range(n)])
    y = np.stack([clean[i + lookback + h - 1] for i in range(n)])
    return torch.from_numpy(x), torch.from_numpy(y)


def multi_windows(maps: np.ndarray, lookback: int, horizons):
    """X (N, lookback, H, W) and Y (N, len(horizons), H, W).

    N is set by the longest horizon, so every horizon's samples are the
    first N of its single-step ``windows`` - the same targets.
    """
    clean = np.nan_to_num(maps, nan=0.0).astype(np.float32)
    n = len(clean) - lookback - max(horizons) + 1
    x = np.stack([clean[i:i + lookback] for i in range(n)])
    y = np.stack([[clean[i + lookback + h - 1] for h in horizons] for i in range(n)])
    return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class RainTransformer(nn.Module):
    """Each frame flattened to a token; the last token decodes the forecast."""

    def __init__(self, spatial_shape, d_model=128, n_heads=4, n_layers=2, n_horizons=1):
        super().__init__()
        self.shape, self.n_horizons = tuple(spatial_shape), n_horizons
        flat = self.shape[0] * self.shape[1]
        self.embedding = nn.Linear(flat, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, n_horizons * flat)

    def forward(self, x):
        b, t, h, w = x.shape
        z = self.transformer(self.embedding(x.view(b, t, -1)))
        out = self.decoder(z[:, -1])
        return out.view(b, h, w) if self.n_horizons == 1 else out.view(b, self.n_horizons, h, w)


class GRUForecaster(nn.Module):
    """A GRU over flattened frames; the last hidden state decodes the forecast."""

    def __init__(self, spatial_shape, hidden_dim=256, n_layers=1, n_horizons=1):
        super().__init__()
        self.shape, self.n_horizons = tuple(spatial_shape), n_horizons
        flat = self.shape[0] * self.shape[1]
        self.gru = nn.GRU(flat, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_horizons * flat)

    def forward(self, x):
        b, t, h, w = x.shape
        _, hidden = self.gru(x.view(b, t, -1))
        out = self.fc(hidden[-1])
        return out.view(b, h, w) if self.n_horizons == 1 else out.view(b, self.n_horizons, h, w)


SimpleMamba = GRUForecaster          # the notebook's name for the same model


def train(model, x_tr, y_tr, x_val, y_val, lr: float, batch_size: int, epochs: int,
          seed: int = 42, device: str = "cpu") -> dict:
    """Adam on MSE with gradient clipping at 1; model left at its best-val state."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr),
                                         batch_size=batch_size, shuffle=True)
    history = {"train": [], "val": [], "best_epoch": None}
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(x_val.to(device)), y_val.to(device)).item()
        history["train"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val"].append(v if np.isfinite(v) else float("inf"))
        if v < best:
            best, history["best_epoch"] = v, epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_grid(factory, splits: dict, lookback: int, horizons, hp: dict, epochs: int,
               seed: int = 42) -> dict:
    """Single-step models per horizon and one multi-horizon model, on test.

    ``splits`` is ``{"train": maps, "val": maps, "test": maps}`` of one input
    map type; ``factory(n_horizons)`` builds a model. Returns
    ``{"single": {h: (N_h, H, W)}, "multi": {h: (N, H, W)}, "histories": ...}``.
    """
    out = {"single": {}, "multi": {}, "histories": {}}
    for h in horizons:
        x_tr, y_tr = windows(splits["train"], lookback, h)
        x_va, y_va = windows(splits["val"], lookback, h)
        x_te, _ = windows(splits["test"], lookback, h)
        model = factory(1)
        out["histories"][f"single_{h}"] = train(model, x_tr, y_tr, x_va, y_va,
                                                hp["lr"], hp["batch_size"], epochs, seed)
        with torch.no_grad():
            out["single"][h] = model.eval()(x_te).numpy()
    x_tr, y_tr = multi_windows(splits["train"], lookback, horizons)
    x_va, y_va = multi_windows(splits["val"], lookback, horizons)
    x_te, _ = multi_windows(splits["test"], lookback, horizons)
    model = factory(len(horizons))
    out["histories"]["multi"] = train(model, x_tr, y_tr, x_va, y_va,
                                      hp["lr"], hp["batch_size"], epochs, seed)
    with torch.no_grad():
        pred = model.eval()(x_te).numpy()
    out["multi"] = {h: pred[:, k] for k, h in enumerate(horizons)}
    return out


# --------------------------------------------------------------------------
# POD + SINDy
# --------------------------------------------------------------------------
class PODSindy:
    """Truncated SVD of the training maps, SINDy on the mode amplitudes.

    STLSQ (threshold 0.05, alpha 0.05), degree-2 polynomials, dt = 0.25 h:
    the notebook's settings. A rollout that fails falls back to persistence
    for that sample, as there.
    """

    def __init__(self, n_modes: int = 8, dt: float = 0.25, seed: int = 42):
        self.n_modes, self.dt, self.seed = n_modes, dt, seed

    def fit(self, train_maps: np.ndarray) -> "PODSindy":
        import pysindy as ps
        from sklearn.decomposition import TruncatedSVD

        flat = np.nan_to_num(train_maps).reshape(len(train_maps), -1)
        self.svd = TruncatedSVD(n_components=self.n_modes, random_state=self.seed)
        z = self.svd.fit_transform(flat)
        self.shape = train_maps.shape[1:]
        self.model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.05, alpha=0.05),
                              feature_library=ps.PolynomialLibrary(degree=2))
        self.model.fit(z, t=self.dt)
        return self

    def predict(self, test_maps: np.ndarray, lookback: int, horizons) -> dict:
        """{h: (N, H, W)} rolled out from each window's last frame."""
        x, _ = multi_windows(test_maps, lookback, horizons)
        z0 = self.svd.transform(x[:, -1].numpy().reshape(len(x), -1))
        t = np.arange(max(horizons) + 1) * self.dt
        traj = np.repeat(z0[:, None, :], max(horizons) + 1, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(z0)):
                try:
                    traj[i] = self.model.simulate(z0[i], t=t)
                except Exception:                              # noqa: BLE001
                    pass                                       # persistence
        return {h: self.svd.inverse_transform(traj[:, h]).reshape(len(z0), *self.shape)
                .astype(np.float32) for h in horizons}


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def persistence(test_maps: np.ndarray, lookback: int, horizons,
                one_step_late: bool = False) -> dict:
    """{h: (N, H, W)}: each window's last observed frame, whatever h is.

    ``one_step_late`` persists frame ``i + lookback`` instead - what the
    notebook's pySTEPS extrapolated from. It is a diagnostic, not a model:
    it shows how much a one-frame leak alone improves a score.
    """
    n = len(test_maps) - lookback - max(horizons) + 1
    shift = 1 if one_step_late else 0
    last = np.nan_to_num(test_maps[lookback - 1 + shift:lookback - 1 + shift + n]).astype(np.float32)
    return {h: last for h in horizons}


def pysteps_forecasts(radar_test: np.ndarray, lookback: int, horizons,
                      faithful: bool = False, extrapolate=None) -> dict:
    """{h: (N, H, W)} Lucas-Kanade extrapolation from three radar frames.

    Aligned like every other model: the last frame used is the window's last,
    ``i + lookback - 1``. ``faithful=True`` reproduces the notebook, which
    used frame ``i + lookback`` - one step later, the frame the h = 1
    forecast is scored against. ``extrapolate(frames, n_steps)`` defaults to
    pySTEPS and can be replaced (tests, or where pySTEPS will not build).
    """
    if extrapolate is None:
        # fail loudly here, not per sample below: without OpenCV, pySTEPS'
        # Lucas-Kanade raises, and a fallback would score persistence as pySTEPS
        import cv2  # noqa: F401
        import pysteps  # noqa: F401
        extrapolate = _pysteps_extrapolate
    n = len(radar_test) - lookback - max(horizons) + 1
    shift = 1 if faithful else 0
    out = {h: np.zeros((n, *radar_test.shape[1:]), dtype=np.float32) for h in horizons}
    failed = 0
    for i in range(n):
        last = i + lookback - 1 + shift
        frames = radar_test[max(0, last - 2):last + 1]
        try:
            fc = extrapolate(frames, max(horizons))
        except ImportError:
            raise
        except Exception:                                      # noqa: BLE001
            failed += 1
            fc = np.repeat(frames[-1:], max(horizons), axis=0)
        for h in horizons:
            out[h][i] = fc[h - 1]
    if failed:
        warnings.warn(f"pySTEPS failed on {failed} of {n} samples; those are persistence",
                      stacklevel=2)
    return out


def _pysteps_extrapolate(frames: np.ndarray, n_steps: int) -> np.ndarray:
    from pysteps import motion, nowcasts
    from pysteps.utils import conversion

    safe = np.where(np.isfinite(frames), frames, 0.0).astype(np.float64)
    dbr, _ = conversion.to_reflectivity(safe, metadata={
        "accutime": 15, "unit": "mm/h", "transform": None, "threshold": 0.1, "zerovalue": 0.0})
    from pysteps.exceptions import MissingOptionalDependency

    try:
        velocity = motion.get_method("lucaskanade")(dbr)
    except MissingOptionalDependency:
        raise
    except Exception:                                          # noqa: BLE001
        # no trackable features (dry frames): no motion is the right answer
        velocity = np.zeros((2, *safe.shape[1:]))
    fc = nowcasts.get_method("extrapolation")(safe[-1], velocity, n_steps)
    return np.nan_to_num(fc, nan=0.0).astype(np.float32)
