"""A motion field learned from simulations where the true motion is known.

Optical flow (LK, VET, DARTS) infers motion from the brightness-constancy
assumption alone; nothing tells it whether it got the motion right, because no
real dataset contains the true wind that carries the rain. The simulator in
``core.simulation`` does: every sequence comes with the exact flow that moved
it. This module trains a network on that supervision - past fields in, the
true motion field out - and gives it the same interface as
:func:`core.nowcast.methods.motion`, so it drops into extrapolation and every
pysteps nowcast.

**Training data.** Sequences from ``core.simulation.spacetime.simulate`` with
every choice drawn at random: the generator (cells, clustered storms, squall
lines, meta-Gaussian, multifractal, cascade, front), the flow (uniform,
rotation, shear, deformation, random divergence-free; 5-60 km/h), the evolution
(frozen, AR(1), scale-dependent cascade, cell life cycles; tau 30-200 min),
then degraded like a sensor would degrade them (random smoothing,
multiplicative noise, a random detection threshold) so the network does not
learn the simulator's exact texture.

**Network.** A U-Net (two poolings, 24-48-96 features) on ``N_PAST`` = 4 fields
as ``log(R + 0.1)``, predicting ``(u, v)`` in pixels per step at every pixel.
Loss: L1 of the vector error, weighted ``1 + 4 x wet`` - motion is observable
only where there is rain, but the network must still fill the dry areas
smoothly. Batches are rotated by multiples of 90 deg and flipped, with the
target vectors rotated alike, so no direction is preferred.

    from core.nowcast import learned_motion as lm
    model = lm.train(n_sequences=600)              # minutes on a CPU
    lm.save(model, "learned_motion.pt")
    v = lm.motion(rate, model)                     # (2, y, x) pixels per step
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

N_PAST = 4
SIZE = 64
EPS = 0.1


# --------------------------------------------------------------------------
# training data from the simulator
# --------------------------------------------------------------------------
def _random_sequence(rng: np.random.Generator, size: int, dt_min: float, n_frames: int):
    from core.simulation import flows as fl
    from core.simulation import generators as gen
    from core.simulation import spacetime as st
    from core.simulation.rain_fields import Grid

    dx = float(rng.choice([0.5, 1.0, 1.0, 2.0]))
    grid = Grid(n=size, dx_km=dx)
    key = str(rng.choice(["convective_cells", "clustered_storms", "squall_line", "stratiform_matern",
                          "banded_anisotropic", "scale_free", "multifractal", "cascade", "frontal",
                          "rainfarm"]))
    seed = int(rng.integers(2 ** 31))
    model = gen.make(key, seed=seed)
    if hasattr(model, "radius_km"):            # keep cells a few pixels wide at every resolution
        model.radius_km = model.radius_km * dx / 0.5 if dx > 1 else model.radius_km
    speed = rng.uniform(5, 60)
    ang = rng.uniform(0, 2 * np.pi)
    mean = (speed * np.cos(ang), speed * np.sin(ang))
    kind = str(rng.choice(["uniform", "rotation", "shear", "deformation", "random"]))
    p = {"mean": mean}
    if kind == "rotation":
        p["omega_deg_h"] = rng.uniform(-80, 80)
    elif kind == "shear":
        p["shear_per_h"] = rng.uniform(-1.5, 1.5) * (1.0 / dx if dx < 1 else 1.0)
    elif kind == "deformation":
        p["rate_per_h"] = rng.uniform(-0.8, 0.8)
    elif kind == "random":
        p.update(rms_kmh=rng.uniform(3, 15), length_km=rng.uniform(10, 40) * dx, seed=seed)
    flow = fl.make_flow(kind, **p)
    evo = str(rng.choice(["frozen", "ar1", "cascade", "lifecycle"]))
    if evo == "lifecycle" and not hasattr(model, "sample_cells"):
        evo = "ar1"
    seq = st.simulate(model, grid, n_frames, dt_min, flow=flow, evolution=evo,
                      tau_min=rng.uniform(30, 200), lifetime_min=rng.uniform(30, 120), seed=seed)
    return seq.frames, seq.velocity_px()


def _degrade(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a clean simulated sequence look measured: blur, noise, a detection threshold."""
    from scipy.ndimage import gaussian_filter
    f = frames.astype(float)
    if rng.random() < 0.6:
        f = gaussian_filter(f, (0, rng.uniform(0.3, 1.5), rng.uniform(0.3, 1.5)))
    if rng.random() < 0.7:
        f = f * np.exp(rng.normal(0, rng.uniform(0.05, 0.4), f.shape))
    f = f * np.exp(rng.normal(0, 0.3))                     # calibration
    f[f < rng.uniform(0.05, 0.5)] = 0.0
    return f


def transform(rate: np.ndarray) -> np.ndarray:
    return np.log(np.clip(rate, 0, None) + EPS) - np.log(EPS)


def simulate_dataset(n_sequences: int, seed: int = 0, size: int = SIZE, dt_min: float = 5.0,
                     windows_per_sequence: int = 4, verbose: bool = True):
    """``(X, Y)``: ``(N, N_PAST, size, size)`` transformed rates and ``(N, 2, size, size)`` motion."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    t0 = time.time()
    for i in range(n_sequences):
        frames, v = _random_sequence(rng, size, dt_min, N_PAST + windows_per_sequence - 1)
        frames = _degrade(frames, rng)
        for w in range(windows_per_sequence):
            x = frames[w:w + N_PAST]
            if (x[-1] > 0.1).mean() < 0.02:
                continue
            X.append(transform(x).astype(np.float32))
            Y.append(v.astype(np.float32))
        if verbose and (i + 1) % 100 == 0:
            print(f"  simulated {i + 1}/{n_sequences} sequences ({time.time() - t0:.0f} s)", flush=True)
    return np.stack(X), np.stack(Y)


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def _net(c_in: int = N_PAST, width: int = 24, prior: Optional[str] = None):
    import torch.nn as nn
    import torch

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.prior = prior
            n_in = c_in + (2 if prior else 0)
            self.e1, self.e2, self.e3 = block(n_in, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
            self.d2 = block(4 * w, 2 * w)
            self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d1 = block(2 * w, w)
            self.out = nn.Conv2d(w, 2, 1)

        def forward(self, x, p=None):
            # inputs as differences to the last frame as well: motion lives in the changes
            x = torch.cat([x[:, -1:], x[:, :-1] - x[:, -1:]], dim=1)
            if self.prior:
                x = torch.cat([x, p / 5.0], dim=1)
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            out = self.out(d1)
            return out + p if self.prior else out

    return UNet()


def _rot_vec(y, k: int, flip: bool):
    """Rotate a motion field ``(N, 2, y, x)`` with its array by ``k`` x 90 deg, then flip x."""
    import torch
    y = torch.rot90(y, k, dims=(2, 3))
    u, v = y[:, 0], y[:, 1]
    # one rot90 of a (row = y, col = x) array sends a displacement (u, v) to (v, -u)
    for _ in range(k % 4):
        u, v = v, -u
    if flip:
        u = -torch.flip(u, dims=(2,))
        v = torch.flip(v, dims=(2,))
    return torch.stack([u, v], 1)


def _augment(x, y, k: int, flip: bool, *fields):
    """Rotate inputs and motion (and any further motion fields) by ``k`` x 90 deg, then flip x."""
    import torch
    x = torch.rot90(x, k, dims=(2, 3))
    if flip:
        x = torch.flip(x, dims=(3,))
    return (x, _rot_vec(y, k, flip), *[_rot_vec(f, k, flip) for f in fields])


def train(n_sequences: int = 600, epochs: int = 25, batch: int = 32, lr: float = 2e-3, seed: int = 0,
          data=None, val_frac: float = 0.1, verbose: bool = True, threads: int = 4,
          prior: Optional[str] = None):
    """Simulate a training set (or use ``data=(X, Y)``) and fit the network; returns it.

    ``prior="LK"``: the network also gets the Lucas-Kanade field of its inputs and learns only
    the correction to it (``data`` may then carry the priors as a third array).
    """
    import torch
    torch.manual_seed(seed)
    # training needs no pysteps, so torch may have its own thread pool here (see run scripts)
    torch.set_num_threads(threads)
    data = data if data is not None else simulate_dataset(n_sequences, seed, verbose=verbose)
    X, Y = data[0], data[1]
    P = (data[2] if len(data) > 2 else prior_fields(X, prior, verbose)) if prior else np.zeros_like(Y)
    n_val = max(1, int(len(X) * val_frac))
    Xt, Yt, Pt = (torch.from_numpy(a[n_val:]) for a in (X, Y, P))
    Xv, Yv, Pv = (torch.from_numpy(a[:n_val]) for a in (X, Y, P))
    net = _net(prior=prior)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    gen = torch.Generator().manual_seed(seed)

    def loss_fn(pred, y, x):
        wet = (x[:, -1] > transform(np.array(0.1)).item() + 1e-6).float()
        w = 1 + 4 * wet
        return (torch.linalg.vector_norm(pred - y, dim=1) * w).sum() / w.sum()

    best, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = 0.0
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            k = int(torch.randint(0, 4, (1,), generator=gen))
            flip = bool(torch.randint(0, 2, (1,), generator=gen))
            x, y, p = _augment(Xt[idx], Yt[idx], k, flip, Pt[idx])
            opt.zero_grad()
            loss = loss_fn(net(x, p), y, x)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xv, Pv), Yv, Xv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if verbose:
            print(f"  epoch {ep + 1:2d}  train {tot / len(Xt):.3f}  val {vl:.3f} px/step", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    net.val_loss = best
    return net


def save(net, path):
    import torch
    torch.save({"state": net.state_dict(), "prior": net.prior}, path)


def load(path):
    import torch
    d = torch.load(path, weights_only=True)
    if "state" not in d:                         # a bare state dict: no prior
        d = {"state": d, "prior": None}
    net = _net(prior=d["prior"])
    net.load_state_dict(d["state"])
    net.eval()
    return net


_META = {"transform": None, "unit": "mm/h", "threshold": 0.1, "zerovalue": 0.0, "accutime": 5.0}


def prior_motion(rate: np.ndarray, method: str = "LK") -> np.ndarray:
    """The physics-based motion field the hybrid network corrects (``core.nowcast.methods``)."""
    from core.nowcast.methods import motion as classic
    return classic(np.asarray(rate, dtype=float), dict(_META), method)


def _prior_chunk(args):
    X, method = args
    return np.stack([prior_motion(np.exp(x + np.log(EPS)) - EPS, method) for x in X]).astype(np.float32)


def prior_fields(X: np.ndarray, method: str = "LK", verbose: bool = True, workers: int = 6) -> np.ndarray:
    """Prior motion for every training sample (inputs back from the log transform).

    Runs in ``workers`` processes: callers need an ``if __name__ == "__main__"`` guard.
    """
    t0 = time.time()
    chunks = [(c, method) for c in np.array_split(X, max(1, workers * 4))]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_prior_chunk, chunks))
    else:
        parts = [_prior_chunk(c) for c in chunks]
    if verbose:
        print(f"  {method} priors for {len(X)} samples ({time.time() - t0:.0f} s)", flush=True)
    return np.concatenate(parts)


def motion(rate: np.ndarray, net) -> np.ndarray:
    """``(2, y, x)`` pixels per step from the last ``N_PAST`` fields of ``rate`` (mm/h).

    Any grid size works: the field is padded to a multiple of 4 by reflection and cropped.
    """
    import torch
    r = np.asarray(rate[-N_PAST:], dtype=float)
    x = transform(r)
    ny, nx = x.shape[1:]
    py, px = (-ny) % 4, (-nx) % 4
    x = np.pad(x, ((0, 0), (0, py), (0, px)), mode="reflect")
    p = None
    if net.prior:
        p = np.pad(prior_motion(r, net.prior), ((0, 0), (0, py), (0, px)), mode="edge")
        p = torch.from_numpy(p[None].astype(np.float32))
    with torch.no_grad():
        v = net(torch.from_numpy(x[None].astype(np.float32)), p)[0].numpy()
    return v[:, :ny, :nx].astype(float)
