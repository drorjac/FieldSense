"""Learned nowcasts: a small U-Net per input set and target, and a hybrid that corrects pysteps.

**Inputs** (the last 30 min, 6 steps; every rate as ``log1p``, then standardised with the
mean and standard deviation of the training samples):

=========  ==========================================================================
sensor     channels per step
=========  ==========================================================================
``R``      radar rate; radar-has-data mask
``C``      link map (0 beyond 15 km of a link); indicator of cells holding a reporting link
``P``      PWS map (0 beyond 15 km); indicator of cells holding a reporting station
=========  ==========================================================================

plus, once, the static coverage of each sensor used (cells within 15 km of a link / PWS).

**Models.** ``in-<set>>tgt-<target>``: input set ``R, C, P, CP, RC, RP, RCP`` (the sensor
ablation; ``C``, ``P`` and ``CP`` are sensor-only models, no radar) and target ``R``
(the future radar) or ``RCP`` (the future radar adjusted with links and PWS). All 12 lead
times are predicted at once as ``log1p`` rates. ``hybrid``: inputs ``RCP`` plus the
pysteps extrapolation of the radar along the radar's motion (12 leads); it predicts a
correction added to that extrapolation (target ``R``).

**Network.** A U-Net (two poolings, 16-32-64 features, ~120k weights) - the architecture of
RainNet (Ayzel et al. 2020), much smaller. Loss: mean squared error of ``log1p`` rates,
weighted by ``1 + observed rate`` so rain counts more than the dry majority, on cells
where the target has data.

**Data split** (no leakage). Training samples come from the training weeks, early stopping
from the validation weeks, scores from the test weeks (``links.split``, the RNN's own split);
a sample's whole window (30 min back, 60 min ahead) lies in one split. Samples: every 5 min
(hybrid: every 15 min) when more than 5% of the radar's domain is wet at the issue time,
plus one dry sample in ten. Adam (lr 1e-3), batches of 32, up to 30 epochs, early stopping
after 5 epochs without improvement; seeds fixed.

**Calibration** (validation weeks only). Trained on a squared error weighted by the
observed rate, the networks blur (drizzle over most of the domain) and their optimum lies
above the mean. Per lead time, two numbers are fitted on the validation samples against the
model's own target: a threshold below which the forecast is set to zero, so the wet area
(> 0.1 mm/h) matches; and a scale, so the volume matches - the frequency and volume matching
pysteps applies to its nowcasts (and ``projects/cml_rnn`` to its RNN). Scored forecasts are
calibrated; the raw ones are kept as ``learned_raw|<model>``.

**Two variants per model**, the one with the lower *validation* loss is scored: ``plain``
as above, and ``aug`` with each training batch flipped at random north-south and east-west
(the inputs, their masks and the target together) and weight decay 1e-4 (AdamW). With
2,300 training samples in Gothenburg the plain models stop improving after a few epochs.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd

from . import scoring
from .cube import Cube, cell_index
from .settings import (CACHE_DIR, INPUT_SETS, MIN_WET_FRACTION, N_HISTORY, N_LEADS, SEED, STEP_MIN,
                       TARGETS)

log = logging.getLogger(__name__)

SENSOR_CHANNELS = {"R": ("R", "Rv"), "C": ("C", "Ci"), "P": ("P", "Pi")}
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 32
LR = 1e-3
VARIANTS = ("plain", "aug")       # aug: random flips of each batch + weight decay 1e-4
WEIGHT_DECAY = 1e-4
# New York's grid is 120 x 120 (Gothenburg 44 x 35): there each model is trained once, in the
# variant that won on Gothenburg's validation weeks (aug), on every second 5-min window, and on
# random 64 x 64 crops (half of them holding the sensor area); validation and forecasts use
# the whole domain.
NETWORK_TRAINING = {
    "openmrg": {"variants": VARIANTS, "stride": 1, "crop": None},
    "openmesh": {"variants": ("aug",), "stride": 2, "crop": 64},
}
DRY_SHARE = 0.1
HYBRID_STRIDE = 3


def model_dir(network: str, name: str, variant: str = "plain"):
    base = CACHE_DIR / network / "learn" / name.replace(">", "_")
    return base if variant == "plain" else base.with_name(base.name + "__" + variant)


def chosen_variant(network: str, name: str) -> str:
    """The variant with the lower validation loss (test weeks never consulted)."""
    best, pick = np.inf, None
    for v in VARIANTS:
        h = model_dir(network, name, v) / "history.json"
        if h.exists() and (model_dir(network, name, v) / "model.pt").exists():
            val = json.loads(h.read_text())["best_val"]
            if val < best:
                best, pick = val, v
    return pick


def model_names() -> list:
    names = [f"in-{s}>tgt-{t}" for t in TARGETS for s in INPUT_SETS]
    return names + ["hybrid"]


def spec(name: str) -> dict:
    if name == "hybrid":
        return {"inputs": INPUT_SETS["RCP"], "target": "R", "hybrid": True}
    s, t = name.removeprefix("in-").split(">tgt-")
    return {"inputs": INPUT_SETS[s], "target": t, "hybrid": False}


# ------------------------------------------------------------------ samples


def _window_ok(cube: Cube) -> np.ndarray:
    """``ok[i]``: steps i-5 ... i+12 contiguous and all in one split."""
    n = cube.times.size
    cont = cube.contiguous
    sp = cube.split
    run = np.zeros(n, int)
    same = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if cont[i] else 0
        same[i] = same[i - 1] + 1 if (cont[i] and sp[i] == sp[i - 1]) else 0
    span = N_HISTORY - 1 + N_LEADS
    ok = np.zeros(n, bool)
    ok[:n - N_LEADS] = same[N_LEADS:] >= span
    return ok


def sample_indices(cube: Cube, split: int, stride: int = 1, seed: int = SEED) -> np.ndarray:
    ok = _window_ok(cube) & (cube.split == split)
    cand = np.flatnonzero(ok)[::stride]
    R = cube["R"]
    wet = np.array([np.nanmean(np.asarray(R[i]) > 0.1) for i in cand]) > MIN_WET_FRACTION
    rng = np.random.default_rng(seed + split)
    dry = (~wet) & (rng.random(cand.size) < DRY_SHARE)
    return cand[wet | dry]


def static_channels(cube: Cube, inputs) -> np.ndarray:
    out = []
    for s in inputs:
        if s in ("C", "P"):
            a = np.asarray(cube[s][: min(2016, cube.times.size)])
            out.append(np.isfinite(a).mean(0) > 0.5)
    return np.array(out, dtype="float32").reshape(len(out), *cube.grid.shape)


def inputs_at(cube: Cube, i: int, inputs) -> np.ndarray:
    """``(channels, y, x)`` for issue index ``i`` (log1p rates and masks, history oldest first)."""
    chans = []
    sl = slice(i - N_HISTORY + 1, i + 1)
    for s in inputs:
        val, mask = SENSOR_CHANNELS[s]
        v = np.asarray(cube[val][sl], dtype="float32")
        m = np.asarray(cube[mask][sl], dtype="float32")
        chans.append(np.log1p(np.clip(np.nan_to_num(v, nan=0.0), 0, None)))
        chans.append(np.nan_to_num(m, nan=0.0))
    return np.concatenate(chans, axis=0)


def target_at(cube: Cube, i: int, target: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(cube[target][i + 1:i + 1 + N_LEADS], dtype="float32")
    valid = np.isfinite(y).astype("float32")
    return np.log1p(np.clip(np.nan_to_num(y, nan=0.0), 0, None)), valid


def extrapolation_at(cube: Cube, i: int, meta: dict) -> np.ndarray:
    """``log1p`` of the pysteps extrapolation of the radar along its own motion, 12 leads."""
    from core.nowcast import methods as M
    hist = np.nan_to_num(np.asarray(cube["R"][i - N_HISTORY + 1:i + 1], dtype=float), nan=0.0)
    try:
        v = M.motion(hist, meta, "LK")
        f = M.deterministic("extrapolation", hist, meta, v, N_LEADS)
    except Exception:
        f = np.repeat(hist[-1:], N_LEADS, axis=0)
    return np.log1p(np.clip(np.nan_to_num(f, nan=0.0), 0, None)).astype("float32")


def _extrap_worker(args):
    network, idx = args
    import warnings
    warnings.filterwarnings("ignore")
    from core.nowcast.grid import metadata
    c = Cube(network)
    meta = metadata(c.grid, STEP_MIN)
    return idx, np.stack([extrapolation_at(c, int(i), meta) for i in idx])


def extrapolations(network: str, idx: np.ndarray, tag: str, workers: int = 6) -> np.ndarray:
    """Cached pysteps extrapolations at ``idx`` (for the hybrid model)."""
    import multiprocessing as mp
    path = CACHE_DIR / network / "learn" / f"extrap_{tag}.npz"
    if path.exists():
        d = np.load(path)
        if np.array_equal(d["idx"], idx):
            return d["X"]
    chunks = [(network, c) for c in np.array_split(idx, max(1, workers * 4)) if c.size]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = dict((int(k[0]), x) for k, x in pool.map(_extrap_worker, chunks))
    X = np.concatenate([parts[int(c[1][0])] for c in chunks])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, idx=idx, X=X)
    return X


# ------------------------------------------------------------------ model


def build_model(c_in: int, c_out: int = N_LEADS, width: int = 16):
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(c_in, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
            self.d2 = block(4 * w, 2 * w)
            self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d1 = block(2 * w, w)
            self.out = nn.Conv2d(w, c_out, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)

    return UNet()


def _pad(shape, mult=4):
    ny, nx = shape
    return (0, (-nx) % mult, 0, (-ny) % mult)          # F.pad order: left, right, top, bottom


class Data:
    """Samples of one model: inputs, targets and masks, read from the memory-mapped cube."""

    def __init__(self, cube: Cube, name: str, idx: np.ndarray, extrap: np.ndarray | None = None,
                 norm: dict | None = None):
        self.cube, self.sp, self.idx = cube, spec(name), idx
        self.static = static_channels(cube, self.sp["inputs"])
        self.extrap = extrap
        if norm is None:
            norm = self._norm()
        self.norm = norm

    def raw(self, k: int):
        i = int(self.idx[k])
        x = inputs_at(self.cube, i, self.sp["inputs"])
        parts = [x, self.static]
        if self.sp["hybrid"]:
            parts.append(self.extrap[k])
        y, m = target_at(self.cube, i, self.sp["target"])
        return np.concatenate(parts, 0), y, m

    def _norm(self) -> dict:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(self.idx), size=min(400, len(self.idx)), replace=False)
        X = np.stack([self.raw(k)[0] for k in pick])
        mu = X.mean(axis=(0, 2, 3))
        sd = X.std(axis=(0, 2, 3))
        return {"mu": mu.astype("float32"), "sd": np.where(sd > 1e-6, sd, 1.0).astype("float32")}

    def batch(self, ks):
        xs, ys, ms = zip(*(self.raw(int(k)) for k in ks))
        X = (np.stack(xs) - self.norm["mu"][None, :, None, None]) / self.norm["sd"][None, :, None, None]
        return X.astype("float32"), np.stack(ys), np.stack(ms)

    def base(self, ks):
        """For the hybrid: the extrapolation (log1p) the network corrects; else zeros."""
        if not self.sp["hybrid"]:
            return None
        return np.stack([self.extrap[int(k)] for k in ks])


def _loss(pred, y, m):
    import torch
    w = m * (1.0 + torch.expm1(y).clamp(max=50.0))
    return (w * (pred - y) ** 2).sum() / w.sum().clamp(min=1.0)


def _forward(model, X, base, pad):
    import torch
    import torch.nn.functional as F
    ny, nx = X.shape[-2:]
    out = model(F.pad(torch.as_tensor(X), pad))[..., :ny, :nx]
    if base is not None:
        out = out + torch.as_tensor(base)
    return out


def _flip(arrs, fy: bool, fx: bool):
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
            continue
        if fy:
            a = a[..., ::-1, :]
        if fx:
            a = a[..., :, ::-1]
        out.append(np.ascontiguousarray(a))
    return out


def _crop(arrs, size: int, area_box, rng):
    """The same random ``size`` x ``size`` window of every array; half the time one holding the
    centre of the sensor area."""
    ny, nx = arrs[0].shape[-2:]
    if rng.random() < 0.5:
        ci, cj = (area_box[0] + area_box[1]) // 2, (area_box[2] + area_box[3]) // 2
        i0 = int(np.clip(ci - rng.integers(8, size - 8), 0, ny - size))
        j0 = int(np.clip(cj - rng.integers(8, size - 8), 0, nx - size))
    else:
        i0, j0 = int(rng.integers(0, ny - size + 1)), int(rng.integers(0, nx - size + 1))
    return [None if a is None else np.ascontiguousarray(a[..., i0:i0 + size, j0:j0 + size]) for a in arrs]


def train_one(network: str, name: str, refresh: bool = False, threads: int = 2, variant: str = "plain") -> dict:
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = model_dir(network, name, variant)
    if (out_dir / "model.pt").exists() and not refresh:
        return json.loads((out_dir / "history.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cube = Cube(network)
    sp = spec(name)
    nt = NETWORK_TRAINING[network]
    stride = HYBRID_STRIDE if sp["hybrid"] else nt["stride"]
    tr, va = sample_indices(cube, 0, stride), sample_indices(cube, 1, stride)
    ex_tr = extrapolations(network, tr, "train") if sp["hybrid"] else None
    ex_va = extrapolations(network, va, "val") if sp["hybrid"] else None
    d_tr = Data(cube, name, tr, ex_tr)
    d_va = Data(cube, name, va, ex_va, norm=d_tr.norm)
    c_in = d_tr.raw(0)[0].shape[0]
    model = build_model(c_in)
    aug = variant == "aug"
    ii, jj = np.nonzero(cube.area)
    area_box = (ii.min(), ii.max(), jj.min(), jj.max())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) if aug else \
        torch.optim.Adam(model.parameters(), lr=LR)
    pad = _pad(cube.grid.shape)
    rng = np.random.default_rng(SEED)
    hist = {"name": name, "variant": variant, "network": network, "n_train": int(tr.size), "n_val": int(va.size), "c_in": int(c_in),
            "n_weights": int(sum(p.numel() for p in model.parameters())), "train": [], "val": []}
    best, bad, t0 = np.inf, 0, time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(tr.size)
        tl = []
        for b in range(0, order.size, BATCH):
            ks = order[b:b + BATCH]
            X, Y, Mk = d_tr.batch(ks)
            base = d_tr.base(ks)
            if nt["crop"]:
                X, Y, Mk, base = _crop([X, Y, Mk, base], nt["crop"], area_box, rng)
            if aug:
                X, Y, Mk, base = _flip([X, Y, Mk, base], rng.random() < 0.5, rng.random() < 0.5)
            pred = _forward(model, X, base, pad)
            loss = _loss(pred, torch.as_tensor(Y), torch.as_tensor(Mk))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss))
        model.eval()
        vl = []
        with torch.no_grad():
            for b in range(0, va.size, 64):
                ks = np.arange(b, min(va.size, b + 64))
                X, Y, Mk = d_va.batch(ks)
                vl.append(float(_loss(_forward(model, X, d_va.base(ks), pad), torch.as_tensor(Y),
                                      torch.as_tensor(Mk))) * ks.size)
        v = float(np.sum(vl) / va.size)
        hist["train"].append(float(np.mean(tl)))
        hist["val"].append(v)
        log.info("%s %s epoch %d: train %.4f val %.4f (%.0f s)", network, name, epoch, np.mean(tl), v, time.time() - t0)
        if v < best - 1e-5:
            best, bad = v, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    hist.update(best_val=best, epochs=len(hist["val"]), seconds=round(time.time() - t0))
    np.savez(out_dir / "norm.npz", **d_tr.norm)
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    return hist


def _train_worker(args):
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    network, name, refresh, threads, variant = args
    return train_one(network, name, refresh, threads, variant)


def train_all(network: str, only=None, refresh: bool = False, procs: int = 4, variants=None) -> None:
    import multiprocessing as mp
    names = only or model_names()
    variants = variants or NETWORK_TRAINING[network]["variants"]
    cube = Cube(network)
    if "hybrid" in names:                      # extrapolations first (they use a pool of their own)
        for split, tag in ((0, "train"), (1, "val")):
            extrapolations(network, sample_indices(cube, split, HYBRID_STRIDE), tag)
    threads = max(1, (os.cpu_count() or 8) // procs)
    with mp.get_context("spawn").Pool(procs) as pool:
        jobs = [(network, n, refresh, threads, v) for v in variants for n in names]
        for h in pool.imap_unordered(_train_worker, jobs):
            log.info("trained %s (%s): %d epochs, best val %.4f, %s s", h["name"], h.get("variant", "plain"),
                     h["epochs"], h["best_val"], h["seconds"])


# ------------------------------------------------------------------ forecasts at the issue times


def predict(network: str, name: str, idx: np.ndarray, extrap: np.ndarray | None = None,
            calibrated: bool = True) -> np.ndarray:
    """``(issue, lead, y, x)`` rain rates (mm/h) of a trained model at issue indices ``idx``
    (with the validation-week calibration of :func:`calibrate` unless ``calibrated=False``)."""
    F = _predict_raw(network, name, idx, extrap)
    if calibrated:
        cal = calibrate(network, name)
        t, sc = np.asarray(cal["threshold"])[None, :, None, None], np.asarray(cal["scale"])[None, :, None, None]
        F = np.where(F >= t, F * sc, 0.0).astype("float32")
    return F


def calibrate(network: str, name: str, refresh: bool = False) -> dict:
    """Per-lead wet threshold and volume scale fitted on the validation samples (see above)."""
    d = model_dir(network, name, chosen_variant(network, name))
    path = d / "calibration.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    cube = Cube(network)
    sp = spec(name)
    stride = HYBRID_STRIDE if sp["hybrid"] else NETWORK_TRAINING[network]["stride"]
    va = sample_indices(cube, 1, stride)
    ex = extrapolations(network, va, "val") if sp["hybrid"] else None
    F = _predict_raw(network, name, va, ex)
    Y = np.stack([np.asarray(cube[sp["target"]][i + 1:i + 1 + N_LEADS], dtype="float32") for i in va])
    thr, scale = [], []
    for li in range(N_LEADS):
        f, y = F[:, li], Y[:, li]
        ok = np.isfinite(y)
        f, y = f[ok], y[ok]
        wet_obs = float(np.mean(y > 0.1))
        t = float(np.quantile(f, 1.0 - wet_obs)) if 0 < wet_obs < 1 else 0.1
        t = max(t, 0.1)
        kept = np.where(f >= t, f, 0.0)
        scale.append(float(y.sum() / kept.sum()) if kept.sum() > 0 else 1.0)
        thr.append(t)
    cal = {"threshold": thr, "scale": scale, "n_val": int(va.size), "variant": chosen_variant(network, name)}
    path.write_text(json.dumps(cal, indent=1))
    return cal


def _predict_raw(network: str, name: str, idx: np.ndarray, extrap: np.ndarray | None = None) -> np.ndarray:
    """``(issue, lead, y, x)`` raw network output as rain rates (mm/h)."""
    import sys
    import torch
    if "cv2" in sys.modules or "pysteps" in sys.modules:
        torch.set_num_threads(1)        # multi-threaded torch after OpenCV crashes on macOS
    d = model_dir(network, name, chosen_variant(network, name))
    cube = Cube(network)
    norm = dict(np.load(d / "norm.npz"))
    data = Data(cube, name, idx, extrap, norm=norm)
    c_in = data.raw(0)[0].shape[0]
    model = build_model(c_in)
    model.load_state_dict(torch.load(d / "model.pt"))
    model.eval()
    pad = _pad(cube.grid.shape)
    out = []
    with torch.no_grad():
        for b in range(0, idx.size, 64):
            ks = np.arange(b, min(idx.size, b + 64))
            X, _, _ = data.batch(ks)
            out.append(np.expm1(_forward(model, X, data.base(ks), pad).numpy().clip(0, 6)))
    return np.concatenate(out).astype("float32")


def score_all(network: str, refresh: bool = False) -> None:
    """Score every trained model at the physics issue times; ``scores/learned.npz``."""
    from .physics import load_issues
    path = CACHE_DIR / network / "scores" / "learned.npz"
    if path.exists() and not refresh:
        return
    iss = load_issues(network)
    idx = iss["idx"].astype(int)
    cube = Cube(network)
    area = cube.area
    g = cube.gauges["g5"]
    gcells = cell_index(cube.grid, g.lat.values, g.lon.values)
    names = [n for n in model_names() if chosen_variant(network, n) is not None]
    ext = extrapolations(network, idx, "test") if "hybrid" in names else None
    ids = [f"learned|{n}" for n in names] + [f"learned_raw|{n}" for n in names]
    S = np.full((idx.size, len(ids), N_LEADS, len(scoring.REGIONS), len(scoring.STATS)), np.nan)
    G = np.full((idx.size, len(ids), N_LEADS, gcells.size), np.nan, dtype="float32")
    for j, name in enumerate(names):
        raw = _predict_raw(network, name, idx, ext if name == "hybrid" else None)
        cal = calibrate(network, name)
        t, sc = np.asarray(cal["threshold"])[:, None, None], np.asarray(cal["scale"])[:, None, None]
        for jj, calibrated in ((j, True), (j + len(names), False)):
            for k, i in enumerate(idx):
                f = np.where(raw[k] >= t, raw[k] * sc, 0.0) if calibrated else raw[k]
                obs = np.asarray(cube["R"][i + 1:i + 1 + N_LEADS], dtype=float)
                masks = {"domain": iss["reach"][k], "area": iss["reach"][k] & area[None]}
                S[k, jj] = scoring.stats(f, obs, masks)
                G[k, jj] = f.reshape(N_LEADS, -1)[:, gcells]
        log.info("%s scored %s", network, name)
    np.savez_compressed(path, ids=np.array(ids), stats=S, gauge=G)
    rows = []
    for name in names:
        for v in VARIANTS:
            hp = model_dir(network, name, v) / "history.json"
            if not hp.exists():
                continue
            h = json.loads(hp.read_text())
            rows.append({**{k: h[k] for k in ("name", "n_train", "n_val", "c_in", "n_weights", "epochs", "best_val",
                                              "seconds")}, "variant": v, "chosen": v == chosen_variant(network, name)})
    pd.DataFrame(rows).to_csv(path.with_name("training.csv"), index=False)
