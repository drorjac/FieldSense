"""Train PyNNcml's two-step RNN on the built datasets.

The network is Habi & Messer's (PyNNcml ``two_step_network``): a GRU (or LSTM) backbone
over the dynamic inputs, a fully connected branch over the link metadata, and two heads -
a rain-rate estimate and a wet probability. Loss, as in the paper: a rain-weighted MSE
on the estimate (``RegressionLoss``: dry hours weigh less, so the network does not learn
to predict zero) plus binary cross-entropy on wet/dry, balanced to be equal at the start.

What is new here is the data, not the network:

- inputs are hourly blocks of 1-minute *excess loss* over a causal baseline, not raw
  RSL/TSL (the networks here report at 1 min with different levels and quantization);
- the target is the average of the references available along each link (radar along
  the path, gauges near it), on three networks at once;
- the loss only counts training hours (``settings.split``); every 96-hour training window
  starts with 24 hours of context that are not scored.

    from cml_rnn.train import Config, train
    model, history = train(["openmrg", "openrainer", "openmesh"], Config())
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import xarray as xr

from core.events import detect_events
from core.opensense.networks import NETWORKS
from core.scientific_packages.pynncml_rnn import RegressionLoss

from .settings import DATA_DIR, PERIODS, split

log = logging.getLogger(__name__)


@dataclass
class Config:
    rnn_type: str = "GRU"
    n_layers: int = 2
    rnn_n_features: int = 128
    metadata_n_features: int = 32
    window: int = 96                 # hours per training sequence
    burn_in: int = 24                # of which not scored
    batch: int = 64
    steps_per_epoch: int = 200
    epochs: int = 60
    patience: int = 10               # epochs without validation improvement
    lr: float = 1e-3
    weight_decay: float = 1e-5
    wet_mm: float = 0.1
    target: str = "target"           # or "radar" / a point set
    log_target: bool = False         # train on log1p(rain); estimate = expm1
    seed: int = 0
    min_link_valid: float = 0.2      # drop links whose target exists in fewer hours
    physics: bool = False            # add the power-law rates of the excess loss as inputs
    neighbours: bool = False         # add what the links within 15 km see
    wet_share: float = 0.0           # share of training windows centred on a wet hour
    wet_mm_window: float = 1.0       # "wet" for that sampling


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def test_windows(network: str, n: int = 10) -> list:
    """The ``n`` largest events of the network (as in multisensor_maps): always test."""
    net = NETWORKS[network]
    ev = pd.concat([detect_events(net.radar_hourly(*p)) for p in PERIODS[network]], ignore_index=True)
    ev = ev.sort_values("total_mm", ascending=False).head(n)
    return list(zip(ev.start, ev.end))


@dataclass
class NetData:
    network: str
    X: np.ndarray            # (link, time, feature)
    M: np.ndarray            # (link, meta)
    y: np.ndarray            # (link, time) target, mm/h
    part: np.ndarray         # (time,) 0 train, 1 val, 2 test
    ds: xr.Dataset

    @classmethod
    def load(cls, network: str, target: str = "target", min_link_valid: float = 0.2,
             physics: bool = False, neighbours: bool = False) -> "NetData":
        from core.cml.rnn import neighbour_features, physics_features
        ds = xr.open_dataset(DATA_DIR / f"{network}.nc").load()
        X, M = ds.X.values.astype("float32"), ds.M.values.astype("float32")
        # neighbours from every link of the network, before the target-coverage filter
        extra = []
        if physics:
            extra.append(physics_features(X, M))
        if neighbours:
            extra.append(neighbour_features(X, M, ds.mid_lat.values, ds.mid_lon.values))
        X = np.concatenate([X] + extra, axis=-1) if extra else X
        y = ds[target].values.astype("float32")
        keep = np.flatnonzero(np.isfinite(y).mean(axis=1) >= min_link_valid)
        ds, X, M = ds.isel(link=keep), X[keep], M[keep]
        part = split(pd.DatetimeIndex(ds.time.values), test_windows(network))
        return cls(network, X, M, ds[target].values.astype("float32"), part, ds)


def normalization(datas: list) -> dict:
    """Exact mean / std of inputs over training hours and of metadata over links."""
    xs = [d.X[:, d.part == 0].reshape(-1, d.X.shape[-1]) for d in datas]
    x = np.concatenate(xs)
    m = np.concatenate([d.M for d in datas])
    std = x.std(0)
    mstd = m.std(0)
    return {"mean_dynamic": x.mean(0), "std_dynamic": np.where(std > 0, std, 1.0),
            "mean_metadata": m.mean(0), "std_metadata": np.where(mstd > 0, mstd, 1.0)}


def _wet_index(d: NetData, cfg: Config, part: int) -> np.ndarray:
    key = ("_wet", part, cfg.wet_mm_window)
    if key not in d.__dict__:
        li, ti = np.nonzero((d.part == part)[None, :] & (np.nan_to_num(d.y) >= cfg.wet_mm_window))
        d.__dict__[key] = np.column_stack([li, ti])
    return d.__dict__[key]


def sample_batch(datas: list, cfg: Config, rng: np.random.Generator, part: int = 0):
    """Random windows from the networks (equal share each), with a loss mask on ``part`` hours.

    A share ``cfg.wet_share`` of the windows is placed around a wet hour of that part, so
    heavy rain - rare among all hours - is seen often enough to be learned.
    """
    X, M, Y, W = [], [], [], []
    for _ in range(cfg.batch):
        d = datas[rng.integers(len(datas))]
        wet = _wet_index(d, cfg, part) if cfg.wet_share > 0 else np.empty((0, 2), int)
        for _attempt in range(50):
            if len(wet) and rng.random() < cfg.wet_share:
                li, tw = wet[rng.integers(len(wet))]
                lo = max(0, tw - cfg.window + 1)
                hi = min(d.X.shape[1] - cfg.window, max(lo, tw - cfg.burn_in))
                t0 = rng.integers(lo, hi + 1) if hi >= lo else lo
            else:
                li = rng.integers(d.X.shape[0])
                t0 = rng.integers(0, d.X.shape[1] - cfg.window)
            sl = slice(t0, t0 + cfg.window)
            mask = (d.part[sl] == part) & np.isfinite(d.y[li, sl])
            mask[:cfg.burn_in] = False
            if mask.sum() >= 12:
                break
        X.append(d.X[li, sl])
        M.append(d.M[li])
        Y.append(np.nan_to_num(d.y[li, sl]))
        W.append(mask)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt)  # noqa: E731
    return t(X), t(M), t(Y), t(W, torch.bool)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(cfg: Config, norm: dict, n_features: int, n_meta: int):
    import pynncml as pnc
    from pynncml.neural_networks import InputNormalizationConfig

    torch.manual_seed(cfg.seed)
    return pnc.scm.rain_estimation.two_step_network(
        n_layers=cfg.n_layers, rnn_type=getattr(pnc.neural_networks.RNNType, cfg.rnn_type),
        normalization_cfg=InputNormalizationConfig(norm["mean_dynamic"], norm["std_dynamic"],
                                                   norm["mean_metadata"], norm["std_metadata"]),
        rnn_input_size=n_features, rnn_n_features=cfg.rnn_n_features,
        metadata_input_size=n_meta, metadata_n_features=cfg.metadata_n_features, pretrained=False)


def forward(model, X, M):
    out, _ = model(X, M, model.init_state(batch_size=X.shape[0]))
    return out[..., 0], out[..., 1]                       # rain (transformed), wet probability


def to_rain(r_hat: torch.Tensor, p: torch.Tensor, cfg: Config, gate: float = 0.5,
            scale: float = 1.0) -> torch.Tensor:
    """Two-step output: the rain estimate where the wet probability exceeds ``gate``."""
    r = torch.expm1(r_hat.clamp(max=6.0)) if cfg.log_target else r_hat
    return scale * r.clamp(min=0.0) * (p > gate)


def losses(model, batch, cfg: Config, est_loss, det_loss):
    X, M, Y, W = batch
    r_hat, p = forward(model, X, M)
    y = torch.log1p(Y) if cfg.log_target else Y
    le = est_loss(r_hat[W][None], y[W][None])
    ld = det_loss(p[W].clamp(1e-6, 1 - 1e-6), (Y[W] > cfg.wet_mm).float())
    return le, ld


def train(networks, cfg: Config, name: str) -> tuple:
    """Train on ``networks``; keep the best validation state; save to DATA_DIR/models/<name>."""
    import scipy.stats

    t_start = time.time()
    datas = [NetData.load(n, cfg.target, cfg.min_link_valid, cfg.physics, cfg.neighbours) for n in networks]
    norm = normalization(datas)
    model = build_model(cfg, norm, datas[0].X.shape[-1], datas[0].M.shape[-1])
    ytr = np.concatenate([d.y[:, d.part == 0][np.isfinite(d.y[:, d.part == 0])] for d in datas])
    ytr = np.log1p(ytr) if cfg.log_target else ytr
    gamma = float(scipy.stats.expon.fit(ytr)[1])
    est_loss, det_loss = RegressionLoss(gamma), torch.nn.BCELoss()
    rng = np.random.default_rng(cfg.seed)
    with torch.no_grad():
        le, ld = zip(*[losses(model, sample_batch(datas, cfg, rng), cfg, est_loss, det_loss) for _ in range(10)])
    lam = float(np.mean([float(x) for x in ld]) / max(np.mean([float(x) for x in le]), 1e-9))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    val_rng = np.random.default_rng(cfg.seed + 1)
    val_batches = [sample_batch(datas, cfg, val_rng, part=1) for _ in range(20)]

    history = {"loss": [], "val": [], "lambda": lam, "gamma": gamma}
    best, best_state, stale = math.inf, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        tot = 0.0
        for _ in range(cfg.steps_per_epoch):
            le, ld = losses(model, sample_batch(datas, cfg, rng), cfg, est_loss, det_loss)
            loss = lam * le + ld
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(lam * a + b) for a, b in
                               (losses(model, vb, cfg, est_loss, det_loss) for vb in val_batches)]))
        sched.step(v)
        history["loss"].append(tot / cfg.steps_per_epoch)
        history["val"].append(v)
        log.info("%s epoch %d  train %.4f  val %.4f  (%.0fs)", name, epoch, history["loss"][-1], v,
                 time.time() - t_start)
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    history["calibration"] = calibrate(model, cfg, datas)
    log.info("%s calibration on validation weeks: %s", name, history["calibration"])
    out = DATA_DIR / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    np.savez(out / "norm.npz", **norm)
    (out / "config.json").write_text(json.dumps({"networks": list(networks), "config": asdict(cfg),
                                                 "history": history, "minutes": (time.time() - t_start) / 60},
                                                indent=2, default=float))
    return model, history


def load(name: str):
    """A trained model, its config and networks."""
    out = DATA_DIR / "models" / name
    meta = json.loads((out / "config.json").read_text())
    cfg = Config(**meta["config"])
    norm = dict(np.load(out / "norm.npz"))
    model = build_model(cfg, norm, len(norm["mean_dynamic"]), len(norm["mean_metadata"]))
    model.load_state_dict(torch.load(out / "model.pt"))
    model.eval()
    return model, cfg, meta


def heads(model, d: NetData, chunk: int = 2000) -> tuple:
    """Both heads for every link and hour, running through the whole record."""
    r = np.zeros(d.y.shape, "float32")
    p = np.zeros(d.y.shape, "float32")
    with torch.no_grad():
        M = torch.as_tensor(d.M)
        state = model.init_state(batch_size=d.X.shape[0])
        for t0 in range(0, d.X.shape[1], chunk):
            o, state = model(torch.as_tensor(d.X[:, t0:t0 + chunk]), M, state)
            r[:, t0:t0 + chunk] = o[..., 0].numpy()
            p[:, t0:t0 + chunk] = o[..., 1].numpy()
    return r, p


def predict(model, cfg: Config, d: NetData, calibration: dict | None = None) -> np.ndarray:
    """Hourly rain (mm) for every link and hour, with the validation calibration."""
    gate, scale = calibration_for(calibration, d.network)
    r, p = heads(model, d)
    return to_rain(torch.as_tensor(r), torch.as_tensor(p), cfg, gate, scale).numpy()


def calibration_for(calibration: dict | None, network: str) -> tuple:
    if not calibration:
        return 0.5, 1.0
    scale = calibration["scale"]
    return calibration["gate"], float(scale.get(network, 1.0) if isinstance(scale, dict) else scale)


def calibrate(model, cfg: Config, datas: list, wet_mm: float = 1.0) -> dict:
    """Wet-probability threshold and per-network scale, fitted on the *validation* weeks.

    Each network's scale makes the estimate's total equal the target's over the *wet* hours
    (target >= ``wet_mm``) of the training and validation weeks. Wet hours, because totals
    and maps are made of them: fitted on all hours the scale is set by drizzle, where the
    network over-reads slightly, and it then pulls heavy rain down (Gothenburg: 0.62, storms
    17% low). Training and validation together, because Gothenburg's three validation weeks
    alone hold too few storms. The network is always known where the model is used. The
    threshold kept is the one with the lowest validation RMSE after scaling. Test weeks are
    never seen.
    """
    H = {d.network: heads(model, d) for d in datas}
    best = None
    for gate in np.round(np.arange(0.1, 0.91, 0.05), 2):
        scale, err, n = {}, 0.0, 0
        for d in datas:
            r, p = H[d.network]
            est = to_rain(torch.as_tensor(r), torch.as_tensor(p), cfg, float(gate)).numpy()
            fit = (d.part <= 1)[None, :] & np.isfinite(d.y) & (np.nan_to_num(d.y) >= wet_mm)
            scale[d.network] = float(d.y[fit].sum() / max(est[fit].sum(), 1e-9))
            sel = (d.part == 1)[None, :] & np.isfinite(d.y)
            err += float(((scale[d.network] * est[sel] - d.y[sel]) ** 2).sum())
            n += int(sel.sum())
        rmse = (err / max(n, 1)) ** 0.5
        if best is None or rmse < best["val_rmse"]:
            best = {"gate": float(gate), "scale": scale, "val_rmse": rmse}
    return best
