"""PyNNcml's two-step RNN as a retrieval method: hourly link rain from 1-minute signals.

Trained in ``projects/retrieval/rnn_three_networks`` (Habi & Messer's network from PyNNcml, trained against the
average of radar along the path and gauges near the link on three networks); this module
is what turns a saved model back into a method any project can run on a link set.

inputs, per link and hour ending T
    ``features``: the 60 one-minute values of *excess loss* in (T - 1 h, T] - total loss
    minus a causal baseline, the median of the previous 24 hours' 15-minute medians - and
    four hourly summaries: share of missing minutes, mean and maximum excess, standard
    deviation of the loss. Needs a day of history before the first hour.
    ``metadata``: frequency (GHz), length (km), vertical polarization, and the ITU-R
    P.838-3 coefficients log10(a) and b.

    from core.cml.rnn import HourlyRNN
    rnn = HourlyRNN("dataset/open_datasets/_cml_rnn/models/gru_all")
    hourly = rnn.estimate(links)          # (link, time) mm per hour, hours after the first day
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import preprocess as pp
from .power_law import itu_ab

N_MIN = 60
FEATURES = [f"excess_{m:02d}" for m in range(N_MIN)] + ["missing", "mean_excess", "max_excess", "std_loss"]
META = ["frequency", "length", "vertical", "log10_a", "b"]


def metadata(links: xr.Dataset) -> np.ndarray:
    f = links.frequency.values.astype(float)
    pol = np.asarray(links.polarization.values, dtype=str)
    a, b = itu_ab(np.clip(f, 1, 100), pol, "ITU_2005")
    return np.column_stack([f, links.length.values.astype(float), (np.char.lower(pol) == "v").astype(float),
                            np.log10(a), b]).astype("float32")


def features(links: xr.Dataset, hours: pd.DatetimeIndex) -> np.ndarray:
    """``(link, hour, 64)`` for hours ending at ``hours``; needs a day of history in ``links``."""
    tl = pp.total_loss(links).transpose("link", "time")
    t = pd.DatetimeIndex(tl.time.values)
    # causal baseline: median over the previous 24 h of 15-minute medians
    q = tl.resample(time="15min").median()
    base = q.rolling(time=96, min_periods=16).median().shift(time=1)
    base = base.reindex(time=t, method="ffill")
    excess = (tl - base).values
    loss = tl.values
    # minute stamps in (T - 1h, T] belong to the hour ending T
    idx = pd.DatetimeIndex(hours)
    X = np.zeros((tl.sizes["link"], idx.size, len(FEATURES)), "float32")
    pos = t.get_indexer(idx)                                   # the minute stamped T
    for k, p in enumerate(pos):
        if p < N_MIN - 1:
            X[:, k, N_MIN] = 1.0
            continue
        sl = slice(p - N_MIN + 1, p + 1)
        e = excess[:, sl]
        missing = np.isnan(e)
        X[:, k, :N_MIN] = np.clip(np.nan_to_num(e, nan=0.0), -20, 60)
        X[:, k, N_MIN] = missing.mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X[:, k, N_MIN + 1] = np.nan_to_num(np.nanmean(e, axis=1), nan=0.0)
            X[:, k, N_MIN + 2] = np.nan_to_num(np.nanmax(np.where(missing, -np.inf, e), axis=1),
                                               nan=0.0, neginf=0.0)
            X[:, k, N_MIN + 3] = np.nan_to_num(np.nanstd(loss[:, sl], axis=1), nan=0.0)
    X[:, :, N_MIN + 1:N_MIN + 3] = np.clip(X[:, :, N_MIN + 1:N_MIN + 3], -20, 60)
    X[:, :, N_MIN + 3] = np.clip(X[:, :, N_MIN + 3], 0, 30)
    return X


PHYSICS = ["pl_mean", "pl_max", "pl_of_mean"]


def physics_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """The ITU-R power law applied to the excess loss: ``(link, time, 3)`` in mm/h.

    Per minute, R = (A / (a L)) ** (1 / b) with A the excess loss (dB, >= 0), a and b the link's
    ITU-R P.838-3 coefficients and L its length; then the hour's mean and maximum, and the
    rate of the hour's mean excess. It scales with heavy rain the way the physics does, which a
    network trained mostly on light rain cannot learn from its rare examples. Computed from
    :func:`features` and :func:`metadata` alone.
    """
    a = (10.0 ** M[:, 3])[:, None, None]
    b = M[:, 4][:, None, None]
    length = np.clip(M[:, 1], 0.05, None)[:, None, None]
    exc = np.clip(X[:, :, :N_MIN], 0, None)
    r = (exc / (a * length)) ** (1.0 / b)
    mean_exc = np.clip(X[:, :, N_MIN + 1:N_MIN + 2], 0, None)
    out = np.concatenate([r.mean(-1, keepdims=True), r.max(-1, keepdims=True),
                          (mean_exc / (a * length)) ** (1.0 / b)], axis=-1)
    return np.clip(out, 0, 300).astype("float32")


NEIGHBOURS = ["nbr_pl_median", "nbr_wet_share"]


def neighbour_features(X: np.ndarray, M: np.ndarray, lat, lon, radius_km: float = 15.0) -> np.ndarray:
    """What the surrounding links see: ``(link, time, 2)``.

    For each link, over the other links within ``radius_km`` of its midpoint: the median of
    their power-law rate of the hour (:func:`physics_features`, ``pl_mean``) and the share of
    them whose mean excess loss exceeds 1 dB. The nearby-link method decides wet or dry and
    each link's reference level this way; a single-link network cannot, and misses it most
    where links are dense. Links with no neighbour get 0.
    """
    from core.geo import haversine_m

    pl = physics_features(X, M)[..., 0]                               # (link, time)
    wet = (X[:, :, N_MIN + 1] > 1.0).astype("float32")
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    near = (d <= radius_km * 1000) & (d > 0)
    out = np.zeros(X.shape[:2] + (2,), "float32")
    for i in range(X.shape[0]):
        j = np.flatnonzero(near[i])
        if j.size:
            out[i, :, 0] = np.median(pl[j], axis=0)
            out[i, :, 1] = wet[j].mean(axis=0)
    return out


class HourlyRNN:
    """A model saved by ``projects/retrieval/rnn_three_networks`` (``model.pt``, ``norm.npz``, ``config.json``)."""

    def __init__(self, model_dir: str | Path, network: str | None = None):
        import pynncml as pnc
        import torch
        from pynncml.neural_networks import InputNormalizationConfig

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "config.json").read_text())
        self.cfg = meta["config"]
        cal = meta["history"].get("calibration") or {"gate": 0.5, "scale": 1.0}
        scale = cal["scale"]
        # the scale is fitted per network on validation weeks; unknown networks get 1
        self.gate = cal["gate"]
        self.scale = float(scale.get(network, 1.0)) if isinstance(scale, dict) else float(scale)
        norm = dict(np.load(self.dir / "norm.npz"))
        self.model = pnc.scm.rain_estimation.two_step_network(
            n_layers=self.cfg["n_layers"], rnn_type=getattr(pnc.neural_networks.RNNType, self.cfg["rnn_type"]),
            normalization_cfg=InputNormalizationConfig(norm["mean_dynamic"], norm["std_dynamic"],
                                                       norm["mean_metadata"], norm["std_metadata"]),
            rnn_input_size=len(norm["mean_dynamic"]), rnn_n_features=self.cfg["rnn_n_features"],
            metadata_input_size=len(norm["mean_metadata"]),
            metadata_n_features=self.cfg["metadata_n_features"], pretrained=False)
        self.model.load_state_dict(torch.load(self.dir / "model.pt"))
        self.model.eval()

    def estimate(self, links: xr.Dataset, start=None) -> xr.DataArray:
        """Hour-ending rain (mm) for every whole hour of ``links`` after ``start``
        (default: one day after the first sample, the history the baseline needs)."""
        import torch

        t = pd.DatetimeIndex(links.time.values)
        start = pd.Timestamp(start) if start is not None else t[0] + pd.Timedelta("1D")
        hours = pd.date_range(t[0].ceil("h") + pd.Timedelta("1h"), t[-1].floor("h"), freq="1h")
        Xn, Mn = features(links, hours), metadata(links)
        extra = []
        if self.cfg.get("physics"):
            extra.append(physics_features(Xn, Mn))
        if self.cfg.get("neighbours"):
            extra.append(neighbour_features(Xn, Mn, links.mid_lat.values, links.mid_lon.values))
        if extra:
            Xn = np.concatenate([Xn] + extra, axis=-1)
        X, M = torch.as_tensor(Xn), torch.as_tensor(Mn)
        with torch.no_grad():
            out, _ = self.model(X, M, self.model.init_state(batch_size=X.shape[0]))
        r, p = out[..., 0], out[..., 1]
        if self.cfg.get("log_target"):
            r = torch.expm1(r.clamp(max=6.0))
        rain = (self.scale * r.clamp(min=0.0) * (p > self.gate)).numpy()
        da = xr.DataArray(rain, dims=("link", "time"), coords={"link": links.link.values, "time": hours},
                          name="rain", attrs={"units": "mm", "hourly": True, "method": "rnn"})
        for c in ("mid_lat", "mid_lon", "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
                  "frequency", "length", "polarization"):
            da.coords[c] = ("link", links[c].values)
        return da.sel(time=slice(start + pd.Timedelta("1h"), None))
