"""Link rain every 5 minutes: the RNN's hourly totals, spread by the power law's timing.

The best link retrieval FieldSense has is the two-step RNN of ``projects/retrieval/rnn_three_networks``
(``gru_physics_nbr``), but it estimates hourly totals. Nowcasting needs 5 minutes. The
RNN's inputs, cached by ``projects/retrieval/rnn_three_networks`` for every link and hour
(``_cml_rnn/<network>.nc``), include the 60 one-minute values of each link's excess loss
(total loss minus a causal baseline). So:

1. the saved model is run on the cached features, as :class:`core.cml.rnn.HourlyRNN`
   does (physics and neighbour features, the per-network calibration scale and wet gate):
   hour-ending totals ``H`` (mm);
2. the ITU-R power law turns each minute's excess loss (less a 0.3 dB noise floor) into a
   rate; the twelve 5-minute means give the *timing* of the rain within the hour;
3. the 5-minute rates are that pattern scaled so the hour's mean equals ``H``; where the
   power law sees nothing but the RNN does, ``H`` is spread evenly.

The power law's own 5-minute rates (step 2, unscaled) are kept as ``pl`` for comparison.
Sublinks of one path are averaged. Link-hours with more than half the minutes missing
are NaN.

Leakage: the RNN was trained against radar along the path and gauges within 3 km of it
(Netatmo and city gauges in Gothenburg, WU PWS in New York) on its *training* weeks. Every
forecast scored in this project is in the RNN's *test* weeks (:func:`split`), the same
weeks for the nowcasting models.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.cml.rnn import N_MIN, neighbour_features, physics_features

from .settings import CACHE_DIR, RNN_DIR, RNN_MODEL, STEP_MIN

log = logging.getLogger(__name__)

NOISE_DB = 0.3
MAX_MISSING = 0.5
PER_HOUR = 60 // STEP_MIN


def split(times, network: str) -> np.ndarray:
    """0 train, 1 validation, 2 test, per time - the rule of ``projects/retrieval/rnn_three_networks``
    (``settings.split``): ISO weeks cycle train, train, validation, test, and the study
    storms of ``projects/maps/multisensor`` (+-1 day) are always test."""
    times = pd.DatetimeIndex(times)
    # hour-ending labels: a 5-min step belongs to the hour that ends at or after it
    hours = times.ceil("h")
    week = np.asarray(hours.isocalendar().week, dtype=int)
    out = np.select([week % 4 == 2, week % 4 == 3], [1, 2], 0)
    for s, e in _study_events().get(network, []):
        sel = (hours > pd.Timestamp(s) - pd.Timedelta("1D")) & (hours <= pd.Timestamp(e) + pd.Timedelta("1D"))
        out[sel] = 2
    return out


def _study_events() -> dict:
    from core.data_paths import REPO_ROOT
    p = REPO_ROOT / "projects" / "maps" / "multisensor" / "results" / "events.csv"
    df = pd.read_csv(p)
    return {n: list(zip(g.start, g.end)) for n, g in df.groupby("network")}


def _model(network: str):
    import pynncml as pnc
    import torch
    from pynncml.neural_networks import InputNormalizationConfig

    d = RNN_DIR / "models" / RNN_MODEL
    meta = json.loads((d / "config.json").read_text())
    cfg = meta["config"]
    cal = meta["history"].get("calibration") or {"gate": 0.5, "scale": 1.0}
    scale = cal["scale"]
    scale = float(scale.get(network, 1.0)) if isinstance(scale, dict) else float(scale)
    norm = dict(np.load(d / "norm.npz"))
    model = pnc.scm.rain_estimation.two_step_network(
        n_layers=cfg["n_layers"], rnn_type=getattr(pnc.neural_networks.RNNType, cfg["rnn_type"]),
        normalization_cfg=InputNormalizationConfig(norm["mean_dynamic"], norm["std_dynamic"],
                                                   norm["mean_metadata"], norm["std_metadata"]),
        rnn_input_size=len(norm["mean_dynamic"]), rnn_n_features=cfg["rnn_n_features"],
        metadata_input_size=len(norm["mean_metadata"]),
        metadata_n_features=cfg["metadata_n_features"], pretrained=False)
    model.load_state_dict(torch.load(d / "model.pt"))
    model.eval()
    return model, cfg, cal["gate"], scale


def rnn_hourly(ds: xr.Dataset, network: str) -> np.ndarray:
    """``(link, hour)`` RNN totals (mm) from the cached features."""
    import torch
    model, cfg, gate, scale = _model(network)
    X = ds.X.values.astype("float32")
    M = ds.M.values.astype("float32")
    extra = []
    if cfg.get("physics"):
        extra.append(physics_features(X, M))
    if cfg.get("neighbours"):
        extra.append(neighbour_features(X, M, ds.mid_lat.values, ds.mid_lon.values))
    if extra:
        X = np.concatenate([X] + extra, axis=-1)
    with torch.no_grad():
        out, _ = model(torch.as_tensor(X), torch.as_tensor(M), model.init_state(batch_size=X.shape[0]))
    r, p = out[..., 0], out[..., 1]
    if cfg.get("log_target"):
        r = torch.expm1(r.clamp(max=6.0))
    return (scale * r.clamp(min=0.0) * (p > gate)).numpy()


def pattern_5min(ds: xr.Dataset) -> np.ndarray:
    """``(link, hour, 12)`` power-law rates (mm/h) of the excess loss, 5-minute means."""
    M = ds.M.values.astype(float)
    a = (10.0 ** M[:, 3])[:, None, None]
    b = M[:, 4][:, None, None]
    length = np.clip(M[:, 1], 0.05, None)[:, None, None]
    exc = np.clip(ds.X.values[:, :, :N_MIN].astype(float) - NOISE_DB, 0, None)
    r = (exc / (a * length)) ** (1.0 / b)
    return np.clip(r, 0, 300).reshape(r.shape[0], r.shape[1], PER_HOUR, -1).mean(-1)


def link_rain(network: str, refresh: bool = False) -> xr.Dataset:
    """5-minute link rain (mm/h) per physical link: ``rnn`` (scaled) and ``pl``, ``(link, time)``."""
    out_path = CACHE_DIR / network / "link_rain_5min.nc"
    if out_path.exists() and not refresh:
        return xr.open_dataset(out_path).load()
    ds = xr.open_dataset(RNN_DIR / f"{network}.nc")
    hours = pd.DatetimeIndex(ds.time.values)
    H = rnn_hourly(ds, network)                                   # (link, hour) mm
    pat = pattern_5min(ds)                                        # (link, hour, 12) mm/h
    pm = pat.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(pm > 0.01, pat * (H[..., None] / pm), H[..., None])
    missing = ds.X.sel(feature="missing").values > MAX_MISSING
    scaled[missing] = np.nan
    pat[missing] = np.nan
    # 5-min steps ending at T-55 ... T for the hour ending T
    offs = pd.to_timedelta(np.arange(-55, 1, STEP_MIN), "min")
    times = (hours.values[:, None] + offs.values[None, :]).ravel()
    n = ds.sizes["link"]
    rnn = scaled.reshape(n, -1)
    pl = pat.reshape(n, -1)
    sub = xr.Dataset({"rnn": (("sublink", "time"), rnn.astype("float32")),
                      "pl": (("sublink", "time"), pl.astype("float32"))},
                     coords={"time": times, "cml_id": ("sublink", ds.cml_id.values)})
    phys = sub.groupby("cml_id").mean("sublink", skipna=True).rename(cml_id="link")
    first = pd.DataFrame({c: ds[c].values for c in ["cml_id", "site_0_lat", "site_0_lon", "site_1_lat",
                                                     "site_1_lon", "mid_lat", "mid_lon", "length"]}) \
        .drop_duplicates("cml_id").set_index("cml_id").loc[phys.link.values]
    phys = phys.assign_coords({c: ("link", first[c].to_numpy()) for c in first.columns})
    phys.attrs = {"units": "mm/h", "step": f"{STEP_MIN}min", "label": "end",
                  "retrieval": f"projects/retrieval/rnn_three_networks {RNN_MODEL} hourly totals, spread over 5-min steps "
                               "by the power-law rate of the 1-min excess loss",
                  "noise_db": NOISE_DB}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phys.to_netcdf(out_path)
    log.info("%s: link rain for %d links, %d steps", network, phys.sizes["link"], phys.sizes["time"])
    return phys
