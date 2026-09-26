"""
The nowcasting pipeline as a few cached stages.

    inputs      radar, gauges, link midpoints, per-link physical rain
    cml_maps    IDW and GMZ maps from the physical per-link rain ("Model 1")
    estimation  the two-step RNN ("Model 2"): per-link rain, then IDW/GMZ maps
    forecasts   persistence baselines, Transformer, GRU, POD-SINDy, pySTEPS
    evaluate    scores vs gauges and radar

Each stage caches its outputs under ``cfg.path(...)``, keyed by a hash of
exactly the settings it depends on (``NowcastConfig.STAGE_KEYS``): change
the epochs and the models retrain, change nothing and nothing reruns. W&B sweeps are not run: hyper-parameters come from ``cfg``
(the notebook's defaults, or the best configs its sweeps saved).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

import data
import evaluation as ev
import forecasting as fc
import maps
from config import NowcastConfig


def _stage(cfg: NowcastConfig, name: str, build):
    path = cfg.path(f"{name}_{cfg.stage_key(name)}.pkl")
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            return pickle.load(f)
    out = build()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def inputs(cfg: NowcastConfig) -> dict:
    """Radar, gauges and links on the radar's 15-minute axis."""
    def build():
        ds = data.link_dataset(cfg)
        rad = data.radar_15min(cfg)
        lon, lat = data.link_midpoints(ds)
        return {"radar": rad, "gauges": data.gauges_15min(ds, rad["times"], cfg.full.start),
                "link_lon": lon, "link_lat": lat,
                "links_physical": data.links_physical(ds, rad["times"], cfg.full.start),
                "crop": data.crop_bounds(rad["lon"], rad["lat"], cfg.lat_range, cfg.lon_range)}
    return _stage(cfg, "inputs", build)


def _maps_from_links(cfg, inp, per_link):
    rad = inp["radar"]
    metric = "degrees" if cfg.faithful else "km"
    idw = maps.idw(per_link, inp["link_lon"], inp["link_lat"], rad["lon"], rad["lat"],
                   cfg.idw_power, metric)
    gmz = maps.gmz(data.link_dataset(cfg).link_set, per_link, rad["lon"], rad["lat"],
                   cfg.gmz_roi, cfg.gmz_points_per_link, cfg.faithful,
                   inp["link_lon"], inp["link_lat"])
    return {"idw": idw, "gmz": gmz}


def cml_maps(cfg: NowcastConfig, inp: dict) -> dict:
    """IDW and GMZ maps of the physical per-link retrieval."""
    return _stage(cfg, "cml_maps",
                  lambda: _maps_from_links(cfg, inp, inp["links_physical"].values))


def estimation(cfg: NowcastConfig, inp: dict) -> dict:
    """Train the two-step RNN on the train split, map its per-link estimates."""
    def build():
        import scipy.stats
        from core.scientific_packages import pynncml_rnn as rnn

        hp = cfg.est_hparams
        tcfg = rnn.TrainConfig(batch_size=hp["batch_size"], rnn_n_features=hp["rnn_n_features"],
                               n_layers=hp["n_layers"], lr=hp["lr"],
                               weight_decay=hp["weight_decay"], n_epochs=cfg.est_epochs,
                               clip_grad=1.0, balance_batches=1, seed=cfg.seed)
        train_ds, val_ds = data.link_dataset(cfg, "train"), data.link_dataset(cfg, "val")
        train_loader = torch.utils.data.DataLoader(train_ds, tcfg.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, tcfg.batch_size)
        rain = np.stack([p.data_array for p in train_ds.point_set]).ravel()
        model = rnn.build_model(tcfg, train_loader)
        history = rnn.train(model, train_loader, tcfg, scipy.stats.expon.fit(rain)[1],
                            progress=False, val_loader=val_loader)
        per_link = _infer_links(model, data.link_dataset(cfg), tcfg, inp, cfg)
        return {"history": history, **_maps_from_links(cfg, inp, per_link)}
    return _stage(cfg, "estimation", build)


def _infer_links(model, dataset, tcfg, inp, cfg) -> np.ndarray:
    """Per-link RNN estimates over the full period, on the radar's axis."""
    from core.scientific_packages import pynncml_rnn as rnn

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        parts = [hat * torch.round(p) for hat, p, _ in rnn.windows(model, batch, tcfg)]
    pred = torch.cat(parts, dim=1).cpu().numpy().T                      # (T', n_links)
    span = (pd.Timestamp(cfg.full.stop) + pd.Timedelta(days=1)
            - pd.Timestamp(cfg.full.start)).total_seconds() / 60
    candidates = np.array([1, 5, 15, 30, 60])
    step = int(candidates[np.argmin(np.abs(candidates - span / max(len(pred), 1)))])
    times = pd.date_range(cfg.full.start, periods=len(pred), freq=f"{step}min")
    return data._to_15min(pred, times, None, inp["radar"]["times"]).values


def _splits(cfg, times, maps_):
    idx = {w: cfg.split_index(times, w) for w in ("train", "val", "test")}
    return {w: maps_[lo:hi] for w, (lo, hi) in idx.items()}, idx


def forecasts(cfg: NowcastConfig, inp: dict, est: dict, cml: dict) -> dict:
    """Every model's test-split forecasts, ``{name: {h: (N_h, H, W)}}``."""
    def build():
        rad = inp["radar"]
        shape = rad["R"].shape[1:]
        radar_splits, idx = _splits(cfg, rad["times"], rad["R"])
        lb, hs = cfg.lookback, cfg.horizons
        out = {"persistence_radar": fc.persistence(radar_splits["test"], lb, hs),
               # diagnostic: the one-frame leak of the notebook's pySTEPS, alone
               "persistence_radar_one_step_late": fc.persistence(radar_splits["test"], lb, hs,
                                                                 one_step_late=True)}
        gauge_field = maps.idw(inp["gauges"]["R"].values, inp["gauges"]["lon"],
                               inp["gauges"]["lat"], rad["lon"], rad["lat"], cfg.idw_power,
                               "degrees" if cfg.faithful else "km")
        out["persistence_gauge"] = fc.persistence(_splits(cfg, rad["times"], gauge_field)[0]["test"], lb, hs)
        out["persistence_cml"] = fc.persistence(_splits(cfg, rad["times"], est["idw"])[0]["test"], lb, hs)
        t_lo = idx["test"][0]
        n = len(radar_splits["test"]) - lb - max(hs) + 1
        # a current-time CML estimate, not a forecast: a reference ceiling
        out["cml_now_physical"] = {h: cml["idw"][t_lo + lb - 1 + h:t_lo + lb - 1 + h + n] for h in hs}

        factories = {
            "transformer": lambda k: fc.RainTransformer(shape, cfg.transformer["d_model"],
                                                         cfg.transformer["n_heads"],
                                                         cfg.transformer["n_layers"], k),
            "gru": lambda k: fc.GRUForecaster(shape, cfg.gru["hidden_dim"],
                                              cfg.gru["n_layers"], k),
        }
        histories = {}
        for map_type in ("idw", "gmz"):
            splits = _splits(cfg, rad["times"], est[map_type])[0]
            for model_name, factory in factories.items():
                hp = cfg.transformer if model_name == "transformer" else cfg.gru
                grid = fc.train_grid(factory, splits, lb, hs, hp, cfg.forecast_epochs, cfg.seed)
                out[f"{model_name}_single_{map_type}"] = grid["single"]
                out[f"{model_name}_multi_{map_type}"] = grid["multi"]
                histories[f"{model_name}_{map_type}"] = grid["histories"]
            sindy = fc.PODSindy(cfg.sindy_modes, seed=cfg.seed).fit(splits["train"])
            out[f"sindy_{map_type}"] = sindy.predict(splits["test"], lb, hs)
        return {"forecasts": out, "histories": histories}
    result = _stage(cfg, "forecasts", build)
    # pySTEPS is cached on its own: it trains nothing, and a run without it
    # installed must not leave a cache that skips it once it is
    result["forecasts"].pop("pysteps", None)
    try:
        result["forecasts"]["pysteps"] = pysteps(cfg, inp)
    except ImportError as e:            # pySTEPS or OpenCV missing: baseline skipped
        print(f"pySTEPS baseline skipped ({e}); pip install pysteps opencv-python-headless")
    return result


def pysteps(cfg: NowcastConfig, inp: dict) -> dict:
    """Lucas-Kanade extrapolation of the radar test split, ``{h: (N_h, H, W)}``."""
    def build():
        rad = inp["radar"]
        test = _splits(cfg, rad["times"], rad["R"])[0]["test"]
        return fc.pysteps_forecasts(test, cfg.lookback, cfg.horizons, faithful=cfg.faithful)
    return _stage(cfg, "pysteps", build)


def evaluate(cfg: NowcastConfig, inp: dict, fcs: dict):
    """(scores vs gauges, scores vs radar) over the test split."""
    rad, g = inp["radar"], inp["gauges"]
    lo, hi = cfg.split_index(rad["times"], "test")
    pixels = ev.gauge_pixels(rad["lon"], rad["lat"], g["lon"], g["lat"])
    return ev.score(fcs["forecasts"], rad["R"][lo:hi], g["R"].values[lo:hi], pixels,
                    cfg.lookback, inp["crop"], cfg.wet_threshold)


def run(cfg: NowcastConfig) -> dict:
    """Every stage, cached; returns everything the notebook shows."""
    inp = inputs(cfg)
    cml = cml_maps(cfg, inp)
    est = estimation(cfg, inp)
    fcs = forecasts(cfg, inp, est, cml)
    vs_gauges, vs_radar = evaluate(cfg, inp, fcs)
    return {"inputs": inp, "cml": cml, "estimation": est, "forecasts": fcs,
            "vs_gauges": vs_gauges, "vs_radar": vs_radar}

