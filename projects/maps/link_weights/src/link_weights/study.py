"""Link error by length learned on half the storms, used as IDW weights on the other half.

For every cached storm of ``maps/multisensor`` and two retrievals, each link's hourly total
is compared with the radar averaged along its path. On the training storms of a network
(every other storm) the mean squared error per length bin gives each bin an error variance;
on the test storms links are mapped three ways - plain IDW, IDW weighted by the inverse of
their bin's error variance, and plain IDW without the links shorter than 1 km - and scored
against the radar and at held-out gauges.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from core.maps.geometry import path_average_points
from core.maps.idw import idw_map
from core.maps.scores import scores

from .settings import EVENT_INPUTS, GAUGES, IDW, LENGTH_BINS_KM, MIN_SHORT_KM, RETRIEVALS, WET_MM

log = logging.getLogger(__name__)
PATH = ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]


def _events() -> pd.DataFrame:
    rows = [{"folder": p, "network": p.name.split("_")[0], "event": p.name}
            for p in sorted(EVENT_INPUTS.glob("*_*")) if (p / "meta.json").exists()]
    df = pd.DataFrame(rows)
    df["split"] = df.groupby("network").cumcount().map(lambda i: "train" if i % 2 == 0 else "test")
    return df


def _load(folder, retrieval):
    meta = json.loads((folder / "meta.json").read_text())
    if retrieval not in meta["links"]:
        return None
    links = xr.open_dataarray(folder / f"links_{retrieval}.nc").load().transpose("link", "time")
    radar = xr.open_dataarray(folder / "radar_0.nc").load()
    return links, radar, meta


def link_errors(events: pd.DataFrame) -> pd.DataFrame:
    """One row per link-hour of every storm: link total, radar along its path, length."""
    rows = []
    for ev in events.itertuples():
        for ret in RETRIEVALS:
            got = _load(ev.folder, ret)
            if got is None:
                continue
            links, radar, _ = got
            table = pd.DataFrame({c: links[c].values for c in PATH}, index=pd.Index(links.link.values, name="link"))
            t = np.intersect1d(links.time.values, radar.time.values)
            rp = path_average_points(radar.sel(time=t), table).transpose("link", "time")
            lk = links.sel(time=t)
            df = pd.DataFrame({"link_mm": lk.values.ravel(), "radar_mm": rp.values.ravel(),
                               "length_km": np.repeat(links.length.values, t.size)})
            df = df[np.isfinite(df.link_mm) & np.isfinite(df.radar_mm)]
            rows.append(df.assign(network=ev.network, event=ev.event, split=ev.split, retrieval=ret))
    return pd.concat(rows, ignore_index=True)


def length_bin(L) -> pd.Categorical:
    labels = [f"{a:g}-{b:g} km" if b < 100 else f">{a:g} km" for a, b in zip(LENGTH_BINS_KM[:-1], LENGTH_BINS_KM[1:])]
    return pd.cut(np.asarray(L, float), LENGTH_BINS_KM, labels=labels, right=False)


def error_model(errors: pd.DataFrame) -> pd.DataFrame:
    """Per network, retrieval and length bin, on training storms: MSE, NRMSE, bias, n."""
    tr = errors[errors.split == "train"].assign(bin=lambda d: length_bin(d.length_km))
    tr = tr[(tr.radar_mm >= WET_MM) | (tr.link_mm >= WET_MM)]
    g = tr.groupby(["network", "retrieval", "bin"], observed=True)
    out = g.apply(lambda d: pd.Series({
        "n": len(d), "mse": float(np.mean((d.link_mm - d.radar_mm) ** 2)),
        "nrmse": float(np.sqrt(np.mean((d.link_mm - d.radar_mm) ** 2)) / max(d.radar_mm.mean(), 1e-9)),
        "rel_bias": float(d.link_mm.sum() / max(d.radar_mm.sum(), 1e-9) - 1)}), include_groups=False)
    return out.reset_index()


def _weights(links, model: pd.DataFrame, network, retrieval) -> np.ndarray:
    m = model[(model.network == network) & (model.retrieval == retrieval)].set_index("bin")["mse"]
    mse = m.reindex(length_bin(links.length.values).astype(str)).to_numpy(float)
    w = 1.0 / np.where(np.isfinite(mse), mse, np.nanmax(m.to_numpy(float)))
    return w / np.nanmean(w)


def _at(field, points):
    lat = xr.DataArray(points.lat.values, dims="station")
    lon = xr.DataArray(points.lon.values, dims="station")
    v = field.sel(lat=lat, lon=lon, method="nearest").transpose("station", "time")
    return v.drop_vars(["lat", "lon"]).assign_coords(station=points.station.values)


def _score(est, ref) -> dict:
    s = scores(est.values, ref.values, wet_threshold=WET_MM)
    return {k: s.get(k, np.nan) for k in ("n", "nrmse", "corr", "rel_bias", "csi")}


def evaluate(events: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Test storms: plain, weighted and no-short-links IDW against radar and gauges."""
    rows = []
    for ev in events[events.split == "test"].itertuples():
        for ret in RETRIEVALS:
            got = _load(ev.folder, ret)
            if got is None:
                continue
            links, radar, meta = got
            grid = Grid(radar.lat.values, radar.lon.values)
            long_ = links.length.values >= MIN_SHORT_KM
            maps = {"idw": idw_map(links, grid, **IDW),
                    "idw weighted by length error": idw_map(links, grid, weights=_weights(links, model, ev.network, ret), **IDW),
                    "idw without links < 1 km": idw_map(links.isel(link=np.flatnonzero(long_)), grid, **IDW)}
            g = GAUGES.get(ev.network)
            gauges = xr.open_dataarray(ev.folder / f"points_{g}.nc").load() if g in meta["points"] else None
            for name, m in maps.items():
                t = np.intersect1d(m.time.values, radar.time.values)
                base = {"network": ev.network, "event": ev.event, "retrieval": ret, "map": name,
                        "short_share": float(1 - long_.mean())}
                rows.append({**base, "reference": "radar", **_score(m.sel(time=t), radar.sel(time=t))})
                if gauges is not None:
                    tg = np.intersect1d(m.time.values, gauges.time.values)
                    rows.append({**base, "reference": "gauges",
                                 **_score(_at(m.sel(time=tg), gauges), gauges.sel(time=tg).transpose("station", "time"))})
        log.info("%s done", ev.event)
    return pd.DataFrame(rows)


def run(results_dir) -> dict:
    ev = _events()
    errors = link_errors(ev)
    model = error_model(errors)
    scores_ = evaluate(ev, model)
    results_dir.mkdir(parents=True, exist_ok=True)
    model.round(4).to_csv(results_dir / "error_model.csv", index=False)
    scores_.round(4).to_csv(results_dir / "test_scores.csv", index=False)
    ev.drop(columns="folder").to_csv(results_dir / "events.csv", index=False)
    return {"model": model, "scores": scores_}


def summary(scores_: pd.DataFrame) -> pd.DataFrame:
    return (scores_.groupby(["reference", "network", "retrieval", "map"])[["nrmse", "corr", "rel_bias", "csi"]]
            .median().round(3).reset_index())
