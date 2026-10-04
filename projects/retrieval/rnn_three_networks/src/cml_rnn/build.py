"""Build one hourly training table per network: link signals in, several references out.

For every link that passes metadata QC and every hour of the network's periods:

inputs ``X(link, time, 64)``
    60 one-minute values of *excess loss* - total loss minus a causal 24-hour baseline
    (the median of the previous day's 15-minute medians) - and four hourly summaries:
    the share of missing minutes (an outage in rain is itself evidence of rain), the mean
    and maximum excess, and the standard deviation of the loss within the hour.
metadata ``M(link, 5)``
    frequency (GHz), length (km), vertical polarization (0/1), and the ITU-R P.838-3
    coefficients log10(a) and b at that frequency and polarization.
references ``(link, time)``, mm in the hour
    ``radar``: the radar averaged along the path; ``<point set>``: the mean of that set's
    gauges within 3 km of the path; ``target``: the mean of whichever are available -
    the average of different sources the RNN is trained on.
power-law estimates ``pl_<method>(link, time)``, mm in the hour
    every method of ``core.cml`` on the same links and hours, for the comparison.

Links are processed ``CHUNK`` days at a time with a day of history in front, so
baselines and RNN context never start cold. Saved as ``DATA_DIR/<network>.nc``.

    python projects/retrieval/rnn_three_networks/src/run.py build --network openrainer
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.cml.estimators import study_estimator
from core.cml.rnn import FEATURES, META, features, metadata
from core.cml.link_qc import QCConfig, metadata_qc
from core.maps.idw import accumulate
from core.opensense.networks import NETWORKS, points_near_links, radar_along_links

from .settings import CHUNK, DATA_DIR, GAUGE_RADIUS_KM, PERIODS, PL_METHODS, SPINUP, TARGET_POINTS

log = logging.getLogger(__name__)

def link_labels(network: str) -> list:
    """Links that pass metadata QC; both directions kept (the RNN learns from each)."""
    net = NETWORKS[network]
    keep, _ = metadata_qc(net.links_table(), QCConfig(domain=net.domain.pad(0.05), duplicate_tolerance_m=-1))
    return keep


def pl_estimates(links: xr.Dataset, t0, t1) -> dict:
    """Hourly link totals of every power-law method, hours ending in ``(t0, t1]``."""
    out = {}
    for m in PL_METHODS:
        try:
            est = study_estimator(m, links.time.values[0], links.time.values[-1], threshold_links=links)
            rain = est.estimate(links)["rain"].sel(time=slice(pd.Timestamp(t0) + pd.Timedelta("1min"), t1))
            out[m] = accumulate(rain, "1h").sel(time=slice(pd.Timestamp(t0) + pd.Timedelta("1h"), t1))
        except Exception as exc:
            log.warning("%s failed: %r", m, exc)
    return out


def build_chunk(network: str, labels: list, t0, t1) -> xr.Dataset:
    net = NETWORKS[network]
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    links = net.links(t0 - pd.Timedelta(SPINUP), t1, labels)
    hours = pd.date_range(t0 + pd.Timedelta("1h"), t1, freq="1h")
    ds = xr.Dataset(coords={"link": np.asarray(labels, dtype=object), "time": hours,
                            "feature": FEATURES})
    ds["X"] = (("link", "time", "feature"), features(links, hours))
    radar = net.radar_hourly(t0, t1).reindex(time=hours)
    ds["radar"] = radar_along_links(radar, links).reindex(time=hours).astype("float32")
    pts = net.points_hourly(t0, t1, sets=TARGET_POINTS[network])
    refs = [ds["radar"]]
    for p in TARGET_POINTS[network]:
        if p in pts:
            ds[p] = points_near_links(pts[p], links, GAUGE_RADIUS_KM).reindex(time=hours).astype("float32")
            refs.append(ds[p])
    with np.errstate(invalid="ignore"):
        ds["target"] = xr.concat(refs, "source").mean("source", skipna=True).astype("float32")
    for m, h in pl_estimates(links, t0, t1).items():
        ds[f"pl_{m}"] = h.reindex(link=ds.link, time=hours).astype("float32")
    return ds


def build(network: str, overwrite: bool = False) -> xr.Dataset:
    path = DATA_DIR / f"{network}.nc"
    if path.exists() and not overwrite:
        return xr.open_dataset(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels = link_labels(network)
    net = NETWORKS[network]
    meta_links = net.links(PERIODS[network][0][0], pd.Timestamp(PERIODS[network][0][0]) + pd.Timedelta("1h"), labels)
    parts = []
    for p0, p1 in PERIODS[network]:
        for c0 in pd.date_range(p0, p1, freq=CHUNK, inclusive="left"):
            c1 = min(c0 + pd.Timedelta(CHUNK), pd.Timestamp(p1))
            log.info("%s %s .. %s (%d links)", network, c0, c1, len(labels))
            parts.append(build_chunk(network, labels, c0, c1))
    ds = xr.concat(parts, "time", data_vars="all", coords="minimal", compat="override")
    ds["M"] = (("link", "meta"), metadata(meta_links))
    ds = ds.assign_coords(meta=META)
    for c in ["frequency", "length", "polarization", "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
              "mid_lat", "mid_lon"]:
        ds.coords[c] = ("link", meta_links[c].values)
    ds.attrs = {"network": network, "gauge_radius_km": GAUGE_RADIUS_KM,
                "target": "mean of radar along the path and the gauges near it"}
    for c in ds.coords:                        # netCDF stores fixed-width strings, not objects
        if ds[c].dtype == object:
            ds.coords[c] = ds[c].astype(str)
    enc = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    return ds
