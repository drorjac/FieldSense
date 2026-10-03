"""Supervised data for learned link-to-map models: link inputs on the grid, radar target.

A learned mapping ``g(links, metadata) -> rain field`` needs, for every time step, the
link observations in a form a model can read, the reference field on the same grid, and
an honest split. This module builds that from the pieces FieldSense already has
(``core.opensense.networks`` for the data, ``core.cml`` for QC and retrieval,
``core.events`` for storm events) and stores it as one ``xarray.Dataset``:

``inputs(time, channel, lat, lon)``
    the links rasterized onto the target grid, NaN-free (0 where no link):
    ``rain`` - path-length-weighted mean of the link values in the cell (mm in the step);
    ``attenuation`` - the same for specific rain-induced attenuation (dB/km);
    ``coverage`` - km of valid link path in the cell; ``length`` - mean length (km) of
    the links crossing the cell
``target(time, lat, lon)``
    the reference field: radar (real networks) or the simulated truth
``link_rain(link, time)``, ``link_attenuation(link, time)``
    the same observations as a table, with ``site_0/1_lat/lon``, ``mid_lat/lon``,
    ``frequency``, ``polarization`` and ``length`` per link: the input of graph models
    and of every classical method in ``core.maps``
``gauges(station, time)``
    point gauges with ``lat``/``lon`` and ``station_set``; never an input, kept for the
    independent check
``distance_km(lat, lon)``, ``observable(lat, lon)``
    distance to the nearest link path, and the cells within ``near_km`` of one, where a
    link map can be scored at all
``event_id(time)``, ``split(time)`` and the event table (``event`` dimension)
    every step belongs to one storm event, every event to one split (train/val/test):
    the split is by event, never by time step, so neighbouring hours of one storm
    never sit on both sides

Rasterization keeps the path average: a link of value ``v`` crossing cells with path
fractions ``w_c`` gives ``sum_c w_c * rain_c = v`` when no other link shares its cells,
and ``sum_c rain_c * coverage_c = sum_i L_i v_i`` always. :func:`sample_paths` is the
discrete forward operator back (field -> path averages), which physics-aware losses and
posterior-sampling methods need.

    from core.maps.learning import build_network_dataset, load_dataset
    ds = build_network_dataset("openmrg", cache_dir=...)       # every radar event of JJA 2015
    train = ds.sel(time=ds.split == "train")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid, haversine_m, to_local_xy

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
CHANNELS = ("rain", "attenuation", "coverage", "length")
LINK_COORDS = ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon", "mid_lat", "mid_lon",
               "frequency", "polarization", "length")


# ---------------------------------------------------------------------------
# geometry: links <-> grid
# ---------------------------------------------------------------------------
def link_lengths_km(links: xr.DataArray | xr.Dataset) -> np.ndarray:
    """Path length (km) of each link: the ``length`` / ``length_km`` coordinate, else haversine."""
    for name in ("length", "length_km"):
        if name in links.coords or name in getattr(links, "data_vars", {}):
            return np.asarray(links[name].values, dtype=float)
    return haversine_m(links.site_0_lat.values, links.site_0_lon.values,
                       links.site_1_lat.values, links.site_1_lon.values) / 1000.0


def path_weights(links: xr.DataArray | xr.Dataset, grid: Grid, step_km: float = 0.05) -> np.ndarray:
    """Fraction of each link's path in each grid cell, ``(cells, links)``.

    The path is cut into equal pieces of at most ``step_km``; each piece goes to the cell
    whose centre is nearest to it. Columns sum to 1 for a path inside the grid, less for
    a path that leaves it.
    """
    la0, lo0 = links.site_0_lat.values.astype(float), links.site_0_lon.values.astype(float)
    la1, lo1 = links.site_1_lat.values.astype(float), links.site_1_lon.values.astype(float)
    length = link_lengths_km(links)
    dlat, dlon = float(grid.lat[1] - grid.lat[0]), float(grid.lon[1] - grid.lon[0])
    W = np.zeros((grid.lat.size * grid.lon.size, la0.size))
    for k in range(la0.size):
        n = max(2, int(np.ceil(length[k] / step_km)))
        s = (np.arange(n) + 0.5) / n
        i = np.round((la0[k] + s * (la1[k] - la0[k]) - grid.lat[0]) / dlat).astype(int)
        j = np.round((lo0[k] + s * (lo1[k] - lo0[k]) - grid.lon[0]) / dlon).astype(int)
        inside = (i >= 0) & (i < grid.lat.size) & (j >= 0) & (j < grid.lon.size)
        np.add.at(W[:, k], i[inside] * grid.lon.size + j[inside], 1.0 / n)
    return W


def rasterize(values: np.ndarray, weights: np.ndarray, length_km: np.ndarray,
              shape: tuple) -> dict:
    """Link values ``(link, time)`` -> ``{"mean", "coverage", "length"}``, each ``(time, *shape)``.

    ``mean`` is the path-length-weighted mean of the valid links crossing a cell (NaN where
    none), ``coverage`` the km of valid path in the cell, ``length`` the mean length of
    those links. A NaN link drops out of that time step only.
    """
    V = np.asarray(values, dtype=float)
    ok = np.isfinite(V)
    C = weights * np.asarray(length_km, dtype=float)[None, :]           # km of path per cell
    num = C @ np.where(ok, V, 0.0)
    den = C @ ok.astype(float)
    lng = C @ (ok * np.asarray(length_km, dtype=float)[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
        length = np.where(den > 0, lng / den, np.nan)
    def to_grid(a):
        return a.T.reshape((V.shape[1],) + tuple(shape))
    return {"mean": to_grid(mean), "coverage": to_grid(den), "length": to_grid(length)}


def sample_paths(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Path average of a field ``(time, lat, lon)`` along each link: ``(link, time)``.

    The discrete forward operator of the links (rain averaged along the path, the
    cells weighted by the share of the path in them); NaN cells drop out.
    """
    F = np.asarray(field, dtype=float).reshape(np.shape(field)[0], -1).T     # (cells, time)
    ok = np.isfinite(F)
    num = weights.T @ np.where(ok, F, 0.0)
    den = weights.T @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def distance_to_links(grid: Grid, links: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Distance (km) from each cell centre to the nearest link path, ``(lat, lon)``."""
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    glat, glon = grid.mesh()
    px, py = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    ax, ay = to_local_xy(links.site_0_lat.values, links.site_0_lon.values, lat0, lon0)
    bx, by = to_local_xy(links.site_1_lat.values, links.site_1_lon.values, lat0, lon0)
    dx, dy = bx - ax, by - ay
    t = ((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(t, 0, 1)
    d = np.hypot(px[:, None] - (ax + t * dx), py[:, None] - (ay + t * dy)).min(axis=1)
    return xr.DataArray((d / 1000.0).reshape(grid.shape), dims=("lat", "lon"),
                        coords={"lat": grid.lat, "lon": grid.lon}, name="distance_km",
                        attrs={"units": "km"})


# ---------------------------------------------------------------------------
# events and splits
# ---------------------------------------------------------------------------
def split_events(event_ids, fractions=(0.6, 0.2, 0.2), seed: int = 0,
                 how: str = "random") -> pd.Series:
    """Assign whole events to ``train``/``val``/``test``: ``Series(event_id -> split)``.

    ``how="random"`` shuffles the events with ``seed``; ``"time"`` keeps their order
    (earliest to train, latest to test). Validation and test get at least one event each
    when there are three or more.
    """
    ids = list(event_ids)
    n = len(ids)
    if how == "random":
        ids = [ids[i] for i in np.random.default_rng(seed).permutation(n)]
    elif how != "time":
        raise ValueError("how must be 'random' or 'time'")
    f = np.asarray(fractions, dtype=float) / np.sum(fractions)
    n_test = max(1 if n >= 3 else 0, int(round(f[2] * n)))
    n_val = max(1 if n >= 3 else 0, int(round(f[1] * n)))
    n_train = n - n_val - n_test
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    out = pd.Series(labels, index=pd.Index(ids, name="event_id"), name="split")
    return out.reindex(list(event_ids))


# ---------------------------------------------------------------------------
# assembling the dataset
# ---------------------------------------------------------------------------
def _link_table(link_rain: xr.DataArray) -> pd.DataFrame:
    t = pd.DataFrame({c: link_rain[c].values for c in ("site_0_lat", "site_0_lon",
                                                        "site_1_lat", "site_1_lon")},
                     index=pd.Index(np.asarray(link_rain.link.values).astype(str), name="link"))
    t["mid_lat"] = (t.site_0_lat + t.site_1_lat) / 2
    t["mid_lon"] = (t.site_0_lon + t.site_1_lon) / 2
    t["frequency"] = np.asarray(link_rain.frequency.values, dtype=float) \
        if "frequency" in link_rain.coords else np.nan
    t["polarization"] = np.asarray(link_rain.polarization.values).astype(str) \
        if "polarization" in link_rain.coords else ""
    t["length"] = link_lengths_km(link_rain)
    return t


def assemble(link_rain: xr.DataArray, target: xr.DataArray, event_of_step, events: pd.DataFrame,
             splits: pd.Series, link_attenuation: xr.DataArray | None = None,
             gauges: xr.DataArray | None = None, near_km: float = 2.0, step_km: float = 0.05,
             extra_maps: dict | None = None, attrs: dict | None = None) -> xr.Dataset:
    """The learning dataset from aligned pieces.

    ``link_rain(link, time)`` with site coordinates (and ``frequency``, ``polarization``,
    ``length``); ``target(time, lat, lon)`` on the output grid, same times;
    ``event_of_step`` the event id of each time step; ``events`` one row per event
    (``event_id``, ``start``, ``end``, ...); ``splits`` event id -> split.
    ``link_attenuation(link, time)`` in dB (path-integrated); without it the
    attenuation channel is NaN-free zeros and flagged in ``attrs``.
    ``gauges(station, time)`` with ``lat``/``lon``. ``extra_maps`` are stored as more
    ``(time, lat, lon)`` variables (e.g. the radar when the target is a simulated truth).
    """
    target = target.transpose("time", "lat", "lon")
    times = target.time.values
    link_rain = link_rain.transpose("link", "time").sel(time=times)
    grid = Grid(target.lat.values, target.lon.values)
    table = _link_table(link_rain)
    W = path_weights(link_rain, grid, step_km)
    L = table.length.to_numpy()

    r = rasterize(link_rain.values, W, L, grid.shape)
    if link_attenuation is not None:
        A = link_attenuation.transpose("link", "time").sel(time=times, link=link_rain.link).values
        spec = rasterize(A / L[:, None], W, L, grid.shape)["mean"]
    else:
        A, spec = np.full(link_rain.shape, np.nan), np.zeros_like(r["mean"])
    chans = np.stack([np.nan_to_num(r["mean"]), np.nan_to_num(spec), r["coverage"],
                      np.nan_to_num(r["length"])], axis=1).astype("float32")

    dist = distance_to_links(grid, link_rain)
    ev = np.asarray(event_of_step).astype(str)
    split = splits.reindex(ev).to_numpy().astype(str)
    events = events.copy()
    events["event_id"] = events.event_id.astype(str)
    events = events.set_index("event_id").loc[list(dict.fromkeys(ev))]

    ds = xr.Dataset(
        {"inputs": (("time", "channel", "lat", "lon"), chans),
         "target": (("time", "lat", "lon"), target.values.astype("float32")),
         "link_rain": (("link", "time"), link_rain.values.astype("float32")),
         "link_attenuation": (("link", "time"), np.asarray(A, dtype="float32")),
         "distance_km": dist,
         "observable": (("lat", "lon"), (dist.values <= near_km).astype("int8")),
         "event_start": ("event", pd.to_datetime(events.start).values.astype("datetime64[ns]")),
         "event_end": ("event", pd.to_datetime(events.end).values.astype("datetime64[ns]")),
         "event_split": ("event", splits.reindex(events.index).to_numpy().astype(str))},
        coords={"time": times, "channel": list(CHANNELS), "lat": grid.lat, "lon": grid.lon,
                "link": table.index.to_numpy().astype(str), "event": events.index.to_numpy(),
                "event_id": ("time", ev), "split": ("time", split),
                **{c: ("link", table[c].to_numpy()) for c in LINK_COORDS}})
    for c in ("total_mm", "duration_h"):
        if c in events:
            ds[f"event_{c}"] = ("event", events[c].to_numpy())
    if gauges is not None and gauges.sizes.get("station", 0):
        g = gauges.transpose("station", "time").reindex(time=times)
        ds["gauges"] = (("station", "time"), g.values.astype("float32"))
        ds = ds.assign_coords(station=np.asarray(g.station.values).astype(str),
                              station_lat=("station", g.lat.values.astype(float)),
                              station_lon=("station", g.lon.values.astype(float)),
                              station_set=("station", np.asarray(
                                  g["station_set"].values if "station_set" in g.coords
                                  else np.full(g.sizes["station"], "gauges")).astype(str)))
    for name, m in (extra_maps or {}).items():
        ds[name] = (("time", "lat", "lon"), m.transpose("time", "lat", "lon").sel(time=times)
                    .values.astype("float32"))
    units = link_rain.attrs.get("units", "mm")
    ds["inputs"].attrs = {"rain": f"{units} (path-weighted mean of the links in the cell)",
                          "attenuation": "dB/km (specific attenuation, same weighting)",
                          "coverage": "km of valid link path in the cell",
                          "length": "km (mean length of the links in the cell)",
                          "fill": "0 where no valid link crosses the cell"}
    ds["target"].attrs = {"units": target.attrs.get("units", "mm"), "source": target.attrs.get("source", "")}
    ds["link_rain"].attrs = {"units": units}
    ds["link_attenuation"].attrs = {"units": "dB", "available": int(link_attenuation is not None)}
    ds["observable"].attrs = {"near_km": near_km, "meaning": "1 = cell centre within near_km of a link path"}
    ds.attrs = {"split_by": "storm event", "time_label": "end of step", **(attrs or {})}
    return ds


def save_dataset(ds: xr.Dataset, path) -> Path:
    """Write to NetCDF (compressed). Strings are stored as fixed-width text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.copy()
    for name in list(ds.variables):
        if ds[name].dtype == object:
            ds[name] = ds[name].copy(data=np.asarray(ds[name].values).astype(str))
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars
           if ds[v].dtype.kind in "fi" and ds[v].ndim >= 2}
    ds.to_netcdf(path, encoding=enc)
    return path


def load_dataset(path) -> xr.Dataset:
    """Read a dataset written by :func:`save_dataset` into memory."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    for c in ("link", "station", "event", "event_id", "split", "channel", "polarization", "station_set"):
        if c in ds.coords:
            ds = ds.assign_coords({c: ds[c].astype(str)})
    return ds


def arrays(ds: xr.Dataset, split: str | None = None, channels=CHANNELS) -> dict:
    """Numpy views for a model: ``x (n, C, H, W)``, ``y (n, H, W)`` (NaN kept), ``mask (H, W)``."""
    sub = ds if split is None else ds.sel(time=ds.split == split)
    return {"x": sub.inputs.sel(channel=list(channels)).values,
            "y": sub.target.values, "mask": ds.observable.values.astype(bool),
            "time": sub.time.values, "event_id": sub.event_id.values}


def link_rain_of(ds: xr.Dataset) -> xr.DataArray:
    """``link_rain`` with its metadata as coordinates, ready for ``core.maps`` (IDW, GMZ)."""
    return ds.link_rain.reset_coords(drop=True).assign_coords(
        {c: ("link", ds[c].values) for c in LINK_COORDS})


# ---------------------------------------------------------------------------
# real networks
# ---------------------------------------------------------------------------
def metadata_qc_links(network: str, qc=None) -> list:
    """The links of a network that pass ``core.cml.link_qc.metadata_qc`` (static)."""
    from core.cml.link_qc import QCConfig, metadata_qc
    from core.opensense.networks import NETWORKS
    net = NETWORKS[network]
    keep, _ = metadata_qc(net.links_table(), qc or QCConfig(domain=net.domain.pad(0.05)))
    return list(keep)


def event_link_data(network: str, start, end, method: str = "constant", spinup: str = "12h",
                    qc=None, keep=None) -> xr.Dataset:
    """Hour-ending link rain (mm) and mean rain-induced attenuation (dB) over one event.

    Metadata QC on the network's table (or the links ``keep`` that passed it, as
    :func:`metadata_qc_links` gives them: the check is static and slow), the links read
    from ``start - spinup`` (the baseline needs dry signal first), time-series QC over
    the event, then a ``core.cml`` retrieval (:func:`core.cml.estimators.study_estimator`).
    Hours ending in ``(start, end]``. Methods without an attenuation output give NaN
    attenuation; the constant baseline's is negative when dry (its wet-antenna offset),
    which :func:`build_network_dataset` clips at 0.
    """
    from core.cml.estimators import study_estimator
    from core.cml.link_qc import QCConfig, metadata_qc, timeseries_qc
    from core.maps.idw import accumulate
    from core.opensense.networks import NETWORKS

    net = NETWORKS[network]
    cfg = qc or QCConfig(domain=net.domain.pad(0.05))
    if keep is None:
        keep, _ = metadata_qc(net.links_table(), cfg)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    # all links, then the selection: a scattered selection is much slower to read
    links = net.links(t0 - pd.Timedelta(spinup), t1).sel(link=keep)
    keep2, _ = timeseries_qc(links.sel(time=slice(t0, t1)), cfg)
    links = links.sel(link=keep2)
    out = study_estimator(method, t0 - pd.Timedelta(spinup), t1).estimate(links)
    hours = slice(t0 + pd.Timedelta("1h"), t1)
    rain = accumulate(out["rain"].sel(time=slice(t0 + pd.Timedelta("1min"), t1)), "1h").sel(time=hours)
    if "attenuation" in out:
        att = accumulate(out["attenuation"].sel(time=slice(t0 + pd.Timedelta("1min"), t1)), "1h")
        att = att.sel(time=hours)        # the mean over the hour (accumulate x 1 h)
    else:
        att = xr.full_like(rain, np.nan)
    coords = {c: ("link", links[c].values) for c in ("site_0_lat", "site_0_lon", "site_1_lat",
                                                     "site_1_lon", "frequency", "polarization",
                                                     "length")}
    ds = xr.Dataset({"rain": rain.reset_coords(drop=True), "attenuation": att.reset_coords(drop=True)})
    ds = ds.assign_coords(coords)
    ds["rain"].attrs = {"units": "mm", "method": method}
    ds["attenuation"].attrs = {"units": "dB", "meaning": "hourly mean rain-induced attenuation"}
    return ds


def _gauges_of(points: dict) -> xr.DataArray | None:
    parts = []
    for name, g in points.items():
        if g.sizes.get("station", 0):
            g = g.reset_coords([c for c in g.coords if c not in ("station", "time", "lat", "lon")],
                               drop=True)
            parts.append(g.assign_coords(station_set=("station", np.full(g.sizes["station"], name))))
    return xr.concat(parts, "station") if parts else None


def build_network_dataset(network: str = "openmrg", events: pd.DataFrame | None = None,
                          method: str = "constant", fractions=(0.6, 0.2, 0.2), seed: int = 0,
                          how: str = "random", near_km: float = 2.0, spinup: str = "12h",
                          cache_dir=None, min_total_mm: float = 1.0) -> xr.Dataset:
    """Every storm event of a network as one learning dataset (hourly, on ``net.grid``).

    ``events``: rows with ``event_id``, ``start``, ``end``; default every event
    :func:`core.events.detect_events` finds in the radar over the network's record
    (OpenMRG: JJA 2015). The radar is the target; the point gauges of all sets are kept
    for the independent check. ``cache_dir``: each event's link data is written there
    and reused, as is the final dataset (``<network>_<method>.nc``).
    """
    from core.events import detect_events
    from core.opensense.networks import NETWORKS

    net = NETWORKS[network]
    cache = Path(cache_dir) if cache_dir is not None else None
    final = cache / f"{network}_{method}_{how}{seed}.nc" if cache is not None else None
    if final is not None and final.exists():
        return load_dataset(final)
    if events is None:
        events = detect_events(net.radar_hourly(*net.period), min_total_mm=min_total_mm)
    events = events.reset_index(drop=True)
    keep = None

    links, atts, targets, gauges, ev_of_step = [], [], [], [], []
    for ev in events.itertuples():
        path = cache / "events" / f"{network}_{method}_{ev.event_id}.nc" if cache is not None else None
        if path is not None and path.exists():
            with xr.open_dataset(path) as f:
                ld = f.load()
        else:
            log.info("%s event %s: links", network, ev.event_id)
            keep = keep if keep is not None else metadata_qc_links(network)
            ld = event_link_data(network, ev.start, ev.end, method, spinup, keep=keep)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                ld.assign_coords(link=ld.link.astype(str), polarization=ld.polarization.astype(str)).to_netcdf(path)
        radar = net.radar_hourly(ev.start, ev.end)
        t = np.intersect1d(radar.time.values, ld.time.values)
        links.append(ld.rain.sel(time=t))
        atts.append(ld.attenuation.sel(time=t).clip(min=0))      # dry: the wet-antenna offset
        targets.append(radar.sel(time=t))
        g = _gauges_of(net.points_hourly(ev.start, ev.end))
        if g is not None:
            gauges.append(g.reindex(time=t))
        ev_of_step += [str(ev.event_id)] * len(t)

    def union(parts):                        # links (or stations) differ between events
        return xr.concat(parts, "time", join="outer", coords="minimal", compat="override",
                         combine_attrs="override")

    link_rain, link_att = union(links), union(atts)
    meta = pd.concat([pd.DataFrame({c: p[c].values for c in ("site_0_lat", "site_0_lon", "site_1_lat",
                                                              "site_1_lon", "frequency", "polarization",
                                                              "length")},
                                   index=np.asarray(p.link.values).astype(str)) for p in links])
    meta = meta[~meta.index.duplicated()].loc[np.asarray(link_rain.link.values).astype(str)]
    link_rain = link_rain.assign_coords({c: ("link", meta[c].to_numpy()) for c in meta})
    link_rain.attrs = {"units": "mm"}
    target = xr.concat(targets, "time")
    gauge_da = None
    if gauges:
        gauge_da = union(gauges)
        st = pd.concat([pd.DataFrame({"lat": g.lat.values, "lon": g.lon.values,
                                      "station_set": g.station_set.values},
                                     index=np.asarray(g.station.values).astype(str)) for g in gauges])
        st = st[~st.index.duplicated()].loc[np.asarray(gauge_da.station.values).astype(str)]
        gauge_da = gauge_da.assign_coords(lat=("station", st.lat.to_numpy()),
                                          lon=("station", st.lon.to_numpy()),
                                          station_set=("station", st.station_set.to_numpy()))
    splits = split_events(events.event_id.astype(str), fractions, seed, how)
    ds = assemble(link_rain, target, ev_of_step, events, splits, link_attenuation=link_att,
                  gauges=gauge_da, near_km=near_km,
                  attrs={"network": network, "retrieval": method, "split_how": how, "split_seed": seed,
                         "target": "radar, hourly accumulation (mm), hour-ending",
                         "grid_res_deg": net.grid_res})
    if final is not None:
        save_dataset(ds, final)
    return ds


# ---------------------------------------------------------------------------
# simulated truth
# ---------------------------------------------------------------------------
def from_synthetic(cases: list, fractions=(0.6, 0.2, 0.2), seed: int = 0, how: str = "random",
                   near_km: float = 2.0, event_ids=None) -> xr.Dataset:
    """The same dataset from ``core.simulation.scenario.SyntheticCase`` objects, one per event.

    The target is the simulated truth (mm per interval); the radar is kept as ``radar``.
    Every case must use the same grid and its own time span (give each scenario its own
    ``start``). Link ids are prefixed with the event id, as every case has its own network.
    """
    ids = list(event_ids) if event_ids is not None else [f"case{k:02d}" for k in range(len(cases))]
    grid0 = cases[0].geo_grid
    links, targets, radars, gauges, rows, ev = [], [], [], [], [], []
    for eid, c in zip(ids, cases):
        if not (np.allclose(c.geo_grid.lat, grid0.lat) and np.allclose(c.geo_grid.lon, grid0.lon)):
            raise ValueError("every case must be on the same grid")
        lk = c.links.assign_coords(link=[f"{eid}/{x}" for x in c.links.link.values])
        lk = lk.reset_coords("true_mm", drop=True) if "true_mm" in lk.coords else lk
        links.append(lk)
        targets.append(c.truth)
        if c.radar is not None:
            radars.append(c.radar)
        if c.gauges is not None:
            g = c.gauges.reset_coords([k for k in ("x_km", "y_km", "true_mm") if k in c.gauges.coords], drop=True)
            gauges.append(g.assign_coords(station=[f"{eid}/{s}" for s in g.station.values],
                                          station_set=("station", np.full(g.sizes["station"], "gauges"))))
        t = c.truth.time.values
        rows.append({"event_id": eid, "start": pd.Timestamp(t[0]) - pd.Timedelta(minutes=c.interval_min),
                     "end": pd.Timestamp(t[-1]), "total_mm": float(c.truth.mean(("lat", "lon")).sum())})
        ev += [eid] * len(t)
    target = xr.concat(targets, "time")
    if pd.Index(target.time.values).has_duplicates:
        raise ValueError("cases overlap in time: give each scenario its own start")
    link_rain = xr.concat(links, "time", join="outer", coords="minimal", compat="override",
                          combine_attrs="override")
    meta = pd.concat([pd.DataFrame({k: lk[k].values for k in ("site_0_lat", "site_0_lon", "site_1_lat",
                                                               "site_1_lon", "frequency", "polarization",
                                                               "length_km")}, index=lk.link.values)
                      for lk in links])
    link_rain = link_rain.assign_coords({k: ("link", meta.loc[link_rain.link.values, k].to_numpy())
                                         for k in meta})
    gauge_da = xr.concat(gauges, "time", join="outer", coords="minimal", compat="override") if gauges else None
    if gauge_da is not None:
        st = pd.concat([pd.DataFrame({"lat": g.lat.values, "lon": g.lon.values}, index=g.station.values)
                        for g in gauges])
        gauge_da = gauge_da.assign_coords(lat=("station", st.loc[gauge_da.station.values, "lat"].to_numpy()),
                                          lon=("station", st.loc[gauge_da.station.values, "lon"].to_numpy()))
    events = pd.DataFrame(rows)
    splits = split_events(ids, fractions, seed, how)
    extra = {"radar": xr.concat(radars, "time")} if len(radars) == len(cases) else None
    return assemble(link_rain.assign_attrs(units="mm"), target, ev, events, splits, gauges=gauge_da,
                    near_km=near_km, extra_maps=extra,
                    attrs={"network": "simulated", "target": "simulated truth (mm per interval)",
                           "interval_min": float(cases[0].interval_min), "split_how": how,
                           "split_seed": seed})
