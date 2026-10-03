"""One 5-minute data cube per network: radar, links, PWS and their combinations on one grid.

Every field is a rain rate (mm/h) on square 2-km pixels (:func:`core.nowcast.grid.square_grid`),
on 5-minute steps labelled by the end of their interval (UTC):

==========  ===================================================================================
channel     content
==========  ===================================================================================
``R``       radar. OpenMRG: SMHI 5-min reflectivity scans, Z = 200 R^1.5 (file attributes);
            OpenMesh: MRMS ``PrecipRate`` (2 min) averaged over each 5-min interval
``Rv``      1 where the radar has data
``C``       links: 5-min link rain (:mod:`.links`) on virtual gauges along each path (one per
            km), IDW (power 2, within 15 km); NaN farther from every link
``Cs, Ci``  links, sparse: mean link rain over the virtual gauges in each cell; indicator
``P``       PWS (quality-controlled with pypwsqc, ``core.opensense.pws_qc``) by IDW within 15 km
``Ps, Pi``  PWS, sparse: value and indicator in the cells holding a reporting station
``CP``      links (virtual gauges) and PWS in one IDW
``RC``      radar adjusted with the links: ``R + IDW(obs - radar at obs)`` - the additive
``RP``      adjustment of ``core.maps.merge`` (radar averaged along each path for a link,
``RCP``     the containing cell for a station), 12 nearest observations within 15 km;
            beyond 15 km from every observation the radar is kept unchanged
==========  ===================================================================================

Independent references (never an input): Gothenburg's 10 city gauges (1 min -> 5 min; the
SMHI gauge is kept hourly in ``gauges_gh`` but not scored); New York's ASOS stations with
1-minute data (EWR, JFK, LGA; 1 min -> 5 min).

The cube is written once under ``CACHE_DIR/<network>/cube/`` as one ``.npy`` per channel,
read back memory-mapped.
"""

from __future__ import annotations

import json
import logging
import time as _time

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from core.geo import Grid, haversine_m, to_local_xy
from core.maps.gmz import virtual_gauges
from core.maps.idw import idw_weights
from core.nowcast.grid import square_grid
from core.opensense.networks import LABELS, NETWORKS as SOURCES, regrid

from . import links as L
from .settings import (CACHE_DIR, LINE_IDW, MERGE_IDW, NETWORKS, PIXEL_KM, PWS_IDW, STEP_MIN)

log = logging.getLogger(__name__)

CHANNELS = ("R", "Rv", "C", "Cs", "Ci", "P", "Ps", "Pi", "CP", "RC", "RP", "RCP")
CHUNK = 2016                      # steps per chunk when mapping (one week)


def grid_for(network: str) -> Grid:
    return square_grid(NETWORKS[network].domain, PIXEL_KM)


# ------------------------------------------------------------------ geometry helpers


def _xy(grid: Grid, lat, lon):
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    return to_local_xy(np.asarray(lat, float), np.asarray(lon, float), lat0, lon0)


def _cells_xy(grid: Grid):
    glat, glon = grid.mesh()
    return _xy(grid, glat.ravel(), glon.ravel())


def cell_index(grid: Grid, lat, lon) -> np.ndarray:
    """Flat index of the cell holding each point (-1 outside the grid)."""
    dlat, dlon = np.diff(grid.lat[:2])[0], np.diff(grid.lon[:2])[0]
    i = np.round((np.asarray(lat) - grid.lat[0]) / dlat).astype(int)
    j = np.round((np.asarray(lon) - grid.lon[0]) / dlon).astype(int)
    ok = (i >= 0) & (i < grid.lat.size) & (j >= 0) & (j < grid.lon.size)
    return np.where(ok, i * grid.lon.size + j, -1)


def _bin_matrix(cells: np.ndarray, ncell: int) -> np.ndarray:
    B = np.zeros((ncell, cells.size), dtype="float32")
    ok = cells >= 0
    B[cells[ok], np.flatnonzero(ok)] = 1.0
    return B


def _interp(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``W (cells, src) @ V (src, time)`` with NaN sources excluded per step -> (cells, time)."""
    valid = np.isfinite(V)
    num = W @ np.where(valid, V, 0.0)
    den = W @ valid.astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype("float32")


def path_weights(grid: Grid, table: pd.DataFrame, n: int = 21) -> np.ndarray:
    """``(link, cells)``: the radar averaged along each path (``n`` points, nearest cell)."""
    s = np.linspace(0, 1, n)
    A = np.zeros((len(table), grid.lat.size * grid.lon.size), dtype="float32")
    for k, r in enumerate(table.itertuples()):
        c = cell_index(grid, r.site_0_lat + s * (r.site_1_lat - r.site_0_lat),
                       r.site_0_lon + s * (r.site_1_lon - r.site_0_lon))
        c = c[c >= 0]
        if c.size:
            np.add.at(A[k], c, 1.0 / c.size)
    return A


# ------------------------------------------------------------------ radar readers


def _openmrg_radar(times: pd.DatetimeIndex, grid: Grid) -> np.ndarray:
    net = SOURCES["openmrg"]
    out = np.full((times.size,) + grid.shape, np.nan, dtype="float32")
    with xr.open_dataset(net.RADAR) as ds:
        za, zb = float(ds.data.attrs.get("zr_a", 200)), float(ds.data.attrs.get("zr_b", 1.5))
        lat, lon = ds.lat.values, ds.lon.values
        src_t = pd.DatetimeIndex(ds.time.values)
        for k0 in range(0, times.size, CHUNK):
            tt = times[k0:k0 + CHUNK]
            sel = src_t.isin(tt)
            if not sel.any():
                continue
            raw = ds.data.isel(time=np.flatnonzero(sel)).values
            rate = (10.0 ** (raw / 10.0) / za) ** (1.0 / zb)
            data = regrid(rate.reshape(rate.shape[0], -1), lat, lon, grid)
            out[k0 + tt.get_indexer(src_t[sel])] = data
    # cells outside the composite (no source pixel within 1.5 pixels)
    c = np.cos(np.radians(np.mean(grid.lat)))
    tree = cKDTree(np.column_stack([np.ravel(lat), np.ravel(lon) * c]))
    glat, glon = grid.mesh()
    d, _ = tree.query(np.column_stack([glat.ravel(), glon.ravel() * c]))
    out[:, (d * 111.195 > 1.5 * PIXEL_KM).reshape(grid.shape)] = np.nan
    return out


def _openmesh_radar(times: pd.DatetimeIndex, grid: Grid, windows: pd.DataFrame) -> np.ndarray:
    """MRMS PrecipRate, 5-min means on the grid; cells MRMS types as snow (``PrecipFlag``, every
    10 min, the latest at or before each step) are blanked - links and PWS see no snow."""
    from core.radar.mrms import SNOW_FLAGS
    from core.radar.mrms_nyc_wide import load_flag, load_rate
    out = np.full((times.size,) + grid.shape, np.nan, dtype="float32")
    for w in windows.itertuples():
        t0, t1 = pd.Timestamp(w.fetch_start), pd.Timestamp(w.fetch_end)
        try:
            r = load_rate(t0 - pd.Timedelta("5min"), t1)
        except Exception as exc:                        # not cached / not available
            log.warning("MRMS %s: %r", w.event_id, exc)
            continue
        if r is None or r.sizes.get("time", 0) == 0:
            continue
        r = r.where(r >= 0)
        # 2-min rates -> mean over each 5-min interval (t - 5 min, t]
        r5 = r.resample(time=f"{STEP_MIN}min", closed="right", label="right").mean()
        tt = pd.DatetimeIndex(r5.time.values)
        keep = tt.isin(times)
        if not keep.any():
            continue
        lon2, lat2 = np.meshgrid(r5.lon.values, r5.lat.values)
        data = regrid(r5.values[keep].reshape(int(keep.sum()), -1), lat2, lon2, grid)
        try:
            fl = load_flag(t0 - pd.Timedelta("10min"), t1)
            snow = fl.isin(list(SNOW_FLAGS)).astype("float32").where(np.isfinite(fl))
            snow = snow.reindex(time=tt[keep], method="ffill")
            frac = regrid(snow.values.reshape(int(keep.sum()), -1), lat2, lon2, grid)
            data[frac > 0.5] = np.nan
        except Exception as exc:
            log.warning("PrecipFlag %s: %r", w.event_id, exc)
        out[times.get_indexer(tt[keep])] = data
        log.info("MRMS %s: %d steps", w.event_id, int(keep.sum()))
    return out


# ------------------------------------------------------------------ point sets


def _to_5min(rain: xr.DataArray, native: str) -> xr.DataArray:
    """``(station, time)`` mm per native step (interval-ending) -> mm/h per 5-min step."""
    n = int(pd.Timedelta(f"{STEP_MIN}min") / pd.Timedelta(native))
    if n <= 1:
        return rain * (60.0 / STEP_MIN)
    r = rain.resample(time=f"{STEP_MIN}min", closed="right", label="right")
    return r.sum(min_count=1).where(r.count() == n) * (60.0 / STEP_MIN)


def _pws(network: str, start, end) -> xr.DataArray:
    """QC'd PWS as ``(station, time)`` mm/h at 5 min (interval-ending)."""
    from core.opensense import pws_qc
    ds = SOURCES[network].points(pd.Timestamp(start) - pd.Timedelta("1h"), end, sets=["pws"])["pws"]
    rain = ds["rain"]
    step = ds.attrs.get("step", "5min")
    lat, lon = rain.lat.values, rain.lon.values
    x, y = to_local_xy(lat, lon, float(np.mean(lat)), float(np.mean(lon)))
    raw = xr.Dataset({"rainfall": (("id", "time"), rain.values)},
                     coords={"id": rain.station.values, "time": rain.time.values,
                             "lat": ("id", lat), "lon": ("id", lon), "x": ("id", x), "y": ("id", y)})
    qc = pws_qc.flag(raw, step="5min")
    keep = pws_qc.usable(qc, max_missing=0.8, max_flagged=0.2)
    amount = qc.rainfall_qc.sel(id=keep)
    out = xr.DataArray(amount.values * (60.0 / STEP_MIN), dims=("station", "time"),
                       coords={"station": amount.id.values, "time": amount.time.values,
                               "lat": ("station", amount.lat.values), "lon": ("station", amount.lon.values)},
                       attrs={"units": "mm/h", "qc": "pypwsqc FZ/HI/SO + rate check", "native_step": step,
                              "n_stations": len(keep), "n_raw": int(rain.sizes["station"])})
    # pws_qc bins [t, t + 5 min) under the label t. OpenMRG2's 5-min amounts are stamped at
    # their interval's end, exactly on the bin edges, so the label is already the end;
    # OpenMesh's irregular (~5.2 min) reports land within 5 min of it.
    return out


def _gauges(network: str, start, end) -> dict:
    """Independent references: ``{"g5": (station, time) mm/h at 5 min, "gh": hourly mm}``."""
    if network == "openmrg":
        pts = SOURCES["openmrg"].points(pd.Timestamp(start) - pd.Timedelta("1h"), end, sets=["city", "smhi"])
        g5 = _to_5min(pts["city"]["rain"], "1min")
        hourly = []
        for name, native in (("city", "1min"), ("smhi", "15min")):
            r = pts[name]["rain"].resample(time="1h", closed="right", label="right")
            per_h = int(pd.Timedelta("1h") / pd.Timedelta(native))
            hourly.append(r.sum(min_count=1).where(r.count() == per_h))
        gh = xr.concat(hourly, "station")
        return {"g5": g5, "gh": gh}
    from core.radar.mrms_nyc_wide import event_windows, load_asos_1min
    w = event_windows()
    w = w[(w.fetch_end >= pd.Timestamp(start)) & (w.fetch_start <= pd.Timestamp(end))]
    parts = []
    for r in w.itertuples():                  # cached per event window by core.radar.mrms_nyc_wide
        try:
            parts.append(load_asos_1min(r.fetch_start, r.fetch_end))
        except Exception as exc:
            log.warning("ASOS %s: %r", r.event_id, exc)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    coords = _asos_coords()
    cols = [c for c in df.columns if c in coords]
    da = xr.DataArray(df[cols].values.T.astype("float32"), dims=("station", "time"),
                      coords={"station": cols, "time": df.index.values,
                              "lat": ("station", [coords[c][0] for c in cols]),
                              "lon": ("station", [coords[c][1] for c in cols])})
    # IEM 1-min stamps name the minute's end
    full = pd.DatetimeIndex(np.unique(np.concatenate(
        [pd.date_range(r.fetch_start - pd.Timedelta("1h"), r.fetch_end, freq="1min").values for r in w.itertuples()])))
    da = da.reindex(time=full)
    g5 = _to_5min(da, "1min")
    r = da.resample(time="1h", closed="right", label="right")
    gh = r.sum(min_count=1).where(r.count() >= 54)
    return {"g5": g5, "gh": gh}


def _asos_coords() -> dict:
    """NWS coordinates of the four ASOS stations (Central Park, LaGuardia, JFK, Newark)."""
    return {"NYC": (40.7789, -73.9692), "LGA": (40.7794, -73.8803),
            "JFK": (40.6386, -73.7622), "EWR": (40.6925, -74.1687)}


# ------------------------------------------------------------------ the cube


class Cube:
    """A built cube, memory-mapped: ``cube["R"]`` is ``(time, y, x)`` float32."""

    def __init__(self, network: str):
        self.network = network
        self.dir = CACHE_DIR / network / "cube"
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.times = pd.DatetimeIndex(np.load(self.dir / "times.npy"))
        self.grid = Grid(np.load(self.dir / "lat.npy"), np.load(self.dir / "lon.npy"))
        self.split = np.load(self.dir / "split.npy")
        self._arr = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._arr:
            self._arr[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._arr[name]

    @property
    def gauges(self) -> dict:
        return {k: xr.open_dataarray(self.dir / f"gauges_{k}.nc").load() for k in ("g5", "gh")}

    @property
    def area(self) -> np.ndarray:
        """Cells within 10 km of a link or PWS: where the sensors can know the rain."""
        return np.load(self.dir / "area.npy")

    @property
    def contiguous(self) -> np.ndarray:
        """``ok[i]`` True when step ``i`` follows step ``i - 1`` by exactly 5 minutes."""
        d = np.diff(self.times.values).astype("timedelta64[m]").astype(int)
        return np.concatenate([[False], d == STEP_MIN])

    @classmethod
    def exists(cls, network: str) -> bool:
        return (CACHE_DIR / network / "cube" / "meta.json").exists()


def _times(network: str) -> tuple[pd.DatetimeIndex, pd.DataFrame | None]:
    s = NETWORKS[network]
    if network == "openmrg":
        return pd.date_range(s.period[0], s.period[1], freq=f"{STEP_MIN}min"), None
    from core.radar.mrms_nyc_wide import event_windows
    w = event_windows()
    w = w[(w.fetch_start >= pd.Timestamp(s.period[0])) & (w.fetch_end <= pd.Timestamp(s.period[1]))]
    parts = [pd.date_range(r.fetch_start.ceil(f"{STEP_MIN}min"), r.fetch_end, freq=f"{STEP_MIN}min")
             for r in w.itertuples()]
    return pd.DatetimeIndex(np.unique(np.concatenate([p.values for p in parts]))), w.reset_index(drop=True)


def build(network: str, refresh: bool = False) -> Cube:
    if Cube.exists(network) and not refresh:
        return Cube(network)
    t_start = _time.time()
    out = CACHE_DIR / network / "cube"
    out.mkdir(parents=True, exist_ok=True)
    grid = grid_for(network)
    ncell = grid.lat.size * grid.lon.size
    times, windows = _times(network)
    log.info("%s: %d steps on a %s grid", network, times.size, grid.shape)

    # radar
    R = _openmrg_radar(times, grid) if network == "openmrg" else _openmesh_radar(times, grid, windows)
    log.info("radar done (%.0f s)", _time.time() - t_start)

    # links
    lr = L.link_rain(network)
    lr = lr.sel(time=lr.time.isin(times))
    table = pd.DataFrame({c: lr[c].values for c in ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
                                                     "mid_lat", "mid_lon"]}, index=lr.link.values)
    inside = (cell_index(grid, table.mid_lat, table.mid_lon) >= 0)
    table, lr = table[inside], lr.isel(link=np.flatnonzero(inside))
    link_v = lr.rnn.reindex(time=times).transpose("link", "time").values.astype("float32")
    vlat, vlon, owner = virtual_gauges(lr, per_km=LINE_IDW["per_km"])
    vg_v = link_v[owner]                                             # (vg, time)

    # PWS
    pws = _pws(network, times[0], times[-1])
    pws = pws.isel(station=np.flatnonzero(cell_index(grid, pws.lat.values, pws.lon.values) >= 0))
    pws_v = pws.reindex(time=times).transpose("station", "time").values.astype("float32")

    # weights
    cx, cy = _cells_xy(grid)
    vx, vy = _xy(grid, vlat, vlon)
    px, py = _xy(grid, pws.lat.values, pws.lon.values)
    W_c = idw_weights(vx, vy, cx, cy, LINE_IDW["power"], LINE_IDW["radius_m"]).astype("float32")
    W_p = idw_weights(px, py, cx, cy, PWS_IDW["power"], PWS_IDW["radius_m"]).astype("float32")
    W_cp = np.concatenate([W_c, W_p], axis=1)
    B_c = _bin_matrix(cell_index(grid, vlat, vlon), ncell)
    B_p = _bin_matrix(cell_index(grid, pws.lat.values, pws.lon.values), ncell)
    # merging: observations at link midpoints (radar along the path) and stations (their cell)
    mx, my = _xy(grid, table.mid_lat.values, table.mid_lon.values)
    A_link = path_weights(grid, table)
    A_pws = _bin_matrix(cell_index(grid, pws.lat.values, pws.lon.values), ncell).T
    W_mc = idw_weights(mx, my, cx, cy, MERGE_IDW["power"], MERGE_IDW["radius_m"], MERGE_IDW["nnear"]).astype("float32")
    W_mp = idw_weights(px, py, cx, cy, MERGE_IDW["power"], MERGE_IDW["radius_m"], MERGE_IDW["nnear"]).astype("float32")
    W_mcp = idw_weights(np.r_[mx, px], np.r_[my, py], cx, cy, MERGE_IDW["power"], MERGE_IDW["radius_m"],
                        MERGE_IDW["nnear"]).astype("float32")

    arrays = {c: np.lib.format.open_memmap(out / f"{c}.npy", mode="w+", dtype="float32",
                                           shape=(times.size,) + grid.shape) for c in CHANNELS}
    for k0 in range(0, times.size, CHUNK):
        sl = slice(k0, min(times.size, k0 + CHUNK))
        n = sl.stop - sl.start
        r = R[sl].reshape(n, -1).T                                      # (cells, time)
        cv, pv = vg_v[:, sl], pws_v[:, sl]
        fields = {
            "R": r, "Rv": np.isfinite(r).astype("float32"),
            "C": _interp(W_c, cv), "Cs": _interp(B_c, cv), "Ci": (B_c @ np.isfinite(cv)) > 0,
            "P": _interp(W_p, pv), "Ps": _interp(B_p, pv), "Pi": (B_p @ np.isfinite(pv)) > 0,
            "CP": _interp(W_cp, np.concatenate([cv, pv])),
        }
        r0 = np.nan_to_num(r)
        obs_c, obs_p = link_v[:, sl], pv
        res_c = obs_c - A_link @ r0
        res_p = obs_p - A_pws @ r0
        for name, W, res in (("RC", W_mc, res_c), ("RP", W_mp, res_p),
                             ("RCP", W_mcp, np.concatenate([res_c, res_p]))):
            corr = _interp(W, res)
            fields[name] = np.where(np.isfinite(corr), np.clip(r + corr, 0, None), r)
        for c in CHANNELS:
            arrays[c][sl] = np.asarray(fields[c], dtype="float32").T.reshape((n,) + grid.shape)
        log.info("%s: mapped steps %d-%d (%.0f s)", network, sl.start, sl.stop, _time.time() - t_start)
    for a in arrays.values():
        a.flush()

    # sensor area: within 10 km of a link (virtual gauge) or a PWS
    d = cKDTree(np.c_[np.r_[vx, px], np.r_[vy, py]]).query(np.c_[cx, cy])[0]
    np.save(out / "area.npy", (d <= 10_000).reshape(grid.shape))
    np.save(out / "times.npy", times.values)
    np.save(out / "lat.npy", grid.lat)
    np.save(out / "lon.npy", grid.lon)
    np.save(out / "split.npy", L.split(times, network))
    g = _gauges(network, times[0], times[-1])
    for k, da in (("g5", g["g5"].reindex(time=times)), ("gh", g["gh"])):
        da = da.assign_coords(station=np.asarray(da.station.values).astype(str))
        da.astype("float32").rename(k).to_netcdf(out / f"gauges_{k}.nc")
    meta = {"network": network, "grid_shape": list(grid.shape), "pixel_km": PIXEL_KM, "n_steps": int(times.size),
            "n_links": int(link_v.shape[0]), "n_virtual_gauges": int(vlat.size), "n_pws": int(pws_v.shape[0]),
            "n_pws_raw": int(pws.attrs.get("n_raw", -1)), "retrieval": lr.attrs.get("retrieval"),
            "radar_valid_share": float(np.isfinite(R).mean()), "build_seconds": round(_time.time() - t_start)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("%s cube built: %s", network, meta)
    return Cube(network)
