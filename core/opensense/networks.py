"""
Three city networks on one footing: links, point gauges and radar.

Each :class:`Network` gives the same three things for any window of its record:

``links(start, end)``         a *link set*: ``rsl(link, time)`` at 1 min, ``tsl`` where it
                              is recorded, and per-link ``frequency`` (GHz), ``length`` (km),
                              ``polarization`` ("v"/"h"), site coordinates, ``mid_lat/lon``
                              - the input of every method in ``core.cml``
``points(start, end)``        ``{name: Dataset}`` of point gauges, ``rain(station, time)`` in
                              mm per native step, with ``lat``/``lon`` and the step and
                              time-label convention in ``attrs``
``radar_hourly(start, end)``  radar accumulations (mm, hour-ending, UTC) on a regular
                              lat/lon :class:`~core.geo.Grid` over the network

| network | links | point gauges | radar |
|---|---|---|---|
| ``openmrg`` (Gothenburg, JJA 2015) | 728 sublinks, TSL+RSL at 10 s -> 1 min | 30 Netatmo PWS (5 min), 10 city gauges (1 min), 1 SMHI (15 min) | SMHI C-band composite, 5 min, 2 km |
| ``openrainer`` (Emilia-Romagna, 2021-22) | 151 links x 2 channels, TSL+RSL, 1 min | 319 ARPAE gauges (15 min) | ARPAE composite, 15-min totals |
| ``openmesh`` (New York City, Oct 2023 - Jun 2024) | 103 sublinks, RSL only, 1 min | 37 WU PWS (5 min), 4 ASOS (hourly) | MRMS Pass 2, hourly, 0.01 deg |

Hourly radar grids are cached per month under ``dataset/open_datasets/<folder>/processed/``.
Time labels: every accumulation here is converted to hour-ENDING (the hour labelled 12:00
is 11:00-12:00 UTC). The per-source conventions are in :data:`LABELS`; they were
established by lagging each source against the links (see ``projects/multisensor_maps``).

    from core.opensense.networks import NETWORKS
    net = NETWORKS["openrainer"]
    links = net.links("2021-09-26", "2021-09-27")
    radar = net.radar_hourly("2021-09-26", "2021-09-27")
    gauges = net.points_hourly("2021-09-26", "2021-09-27")
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import NYC, OPENMESH, Domain, Grid, haversine_m
from core import data_paths as dp
from core.opensense.fetch import DATA_ROOT

# Which edge of its interval each accumulation's time stamp names, from lagging each
# source against the 1-minute link signal (network means, one wet week or more):
# decisive for SMHI's 15-min gauge (end: r 0.69 vs 0.57 as start) and OpenMesh's PWS
# (end, and reports ~5 min late: r 0.95 one step later vs 0.91); flat within 0.01 over
# +-1 step for the others, where it cannot matter at hourly resolution. OpenRainER's
# "end" was established on a convective event by opensense_pipeline/ingest_openrainer.
LABELS = {
    ("openmrg", "radar"): "end",
    ("openmrg", "pws"): "end",
    ("openmrg", "city"): "end",
    ("openmrg", "smhi"): "end",
    ("openrainer", "radar"): "end",
    ("openrainer", "gauges"): "end",
    ("openmesh", "pws"): "end",
}


# ---------------------------------------------------------------------------
# helpers shared by the networks
# ---------------------------------------------------------------------------
def to_hourly(da: xr.DataArray, label: str = "end", min_coverage: float = 0.8) -> xr.DataArray:
    """Per-step accumulations (mm) -> hour-ending totals (mm).

    ``label`` says whether a sample's stamp is the end or the start of its interval.
    Hours with fewer than ``min_coverage`` of their samples are NaN, never zero.
    """
    t = pd.DatetimeIndex(da.time.values)
    step = pd.Series(t).diff().median()
    per_hour = pd.Timedelta("1h") / step
    closed = "right" if label == "end" else "left"
    r = da.resample(time="1h", closed=closed, label="right")
    out = r.sum(min_count=1).where(r.count() >= min_coverage * per_hour)
    out.attrs = dict(da.attrs, units="mm", time_label="end of hour (UTC)")
    return out


def regrid(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, grid: Grid) -> np.ndarray:
    """Source pixels (time, npix) with centres ``lat``/``lon`` (npix) -> ``grid`` (time, nlat, nlon).

    Each target cell is the mean of the source pixels whose centres fall in it; cells
    with none take the nearest source pixel (the target is finer than the source).
    NaN source values are skipped.
    """
    lat, lon = np.ravel(lat), np.ravel(lon)
    res_lat = float(np.diff(grid.lat[:2])[0])
    res_lon = float(np.diff(grid.lon[:2])[0])
    i = np.round((lat - grid.lat[0]) / res_lat).astype(int)
    j = np.round((lon - grid.lon[0]) / res_lon).astype(int)
    inside = (i >= 0) & (i < grid.lat.size) & (j >= 0) & (j < grid.lon.size)
    cell = np.where(inside, i * grid.lon.size + j, -1)
    ncell = grid.lat.size * grid.lon.size
    V = np.asarray(values, dtype="float64").reshape(values.shape[0], -1)
    ok = np.isfinite(V)
    total = np.zeros((V.shape[0], ncell))
    count = np.zeros((V.shape[0], ncell))
    sel = cell >= 0
    np.add.at(total.T, cell[sel], np.where(ok, V, 0.0)[:, sel].T)
    np.add.at(count.T, cell[sel], ok[:, sel].T.astype(float))
    with np.errstate(invalid="ignore"):
        out = total / count
    # empty cells: nearest source pixel
    empty = np.flatnonzero(count.max(axis=0) == 0)
    if empty.size:
        from scipy.spatial import cKDTree
        glat, glon = grid.mesh()
        c = np.cos(np.radians(np.mean(grid.lat)))
        tree = cKDTree(np.column_stack([lat, lon * c]))
        _, nearest = tree.query(np.column_stack([glat.ravel()[empty], glon.ravel()[empty] * c]))
        out[:, empty] = V[:, nearest]
    return out.reshape(V.shape[0], grid.lat.size, grid.lon.size).astype("float32")


def _grid_da(data, times, grid: Grid, name: str, **attrs) -> xr.DataArray:
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": pd.DatetimeIndex(times), "lat": grid.lat, "lon": grid.lon},
                        name=name, attrs=attrs)


def _link_set(rsl: np.ndarray, times, meta: pd.DataFrame, tsl: np.ndarray | None = None,
              source: str = "") -> xr.Dataset:
    """Assemble a link set from (link, time) arrays and a metadata frame indexed by label."""
    coords = {"link": np.asarray(meta.index, dtype=object), "time": pd.DatetimeIndex(times)}
    for col in ["cml_id", "sublink_id", "frequency", "length", "polarization", "site_0_lat",
                "site_0_lon", "site_1_lat", "site_1_lon"]:
        coords[col] = ("link", meta[col].to_numpy())
    coords["mid_lat"] = ("link", ((meta.site_0_lat + meta.site_1_lat) / 2).to_numpy())
    coords["mid_lon"] = ("link", ((meta.site_0_lon + meta.site_1_lon) / 2).to_numpy())
    data = {"rsl": (("link", "time"), np.asarray(rsl, dtype="float32"))}
    if tsl is not None:
        data["tsl"] = (("link", "time"), np.asarray(tsl, dtype="float32"))
    out = xr.Dataset(data, coords=coords)
    out["rsl"].attrs = {"units": "dBm"}
    out["frequency"].attrs["units"] = "GHz"
    out["length"].attrs["units"] = "km"
    out.attrs = {"source": source}
    return out


def _month_starts(start, end) -> list[pd.Timestamp]:
    s, e = pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M")
    return [p.to_timestamp() for p in pd.period_range(s, e, freq="M")]


# ---------------------------------------------------------------------------
@dataclass
class Network:
    """Common interface; see the module docstring."""

    key: str
    name: str
    domain: Domain
    period: tuple
    grid_res: float
    folder: str
    radar_name: str = "radar"
    point_names: tuple = ()
    _table: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def grid(self) -> Grid:
        return Grid.from_domain(self.domain, self.grid_res)

    @property
    def processed(self) -> Path:
        p = DATA_ROOT / self.folder / "processed" / "networks"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def links_table(self) -> pd.DataFrame:
        raise NotImplementedError

    def links(self, start, end, labels=None) -> xr.Dataset:
        raise NotImplementedError

    def points(self, start, end, sets=None) -> dict:
        raise NotImplementedError

    def points_hourly(self, start, end, min_coverage: float = 0.8, sets=None) -> dict:
        """``{name: rain(station, time)}`` hour-ending mm, hours ending in ``(start, end]``.

        ``sets`` restricts which point sets are read (default: all)."""
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        out = {}
        for name, ds in self.points(t0 - pd.Timedelta("1h"), t1, sets).items():
            h = ds["rain"] if ds.attrs.get("step") == "1h" else to_hourly(
                ds["rain"], ds.attrs.get("label", "end"), min_coverage)
            h = h.sel(time=slice(t0 + pd.Timedelta("1h"), t1))
            out[name] = h.assign_attrs(units="mm", source=name)
        return out

    def radar_hourly(self, start, end) -> xr.DataArray:
        """Hour-ending radar totals (mm) on :attr:`grid`, hours ending in ``(start, end]``."""
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        # the hour ending at midnight on the 1st belongs to the previous month's cache
        parts = [self._radar_month(m) for m in _month_starts(t0 + pd.Timedelta("1h"),
                                                              t1 - pd.Timedelta("1h"))]
        da = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
        return da.sel(time=slice(t0 + pd.Timedelta("1h"), t1))

    def _radar_month(self, month: pd.Timestamp) -> xr.DataArray:
        path = self.processed / f"radar_hourly_{month:%Y%m}_{self.domain.key}_{self.grid_res:g}.nc"
        if not path.exists():
            da = self._build_radar_month(month)
            da.to_netcdf(path, encoding={da.name: {"zlib": True, "complevel": 4}})
        with xr.open_dataarray(path) as da:
            return da.load()

    def _build_radar_month(self, month: pd.Timestamp) -> xr.DataArray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenMRG - Gothenburg, JJA 2015
# ---------------------------------------------------------------------------
class OpenMRG(Network):
    # inputs, from ~/data/cml (or data/interim): see core/data_paths.py
    CML = dp.data_path(dp.OPENMRG_CML)
    CML_META = dp.data_path(dp.OPENMRG_CML_META)
    RADAR = dp.data_path(dp.OPENMRG_RADAR)
    PWS = dp.data_path(dp.OPENMRG2)

    def links_table(self) -> pd.DataFrame:
        if self._table is None:
            m = pd.read_csv(self.CML_META)
            t = pd.DataFrame({
                "cml_id": m.Link.astype(int), "sublink_id": m.Direction.astype(str),
                "sublink": m.Sublink.astype(int), "frequency": m.Frequency_GHz.astype(float),
                "length": m.Length_km.astype(float),
                "polarization": np.where(m.Polarization.str.lower().str.startswith("v"), "v", "h"),
                "site_0_lat": m.NearLatitude_DecDeg, "site_0_lon": m.NearLongitude_DecDeg,
                "site_1_lat": m.FarLatitude_DecDeg, "site_1_lon": m.FarLongitude_DecDeg})
            t.index = [f"{c}/{d}" for c, d in zip(t.cml_id, t.sublink_id)]
            t.index.name = "link"
            self._table = t.sort_index()
        return self._table

    def links(self, start, end, labels=None) -> xr.Dataset:
        """10-s TSL/RSL averaged to 1 min (a minute is labelled by its start)."""
        table = self.links_table()
        labels = list(table.index) if labels is None else list(labels)
        meta = table.loc[labels]
        t0 = pd.Timestamp(start).floor("min")
        t1 = pd.Timestamp(end).floor("min") + pd.Timedelta("59s")
        with xr.open_dataset(self.CML) as raw:
            sub = raw.sel(time=slice(t0, t1), sublink=meta.sublink.to_numpy()).load()
        minute = sub.resample(time="1min", closed="left", label="left").mean()
        # (time, sublink) -> (link, time)
        return _link_set(minute.rsl.to_numpy().T, minute.time.values, meta,
                         tsl=minute.tsl.to_numpy().T, source="OpenMRG (zenodo 7107689)")

    def points(self, start, end, sets=None) -> dict:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        out = {}
        for name, file, var, step in [("pws", "OpenMRGplus_rain.nc", "rainfall", "5min"),
                                      ("city", "city_gauges.nc", "rainfall_amount", "1min"),
                                      ("smhi", "smhi_gauges.nc", "rainfall_amount", "15min")]:
            if sets is not None and name not in sets:
                continue
            with xr.open_dataset(self.PWS / file) as ds:
                ds = ds.sel(time=slice(t0, t1)).load()
            lat = ds["latitude" if "latitude" in ds.coords else "lat"].to_numpy()
            lon = ds["longitude" if "longitude" in ds.coords else "lon"].to_numpy()
            rain = ds[var].transpose("id", "time").to_numpy().astype("float32")
            rain = np.where(rain < 0, np.nan, rain)
            out[name] = xr.Dataset(
                {"rain": (("station", "time"), rain)},
                coords={"station": np.asarray([f"{name}_{s}" for s in ds.id.values], dtype=object),
                        "time": ds.time.values, "lat": ("station", lat), "lon": ("station", lon)},
                attrs={"step": step, "label": LABELS[("openmrg", name)], "units": "mm"})
        return out

    def _build_radar_month(self, month: pd.Timestamp) -> xr.DataArray:
        end = month + pd.offsets.MonthBegin(1)
        with xr.open_dataset(self.RADAR) as ds:
            # 5-min scans ending in (month - 1h, end]
            ds = ds.sel(time=slice(month - pd.Timedelta("55min"), end)).load()
            za, zb = float(ds.data.attrs.get("zr_a", 200)), float(ds.data.attrs.get("zr_b", 1.5))
            rate = (10.0 ** (ds.data / 10.0) / za) ** (1.0 / zb)          # mm/h
            lat, lon = ds.lat.to_numpy(), ds.lon.to_numpy()
        amount = rate / 12.0                                               # mm per 5 min
        hourly = to_hourly(amount, LABELS[("openmrg", "radar")])
        hourly = hourly.sel(time=slice(month + pd.Timedelta("1h"), end))
        values = hourly.to_numpy().reshape(hourly.sizes["time"], -1)
        data = regrid(values, lat, lon, self.grid)
        return _grid_da(data, hourly.time.values, self.grid, "radar", units="mm",
                        source="SMHI radar composite, Z = 200 R^1.5 (file attributes)",
                        time_label="end of hour (UTC)")


# ---------------------------------------------------------------------------
# OpenRainER - Emilia-Romagna, 2021-2022
# ---------------------------------------------------------------------------
class OpenRainER(Network):
    # inputs, from ~/data/cml (or data/interim): see core/data_paths.py
    RAW = dp.data_path(dp.OPENRAINER_DOWNLOAD)

    def _file(self, prefix: str, month: pd.Timestamp, archive: str) -> Path:
        """Extract ``<prefix>_<YYYYMM>*.nc`` from its tar if needed; return the .nc path.

        Monthly files live one folder per product (``dp.OPENRAINER_MONTHLY``); a
        missing month is unpacked from the tar into that same shared folder.
        """
        tag = f"{prefix}_{month:%Y%m}"
        folder = dp.data_path(dp.OPENRAINER_MONTHLY[prefix])
        found = sorted(p for p in folder.glob(f"{tag}*.nc"))
        if found:
            return found[0]
        folder.mkdir(parents=True, exist_ok=True)
        listing = subprocess.run(["tar", "-tf", str(self.RAW / archive)], capture_output=True,
                                 text=True, check=True).stdout.split()
        member = next(m for m in listing if Path(m).name.startswith(tag))
        subprocess.run(["tar", "-xf", str(self.RAW / archive), "-C", str(folder), member],
                       check=True)
        gz = folder / member
        target = folder / Path(member).name.removesuffix(".gz")
        with gzip.open(gz, "rb") as fin, target.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz.unlink()
        return target

    def links_table(self) -> pd.DataFrame:
        if self._table is None:
            from core.opensense import conventions as cv
            with xr.open_dataset(self._file("CML", pd.Timestamp(self.period[0]), "CML.tar")) as ds:
                rows = []
                length = cv.to_km(ds.length)
                freq = cv.to_ghz(ds.frequency)
                pol = cv.normalize_polarization(ds.polarization.values.ravel()).reshape(ds.polarization.shape)
                for i, c in enumerate(ds.cml_id.values):
                    for j, s in enumerate(ds.sublink_id.values):
                        rows.append({"cml_id": str(c), "sublink_id": str(s),
                                     "frequency": float(np.asarray(freq)[i, j]),
                                     "length": float(np.asarray(length)[i]),
                                     "polarization": "v" if str(pol[i, j]).startswith("v") else "h",
                                     "site_0_lat": float(ds.site_0_lat[i]), "site_0_lon": float(ds.site_0_lon[i]),
                                     "site_1_lat": float(ds.site_1_lat[i]), "site_1_lon": float(ds.site_1_lon[i])})
            t = pd.DataFrame(rows)
            t.index = [f"{c}/{s}" for c, s in zip(t.cml_id, t.sublink_id)]
            t.index.name = "link"
            self._table = t
        return self._table

    def links(self, start, end, labels=None) -> xr.Dataset:
        table = self.links_table()
        labels = list(table.index) if labels is None else list(labels)
        meta = table.loc[labels]
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        full = pd.date_range(t0.floor("min"), t1.floor("min"), freq="1min")
        rsl = np.full((len(labels), full.size), np.nan, "float32")
        tsl = np.full_like(rsl, np.nan)
        pos = {lab: k for k, lab in enumerate(labels)}
        for month in _month_starts(t0, t1):
            with xr.open_dataset(self._file("CML", month, "CML.tar")) as ds:
                ds = ds.sel(time=slice(t0, t1)).load()
            idx = full.get_indexer(pd.DatetimeIndex(ds.time.values))
            ok = idx >= 0
            for i, c in enumerate(ds.cml_id.values):
                for j, s in enumerate(ds.sublink_id.values):
                    k = pos.get(f"{c}/{s}")
                    if k is None:
                        continue
                    rsl[k, idx[ok]] = ds.rsl.values[i, j][ok]
                    tsl[k, idx[ok]] = ds.tsl.values[i, j][ok]
        return _link_set(rsl, full, meta, tsl=tsl, source="OpenRainER (zenodo 22829808)")

    def points(self, start, end, sets=None) -> dict:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        parts = []
        for month in _month_starts(t0, t1):
            with xr.open_dataset(self._file("AWS", month, "AWS.tar")) as ds:
                parts.append(ds[["rainfall_amount"]].sel(time=slice(t0, t1)).load())
        ds = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
        rain = ds.rainfall_amount.transpose("id", "time").to_numpy().astype("float32")
        rain = np.where(rain < 0, np.nan, rain)
        g = xr.Dataset({"rain": (("station", "time"), rain)},
                       coords={"station": np.asarray(ds.id.values, dtype=object), "time": ds.time.values,
                               "lat": ("station", ds.latitude.to_numpy()),
                               "lon": ("station", ds.longitude.to_numpy())},
                       attrs={"step": "15min", "label": LABELS[("openrainer", "gauges")], "units": "mm"})
        inside = self.domain.pad(0.1).contains(g.lat.values, g.lon.values)
        return {"gauges": g.isel(station=np.flatnonzero(inside))}

    def _build_radar_month(self, month: pd.Timestamp) -> xr.DataArray:
        path = self._file("RADrain", month, "RADrain.tar")
        d = self.domain.pad(0.05)
        with xr.open_dataset(path) as ds:
            lat_desc = ds.lat.values[0] > ds.lat.values[-1]
            ds = ds.sel(lon=slice(d.lon_min, d.lon_max),
                        lat=slice(d.lat_max, d.lat_min) if lat_desc else slice(d.lat_min, d.lat_max))
            amount = ds.rainfall_amount.load()                               # mm per 15 min
        amount = amount.where(amount >= 0)
        hourly = to_hourly(amount, LABELS[("openrainer", "radar")])
        end = month + pd.offsets.MonthBegin(1)
        hourly = hourly.sel(time=slice(month + pd.Timedelta("1h"), end))
        lon2, lat2 = np.meshgrid(hourly.lon.values, hourly.lat.values)
        data = regrid(hourly.to_numpy().reshape(hourly.sizes["time"], -1), lat2, lon2, self.grid)
        return _grid_da(data, hourly.time.values, self.grid, "radar", units="mm",
                        source="ARPAE-SIMC radar 15-min rain depth", time_label="end of hour (UTC)")


# ---------------------------------------------------------------------------
# OpenMesh - New York City, 2023-24
# ---------------------------------------------------------------------------
class OpenMesh(Network):
    def links_table(self) -> pd.DataFrame:
        if self._table is None:
            from core.opensense import openmesh as om
            self._table = om.sublinks_table()
        return self._table

    def links(self, start, end, labels=None) -> xr.Dataset:
        from core.opensense import openmesh as om
        return om.load_links(start, end, labels)

    def points(self, start, end, sets=None) -> dict:
        from core.asos import fetch_asos, hourly_precip
        from core.opensense import openmesh as om
        pws = om.load_pws(start, end)
        pws = pws.assign_attrs(step="5min", label=LABELS[("openmesh", "pws")], units="mm")
        pws = pws.assign_coords(station=np.asarray([f"pws_{s}" for s in pws.station.values], dtype=object))
        out = {"pws": pws}
        if sets is not None and "asos" not in sets:
            return out
        try:
            a = fetch_asos(pd.Timestamp(start) - pd.Timedelta("1h"), pd.Timestamp(end) + pd.Timedelta("1h"))
            h = hourly_precip(a).loc[pd.Timestamp(start):pd.Timestamp(end)]
            st = a.groupby("station")[["lat", "lon"]].first().loc[h.columns]
            out["asos"] = xr.Dataset(
                {"rain": (("station", "time"), h.to_numpy().T.astype("float32"))},
                coords={"station": np.asarray([f"asos_{s}" for s in h.columns], dtype=object),
                        "time": h.index.values.astype("datetime64[ns]"),
                        "lat": ("station", st.lat.to_numpy()), "lon": ("station", st.lon.to_numpy())},
                attrs={"step": "1h", "label": "end", "units": "mm"})
        except Exception:                        # IEM unreachable: PWS alone
            pass
        return out

    def _build_radar_month(self, month: pd.Timestamp) -> xr.DataArray:
        from core.radar.mrms import hourly_rainfall
        end = month + pd.offsets.MonthBegin(1)
        start = max(month, pd.Timestamp(self.period[0]))
        stop = min(end, pd.Timestamp(self.period[1]))
        qpe = hourly_rainfall(start, stop, NYC)
        g = self.grid
        data = qpe.reindex(lat=g.lat, lon=g.lon, method="nearest", tolerance=self.grid_res / 2)
        return _grid_da(data.to_numpy().astype("float32"), qpe.time.values, g, "radar", units="mm",
                        source="MRMS MultiSensor QPE 01H Pass 2", time_label="end of hour (UTC)")


NETWORKS = {
    "openmrg": OpenMRG("openmrg", "OpenMRG (Gothenburg)", Domain(57.30, 58.00, 11.55, 12.55, "openmrg"),
                       ("2015-06-01", "2015-09-01"), 0.02, "OpenMRG_Sweden",
                       point_names=("pws", "city", "smhi")),
    "openrainer": OpenRainER("openrainer", "OpenRainER (Emilia-Romagna)",
                             Domain(43.75, 45.00, 9.25, 12.65, "openrainer"),
                             ("2021-01-01", "2023-01-01"), 0.02, "OpenRainER_Italy",
                             point_names=("gauges",)),
    "openmesh": OpenMesh("openmesh", "OpenMesh (New York City)", OPENMESH,
                         ("2023-10-29", "2024-07-01"), 0.01, "OpenMesh_NYC",
                         point_names=("pws", "asos")),
}


# ---------------------------------------------------------------------------
# references along links
# ---------------------------------------------------------------------------
def radar_along_links(radar: xr.DataArray, links: pd.DataFrame | xr.Dataset) -> xr.DataArray:
    """Radar averaged along each link path: ``(link, time)``, same units as ``radar``."""
    from core.radar.mrms import path_average
    table = links if isinstance(links, pd.DataFrame) else pd.DataFrame(
        {c: links[c].values for c in ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]},
        index=pd.Index(links.link.values, name="link"))
    return path_average(radar, table).transpose("link", "time")


def points_near_links(points: xr.DataArray, links: pd.DataFrame | xr.Dataset,
                      radius_km: float = 3.0) -> xr.DataArray:
    """Mean of the point gauges within ``radius_km`` of each link's path, ``(link, time)``.

    The distance is to the nearest of 5 points along the path. Links with no gauge in
    range are NaN.
    """
    if isinstance(links, xr.Dataset):
        links = pd.DataFrame({c: links[c].values for c in
                              ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]},
                             index=pd.Index(links.link.values, name="link"))
    s = np.linspace(0, 1, 5)
    rows = []
    for _, r in links.iterrows():
        la = r.site_0_lat + s[:, None] * (r.site_1_lat - r.site_0_lat)
        lo = r.site_0_lon + s[:, None] * (r.site_1_lon - r.site_0_lon)
        d = haversine_m(la, lo, points.lat.values[None, :], points.lon.values[None, :]).min(axis=0)
        near = np.flatnonzero(d <= radius_km * 1000)
        rows.append(points.isel(station=near).mean("station", skipna=True).to_numpy()
                    if near.size else np.full(points.sizes["time"], np.nan))
    return xr.DataArray(np.asarray(rows, dtype="float32"), dims=("link", "time"),
                        coords={"link": np.asarray(links.index, dtype=object), "time": points.time.values},
                        attrs={"radius_km": radius_km, "units": points.attrs.get("units", "mm")})
