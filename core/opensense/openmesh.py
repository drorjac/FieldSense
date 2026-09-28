"""OpenMesh NYC wireless-link dataset (and its companion PWS gauge dataset).

OpenMesh (Jacoby et al., Zenodo 10.5281/zenodo.15287692, CC BY 4.0) holds 1-min received
signal level (RSL, dBm) from the NYC Mesh community network: 75 links x up to 3
sublinks, of which **103 sublinks** carry data, 2023-10-29 .. 2024-07-01 UTC, in the
OpenSense CML NetCDF convention: ``rsl(cml_id, sublink_id, time)`` with per-sublink
``frequency`` (MHz), ``polarization`` and per-link ``length`` (m) and site coordinates.
There is no transmitted-power (TSL) variable; TSL is constant and taken as 0 dBm, so
total loss = -RSL and only *changes* in RSL matter.

The companion PWS dataset (Zenodo 10.5281/zenodo.17508286, CC BY-NC 4.0) has 37 Weather
Underground stations, one NetCDF group each, ~5-min ``rainfall_amount`` (mm).

The central in-memory object is a **link set**: an ``xarray.Dataset`` with ``rsl(link,
time)`` where ``link`` is a flat ``"<cml_id>/<sublink_id>"`` label, and link metadata
as coordinates (``cml_id, sublink_id, frequency [GHz], length [km], polarization,
site_0_lat/lon, site_1_lat/lon, mid_lat/lon``). Every estimator in
the ``nyc_rain_maps`` project consumes this object.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from core.opensense import fetch

log = logging.getLogger(__name__)

RAW = fetch.DATA_ROOT / "OpenMesh_NYC" / "raw"
EXTRACTED = RAW / "extracted"

OPENMESH_FILE = "ds_openmesh.nc"
LINKS_META_FILE = "links_metadata.csv"
PWS_FILE = "pws_wu_os.nc"
PWS_META_FILE = "pws_metadata.csv"


# ------------------------------------------------------------------ locate / fetch


def openmesh_path(download: bool = True) -> Path:
    """Path to ``ds_openmesh.nc``.

    Resolution: ``$OPENMESH_NC`` (a copy you already have) -> ``raw/extracted/`` ->
    download ``OpenMesh.zip`` from Zenodo with :mod:`core.opensense.fetch` (checksum
    verified) and extract it, if ``download``.
    """
    return _locate("OPENMESH_NC", OPENMESH_FILE, "openmesh", "OpenMesh.zip",
                   (OPENMESH_FILE, LINKS_META_FILE, PWS_META_FILE), download)


def pws_path(download: bool = True) -> Path:
    """Path to ``pws_wu_os.nc`` (``$OPENMESH_PWS_NC`` -> ``raw/extracted/`` -> Zenodo)."""
    return _locate("OPENMESH_PWS_NC", PWS_FILE, "openmesh_pws", "PWS_NYC_WU.zip",
                   (PWS_FILE, PWS_META_FILE), download)


def _locate(env: str, name: str, source: str, archive: str, members, download: bool) -> Path:
    local = os.environ.get(env)
    if local and Path(local).expanduser().exists():
        return Path(local).expanduser()
    path = EXTRACTED / name
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} not found; run python -m core.opensense.fetch "
                                f"--dataset {source}, or set {env}")
    zip_path = RAW / archive
    if not zip_path.exists():
        fetch.fetch(source)
    _extract(zip_path, {m: m for m in members}, EXTRACTED)
    return path


def _extract(zip_path: Path, wanted: Mapping[str, str], dest: Path) -> None:
    """Extract members whose basename is a key of ``wanted`` to ``dest/<value>``."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in wanted:
                with z.open(member) as src, open(dest / wanted[name], "wb") as out:
                    shutil.copyfileobj(src, out)


# ------------------------------------------------------------------------ links


_OPEN: dict[Path, xr.Dataset] = {}


def open_openmesh(path: str | Path | None = None) -> xr.Dataset:
    """Open ``ds_openmesh.nc`` lazily (nothing is read until selected).

    The handle is opened once per process and reused: opening the same HDF5 file
    through several live netCDF4 handles can crash the HDF5 library.
    """
    p = Path(path or openmesh_path()).resolve()
    if p not in _OPEN:
        _OPEN[p] = xr.open_dataset(p)
    return _OPEN[p]


def link_label(cml_id, sublink_id) -> str:
    return f"{int(cml_id)}/{sublink_id}"


def sublinks_table(ds: xr.Dataset | None = None, only_with_data: bool = True) -> pd.DataFrame:
    """One row per sublink with metadata and data availability.

    Columns: ``cml_id, sublink_id, frequency (GHz), length (km), polarization,
    site_0_lat, site_0_lon, site_1_lat, site_1_lon, mid_lat, mid_lon, availability``
    (fraction of non-NaN minutes over the full record). Indexed by link label.
    With ``only_with_data`` (default) the 103 sublinks that carry data are returned.
    """
    ds = ds if ds is not None else open_openmesh()
    avail = ds["rsl"].notnull().mean("time").load()
    rows = []
    for cml in ds.cml_id.values:
        for sub in ds.sublink_id.values:
            a = float(avail.sel(cml_id=cml, sublink_id=sub))
            if only_with_data and a == 0:
                continue
            c = ds.sel(cml_id=cml)
            rows.append({
                "cml_id": int(cml), "sublink_id": str(sub),
                "frequency": float(ds.frequency.sel(cml_id=cml, sublink_id=sub)) / 1000.0,
                "length": float(c.length) / 1000.0,
                "polarization": str(ds.polarization.sel(cml_id=cml, sublink_id=sub).values).lower(),
                "site_0_lat": float(c.site_0_lat), "site_0_lon": float(c.site_0_lon),
                "site_1_lat": float(c.site_1_lat), "site_1_lon": float(c.site_1_lon),
                "availability": a,
            })
    df = pd.DataFrame(rows)
    df["mid_lat"] = (df.site_0_lat + df.site_1_lat) / 2
    df["mid_lon"] = (df.site_0_lon + df.site_1_lon) / 2
    df.index = [link_label(r.cml_id, r.sublink_id) for r in df.itertuples()]
    df.index.name = "link"
    return df


def load_links(start, end, links: Iterable[str] | Mapping[int, str] | None = None,
               ds: xr.Dataset | None = None) -> xr.Dataset:
    """Load a link set: ``rsl(link, time)`` in dBm for ``[start, end]`` (inclusive).

    ``links`` may be ``None`` (all 103 sublinks with data), an iterable of labels
    (``"8/sublink_1"``), or a ``{cml_id: sublink_id}`` mapping as used by
    implementation_2's ``sublink_choice``. Order is preserved. Gaps stay NaN; filling
    is a method decision, made explicitly by the retrieval.
    """
    ds = ds if ds is not None else open_openmesh()
    table = sublinks_table(ds, only_with_data=False)
    if links is None:
        labels = list(sublinks_table(ds).index)
    elif isinstance(links, Mapping):
        labels = [link_label(k, v) for k, v in links.items()]
    else:
        labels = list(links)
    unknown = [lab for lab in labels if lab not in table.index]
    if unknown:
        raise KeyError(f"unknown links: {unknown}")
    meta = table.loc[labels]

    sub = ds["rsl"].sel(time=slice(pd.Timestamp(start), pd.Timestamp(end)))
    arrays = [sub.sel(cml_id=str(r.cml_id), sublink_id=r.sublink_id).values for r in meta.itertuples()]
    rsl = np.stack(arrays).astype("float32")
    coords = {"link": labels, "time": sub.time.values}
    for col in ["cml_id", "sublink_id", "frequency", "length", "polarization", "site_0_lat",
                "site_0_lon", "site_1_lat", "site_1_lon", "mid_lat", "mid_lon"]:
        # to_numpy: under pandas 3 a string column's .values is an extension array,
        # which xarray < 2024.9 cannot copy
        coords[col] = ("link", meta[col].to_numpy())
    out = xr.Dataset({"rsl": (("link", "time"), rsl)}, coords=coords)
    out["rsl"].attrs = {"units": "dBm", "long_name": "received signal level"}
    out["frequency"].attrs["units"] = "GHz"
    out["length"].attrs["units"] = "km"
    out.attrs = {"source": "OpenMesh (Zenodo 10.5281/zenodo.15287692)",
                 "start": str(pd.Timestamp(start)), "end": str(pd.Timestamp(end))}
    return out


def links_frame(links: xr.Dataset) -> pd.DataFrame:
    """Metadata of a link set as a DataFrame indexed by link label."""
    cols = [c for c in links.coords if links[c].dims == ("link",) and c != "link"]
    return pd.DataFrame({c: links[c].values for c in cols}, index=pd.Index(links.link.values, name="link"))


# ------------------------------------------------------------------------- PWS


def load_pws(start=None, end=None, path: str | Path | None = None,
             max_mm_per_5min: float = 15.0) -> xr.Dataset:
    """PWS gauges as ``rain(station, time)`` in mm per 5-min interval.

    Each station's irregular record is summed into regular 5-min bins
    (interval-ENDING labels). Basic QC: negative amounts and single-interval
    amounts above ``max_mm_per_5min`` (180 mm/h, beyond NYC records) are NaN.
    Station coordinates are the ``lat``/``lon`` coordinates.
    """
    import netCDF4

    path = path or pws_path()
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    with netCDF4.Dataset(path) as root:
        stations = list(root.groups)
    series, lat, lon = {}, [], []
    for st in stations:
        g = xr.open_dataset(path, group=st)
        s = g["rainfall_amount"].isel(id=0).to_series()
        lat.append(float(g["lat"].values[0]))
        lon.append(float(g["lon"].values[0]))
        g.close()
        if start is not None or end is not None:
            s = s.loc[start - pd.Timedelta("5min") if start is not None else None:end]
        s = s.where((s >= 0) & (s <= max_mm_per_5min))
        series[st] = s.resample("5min", label="right", closed="right").sum(min_count=1)
    df = pd.DataFrame(series)
    if start is not None or end is not None:
        df = df.loc[start:end]
    out = xr.Dataset({"rain": (("time", "station"), df.values.astype("float32"))},
                     coords={"time": df.index.values, "station": stations,
                             "lat": ("station", lat), "lon": ("station", lon)})
    out = out.transpose("station", "time")
    out["rain"].attrs = {"units": "mm", "interval": "5min", "time_label": "interval end (UTC)"}
    out.attrs = {"source": "Weather Underground PWS via OpenMesh PWS (Zenodo 10.5281/zenodo.17508286)"}
    return out
