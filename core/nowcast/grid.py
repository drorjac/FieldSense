"""The nowcasting grid and the bridge from xarray fields to pysteps.

Shared by ``projects/nowcasting/pysteps`` and ``projects/nowcasting/multisensor``.

pysteps measures motion in pixels and takes ``kmperpixel`` for its spectral cascade, so
the pixels should be square. The training school read OpenRainER on its native lat/lon
grid (0.0127 x 0.0090 deg, i.e. 1.0 x 1.0 km only by accident of latitude) with
``cartesian_unit = "degree"``. Here every network gets a grid of square ``pixel_km``
pixels in a local equirectangular projection around the domain centre - which is itself a
regular lat/lon grid, with ``dlon = dlat / cos(lat0)`` - so the same
:class:`~core.geo.Grid` serves the link maps, the merging code and pysteps, and the
metadata carry a real projection in metres.

    grid = square_grid(domain, 2.0)
    precip, meta = to_pysteps(field_mm, step_min=15)      # (time, y, x) mm/h + metadata
    dbr, meta_db = to_dbr(precip, meta)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import EARTH_RADIUS_M, Domain, Grid

KM_PER_DEG = np.pi * EARTH_RADIUS_M / 180.0 / 1000.0


def square_grid(domain: Domain, pixel_km: float = 2.0) -> Grid:
    """Regular lat/lon grid over ``domain`` whose cells are ``pixel_km`` square at its centre."""
    lat0 = 0.5 * (domain.lat_min + domain.lat_max)
    dlat = pixel_km / KM_PER_DEG
    dlon = dlat / np.cos(np.radians(lat0))
    lat = np.arange(domain.lat_min + dlat / 2, domain.lat_max, dlat)
    lon = np.arange(domain.lon_min + dlon / 2, domain.lon_max, dlon)
    return Grid(np.round(lat, 6), np.round(lon, 6))


def pixel_km(grid: Grid) -> float:
    return float(np.diff(grid.lat[:2])[0]) * KM_PER_DEG


def metadata(grid: Grid, step_min: int, times=None, unit: str = "mm/h",
             product: str = "nowcast") -> dict:
    """pysteps metadata for ``grid``: an equirectangular projection in metres."""
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    px = pixel_km(grid) * 1000.0
    ny, nx = grid.shape
    return {
        "projection": f"+proj=eqc +lat_ts={lat0:.6f} +lat_0={lat0:.6f} +lon_0={lon0:.6f} "
                      "+ellps=sphere +R=6371008.8 +units=m",
        "cartesian_unit": "m",
        "x1": -nx * px / 2, "x2": nx * px / 2, "y1": -ny * px / 2, "y2": ny * px / 2,
        "xpixelsize": px, "ypixelsize": px,
        "yorigin": "lower",                          # row 0 is the southernmost latitude
        "accutime": float(step_min),
        "unit": unit, "transform": None, "threshold": 0.1, "zerovalue": 0.0,
        "zr_a": 200.0, "zr_b": 1.6,
        "timestamps": None if times is None else list(pd.DatetimeIndex(times).to_pydatetime()),
        "institution": "FieldSense", "product": product,
    }


def to_pysteps(field_mm: xr.DataArray, step_min: int, fill: float | None = 0.0) -> tuple[np.ndarray, dict]:
    """``(time, lat, lon)`` accumulations in mm per step -> rain rate (mm/h) array and metadata.

    ``fill`` replaces non-finite values (the session set them to the field's minimum, i.e.
    zero rain); ``None`` keeps the NaNs.
    """
    f = field_mm.transpose("time", "lat", "lon")
    rate = f.values.astype("float64") * (60.0 / step_min)
    if fill is not None:
        rate = np.where(np.isfinite(rate), rate, fill)
    grid = Grid(f.lat.values, f.lon.values)
    return rate, metadata(grid, step_min, f.time.values)


DB_THRESHOLD = 0.1          # mm/h: the training school's dB transform
DB_ZEROVALUE = -15.0        # dBR assigned below the threshold


def to_dbr(rate: np.ndarray, meta: dict, threshold: float = DB_THRESHOLD,
           zerovalue: float = DB_ZEROVALUE) -> tuple[np.ndarray, dict]:
    """The session's transform: mm/h -> dBR, threshold 0.1 mm/h, -15 dBR below it."""
    from pysteps.utils import transformation
    return transformation.dB_transform(rate, meta, threshold=threshold, zerovalue=zerovalue)


def from_dbr(dbr: np.ndarray, meta_db: dict) -> np.ndarray:
    """dBR back to mm/h (values below the threshold become 0)."""
    from pysteps.utils import transformation
    out = transformation.dB_transform(dbr, threshold=meta_db["threshold"], inverse=True)[0]
    return np.where(np.isfinite(out), out, 0.0)
