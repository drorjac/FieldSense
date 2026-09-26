"""
Normalizing CML datasets to one internal convention.

Every source in this pipeline describes the same physical quantities
differently, and the disagreements are silent: the wrong unit does not raise,
it just retrieves the wrong rain rate. Collected here so there is one place
that knows about them.

What actually differs across the sources handled here:

=========================  ==================  ====================  ==================
source                     ``length``          ``frequency``         ``polarization``
=========================  ==================  ====================  ==================
OpenMRG raw (Zenodo)       km, declared        GHz, declared         ``Vertical``
OpenRainER raw (Zenodo)    m, declared         MHz, declared         ``vertical``
OpenMRG example subset     m, declared         MHz, **undeclared**   ``v``
OpenRainER example subset  m, declared         MHz, **undeclared**   ``vertical``
OpenMesh example subset    m, declared         MHz, declared         ``v``
=========================  ==================  ====================  ==================

The undeclared cases are the dangerous ones. ``to_ghz`` therefore falls back to
inspecting the magnitude when no unit is attached: nothing in a terrestrial CML
network transmits at 7,456 GHz, so a value that large is MHz.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# ITU-R P.838-3 is tabulated to 1000 GHz; real CML bands sit well below.
_MAX_PLAUSIBLE_GHZ = 1000.0


def to_km(da) -> np.ndarray:
    """Path length in km, from the declared units."""
    units = str(getattr(da, "attrs", {}).get("units", "")).strip().lower()
    v = np.asarray(da, dtype=float)
    if units in ("m", "metre", "meter", "metres", "meters"):
        return v / 1000.0
    if units in ("km", "kilometre", "kilometer", "kilometres", "kilometers"):
        return v
    if units:
        raise ValueError(f"unexpected length units {units!r}")
    # Undeclared: a terrestrial CML is not 10,000 km long.
    return v / 1000.0 if np.nanmedian(v) > 100.0 else v


def to_ghz(da) -> np.ndarray:
    """Frequency in GHz, from the declared units, or from magnitude if absent."""
    units = str(getattr(da, "attrs", {}).get("units", "")).strip().lower()
    v = np.asarray(da, dtype=float)
    if units == "mhz":
        return v / 1000.0
    if units == "hz":
        return v / 1e9
    if units == "ghz":
        return v
    if units:
        raise ValueError(f"unexpected frequency units {units!r}")
    # Undeclared. The example subsets ship MHz with no units attribute, and
    # feeding those straight into the ITU-R table clamps every link to the top
    # of the table and retrieves near-zero rain, silently.
    return v / 1000.0 if np.nanmedian(v) > _MAX_PLAUSIBLE_GHZ else v


def normalize_polarization(values) -> np.ndarray:
    """Map any spelling onto ``"horizontal"`` / ``"vertical"``.

    Unrecognized entries become ``"vertical"``, which is the ITU-R default
    assumption when polarization is unknown.
    """
    out = []
    for p in np.atleast_1d(np.asarray(values)).ravel():
        s = str(p).strip().lower()
        if s.startswith("h"):
            out.append("horizontal")
        elif s.startswith("v"):
            out.append("vertical")
        else:
            out.append("vertical")
    return np.array(out)


def itu_coefficients(freq_ghz: np.ndarray, pol: np.ndarray) -> tuple:
    """Per-link ITU-R P.838-3 ``(k, alpha)`` arrays."""
    from core.itu_p838 import get_k_alpha

    freq_ghz = np.asarray(freq_ghz, dtype=float)
    pol = np.asarray(pol)
    flat_f = freq_ghz.ravel()
    flat_p = pol.ravel() if pol.size == flat_f.size else np.full(flat_f.size, "vertical")

    k = np.empty(flat_f.size)
    alpha = np.empty(flat_f.size)
    for i, (f, p) in enumerate(zip(flat_f, flat_p)):
        k[i], alpha[i] = get_k_alpha(float(f), str(p))
    return k.reshape(freq_ghz.shape), alpha.reshape(freq_ghz.shape)


def project_cml(ds: xr.Dataset, crs: str = "EPSG:32632") -> xr.Dataset:
    """Attach projected endpoint and midpoint coordinates to a CML dataset.

    ``mergeplg`` builds its line geometry from ``site_0_x/y`` and
    ``site_1_x/y`` in metres, so lat/lon alone is not enough.
    """
    import poligrain as plg

    out = ds
    x0, y0 = plg.spatial.project_point_coordinates(
        out.site_0_lon, out.site_0_lat, crs)
    x1, y1 = plg.spatial.project_point_coordinates(
        out.site_1_lon, out.site_1_lat, crs)
    dim = out.site_0_lon.dims[0]
    for name, val in (("site_0_x", x0), ("site_0_y", y0),
                      ("site_1_x", x1), ("site_1_y", y1)):
        out.coords[name] = (dim, np.asarray(val))
    out.coords["x"] = (dim, (np.asarray(x0) + np.asarray(x1)) / 2)
    out.coords["y"] = (dim, (np.asarray(y0) + np.asarray(y1)) / 2)
    return out


def project_points(ds: xr.Dataset, crs: str = "EPSG:32632",
                   lon: str = "lon", lat: str = "lat") -> xr.Dataset:
    """Attach projected x/y to a point dataset (gauges, PWS, ASOS)."""
    import poligrain as plg

    dim = ds[lon].dims[0]
    x, y = plg.spatial.project_point_coordinates(ds[lon], ds[lat], crs)
    ds.coords["x"] = (dim, np.asarray(x))
    ds.coords["y"] = (dim, np.asarray(y))
    return ds


def project_grid(ds: xr.Dataset, crs: str, lon: str = "lon",
                 lat: str = "lat") -> xr.Dataset:
    """Attach the projected grid coordinates mergeplg and poligrain read.

    Adds 2-D ``x_grid``/``y_grid`` (exact, per cell) and 1-D ``x``/``y`` axes,
    plus 2-D ``lon``/``lat`` if the source had 1-D axes. ``ds`` must be on
    (..., y, x) dimensions, or on (..., lat, lon), which are renamed.

    The 1-D axes are the centre row and centre column of the projected grid.
    A lat/lon grid is not exactly rectilinear once projected, so these are
    bookkeeping only - the same approximation ``mergeplg.io`` makes. The
    geometry itself is always taken from ``x_grid``/``y_grid``.
    """
    import poligrain as plg

    lon2d, lat2d = np.asarray(ds[lon]), np.asarray(ds[lat])
    if lon2d.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)
    if lat in ds.dims and lon in ds.dims:
        ds = ds.rename({lat: "y", lon: "x"})
        lon, lat = "lon", "lat"
    # Any existing x/y are in the source's own projection (OpenMRG ships a
    # polar stereographic grid); they are replaced, not trusted.
    ds = ds.drop_vars([c for c in (lon, lat, "x", "y") if c in ds.variables])
    ds.coords["lon"] = (("y", "x"), lon2d)
    ds.coords["lat"] = (("y", "x"), lat2d)
    # older code in this repository used these names; keep both
    ds.coords["longitudes"] = ds.lon
    ds.coords["latitudes"] = ds.lat

    xs, ys = plg.spatial.project_point_coordinates(ds.lon, ds.lat, crs)
    xv, yv = np.asarray(xs), np.asarray(ys)
    ds.coords["x_grid"] = (("y", "x"), xv)
    ds.coords["y_grid"] = (("y", "x"), yv)
    ds.coords["x"] = ("x", xv[xv.shape[0] // 2, :])
    ds.coords["y"] = ("y", yv[:, xv.shape[1] // 2])
    ds.attrs["crs"] = crs
    return ds


# Required by the OpenSense-1.0 CML convention, for ``check_format``.
_CML_DIMS = ("time", "cml_id", "sublink_id")
_CML_COORDS = ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
               "frequency", "length")


def check_format(ds: xr.Dataset):
    """OpenSense-1.0 CML format checks, one row per check.

    Returns a DataFrame with ``check``, ``passed`` and ``detail``, so a
    failure says what was found rather than just that something is off.
    Units are checked for being declared, because an undeclared unit is the
    failure that goes unnoticed (see the module docstring).
    """
    import pandas as pd

    rows = []

    def add(check, passed, detail=""):
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    missing = [d for d in _CML_DIMS if d not in ds.dims]
    add("dimensions time, cml_id, sublink_id", not missing,
        f"missing {missing}" if missing else "")
    if "time" in ds.coords:
        add("time is datetime64", ds.time.dtype.kind == "M", str(ds.time.dtype))
    for name in ("cml_id", "sublink_id"):
        if name in ds.coords:
            add(f"{name} is a string", ds[name].dtype.kind in "USO", str(ds[name].dtype))
    missing = [c for c in _CML_COORDS if c not in ds.variables]
    add("site coordinates, frequency, length", not missing,
        f"missing {missing}" if missing else "")
    signals = [v for v in ("rsl", "tsl") if v in ds.data_vars]
    add("rsl present (tsl optional)", "rsl" in signals, f"found {signals}")
    for v in signals:
        add(f"{v} in dBm", str(ds[v].attrs.get("units", "")).lower() == "dbm",
            f"units={ds[v].attrs.get('units', '(none)')!r}")
    for c in ("frequency", "length"):
        if c in ds.variables:
            units = ds[c].attrs.get("units")
            add(f"{c} units declared", bool(units), f"units={units!r}")
    return pd.DataFrame(rows)
