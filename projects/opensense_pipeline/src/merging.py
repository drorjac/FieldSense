"""
Thin, uniform wrapper over the ``mergeplg`` interpolation and merging methods.

Every method in ``mergeplg`` follows the same two-step contract: ``update()``
caches geometry and intersection weights, then ``adjust()`` (merging) or
``interpolate()`` (CML/gauge only) produces the field. The signatures differ
though - IDW takes ``p``/``nnear``/``max_distance``, the kriging methods take
``variogram_model``/``variogram_parameters``, and KED takes ``n_closest``. This
module hides those differences behind one ``run(...)`` call so a benchmark can
loop over methods without special-casing each one.

Two baselines are included alongside the ``mergeplg`` methods so the *gain*
from merging is measurable rather than assumed:

``radar_only``
    the radar field, untouched. Any merge that cannot beat this is not earning
    its complexity.
``cml_only_idw``
    CML observations interpolated with no radar input at all.

Two mergeplg APIs are supported. The 0.1.0 release (pinned, the published
results) builds an object and calls ``update()`` then ``adjust()``; mergeplg
``main`` takes the geometry in the constructor and is called per timestep,
and renames and re-defaults several parameters. ``NEW_API`` says which is
installed; each method carries explicit settings for both, so a run on
``main`` compares like with like. Two methods exist only on ``main`` and are
listed only there: RADOLAN (the DWD operational adjustment) and difference
kriging with the nugget estimated from link geometry (``c0_within``).
Results from ``main`` are written with the ``API_TAG`` suffix, never over
the published files.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import xarray as xr
from mergeplg import interpolate, merge


# Any estimate above this is a numerical blow-up, not rain.
IMPLAUSIBLE_MM_H = 300.0

NEW_API = "ds_rad" in inspect.signature(merge.MergeDifferenceIDW.__init__).parameters
API_TAG = "_mergeplg-main" if NEW_API else ""


@dataclass(frozen=True)
class Method:
    """One named reconstruction method."""

    key: str
    label: str
    uses_radar: bool
    uses_cml: bool
    build: Callable | None = None      # constructs the mergeplg object (0.1.0)
    kwargs: dict = field(default_factory=dict)
    new_cls: str = ""                  # "merge.X" / "interpolate.X" on mergeplg main
    new_kwargs: dict = field(default_factory=dict)
    main_only: bool = False

    @property
    def family(self) -> str:
        if not self.uses_radar:
            return "CML only"
        if not self.uses_cml:
            return "radar only"
        return "merged"


# The variogram is estimated per timestep by mergeplg when parameters are None.
# Letting it fit rather than pinning a guess keeps the comparison fair across
# the very different fields in the synthetic benchmark.
METHODS: tuple[Method, ...] = (
    Method("radar_only", "Radar only", True, False),
    Method(
        "cml_idw", "CML only, IDW", False, True,
        build=lambda: interpolate.InterpolateIDW(min_observations=1),
        kwargs=dict(p=2, idw_method="radolan", nnear=8, max_distance=30000),
        new_cls="interpolate.InterpolateIDW",
        new_kwargs=dict(min_observations=1, p=2, idw_method="radolan", nnear=8,
                        max_distance=30000),
    ),
    Method(
        "cml_okrig", "CML only, block kriging", False, True,
        build=lambda: interpolate.InterpolateOrdinaryKriging(
            discretization=8, min_observations=1),
        kwargs=dict(variogram_model="spherical", nnear=8, full_line=True),
        new_cls="interpolate.InterpolateOrdinaryKriging",
        new_kwargs=dict(discretization=8, min_observations=1, variogram_model="spherical",
                        nnear=8, full_line=True),
    ),
    Method(
        "merge_idw_add", "Merge: difference IDW (additive)", True, True,
        build=lambda: merge.MergeDifferenceIDW(min_observations=1),
        kwargs=dict(p=2, idw_method="radolan", nnear=8, max_distance=30000,
                    method="additive"),
        new_cls="merge.MergeDifferenceIDW",
        new_kwargs=dict(min_observations=1, p=2, idw_method="radolan", nnear=8,
                        max_distance=30000, method="additive"),
    ),
    Method(
        "merge_idw_mult", "Merge: difference IDW (multiplicative)", True, True,
        build=lambda: merge.MergeDifferenceIDW(min_observations=1),
        kwargs=dict(p=2, idw_method="radolan", nnear=8, max_distance=30000,
                    method="multiplicative"),
        new_cls="merge.MergeDifferenceIDW",
        new_kwargs=dict(min_observations=1, p=2, idw_method="radolan", nnear=8,
                        max_distance=30000, method="multiplicative"),
    ),
    Method(
        "merge_okrig_add", "Merge: difference kriging (additive)", True, True,
        build=lambda: merge.MergeDifferenceOrdinaryKriging(
            discretization=8, min_observations=1),
        kwargs=dict(variogram_model="spherical", nnear=8, full_line=True,
                    method="additive"),
        new_cls="merge.MergeDifferenceOrdinaryKriging",
        new_kwargs=dict(discretization=8, min_observations=1, variogram_model="spherical",
                        nnear=8, full_line=True, method="additive"),
    ),
    Method(
        "merge_ked", "Merge: kriging with external drift", True, True,
        build=lambda: merge.MergeKrigingExternalDrift(
            discretization=8, min_observations=1),
        kwargs=dict(variogram_model="spherical", n_closest=8),
        new_cls="merge.MergeKrigingExternalDrift",
        new_kwargs=dict(discretization=8, min_observations=1, variogram_model="spherical",
                        nnear=8),
    ),
    Method(
        "merge_okrig_add_c0", "Merge: difference kriging, nugget from link geometry",
        True, True, main_only=True,
        new_cls="merge.MergeDifferenceOrdinaryKriging",
        new_kwargs=dict(discretization=8, min_observations=1, variogram_model="spherical",
                        nnear=8, full_line=True, method="additive", c0_within=True),
    ),
    Method(
        "merge_radolan", "Merge: RADOLAN (DWD)", True, True, main_only=True,
        new_cls="merge.MergeRADOLAN",
        new_kwargs=dict(nnear=8, max_distance=60000),
    ),
)
METHODS = tuple(m for m in METHODS if NEW_API or not m.main_only)

METHODS_BY_KEY = {m.key: m for m in METHODS}


def run(method: Method, da_rad: xr.DataArray, da_cml: xr.DataArray | None,
        da_gauge: xr.DataArray | None = None) -> xr.DataArray:
    """Produce a rainfall field for one method at one timestep.

    ``da_rad`` must carry ``x``/``y`` coordinates; ``da_cml`` must carry the
    ``site_0_x/y`` and ``site_1_x/y`` endpoint coordinates that mergeplg uses
    to build line geometry.

    Returns a DataArray on the radar grid. On failure - too few observations,
    a singular kriging system - returns an all-NaN field rather than raising,
    so one bad timestep cannot abort a long benchmark.
    """
    da_rad = conform_radar(da_rad)
    da_cml = conform_cml(da_cml)
    da_gauge = conform_gauge(da_gauge)

    if method.key == "radar_only":
        return da_rad
    if NEW_API:
        return _run_new(method, da_rad, da_cml, da_gauge)

    obj = method.build()
    kwargs = dict(method.kwargs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method.uses_radar:
                obj.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)
                out = obj.adjust(da_rad, da_cml=da_cml, da_gauge=da_gauge,
                                 **kwargs)
            else:
                obj.update(da_cml=da_cml, da_gauge=da_gauge)
                out = obj.interpolate(da_rad, da_cml=da_cml,
                                      da_gauge=da_gauge, **kwargs)
    except Exception as exc:                        # noqa: BLE001
        return xr.full_like(da_rad, np.nan).assign_attrs(
            merge_error=f"{type(exc).__name__}: {exc}")

    out = xr.DataArray(np.asarray(out), dims=da_rad.dims, coords=da_rad.coords)
    return out.clip(min=0.0)


def _run_new(method: Method, da_rad, da_cml, da_gauge) -> xr.DataArray:
    """``run`` on mergeplg main: geometry in the constructor, one call per step."""
    module, name = method.new_cls.split(".")
    cls = getattr(merge if module == "merge" else interpolate, name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if module == "interpolate":
                obj = cls(ds_grid=da_rad, ds_cmls=da_cml, ds_gauges=da_gauge,
                          **method.new_kwargs)
                out = obj(da_cmls=da_cml, da_gauges=da_gauge)
            else:
                extra = {}
                if name == "MergeRADOLAN":
                    # its station table is built from the inputs' dtype and then
                    # given float64 values, which pandas 3 refuses for float32
                    da_rad = da_rad.astype("float64")
                    da_cml = None if da_cml is None else da_cml.astype("float64")
                    da_gauge = None if da_gauge is None else da_gauge.astype("float64")
                    extra = {"start_index_in_relevant_stations": 0}   # not random
                obj = cls(ds_rad=da_rad, ds_cmls=da_cml, ds_gauges=da_gauge,
                          **method.new_kwargs)
                out = obj(da_rad, da_cmls=da_cml, da_gauges=da_gauge, **extra)
                if isinstance(out, xr.Dataset):       # RADOLAN returns every product
                    out = out["RW_not_rounded"]
    except Exception as exc:                        # noqa: BLE001
        return xr.full_like(da_rad, np.nan).assign_attrs(
            merge_error=f"{type(exc).__name__}: {exc}")
    out = xr.DataArray(np.asarray(out), dims=da_rad.dims, coords=da_rad.coords)
    return out.clip(min=0.0)


# --------------------------------------------------------------------------
def score(truth: np.ndarray, estimate: np.ndarray) -> dict:
    """Agreement between a reconstructed field and a reference.

    Pixels where either side is NaN are dropped, and the count kept, so a
    method that quietly fails over most of the domain cannot post a good score
    on the handful of pixels it did fill.
    """
    t = np.asarray(truth, dtype=float).ravel()
    e = np.asarray(estimate, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(e)
    n = int(ok.sum())
    if n < 2:
        keys = ("rmse", "mae", "median_abs_err", "p95_abs_err", "bias",
                "corr", "max_estimate", "frac_implausible")
        return {k: float("nan") for k in keys} | {
            "n_valid": n, "coverage": n / t.size if t.size else 0.0}

    t, e = t[ok], e[ok]
    err = e - t
    denom = t.std() * e.std()

    # Multiplicative merging divides by the radar field, so a near-zero radar
    # pixel under a raining link sends the estimate to absurd values; KED can
    # extrapolate the same way. A single such pixel dominates RMSE and hides
    # how the method behaves everywhere else, so robust statistics and an
    # explicit blow-up rate are reported alongside it. 300 mm/h is far above
    # any rate observed in these datasets.
    implausible = float((e > IMPLAUSIBLE_MM_H).mean())

    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "median_abs_err": float(np.median(np.abs(err))),
        "p95_abs_err": float(np.percentile(np.abs(err), 95)),
        "bias": float(err.mean()),
        "corr": float(((t - t.mean()) * (e - e.mean())).mean() / denom)
                if denom > 0 else float("nan"),
        "max_estimate": float(e.max()),
        "frac_implausible": implausible,
        "n_valid": n,
        "coverage": float(n / ok.size),
    }


# --------------------------------------------------------------------------
# conforming data to the mergeplg contract
# --------------------------------------------------------------------------
# mergeplg reads geometry off specific coordinate names, and they do not all
# match what the OpenSense-1.0 files (or mergeplg's own io loader) provide:
#   da_rad   needs 2-D  x_grid / y_grid   (the loader supplies xs / ys)
#   da_gauge needs      id, x, y          (files supply station_id)
#   da_cml   needs      cml_id, site_0_x/y, site_1_x/y
# These helpers rename rather than copy, so the arrays stay views where xarray
# allows it.

def conform_radar(da_rad: xr.DataArray) -> xr.DataArray:
    """Give a radar DataArray the 2-D ``x_grid``/``y_grid`` coords mergeplg wants."""
    da = da_rad
    if "x_grid" not in da.coords or "y_grid" not in da.coords:
        if "xs" in da.coords and "ys" in da.coords:
            da = da.assign_coords(x_grid=da.xs, y_grid=da.ys)
        elif "x" in da.coords and "y" in da.coords:
            xg, yg = np.meshgrid(np.asarray(da.x), np.asarray(da.y))
            da = da.assign_coords(
                x_grid=(("y", "x"), xg), y_grid=(("y", "x"), yg))
        else:
            raise ValueError("radar needs x/y or xs/ys coordinates")
    return da


def conform_gauge(da_gauge: xr.DataArray | None,
                  crs: str = "EPSG:32632") -> xr.DataArray | None:
    """Rename the station dimension to ``id`` and ensure x/y *and* lon/lat.

    poligrain's ``GridAtPoints.__call__`` builds its output with
    ``da_point_data.lon`` unconditionally, even when it was constructed with
    ``use_lon_lat=False`` and did every calculation in projected metres. So
    lon/lat have to be present even though nothing reads them numerically;
    they are back-projected from x/y when a caller did not supply them.
    """
    if da_gauge is None:
        return None
    da = da_gauge
    for candidate in ("station_id", "gauge_id", "name"):
        if candidate in da.dims:
            da = da.rename({candidate: "id"})
            break
    if "id" not in da.coords:
        da = da.assign_coords(id=("id", np.arange(da.sizes["id"])))

    missing = [c for c in ("x", "y") if c not in da.coords]
    if missing:
        raise ValueError(f"gauge data is missing projected coords: {missing}")

    if "lon" not in da.coords or "lat" not in da.coords:
        from pyproj import Transformer
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(np.asarray(da.x), np.asarray(da.y))
        da = da.assign_coords(lon=("id", lon), lat=("id", lat))
    return da


def conform_cml(da_cml: xr.DataArray | None) -> xr.DataArray | None:
    """Check the CML endpoint coordinates mergeplg builds line geometry from."""
    if da_cml is None:
        return None
    needed = ("site_0_x", "site_0_y", "site_1_x", "site_1_y")
    missing = [c for c in needed if c not in da_cml.coords]
    if missing:
        raise ValueError(f"CML data is missing endpoint coords: {missing}")
    if "cml_id" not in da_cml.dims:
        raise ValueError("CML data must have a 'cml_id' dimension")
    return da_cml
