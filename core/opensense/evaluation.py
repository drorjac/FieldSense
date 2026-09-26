"""
Scoring CML retrievals and rainfall maps against other sensors, with poligrain.

Three sensors, three geometries - a link measures a path average, a radar a
grid-cell average, a gauge a point - and every comparison between them first
has to put both on the same support. ``poligrain`` has the matching tools;
this module wraps them in the shapes the pipeline uses, and records the
places where calling them directly silently goes wrong.

Matching

``radar_along_links``   radar path-averaged along each link
                        (``poligrain.spatial.GridAtLines``) - the like-for-like
                        way to compare radar with a CML, rather than
                        interpolating the CMLs onto the grid
``closest_gauges``      the gauges within a distance of each link *path*
                        (``poligrain.spatial.get_closest_points_to_line``)
``gauge_series_at_links`` a gauge time series per link from that matching
``grid_at_points``      a gridded field at gauge locations
                        (``poligrain.spatial.GridAtPoints``)
``distance_to_network`` distance from any point to the nearest link path

Scoring

``rainfall_metrics``    continuous and wet/dry skill in one dict
                        (``poligrain.validation``), with the threshold
                        semantics made explicit - see its docstring

Example
-------
>>> from core.opensense import evaluation as ev
>>> radar_path = ev.radar_along_links(data["radar"].R, cml)     # (time, cml_id)
>>> ev.rainfall_metrics(radar_path, cml_rain)["pbias"]
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Wet/dry threshold used throughout, mm/h. Below this a 10 s CML sample, a
# 5-minute radar pixel and a tipping bucket are all indistinguishable from
# noise or quantization.
WET_THRESHOLD_MM_H = 0.1


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def _line_geometry(ds_cml: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Just the per-link geometry poligrain needs, with no time axis."""
    names = ("site_0_x", "site_0_y", "site_1_x", "site_1_y",
             "site_0_lon", "site_0_lat", "site_1_lon", "site_1_lat")
    missing = [n for n in names[:4] if n not in ds_cml.coords]
    if missing:
        raise ValueError(f"CML data is missing projected endpoints {missing}; "
                         f"run conventions.project_cml first")
    geo = xr.Dataset(coords={"cml_id": ds_cml.cml_id})
    for n in names:
        if n in ds_cml.coords:
            c = ds_cml[n]
            if "sublink_id" in c.dims:
                c = c.isel(sublink_id=0, drop=True)
            geo.coords[n] = ("cml_id", np.asarray(c))
    # GridAtLines copies site lon/lat onto its output for plotting even when
    # it computed in projected metres, and fails without them. They are not
    # read numerically, so NaN placeholders are honest when a caller has
    # projected coordinates only.
    for n in names[4:]:
        if n not in geo.coords:
            geo.coords[n] = ("cml_id", np.full(geo.sizes["cml_id"], np.nan))
    return geo


def radar_along_links(da_rad: xr.DataArray, ds_cml: xr.Dataset | xr.DataArray,
                      grid_point_location: str = "center") -> xr.DataArray:
    """Radar averaged along each link path, weighted by intersection length.

    ``da_rad`` is (time, y, x) or (y, x) with 2-D ``x_grid``/``y_grid`` in the
    same projected CRS as the links' ``site_*_x/y`` (see
    ``conventions.project_grid``). Returns (time, cml_id).

    This is the comparison a CML actually supports: a link reports one
    number for its whole path, so the radar is reduced to the same path
    rather than the link smeared out onto the grid. Links that leave the
    radar domain get the average of the part inside it.
    """
    import poligrain as plg

    if "time" in da_rad.dims:
        da_rad = da_rad.transpose("time", "y", "x")
    geo = _line_geometry(ds_cml)
    gal = plg.spatial.GridAtLines(da_rad, geo, grid_point_location=grid_point_location,
                                  use_lon_lat=False)
    out = gal(da_rad)
    out.attrs.update(units=da_rad.attrs.get("units", "mm h-1"),
                     long_name="radar path average")
    return out


def closest_gauges(ds_cml: xr.Dataset | xr.DataArray, ds_gauge: xr.Dataset,
                   max_distance_m: float = 1000.0,
                   n_closest: int = 1) -> xr.Dataset:
    """Gauges within ``max_distance_m`` of each link path.

    Wraps ``poligrain.spatial.get_closest_points_to_line``, with one trap
    removed. poligrain searches a radius of ``length / 2 + max_distance``
    around each link midpoint, reading ``length`` **in the units of the
    coordinates** - metres. This pipeline, like the OpenSense files after
    ``example_data.normalize_cml``, carries ``length`` in km. Passed straight
    through, the search radius around a 4 km link shrinks from 3 km to
    1.002 km and gauges beside the far half of the path are silently missed.
    The length here is always recomputed from the projected endpoints.

    Returns poligrain's dataset: ``distance`` and ``neighbor_id`` on
    (cml_id, n_closest), with ``inf``/``None`` where fewer gauges qualify.
    """
    import poligrain as plg

    geo = _line_geometry(ds_cml)
    geo.coords["length"] = ("cml_id", np.hypot(
        np.asarray(geo.site_1_x) - np.asarray(geo.site_0_x),
        np.asarray(geo.site_1_y) - np.asarray(geo.site_0_y)))
    pts = xr.Dataset(coords={"id": ds_gauge.id,
                             "x": ("id", np.asarray(ds_gauge.x)),
                             "y": ("id", np.asarray(ds_gauge.y))})
    return plg.spatial.get_closest_points_to_line(
        geo, pts, max_distance=max_distance_m, n_closest=n_closest)


def gauge_series_at_links(da_gauge: xr.DataArray, closest: xr.Dataset) -> xr.DataArray:
    """(time, id) gauge data -> (time, cml_id), the nearest gauge per link.

    Links with no gauge within the matching distance are all-NaN, so a
    straight elementwise comparison with the CML series only scores the
    matched pairs.
    """
    nearest = closest.neighbor_id.isel(n_closest=0).values
    ids = list(np.asarray(da_gauge.id))
    g = da_gauge.transpose("time", "id").values
    out = np.full((g.shape[0], nearest.size), np.nan)
    for j, gid in enumerate(nearest):
        if gid is not None and gid in ids:
            out[:, j] = g[:, ids.index(gid)]
    da = xr.DataArray(out, dims=("time", "cml_id"),
                      coords={"time": da_gauge.time, "cml_id": closest.cml_id})
    da.coords["gauge_id"] = ("cml_id", np.array(
        [g if g is not None else "" for g in nearest], dtype=object))
    da.coords["gauge_distance_m"] = ("cml_id", closest.distance.isel(n_closest=0).values)
    return da


def grid_at_points(da_grid: xr.DataArray, da_points: xr.DataArray,
                   nnear: int = 1, stat: str = "best") -> xr.DataArray:
    """A gridded field at point locations, via ``poligrain.GridAtPoints``.

    ``da_points`` needs ``id``, projected ``x``/``y`` and ``lon``/``lat``:
    poligrain builds its output from ``da_point_data.lon`` even when it did
    every calculation in projected metres. ``stat="best"`` with ``nnear > 1``
    picks, per point, the neighbouring pixel closest to the observation -
    an optimistic but standard choice for radar-gauge comparison; the default
    ``nnear=1`` is the plain nearest pixel.
    """
    import poligrain as plg

    gap = plg.spatial.GridAtPoints(da_grid, da_points, nnear=nnear, stat=stat,
                                   use_lon_lat=False)
    return gap(da_grid, da_points)


def distance_to_network(ds_cml, x, y, chunk: int = 20000) -> np.ndarray:
    """Shortest distance (km) from each point to any link path.

    Point-to-segment, not point-to-midpoint: a CML measures along its whole
    line, so a point beside the middle of a long link is well covered even
    though both endpoints are far away. ``x``/``y`` can have any shape (a
    gauge list, a 2-D grid); the result has the same shape.
    """
    x0, y0, x1, y1 = (np.asarray(ds_cml[c], dtype=float).ravel()
                      for c in ("site_0_x", "site_0_y", "site_1_x", "site_1_y"))
    vx, vy = x1 - x0, y1 - y0
    len2 = np.maximum(vx * vx + vy * vy, 1e-9)

    px = np.asarray(x, dtype=float)
    py = np.asarray(y, dtype=float)
    flat_x, flat_y = px.ravel(), py.ravel()
    out = np.empty(flat_x.size)
    for s in range(0, flat_x.size, chunk):
        qx = flat_x[s:s + chunk, None]
        qy = flat_y[s:s + chunk, None]
        t = np.clip(((qx - x0) * vx + (qy - y0) * vy) / len2, 0.0, 1.0)
        out[s:s + chunk] = np.hypot(qx - (x0 + t * vx), qy - (y0 + t * vy)).min(axis=1)
    return (out / 1000.0).reshape(px.shape)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _paired(reference, estimate):
    """Align two DataArrays by coordinate and dimension order.

    Flattening a (cml_id, time) array against a (time, cml_id) one pairs
    values from different links and times and returns a correlation near
    zero rather than an error - the OpenMRG reference ``R`` is stored
    (sublink_id, cml_id, time) while its ``rsl`` is (time, sublink_id,
    cml_id). Plain arrays are passed through and must already match.
    """
    if isinstance(reference, xr.DataArray) and isinstance(estimate, xr.DataArray):
        if set(reference.dims) != set(estimate.dims):
            raise ValueError(f"cannot pair dims {reference.dims} with {estimate.dims}")
        reference, estimate = xr.align(reference, estimate, join="inner")
        estimate = estimate.transpose(*reference.dims)
        return reference.values, estimate.values
    ref, est = np.asarray(reference), np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {est.shape}")
    return ref, est


def rainfall_metrics(reference, estimate,
                     wet_threshold: float = WET_THRESHOLD_MM_H) -> dict:
    """Continuous and detection skill of ``estimate`` against ``reference``.

    Built on ``poligrain.validation``, whose threshold semantics differ
    between its two functions and are easy to get wrong:
    ``calculate_rainfall_metrics`` **drops** pairs below the threshold, while
    ``calculate_wet_dry_metrics`` keeps them and calls them dry. A percent
    bias computed with a 0.1 mm/h threshold therefore ignores all the light
    rain - which is where a CML baseline error lives. So the continuous
    metrics here use no threshold (the whole record) and ``wet_threshold``
    applies to detection only.

    Returns short, stable keys: ``r``, ``rmse``, ``mae``, ``pbias`` (%),
    ``ratio`` (estimate total / reference total), ``mcc``, ``tpr``, ``fpr``,
    ``n`` (valid pairs), ``mean_ref``, ``mean_est``.
    """
    import poligrain as plg

    reference, estimate = _paired(reference, estimate)
    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    ok = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[ok], est[ok]
    nan_row = {k: np.nan for k in ("r", "rmse", "mae", "pbias", "ratio", "mcc",
                                   "tpr", "fpr", "mean_ref", "mean_est")}
    if ref.size < 3:
        return nan_row | {"n": int(ref.size)}

    cont = plg.validation.calculate_rainfall_metrics(ref, est, 0.0, 0.0)
    det = plg.validation.calculate_wet_dry_metrics(ref, est, wet_threshold,
                                                   wet_threshold)
    total_ref = ref.sum()
    return {
        "r": float(cont["pearson_correlation_coefficient"]),
        "rmse": float(cont["root_mean_square_error"]),
        "mae": float(cont["mean_absolute_error"]),
        "pbias": float(cont["percent_bias"]),
        "ratio": float(est.sum() / total_ref) if total_ref > 0 else np.nan,
        "mcc": float(det["matthews_correlation_coefficient"]),
        "tpr": float(det["true_positive_ratio"]),
        "fpr": float(det["false_positive_ratio"]),
        "mean_ref": float(ref.mean()),
        "mean_est": float(est.mean()),
        "n": int(ref.size),
    }


def skill_table(estimates: dict, references: dict,
                metrics=("ratio", "r", "mcc")):
    """Score several per-link estimates against several references at once.

    ``estimates`` maps a name to a (time, cml_id) rain rate at the
    retrieval's own resolution. ``references`` maps a name to
    ``(reference, step)`` or ``(reference, step, label)``: the reference on
    (time, cml_id) - radar along links, gauges at links - and the step and
    timestamp convention to aggregate the estimate to before comparing.
    Returns a DataFrame, one row per estimate, one column per
    (reference, metric).

    >>> skill_table({"default": r1, "improved": r2},
    ...             {"gauge": (gauge_at_links, "15min"), "radar": (radar_path, "5min")})
    """
    import pandas as pd

    rows = {}
    for name, est in estimates.items():
        row = {}
        for ref_name, spec in references.items():
            ref, step, *rest = spec
            label = rest[0] if rest else "start"
            m = rainfall_metrics(ref, aggregate(est, step, label=label))
            row.update({(ref_name, k): m[k] for k in metrics})
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def aggregate(da: xr.DataArray, freq: str, label: str = "start") -> xr.DataArray:
    """Mean rain rate over ``freq`` bins; NaN only where a bin is all-NaN.

    Rates average, accumulations sum - everything in this module is a rate.

    ``label`` says which edge of a bin its timestamp names, and has to match
    the reference being compared against. ``"start"`` is pandas' default:
    ``[t, t + freq)`` stamped ``t``. ``"end"`` gives ``(t - freq, t]``
    stamped ``t``, the convention of OpenRainER's 15-minute gauge and radar
    accumulations. Aggregating a CML with the wrong one shifts it a whole
    bin against the reference; on OpenRainER that halves the apparent
    CML-gauge correlation (0.69 -> 0.35) without any error. Datasets declare
    theirs as ``example_data.DATASETS[key].accumulation_label``.
    """
    if label == "start":
        return da.resample(time=freq).mean(skipna=True)
    if label == "end":
        return da.resample(time=freq, closed="right", label="right").mean(skipna=True)
    raise ValueError(f"label must be 'start' or 'end', got {label!r}")
