"""
Scores at the gauges, as ``4_analysis``: poligrain's ``calculate_rainfall_metrics`` on all
gauge-hours pooled, with a 0.2 mm threshold on both reference and estimate (pairs where
both are below it are left out, a value below it on one side counts as 0).

On top of the notebook's tables: intensity classes of the gauge value, per-gauge RMSE and
correlation over time (as its PCC/RMSE maps), and gauges grouped by distance to the
nearest link path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from radar_adjustment.settings import CLASSES, DISTANCE_BANDS_KM, FIELDS, THRESHOLD

KEEP = {"pearson_correlation_coefficient": "pcc", "root_mean_square_error": "rmse",
        "mean_absolute_error": "mae", "percent_bias": "pbias",
        "coefficient_of_variation": "cv", "reference_mean_rainfall": "ref_mean",
        "estimate_mean_rainfall": "est_mean", "N_all": "n_all", "N_nan": "n_nan"}


def metrics(ref, est, thresh: float = THRESHOLD) -> dict:
    import poligrain as plg
    ref, est = np.asarray(ref, float).ravel(), np.asarray(est, float).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = plg.validation.calculate_rainfall_metrics(reference=ref, estimate=est,
                                                      ref_thresh=thresh, est_thresh=thresh)
    out = {KEEP[k]: float(v) for k, v in m.items() if k in KEEP}
    ok = np.isfinite(ref) & np.isfinite(est) & ~((ref < thresh) & (est < thresh))
    out["n_pairs"] = int(ok.sum())
    return out


def radar_at_gauges(rad: xr.DataArray, gauges: xr.DataArray) -> xr.DataArray:
    import poligrain as plg
    g = plg.spatial.GridAtPoints(da_gridded_data=rad.isel(time=0), da_point_data=gauges.isel(time=0),
                                 nnear=1, stat="best")
    return g(da_gridded_data=rad, da_point_data=gauges).transpose("id", "time")


def load_product(version: str, network: str, checks: str, tag: str) -> xr.DataArray | None:
    files = sorted((FIELDS / version / network / checks).glob(f"{tag}_????-??.nc"))
    if not files:
        return None
    parts = [xr.open_dataset(f).at_gauges.load() for f in files]
    return xr.concat(parts, "time").sortby("time")


def distance_to_links_km(gauges: xr.DataArray, cml: xr.DataArray) -> pd.Series:
    """Distance (km) from each gauge to the nearest link path, in the UTM plane."""
    px, py = gauges.x.values[:, None], gauges.y.values[:, None]
    ax, ay = cml.site_0_x.values[None, :], cml.site_0_y.values[None, :]
    bx, by = cml.site_1_x.values[None, :], cml.site_1_y.values[None, :]
    dx, dy = bx - ax, by - ay
    L2 = np.where(dx ** 2 + dy ** 2 > 0, dx ** 2 + dy ** 2, 1.0)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    d = np.hypot(px - (ax + t * dx), py - (ay + t * dy)).min(axis=1)
    return pd.Series(d / 1000, index=gauges.id.values, name="dist_km")


def score_products(products: dict, ref: xr.DataArray, dist_km: pd.Series | None = None) -> dict:
    """``products``: {name: (id, time) at gauges}. Returns the notebook's tables and ours."""
    overall, classes, per_gauge, bands = [], [], [], []
    R = ref.transpose("id", "time")
    for name, est in products.items():
        E = est.transpose("id", "time").reindex(id=R.id, time=R.time)
        overall.append({"product": name, **metrics(R.values, E.values)})
        for lo, hi in CLASSES:
            inc = (R >= lo) & (R < hi)
            r, e = R.where(inc), E.where(inc)
            classes.append({"product": name, "class": f"[{lo:g},{hi:g})" if hi < 1000 else f">={lo:g}",
                            "n": int(np.isfinite(r.values).sum()), **metrics(r.values, e.values)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmse = np.sqrt(((R - E) ** 2).mean("time"))
            pcc = xr.corr(R, E, dim="time")
        for i, gid in enumerate(R.id.values):
            per_gauge.append({"product": name, "id": gid, "rmse": float(rmse[i]), "pcc": float(pcc[i]),
                              "lon": float(R.lon[i]), "lat": float(R.lat[i]),
                              "dist_km": float(dist_km.loc[gid]) if dist_km is not None else np.nan})
        if dist_km is not None:
            edges = DISTANCE_BANDS_KM
            for lo, hi in zip(edges[:-1], edges[1:]):
                ids = dist_km.index[(dist_km >= lo) & (dist_km < hi)]
                if len(ids) == 0:
                    continue
                band = f"{lo:g}-{hi:g} km" if hi < 1000 else f">{lo:g} km"
                bands.append({"product": name, "band": band, "n_gauges": len(ids),
                              **metrics(R.sel(id=ids).values, E.sel(id=ids).values)})
    return {"overall": pd.DataFrame(overall), "classes": pd.DataFrame(classes),
            "per_gauge": pd.DataFrame(per_gauge), "bands": pd.DataFrame(bands)}
