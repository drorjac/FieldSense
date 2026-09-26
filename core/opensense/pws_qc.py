"""
Quality control for personal weather stations (PWS), with pypwsqc.

PWS are dense and free, and a fraction of them is wrong in ways a single
station cannot reveal: a clogged funnel reports zeros through a storm, a
counter reset logs 100 mm in five minutes. The PWSQC filters of de Vos et al.
(2019), ported to Python as ``pypwsqc``, judge each station against its
neighbours:

``fz``    faulty zeros - a run of zeros while the neighbours report rain
``hi``    high influx - far more than the neighbours, or > ``hi_thres_b``
          mm per step when they are nearly dry
``so``    station outlier - a rolling correlation with the neighbours below
          ``so_gamma`` (needs a long record: 28 days by default)
``rate``  added here, not in pypwsqc: a published ``rainfall_rate`` while the
          measured amount is zero. On OpenMesh one station reports 192 mm/h
          for 15 minutes after a gap with no rain in its bucket.

The filters run on per-step *amounts* on a regular axis (``regularize``),
which is what the bucket measures; flags are 1 (flagged), 0 (passed) and -1
(could not be evaluated: too few neighbours reporting, or no history yet).

    from core.opensense import example_data, pws_qc
    pws = example_data.load("openmesh", "20d", components=["pws"])["pws"]
    qc = pws_qc.flag(pws, so_evaluation_period=2016, so_mmatch=30)
    pws_qc.summary(qc)          # one row per station
    qc.rainfall_qc              # amounts with every flagged step removed

Defaults are those of pypwsqc's documentation examples (5-minute data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

FLAGS = ("fz_flag", "hi_flag", "so_flag", "rate_flag")


def regularize(pws: xr.Dataset, step: str = "5min") -> xr.Dataset:
    """Per-step amounts on a regular axis, as ``rainfall`` (id, time), in mm.

    PWS report irregularly (OpenMesh: every ~5.2 min); the filters count
    steps, so they need a fixed one. A step with no report is NaN, not 0.
    """
    # OpenSense files name the per-step amount rainfall_amount; the Amsterdam
    # set (the one PWSQC was built on) calls it rainfall
    name = "rainfall_amount" if "rainfall_amount" in pws else "rainfall"
    amount = pws[name].transpose("id", "time")
    out = xr.Dataset({"rainfall": amount.resample(time=step).sum(min_count=1)})
    for c in ("x", "y", "lon", "lat"):
        if c in pws.coords:
            out.coords[c] = pws[c]
    if "rainfall_rate" in pws:
        out["rate_max"] = pws.rainfall_rate.transpose("id", "time").resample(time=step).max()
    return out


def distances(ds: xr.Dataset) -> xr.DataArray:
    """Station-to-station distance (m) on projected x/y, poligrain's layout."""
    import poligrain as plg

    return plg.spatial.calc_point_to_point_distances(ds, ds)


def neighbour_reference(ds: xr.Dataset, dist: xr.DataArray,
                        max_distance_m: float = 10e3) -> xr.Dataset:
    """Add ``reference`` (neighbours' median) and ``nbrs_not_nan`` (how many report).

    Neighbours are the other stations within ``max_distance_m``; the station
    itself never counts toward its own reference.
    """
    near = (dist < max_distance_m) & (dist > 0)
    rain = ds.rainfall
    ref, count = [], []
    for sid in ds.id.values:
        ids = dist.id_neighbor.values[near.sel(id=sid).values]
        sub = rain.sel(id=ids)
        if len(ids):
            ref.append(sub.median("id"))
            count.append(sub.notnull().sum("id"))
        else:                                   # isolated: nothing to compare with
            ref.append(xr.full_like(rain.isel(id=0, drop=True), np.nan))
            count.append(xr.zeros_like(rain.isel(id=0, drop=True), dtype=int))
    ds = ds.copy()
    ds["reference"] = xr.concat(ref, dim="id").assign_coords(id=ds.id).transpose("id", "time")
    ds["nbrs_not_nan"] = xr.concat(count, dim="id").assign_coords(id=ds.id).transpose("id", "time")
    return ds


def rate_flag(ds: xr.Dataset, min_amount_mm: float = 0.5) -> xr.DataArray:
    """1 where the published rate implies ``min_amount_mm`` in a step the bucket saw none of.

    A tipping bucket records in 0.254 mm tips, so light rain legitimately
    leaves a 5-minute step at zero (1 mm/h is a third of a tip); the default
    asks for two tips' worth before calling the rate wrong.
    """
    if "rate_max" not in ds:
        return xr.zeros_like(ds.rainfall, dtype=int)
    step_h = pd.Timedelta(ds.attrs.get("step", "5min")).total_seconds() / 3600.0
    bad = (ds.rate_max * step_h >= min_amount_mm) & (ds.rainfall.fillna(0.0) == 0.0)
    return xr.where(ds.rate_max.isnull(), -1, bad.astype(int))


def flag(pws: xr.Dataset, step: str = "5min", max_distance_m: float = 10e3,
         nint: int = 6, n_stat: int = 5, hi_thres_a: float = 0.4,
         hi_thres_b: float = 10.0, so_evaluation_period: int = 8064,
         so_mmatch: int = 200, so_gamma: float = 0.15,
         rate_min_amount_mm: float = 0.5) -> xr.Dataset:
    """Run FZ, HI, SO and the rate check; return flags and ``rainfall_qc``.

    ``so_evaluation_period`` and ``so_mmatch`` are in steps: the default
    28 days needs a record longer than that - for a 20-day subset use about
    a week (2016) and far fewer matching wet steps (tens), or read the SO
    flags as "not evaluated" (-1).
    """
    from pypwsqc import flagging

    ds = regularize(pws, step)
    ds.attrs["step"] = step
    dist = distances(ds)
    ds = neighbour_reference(ds, dist, max_distance_m)
    ds = flagging.fz_filter(ds, nint=nint, n_stat=n_stat)
    ds = flagging.hi_filter(ds, hi_thres_a=hi_thres_a, hi_thres_b=hi_thres_b,
                            nint=nint, n_stat=n_stat)
    # so_filter writes into these rather than creating them (pypwsqc 0.2.1)
    ds["so_flag"] = xr.full_like(ds.rainfall, -1.0)
    ds["median_corr_nbrs"] = xr.full_like(ds.rainfall, np.nan)
    if ds.sizes["time"] > so_evaluation_period:
        ds = flagging.so_filter(ds, distance_matrix=dist,
                                evaluation_period=so_evaluation_period, mmatch=so_mmatch,
                                gamma=so_gamma, n_stat=n_stat, max_distance=max_distance_m)
    ds["so_flag"] = ds.so_flag.fillna(-1).astype(int)
    ds["rate_flag"] = rate_flag(ds, rate_min_amount_mm)

    flagged = sum((ds[f] == 1) for f in FLAGS) > 0
    ds["rainfall_qc"] = ds.rainfall.where(~flagged)
    ds.attrs.update(qc="pypwsqc FZ/HI/SO + rate-amount check", step=step,
                    max_distance_m=max_distance_m)
    return ds


def summary(qc: xr.Dataset) -> pd.DataFrame:
    """One row per station: missing share, share of steps each flag fired, totals."""
    table = pd.DataFrame({
        "missing": qc.rainfall.isnull().mean("time").values,
        **{f.removesuffix("_flag"): (qc[f] == 1).mean("time").values for f in FLAGS},
        "so_evaluated": (qc.so_flag != -1).mean("time").values,
        "total_mm": qc.rainfall.sum("time").values,
        "total_qc_mm": qc.rainfall_qc.sum("time").values,
    }, index=pd.Index(qc.id.values, name="station"))
    return table.sort_values("total_qc_mm")


def usable(qc: xr.Dataset, max_missing: float = 0.5, max_flagged: float = 0.05) -> list:
    """Stations fit to serve as references: mostly present, rarely flagged."""
    t = summary(qc)
    flagged = t[[f.removesuffix("_flag") for f in FLAGS]].sum(axis=1)
    return t.index[(t.missing <= max_missing) & (flagged <= max_flagged)].tolist()
