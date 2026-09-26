"""
Point sensors over the OpenMesh network: PWS and ASOS, summarized and plotted.

The analysis ``read_pws_sample.ipynb`` and ``wu_pipeline.ipynb`` wrote into
cells: per-station totals and rain statistics, accumulation over time, and
which stations can be trusted. Works on any OpenSense point dataset with a
rain rate ``R`` on (time, id) - ``core.opensense.example_data`` adds one.

    from stations import totals, plot_accumulation
    table = totals(data["pws"])
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from core import viz_style as vs


def _step_hours(ds: xr.Dataset) -> xr.DataArray:
    """Hours each sample stands for; PWS report irregularly, so per sample."""
    t = ds.time.values
    dt = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
    dt = np.append(dt, np.median(dt))
    # a reporting gap is not rain time: cap each step at 3x the typical one
    return xr.DataArray(np.minimum(dt, 3 * np.median(dt)), dims="time",
                        coords={"time": ds.time})


def accumulation(ds: xr.Dataset) -> xr.DataArray:
    """Cumulative rain (mm) per station from the rate ``R`` (mm/h)."""
    return (ds.R.fillna(0.0) * _step_hours(ds)).cumsum("time")


def totals(ds: xr.Dataset, wet_threshold: float = 0.1) -> pd.DataFrame:
    """One row per station: total, wet hours, peak rate, share of missing samples.

    Sorted by total. A station reporting far less than its neighbours, or
    mostly missing, is the first thing to look at before using it as a
    reference.
    """
    hours = _step_hours(ds)
    rate = ds.R.transpose("time", "id")
    table = pd.DataFrame({
        "total_mm": (rate.fillna(0.0) * hours).sum("time").values,
        "wet_hours": ((rate >= wet_threshold) * hours).sum("time").values,
        "peak_mm_h": rate.max("time").values,
        "missing": rate.isnull().mean("time").values,
    }, index=pd.Index(ds.id.values, name="station"))
    median = np.nanmedian(table.total_mm)
    table["vs_median"] = table.total_mm / median if median > 0 else np.nan
    return table.sort_values("total_mm", ascending=False)


def plot_accumulation(ds: xr.Dataset, top: int = 8, title: str = ""):
    """Cumulative rain of the ``top`` wettest stations, the rest in grey."""
    acc = accumulation(ds)
    order = acc.isel(time=-1).to_series().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 4))
    for sid in order[top:]:
        ax.plot(acc.time, acc.sel(id=sid), color=vs.GRIDLINE, lw=0.8)
    palette = list(vs.SERIES.values())
    for k, sid in enumerate(order[:top]):
        color = palette[k] if k < len(palette) else vs.INK_SECONDARY
        ax.plot(acc.time, acc.sel(id=sid), lw=1.4, color=color,
                ls="-" if k < len(palette) else "--", label=str(sid))
    ax.set(ylabel="accumulated rain (mm)", title=title)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    return fig


def plot_totals(table: pd.DataFrame, title: str = ""):
    """Station totals as bars, with the network median for reference."""
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(table)), 3.6))
    ax.bar(range(len(table)), table.total_mm, color=vs.SERIES["stratiform"], width=0.8)
    ax.axhline(np.nanmedian(table.total_mm), color=vs.INK_PRIMARY, ls="--", lw=1,
               label="median")
    ax.set_xticks(range(len(table)), table.index, rotation=90, fontsize=7)
    ax.set(ylabel="total (mm)", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    return fig
