"""The network with the KNMI gauges, and the wettest day mapped from links against the gauges."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection

from core.maps import wet_area as wa
from core.maps.idw import idw_map
from core.viz_style import CMAP_RAIN, SERIES, use_style

from .settings import CACHE, END, FIGURES_DIR, IDW, P_CUT, START, WET_MM
from .study import grid, hourly, retrieve

STORM_DAY = CACHE / "storm_day.nc"
NOKIA, NEC = SERIES["stratiform"], SERIES["convective"]
LABEL = {"NOKIA": "Nokia", "NEC": "NEC"}


def wettest_day(gauges: xr.DataArray) -> pd.Timestamp:
    """The day (hours ending 01-24 UTC) with the largest gauge-mean total."""
    d = gauges.resample(time="1D", closed="right", label="left").sum(min_count=20)
    return pd.Timestamp(d.mean("station").idxmax("time").item())


def storm_day(force: bool = False) -> xr.Dataset:
    """Daily link maps (IDW, wet-masked IDW) and gauge totals of the wettest day; cached."""
    if STORM_DAY.exists() and not force:
        return xr.open_dataset(STORM_DAY).load()
    links, gauges = hourly()
    day = wettest_day(gauges)
    hrs = slice(day + pd.Timedelta("1h"), day + pd.Timedelta("1D"))
    chunk, g = links.sel(time=hrs), grid()
    m = idw_map(chunk, g, **IDW).sum("time", min_count=20)
    mm = wa.masked_idw(chunk, g, WET_MM, P_CUT, **IDW).sum("time", min_count=20)
    gs = gauges.sel(time=hrs).sum("time", min_count=20)
    ds = xr.Dataset({"idw": m, "masked": mm})
    ds["gauge_total"] = gs.drop_vars([c for c in gs.coords if c not in ("station",)])
    ds = ds.assign_coords(gauge_lat=("station", gauges.lat.values),
                          gauge_lon=("station", gauges.lon.values))
    ds.attrs["day"] = str(day.date())
    ds.to_netcdf(STORM_DAY)
    return ds


def plot_network(ax, sub: xr.Dataset, gauges: xr.DataArray) -> None:
    """Link paths coloured by vendor, KNMI gauges as triangles."""
    cols =["path", "vendor", "site_0_lon", "site_0_lat", "site_1_lon", "site_1_lat"]
    paths = pd.DataFrame({c: sub[c].values for c in cols}).drop_duplicates("path")
    for vendor, colour in (("NOKIA", NOKIA), ("NEC", NEC)):
        sel = (paths.vendor == vendor).to_numpy()
        p = paths[sel]
        segs = np.stack([p[["site_0_lon", "site_0_lat"]].to_numpy(),
                         p[["site_1_lon", "site_1_lat"]].to_numpy()], axis=1)
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=0.8,
                                         label=f"{LABEL[vendor]} ({sel.sum()} paths)"))
    ax.scatter(gauges.lon, gauges.lat, marker="^", s=28, c="k", zorder=3,
               label=f"KNMI gauges ({gauges.sizes['station']})")
    ax.set_xlim(3.3, 7.3)
    ax.set_ylim(50.7, 53.6)
    ax.set_aspect(1 / np.cos(np.radians(52.2)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def plot_storm(axes, ds: xr.Dataset, vmax: float | None = None) -> None:
    """The day's link maps with the gauge totals drawn on the same colour scale."""
    vmax = vmax or float(np.nanpercentile(np.r_[ds.gauge_total.values, ds.idw.values.ravel()], 99))
    for ax, name, title in zip(axes, ("idw", "masked"), ("IDW", "IDW with wet mask")):
        im = ax.pcolormesh(ds.lon, ds.lat, ds[name], cmap=CMAP_RAIN, vmin=0, vmax=vmax,
                           shading="nearest")
        ax.scatter(ds.gauge_lon, ds.gauge_lat, c=ds.gauge_total, cmap=CMAP_RAIN, vmin=0,
                   vmax=vmax, s=60, edgecolors="k", linewidths=0.8, zorder=3)
        ax.set_aspect(1 / np.cos(np.radians(52.2)))
        ax.set_title(f"{title}, {ds.attrs['day']} (circles: KNMI gauges)", fontsize=9)
        ax.set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    plt.colorbar(im, ax=axes, shrink=0.8, label="daily rain (mm)")


def write() -> None:
    use_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sub = retrieve()
    _, gauges = hourly()
    fig, ax = plt.subplots(figsize=(6, 6.5))
    plot_network(ax, sub, gauges)
    ax.set_title(f"Dutch CML network, {START.date()} to {(END - pd.Timedelta('1D')).date()}",
                 fontsize=10)
    fig.savefig(FIGURES_DIR / "network.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    ds = storm_day()
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    plot_storm(axes, ds)
    fig.savefig(FIGURES_DIR / "storm_day.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
