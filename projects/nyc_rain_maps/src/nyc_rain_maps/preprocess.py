"""Signal preprocessing: total loss, gap handling and aggregation.

OpenMesh has no transmitted-power record, so total loss is ``TL = TSL - RSL`` with a
constant TSL (0 dBm by default); only changes in TL matter because every method
subtracts a baseline.

Gap handling is where the two implementations differ most, and it matters: during
heavy rain a link can lose sync and report nothing, so a gap is *evidence of rain*, not
absence of data. Filling policies:

* :func:`fill_gaps_min_rsl` (implementation_2): every gap in the event window is set
  to the link's deepest fade (minimum RSL) in that window - rain or not.
* :func:`fill_gaps_gauge_gated` (implementation_1): gaps are filled only while nearby
  PWS gauges report rain, with the link's maximum (or 99th-percentile) attenuation;
  gaps in dry weather stay NaN (and end up as zero rain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import haversine_m


def total_loss(links: xr.Dataset, tsl: float = 0.0) -> xr.DataArray:
    """Total path loss ``TSL - RSL`` (dB), same shape as ``links.rsl``."""
    tl = tsl - links["rsl"]
    tl.name = "total_loss"
    tl.attrs = {"units": "dB", "tsl_assumed_dBm": tsl}
    return tl


def nan_mask(links: xr.Dataset) -> xr.DataArray:
    """True where the raw RSL is missing."""
    return links["rsl"].isnull().rename("nan_mask")


def fill_gaps_min_rsl(links: xr.Dataset) -> xr.Dataset:
    """implementation_2: replace every NaN with the link's minimum RSL in the window.

    The original gap positions are kept as ``links['nan_mask']``.
    """
    out = links.copy()
    out["nan_mask"] = nan_mask(links)
    out["rsl"] = links["rsl"].fillna(links["rsl"].min("time"))
    out["rsl"].attrs = dict(links["rsl"].attrs, gap_fill="min RSL over window (all gaps)")
    return out


def interpolate_short_gaps(links: xr.Dataset, max_gap: str = "5min") -> xr.Dataset:
    """Linear interpolation across gaps no longer than ``max_gap``."""
    out = links.copy()
    out["rsl"] = links["rsl"].interpolate_na("time", method="linear", max_gap=pd.Timedelta(max_gap))
    return out


def gauge_wet_reference(links: xr.Dataset, pws: xr.Dataset, radius_m: float = 5000.0,
                        bin_minutes: int = 15) -> xr.DataArray:
    """implementation_1's gauge reference: is it raining near each link?

    For every link, average the PWS 5-min rain of all stations within ``radius_m`` of
    the link midpoint, take the mean over consecutive ``bin_minutes`` bins (stamped at
    the bin start), and linearly interpolate that onto the link's 1-min time axis
    (0 outside the gauge record). Returns the interpolated reference (>0 = wet).
    """
    t_link = links.time.values
    t0 = pd.Timestamp(t_link[0]).floor(f"{bin_minutes}min")
    out = np.zeros((links.sizes["link"], t_link.size))
    for i, (lat, lon) in enumerate(zip(links.mid_lat.values, links.mid_lon.values)):
        d = haversine_m(lat, lon, pws.lat.values, pws.lon.values)
        near = pws["rain"].isel(station=np.flatnonzero(d <= radius_m))
        if near.sizes["station"] == 0:
            continue
        ref = near.mean("station", skipna=True).to_series()
        ref = ref.loc[t0:]
        binned = ref.resample(f"{bin_minutes}min", label="left", closed="left", origin=t0).mean()
        x = binned.index.values.astype("datetime64[s]").astype(float)
        y = np.nan_to_num(binned.values, nan=0.0)
        out[i] = np.interp(t_link.astype("datetime64[s]").astype(float), x, y, left=0.0, right=0.0)
    return xr.DataArray(out, dims=("link", "time"), coords={"link": links.link, "time": links.time},
                        name="gauge_reference", attrs={"radius_m": radius_m, "bin_minutes": bin_minutes})


def fill_gaps_gauge_gated(attenuation: xr.DataArray, wet_reference: xr.DataArray,
                          stat: str = "q99") -> xr.DataArray:
    """implementation_1: fill attenuation gaps only where the gauge reference is wet.

    ``stat``: ``"q99"`` (99th percentile of the link's non-NaN attenuation in the
    window; current implementation_1 code) or ``"max"`` (maximum; what produced the
    committed ``map1.pkl``/``map2.pkl``).
    """
    if stat == "max":
        fill = attenuation.max("time", skipna=True)
    elif stat == "q99":
        fill = attenuation.quantile(0.99, dim="time", skipna=True).drop_vars("quantile")
    else:
        raise ValueError("stat must be 'max' or 'q99'")
    wet_gap = attenuation.isnull() & (wet_reference > 0)
    out = attenuation.where(~wet_gap, fill)
    out.attrs = dict(attenuation.attrs, gap_fill=f"{stat} attenuation where gauges wet")
    return out


def min_max(links: xr.Dataset, window: str = "15min", interpolate_gap: str | None = "5min"
            ) -> xr.Dataset:
    """Min and max RSL over ``window`` (interval-ENDING labels), as ``rsl_min``/``rsl_max``.

    Mirrors the 15-min min/max aggregation of operational CML data (and of the nearby-
    link method). Short gaps are interpolated first if ``interpolate_gap`` is set.
    """
    src = interpolate_short_gaps(links, interpolate_gap) if interpolate_gap else links
    r = src["rsl"].resample(time=window, label="right", closed="right")
    out = xr.Dataset({"rsl_min": r.min(), "rsl_max": r.max()})
    return out.assign_coords({c: links[c] for c in links.coords if links[c].dims == ("link",)})
