"""
Wet/dry masks for the CML retrieval, beyond the rolling-std default.

The wet/dry step decides which samples the dry-weather baseline may learn
from. ``retrieval.retrieve`` uses the rolling standard deviation of the
signal (Schleiss & Berne 2010) unless given a mask; this module builds the two
alternatives the OpenSense ecosystem provides, both returned per link on the
CML's own time axis, ready for ``retrieval.retrieve_dataset(ds, wet=mask)``:

``from_radar``      radar path-averaged along each link (poligrain)
``nearby_links``    Overeem et al. (2016): a link is wet when it and its
                    neighbours drop together (pycomlink)

Both are NaN where they cannot decide (no radar coverage, too few
neighbours, not enough history). ``fill_undecided`` resolves those samples
with the rolling-std classifier so every sample gets a flag.

What ``retrieval_benchmark.py`` found on 8 days of OpenMRG: a better mask on
its own makes the retrieval over-read *more*, because the rolling-std mask
lets rain leak into the baseline and that leak had been hiding a missing
wet-antenna correction. A better mask pays off only together with a
wet-antenna model - see the project README.

Example
-------
>>> from core.opensense import retrieval as rt, wet_dry
>>> mask = wet_dry.nearby_links(cml)
>>> out = rt.retrieve_dataset(cml, rt.RetrievalConfig.for_interval(10,
...                           waa_model="pastorek2021"),
...                           wet=wet_dry.fill_undecided(mask, cml))
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from core.opensense import retrieval as rt

WET_THRESHOLD_MM_H = 0.1


def from_radar(radar_path: xr.DataArray, cml_time,
               threshold: float = WET_THRESHOLD_MM_H,
               dilate_steps: int = 1) -> xr.DataArray:
    """Wet where the radar path average reaches ``threshold``.

    ``radar_path`` is (time, cml_id), from ``evaluation.radar_along_links``.
    The mask is dilated by ``dilate_steps`` radar steps either side, because
    the baseline must not learn from the edges of a rain spell, and a
    5-minute radar and a 10 s link need not share a timestamp convention.
    Returned as float (1 wet, 0 dry, NaN no radar) on ``cml_time``.
    """
    have = np.isfinite(radar_path)
    wet = (radar_path >= threshold).astype(float).where(have)
    if dilate_steps:
        # dilate, but never invent a flag where the radar had no data
        wet = wet.rolling(time=2 * dilate_steps + 1, center=True,
                          min_periods=1).max().where(have)
    return wet.transpose("time", "cml_id").reindex(time=cml_time, method="ffill")


def nearby_links(cml: xr.Dataset, radius_km: float = 15.0,
                 interval_min: int = 15, **kwargs) -> xr.DataArray:
    """Overeem et al. (2016) nearby-link wet/dry, via pycomlink.

    Designed for the 15-minute min/max received power operators archive;
    ``pmin`` is formed here from the raw signals the same way (per link, the
    mean of its sublinks' minimum of ``RSL - TSL``). A link is wet when it
    *and* the median of its neighbours within ``radius_km`` fall well below
    their 24-hour maximum: rain is spatially coherent, hardware glitches are
    not. ``kwargs`` pass through to ``pycomlink``'s ``nearby_wetdry``
    (thresholds, ``min_links``).

    Needs ``length_km`` and site lat/lon, i.e. a normalized dataset. NaN
    during the first 6 hours (the 24 h maximum needs history) and for links
    with too few neighbours. Returned as float on the CML's time axis.
    """
    from pycomlink.processing.wet_dry import nearby_wetdry as nw

    rx = -(cml.tsl - cml.rsl) if "tsl" in cml else cml.rsl
    if "sublink_id" in rx.dims:
        pmin = rx.resample(time=f"{interval_min}min").min().mean("sublink_id")
    else:
        pmin = rx.resample(time=f"{interval_min}min").min()
    length = cml.length_km
    if "sublink_id" in length.dims:
        length = length.isel(sublink_id=0, drop=True)
    pmin = pmin.transpose("cml_id", "time").assign_coords(
        length=("cml_id", np.asarray(length)))
    geo = cml.isel(time=0, drop=True)
    # pycomlink wraps both loops in a tqdm bar per call; a library called
    # from notebooks and batch jobs should not print, so it is swapped out
    # for the duration of the call.
    tqdm_orig, nw.tqdm = nw.tqdm, (lambda it, *a, **k: it)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist = nw.calc_distance_between_cml_endpoints(
            cml_ids=geo.cml_id.values,
            site_a_latitude=geo.site_0_lat.values,
            site_a_longitude=geo.site_0_lon.values,
            site_b_latitude=geo.site_1_lat.values,
                site_b_longitude=geo.site_1_lon.values)
            wet, _ = nw.nearby_wetdry(pmin, dist, radius=radius_km,
                                      interval=interval_min, **kwargs)
    finally:
        nw.tqdm = tqdm_orig
    return (wet.transpose("time", "cml_id").astype(float)
            .reindex(time=cml.time, method="ffill"))


def fill_undecided(mask: xr.DataArray, cml: xr.Dataset,
                   cfg: rt.RetrievalConfig | None = None) -> xr.DataArray:
    """Resolve NaN samples of a link mask with the rolling-std classifier.

    Returns a boolean (time, cml_id, sublink_id) mask - or (time, cml_id)
    for datasets without sublinks - for ``retrieve_dataset(wet=...)``.
    """
    cfg = cfg or rt.RetrievalConfig.for_interval(rt.sampling_interval_s(cml.time))
    dims = ("time", "cml_id", "sublink_id") if "sublink_id" in cml.dims \
        else ("time", "cml_id")
    tsl = cml.tsl.transpose(*dims).values if "tsl" in cml else None
    loss = rt.total_loss_from(cml.rsl.transpose(*dims).values, tsl)
    shape = loss.shape
    flat = pd.DataFrame(loss.reshape(shape[0], -1)).ffill().bfill().to_numpy()
    fallback = rt.wet_dry_rolling_std(flat, cfg.wet_window,
                                      cfg.wet_threshold_db).reshape(shape)

    m = mask.transpose("time", "cml_id").values.astype(float)
    if len(dims) == 3:
        m = np.repeat(m[:, :, None], shape[2], axis=2)
    out = np.where(np.isfinite(m), m > 0.5, fallback)
    return xr.DataArray(out, dims=dims,
                        coords={d: cml[d] for d in dims})
