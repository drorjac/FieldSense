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
``cnn``             Polz et al. (2020): a CNN trained on German links against
                    radar, run on 3-hour windows of 1-minute total loss
                    (pycomlink's model loader, PyTorch)

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
>>> from core.opensense import conventions as cv
from core.opensense import retrieval as rt, wet_dry
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

from core.opensense import conventions as cv
from core.opensense import retrieval as rt

WET_THRESHOLD_MM_H = 0.1

# Polz et al. (2020), AMT 13, 3835 - pinned to the commit that published it
# (BSD-3-Clause, github.com/jpolz/cml_wd_pytorch)
POLZ2020_MODEL = ("https://github.com/jpolz/cml_wd_pytorch/raw/"
                  "c346b8bcf830678e495a2a308b41891e2133d988/data/amt_model/pytorch_model_jit.pt")
CNN_WINDOW = 180        # minutes of context per prediction
CNN_TARGET = 150        # the prediction belongs to window start + 150 (pycomlink)


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

    Needs site lat/lon and a length in any declared unit. NaN
    during the first 6 hours (the 24 h maximum needs history) and for links
    with too few neighbours. Returned as float on the CML's time axis.
    """
    from pycomlink.processing.wet_dry import nearby_wetdry as nw

    rx = -(cml.tsl - cml.rsl) if "tsl" in cml else cml.rsl
    if "sublink_id" in rx.dims:
        pmin = rx.resample(time=f"{interval_min}min").min().mean("sublink_id")
    else:
        pmin = rx.resample(time=f"{interval_min}min").min()
    # normalized datasets carry length_km; raw OpenSense files only length,
    # in whatever units they declare
    if "length_km" in cml.coords:
        length = cml.length_km
    else:
        length = xr.DataArray(cv.to_km(cml.length), dims=cml.length.dims)
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


def cnn(cml: xr.Dataset, model: str = POLZ2020_MODEL, threshold: float | None = 0.82,
        batch_size: int = 4096, links_per_chunk: int = 16, max_gap_min: int = 5,
        net=None) -> xr.DataArray:
    """Polz et al. (2020) CNN wet/dry on 1-minute total loss.

    Each link's total loss (``TSL - RSL``, or ``-RSL``) is averaged to 1
    minute and its median removed; gaps up to ``max_gap_min`` are
    interpolated and longer ones set to the median, because the network has
    no notion of missing data. A prediction for minute ``t`` sees
    ``[t - 150, t + 30)``, as in pycomlink. The model takes two channels: a
    link with one sublink gets it twice, a link with more keeps its first two.

    Returns P(wet), or with ``threshold`` a 1/0 mask, as float on the CML's
    time axis; NaN where the link had no data in that minute or the window
    does not fit (the first 150 and last 29 minutes). ``net`` passes an
    already loaded model (tests; repeated calls).

    pycomlink's own ``cnn_wd`` is not used: under NumPy >= 2.4 it returns
    all-NaN predictions (a one-element array assigned to a scalar raises,
    and the error is swallowed), and it builds every window as a Python list,
    which does not fit in memory for a network over days.
    """
    import contextlib
    import io

    import torch

    if net is None:
        from pycomlink.processing.pytorch_utils.inference_utils import get_model
        with contextlib.redirect_stdout(io.StringIO()):    # it prints on every load
            net = get_model(model)

    loss = (cml.tsl - cml.rsl) if "tsl" in cml else -cml.rsl
    if "sublink_id" not in loss.dims:
        loss = loss.expand_dims(sublink_id=[0])
    loss = loss.transpose("time", "cml_id", "sublink_id").resample(time="1min").mean()
    if loss.sizes["sublink_id"] == 1:
        loss = xr.concat([loss, loss], dim="sublink_id")
    loss = loss.isel(sublink_id=slice(0, 2))
    have = loss.notnull().any("sublink_id")
    normed = (loss - loss.median("time")).interpolate_na(
        "time", max_gap=np.timedelta64(max_gap_min, "m")).fillna(0.0)

    x = normed.values.astype(np.float32)                   # (time, cml, 2)
    n_t, n_links = x.shape[:2]
    prob = np.full((n_t, n_links), np.nan)
    if n_t >= CNN_WINDOW:
        for lo in range(0, n_links, links_per_chunk):
            chunk = x[:, lo:lo + links_per_chunk]
            # (windows, links, 2, 180) -> (links * windows, 180, 2)
            win = np.lib.stride_tricks.sliding_window_view(chunk, CNN_WINDOW, axis=0)
            win = np.ascontiguousarray(win.transpose(1, 0, 3, 2)).reshape(-1, CNN_WINDOW, 2)
            out = []
            with torch.no_grad():
                for b in range(0, len(win), batch_size):
                    out.append(net(torch.from_numpy(win[b:b + batch_size])).numpy().ravel())
            p = np.concatenate(out).reshape(chunk.shape[1], -1).T   # (windows, links)
            prob[CNN_TARGET:CNN_TARGET + len(p), lo:lo + chunk.shape[1]] = p

    out = xr.DataArray(prob, dims=("time", "cml_id"),
                       coords={"time": loss.time, "cml_id": loss.cml_id}).where(have)
    if threshold is not None:
        out = (out > threshold).astype(float).where(out.notnull())
    return out.reindex(time=cml.time, method="ffill", tolerance=np.timedelta64(59, "s"))


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
