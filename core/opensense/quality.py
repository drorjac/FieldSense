"""
Quality control of raw CML signals before retrieval.

``censored_at_floor``
    samples where the receiver sits at its sensitivity floor. There the
    recorded loss is a lower bound, not a measurement - the true attenuation
    may be anything above it - so no rain rate can be inverted from it.

Found the hard way. On OpenRainER, 26 September 2021, link 54 drops to
-100 dBm and stays there for ~16 hours. The rolling-std wet/dry classifier
calls the flat plateau dry and the baseline climbs onto it, so the default
chain returns ~0 - right by accident. The nearby-link mask sees wet
neighbours, correctly calls the link wet, holds the baseline at the
pre-event level, and turns 60 dB of outage into 180 mm/h for seven hours.
One such link dominates every pooled score. The better mask did not cause
the error; it removed the accident that had been hiding it.

Receivers clip at different levels - OpenMRG and OpenMesh at -90 dBm,
OpenRainER's at -100 to -103 dBm - so no absolute threshold works. What
generalizes is the shape: a real fade touches its deepest point briefly,
a clipped receiver lingers there. A sample is censored when its sublink

* is within ``margin_db`` of its own minimum over the window,
* that minimum is at least ``depth_db`` below its median level (so a flat,
  never-raining sublink sitting at its usual level is not flagged), and
* stays there for at least ``min_minutes`` consecutively.

On the example subsets this flags 3 of 728 OpenMRG sublinks, 24 of 302
OpenRainER sublinks, and none of OpenMesh's 225.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _long_runs(flag: np.ndarray, n: int) -> np.ndarray:
    """True where ``flag`` is part of a run of at least ``n`` along axis 0."""
    flag = np.asarray(flag, dtype=bool)
    out = np.zeros_like(flag)
    for j in range(flag.shape[1]):
        col = flag[:, j]
        if not col.any():
            continue
        # run boundaries from the padded difference
        edges = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            if e - s >= n:
                out[s:e, j] = True
    return out


def censored_at_floor(rsl: np.ndarray, interval_s: float, margin_db: float = 1.0,
                      depth_db: float = 20.0, min_minutes: float = 10.0) -> np.ndarray:
    """Boolean (time, sublink) mask of samples at the receiver floor.

    ``rsl`` is received power in dBm, (time, sublink). See the module
    docstring for the criteria; defaults were checked on OpenMRG, OpenRainER
    and OpenMesh.
    """
    rsl = np.asarray(rsl, dtype=float)
    with np.errstate(invalid="ignore"), _quiet():
        floor = np.nanmin(rsl, axis=0)
        median = np.nanmedian(rsl, axis=0)
    deep = floor <= median - depth_db
    near = (rsl <= (floor + margin_db)[None, :]) & deep[None, :]
    n = max(2, int(round(min_minutes * 60.0 / interval_s)))
    return _long_runs(near, n)


def censored_dataset(ds: xr.Dataset, **kwargs) -> xr.DataArray:
    """``censored_at_floor`` on an OpenSense CML dataset's ``rsl``.

    Returns a boolean DataArray on the same dimensions as ``ds.rsl``.
    """
    from core.opensense.retrieval import sampling_interval_s

    rsl = ds.rsl
    other = [d for d in rsl.dims if d != "time"]
    flat = rsl.transpose("time", *other)
    shape = flat.shape
    mask = censored_at_floor(flat.values.reshape(shape[0], -1),
                             sampling_interval_s(ds.time), **kwargs)
    return xr.DataArray(mask.reshape(shape), dims=flat.dims,
                        coords={d: flat[d] for d in flat.dims}).transpose(*rsl.dims)


class _quiet:
    """Silence all-NaN slice warnings for sublinks with no data."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
