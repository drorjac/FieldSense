"""Storm events from the radar, and a train/test split by event.

A model that learns from link data must be tested on storms it has not seen: two
minutes of the same storm are not independent, so a split by sample leaks. Events
are found by ``core.events.detect_events`` on the hourly radar over the network;
each event is extended by ``pad_after`` so its drying tail stays with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.events import detect_events


def radar_events(rate: xr.DataArray, links: xr.Dataset | None = None, margin_m: float = 5000.0,
                 wet_mm: float = 0.1, min_gap_h: int = 3, min_total_mm: float = 0.5,
                 label: str = "start") -> pd.DataFrame:
    """Events in a ``(time, y, x)`` radar rain rate (mm/h).

    With ``links`` (projected ``site_*_x/y``, as ``rate``'s ``x``/``y``) the radar is
    first cut to the network's bounding box plus ``margin_m``, so an event is rain
    over the links, not anywhere in the radar's view. The rate is averaged to
    hour-ending totals (mm) and passed to ``core.events.detect_events``, whose
    thresholds apply to the mean over the box. ``label`` is the radar's own
    time-stamp convention (``"start"`` or ``"end"``).
    """
    if links is not None:
        xs = np.r_[links.site_0_x.values, links.site_1_x.values]
        ys = np.r_[links.site_0_y.values, links.site_1_y.values]
        rate = rate.where((rate.x >= xs.min() - margin_m) & (rate.x <= xs.max() + margin_m)
                          & (rate.y >= ys.min() - margin_m) & (rate.y <= ys.max() + margin_m),
                          drop=True)
    closed = "left" if label == "start" else "right"
    hourly = rate.resample(time="1h", closed=closed, label="right").mean()
    hourly = xr.DataArray(hourly.transpose("time", "y", "x").values, dims=("time", "lat", "lon"),
                          coords={"time": hourly.time.values})
    ev = detect_events(hourly, wet_mm=wet_mm, min_gap_h=min_gap_h, min_total_mm=min_total_mm)
    for c in ("start", "end", "peak_hour"):              # detect_events writes strings
        ev[c] = pd.to_datetime(ev[c])
    return ev


def split_events(events: pd.DataFrame, test_every: int = 2, offset: int = 1) -> pd.DataFrame:
    """``events`` with a ``split`` column: every ``test_every``-th event (from ``offset``)
    is ``"test"``, the rest ``"train"``. Alternating keeps both halves spread over the
    record, so neither is all one weather type."""
    out = events.sort_values("start").reset_index(drop=True).copy()
    idx = np.arange(len(out))
    out["split"] = np.where((idx - offset) % test_every == 0, "test", "train")
    return out


def event_mask(times, events: pd.DataFrame, split: str | None = None,
               pad_before: str = "1h", pad_after: str = "3h") -> np.ndarray:
    """Boolean over ``times``: inside an event (of ``split``, if given), padded.

    The padding after an event keeps the drying tail with the storm that made it.
    """
    t = pd.DatetimeIndex(times)
    mask = np.zeros(t.size, bool)
    ev = events if split is None else events[events.split == split]
    for _, e in ev.iterrows():
        mask |= (t >= e.start - pd.Timedelta(pad_before)) & (t < e.end + pd.Timedelta(pad_after))
    return mask
