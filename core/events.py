"""Precipitation events in an hourly radar record.

An hour is wet when the domain-mean accumulation reaches ``wet_mm``; wet hours separated
by fewer than ``min_gap_h`` dry hours belong to the same event; events whose domain-mean
total is below ``min_total_mm`` are dropped. Hours where less than ``min_valid_fraction``
of the domain has data count as dry, so archive gaps never merge or invent events.

    from core.events import detect_events
    events = detect_events(radar_hourly)          # one row per event

``projects/nyc_rain_maps`` adds precipitation type (MRMS PrecipFlag + ASOS) on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

COLUMNS = ["event_id", "start", "end", "duration_h", "wet_hours", "total_mm", "max_cell_mm",
           "peak_hour", "peak_hourly_mm", "max_cell_hourly_mm"]


def detect_events(hourly: xr.DataArray, wet_mm: float = 0.1, min_gap_h: int = 6,
                  min_total_mm: float = 1.0, min_valid_fraction: float = 0.8) -> pd.DataFrame:
    """Events of an hour-ending ``(time, lat, lon)`` record in mm; ``start`` is the start
    of the first wet hour, ``end`` the end of the last."""
    valid = hourly.notnull().mean(("lat", "lon"))
    mean = hourly.mean(("lat", "lon")).where(valid >= min_valid_fraction)
    s = mean.to_series()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full)                         # archive gaps become NaN (treated as dry)
    wet = (s >= wet_mm).fillna(False).values
    idx = np.flatnonzero(wet)
    if idx.size == 0:
        return pd.DataFrame(columns=COLUMNS)
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] - 1 < min_gap_h:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    rows = []
    for g in groups:
        t_first, t_last = full[g[0]], full[g[-1]]          # hour-ending labels
        sub = hourly.sel(time=slice(t_first, t_last))
        total_map = sub.sum("time", min_count=1)
        seg = s.loc[t_first:t_last]
        total = float(seg.sum())
        if total < min_total_mm:
            continue
        start = t_first - pd.Timedelta("1h")
        rows.append({"event_id": f"{start:%Y%m%dT%H}", "start": str(start), "end": str(t_last),
                     "duration_h": int(len(seg)), "wet_hours": int((seg >= wet_mm).sum()),
                     "total_mm": round(total, 2), "max_cell_mm": round(float(total_map.max()), 2),
                     "peak_hour": str(seg.idxmax()), "peak_hourly_mm": round(float(seg.max()), 2),
                     "max_cell_hourly_mm": round(float(sub.max()), 2)})
    return pd.DataFrame(rows, columns=COLUMNS)
