"""Precipitation-event catalog from MRMS radar (with ASOS surface confirmation).

Pipeline
--------
1. **Hourly record** - MRMS ``MultiSensor_QPE_01H_Pass2`` over the domain for the
   whole period; the domain-mean hourly accumulation is the event signal.
2. **Detection** - an hour is wet when the domain mean >= ``wet_mm`` (0.1 mm); wet hours
   separated by fewer than ``min_gap_h`` (6) dry hours belong to the same event; events
   with a domain-mean total below ``min_total_mm`` are discarded.
3. **Precipitation type** - two independent sources:

   * MRMS ``PrecipFlag`` sampled every ``flag_step`` (10 min) during wet hours; each
     flagged cell is weighted by that hour's QPE in the cell, giving the
     *snow fraction* of the event's precipitation (flag 3) versus rain flags
     (1, 6, 7, 10, 91, 96);
   * ASOS present weather (NYC, LGA, JFK, EWR): station-hours with rain / snow /
     mixed (ice pellets, rain+snow) / freezing rain.

   Classification (both sources must agree for a "pure" type):

   * ``snow`` - snow fraction >= 0.75 and < 25 % of ASOS precipitation station-hours
     report liquid-only rain;
   * ``rain`` - snow fraction <= 0.05 and no material ASOS frozen/freezing reports
     (fewer than 2 station-hours or < 5 % of precipitation station-hours);
   * ``mix``  - everything in between (both phases present, or sources disagree).
4. **Ranking** - by domain-mean total (liquid equivalent, mm); the top ``n_per_type``
   of each type form the catalog.

Snow totals are liquid-water equivalent, as MRMS reports them; radar QPE in snow is
far less certain than in rain (and CMLs barely respond to dry snow), which is noted
in every snow event's README.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .settings import get_paths
from core.geo import NYC, Domain
from core.asos import NYC_STATIONS, fetch_asos, hourly_ptype, hourly_precip
from core.radar.mrms import MRMSClient
from core.radar.mrms import RAIN_FLAGS, SNOW_FLAGS, COOL_RAIN_FLAGS

log = logging.getLogger(__name__)

QPE = "MultiSensor_QPE_01H_Pass2"
OPENMESH_PERIOD = ("2023-10-29", "2024-07-01")


@dataclass
class Event:
    event_id: str
    start: str                 # start of first wet hour (UTC)
    end: str                   # end of last wet hour (UTC)
    duration_h: int
    wet_hours: int
    total_mm: float            # domain-mean accumulation (liquid equivalent)
    max_cell_mm: float         # largest single-cell total in the domain
    peak_hour: str             # hour-ending label of the wettest hour
    peak_hourly_mm: float      # domain-mean accumulation in that hour
    max_cell_hourly_mm: float  # largest single-cell hourly accumulation
    ptype: str = "unclassified"
    snow_fraction: float = float("nan")
    cool_rain_fraction: float = float("nan")
    flag_samples: int = 0
    asos: dict = field(default_factory=dict)
    asos_precip_mm: dict = field(default_factory=dict)
    min_temp_c: float = float("nan")
    rank: int = 0
    in_openmesh: bool = True

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.start), pd.Timestamp(self.end)


# ----------------------------------------------------------------- detection


def hourly_record(start, end, domain: Domain = NYC, client: MRMSClient | None = None) -> xr.DataArray:
    """Hourly MRMS QPE (mm, hour-ending) over ``domain`` for ``(start, end]``."""
    client = client or MRMSClient()
    return client.load(QPE, pd.Timestamp(start) + pd.Timedelta("1h"), pd.Timestamp(end), domain,
                       progress=True)


def detect_events(hourly: xr.DataArray, wet_mm: float = 0.1, min_gap_h: int = 6,
                  min_total_mm: float = 1.0, min_valid_fraction: float = 0.8) -> list[Event]:
    """Split the hourly record into events (see module docstring)."""
    valid = hourly.notnull().mean(("lat", "lon"))
    mean = hourly.mean(("lat", "lon")).where(valid >= min_valid_fraction)
    s = mean.to_series()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full)                         # archive gaps become NaN (treated as dry)
    wet = (s >= wet_mm).fillna(False).values
    idx = np.flatnonzero(wet)
    if idx.size == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] - 1 < min_gap_h:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    events = []
    for g in groups:
        t_first, t_last = full[g[0]], full[g[-1]]          # hour-ending labels
        sub = hourly.sel(time=slice(t_first, t_last))
        total_map = sub.sum("time", min_count=1)
        seg = s.loc[t_first:t_last]
        total = float(seg.sum())
        if total < min_total_mm:
            continue
        peak = seg.idxmax()
        start = t_first - pd.Timedelta("1h")
        events.append(Event(
            event_id=f"{start:%Y%m%dT%H}", start=str(start), end=str(t_last),
            duration_h=int(len(seg)), wet_hours=int((seg >= wet_mm).sum()),
            total_mm=round(total, 2), max_cell_mm=round(float(total_map.max()), 2),
            peak_hour=str(peak), peak_hourly_mm=round(float(seg.max()), 2),
            max_cell_hourly_mm=round(float(sub.max()), 2)))
    return events


# ------------------------------------------------------------ classification


def precip_flag_fractions(event: Event, hourly: xr.DataArray, domain: Domain,
                          client: MRMSClient, flag_step: str = "10min", wet_mm: float = 0.1
                          ) -> dict:
    """QPE-weighted fractions of MRMS PrecipFlag types over the event's wet hours."""
    t0, t1 = event.window
    sub = hourly.sel(time=slice(t0 + pd.Timedelta("1h"), t1))
    wet_hours = sub.time.values[(sub.mean(("lat", "lon")) >= wet_mm).values]
    snow = rain = cool = 0.0
    n = 0
    for h in wet_hours:
        h = pd.Timestamp(h)
        try:
            flags = client.load("PrecipFlag", h - pd.Timedelta("1h") + pd.Timedelta(flag_step), h,
                                domain, freq=flag_step)
        except Exception as exc:                     # archive gap: skip this hour
            log.warning("PrecipFlag unavailable for hour ending %s: %s", h, exc)
            continue
        qpe = hourly.sel(time=h).reindex_like(flags.isel(time=0), method="nearest", tolerance=0.006)
        w = qpe.fillna(0).values / flags.sizes["time"]
        F = flags.values
        snow += float(np.nansum(np.isin(F, SNOW_FLAGS) * w))
        rain += float(np.nansum(np.isin(F, RAIN_FLAGS) * w))
        cool += float(np.nansum(np.isin(F, COOL_RAIN_FLAGS) * w))
        n += flags.sizes["time"]
    tot = snow + rain
    return {"snow_fraction": snow / tot if tot > 0 else np.nan,
            "cool_rain_fraction": cool / tot if tot > 0 else np.nan, "flag_samples": n}


def asos_summary(event: Event, asos: pd.DataFrame) -> tuple[dict, dict, float]:
    """(station-hour counts per type, station precip totals mm, min temperature C)."""
    t0, t1 = event.window
    sel = asos[(asos.valid > t0) & (asos.valid <= t1)]
    if sel.empty:
        return {}, {}, float("nan")
    pt = hourly_ptype(sel)
    counts = pd.Series(pt.values.ravel()).value_counts().to_dict()
    counts = {k: int(v) for k, v in counts.items() if k != "none"}
    pr = hourly_precip(sel).sum(min_count=1).round(1)
    return counts, {k: float(v) for k, v in pr.items() if np.isfinite(v)}, round(float(sel.temp_c.min()), 1)


def classify(event: Event) -> str:
    """Apply the snow / rain / mix rule (module docstring) to a filled-in event."""
    sf = event.snow_fraction
    a = event.asos
    liquid = a.get("rain", 0)
    snow = a.get("snow", 0)
    frozen = snow + a.get("mix", 0) + a.get("freezing", 0)
    precip_hours = liquid + frozen
    if np.isnan(sf):
        return "unclassified"
    if sf >= 0.75 and (precip_hours == 0 or (precip_hours - snow) / precip_hours < 0.25):
        return "snow"
    # A lone frozen report in a long rain event (a single sleet observation) is not a
    # mixed event: frozen reports must be >= 2 station-hours and >= 5 % of the total.
    material_frozen = frozen >= 2 and (precip_hours == 0 or frozen / precip_hours >= 0.05)
    if sf <= 0.05 and not material_frozen:
        return "rain"
    return "mix"


# ------------------------------------------------------------------- catalog


def build_catalog(start: str = OPENMESH_PERIOD[0], end: str = OPENMESH_PERIOD[1],
                  domain: Domain = NYC, n_per_type: int = 5, min_total_mm: float = 1.0,
                  classify_top: int | None = None, client: MRMSClient | None = None,
                  stations=NYC_STATIONS, warm_c: float = 6.0,
                  extend: tuple | None = ("2020-11-01", "2026-05-01")) -> pd.DataFrame:
    """Detect, classify and rank events; returns one row per detected event.

    PrecipFlag is downloaded for every event except warm ones: minimum ASOS
    temperature >= ``warm_c`` and no frozen present-weather report, where snow is
    physically excluded (snow fraction set to 0). ``classify_top`` forces PrecipFlag
    for the largest N events regardless.

    ``extend``: if the main period yields fewer than ``n_per_type`` snow or mix events,
    search this longer period for them (radar/ASOS only - no link data there); see
    :func:`extended_frozen_events`. ``None`` disables it.
    """
    client = client or MRMSClient()
    hourly = hourly_record(start, end, domain, client)
    events = detect_events(hourly, min_total_mm=min_total_mm)
    log.info("%d events >= %.1f mm", len(events), min_total_mm)

    asos = fetch_asos(pd.Timestamp(start) - pd.Timedelta("1D"), pd.Timestamp(end) + pd.Timedelta("1D"),
                      stations)
    by_size = sorted(events, key=lambda e: -e.total_mm)
    forced = set(id(e) for e in by_size[:classify_top]) if classify_top else set()
    for i, e in enumerate(events):
        e.asos, e.asos_precip_mm, e.min_temp_c = asos_summary(e, asos)
        frozen_reported = bool(set(e.asos) & {"snow", "mix", "freezing"})
        warm = np.isfinite(e.min_temp_c) and e.min_temp_c >= warm_c
        if warm and not frozen_reported and id(e) not in forced:
            # Surface >= warm_c everywhere ASOS observes and no frozen reports: snow is
            # physically excluded, PrecipFlag is not needed (saves ~50 s per event).
            e.snow_fraction, e.flag_samples = 0.0, 0
        else:
            for k, v in precip_flag_fractions(e, hourly, domain, client).items():
                setattr(e, k, v)
        e.ptype = classify(e)
        log.info("event %d/%d %s: %s", i + 1, len(events), e.event_id, e.ptype)

    for e in events:
        e.in_openmesh = _overlaps(e, *OPENMESH_PERIOD)
    if extend is not None:
        counts = {t: sum(e.ptype == t for e in events) for t in ("snow", "mix")}
        need = {t for t, c in counts.items() if c < n_per_type}
        if need:
            events += extended_frozen_events(extend[0], extend[1], domain, client, stations,
                                             exclude=(start, end), min_total_mm=min_total_mm,
                                             warm_c=warm_c)

    df = pd.DataFrame([asdict(e) for e in events])
    df["selected"] = False
    for ptype in ("snow", "rain", "mix"):
        # events with link data first (they can be mapped from CMLs), then radar-only ones
        sub = df[df.ptype == ptype].sort_values(["in_openmesh", "total_mm"], ascending=[False, False])
        df.loc[sub.index, "rank"] = np.arange(1, len(sub) + 1)
        df.loc[sub.index[:n_per_type], "selected"] = True
    df.attrs = {"domain": asdict(domain), "period": [str(start), str(end)],
                "extended_period": [str(x) for x in extend] if extend else None,
                "min_total_mm": min_total_mm, "n_per_type": n_per_type}
    return df


def _overlaps(e: Event, start, end) -> bool:
    t0, t1 = e.window
    return t1 > pd.Timestamp(start) and t0 < pd.Timestamp(end)


def frozen_candidate_days(start, end, stations=NYC_STATIONS, min_station_hours: int = 3
                          ) -> pd.DatetimeIndex:
    """UTC days on which ASOS reported snow / mix / freezing in >= ``min_station_hours``."""
    asos = fetch_asos(start, end, stations)
    pt = hourly_ptype(asos)
    frozen = pt.isin(["snow", "mix", "freezing"]).sum(axis=1)
    daily = frozen.groupby(frozen.index.floor("D")).sum()
    return pd.DatetimeIndex(daily[daily >= min_station_hours].index)


def extended_frozen_events(start, end, domain: Domain, client: MRMSClient, stations=NYC_STATIONS,
                           exclude=None, min_total_mm: float = 1.0, warm_c: float = 6.0,
                           flag_step: str = "30min") -> list[Event]:
    """Snow / mix events outside the main period, screened cheaply with ASOS first.

    Days with frozen ASOS reports are grouped into clusters (+-1 day), hourly MRMS is
    fetched only for those clusters, and the normal detection + classification is run.
    PrecipFlag is sampled every ``flag_step`` (30 min: plenty for an event-level snow
    fraction, a third of the downloads). Events overlapping ``exclude`` (the main
    period) are dropped. Returned events have
    ``in_openmesh`` set from the OpenMesh record (normally False).
    """
    days = frozen_candidate_days(start, end, stations)
    if exclude is not None:
        days = days[(days < pd.Timestamp(exclude[0]) - pd.Timedelta("1D")) |
                    (days > pd.Timestamp(exclude[1]) + pd.Timedelta("1D"))]
    log.info("extended search: %d candidate days with frozen ASOS reports", len(days))
    clusters, cur = [], []
    for d in days:
        if cur and d - cur[-1] > pd.Timedelta("2D"):
            clusters.append(cur)
            cur = []
        cur.append(d)
    if cur:
        clusters.append(cur)

    out = []
    asos = fetch_asos(start, end, stations)
    for c in clusters:
        t0, t1 = c[0] - pd.Timedelta("1D"), c[-1] + pd.Timedelta("2D")
        try:
            hourly = client.load(QPE, t0 + pd.Timedelta("1h"), t1, domain)
        except Exception as exc:
            log.warning("no MRMS for %s..%s: %s", t0, t1, exc)
            continue
        for e in detect_events(hourly, min_total_mm=min_total_mm):
            e.asos, e.asos_precip_mm, e.min_temp_c = asos_summary(e, asos)
            if not set(e.asos) & {"snow", "mix", "freezing"}:
                continue                      # a rain event that happens to sit near frozen days
            for k, v in precip_flag_fractions(e, hourly, domain, client, flag_step=flag_step).items():
                setattr(e, k, v)
            e.ptype = classify(e)
            e.in_openmesh = _overlaps(e, *OPENMESH_PERIOD)
            if e.ptype in ("snow", "mix"):
                out.append(e)
                log.info("extended: %s %s %.1f mm", e.event_id, e.ptype, e.total_mm)
    return out


def save_catalog(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or get_paths().events / "catalog_all.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in ("asos", "asos_precip_mm"):
        out[c] = out[c].map(json.dumps)
    out.to_csv(path, index=False)
    (path.with_suffix(".json")).write_text(json.dumps(df.attrs, indent=2, default=str))
    return path


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    path = path or get_paths().events / "catalog_all.csv"
    df = pd.read_csv(path)
    for c in ("asos", "asos_precip_mm"):
        df[c] = df[c].map(json.loads)
    meta = path.with_suffix(".json")
    if meta.exists():
        df.attrs = json.loads(meta.read_text())
    return df


def event_from_row(row: pd.Series) -> Event:
    fields = Event.__dataclass_fields__
    return Event(**{k: row[k] for k in fields if k in row})


def write_catalog(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the catalog as ``events/all_detected_events.csv`` (dict columns as JSON)."""
    from .settings import EVENTS_DIR
    path = path or EVENTS_DIR / "all_detected_events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in ("asos", "asos_precip_mm"):
        out[c] = out[c].map(json.dumps)
    out.to_csv(path, index=False)
    return path
