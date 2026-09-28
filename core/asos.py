"""ASOS surface observations (METAR) from the Iowa Environmental Mesonet.

Used as an independent, surface-level check on precipitation *type* (MRMS PrecipFlag
is model-temperature based) and as point rainfall. Present-weather codes (``wxcodes``)
are decoded into rain / snow / ice pellets / freezing rain.

Stations used by default (all within the NYC domain):
``NYC`` Central Park, ``LGA`` LaGuardia, ``JFK`` Kennedy, ``EWR`` Newark.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time

import pandas as pd
import requests

from core.opensense.fetch import DATA_ROOT

log = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
NYC_STATIONS = ("NYC", "LGA", "JFK", "EWR")
CACHE = DATA_ROOT / "ASOS"          # downloaded reports, not tracked

# METAR present-weather precipitation groups, in priority order for classification.
_FREEZING = re.compile(r"FZ(RA|DZ)")
_PELLETS = re.compile(r"PL|GS|IC")
_SNOW = re.compile(r"SN|SG")
_RAIN = re.compile(r"(?<!FZ)(RA|DZ)")
_UNKNOWN = re.compile(r"UP")


def classify_wxcodes(code: str | float | None) -> str:
    """Map one METAR ``wxcodes`` string to none/rain/snow/mix/freezing/unknown.

    ``mix`` = snow or ice pellets together with liquid in the same report (e.g. ``RASN``,
    ``-PLRA``); ``freezing`` = freezing rain/drizzle; ice pellets alone count as ``mix``
    because they only occur in transition layers.
    """
    if not isinstance(code, str) or not code.strip():
        return "none"
    c = code.upper()
    if _FREEZING.search(c):
        return "freezing"
    snow, pellets, rain = bool(_SNOW.search(c)), bool(_PELLETS.search(c)), bool(_RAIN.search(c))
    if (snow or pellets) and rain or pellets:
        return "mix"
    if snow:
        return "snow"
    if rain:
        return "rain"
    if _UNKNOWN.search(c):
        return "unknown"
    return "none"


def fetch_asos(start, end, stations=NYC_STATIONS, use_cache: bool = True,
               timeout: float = 120, retries: int = 4) -> pd.DataFrame:
    """Routine + special METARs between ``start`` and ``end`` (UTC).

    Columns: ``station, valid (UTC), lat, lon, tmpf, temp_c, p01m (mm since last
    routine report), wxcodes, ptype``. Cached as CSV per station set and period.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    stations = tuple(sorted(stations))
    tag = hashlib.sha1(",".join(stations).encode()).hexdigest()[:6]
    path = CACHE / f"metar_{tag}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv"
    if use_cache and path.exists():
        return _finish(pd.read_csv(path, parse_dates=["valid"]))

    frames = []
    for c0 in pd.date_range(start.normalize(), end, freq="MS").union([start]).sort_values():
        c1 = min(c0 + pd.offsets.MonthBegin(1), end)
        if c1 <= c0:
            continue
        if frames:
            time.sleep(1.0)            # IEM rate-limits bursts (HTTP 429)
        frames.append(_request(c0, c1, stations, timeout, retries))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["station", "valid"])
    df = df[(df.valid >= start) & (df.valid <= end)].sort_values(["station", "valid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return _finish(df)


def _request(t0, t1, stations, timeout, retries) -> pd.DataFrame:
    params = [("station", s) for s in stations] + [
        ("data", "tmpf"), ("data", "p01m"), ("data", "wxcodes"),
        ("sts", f"{t0:%Y-%m-%dT%H:%M}Z"), ("ets", f"{t1:%Y-%m-%dT%H:%M}Z"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
        ("missing", "empty"), ("trace", "0.0001"), ("report_type", "3"), ("report_type", "4"),
    ]
    for attempt in range(retries):
        try:
            r = requests.get(IEM_ASOS_URL, params=params, timeout=timeout,
                             headers={"User-Agent": "FieldSense"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["valid"])
            return df
        except (requests.RequestException, pd.errors.ParserError) as exc:
            log.warning("IEM ASOS request failed (%s), retry %d", exc, attempt + 1)
            time.sleep(2 ** attempt * 5)   # IEM asks clients to back off
    raise RuntimeError(f"IEM ASOS request failed for {t0}..{t1}")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_c"] = (df["tmpf"] - 32) * 5 / 9
    df["ptype"] = df["wxcodes"].map(classify_wxcodes)
    return df


def hourly_ptype(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-ending precipitation type per station.

    Returns a frame indexed by hour (label = end of hour, UTC) with one column per
    station; the value is the "most frozen" type reported in that hour
    (freezing > mix > snow > rain > unknown > none), except that snow and rain reported
    in the same hour become ``mix``.
    """
    order = ["none", "unknown", "rain", "snow", "mix", "freezing"]
    rank = {k: i for i, k in enumerate(order)}
    d = df.assign(hour=df["valid"].dt.ceil("h"), r=df["ptype"].map(rank))

    def reduce(s: pd.Series) -> str:
        kinds = set(s)
        if {"rain", "snow"} <= kinds and not kinds & {"mix", "freezing"}:
            return "mix"
        return max(kinds, key=rank.get)

    out = d.groupby(["hour", "station"])["ptype"].agg(reduce).unstack("station")
    return out.fillna("none")


def hourly_precip(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-ending precipitation (mm) per station from routine METAR ``p01m``.

    Routine ASOS reports at these stations are issued at :51 and carry precipitation since
    the previous routine report; specials (SPECI) in between only carry precipitation since
    the last routine report, so for each station-hour the report closest to :51 is used.
    """
    d = df[df["valid"].dt.minute.between(45, 59)].copy()
    d["hour"] = d["valid"].dt.ceil("h")
    d["off51"] = (d["valid"].dt.minute - 51).abs()
    d = d.sort_values("off51").drop_duplicates(["hour", "station"])
    out = d.set_index(["hour", "station"])["p01m"].unstack("station").sort_index()
    return out.clip(lower=0)


IEM_ASOS_1MIN_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"


def fetch_asos_1min(start, end, stations=("LGA",), use_cache: bool = True, timeout: float = 120,
                    retries: int = 4) -> pd.DataFrame:
    """1-minute ASOS precipitation (mm per minute) from IEM, as used by implementation_2.

    Returns a frame indexed by UTC minute with one column per station. Requested in
    month-long chunks; the chunk end is exclusive, so no day is dropped at chunk edges
    (implementation_2's ``asos_io`` lost the last day of each month).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    stations = tuple(sorted(stations))
    path = CACHE / f"asos1min_{'-'.join(stations)}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frames = []
    edges = list(pd.date_range(start.normalize(), end, freq="MS").union([start, end]).sort_values())
    for c0, c1 in zip(edges[:-1], edges[1:]):
        if c1 <= c0:
            continue
        params = [("station", s) for s in stations] + [
            ("vars", "precip"), ("sts", f"{c0:%Y-%m-%dT%H:%M}Z"), ("ets", f"{c1:%Y-%m-%dT%H:%M}Z"),
            ("sample", "1min"), ("what", "download"), ("tz", "UTC")]
        for attempt in range(retries):
            try:
                r = requests.get(IEM_ASOS_1MIN_URL, params=params, timeout=timeout,
                                 headers={"User-Agent": "FieldSense"})
                r.raise_for_status()
                frames.append(pd.read_csv(io.StringIO(r.text)))
                break
            except (requests.RequestException, pd.errors.ParserError) as exc:
                log.warning("IEM 1-min request failed (%s), retry %d", exc, attempt + 1)
                time.sleep(2 ** attempt * 5)
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True)
    df["valid"] = pd.to_datetime(df["valid(UTC)"])
    df["precip"] = pd.to_numeric(df["precip"], errors="coerce") * 25.4
    out = df.pivot_table(index="valid", columns="station", values="precip", aggfunc="first")
    out = out.loc[(out.index >= start) & (out.index <= end)].sort_index()
    out.columns.name = None
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out
