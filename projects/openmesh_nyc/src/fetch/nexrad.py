"""
NEXRAD reflectivity over the OpenMesh network, for rain and snow days.

OpenMesh publishes CMLs, personal weather stations and ASOS, but **no radar**,
so there is no gridded reference to compare a CML-derived field against. This
pulls one.

Source
------
The Iowa Environmental Mesonet archive of **KOKX** (Upton, NY), the NEXRAD
that covers New York City: product ``N0B``, super-resolution base
reflectivity, ~1.2 km, irregular 2-5 minute cadence, no authentication. Each
frame is a palette PNG plus an ESRI world file.

Two alternatives were rejected. NOAA's Level II archive on S3 is richer but
needs a full radar decoder and signed listing. IEM's CONUS ``n0q`` composite
is higher resolution (0.005 deg) but each frame is a 6 MB nationwide image
for a crop of a few hundred pixels - 1.7 GB per day against 30 MB here, for
one city. Single-site also avoids compositing artefacts between overlapping
radars.

Why rain and snow are handled separately
----------------------------------------
Reflectivity is converted to a precipitation rate through ``Z = a R**b``, and
the coefficients for snow are not the coefficients for rain. Using the rain
relation on a snow day understates the rate badly, because snow of a given
water-equivalent rate scatters far more than rain. This module classifies each
day from METAR present-weather codes and applies:

* rain: ``Z = 200 R**1.6``   (Marshall-Palmer)
* snow: ``Z = 180 S**2.0``   (Sekhon-Srivastava, the WSR-88D operational pair)

Both give a liquid-water-equivalent rate in mm/h.

    python -m nexrad --classify
    python -m nexrad --date 2024-01-16          # a snow day
    python -m nexrad --date 2024-03-23          # a strong rain day
    python -m nexrad --best 2 --kind snow
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
CACHE = REPO_ROOT / "dataset" / "open_datasets" / "OpenMesh_NYC" / "radar"

IEM_ARCHIVE = "https://mesonet.agron.iastate.edu/archive/data"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# NYC ASOS stations around the mesh network.
ASOS_STATIONS = ("NYC", "LGA", "JFK", "EWR")

# The OpenMesh network spans lower Manhattan and Brooklyn; PWS reach a little
# wider. Padded to give the radar some context around the links.
BBOX = {"lon_min": -74.10, "lon_max": -73.80,
        "lat_min": 40.55, "lat_max": 40.92}

# KOKX (Upton, NY) is the NEXRAD covering New York City. Georeferencing is
# read from each product's world file rather than hardcoded, since the grid
# differs between products.
RADAR_SITE = "OKX"
RADAR_PRODUCT = "N0B"

# Z-R and Z-S pairs. Snow is far more reflective per mm/h of melted water.
ZR = {"rain": (200.0, 1.6), "snow": (180.0, 2.0),
      # A mixed day has both phases present and no single relation is right.
      # The rain pair is used and the result flagged, rather than inventing a
      # blend that would look more precise than it is.
      "mixed": (200.0, 1.6)}

# OpenMesh covers this period; classification is restricted to it.
PERIOD = ("2023-10-01", "2024-07-31")


@dataclass(frozen=True)
class Day:
    date: str
    kind: str            # "rain" or "snow"
    precip_mm: float
    snow_obs: int
    rain_obs: int


# --------------------------------------------------------------------------
# day classification
# --------------------------------------------------------------------------
def _asos_frame(start: str = PERIOD[0], end: str = PERIOD[1]) -> pd.DataFrame:
    """METAR observations for the NYC stations over the OpenMesh period."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    params = [("data", "tmpf"), ("data", "p01i"), ("data", "wxcodes"),
              ("year1", s.year), ("month1", s.month), ("day1", s.day),
              ("year2", e.year), ("month2", e.month), ("day2", e.day),
              ("tz", "UTC"), ("format", "onlycomma"), ("missing", "empty"),
              ("trace", "0.0001")]
    params += [("station", st) for st in ASOS_STATIONS]

    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"asos_{s.date()}_{e.date()}.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["valid"])

    r = requests.get(IEM_ASOS, params=params, timeout=300)
    r.raise_for_status()
    cached.write_text(r.text)
    return pd.read_csv(io.StringIO(r.text), parse_dates=["valid"])


def classify_days(min_precip_mm: float = 2.0) -> pd.DataFrame:
    """Daily rain/snow classification for the OpenMesh period.

    ``p01i`` is a one-hour running total reported at every observation, so
    summing it straight would multiply the precipitation by the number of
    observations per hour. It is reduced to one value per station-hour first.
    """
    df = _asos_frame()
    df["p01i"] = pd.to_numeric(df["p01i"], errors="coerce").fillna(0.0)
    wx = df["wxcodes"].fillna("")

    df["is_snow"] = wx.str.contains("SN|SG|IC|PL", regex=True)
    df["is_rain"] = wx.str.contains("RA|DZ|TS", regex=True)
    df["hour"] = df["valid"].dt.floor("h")
    df["date"] = df["valid"].dt.date

    hourly = (df.groupby(["station", "hour"])["p01i"].max()
              .reset_index())
    hourly["date"] = hourly["hour"].dt.date
    daily_mm = (hourly.groupby(["date", "station"])["p01i"].sum()
                .groupby("date").mean() * 25.4)          # inches -> mm

    obs = df.groupby("date")[["is_snow", "is_rain"]].sum()
    out = obs.join(daily_mm.rename("precip_mm")).fillna(0.0)

    # A single snow observation in a day of rain does not make a snow day -
    # 2024-01-28 has 2 frozen obs against 706 liquid ones. Classify on which
    # phase dominates, and call it mixed when both are well represented,
    # because the Z-S and Z-R relations disagree most there.
    total = out["is_snow"] + out["is_rain"]
    snow_frac = np.where(total > 0, out["is_snow"] / np.maximum(total, 1), 0.0)
    out["snow_fraction"] = snow_frac
    out["kind"] = np.where(snow_frac >= 0.7, "snow",
                           np.where(snow_frac <= 0.1, "rain", "mixed"))
    out = out[out["precip_mm"] >= min_precip_mm]
    return out.rename(columns={"is_snow": "snow_obs", "is_rain": "rain_obs"})


def best_days(kind: str, n: int = 3) -> list[Day]:
    """The n strongest days of one kind."""
    table = classify_days()
    sel = table[table["kind"] == kind]
    sort_by = ["snow_obs", "precip_mm"] if kind == "snow" else ["precip_mm"]
    sel = sel.sort_values(sort_by, ascending=False).head(n)
    return [Day(str(d), kind, float(r.precip_mm), int(r.snow_obs),
                int(r.rain_obs)) for d, r in sel.iterrows()]


def classify_one(date: str) -> str:
    """The kind assigned to a single day."""
    import pandas as pd

    table = classify_days(min_precip_mm=0.0)
    key = pd.Timestamp(date).date()
    return str(table.loc[key, "kind"]) if key in table.index else "rain"


# --------------------------------------------------------------------------
# radar
# --------------------------------------------------------------------------
def _day_url(date: str) -> str:
    d = pd.Timestamp(date)
    return (f"{IEM_ARCHIVE}/{d:%Y/%m/%d}/GIS/ridge/"
            f"{RADAR_SITE}/{RADAR_PRODUCT}")


def list_frames(date: str) -> list[str]:
    """Timestamps available for one day, from the archive directory listing.

    The cadence is irregular - 2 to 5 minutes depending on the volume scan
    mode - so the available frames are discovered rather than assumed.
    """
    r = requests.get(_day_url(date) + "/", timeout=120)
    r.raise_for_status()
    pattern = rf"{RADAR_SITE}_{RADAR_PRODUCT}_(\d{{12}})\.png"
    return sorted(set(re.findall(pattern, r.text)))


def _geotransform(date: str, stamp: str,
                  session: requests.Session | None = None) -> tuple:
    """(pixel size, origin lon, origin lat) from the product's world file."""
    get = (session or requests).get
    url = f"{_day_url(date)}/{RADAR_SITE}_{RADAR_PRODUCT}_{stamp}.wld"
    r = get(url, timeout=60)
    r.raise_for_status()
    v = [float(x) for x in r.text.split()]
    return v[0], v[4], v[5]          # x step, origin lon, origin lat


def fetch_frame(date: str, stamp: str,
                session: requests.Session | None = None):
    """One frame as a full-tile dBZ array, or ``None`` if unavailable."""
    from PIL import Image

    get = (session or requests).get
    url = f"{_day_url(date)}/{RADAR_SITE}_{RADAR_PRODUCT}_{stamp}.png"
    try:
        r = get(url, timeout=120)
        if r.status_code != 200:
            return None
    except requests.RequestException:
        return None

    tile = np.array(Image.open(io.BytesIO(r.content)))
    # IEM ramp: 0 means no data; 1..255 maps to -32 dBZ upward in 0.5 dB steps.
    return np.where(tile == 0, np.nan,
                    -32.0 + (tile.astype(float) - 1.0) * 0.5)


def to_rate(dbz: np.ndarray, kind: str) -> np.ndarray:
    """Reflectivity -> liquid-water-equivalent rate (mm/h) for rain or snow."""
    a, b = ZR[kind]
    z = 10.0 ** (np.asarray(dbz, dtype=float) / 10.0)
    return (z / a) ** (1.0 / b)


def build_day(date: str, kind: str, step_minutes: int = 5,
              cache: bool = True) -> xr.Dataset:
    """Every available frame for one day, cropped to the bbox."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / f"nexrad_{RADAR_SITE}_{RADAR_PRODUCT}_{date}_{kind}.nc"
    if cache and out_path.exists():
        print(f"  [cache] {out_path.name}")
        return xr.open_dataset(out_path)

    stamps = list_frames(date)
    if not stamps:
        raise SystemExit(f"no {RADAR_SITE} {RADAR_PRODUCT} frames for {date}")

    # Thin to roughly the requested spacing; the native cadence is irregular.
    kept, last = [], None
    for st in stamps:
        t = datetime.strptime(st, "%Y%m%d%H%M")
        if last is None or (t - last) >= timedelta(minutes=step_minutes):
            kept.append((st, t))
            last = t

    session = requests.Session()
    step, lon0, lat0 = _geotransform(date, kept[0][0], session)

    # Crop indices into the full tile, from the world file.
    def _cols_rows(shape):
        n_rows, n_cols = shape
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)
        lons = lon0 + (cols + 0.5) * step
        lats = lat0 - (rows + 0.5) * step
        cmask = (lons >= BBOX["lon_min"]) & (lons <= BBOX["lon_max"])
        rmask = (lats >= BBOX["lat_min"]) & (lats <= BBOX["lat_max"])
        return cmask, rmask, lons[cmask], lats[rmask]

    frames, times, lons, lats = [], [], None, None
    print(f"  {len(stamps)} frames available, keeping {len(kept)} "
          f"at ~{step_minutes} min")
    for i, (st, t) in enumerate(kept):
        full = fetch_frame(date, st, session)
        if full is None:
            continue
        if lons is None:
            cmask, rmask, lons, lats = _cols_rows(full.shape)
        frames.append(full[np.ix_(rmask, cmask)])
        times.append(np.datetime64(t, "ns"))
        if (i + 1) % 60 == 0:
            print(f"    {i + 1}/{len(kept)} fetched", flush=True)

    if not frames:
        raise SystemExit(f"no frames retrieved for {date}")

    dbz = np.stack(frames)
    ds = xr.Dataset(
        {"dbz": (("time", "lat", "lon"), dbz.astype("float32")),
         "R": (("time", "lat", "lon"), to_rate(dbz, kind).astype("float32"))},
        coords={"time": np.array(times), "lat": lats, "lon": lons},
    )
    a, b = ZR[kind]
    ds.dbz.attrs.update(units="dBZ",
                        long_name=f"{RADAR_SITE} {RADAR_PRODUCT} base reflectivity")
    ds.R.attrs.update(units="mm h-1",
                      long_name="liquid water equivalent precipitation rate",
                      note=f"Z = {a} R^{b} ({kind})")
    ds.attrs.update(
        source=f"Iowa Environmental Mesonet NEXRAD {RADAR_SITE} {RADAR_PRODUCT}",
        url=_day_url(date), date=date, precipitation_kind=kind,
        bbox=str(BBOX), grid_step_deg=step,
        note="covers the OpenMesh NYC network; OpenMesh ships no radar")

    if cache:
        ds.to_netcdf(out_path)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} "
              f"({out_path.stat().st_size/1e6:.1f} MB, {len(frames)} frames)")
    return ds


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classify", action="store_true",
                    help="list the strongest rain and snow days, then exit")
    ap.add_argument("--date", help="fetch one day, YYYY-MM-DD")
    ap.add_argument("--kind", choices=sorted(ZR), help="rain or snow")
    ap.add_argument("--best", type=int, metavar="N",
                    help="fetch the N strongest days of --kind")
    ap.add_argument("--step", type=int, default=5,
                    help="minimum minutes between kept frames "
                         "(default 5; native cadence is 2-5)")
    args = ap.parse_args()

    if args.classify or (not args.date and not args.best):
        for kind in ("snow", "rain"):
            print(f"\n=== strongest {kind} days over the OpenMesh period ===")
            print(f"{'date':>12}{'precip mm':>11}{'snow obs':>10}{'rain obs':>10}")
            for d in best_days(kind, 8):
                print(f"{d.date:>12}{d.precip_mm:>11.1f}"
                      f"{d.snow_obs:>10d}{d.rain_obs:>10d}")
        print(f"\ncached under {CACHE.relative_to(REPO_ROOT)}/")
        return

    if args.best:
        if not args.kind:
            ap.error("--best needs --kind")
        targets = [(d.date, d.kind) for d in best_days(args.kind, args.best)]
    else:
        kind = args.kind
        if kind is None:
            table = classify_days()
            key = pd.Timestamp(args.date).date()
            kind = table.loc[key, "kind"] if key in table.index else "rain"
            print(f"  classified {args.date} as {kind}")
        targets = [(args.date, kind)]

    for date, kind in targets:
        print(f"\n{date}  ({kind})")
        ds = build_day(date, kind, args.step)
        r = np.asarray(ds.R)
        print(f"  {ds.sizes['time']} frames, grid {ds.sizes['lat']} x "
              f"{ds.sizes['lon']}, peak {np.nanmax(r):.1f} mm/h")


if __name__ == "__main__":
    main()
