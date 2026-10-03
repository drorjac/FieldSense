"""MRMS radar over a wide box around New York City, for nowcasting with the OpenMesh links.

The OpenMesh network spans about 30 x 30 km; a nowcast needs to see rain before it reaches
the city. ``NYC_WIDE`` is a box of about 240 x 240 km centred on the network (the centre of
:data:`core.geo.OPENMESH`), cut from the CONUS MRMS grid at its native 0.01 deg:
215 rows x 284 columns (latitude x longitude), about 1.1 km x 0.84 km per cell.

This module defines the box, lists the windows to fetch (the rain events of
``projects/nyc_rain_maps/events/all_detected_events.csv`` inside the OpenMesh period,
padded 3 h before and 1 h after), and fills the shared MRMS cache:

* ``PrecipRate`` (instantaneous surface rate, mm/h) at its native 2 minutes;
* ``PrecipFlag`` (surface precipitation type) every 10 minutes, to drop snow and mix;
* ASOS 1-minute precipitation at NYC, LGA, JFK and EWR (IEM), as the independent reference.

Loading a window afterwards reads only the cache:

    from core.radar.mrms_nyc_wide import load_rate
    rate = load_rate("2023-12-18 00:00", "2023-12-18 12:00")   # (time, lat, lon) mm/h, 2 min

Prefetch (resumable; finished days are read from the cache):

    python -m core.radar.mrms_nyc_wide            # all events, PrecipRate + PrecipFlag + ASOS
    python -m core.radar.mrms_nyc_wide --list     # the windows only
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import data_paths as dp
from core.geo import OPENMESH, Domain

HALF_SIDE_KM = 120.0
_LAT0 = 0.5 * (OPENMESH.lat_min + OPENMESH.lat_max)
_LON0 = 0.5 * (OPENMESH.lon_min + OPENMESH.lon_max)
_DLAT = HALF_SIDE_KM / 111.32
_DLON = HALF_SIDE_KM / (111.32 * np.cos(np.radians(_LAT0)))

NYC_WIDE = Domain(round(float(_LAT0 - _DLAT), 4), round(float(_LAT0 + _DLAT), 4),
                  round(float(_LON0 - _DLON), 4), round(float(_LON0 + _DLON), 4), name="nyc_wide")

EVENTS_CSV = Path(__file__).resolve().parents[2] / "projects/nyc_rain_maps/events/all_detected_events.csv"
PAD_BEFORE = pd.Timedelta(hours=3)
PAD_AFTER = pd.Timedelta(hours=1)
MAX_SNOW_FRACTION = 0.1          # "mostly rain": mix events with at most 10% snow samples
FLAG_FREQ = "10min"
LOG_DIR = dp.OUTPUTS / "_multisensor_nowcasting" / "logs"

log = logging.getLogger(__name__)


def event_windows(events_csv: Path = EVENTS_CSV) -> pd.DataFrame:
    """Rain (or mostly-rain) events in the OpenMesh period, with padded fetch windows."""
    e = pd.read_csv(events_csv, parse_dates=["start", "end"])
    rain = (e.ptype == "rain") | ((e.ptype == "mix") & (e.snow_fraction <= MAX_SNOW_FRACTION))
    e = e[e.in_openmesh.astype(bool) & rain].copy()
    e["fetch_start"] = e.start - PAD_BEFORE
    e["fetch_end"] = e.end + PAD_AFTER
    return e[["event_id", "ptype", "start", "end", "fetch_start", "fetch_end"]].reset_index(drop=True)


def _client(processes: int | None = None):
    from core.radar.mrms import MRMSClient
    return MRMSClient(processes=processes)


def load_rate(start, end, client=None):
    """``PrecipRate`` (mm/h, 2 min) on :data:`NYC_WIDE` for ``[start, end]``; fetches what is not cached."""
    return (client or _client()).load("PrecipRate", start, end, NYC_WIDE)


def load_flag(start, end, client=None):
    """``PrecipFlag`` categories on :data:`NYC_WIDE`, every 10 minutes."""
    return (client or _client()).load("PrecipFlag", start, end, NYC_WIDE, freq=FLAG_FREQ)


def load_asos_1min(start, end):
    """ASOS 1-minute precipitation (mm per minute) at NYC, LGA, JFK, EWR."""
    from core.asos import NYC_STATIONS, fetch_asos_1min
    return fetch_asos_1min(start, end, stations=NYC_STATIONS)


def prefetch(products=("PrecipRate", "PrecipFlag"), asos: bool = True) -> pd.DataFrame:
    """Fill the caches for every event window; one row of counts per event and product."""
    windows = event_windows()
    client = _client()
    rows = []
    t_all = time.time()
    for i, w in windows.iterrows():
        for product in products:
            t0 = time.time()
            freq = FLAG_FREQ if product == "PrecipFlag" else None
            try:
                da = client.load(product, w.fetch_start, w.fetch_end, NYC_WIDE, freq=freq)
                n_valid, n_missing = int(da.sizes["time"]), len(da.attrs.get("missing_times", []))
                shape = tuple(int(n) for n in da.shape[1:])
            except Exception as exc:                       # noqa: BLE001 - log and continue
                log.warning("%s %s failed: %s", w.event_id, product, exc)
                n_valid, n_missing, shape = 0, -1, ()
            rows.append(dict(event_id=w.event_id, product=product, valid=n_valid,
                             missing=n_missing, shape=str(shape), seconds=round(time.time() - t0, 1)))
            log.info("[%d/%d] %s %s: %d valid, %d missing, %.0fs (total %.0f min)", i + 1,
                     len(windows), w.event_id, product, n_valid, n_missing,
                     time.time() - t0, (time.time() - t_all) / 60)
        if asos:
            try:
                a = load_asos_1min(w.fetch_start, w.fetch_end)
                rows.append(dict(event_id=w.event_id, product="ASOS_1min", valid=int(a.notna().any(axis=1).sum()),
                                 missing=int(a.isna().all(axis=1).sum()), shape=str(tuple(a.columns)), seconds=0))
            except Exception as exc:                       # noqa: BLE001
                log.warning("%s ASOS failed: %s", w.event_id, exc)
    out = pd.DataFrame(rows)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG_DIR / "mrms_nyc_wide_prefetch.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the event windows and exit")
    ap.add_argument("--no-flag", action="store_true", help="skip PrecipFlag")
    ap.add_argument("--no-asos", action="store_true", help="skip ASOS 1-minute")
    args = ap.parse_args()
    if args.list:
        print(NYC_WIDE)
        print(event_windows().to_string())
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "mrms_nyc_wide_prefetch.log"),
                                  logging.StreamHandler()])
    products = ("PrecipRate",) if args.no_flag else ("PrecipRate", "PrecipFlag")
    out = prefetch(products, asos=not args.no_asos)
    print(out.groupby("product")[["valid", "missing"]].sum())


if __name__ == "__main__":       # the MRMS client decodes in a process pool
    main()
