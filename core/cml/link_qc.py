"""Link quality control, implementing the procedure in the project README.

Three stages, cheapest first; every rejection is logged with its reason so the
surviving link set is reproducible:

1. **Metadata QC** (static): coordinates outside the study area, zero/implausible
   path length, frequency outside the useful band, duplicate paths (including the two
   directions of a bidirectional pair).
2. **Time-series QC** (per link, per window): availability, constant (stuck) signal,
   physically impossible values and isolated spikes.
3. **Retrieval QC** (after rain estimation): links wet while their neighbours are dry,
   and totals far off their neighbours'. These are *flagged*, never auto-dropped -
   in convective storms a genuine outlier is signal.

Thresholds are dataclass fields with the reasoning next to them; change them
explicitly and they end up in the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import NYC, Domain, haversine_m


@dataclass
class QCConfig:
    domain: Domain = NYC
    # Below ~10 GHz rain attenuation is too weak to separate from 1-dB quantization
    # on city-length paths (a = 4e-4 at 5.8 GHz: 10 mm/h over 2 km gives < 0.1 dB).
    min_frequency_ghz: float = 10.0
    max_frequency_ghz: float = 100.0      # ITU-R P.838 table range
    # Short paths accumulate too little attenuation (0.3 km at 68 GHz, 5 mm/h ~ 0.8 dB);
    # long paths break the uniform-rain-along-path assumption.
    min_length_km: float = 0.3
    max_length_km: float = 20.0
    # Endpoints within this distance are "the same site" for duplicate detection.
    duplicate_tolerance_m: float = 100.0
    min_availability: float = 0.8         # fraction of non-NaN samples in the window
    min_std_db: float = 0.05              # below: stuck / synthetic feed
    rsl_range_dbm: tuple = (-100.0, -10.0)
    spike_db: float = 15.0                # single-sample excursion w.r.t. both neighbours
    # retrieval QC
    neighbour_radius_km: float = 5.0
    wet_alone_mm: float = 2.0             # link total while neighbours' median < 0.2 mm
    outlier_ratio: float = 5.0            # total vs neighbour median (both directions)


def _record(rows, link, stage, reason, dropped=True, value=None):
    rows.append({"link": link, "stage": stage, "reason": reason, "dropped": dropped,
                 "value": value})


def metadata_qc(table: pd.DataFrame, cfg: QCConfig = QCConfig()) -> tuple[list[str], pd.DataFrame]:
    """Stage 1 on a :func:`~core.opensense.openmesh.sublinks_table`. Returns (kept, log)."""
    rows = []
    keep = []
    for lk, r in table.iterrows():
        inside = cfg.domain.contains([r.site_0_lat, r.site_1_lat], [r.site_0_lon, r.site_1_lon])
        if not np.all(np.isfinite([r.site_0_lat, r.site_0_lon, r.site_1_lat, r.site_1_lon])) or not inside.all():
            _record(rows, lk, "metadata", "coordinates missing or outside study area")
        elif r.length <= 0 or haversine_m(r.site_0_lat, r.site_0_lon, r.site_1_lat, r.site_1_lon) < 1:
            _record(rows, lk, "metadata", "zero-length path", value=r.length)
        elif r.length < cfg.min_length_km:
            _record(rows, lk, "metadata", f"path shorter than {cfg.min_length_km} km", value=r.length)
        elif r.length > cfg.max_length_km:
            _record(rows, lk, "metadata", f"path longer than {cfg.max_length_km} km", value=r.length)
        elif not cfg.min_frequency_ghz <= r.frequency <= cfg.max_frequency_ghz:
            _record(rows, lk, "metadata",
                    f"frequency outside {cfg.min_frequency_ghz}-{cfg.max_frequency_ghz} GHz", value=r.frequency)
        else:
            keep.append(lk)

    # duplicates: same endpoints (either direction) and same band; keep best availability
    kept = table.loc[keep]
    if "availability" in kept:            # prefer the better-recorded one of a duplicate pair
        kept = kept.sort_values("availability", ascending=False)
    chosen = []
    for lk, r in kept.iterrows():
        dup_of = None
        for other in chosen:
            o = table.loc[other]
            same = (haversine_m(r.site_0_lat, r.site_0_lon, o.site_0_lat, o.site_0_lon) < cfg.duplicate_tolerance_m and
                    haversine_m(r.site_1_lat, r.site_1_lon, o.site_1_lat, o.site_1_lon) < cfg.duplicate_tolerance_m)
            flip = (haversine_m(r.site_0_lat, r.site_0_lon, o.site_1_lat, o.site_1_lon) < cfg.duplicate_tolerance_m and
                    haversine_m(r.site_1_lat, r.site_1_lon, o.site_0_lat, o.site_0_lon) < cfg.duplicate_tolerance_m)
            if (same or flip) and abs(r.frequency - o.frequency) < 10:
                dup_of = other
                break
        if dup_of is not None:
            _record(rows, lk, "metadata", "duplicate path (other direction/sublink kept)", value=dup_of)
        else:
            chosen.append(lk)
    order = [str(lk) for lk in table.index if lk in chosen]
    return order, pd.DataFrame(rows, columns=["link", "stage", "reason", "dropped", "value"])


def timeseries_qc(links: xr.Dataset, cfg: QCConfig = QCConfig()) -> tuple[list[str], pd.DataFrame]:
    """Stage 2 on a loaded link set (event window). Returns (kept, log)."""
    rows, keep = [], []
    rsl = links["rsl"]
    lo, hi = cfg.rsl_range_dbm
    for lk in links.link.values:
        x = rsl.sel(link=lk).values.astype(float)
        avail = np.isfinite(x).mean()
        if avail < cfg.min_availability:
            _record(rows, lk, "timeseries", f"availability below {cfg.min_availability:.0%}", value=avail)
            continue
        if np.nanstd(x) < cfg.min_std_db:
            _record(rows, lk, "timeseries", "constant signal (stuck feed)", value=float(np.nanstd(x)))
            continue
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        if bad.mean() > 0.01:
            _record(rows, lk, "timeseries", f"RSL outside {lo}..{hi} dBm", value=float(bad.mean()))
            continue
        d_prev, d_next = x[1:-1] - x[:-2], x[1:-1] - x[2:]
        spikes = int(np.sum((np.abs(d_prev) > cfg.spike_db) & (np.abs(d_next) > cfg.spike_db)
                            & (np.sign(d_prev) == np.sign(d_next))))
        if spikes:
            _record(rows, lk, "timeseries", "isolated spikes (flagged, kept)", dropped=False, value=spikes)
        keep.append(str(lk))
    return keep, pd.DataFrame(rows, columns=["link", "stage", "reason", "dropped", "value"])


def remove_spikes(links: xr.Dataset, spike_db: float = 15.0) -> xr.Dataset:
    """NaN-out isolated single-sample spikes (both neighbours differ by > ``spike_db``)."""
    x = links["rsl"].values.astype(float).copy()
    d_prev, d_next = x[:, 1:-1] - x[:, :-2], x[:, 1:-1] - x[:, 2:]
    spk = (np.abs(d_prev) > spike_db) & (np.abs(d_next) > spike_db) & (np.sign(d_prev) == np.sign(d_next))
    x[:, 1:-1][spk] = np.nan
    out = links.copy()
    out["rsl"] = (links["rsl"].dims, x.astype("float32"), links["rsl"].attrs)
    return out


def retrieval_qc(rain: xr.DataArray, cfg: QCConfig = QCConfig()) -> pd.DataFrame:
    """Stage 3 on estimated ``rain(link, time)`` in mm/h. Flags only (nothing dropped)."""
    t = pd.DatetimeIndex(rain.time.values)
    dt_h = (t[1] - t[0]) / pd.Timedelta("1h") if len(t) > 1 else 1.0
    total = (rain.fillna(0) * dt_h).sum("time").to_series()
    lat, lon = rain.mid_lat.values, rain.mid_lon.values
    rows = []
    for i, lk in enumerate(rain.link.values):
        d = haversine_m(lat[i], lon[i], lat, lon)
        nb = (d <= cfg.neighbour_radius_km * 1000) & (np.arange(lat.size) != i)
        if nb.sum() < 2:
            _record(rows, lk, "retrieval", "fewer than 2 neighbours: not checked", dropped=False)
            continue
        med = float(np.median(total.values[nb]))
        tot = float(total.iloc[i])
        if med < 0.2 and tot >= cfg.wet_alone_mm:
            _record(rows, lk, "retrieval", "wet while neighbours dry", dropped=False, value=tot)
        elif med >= 1.0 and (tot > cfg.outlier_ratio * med or tot < med / cfg.outlier_ratio):
            _record(rows, lk, "retrieval", f"total {tot:.1f} mm vs neighbour median {med:.1f} mm",
                    dropped=False, value=tot / med)
    return pd.DataFrame(rows, columns=["link", "stage", "reason", "dropped", "value"])


def qc_summary(log: pd.DataFrame, n_in: int) -> dict:
    """Links in, links out and a count per rejection reason (README reporting rule)."""
    flag = log["dropped"].astype(bool)
    dropped = log[flag]
    return {"links_in": n_in, "links_dropped": int(dropped.link.nunique()),
            "links_out": n_in - int(dropped.link.nunique()),
            "by_reason": dropped.groupby("reason").link.count().to_dict(),
            "flagged": log[~flag].groupby("reason").link.count().to_dict()}


def run_qc(table: pd.DataFrame, links: xr.Dataset, cfg: QCConfig = QCConfig()
           ) -> tuple[list[str], pd.DataFrame, dict]:
    """Stages 1-2. Returns (surviving labels, full log, summary)."""
    keep1, log1 = metadata_qc(table, cfg)
    keep2, log2 = timeseries_qc(links.sel(link=[lk for lk in links.link.values if lk in keep1]), cfg)
    log = pd.concat([x for x in (log1, log2) if len(x)], ignore_index=True) if len(log1) + len(log2) \
        else log1
    summary = qc_summary(log, len(table))
    summary["config"] = {k: (v if not isinstance(v, Domain) else asdict(v)) for k, v in asdict(cfg).items()}
    return keep2, log, summary
