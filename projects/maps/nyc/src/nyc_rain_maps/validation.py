"""Data validation: audit the radar fetching, then score every source against official gauges.

Part A - is the MRMS data we fetch the data NOAA published, placed where it belongs?

* **A1 sources** - the same file from the NOAA AWS bucket and the IEM mirror decodes to the
  same field;
* **A2 decoder** - our in-memory ecCodes crop equals an independent decode of the whole
  CONUS file with xarray/cfgrib, value for value, on the same coordinates;
* **A3 georeferencing** - the value we sample at each official gauge equals ecCodes'
  own nearest-grid-point lookup in the original GRIB;
* **A4 accumulation** - 24 of our hourly Pass-2 accumulations sum to MRMS's independently
  produced 24-h Pass-2 accumulation;
* **A5 completeness** - hours present / missing in the cached record.

Part B - how close is each data source to the official gauges?

The NWS ASOS stations (Central Park, LaGuardia, JFK, Newark) are the only official,
maintained gauges in the domain. Hour by hour (hour-ending, routine METAR ``p01m``) they are
compared with: MRMS Pass 2 (the reference used everywhere in this project), MRMS Pass 1 and
radar-only QPE, the PWS network (nearest station and IDW of all stations) and, where the
links reach, the CML maps of both implementations. Hours are typed by the station's own
present-weather report (rain / snow / mix), because gauges and radar fail differently in
frozen precipitation.

    from nyc_rain_maps.validation import run_validation
    v = run_validation()
    v.write_report("results/validation")
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from core.geo import NYC, Domain, haversine_m
from core.maps.scores import scores
from core.opensense import openmesh as om
from core.asos import NYC_STATIONS, fetch_asos, hourly_precip, hourly_ptype
from core.radar.mrms import MRMSClient, decode_grib, file_url
from core.radar.mrms import get_product

log = logging.getLogger(__name__)

QPE = "MultiSensor_QPE_01H_Pass2"
PERIOD = ("2023-10-29", "2024-07-01")


# ------------------------------------------------------------------ part A


def check_sources(times, product: str = QPE, domain: Domain = NYC) -> pd.DataFrame:
    """A1: download each file from AWS and from IEM separately and compare."""
    rows = []
    for t in pd.DatetimeIndex(times):
        rec = {"time": t}
        fields = {}
        for src in ("aws", "iem"):
            r = requests.get(file_url(product, t, src), timeout=120)
            rec[f"{src}_http"] = r.status_code
            if r.ok:
                raw = gzip.decompress(r.content)
                rec[f"{src}_md5"] = hashlib.md5(raw).hexdigest()[:12]
                fields[src] = decode_grib(raw, domain)[0]
        rec["identical_grib"] = rec.get("aws_md5") == rec.get("iem_md5") and "aws_md5" in rec
        if len(fields) == 2:
            rec["max_abs_diff"] = float(np.nanmax(np.abs(fields["aws"] - fields["iem"])))
        rows.append(rec)
    return pd.DataFrame(rows)


def check_decoder(times, product: str = QPE, domain: Domain = NYC) -> pd.DataFrame:
    """A2: our crop vs a full independent cfgrib decode cropped afterwards."""
    rows = []
    p = get_product(product)
    for t in pd.DatetimeIndex(times):
        raw = gzip.decompress(requests.get(file_url(p, t, "aws"), timeout=120).content)
        ours, lat, lon, valid = decode_grib(raw, domain)
        with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
            f.write(raw)
            f.flush()
            # decode_timedelta=False: only the field is compared, and xarray < 2024.9 cannot
            # decode cfgrib's forecast step under pandas 3
            ds = xr.open_dataset(f.name, engine="cfgrib", backend_kwargs={"indexpath": ""},
                                 decode_timedelta=False)
            da = ds[list(ds.data_vars)[0]]
            sub = da.sel(latitude=lat, longitude=lon % 360, method="nearest").load().values
            ds.close()
        ref = np.where(sub < 0, np.nan, sub)
        same_nan = np.array_equal(np.isnan(ours), np.isnan(ref))
        rows.append({"time": t, "cells": ours.size, "valid_time_ok": valid == t,
                     "same_missing_mask": bool(same_nan),
                     "max_abs_diff": float(np.nanmax(np.abs(ours - ref))) if np.isfinite(ours).any() else 0.0})
    return pd.DataFrame(rows)


def check_georef(times, stations: pd.DataFrame, product: str = QPE, client: MRMSClient | None = None
                 ) -> pd.DataFrame:
    """A3: our cached value at each station vs ecCodes' nearest grid point in the raw file."""
    import eccodes

    client = client or MRMSClient()
    rows = []
    for t in pd.DatetimeIndex(times):
        raw = gzip.decompress(requests.get(file_url(product, t, "aws"), timeout=120).content)
        cached = client.load(product, t, t, NYC).isel(time=0)
        h = eccodes.codes_new_from_message(raw)
        try:
            for st, r in stations.iterrows():
                n = eccodes.codes_grib_find_nearest(h, float(r.lat), float(r.lon))[0]
                ours = float(cached.sel(lat=r.lat, lon=r.lon, method="nearest"))
                val = np.nan if n.value < 0 else float(n.value)
                rows.append({"time": t, "station": st, "grib_lat": round(n.lat, 4),
                             "grib_lon": round(n.lon - 360, 4), "grib_value": val, "our_value": ours,
                             "equal": bool((np.isnan(val) and np.isnan(ours)) or abs(val - ours) < 1e-6)})
        finally:
            eccodes.codes_release(h)
    return pd.DataFrame(rows)


def check_accumulation(days, client: MRMSClient | None = None) -> pd.DataFrame:
    """A4: sum of 24 hourly Pass-2 fields vs the 24-h Pass-2 field valid at the next 00 UTC."""
    client = client or MRMSClient()
    rows = []
    for d in pd.DatetimeIndex(days):
        h = client.load(QPE, d + pd.Timedelta("1h"), d + pd.Timedelta("24h"), NYC)
        if h.sizes["time"] < 24:
            continue
        s24 = h.sum("time", min_count=24)
        q24 = client.load("MultiSensor_QPE_24H_Pass2", d + pd.Timedelta("24h"), d + pd.Timedelta("24h"),
                          NYC).isel(time=0)
        ok = np.isfinite(s24.values) & np.isfinite(q24.values)
        a, b = s24.values[ok], q24.values[ok]
        rows.append({"day": d.date(), "mean_sum_of_hourly": float(a.mean()), "mean_24h_product": float(b.mean()),
                     "rel_diff": float(a.mean() / b.mean() - 1) if b.mean() > 0 else np.nan,
                     "max_abs_diff_mm": float(np.abs(a - b).max()),
                     "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan})
    return pd.DataFrame(rows)


def check_completeness(start=PERIOD[0], end=PERIOD[1], client: MRMSClient | None = None) -> dict:
    """A5: hours expected vs present in the hourly Pass-2 record."""
    client = client or MRMSClient()
    h = client.load(QPE, pd.Timestamp(start) + pd.Timedelta("1h"), pd.Timestamp(end), NYC)
    expected = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta("1h"))
    missing = h.attrs.get("missing_times", [])
    return {"hours_expected": expected, "hours_present": int(h.sizes["time"]),
            "hours_missing": len(missing), "missing": missing,
            "cells_no_coverage_fraction": float(h.isnull().mean())}


# ------------------------------------------------------------------ part B


def official_hourly(start=PERIOD[0], end=PERIOD[1], stations=NYC_STATIONS, asos: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ASOS hourly precipitation (mm), hourly precipitation type, station coordinates."""
    a = asos if asos is not None else fetch_asos(pd.Timestamp(start) - pd.Timedelta("1D"),
                                                 pd.Timestamp(end) + pd.Timedelta("1D"), stations)
    a = a[(a.valid > pd.Timestamp(start) - pd.Timedelta("2h")) & (a.valid <= pd.Timestamp(end) + pd.Timedelta("1h"))]
    pr = hourly_precip(a).loc[pd.Timestamp(start) + pd.Timedelta("1h"):pd.Timestamp(end)]
    pt = hourly_ptype(a).reindex(pr.index).fillna("none")
    coords = a.groupby("station")[["lat", "lon"]].first()
    return pr, pt, coords


def _type_of_hour(ptype: str) -> str:
    return {"rain": "rain", "snow": "snow", "mix": "mix", "freezing": "mix"}.get(ptype, "none")


def radar_at_stations(start, end, coords: pd.DataFrame, product: str = QPE,
                      client: MRMSClient | None = None) -> pd.DataFrame:
    client = client or MRMSClient()
    h = client.load(product, pd.Timestamp(start) + pd.Timedelta("1h"), pd.Timestamp(end), NYC)
    s = h.sel(lat=xr.DataArray(coords.lat.values, dims="station"),
              lon=xr.DataArray(coords.lon.values, dims="station"), method="nearest")
    return pd.DataFrame(s.transpose("time", "station").values, index=pd.DatetimeIndex(h.time.values),
                        columns=coords.index)


def pws_at_stations(start, end, coords: pd.DataFrame, radius_m: float = 3000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PWS hourly (mm) at each station: nearest PWS within ``radius_m``, and IDW of all PWS."""
    pws = om.load_pws(pd.Timestamp(start), pd.Timestamp(end))
    r = pws["rain"].resample(time="1h", label="right", closed="right")
    h = r.sum(min_count=1).where(r.count() >= 10)
    near, idw = {}, {}
    for st, c in coords.iterrows():
        d = haversine_m(c.lat, c.lon, pws.lat.values, pws.lon.values)
        i = int(np.argmin(d))
        near[st] = h.isel(station=i).to_series() if d[i] <= radius_m else pd.Series(np.nan, index=h.time.values)
        w = 1.0 / np.maximum(d, 50.0) ** 2
        w[d > 10_000] = 0
        v = h.values                                         # (station, time)
        ok = np.isfinite(v)
        num = (np.nan_to_num(v) * w[:, None]).sum(0)
        den = (ok * w[:, None]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            idw[st] = pd.Series(np.where(den > 0, num / den, np.nan), index=pd.DatetimeIndex(h.time.values))
    return pd.DataFrame(near), pd.DataFrame(idw)


def score_vs_official(sources: dict, official: pd.DataFrame, ptype: pd.DataFrame,
                      wet_threshold: float = 0.1) -> pd.DataFrame:
    """Scores of every source against the official gauges, per hour type and station."""
    rows = []
    for name, df in sources.items():
        for st in official.columns:
            if st not in df.columns:
                continue
            idx = official.index.intersection(df.index)
            o, e = official.loc[idx, st], df.loc[idx, st]
            t = ptype.loc[idx, st].map(_type_of_hour) if st in ptype.columns else pd.Series("none", index=idx)
            # dry hours (no precipitation reported, no type) count as "all" only
            for typ in ("all", "rain", "snow", "mix"):
                m = (t == typ) if typ != "all" else pd.Series(True, index=idx)
                s = scores(e[m].values, o[m].values, wet_threshold)
                if s.get("n", 0) == 0:
                    continue
                ok = np.isfinite(e[m].values) & np.isfinite(o[m].values)
                rows.append({"source": name, "station": st, "hours": typ, **s,
                             "total_official_mm": float(o[m].values[ok].sum()),
                             "total_source_mm": float(e[m].values[ok].sum())})
    return pd.DataFrame(rows)


def pooled_vs_official(sources: dict, official: pd.DataFrame, ptype: pd.DataFrame,
                       wet_threshold: float = 0.1, common: bool = True) -> pd.DataFrame:
    """Scores pooled over stations, per hour type.

    With ``common`` (default) every source is scored on the same station-hours - those where
    the official gauge and *all* sources have a value - so the ranking is fair; ``coverage``
    reports the share of official station-hours each source has at all.
    """
    rows = []
    stations = [st for st in official.columns if all(st in df.columns for df in sources.values())]
    for typ in ("all", "rain", "snow", "mix"):
        blocks = {name: [] for name in sources}
        off_blocks, cov = [], {name: [0, 0] for name in sources}
        for st in stations:
            idx = official.index
            t = ptype.reindex(idx)[st].map(_type_of_hour) if st in ptype.columns else pd.Series("none", index=idx)
            m = ((t == typ) if typ != "all" else pd.Series(True, index=idx)) & official[st].notna()
            vals = {name: df[st].reindex(idx)[m] for name, df in sources.items()}
            for name, v in vals.items():
                cov[name][0] += int(v.notna().sum())
                cov[name][1] += int(m.sum())
            keep = pd.concat(vals.values(), axis=1).notna().all(axis=1) if common else pd.Series(True, index=idx[m])
            off_blocks.append(official[st][m][keep].values)
            for name, v in vals.items():
                blocks[name].append(v[keep].values)
        o = np.concatenate(off_blocks) if off_blocks else np.array([])
        for name in sources:
            e = np.concatenate(blocks[name]) if blocks[name] else np.array([])
            sc = scores(e, o, wet_threshold)
            if sc.get("n", 0):
                ok = np.isfinite(e) & np.isfinite(o)
                rows.append({"hours": typ, "source": name, "coverage": cov[name][0] / max(cov[name][1], 1), **sc,
                             "total_official_mm": float(o[ok].sum()), "total_source_mm": float(e[ok].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver


@dataclass
class Validation:
    sources_check: pd.DataFrame
    decoder_check: pd.DataFrame
    georef_check: pd.DataFrame
    accumulation_check: pd.DataFrame
    completeness: dict
    period_scores: pd.DataFrame           # whole record: radar / PWS vs official, per station
    period_pooled: pd.DataFrame
    products_pooled: pd.DataFrame         # Pass2 / Pass1 / radar-only on catalog events
    cml_official: pd.DataFrame            # CML maps (study) at the official gauges
    series: dict = field(default_factory=dict)

    def write_report(self, out_dir: str | Path) -> Path:
        from . import plots as plotting
        out = Path(out_dir)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        for name in ["sources_check", "decoder_check", "georef_check", "accumulation_check", "period_scores",
                     "period_pooled", "products_pooled", "cml_official"]:
            df = getattr(self, name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.round(4).to_csv(out / f"{name}.csv", index=False)
        plotting.validation_cumulative_figure(self, out / "figures" / "cumulative_at_official_gauges.png")
        plotting.validation_scatter_figure(self, out / "figures" / "hourly_vs_official.png")
        plotting.validation_accumulation_figure(self, out / "figures" / "hourly_vs_24h_product.png")
        (out / "README.md").write_text(self._markdown())
        return out

    def _markdown(self) -> str:
        c = self.completeness
        sc, dc, gc, ac = self.sources_check, self.decoder_check, self.georef_check, self.accumulation_check
        L = ["# Data validation: radar fetching and all sources vs the official gauges", "",
             "Generated by `python src/run.py validate` (`nyc_rain_maps.validation`). Official reference: NWS ASOS hourly "
             "precipitation (routine METAR, report nearest :51) at Central Park (NYC), LaGuardia (LGA), "
             "JFK and Newark (EWR); hours typed by each station's own present-weather report.", "",
             "## A. Is the radar data what NOAA published, where it belongs?", "",
             "| check | what | result |", "|---|---|---|",
             f"| A1 sources | {len(sc)} files fetched separately from NOAA AWS and IEM | "
             f"{int(sc.identical_grib.sum())}/{len(sc)} byte-identical GRIB after gunzip; max field difference "
             f"{sc.max_abs_diff.max():.3g} mm |",
             f"| A2 decoder | our ecCodes crop vs independent full cfgrib decode, {len(dc)} files x "
             f"{int(dc.cells.iloc[0]) if len(dc) else 0} cells | max difference {dc.max_abs_diff.max():.3g} mm; "
             f"missing-data mask identical in {int(dc.same_missing_mask.sum())}/{len(dc)}; valid time correct in "
             f"{int(dc.valid_time_ok.sum())}/{len(dc)} |",
             f"| A3 georeferencing | cached value at each ASOS station vs ecCodes nearest grid point, "
             f"{len(gc)} station-hours | {int(gc.equal.sum())}/{len(gc)} equal |",
             f"| A4 accumulation | 24 hourly Pass-2 fields vs the 24-h Pass-2 product, {len(ac)} wet days | "
             f"domain-mean difference {ac.rel_diff.abs().median():.1%} median, "
             f"{ac.rel_diff.abs().max():.1%} worst; cell correlation {ac['corr'].median():.3f} median |",
             f"| A5 completeness | hourly Pass 2, {PERIOD[0]} to {PERIOD[1]} | {c['hours_present']}/"
             f"{c['hours_expected']} hours present ({c['hours_missing']} missing on every archive); "
             f"{c['cells_no_coverage_fraction']:.2%} of cell-hours without radar coverage |", "",
             "A4 matches to the last digit because MRMS builds its 24-h Pass-2 product from its hourly "
             "Pass-2 fields; the exact match also confirms our hour-ending time labels (a one-hour shift "
             "would break it).", "",
             "![hourly vs 24h](figures/hourly_vs_24h_product.png)", "",
             "## B. Every source against the official gauges", "",
             "### Whole OpenMesh record (hourly, pooled over the four stations)", "",
             "Every source is scored on the **same station-hours** (official gauge and all sources "
             "valid); *coverage* is the share of official station-hours a source has at all.", "",
             "| hours | source | coverage | station-hours | official total mm | source total mm | rel. bias | NRMSE "
             "| corr | POD | FAR | CSI |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        pp = self.period_pooled.assign(_t=self.period_pooled.hours.map({"all": 0, "rain": 1, "mix": 2, "snow": 3}))
        for r in pp.sort_values(["_t", "nrmse"]).itertuples():
            L.append(f"| {r.hours} | {r.source} | {r.coverage:.0%} | {r.n:,} | {r.total_official_mm:.0f} | {r.total_source_mm:.0f} | "
                     f"{r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} | {r.pod:.2f} | {r.far:.2f} | {r.csi:.2f} |")
        L += ["", "![cumulative](figures/cumulative_at_official_gauges.png)", "",
              "![scatter](figures/hourly_vs_official.png)", ""]
        if not self.products_pooled.empty:
            L += ["### MRMS products on the catalog events (all 15, pooled over stations)", "",
                  "| hours | product | station-hours | rel. bias | NRMSE | corr |", "|---|---|---|---|---|---|"]
            q = self.products_pooled.assign(_t=self.products_pooled.hours.map({"all": 0, "rain": 1, "mix": 2, "snow": 3}))
            for r in q.sort_values(["_t", "nrmse"]).itertuples():
                L.append(f"| {r.hours} | {r.source} | {r.n:,} | {r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} |")
            L.append("")
        if not self.cml_official.empty:
            L += ["### CML maps at the official gauges (study events, stations inside the link network's reach)", "",
                  "| type | map | station-hours | rel. bias | NRMSE | corr |", "|---|---|---|---|---|---|"]
            k = self.cml_official.assign(_t=self.cml_official.ptype.map({"rain": 0, "mix": 1, "snow": 2}))
            for r in k.sort_values(["_t", "nrmse"]).itertuples():
                L.append(f"| {r.ptype} | {r.map} | {r.n} | {r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} |")
            L.append("")
        L += ["## Per station (whole record, all hours)", "",
              "| station | source | station-hours | rel. bias | NRMSE | corr |", "|---|---|---|---|---|---|"]
        ps = self.period_scores[self.period_scores.hours == "all"]
        for r in ps.sort_values(["station", "nrmse"]).itertuples():
            L.append(f"| {r.station} | {r.source} | {r.n:,} | {r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} |")
        L += ["", "## Reading the numbers", "", *self._findings(), "",
              "## Caveats", "",
              "- ASOS heated tipping buckets under-catch snow and wind-driven precipitation; in snow and mixed "
              "hours the \"official\" value is itself low.",
              "- A point gauge against a 1 km radar cell or an interpolated map includes representativeness "
              "error even when both are perfect; that sets a floor on every NRMSE here.",
              "- METAR hourly precipitation is the routine report's `p01m`; hours with a missing routine "
              "report are excluded, never zero-filled.", ""]
        return "\n".join(L)

    def _findings(self) -> list:
        out = []
        pp = self.period_pooled
        for typ in ("rain", "snow", "mix"):
            s = pp[pp.hours == typ].sort_values("nrmse")
            if s.empty:
                continue
            best = s.iloc[0]
            out.append(f"- **{typ} hours:** closest to the official gauges is `{best.source}` (NRMSE "
                       f"{best.nrmse:.2f}, bias {best.rel_bias:+.0%}); "
                       + "; ".join(f"`{r.source}` {r.nrmse:.2f} ({r.rel_bias:+.0%})" for r in s.iloc[1:].itertuples()) + ".")
        return out


def _cml_at_official() -> pd.DataFrame:
    from .settings import repo_reports_dir
    p = repo_reports_dir() / "study" / "asos_point_scores.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def run_validation(n_files: int = 12, n_days: int = 20, client: MRMSClient | None = None) -> Validation:
    """Run parts A and B (network needed for A1-A3)."""
    client = client or MRMSClient()
    official, ptype, coords = official_hourly()
    wet_hours = official.index[(official.fillna(0) > 1.0).any(axis=1)]
    rng = np.random.default_rng(0)
    sample = pd.DatetimeIndex(sorted(rng.choice(wet_hours, size=min(n_files, len(wet_hours)), replace=False)))
    log.info("A1-A3 on %d wet hours", len(sample))
    sources = check_sources(sample[: max(4, n_files // 2)])
    decoder = check_decoder(sample[: max(4, n_files // 2)])
    georef = check_georef(sample, coords, client=client)
    daily = official.resample("1D", label="left", closed="right").sum(min_count=12).max(axis=1)
    days = pd.DatetimeIndex(daily[daily > 5].index[:n_days])
    accum = check_accumulation(days, client=client)
    complete = check_completeness(client=client)

    radar = radar_at_stations(*PERIOD, coords, client=client)
    pws_near, pws_idw = pws_at_stations(*PERIOD, coords)
    src = {"MRMS Pass 2": radar, "PWS nearest (<3 km)": pws_near, "PWS IDW": pws_idw}
    period_scores = score_vs_official(src, official, ptype)
    period_pooled = pooled_vs_official(src, official, ptype)

    # MRMS products on the catalog events
    from .settings import repo_events_dir
    cat = pd.read_csv(repo_events_dir() / "all_detected_events.csv")
    cat = cat[cat.selected.astype(bool)]
    prods = {"MRMS Pass 2": [], "MRMS Pass 1": [], "MRMS radar-only": []}
    names = {"MRMS Pass 2": QPE, "MRMS Pass 1": "MultiSensor_QPE_01H_Pass1", "MRMS radar-only": "RadarOnly_QPE_01H"}
    for ev in cat.itertuples():
        for lab, prod in names.items():
            try:
                prods[lab].append(radar_at_stations(ev.start, ev.end, coords, product=prod, client=client))
            except Exception as exc:
                log.warning("%s %s: %s", prod, ev.event_id, exc)
    long_record = fetch_asos("2020-11-01", "2026-05-01")          # the event catalog's ASOS record
    offs, pts = [], []
    for ev in cat.itertuples():
        o, t, _ = official_hourly(ev.start, ev.end, asos=long_record)
        offs.append(o)
        pts.append(t)
    ev_off = pd.concat(offs).sort_index()
    ev_pt = pd.concat(pts).sort_index()
    ev_off, ev_pt = ev_off[~ev_off.index.duplicated()], ev_pt[~ev_pt.index.duplicated()]
    prod_src = {k: pd.concat(v).sort_index() for k, v in prods.items() if v}
    prod_src = {k: v[~v.index.duplicated()] for k, v in prod_src.items()}
    products_pooled = pooled_vs_official(prod_src, ev_off, ev_pt)

    series = {"official": official, "MRMS Pass 2": radar, "PWS nearest (<3 km)": pws_near, "PWS IDW": pws_idw,
              "ptype": ptype}
    return Validation(sources, decoder, georef, accum, complete, period_scores, period_pooled, products_pooled,
                      _cml_at_official(), series)
