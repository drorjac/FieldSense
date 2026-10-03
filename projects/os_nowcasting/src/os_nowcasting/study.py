"""The study: every product of every event nowcast at many issue times, verified three ways.

For each network, event and issue time (hourly, while at least 5% of the domain has rain
on the radar), and for each product (radar, link maps, merged, PWS):

1. motion by Lucas-Kanade on the product's own last four fields (dBR);
2. the deterministic nowcasts (persistence, extrapolation, S-PROG, ANVIL; LINDA on the
   radar) and the STEPS ensemble, to ``n_leadtimes`` steps (90 min);
3. each lead time scored, on the cells the product's motion can reach from inside the
   domain (rain advected in from outside is unknown to every method; on OpenMRG's
   88 x 70 km domain half of it is gone within 45 min), against
   ``own``     the product's own field at that time - can the product be nowcast?
   ``radar``   the radar at that time - how good a rain forecast is it?
   ``gauges``  the independent gauges (never used to make any product), at their cells.

Two side studies on the radar: the four motion methods (and LK without the dB transform)
compared by the skill of the extrapolation nowcast they drive; and how far each product's
motion field is from the radar's. On OpenMRG, hourly totals with and without advection
interpolation (block 03a) are scored at the city gauges.

Scores are pooled over all issue times of a network (``verify``). State is checkpointed
after each event under ``CACHE_DIR/<network>/``, so an interrupted run resumes.
"""

from __future__ import annotations

import logging
import pickle
import time

import numpy as np
import pandas as pd
import xarray as xr

from . import data, interp, nowcast, products, verify
from .grid import pixel_km, to_pysteps
from .settings import (CACHE_DIR, DETERMINISTIC, ENSEMBLE, EVENTS, ISSUE_EVERY, LINDA_PRODUCTS,
                       MIN_WET_FRACTION, N_ENS_MEMBERS, N_PAST, NETWORKS, PRODUCTS)

log = logging.getLogger(__name__)

HISTORY_STEPS = max(N_PAST.values())          # fields needed before an issue time
SAL_LEADS_MIN = (30, 60)
# LINDA is slow (a minute or more per nowcast in heavy rain on OpenRainER's grid): the radar
# only, LINDA on every 3rd issue time and LINDA-P on every 6th. On those issue times the other
# methods are scored a second time under the product name "radar (LINDA subset)" /
# "radar (LINDA-P subset)", so LINDA is compared with them on the same forecasts.
LINDA_EVERY = 3
LINDA_P_EVERY = 6
REFERENCES = ("own", "radar", "gauges")


class NetworkStudy:
    """Accumulated scores of one network; picklable, so it can be checkpointed."""

    def __init__(self, network: str):
        self.network = network
        s = NETWORKS[network]
        self.px = pixel_km(data.grid_for(network))
        self.det = verify.Collector(lambda: verify.Deterministic(self.px))
        self.det_points = verify.Collector(lambda: verify.Deterministic(self.px, gridded=False))
        self.ens = verify.Collector(lambda: verify.Ensemble(N_ENS_MEMBERS))
        self.ens_points = verify.Collector(lambda: verify.Ensemble(N_ENS_MEMBERS))
        self.motion = verify.Collector(lambda: verify.Deterministic(self.px))
        self.motion_diff = []                 # rows: product vs radar motion, per issue time
        self.accum = []                       # rows: hourly totals at gauges, plain vs interpolated
        self.issues = []                      # rows: one per issue time
        self.reach = []                       # rows: share of the domain verifiable, per product
        self.done = []                        # finished events
        self.step = s.step_min

    # ------------------------------------------------------------------ pickling
    def __getstate__(self):
        d = self.__dict__.copy()
        for k in ("det", "det_points", "ens", "ens_points", "motion"):
            d[k] = dict(d[k].acc)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        px = d["px"]
        makers = {"det": lambda: verify.Deterministic(px),
                  "det_points": lambda: verify.Deterministic(px, gridded=False),
                  "ens": lambda: verify.Ensemble(N_ENS_MEMBERS),
                  "ens_points": lambda: verify.Ensemble(N_ENS_MEMBERS),
                  "motion": lambda: verify.Deterministic(px)}
        for k, f in makers.items():
            c = verify.Collector(f)
            c.acc.update(d[k])
            setattr(self, k, c)

    @property
    def path(self):
        return CACHE_DIR / self.network / "study_state.pkl"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_or_new(cls, network: str) -> "NetworkStudy":
        p = CACHE_DIR / network / "study_state.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return cls(network)

    # ------------------------------------------------------------------ tables
    def tables(self) -> dict:
        names = ("product", "method", "reference", "lead_min")
        det = pd.DataFrame(self.det.rows(names) + self.det_points.rows(names))
        ens = pd.DataFrame(self.ens.rows(names) + self.ens_points.rows(names))
        mot = pd.DataFrame(self.motion.rows(("motion", "lead_min")))
        for df in (det, ens, mot):
            df.insert(0, "network", self.network)
        return {"deterministic": det, "ensemble": ens, "motion": mot,
                "motion_vs_radar": pd.DataFrame(self.motion_diff),
                "accumulation": pd.DataFrame(self.accum), "issues": pd.DataFrame(self.issues),
                "reach": pd.DataFrame(self.reach)}


def issue_times(radar: xr.DataArray, start, end) -> pd.DatetimeIndex:
    """Hourly issue times in ``[start + history, end]`` when the radar is wet enough."""
    step = pd.Timedelta(minutes=int(radar.attrs.get("step", "15min").removesuffix("min")))
    t0 = pd.Timestamp(start) + HISTORY_STEPS * step
    cand = pd.date_range(t0.ceil(ISSUE_EVERY), pd.Timestamp(end), freq=ISSUE_EVERY)
    cand = cand[cand.isin(pd.DatetimeIndex(radar.time.values))]
    rate = radar.sel(time=cand) * (60.0 / (step / pd.Timedelta("1min")))
    wet = (rate > 0.1).sum(("lat", "lon")) / np.isfinite(rate).sum(("lat", "lon"))
    return cand[wet.values >= MIN_WET_FRACTION]


reachable = nowcast.reachable


def gauge_cells(g: xr.DataArray, grid) -> np.ndarray:
    i = np.abs(grid.lat[None, :] - g.lat.values[:, None]).argmin(1)
    j = np.abs(grid.lon[None, :] - g.lon.values[:, None]).argmin(1)
    return i * grid.lon.size + j


def run_event(st: NetworkStudy, event_id: str, start, end):
    s = NETWORKS[st.network]
    step = pd.Timedelta(minutes=s.step_min)
    t_load0 = pd.Timestamp(start) - (HISTORY_STEPS + 1) * step
    t_load1 = pd.Timestamp(end) + s.n_leadtimes * step
    radar = data.radar(st.network, t_load0, t_load1)
    t = time.time()
    prods = products.build(st.network, t_load0, t_load1, radar)
    prods.pop("_links", None)
    log.info("%s %s: products %s in %.0f s", st.network, event_id, list(prods), time.time() - t)
    grid = data.grid_for(st.network)
    g = data.points(st.network, t_load0, t_load1, s.gauges).reindex(time=radar.time.values)
    cells = gauge_cells(g, grid)
    g_rate = g.transpose("station", "time").values.T * (60.0 / s.step_min)      # (time, station)

    rates = {}
    for p in PRODUCTS[st.network]:
        if p in prods:
            rates[p], meta = to_pysteps(prods[p], s.step_min, fill=None)
    obs_radar = rates["radar"]
    times = pd.DatetimeIndex(radar.time.values)
    issues = issue_times(radar, start, end)
    log.info("%s %s: %d issue times", st.network, event_id, len(issues))

    for k, ti in enumerate(issues):
        i = times.get_loc(ti)
        if i + s.n_leadtimes >= len(times):
            continue
        t_issue = time.time()
        leads = range(1, s.n_leadtimes + 1)
        future = slice(i + 1, i + 1 + s.n_leadtimes)
        v_radar = None
        for p, rate in rates.items():
            past = np.nan_to_num(rate[i - HISTORY_STEPS + 1:i + 1], nan=0.0)
            v = nowcast.motion(past, meta, "LK")
            if p == "radar":
                v_radar = v
            # cells the motion can reach from inside the domain; elsewhere rain flows in from
            # outside and no nowcast knows it, so every method is scored on these cells only
            reach = reachable(v, past.shape[1:], s.n_leadtimes)
            refs = {"own": np.where(reach, rate[future], np.nan),
                    "radar": np.where(reach, obs_radar[future], np.nan)}
            g_ref = np.where(verify.at_points(reach, cells), g_rate[future], np.nan)      # (lead, station)
            st.reach.append({"network": st.network, "event": event_id, "issue": ti, "product": p,
                             **{f"reach_{lead * s.step_min}": float(reach[lead - 1].mean())
                                for lead in (3 if s.step_min == 15 else 9, s.n_leadtimes)}})
            linda = p in LINDA_PRODUCTS and k % LINDA_EVERY == 0
            methods = list(DETERMINISTIC) + (["linda"] if linda else [])
            for m in methods:
                try:
                    f = nowcast.deterministic(m, past, meta, v, s.n_leadtimes)
                except Exception as exc:                    # one failure must not stop a run
                    log.warning("%s %s %s %s: %r", st.network, ti, p, m, exc)
                    continue
                for li, lead in enumerate(leads):
                    lm = lead * s.step_min
                    names = [p] + ([f"{p} (LINDA subset)"] if linda else [])
                    for name in names:
                        for ref, obs in refs.items():
                            st.det[(name, m, ref, lm)].add(f[li], obs[li], sal=lm in SAL_LEADS_MIN)
                        st.det_points[(name, m, "gauges", lm)].add(verify.at_points(f[li], cells),
                                                                   g_ref[li])
            linda_p = p in LINDA_PRODUCTS and k % LINDA_P_EVERY == 0
            ens_methods = [ENSEMBLE] + (["linda_p"] if linda_p else [])
            for m in ens_methods:
                try:
                    e = nowcast.ensemble(m, past, meta, v, s.n_leadtimes, n_members=N_ENS_MEMBERS)
                except Exception as exc:
                    log.warning("%s %s %s %s: %r", st.network, ti, p, m, exc)
                    continue
                for li, lead in enumerate(leads):
                    lm = lead * s.step_min
                    names = [p] + ([f"{p} (LINDA-P subset)"] if linda_p else [])
                    for name in names:
                        for ref, obs in refs.items():
                            st.ens[(name, m, ref, lm)].add(e[:, li], obs[li])
                        st.ens_points[(name, m, "gauges", lm)].add(verify.at_points(e[:, li], cells),
                                                                   g_ref[li])
            if p != "radar" and v_radar is not None:
                wet = past[-1] > 0.1
                d = np.hypot(*(v - v_radar)) * st.px * 60.0 / s.step_min        # km/h
                sp = np.hypot(*v_radar) * st.px * 60.0 / s.step_min
                st.motion_diff.append({"network": st.network, "event": event_id, "issue": ti,
                                       "product": p, "wet_cells": int(wet.sum()),
                                       "mean_diff_kmh": float(d[wet].mean()) if wet.any() else np.nan,
                                       "radar_speed_kmh": float(sp[wet].mean()) if wet.any() else np.nan})
        # motion methods on the radar: the extrapolation nowcast each one drives
        # (scored on the cells all five motion fields can reach)
        past = np.nan_to_num(obs_radar[i - HISTORY_STEPS + 1:i + 1], nan=0.0)
        fc, reach_all = {}, np.ones((s.n_leadtimes,) + past.shape[1:], bool)
        for mname, method, tr in [("LK", "LK", True), ("LK (no dB)", "LK", False), ("VET", "VET", True),
                                  ("DARTS", "DARTS", True), ("proesmans", "proesmans", True)]:
            try:
                v = nowcast.motion(past, meta, method, transform=tr)
                fc[mname] = nowcast.deterministic("extrapolation", past, meta, v, s.n_leadtimes)
                reach_all &= reachable(v, past.shape[1:], s.n_leadtimes)
            except Exception as exc:
                log.warning("motion %s at %s: %r", mname, ti, exc)
        for mname, f in fc.items():
            for li, lead in enumerate(leads):
                st.motion[(mname, lead * s.step_min)].add(f[li], np.where(reach_all[li], obs_radar[i + lead], np.nan))
        st.issues.append({"network": st.network, "event": event_id, "issue": ti,
                          "seconds": round(time.time() - t_issue, 1)})

    if st.network == "openmrg":
        accumulation_check(st, event_id, rates["radar"], meta, times, g, cells)


def accumulation_check(st: NetworkStudy, event_id, rate, meta, times, g, cells):
    """Hourly totals from 5-min scans, plain vs advection-interpolated, at the gauges."""
    per_hour = 60 // st.step
    hours = times[(times.minute == 0)]
    g_h = g.resample(time="1h", closed="right", label="right").sum(min_count=per_hour)
    for h in hours:
        i = times.get_loc(h)
        if i < per_hour or h not in g_h.time.values:
            continue
        scans = np.nan_to_num(rate[i - per_hour:i + 1], nan=0.0)
        if scans.max() <= 0.1:
            continue
        plain = interp.hourly_total(scans, meta, interpolate=False)
        adv = interp.hourly_total(scans, meta, interpolate=True)
        obs = g_h.sel(time=h).values
        for name, fld in [("plain", plain), ("advection", adv)]:
            for sid, c, o in zip(g.station.values, cells, obs):
                st.accum.append({"network": st.network, "event": event_id, "hour": h, "station": sid,
                                 "method": name, "estimate_mm": float(fld.ravel()[c]),
                                 "gauge_mm": float(o)})


def run_network(network: str, events=None, fresh: bool = False) -> NetworkStudy:
    st = NetworkStudy(network) if fresh else NetworkStudy.load_or_new(network)
    for ev_id, start, end in (events or EVENTS[network]):
        if ev_id in st.done:
            log.info("%s %s already done", network, ev_id)
            continue
        t = time.time()
        run_event(st, ev_id, start, end)
        st.done.append(ev_id)
        st.save()
        log.info("%s %s done in %.0f s", network, ev_id, time.time() - t)
    return st
