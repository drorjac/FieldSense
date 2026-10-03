"""The largest events of each network, every sensor on one grid, pooled per network.

    from multisensor_maps.study import run_study
    study = run_study()                 # -> results/<network>/..., results/report.md

Pooled scores weigh every cell-hour (maps) or station-hour (points) equally across a
network's events, and use only samples every compared map has for that event, so a
method cannot improve its score by leaving gaps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.events import detect_events
from core.maps.scores import scores
from core.opensense.networks import NETWORKS

from .event import EventResult, loo_at_points, run_event, sample_at
from .settings import (CHECK_POINTS, METHODS, N_EVENTS, OPENRAINER_SECOND_SEASON, PERIODS,
                       RESULTS_DIR)

log = logging.getLogger(__name__)


def network_events(network: str, n: int = N_EVENTS) -> pd.DataFrame:
    """The ``n`` largest events of a network's study period, by domain-mean radar total."""
    net = NETWORKS[network]
    periods = [PERIODS[network]] + ([OPENRAINER_SECOND_SEASON] if network == "openrainer" else [])
    found = pd.concat([detect_events(net.radar_hourly(*p)) for p in periods], ignore_index=True)
    found = found.sort_values("total_mm", ascending=False).head(n)
    return found.sort_values("start").reset_index(drop=True).assign(network=network)


def _map_pairs(res: EventResult, near_km: float) -> list[pd.DataFrame]:
    """Jointly valid cell-hours of every (estimate, reference) pair, near links or all."""
    t = res._times()
    mask = (res.distance_km <= near_km).values if near_km is not None else np.ones(res.distance_km.shape, bool)
    pairs = res.pairwise(near_km=None).loc[:, ["estimate", "reference"]].itertuples(index=False)
    vals = {m: res.maps[m].sel(time=t).values[:, mask] for m in res.maps}
    common = _common(vals)
    out = []
    for est, ref in pairs:
        sel = common & np.isfinite(vals[est]) & np.isfinite(vals[ref])
        out.append(pd.DataFrame({"est": vals[est][sel], "ref": vals[ref][sel], "estimate": est,
                                 "reference": ref, "base": int(common.sum())}))
    return out


def _common(vals: dict, min_coverage: float = 0.9) -> np.ndarray:
    """Samples every *full-coverage* map has. A map with gaps (under ``min_coverage`` of the
    best-covered one) neither shrinks this sample nor chooses its own: it is scored on its
    share of it, and its coverage is reported."""
    n = {k: int(np.isfinite(v).sum()) for k, v in vals.items()}
    full = [k for k in vals if n[k] >= min_coverage * max(n.values())]
    return np.all([np.isfinite(vals[k]) for k in full], axis=0)


def _point_pairs(res: EventResult) -> list[pd.DataFrame]:
    check = CHECK_POINTS[res.network]
    g = res.points[check]
    t = np.intersect1d(res._times(), g.time.values)
    ref = g.sel(time=t).values
    est = {f"{check} (leave-one-out)": loo_at_points(g).sel(time=t).values}
    for name, m in res.maps.items():
        if name != check:
            est[name] = sample_at(m.sel(time=t), g.lat.values, g.lon.values).values
    near = sample_at(res.distance_km.expand_dims(time=[0]), g.lat.values, g.lon.values).values[:, 0] <= 5.0
    common = np.isfinite(ref) & _common(est)
    rows = []
    for k, v in est.items():
        for sub, base in (("all gauges", common), ("gauges <= 5 km of a link", common & near[:, None])):
            sel = base & np.isfinite(v)
            rows.append(pd.DataFrame({"est": v[sel], "ref": ref[sel], "estimate": k, "subset": sub,
                                      "base": int(base.sum())}))
    return rows


@dataclass
class Study:
    events: pd.DataFrame
    event_scores: pd.DataFrame
    link_scores: pd.DataFrame
    pooled_maps: pd.DataFrame
    pooled_points: pd.DataFrame
    errors: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)       # network -> EventResult (largest event)

    def save(self, out=RESULTS_DIR):
        out.mkdir(parents=True, exist_ok=True)
        self.events.to_csv(out / "events.csv", index=False)
        self.event_scores.round(4).to_csv(out / "event_scores.csv", index=False)
        self.link_scores.round(4).to_csv(out / "link_scores.csv", index=False)
        self.pooled_maps.round(4).to_csv(out / "pooled_map_scores.csv", index=False)
        self.pooled_points.round(4).to_csv(out / "pooled_point_scores.csv", index=False)
        (out / "errors.json").write_text(json.dumps(self.errors, indent=2))
        return out


def run_study(networks=("openmrg", "openrainer", "openmesh"), methods=METHODS, extra_factory=None,
              near_km: float = 2.0) -> Study:
    """``extra_factory(network) -> {name: f(links)}`` adds methods from elsewhere (the RNN)."""
    ev_rows, ev_scores, link_rows, errors, examples = [], [], [], {}, {}
    map_pairs, point_pairs = {}, {}
    for network in networks:
        events = network_events(network)
        ev_rows.append(events)
        extra = extra_factory(network) if extra_factory else None
        for ev in events.itertuples():
            label = f"{network} {pd.Timestamp(ev.start):%Y-%m-%d %H}"
            log.info("study: %s", label)
            try:
                res = run_event(network, ev.start, ev.end, methods, extra=extra)
            except Exception as exc:
                errors[label] = repr(exc)
                log.exception("event failed: %s", label)
                continue
            for k, v in res.errors.items():
                errors[f"{label} / {k}"] = v
            for near in (near_km, None):
                pw = res.pairwise(near_km=near).assign(event=label)
                ev_scores.append(pw)
                for df in _map_pairs(res, near):
                    map_pairs.setdefault((network, near), []).append(df.assign(event=label))
            link_rows.append(res.link_check().assign(event=label))
            for df in _point_pairs(res):
                point_pairs.setdefault(network, []).append(df.assign(event=label))
            if ev.total_mm == events.total_mm.max():
                examples[network] = res
    pooled_maps, pooled_points = [], []
    for (network, near), frames in map_pairs.items():
        df = pd.concat(frames, ignore_index=True)
        for (e, r), g in df.groupby(["estimate", "reference"], sort=False):
            base = g.groupby("event").base.first().sum()
            pooled_maps.append({"network": network, "estimate": e, "reference": r,
                                "cells": f"<= {near_km:g} km of a link" if near else "all",
                                "events": g.event.nunique(), "coverage": len(g) / max(base, 1),
                                **scores(g.est, g.ref)})
    for network, frames in point_pairs.items():
        df = pd.concat(frames, ignore_index=True)
        for (e, sub), g in df.groupby(["estimate", "subset"], sort=False):
            base = g.groupby("event").base.first().sum()
            pooled_points.append({"network": network, "check": CHECK_POINTS[network], "estimate": e,
                                  "subset": sub, "events": g.event.nunique(),
                                  "coverage": len(g) / max(base, 1), **scores(g.est, g.ref)})
    return Study(pd.concat(ev_rows, ignore_index=True),
                 pd.concat(ev_scores, ignore_index=True) if ev_scores else pd.DataFrame(),
                 pd.concat(link_rows, ignore_index=True) if link_rows else pd.DataFrame(),
                 pd.DataFrame(pooled_maps), pd.DataFrame(pooled_points), errors, examples)
