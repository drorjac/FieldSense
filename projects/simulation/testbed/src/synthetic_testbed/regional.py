"""The regional study: every method on random 64 km scenarios, scored against the truth.

Two scenario sets:

``mixed``      ``core.simulation.scenario.random_scenario`` - every rain model, flow, evolution
               and radar band at random
``lifecycle``  the three cell models only, every scenario with cells that are born, grow and
               decay (``evolution="lifecycle"``, lifetimes 30-90 min) - convective rain, the
               hardest case, which the mixed set draws only now and then

In each scenario: every map and merging method (``benchmark.maps``), every motion method on
four products at several issue times, and nowcasts from the truth, the radar and the KED map.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from core.simulation import benchmark as bm
from core.simulation.scenario import random_scenario

from .settings import CELL_MODELS, MOTION_PRODUCTS, NOWCAST_PRODUCTS

SETS = ("mixed", "lifecycle")


def scenario(seed: int, scenario_set: str = "mixed"):
    if scenario_set == "mixed":
        return random_scenario(seed)
    if scenario_set == "lifecycle":
        rng = np.random.default_rng(20_000 + seed)
        return random_scenario(seed, model=str(rng.choice(CELL_MODELS)), evolution="lifecycle",
                               evolution_params={"lifetime_min": float(rng.uniform(30, 90))})
    raise ValueError(f"scenario_set must be one of {SETS}")


def study(n_scenarios: int = 24, first_seed: int = 100, nets=None, steps_members: int = 8,
          issue_every: int = 6, scenario_set: str = "mixed", log=print, checkpoint=None):
    """Maps, motion and nowcasts on ``n_scenarios`` scenarios of ``scenario_set``.

    ``nets``: ``{label: network}`` learned motion estimators, scored next to the classical
    ones and used for extrapolation nowcasts. Scenario seeds come from ``random_scenario``'s
    own stream, separate from the training simulations. Returns ``(maps, motion, nowcast,
    scenarios)`` data frames; ``checkpoint`` receives them after every scenario.
    """
    from core.nowcast import learned_motion as lm
    from core.nowcast.verify import Collector, Deterministic
    map_rows, motion_rows, scen_rows = [], [], []
    col = None
    for s in range(first_seed, first_seed + n_scenarios):
        t0 = time.time()
        sc = scenario(s, scenario_set)
        try:
            case = sc.run()
        except Exception as e:
            log(f"scenario {s}: simulation failed: {e!r}")
            continue
        t_sim = time.time() - t0
        timings = {}
        products = bm.maps(case, timings=timings)
        df = bm.score_maps(case, products)
        df["scenario"] = s
        map_rows.append(df)
        t_maps = time.time() - t0 - t_sim

        # motion: from 4 past intervals, at several issue times, every product, every method
        meta = bm.nowcast_meta(case)
        rates = {p: (case.rate if p == "truth" else case.as_rate(products[p]))
                 for p in MOTION_PRODUCTS if p == "truth" or products.get(p) is not None}
        px_kmh = case.pixel_km * 60.0 / case.interval_min
        issues = list(range(8, case.truth.sizes["time"] - 12, issue_every))
        for t in issues:
            wet = case.rate[t] > 0.1
            if wet.mean() < 0.02:
                continue
            for p, r in rates.items():
                ests = bm.motion(r[:t + 1], meta)
                for label, net in (nets or {}).items():
                    ests[label] = lm.motion(r[:t + 1], net)
                for m, v in ests.items():
                    for region, mask in (("wet", wet), ("all", None)):
                        motion_rows.append({"scenario": s, "issue": t, "product": p, "method": m,
                                            "region": region, "flow": case.meta["flow"].split()[0],
                                            **bm.score_motion(v, case.velocity, mask, px_kmh)})
        t_motion = time.time() - t0 - t_sim - t_maps

        # nowcasts
        extra = {label: (lambda h, net=net: lm.motion(h, net)) for label, net in (nets or {}).items()}
        nrates = {p: rates[p] for p in NOWCAST_PRODUCTS if p in rates}
        col = bm.nowcasts(case, nrates, issues, n_leads=12, steps_members=steps_members,
                          extra_motion=extra, collector=col or Collector(lambda: Deterministic(case.pixel_km)))
        t_now = time.time() - t0 - t_sim - t_maps - t_motion
        truth = case.truth.values
        scen_rows.append({"scenario": s, **case.meta, "evolution_params": json.dumps(sc.evolution_params),
                          "wet_fraction": float((truth > 0.1 * case.interval_min / 60).mean()),
                          "mean_mm_h": float(case.rate.mean()),
                          "speed_kmh": float(np.hypot(*case.velocity_kmh).mean()),
                          "t_sim": t_sim, "t_maps": t_maps, "t_motion": t_motion, "t_nowcast": t_now,
                          "errors": json.dumps({k: v for k, v in timings.items() if k.endswith(":error")})})
        log(f"scenario {s}: {case.meta['model']}, {case.meta['flow']}, {case.meta['evolution']}, "
            f"{case.meta['radar_band']}-band, {case.meta['n_links']} links, {case.meta['n_gauges']} gauges, "
            f"{case.meta['n_pws']} PWS - sim {t_sim:.0f}s maps {t_maps:.0f}s motion {t_motion:.0f}s "
            f"nowcast {t_now:.0f}s")
        if checkpoint is not None:
            checkpoint(pd.concat(map_rows, ignore_index=True), pd.DataFrame(motion_rows),
                       pd.DataFrame(col.rows(("product", "method", "lead_min"))), pd.DataFrame(scen_rows))
    maps = pd.concat(map_rows, ignore_index=True)
    nowcast = pd.DataFrame(col.rows(("product", "method", "lead_min"))) if col else pd.DataFrame()
    return maps, pd.DataFrame(motion_rows), nowcast, pd.DataFrame(scen_rows)
