"""The city study: how accurate is each rain map at street level, and how fast does it improve
with coarser space and time?

A 25.6 km city at 100 m and 1 min of truth - the scale of a street, a few minutes of a
storm. Over it:

* an operational C-band radar 20-60 km away, 800 m pixels, no attenuation correction;
* an urban X-band radar just outside the city, 200 m pixels, dual-polarisation attenuation
  correction (85% of the path-integrated attenuation restored);
* 80-300 links (the whole city, at one tower per ~9 km^2), 2-12 official gauges and
  20-150 personal weather stations.

Every map is made on a 200 m analysis grid at 5 min, then scored against the truth at
100 m (each 200 m pixel read for its four 100 m street pixels), 200 m, 400 m, ... 6.4 km,
and at 5, 15, 30 and 60 min (``benchmark.score_scales``), and at 200 m by the distance to the
nearest link. Cell models get a small-scale texture (``texture_strength``) and cell numbers
per area as on the regional domain, so the rain has structure at the street scale.

Each scenario is written to its own files, so the study resumes and can run in parallel.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.simulation import benchmark as bm
from core.simulation import sensors as sn
from core.simulation.scenario import Scenario

from .settings import CELL_MODELS, CITY_DURATION_MIN, CITY_DX_KM, CITY_FACTOR, CITY_N, LINK_BANDS_KM, TABLES

MODELS = CELL_MODELS + ("stratiform_matern", "scale_free", "multifractal", "rainfarm", "cascade")
OUT = TABLES / "city"


def scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(30_000 + seed)
    model = str(rng.choice(MODELS))
    size = CITY_N * CITY_DX_KM
    params = {}
    if model in CELL_MODELS:
        from core.simulation import generators as gen
        n_cells = gen.make(model).n_cells * (size / 64.0) ** 2          # the regional density
        params = {"n_cells": max(3.0, n_cells), "texture_strength": 0.35}
        if gen.make(model).n_storms:
            params["n_storms"] = 1
    flow = str(rng.choice(["uniform", "uniform", "rotation", "shear", "random"]))
    speed, ang = rng.uniform(10, 45), rng.uniform(0, 2 * np.pi)
    fp = {"mean": (float(speed * np.cos(ang)), float(speed * np.sin(ang)))}
    if flow == "rotation":
        fp["omega_deg_h"] = float(rng.uniform(-60, 60))
    elif flow == "shear":
        fp["shear_per_h"] = float(rng.uniform(-1.5, 1.5))
    elif flow == "random":
        fp.update(rms_kmh=float(rng.uniform(4, 12)), length_km=float(rng.uniform(8, 20)))
    if model in CELL_MODELS and rng.random() < 0.6:
        evolution, ep = "lifecycle", {"lifetime_min": float(rng.uniform(30, 90))}
    else:
        evolution = str(rng.choice(["ar1", "cascade"]))
        ep = {"tau_min": float(rng.uniform(40, 150)), "evolve_every": 5}
    a = rng.uniform(0, 2 * np.pi)
    r_c = rng.uniform(20, 60)
    c_band = sn.RadarConfig(band="C", resolution_km=0.8, site_km=(size / 2 + (size / 2 + r_c) * np.cos(a),
                                                                   size / 2 + (size / 2 + r_c) * np.sin(a)),
                            calibration_db=float(rng.normal(0, 1.5)), dsd_sigma_db=float(rng.uniform(1, 2.5)))
    b = rng.uniform(0, 2 * np.pi)
    x_band = sn.RadarConfig(band="X", resolution_km=0.2, beam_width_deg=1.0, pia_correction=0.85,
                            site_km=(size / 2 + (size / 2 + 3) * np.cos(b), size / 2 + (size / 2 + 3) * np.sin(b)),
                            calibration_db=float(rng.normal(0, 1.0)), dsd_sigma_db=float(rng.uniform(1, 2.5)))
    return Scenario(model=model, model_params=params, flow=flow, flow_params=fp, evolution=evolution,
                    evolution_params=ep, n=CITY_N, dx_km=CITY_DX_KM, analysis_factor=CITY_FACTOR,
                    duration_min=CITY_DURATION_MIN, radar=c_band, extra_radars={"xband": x_band},
                    n_links=(80, 300), n_gauges=(2, 12), n_pws=(20, 150), seed=seed)


def products(case) -> dict:
    """The C-band radar and everything made with it; the X-band radar and its cheap adjustments."""
    p = bm.maps(case, sensor_sets=("gauges", "links", "links+gauges", "links+gauges+pws"),
                with_mergeplg=False)
    p.update({k: v for k, v in bm.maps(case, sensor_sets=("links+gauges",)).items()
              if k.startswith(("ked", "okrig", "idw_add", "idw_mul"))})
    p.update(bm.maps_with(case, "xband", "X ", sensor_sets=("gauges", "links+gauges", "links+gauges+pws"),
                          with_mergeplg=False))
    if case.pws is not None:
        from core.maps import merge
        obs, _ = merge.observations(pws=case.pws)
        p["idw pws"] = merge.merge_idw(obs, case.geo_grid, bm.IDW_FULL)
    return {k: v for k, v in p.items() if v is not None}


def run_one(seed: int, log=print) -> pd.DataFrame:
    """Simulate one city scenario, make every map, score it; written to ``tables/city/``."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scenario_{seed:04d}.csv"
    if path.exists():
        return pd.read_csv(path)
    t0 = time.time()
    case = scenario(seed).run()
    t_sim = time.time() - t0
    prods = products(case)
    t_maps = time.time() - t0 - t_sim
    scales = bm.score_scales(case, prods).assign(region="all")
    dist = bm.link_distance_km(case)
    bands = []
    for lo, hi in zip(LINK_BANDS_KM[:-1], LINK_BANDS_KM[1:]):
        d = bm.score_scales(case, prods, space_km=(0.2,), time_min=(5, 60),
                            region=(dist >= lo) & (dist < hi))
        bands.append(d.assign(region=f"{lo:g}-{hi:g} km" if hi < 1e8 else f">{lo:g} km",
                              band_lo=lo, cell_share=float(((dist >= lo) & (dist < hi)).mean())))
    out = pd.concat([scales] + bands, ignore_index=True)
    m = case.meta
    out = out.assign(scenario=seed, model=m["model"], flow=m["flow"].split()[0], evolution=m["evolution"],
                     n_links=m["n_links"], n_gauges=m["n_gauges"], n_pws=m["n_pws"],
                     wet_fraction=float((case.truth_fine > 0.1 * case.interval_min / 60).mean()),
                     mean_mm_h=float(case.truth_fine.mean() * 60 / case.interval_min))
    out.to_csv(path, index=False)
    log(f"city {seed}: {m['model']}, {m['flow']}, {m['evolution']}, {m['n_links']} links, "
        f"{m['n_gauges']} gauges, {m['n_pws']} PWS - sim {t_sim:.0f}s maps {t_maps:.0f}s "
        f"total {time.time() - t0:.0f}s")
    return out


def study(seeds, log=print) -> pd.DataFrame:
    return pd.concat([run_one(s, log) for s in seeds], ignore_index=True)


def load() -> pd.DataFrame:
    files = sorted(OUT.glob("scenario_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def example(seed: int = 0):
    """One city case and its products, for the figures and the notebook."""
    case = scenario(seed).run()
    return case, products(case)
