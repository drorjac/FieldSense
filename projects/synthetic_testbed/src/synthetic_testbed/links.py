"""
What do the links' errors cost the maps? Re-run the maps of the first study scenarios with the
same rain, radar and gauges, and links processed three ways:

* ``no wet/dry``  every sample retrieved from the raw attenuation: baseline offsets and noise
                  read as rain
* ``default``     the study's chain: rolling-std wet/dry, baseline from preceding dry
                  samples, a (deliberately mismatched) wet-antenna correction
* ``ideal``       no wet antenna, baseline error, noise or quantisation (only the
                  path-averaging non-linearity is left)

    python projects/synthetic_testbed/src/run.py links --n 8
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.simulation import benchmark as bm
from core.simulation.scenario import random_scenario

VARIANTS = {
    "no wet/dry": {"wet_dry": False},
    "default": {},
    "ideal": {"waa_max_db": 0.0, "waa_max_db_assumed": 0.0, "baseline_sigma_db": 0.0,
              "baseline_drift_db_per_sqrt_h": 0.0, "noise_sigma_db": 0.0, "quantization_db": 0.0,
              "wet_dry": False},
}
METHODS = ("radar", "add [gauges]", "add [links]", "add [links+gauges]", "mul [links+gauges]",
           "okrig_add [links+gauges]", "ked [links+gauges]", "idw links", "idw links+gauges")



def study(n: int = 8, first_seed: int = 100, log=print) -> pd.DataFrame:
    rows = []
    for s in range(first_seed, first_seed + n):
        for v, params in VARIANTS.items():
            case = random_scenario(s, link_params=params).run()
            L = case.links
            true = np.where(np.isfinite(L.values), L.true_mm.values, 0)
            bias = float(np.nansum(L.values) / np.nansum(true) - 1)
            df = bm.score_maps(case, bm.maps(case))
            rows.append(df[df.method.isin(METHODS)].assign(scenario=s, links=v, link_bias=bias))
            log(f"scenario {s} {v}: link bias {bias:+.0%}")
    return pd.concat(rows, ignore_index=True)
