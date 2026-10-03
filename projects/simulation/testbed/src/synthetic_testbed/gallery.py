"""What the generators make: one field each, the four evolutions, how predictable each leaves the rain."""

from __future__ import annotations

import pandas as pd

from core.simulation import flows as fl
from core.simulation import generators as gen
from core.simulation import spacetime as st
from core.simulation.rain_fields import Grid, field_stats

from .settings import GALLERY


def fields(n: int = 128, dx_km: float = 0.5, seed: int = 3) -> tuple[dict, pd.DataFrame]:
    """One realisation of every generator (and the cloud model after spin-up), with statistics."""
    from core.simulation.cloud_model import WarmRainModel
    grid = Grid(n=n, dx_km=dx_km)
    out, rows = {}, []
    for key in GALLERY:
        out[key] = gen.make(key, seed=seed).build(grid)
    out["cloud_model"] = WarmRainModel(seed=seed).simulate(grid, 1).frames[0]
    for key, f in out.items():
        s = field_stats(f, grid)
        rows.append({"model": key, "wet_fraction": s["war"], "mean_mm_h": s["imf"],
                     "mean_wet_mm_h": s["cmf"], "max_mm_h": s["max"],
                     "decorrelation_km": s["decorrelation_km"], "water_in_top5pct": s["frac_flux_top5pct"]})
    return out, pd.DataFrame(rows)


def evolutions(n: int = 128, dx_km: float = 0.5, seed: int = 5, n_steps: int = 13, dt_min: float = 5.0):
    """Clustered storms in a rotating flow under each evolution (and the cloud model), with the
    correlation the true motion alone keeps at every lead."""
    from core.simulation.cloud_model import WarmRainModel
    grid = Grid(n=n, dx_km=dx_km)
    flow = fl.RotationFlow(omega_deg_h=30, mean=(20.0, 5.0))
    model = gen.make("clustered_storms", seed=seed)
    seqs = {ev: st.simulate(model, grid, n_steps, dt_min, flow=flow, evolution=ev, tau_min=60,
                            lifetime_min=60, seed=seed)
            for ev in ("frozen", "ar1", "cascade", "lifecycle")}
    seqs["cloud_model"] = WarmRainModel(seed=seed).simulate(
        grid, n_steps, dt_min, flow=fl.RotationFlow(omega_deg_h=30, mean=(20.0, 5.0)))
    pred = pd.DataFrame([{"evolution": k, "lead_min": h * dt_min, "correlation": s.predictability(h)}
                         for k, s in seqs.items() for h in range(1, n_steps - 1)])
    return seqs, pred


def predictability(seeds=range(4), n: int = 96, dx_km: float = 0.5) -> pd.DataFrame:
    """Correlation kept by the true motion at 30 and 60 min, per generator and evolution."""
    grid = Grid(n=n, dx_km=dx_km)
    rows = []
    for key in GALLERY:
        for ev in ("ar1", "cascade", "lifecycle"):
            if ev == "lifecycle" and not hasattr(gen.make(key), "sample_cells"):
                continue
            for s in seeds:
                seq = st.simulate(gen.make(key, seed=s), grid, 13, 5.0, evolution=ev, tau_min=60,
                                  lifetime_min=60, seed=s)
                rows.append({"model": key, "evolution": ev, "seed": s,
                             "r30": seq.predictability(6), "r60": seq.predictability(12)})
    return pd.DataFrame(rows).groupby(["model", "evolution"])[["r30", "r60"]].mean().reset_index()
