"""
A 2-D column model of warm-rain clouds: rain that grows and decays for physical reasons.

The statistical generators reproduce what rain *looks like*; in them a cell
grows because an envelope says so. Here rain is produced by a process: water
vapour is lifted by updrafts, condenses into cloud water, cloud water turns
into rain by Kessler (1969) warm-rain microphysics, rain falls out or
evaporates, and everything is carried by the wind. Column-integrated, so every
quantity is a water path in mm (kg m^-2):

    dV/dt = -(1 - f_conv) C + E_c + E_r + (V_env - V) / tau_supply    vapour
    dQc/dt = C - E_c - A - K                                  cloud water
    dQr/dt = A + K - E_r - P                                  rain water
    C   = c_cond * w+ * V * f(V / V_sat)                      condensation in updrafts
    f_conv                                                    share of C fed by low-level
                                                              moisture convergence into the updraft
    A   = k_auto * max(Qc - qc0, 0)                           autoconversion (Kessler)
    K   = k_accr * Qc * Qr^0.875                              accretion (Kessler)
    E_c = k_evap_c * Qc * (w- + subsaturation)                cloud evaporation
    E_r = k_evap_r * Qr^0.5 * (1 - V / V_sat)                 rain evaporation
    P   = Qr / tau_fall                                       surface rain rate (mm/h)

Updrafts ``w`` (m/s) are thermals with a life cycle (born, strengthen, decay,
as ``GaussianCells`` cells) plus a weak large-scale vertical motion (raise
``w_large_scale_ms`` for widespread stratiform rain).
New thermals are born preferentially at the edge of existing rain
(``trigger_gain``): the cold-pool outflow that lifts the air around a dying
cell and organises convection into clusters and lines.

All rates are per hour. The defaults give scattered convection with peaks of
25-55 mm/h, a wet fraction of 0.2-0.35 and cells that live 30-60 minutes; the model is a toy, not a
cloud-resolving model, but its rain has a causal life cycle, a delay between
cloud and rain, and decay by evaporation, which no statistical generator has.

    from core.simulation.cloud_model import WarmRainModel
    seq = WarmRainModel(seed=1).simulate(Grid(n=128, dx_km=0.5), n_steps=48, dt_min=5)
    seq.frames                        # rain rate mm/h
    seq.extras["cloud_water"]         # cloud water path, mm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.simulation import flows as fl
from core.simulation import random_fields as rf
from core.simulation.rain_fields import Grid


@dataclass
class WarmRainModel:
    name: str = "Warm-rain cloud model"
    key: str = "cloud_model"
    # environment
    vapour_env_mm: float = 40.0
    vapour_sat_mm: float = 44.0
    vapour_sd_mm: float = 2.0
    tau_supply_h: float = 1.5
    convergence_frac: float = 0.8
    # thermals (updrafts)
    n_thermals: float = 10.0
    w_peak_ms: float = 6.0
    w_peak_cv: float = 0.4
    thermal_radius_km: float = 2.0
    thermal_life_min: float = 45.0
    trigger_gain: float = 4.0
    w_large_scale_ms: float = 0.0
    w_large_scale_sd_ms: float = 0.05
    # microphysics, per hour
    c_cond: float = 0.5
    k_auto: float = 3.6
    qc0_mm: float = 0.5
    k_accr: float = 6.0
    k_evap_c: float = 1.0
    k_evap_r: float = 1.5
    tau_fall_min: float = 8.0
    # numerics and motion
    dt_int_min: float = 1.0
    spinup_min: float = 60.0
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def simulate(self, grid: Grid, n_steps: int, dt_min: float = 5.0, flow: Optional[fl.Flow] = None,
                 pad_frac: float = 0.25):
        """Run the model; returns a ``spacetime.FieldSequence`` of surface rain rate."""
        from core.simulation.spacetime import FieldSequence
        rng = np.random.default_rng(self.seed)
        flow = flow or fl.UniformFlow(mean=self.advection_kmh)
        pad = 0 if flow.uniform else int(round(pad_frac * grid.n))
        sim = Grid(n=grid.n + 2 * pad, dx_km=grid.dx_km)
        origin = (-pad * grid.dx_km, -pad * grid.dx_km)
        flow.prepare(grid.size_km, sim, origin)
        xx, yy = sim.meshgrid()
        xx, yy = xx + origin[0], yy + origin[1]
        area_ratio = (sim.size_km / grid.size_km) ** 2

        V = self.vapour_env_mm + self.vapour_sd_mm * rf.grf(sim.n, sim.dx_km, rng, "matern", 20.0, 1.5)
        V_env = V.copy()
        Qc = np.zeros(sim.shape)
        Qr = np.zeros(sim.shape)
        w_ls = self.w_large_scale_ms + self.w_large_scale_sd_ms * rf.grf(sim.n, sim.dx_km, rng, "matern",
                                                                         25.0, 1.5)
        th = {k: np.empty(0) for k in ("x", "y", "w", "r", "age", "life")}
        h = self.dt_int_min / 60.0
        life_h = self.thermal_life_min / 60.0
        births = self.n_thermals * area_ratio * h / life_h
        rain_rate = np.zeros(sim.shape)

        def spawn(n, rain):
            if n == 0:
                return
            # birth probability: uniform, raised at the edge of existing rain (outflow triggering)
            wet = (rain > 1.0).astype(float)
            ring = np.clip(gaussian_filter(wet, 3.0 / sim.dx_km, mode="wrap") - wet, 0, None)
            p = 1.0 + self.trigger_gain * ring / max(ring.max(), 1e-9) * (ring.max() > 0)
            p = (p / p.sum()).ravel()
            idx = rng.choice(p.size, size=n, p=p)
            shape_w = 1 / self.w_peak_cv ** 2
            new = {"x": xx.ravel()[idx] + rng.uniform(-.5, .5, n) * sim.dx_km,
                   "y": yy.ravel()[idx] + rng.uniform(-.5, .5, n) * sim.dx_km,
                   "w": rng.gamma(shape_w, self.w_peak_ms / shape_w, n),
                   "r": self.thermal_radius_km * rng.uniform(0.6, 1.4, n),
                   "age": np.zeros(n), "life": rng.gamma(4.0, life_h / 4.0, n)}
            for k in th:
                th[k] = np.concatenate([th[k], new[k]])

        def updraft():
            w = w_ls.copy()
            env = np.sin(np.pi * np.clip(th["age"] / th["life"], 0, 1))
            for i in range(th["x"].size):
                d2 = ((xx - th["x"][i]) ** 2 + (yy - th["y"][i]) ** 2) / th["r"][i] ** 2
                near = d2 < 16
                w[near] += th["w"][i] * env[i] * (np.exp(-0.5 * d2[near]) - 0.4 * np.exp(-0.125 * d2[near]))
            return w      # compensating subsidence ring around each thermal

        spawn(int(rng.poisson(self.n_thermals * area_ratio)), rain_rate)
        th["age"] = rng.uniform(0, 1, th["x"].size) * th["life"]

        n_spin = int(round(self.spinup_min / self.dt_int_min))
        per_out = int(round(dt_min / self.dt_int_min))
        total = n_spin + (n_steps - 1) * per_out + 1
        frames, cloud, vapour, wfield = [], [], [], []
        for it in range(total):
            w = updraft()
            wp, wm = np.clip(w, 0, None), np.clip(-w, 0, None)
            sat = np.clip(V / self.vapour_sat_mm, 0, 1.2)
            C = self.c_cond * wp * V * np.clip((sat - 0.8) / 0.2, 0, None)
            A = self.k_auto * np.clip(Qc - self.qc0_mm, 0, None)
            K = self.k_accr * Qc * np.clip(Qr, 0, None) ** 0.875
            Ec = self.k_evap_c * Qc * (wm + np.clip(1 - sat, 0, None))
            Er = self.k_evap_r * np.sqrt(np.clip(Qr, 0, None)) * np.clip(1 - sat, 0, None)
            P = Qr / (self.tau_fall_min / 60.0)
            V = V + h * (-(1 - self.convergence_frac) * C + Ec + Er + (V_env - V) / self.tau_supply_h)
            Qc = np.clip(Qc + h * (C - Ec - A - K), 0, None)
            Qr = np.clip(Qr + h * (A + K - Er - P), 0, None)
            rain_rate = P

            # carry everything with the wind
            V = fl.advect(V, flow, sim, origin, h, order=1)
            V_env = fl.advect(V_env, flow, sim, origin, h, order=1)
            Qc = np.clip(fl.advect(Qc, flow, sim, origin, h, order=1), 0, None)
            Qr = np.clip(fl.advect(Qr, flow, sim, origin, h, order=1), 0, None)
            w_ls = fl.advect(w_ls, flow, sim, origin, h, order=1)
            th["x"], th["y"] = _forward(flow, th["x"], th["y"], h)
            th["age"] = th["age"] + h
            lo, hi = origin[0], origin[0] + sim.size_km
            if flow.uniform:     # periodic domain: thermals wrap with the fields
                th["x"] = lo + np.mod(th["x"] - lo, hi - lo)
                th["y"] = lo + np.mod(th["y"] - lo, hi - lo)
            keep = (th["age"] < th["life"]) & (th["x"] > lo) & (th["x"] < hi) & (th["y"] > lo) & (th["y"] < hi)
            for k in th:
                th[k] = th[k][keep]
            spawn(int(rng.poisson(births)), rain_rate)

            if it >= n_spin and (it - n_spin) % per_out == 0:
                c = slice(pad, pad + grid.n)
                frames.append(rain_rate[c, c].copy())
                cloud.append(Qc[c, c].copy())
                vapour.append(V[c, c].copy())
                wfield.append(w[c, c].copy())
        frames = np.stack(frames).astype(np.float32)
        frames[frames < 0.1] = 0.0
        return FieldSequence(frames, grid, dt_min, flow.field(grid), flow, self, "cloud_model",
                             {"seed": self.seed},
                             {"cloud_water": np.stack(cloud).astype(np.float32),
                              "vapour": np.stack(vapour).astype(np.float32),
                              "updraft": np.stack(wfield).astype(np.float32)})

    def build(self, grid: Grid) -> np.ndarray:
        """One rain field after spin-up (so the model can stand in as a generator)."""
        return self.simulate(grid, 1).frames[0]


def _forward(flow, x, y, hours):
    u, v = flow.at(x, y)
    u2, v2 = flow.at(x + 0.5 * hours * u, y + 0.5 * hours * v)
    return x + hours * u2, y + hours * v2
