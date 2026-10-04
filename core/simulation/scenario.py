"""
A synthetic case: truth, radar, links and gauges, in the formats the rest of FieldSense reads.

The mapping, merging and nowcasting code in ``core.maps`` and ``core.nowcast``
works on lat/lon xarray data from real networks. ``Scenario.run`` produces the
same structures from a simulation, with the truth alongside, so every method
can be run unchanged and scored against what really fell:

    from core.simulation.scenario import Scenario
    case = Scenario(model="clustered_storms", flow="rotation", n_links=(30, 150),
                    n_gauges=(5, 40), seed=4).run()
    case.truth      # (time, lat, lon) mm per interval on the analysis grid
    case.radar      # same grid
    case.links      # (link, time) mm, with site_*_lat/lon, mid_lat/lon, frequency, polarization
    case.gauges     # (station, time) mm, with lat/lon
    case.velocity   # (2, y, x) true motion, pixels per interval (pysteps convention)

A count given as a ``(low, high)`` tuple is drawn uniformly per scenario, so a
batch of scenarios spans sparse and dense networks. The domain is placed at
``origin_latlon`` in a local equirectangular projection (any location works:
nothing here depends on it except the labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from core.simulation import flows as fl
from core.simulation import generators as gen
from core.simulation import sensors as sn
from core.simulation import spacetime as st
from core.simulation.rain_fields import Grid

KM_PER_DEG = 111.19492664455873          # mean Earth radius, as core.nowcast.grid

Count = Union[int, tuple]


def _draw(v: Count, rng) -> int:
    return int(rng.integers(v[0], v[1] + 1)) if isinstance(v, tuple) else int(v)


@dataclass
class SyntheticCase:
    truth: xr.DataArray                 # analysis grid, mm per interval
    radar: Optional[xr.DataArray]
    links: Optional[xr.DataArray]
    gauges: Optional[xr.DataArray]
    pws: Optional[xr.DataArray]
    velocity: np.ndarray                # (2, y, x) pixels per interval on the analysis grid
    velocity_kmh: np.ndarray            # (2, y, x) km/h
    sequence: st.FieldSequence          # the fine-resolution truth
    geo_grid: object                    # core.geo.Grid of the analysis grid
    interval_min: float
    pixel_km: float
    meta: dict = dc_field(default_factory=dict)
    truth_fine: Optional[np.ndarray] = None   # (time, n, n) mm per interval on the truth grid
    radars: dict = dc_field(default_factory=dict)   # every radar product by name ("radar" too)
    network: object = None                  # the CMLNetwork (km), for distances to the links
    dx_km: float = 0.0                      # truth grid spacing

    @property
    def rate(self) -> np.ndarray:
        """Truth as mm/h, ``(time, y, x)``."""
        return self.truth.values.astype(float) * 60.0 / self.interval_min

    def as_rate(self, da: xr.DataArray) -> np.ndarray:
        return da.transpose("time", "lat", "lon").values.astype(float) * 60.0 / self.interval_min


@dataclass
class Scenario:
    """Everything that defines one synthetic case; ``run()`` simulates it."""

    model: Union[str, object] = "convective_cells"
    model_params: dict = dc_field(default_factory=dict)
    flow: Union[str, fl.Flow] = "uniform"
    flow_params: dict = dc_field(default_factory=dict)
    evolution: str = "ar1"
    evolution_params: dict = dc_field(default_factory=lambda: {"tau_min": 60.0})
    n: int = 128                    # truth grid
    dx_km: float = 0.5
    analysis_factor: int = 2        # analysis pixel = factor * dx (1 km by default)
    duration_min: float = 180.0
    truth_dt_min: float = 1.0
    interval_min: float = 5.0
    radar: Optional[sn.RadarConfig] = dc_field(default_factory=sn.RadarConfig)
    extra_radars: dict = dc_field(default_factory=dict)     # {name: RadarConfig}, e.g. an X-band
    n_links: Count = 80
    link_params: dict = dc_field(default_factory=dict)
    n_gauges: Count = 20
    gauge_params: dict = dc_field(default_factory=dict)
    n_pws: Count = 0
    pws_params: dict = dc_field(default_factory=dict)
    origin_latlon: tuple = (45.0, 10.0)
    start: str = "2026-07-01 12:00"
    seed: int = 0

    def run(self) -> SyntheticCase:
        rng = np.random.default_rng(self.seed)
        grid = Grid(n=self.n, dx_km=self.dx_km)
        model = self.model if not isinstance(self.model, str) else \
            gen.make(self.model, **{"seed": self.seed, **self.model_params})
        flow = self.flow if isinstance(self.flow, fl.Flow) else \
            fl.make_flow(self.flow, **({"seed": self.seed} if self.flow == "random" else {}),
                         **self.flow_params)
        n_steps = int(round(self.duration_min / self.truth_dt_min)) + 1
        if self.evolution == "cloud_model":
            seq = model.simulate(grid, n_steps, self.truth_dt_min, flow=flow)
        else:
            seq = st.simulate(model, grid, n_steps, self.truth_dt_min, flow=flow,
                              evolution=self.evolution, seed=self.seed, **self.evolution_params)
        frames = seq.frames
        per_out = int(round(self.interval_min / self.truth_dt_min))
        n_out = (n_steps - 1) // per_out
        f = self.analysis_factor
        ana_km = self.dx_km * f

        # truth: interval-ending mean rate -> mm, on the analysis grid
        fine = np.stack([frames[j * per_out + 1:(j + 1) * per_out + 1].mean(0) for j in range(n_out)])
        truth = sn.block_mean(fine, f) * self.interval_min / 60.0

        geo, lat, lon = self._geo(ana_km, self.n // f)
        times = pd.date_range(self.start, periods=n_out + 1, freq=f"{self.interval_min:g}min")[1:]
        times = times.values.astype("datetime64[ns]")
        coords = {"time": times, "lat": lat, "lon": lon}

        def field(a, name):
            return xr.DataArray(a.astype("float32"), dims=("time", "lat", "lon"), coords=coords,
                                name=name, attrs={"units": "mm"})

        radars = {}
        meta = {"model": self.model if isinstance(self.model, str) else getattr(model, "key", type(model).__name__), "flow": flow.describe(),
                "evolution": self.evolution, "seed": self.seed}
        configs = ({"radar": self.radar} if self.radar is not None else {}) | dict(self.extra_radars)
        for k, (name, rcfg) in enumerate(configs.items()):
            rcfg = sn.RadarConfig(**{**rcfg.__dict__, "seed": self.seed + 7 + 100 * k})
            r = sn.Radar(rcfg, grid).observe(frames, self.truth_dt_min, self.interval_min)
            rf_ = max(1, int(round(rcfg.resolution_km / self.dx_km)))
            # put the radar product on the analysis grid (each radar pixel repeated, or averaged)
            if rf_ >= f:
                r = np.kron(r, np.ones((1, rf_ // f, rf_ // f)))
            else:
                r = sn.block_mean(r, f // rf_)
            radars[name] = field(r, name)
            meta[f"{name}_band"] = rcfg.band
            meta[f"{name}_km"] = rcfg.resolution_km
        radar = radars.get("radar")
        if radar is not None:
            meta.update(radar_band=self.radar.band, radar_site_km=self.radar.site_km)

        links, net = None, None
        n_links = _draw(self.n_links, rng)
        if n_links > 0:
            cml = sn.CMLs(sn.CMLConfig(n_links=n_links, seed=self.seed + 11, **self.link_params), grid)
            o = cml.observe(frames, self.truth_dt_min, self.interval_min)
            links, net = self._links(cml.net, o, times), cml.net
        gauges = self._points(_draw(self.n_gauges, rng), self.gauge_params, "gauges", 13,
                              frames, grid, times)
        pws = self._points(_draw(self.n_pws, rng), {**sn.PWS_DEFAULTS, **self.pws_params}, "pws", 17,
                           frames, grid, times)
        meta.update(n_links=0 if links is None else links.sizes["link"],
                    n_gauges=0 if gauges is None else gauges.sizes["station"],
                    n_pws=0 if pws is None else pws.sizes["station"])

        vel_kmh = sn.block_mean(seq.velocity, f)
        vel_px = fl.velocity_px(vel_kmh, ana_km, self.interval_min)
        return SyntheticCase(field(truth, "truth"), radar, links, gauges, pws, vel_px, vel_kmh, seq,
                             geo, self.interval_min, ana_km, meta,
                             truth_fine=(fine * self.interval_min / 60.0).astype(np.float32),
                             radars=radars, network=net, dx_km=self.dx_km)

    # ------------------------------------------------------------------ geo
    def _latlon(self, x_km, y_km):
        lat0, lon0 = self.origin_latlon
        latc = lat0 + self.n * self.dx_km / 2 / KM_PER_DEG
        return lat0 + np.asarray(y_km) / KM_PER_DEG, \
            lon0 + np.asarray(x_km) / (KM_PER_DEG * np.cos(np.radians(latc)))

    def _geo(self, ana_km, m):
        from core.geo import Grid as GeoGrid
        c = (np.arange(m) + 0.5) * ana_km
        lat, _ = self._latlon(0 * c, c)
        _, lon = self._latlon(c, 0 * c)
        lat, lon = np.round(lat, 7), np.round(lon, 7)
        return GeoGrid(lat, lon), lat, lon

    def _links(self, net, o, times) -> xr.DataArray:
        la0, lo0 = self._latlon(net.xa, net.ya)
        la1, lo1 = self._latlon(net.xb, net.yb)
        ids = np.array([f"cml{i:03d}" for i in range(net.n_links)], dtype=object)
        return xr.DataArray(
            o["rain_mm"].astype("float32"), dims=("link", "time"), name="links",
            coords={"link": ids, "time": times,
                    "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                    "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                    "mid_lat": ("link", (la0 + la1) / 2), "mid_lon": ("link", (lo0 + lo1) / 2),
                    "frequency": ("link", net.freq_ghz), "polarization": ("link", net.pol.astype(object)),
                    "length_km": ("link", net.length_km),
                    "true_mm": (("link", "time"), o["true_mm"].astype("float32"))},
            attrs={"units": "mm"})

    def _points(self, n, params, kind, offset, frames, grid, times) -> Optional[xr.DataArray]:
        if n <= 0:
            return None
        g = sn.Gauges(sn.GaugeConfig(n=n, seed=self.seed + offset, **{"kind": kind, **params}), grid)
        o = g.observe(frames, self.truth_dt_min, self.interval_min)
        lat, lon = self._latlon(g.x, g.y)
        ids = np.array([f"{kind}{i:03d}" for i in range(n)], dtype=object)
        return xr.DataArray(o["rain_mm"].astype("float32"), dims=("station", "time"), name=kind,
                            coords={"station": ids, "time": times, "lat": ("station", lat),
                                    "lon": ("station", lon), "x_km": ("station", g.x),
                                    "y_km": ("station", g.y),
                                    "true_mm": (("station", "time"), o["true_mm"].astype("float32"))},
                            attrs={"units": "mm"})


def random_scenario(seed: int, **overrides) -> Scenario:
    """A scenario with every choice drawn at random: model, flow, evolution, sensor counts, radar."""
    rng = np.random.default_rng(10_000 + seed)
    model = rng.choice(["convective_cells", "clustered_storms", "squall_line", "stratiform_matern",
                        "banded_anisotropic", "scale_free", "multifractal", "frontal"])
    flow = rng.choice(["uniform", "uniform", "rotation", "shear", "random"])
    speed, ang = rng.uniform(10, 45), rng.uniform(0, 2 * np.pi)
    mean = (float(speed * np.cos(ang)), float(speed * np.sin(ang)))
    flow_params = {"mean": mean}
    if flow == "rotation":
        flow_params["omega_deg_h"] = float(rng.uniform(-60, 60))
    elif flow == "shear":
        flow_params["shear_per_h"] = float(rng.uniform(-1, 1))
    elif flow == "random":
        flow_params["rms_kmh"] = float(rng.uniform(4, 12))
    evolution = "lifecycle" if model in ("convective_cells", "clustered_storms", "squall_line") \
        and rng.random() < 0.5 else str(rng.choice(["ar1", "cascade"]))
    evo = {"lifetime_min": float(rng.uniform(40, 90))} if evolution == "lifecycle" else \
        {"tau_min": float(rng.uniform(40, 150))}
    band = str(rng.choice(["S", "C", "C", "X"]))
    radar = sn.RadarConfig(band=band, site_km=(float(rng.uniform(-60, -5)), float(rng.uniform(-60, 70))),
                           calibration_db=float(rng.normal(0, 1.5)), dsd_sigma_db=float(rng.uniform(1, 2.5)))
    kw = dict(model=str(model), flow=str(flow), flow_params=flow_params, evolution=evolution,
              evolution_params=evo, radar=radar, n_links=(20, 200), n_gauges=(3, 50),
              n_pws=(0, 60), seed=seed)
    kw.update(overrides)
    return Scenario(**kw)
