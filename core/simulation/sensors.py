"""
Simulated sensors: a weather radar, a CML network and rain gauges over a known truth.

Each sensor turns a simulated sequence of rain rates (``spacetime.FieldSequence``
at a fine grid and time step) into what that instrument would report, with the
errors that dominate it in practice, and nothing else is known to the methods
downstream. All three report accumulations per product interval (mm).

``Radar``  (``RadarConfig``)
    Instantaneous scans from a site that may be outside the domain. Truth ->
    reflectivity through a Z-R relation with spatially correlated DSD
    variability -> two-way path-integrated attenuation along each ray (ITU-R
    P.838-3 at the band's frequency; strong at X, moderate at C, negligible at
    S; ``pia_correction`` restores a share of it, as a dual-polarisation correction does) ->
    beam broadening (smoothing that grows with range) -> beam height and
    overshooting (Z falls off above the echo top) -> partial beam blockage ->
    averaging to the radar resolution -> calibration bias, noise, minimum
    detectable reflectivity that rises with range, 0.5 dB quantisation ->
    retrieval with an *assumed* Z-R relation. Accumulation = rate x interval,
    as operational products do between scans.

``CMLs``  (``CMLConfig``)
    A backhaul topology from ``cml_network.synthesize_network``; at every truth
    step the path integral of ``k R^alpha`` (ITU-R), wet-antenna attenuation,
    a per-link baseline offset that drifts as a random walk, receiver noise,
    quantisation and outages; then processed as real data are: rolling-std
    wet/dry classification, a baseline from the preceding dry samples, a
    wet-antenna correction with deliberately wrong coefficients, the power law,
    and the mean over the interval.

``Gauges``  (``GaugeConfig``; ``PWS_DEFAULTS`` for personal weather stations)
    Points placed uniformly, clustered (urban) or on a grid; tipping-bucket
    resolution, per-gauge undercatch, and faults: dead gauges (NaN) and
    gauges stuck at zero (the classic PWS failure).

Positions are km with the domain's lower-left corner at (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from core.simulation import cml_network as cn
from core.simulation import random_fields as rf
from core.simulation.rain_fields import Grid

BAND_GHZ = {"S": 2.8, "C": 5.6, "X": 9.4}
EFFECTIVE_EARTH_RADIUS_KM = 8494.0          # 4/3 of the Earth's radius


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over ``f x f`` blocks of the last two axes."""
    if f == 1:
        return a
    *lead, ny, nx = a.shape
    return a.reshape(*lead, ny // f, f, nx // f, f).mean(axis=(-3, -1))


def _bilinear_points(field: np.ndarray, grid: Grid, x, y) -> np.ndarray:
    return map_coordinates(field, [np.asarray(y) / grid.dx_km - 0.5, np.asarray(x) / grid.dx_km - 0.5],
                           order=1, mode="nearest")


# --------------------------------------------------------------------------
# radar
# --------------------------------------------------------------------------
@dataclass
class RadarConfig:
    site_km: Tuple[float, float] = (-20.0, -20.0)
    band: str = "C"
    resolution_km: float = 1.0
    beam_width_deg: float = 1.0
    elevation_deg: float = 0.5
    zr_true: Tuple[float, float] = (200.0, 1.6)      # what nature does on average
    zr_assumed: Tuple[float, float] = (200.0, 1.6)   # what the retrieval uses
    dsd_sigma_db: float = 1.5                        # DSD variability around zr_true
    dsd_length_km: float = 8.0
    dsd_tau_min: float = 30.0
    calibration_db: float = 0.0
    noise_db: float = 1.0
    quantization_db: float = 0.5
    mds_dbz_1km: float = -30.0                       # min detectable reflectivity at 1 km
    attenuation: bool = True
    pia_correction: float = 0.0                      # share of the PIA restored (dual-pol: ~0.8-0.9)
    echo_top_km: float = 5.0
    vpr_db_per_km: float = -6.0                      # Z lapse above the echo top
    blockage: Tuple[Tuple[float, float, float], ...] = ()   # (azimuth from, to, loss dB)
    scan_min: Optional[float] = None                 # default: the product interval
    seed: int = 0

    @property
    def freq_ghz(self) -> float:
        return BAND_GHZ[self.band]


class Radar:
    """Radar simulator over a truth sequence on ``grid``."""

    def __init__(self, cfg: RadarConfig, grid: Grid):
        from core.itu_p838 import get_k_alpha
        self.cfg, self.grid = cfg, grid
        self.k_att, self.a_att = get_k_alpha(cfg.freq_ghz, "horizontal")
        xx, yy = grid.meshgrid()
        dx, dy = xx - cfg.site_km[0], yy - cfg.site_km[1]
        self.range_km = np.maximum(np.hypot(dx, dy), 0.1)
        self.azimuth = np.mod(np.degrees(np.arctan2(dx, dy)), 360)       # clockwise from north
        self.beam_height_km = self.range_km * np.sin(np.radians(cfg.elevation_deg)) \
            + self.range_km ** 2 / (2 * EFFECTIVE_EARTH_RADIUS_KM)
        self.beam_sigma_km = self.range_km * np.radians(cfg.beam_width_deg) / 2.355
        self.rng = np.random.default_rng(cfg.seed)
        self._dsd = None
        self.factor = max(1, int(round(cfg.resolution_km / grid.dx_km)))

    # per-pixel static effects, in dB
    def static_loss_db(self) -> np.ndarray:
        c = self.cfg
        over = np.clip(self.beam_height_km - c.echo_top_km, 0, None) * c.vpr_db_per_km
        block = np.zeros(self.grid.shape)
        for a0, a1, loss in c.blockage:
            inside = (self.azimuth >= a0) & (self.azimuth <= a1) if a0 <= a1 else \
                (self.azimuth >= a0) | (self.azimuth <= a1)
            block[inside] -= loss
        return over + block

    def min_dbz(self) -> np.ndarray:
        return self.cfg.mds_dbz_1km + 20 * np.log10(self.range_km)

    def _dsd_field(self, dt_min: float) -> np.ndarray:
        c, g = self.cfg, self.grid
        new = rf.grf(g.n, g.dx_km, self.rng, "matern", c.dsd_length_km, nu=1.5)
        if self._dsd is None:
            self._dsd = new
        else:
            rho = np.exp(-dt_min / c.dsd_tau_min)
            self._dsd = rho * self._dsd + np.sqrt(1 - rho ** 2) * new
        return c.dsd_sigma_db * self._dsd

    def pia_db(self, rain: np.ndarray, n_samples: int = 64) -> np.ndarray:
        """Two-way path-integrated attenuation to every pixel (dB); rain outside the domain is 0."""
        c, g = self.cfg, self.grid
        k = self.k_att * np.clip(rain, 0, None) ** self.a_att                 # dB/km one way
        xx, yy = g.meshgrid()
        s = np.linspace(0, 1, n_samples)[:, None, None]
        px = c.site_km[0] + s * (xx - c.site_km[0])[None]
        py = c.site_km[1] + s * (yy - c.site_km[1])[None]
        col, row = px / g.dx_km - 0.5, py / g.dx_km - 0.5
        inside = (col >= -0.5) & (col <= g.n - 0.5) & (row >= -0.5) & (row <= g.n - 0.5)
        kv = map_coordinates(k, [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(px.shape)
        kv = np.where(inside, kv, 0.0)
        return 2.0 * kv.mean(axis=0) * self.range_km

    def scan(self, rain: np.ndarray, dt_min: float) -> np.ndarray:
        """Retrieved rain rate (mm/h) on the radar grid (``n / factor`` square) from one scan."""
        c = self.cfg
        a, b = c.zr_true
        with np.errstate(divide="ignore"):
            z_true_db = np.where(rain > 0, 10 * np.log10(a * np.clip(rain, 1e-6, None) ** b), -99.0)
        z_db = z_true_db + np.where(rain > 0, self._dsd_field(dt_min), 0.0) + self.static_loss_db()
        if c.attenuation:
            z_db = z_db - (1.0 - c.pia_correction) * self.pia_db(rain)
        # beam broadening: blend Z smoothed at a few widths by the local beam width
        Z = np.where(z_db > -90, 10 ** (z_db / 10), 0.0)
        sig_px = self.beam_sigma_km / self.grid.dx_km
        levels = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        smooth = np.stack([Z if s == 0 else gaussian_filter(Z, s, mode="nearest") for s in levels])
        idx = np.clip(np.interp(sig_px, levels, np.arange(levels.size)), 0, levels.size - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, levels.size - 1)
        w = idx - i0
        rows, cols = np.indices(Z.shape)
        Z = (1 - w) * smooth[i0, rows, cols] + w * smooth[i1, rows, cols]
        # sample volume average at the radar resolution, then the receiver
        Zr = block_mean(Z, self.factor)
        mdz = block_mean(self.min_dbz(), self.factor)
        with np.errstate(divide="ignore"):
            dbz = np.where(Zr > 0, 10 * np.log10(np.clip(Zr, 1e-12, None)), -99.0)
        dbz = dbz + c.calibration_db + self.rng.normal(0, c.noise_db, dbz.shape)
        if c.quantization_db > 0:
            dbz = np.round(dbz / c.quantization_db) * c.quantization_db
        a2, b2 = c.zr_assumed
        r = np.where(dbz >= mdz, (10 ** (dbz / 10) / a2) ** (1 / b2), 0.0)
        r[r < 0.1] = 0.0
        return r

    def observe(self, frames: np.ndarray, dt_min: float, product_min: float) -> np.ndarray:
        """Accumulation (mm) per product interval, ``(T_out, n/f, n/f)``.

        Scans every ``scan_min`` (default: one per interval, at its end); each scan's rate
        is held over its share of the interval.
        """
        scan_min = self.cfg.scan_min or product_min
        per_scan = int(round(scan_min / dt_min))
        per_out = int(round(product_min / dt_min))
        n_out = (frames.shape[0] - 1) // per_out
        out = []
        for j in range(n_out):
            idx = range(j * per_out + per_scan, (j + 1) * per_out + 1, per_scan)
            rates = [self.scan(frames[i], scan_min) for i in idx]
            out.append(np.mean(rates, axis=0) * product_min / 60.0)
        return np.stack(out)


# --------------------------------------------------------------------------
# CMLs
# --------------------------------------------------------------------------
@dataclass
class CMLConfig(cn.SensorConfig):
    n_links: int = 80
    nodes_per_link: float = 0.55
    node_spacing_km: float = 3.0
    min_length_km: float = 0.5
    max_length_km: float = 12.0
    baseline_drift_db_per_sqrt_h: float = 0.1
    outage_frac: float = 0.01
    dead_link_frac: float = 0.0
    # processing, as for real data: rolling-std wet/dry (Schleiss & Berne 2010), baseline from
    # the preceding dry samples, then wet antenna and power law. ``wet_dry=False`` retrieves
    # every sample from the raw attenuation (any offset or noise then reads as rain).
    wet_dry: bool = True
    wet_dry_window_min: float = 30.0
    wet_dry_threshold_db: float = 0.3
    baseline_window_min: float = 60.0


class CMLs:
    """Links over ``grid``. Towers sit at a realistic density (one per ``node_spacing_km``
    squared): a small network covers part of the domain densely, as a city network does,
    rather than spreading a few very long links over all of it."""

    def __init__(self, cfg: CMLConfig, grid: Grid):
        self.cfg, self.grid = cfg, grid
        n_nodes = max(4, int(round(cfg.n_links * cfg.nodes_per_link)))
        side = min(grid.size_km, np.sqrt(n_nodes) * cfg.node_spacing_km)
        rng = np.random.default_rng(cfg.seed + 3)
        x0, y0 = rng.uniform(0, grid.size_km - side, 2)
        net = cn.synthesize_network(Grid(n=grid.n, dx_km=side / grid.n), n_links=cfg.n_links,
                                    n_nodes=n_nodes, min_length_km=cfg.min_length_km,
                                    max_length_km=cfg.max_length_km, seed=cfg.seed)
        net.xa, net.xb = net.xa + x0, net.xb + x0
        net.ya, net.yb = net.ya + y0, net.yb + y0
        self.net = net

    def wet_dry_baseline(self, A: np.ndarray, dt_min: float):
        """Wet where the centred rolling std of the attenuation exceeds the threshold; the
        baseline is the mean of the last ``baseline_window_min`` of dry samples, held
        through wet periods (``(links, time)`` each)."""
        import pandas as pd
        c = self.cfg
        w = max(3, int(round(c.wet_dry_window_min / dt_min)))
        sd = pd.DataFrame(A.T).rolling(w, center=True, min_periods=w // 2).std().to_numpy().T
        wet = np.nan_to_num(sd) > c.wet_dry_threshold_db
        nb = max(1, int(round(c.baseline_window_min / dt_min)))
        base = pd.DataFrame(np.where(wet, np.nan, A).T).rolling(nb, min_periods=1).mean().ffill()
        base = base.to_numpy().T
        # before the first dry sample there is no reference: the first value
        base = np.where(np.isfinite(base), base, A[:, :1])
        return wet, base

    def observe(self, frames: np.ndarray, dt_min: float, product_min: float) -> dict:
        """Per link and product interval: ``rain_mm`` retrieved, ``true_mm`` path average."""
        c, net = self.cfg, self.net
        rng = np.random.default_rng(c.seed + 1)
        L, nl = net.length_km, net.n_links
        x, y = net.path_points(cn.N_PATH_SAMPLES)
        col, row = x / self.grid.dx_km - 0.5, y / self.grid.dx_km - 0.5
        baseline = rng.normal(0, c.baseline_sigma_db, nl)
        dead = rng.random(nl) < c.dead_link_frac
        h = dt_min / 60.0
        per_out = int(round(product_min / dt_min))
        n_out = (frames.shape[0] - 1) // per_out
        T = frames.shape[0]
        A_obs = np.zeros((nl, T))
        true_path = np.zeros((nl, T))
        for t in range(T):
            s = map_coordinates(frames[t], [row.ravel(), col.ravel()], order=1, mode="nearest")
            s = np.clip(s.reshape(x.shape), 0, None)
            true_path[:, t] = s.mean(axis=1)
            A = net.k * (s ** net.alpha[:, None]).mean(axis=1) * L
            A = A + cn.wet_antenna_db(true_path[:, t], c.waa_max_db, c.waa_rate_per_mm_h)
            baseline = baseline + rng.normal(0, c.baseline_drift_db_per_sqrt_h * np.sqrt(h), nl)
            A = A + baseline + rng.normal(0, c.noise_sigma_db, nl)
            if c.quantization_db > 0:
                A = np.round(A / c.quantization_db) * c.quantization_db
            A_obs[:, t] = A
        self.wet, self.A_obs = None, A_obs
        if c.wet_dry:
            wet, base = self.wet_dry_baseline(A_obs, dt_min)
            retrieved = np.where(wet, cn.retrieve_rain(np.clip(A_obs - base, 0, None).T, net, c).T, 0.0)
            self.wet = wet
        else:
            retrieved = cn.retrieve_rain(np.clip(A_obs, 0, None).T, net, c).T
        retrieved[rng.random(retrieved.shape) < c.outage_frac] = np.nan
        retrieved[dead] = np.nan
        out_r, out_t = np.zeros((nl, n_out)), np.zeros((nl, n_out))
        for j in range(n_out):
            sl = slice(j * per_out + 1, (j + 1) * per_out + 1)       # interval-ending
            with np.errstate(invalid="ignore"):
                valid = np.isfinite(retrieved[:, sl]).mean(1)
                out_r[:, j] = np.where(valid >= 0.5, np.nanmean(retrieved[:, sl], axis=1), np.nan)
            out_t[:, j] = true_path[:, sl].mean(1)
        f = product_min / 60.0
        return {"rain_mm": out_r * f, "true_mm": out_t * f}


# --------------------------------------------------------------------------
# gauges
# --------------------------------------------------------------------------
@dataclass
class GaugeConfig:
    n: int = 20
    placement: str = "uniform"          # uniform | clustered | grid
    n_clusters: int = 3
    cluster_km: float = 4.0
    margin_km: float = 1.0
    tip_mm: float = 0.2
    undercatch_mean: float = 0.95
    undercatch_sd: float = 0.03
    dead_frac: float = 0.0
    stuck_zero_frac: float = 0.0
    kind: str = "gauges"
    seed: int = 0


PWS_DEFAULTS = dict(placement="clustered", tip_mm=0.101, undercatch_mean=0.85, undercatch_sd=0.12,
                    dead_frac=0.05, stuck_zero_frac=0.08, kind="pws")


class Gauges:
    def __init__(self, cfg: GaugeConfig, grid: Grid):
        self.cfg, self.grid = cfg, grid
        rng = np.random.default_rng(cfg.seed)
        lo, hi = cfg.margin_km, grid.size_km - cfg.margin_km
        if cfg.placement == "grid":
            m = int(np.ceil(np.sqrt(cfg.n)))
            g = np.linspace(lo, hi, m + 2)[1:-1]
            xx, yy = np.meshgrid(g, g)
            self.x, self.y = xx.ravel()[:cfg.n], yy.ravel()[:cfg.n]
        elif cfg.placement == "clustered":
            cx, cy = rng.uniform(lo, hi, cfg.n_clusters), rng.uniform(lo, hi, cfg.n_clusters)
            k = rng.integers(0, cfg.n_clusters, cfg.n)
            self.x = np.clip(cx[k] + rng.normal(0, cfg.cluster_km, cfg.n), lo, hi)
            self.y = np.clip(cy[k] + rng.normal(0, cfg.cluster_km, cfg.n), lo, hi)
        else:
            self.x, self.y = rng.uniform(lo, hi, cfg.n), rng.uniform(lo, hi, cfg.n)
        self.catch = np.clip(rng.normal(cfg.undercatch_mean, cfg.undercatch_sd, cfg.n), 0.3, 1.1)
        self.dead = rng.random(cfg.n) < cfg.dead_frac
        self.stuck = (rng.random(cfg.n) < cfg.stuck_zero_frac) & ~self.dead

    def observe(self, frames: np.ndarray, dt_min: float, product_min: float) -> dict:
        c = self.cfg
        h = dt_min / 60.0
        rates = np.stack([_bilinear_points(f, self.grid, self.x, self.y) for f in frames], axis=1)
        per_out = int(round(product_min / dt_min))
        n_out = (frames.shape[0] - 1) // per_out
        # depth in each truth step (interval-ending), cumulated and tipped
        depth = np.concatenate([np.zeros((c.n, 1)), rates[:, 1:] * h], axis=1)
        true_cum = np.cumsum(depth, axis=1)
        caught = np.cumsum(depth * self.catch[:, None], axis=1)
        tipped = np.floor(caught / c.tip_mm) * c.tip_mm if c.tip_mm > 0 else caught
        ends = np.arange(n_out + 1) * per_out
        obs = np.diff(tipped[:, ends], axis=1)
        true = np.diff(true_cum[:, ends], axis=1)
        obs[self.stuck] = 0.0
        obs[self.dead] = np.nan
        return {"rain_mm": obs, "true_mm": true}
