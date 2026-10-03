"""
Commercial microwave link (CML) network: geometry, forward model, retrieval.

A CML measures the attenuation of a point-to-point microwave beam. Rain along
the path attenuates the signal, so each link reports one number that is a
*line integral* over the rain field, not a point sample. Turning a set of those
numbers back into a 2-D field is the problem this project demonstrates.

The forward chain modelled here, in order:

1. Line integral of specific attenuation along the path (ITU-R P.838-3).
2. Wet-antenna attenuation - water on the radome, not in the air. Static in
   the path rain by default; with memory (builds up, dries) when a
   ``wet_antenna.DynamicWetAntenna`` is passed.
3. Baseline error - uncertainty in the dry-weather reference level.
4. Receiver noise.
5. Quantization - the RSL is reported in discrete steps.

The retrieval inverts 1-2 and is blind to 3-5. Two error sources survive even a
perfect receiver, and both are demonstrated separately:

*Path-averaging bias*: attenuation integrates ``R**alpha``, but the retrieval
solves for a single ``R``. By Jensen's inequality the retrieved path average
exceeds the true one when ``alpha > 1`` and falls below it when ``alpha < 1``.
Since alpha crosses 1 near 23 GHz, the bias changes sign across a real network.

*Wet-antenna mismatch*: the correction applied in retrieval never has the true
coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from core.simulation.rain_fields import Grid

# Typical backhaul allocations. Longer hops use lower frequencies because the
# rain-fade margin has to cover more kilometres.
_FREQUENCY_PLAN = (
    # (max length km, frequency GHz)
    (1.5, 38.0),
    (3.5, 26.0),
    (6.0, 23.0),
    (9.0, 18.0),
    (np.inf, 15.0),
)

N_PATH_SAMPLES = 128


@dataclass
class CMLNetwork:
    """A set of links over the domain. Coordinates in km."""

    xa: np.ndarray
    ya: np.ndarray
    xb: np.ndarray
    yb: np.ndarray
    freq_ghz: np.ndarray
    pol: np.ndarray          # array of "horizontal" / "vertical"
    k: np.ndarray            # ITU-R coefficient per link
    alpha: np.ndarray        # ITU-R coefficient per link

    @property
    def n_links(self) -> int:
        return self.xa.size

    @property
    def length_km(self) -> np.ndarray:
        return np.hypot(self.xb - self.xa, self.yb - self.ya)

    @property
    def midpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        return 0.5 * (self.xa + self.xb), 0.5 * (self.ya + self.yb)

    def path_points(self, n_samples: int = N_PATH_SAMPLES):
        """Sample points along every path. Returns (x, y), each (n_links, n_samples)."""
        s = np.linspace(0.0, 1.0, n_samples)[None, :]
        x = self.xa[:, None] + s * (self.xb - self.xa)[:, None]
        y = self.ya[:, None] + s * (self.yb - self.ya)[:, None]
        return x, y

    def summary(self) -> dict:
        L = self.length_km
        return {
            "n_links": self.n_links,
            "total_path_km": float(L.sum()),
            "mean_length_km": float(L.mean()),
            "min_length_km": float(L.min()),
            "max_length_km": float(L.max()),
            "median_freq_ghz": float(np.median(self.freq_ghz)),
            "frac_alpha_gt_1": float((self.alpha > 1.0).mean()),
        }


def _assign_frequency(length_km: float, rng: np.random.Generator) -> float:
    for max_len, freq in _FREQUENCY_PLAN:
        if length_km <= max_len:
            # a little spread so the network is not perfectly banded
            return float(np.clip(freq * rng.normal(1.0, 0.06), 10.0, 42.0))
    return 15.0


def synthesize_network(grid: Grid, n_links: int = 90, n_nodes: int = 46,
                       min_length_km: float = 0.6, max_length_km: float = 11.0,
                       neighbours: int = 3,
                       seed: int = 5) -> CMLNetwork:
    """Build a plausible backhaul topology over the domain.

    Towers are scattered uniformly, then each tower is connected to its nearest
    few neighbours within a length window. That produces the length
    distribution and the clustering of a real network far better than drawing
    independent random segments, which would leave the domain covered
    uniformly and make the reconstruction look better than it is.
    """
    from core.itu_p838 import get_k_alpha  # local import: keeps module import cheap

    rng = np.random.default_rng(seed)
    nx = rng.uniform(0.0, grid.size_km, n_nodes)
    ny = rng.uniform(0.0, grid.size_km, n_nodes)

    candidates = set()
    for i in range(n_nodes):
        d = np.hypot(nx - nx[i], ny - ny[i])
        d[i] = np.inf
        order = np.argsort(d)
        taken = 0
        for j in order:
            if taken >= neighbours:
                break
            if min_length_km <= d[j] <= max_length_km:
                candidates.add((min(i, int(j)), max(i, int(j))))
                taken += 1

    pairs = sorted(candidates)
    rng.shuffle(pairs)
    pairs = pairs[:n_links]
    if not pairs:
        raise RuntimeError("no links satisfied the length window")

    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    xa, ya, xb, yb = nx[ia], ny[ia], nx[ib], ny[ib]
    lengths = np.hypot(xb - xa, yb - ya)

    freqs = np.array([_assign_frequency(L, rng) for L in lengths])
    pols = np.array(["vertical" if u < 0.5 else "horizontal"
                     for u in rng.uniform(size=freqs.size)])

    ka = np.empty_like(freqs)
    al = np.empty_like(freqs)
    for i, (f, p) in enumerate(zip(freqs, pols)):
        ka[i], al[i] = get_k_alpha(float(f), str(p))

    return CMLNetwork(xa=xa, ya=ya, xb=xb, yb=yb, freq_ghz=freqs, pol=pols,
                      k=ka, alpha=al)


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------
def sample_along_paths(rain: np.ndarray, grid: Grid, net: CMLNetwork,
                       n_samples: int = N_PATH_SAMPLES) -> np.ndarray:
    """Bilinear-sample the rain field at points along every path.

    Returns an (n_links, n_samples) array of rain rates in mm/h.
    """
    x, y = net.path_points(n_samples)
    # grid cell centres sit at (i + 0.5) * dx, so the fractional index is
    # x/dx - 0.5; the grid is periodic, hence mode="grid-wrap".
    col = x / grid.dx_km - 0.5
    row = y / grid.dx_km - 0.5
    samples = map_coordinates(rain, [row.ravel(), col.ravel()], order=1,
                              mode="grid-wrap")
    return np.clip(samples.reshape(x.shape), 0.0, None)


@dataclass
class SensorConfig:
    """Impairments applied on top of the clean path integral."""

    waa_max_db: float = 2.3          # Schleiss-type saturating wet-antenna term
    waa_rate_per_mm_h: float = 0.28
    waa_max_db_assumed: float = 1.8  # what the retrieval believes - deliberately off
    waa_rate_assumed: float = 0.35
    baseline_sigma_db: float = 0.25  # dry-reference uncertainty, per link
    noise_sigma_db: float = 0.12     # receiver noise
    quantization_db: float = 0.3     # reported RSL step
    seed: int = 101


def wet_antenna_db(rain_path_mm_h: np.ndarray, a_max: float,
                   rate: float) -> np.ndarray:
    """Saturating wet-antenna attenuation, after Schleiss et al. (2013)."""
    return a_max * (1.0 - np.exp(-rate * np.clip(rain_path_mm_h, 0.0, None)))


def forward_model(rain: np.ndarray, grid: Grid, net: CMLNetwork,
                  cfg: Optional[SensorConfig] = None,
                  n_samples: int = N_PATH_SAMPLES,
                  wet_antenna=None, waa_state: Optional[np.ndarray] = None,
                  dt_min: float = 1.0) -> dict:
    """Run the full sensor chain for one rain field.

    ``wet_antenna`` (opt-in) is a ``wet_antenna.DynamicWetAntenna``: the wet
    antenna then has memory. ``waa_state`` is its attenuation per link before
    this field (zero if not given), and the field is applied for ``dt_min``
    minutes; the new state is returned as ``A_waa``, to be passed back as
    ``waa_state`` with the next field (``forward_series`` does this). Left at
    ``None``, the static ``SensorConfig`` wet antenna is used, as before. The
    retrieval always corrects with the static, assumed model.

    Returns a dict of per-link quantities, all arrays of length ``n_links``:

    ``R_path_true``
        True path-averaged rain rate (mm/h). This is what a perfect CML would
        report and what the reconstruction ideally receives.
    ``A_rain``
        True rain-induced attenuation, ``k * mean(R**alpha) * L`` (dB).
    ``A_uniform``
        Attenuation the path *would* show if the rain were uniform at
        ``R_path_true``. The gap against ``A_rain`` is the path-averaging
        non-linearity.
    ``A_observed``
        What the receiver reports, after wet antenna, baseline error, noise and
        quantization.
    ``R_retrieved``
        Rain rate recovered from ``A_observed``.
    ``R_retrieved_clean``
        Rain rate recovered from ``A_rain`` alone - isolates the path-averaging
        bias from every other impairment.
    """
    cfg = cfg or SensorConfig()
    rng = np.random.default_rng(cfg.seed)

    samples = sample_along_paths(rain, grid, net, n_samples)
    L = net.length_km
    k = net.k
    alpha = net.alpha

    R_path_true = samples.mean(axis=1)

    # The physical line integral: attenuation integrates R**alpha, not R.
    A_rain = k * (samples ** alpha[:, None]).mean(axis=1) * L
    A_uniform = k * R_path_true**alpha * L

    if wet_antenna is None:
        A_waa = wet_antenna_db(R_path_true, cfg.waa_max_db, cfg.waa_rate_per_mm_h)
    else:
        prev = np.zeros(net.n_links) if waa_state is None else waa_state
        A_waa = wet_antenna.step(prev, R_path_true, dt_min)
    baseline = rng.normal(0.0, cfg.baseline_sigma_db, net.n_links)
    noise = rng.normal(0.0, cfg.noise_sigma_db, net.n_links)

    A_total = A_rain + A_waa + baseline + noise
    if cfg.quantization_db > 0:
        A_observed = np.round(A_total / cfg.quantization_db) * cfg.quantization_db
    else:
        A_observed = A_total
    A_observed = np.clip(A_observed, 0.0, None)

    R_retrieved = retrieve_rain(A_observed, net, cfg)
    R_retrieved_clean = _invert_power_law(A_rain, net)

    return {
        "samples": samples,
        "R_path_true": R_path_true,
        "A_rain": A_rain,
        "A_uniform": A_uniform,
        "A_waa": A_waa,
        "A_observed": A_observed,
        "R_retrieved": R_retrieved,
        "R_retrieved_clean": R_retrieved_clean,
    }


def forward_series(frames: np.ndarray, grid: Grid, net: CMLNetwork, dt_min: float,
                   cfg: Optional[SensorConfig] = None, wet_antenna=None,
                   n_samples: int = N_PATH_SAMPLES) -> dict:
    """The sensor chain over a sequence of fields ``(time, n, n)``, ``dt_min`` apart.

    As :func:`forward_model` frame by frame, with what a time series adds: the
    baseline offset is drawn once per link and held, the receiver noise is
    fresh at every step, and with ``wet_antenna`` (a
    ``wet_antenna.DynamicWetAntenna``) the wet antenna carries over from one
    step to the next. Returns ``R_path_true``, ``A_rain``, ``A_uniform``,
    ``A_waa``, ``baseline`` (per link), ``A_observed`` and ``R_retrieved``,
    each ``(n_links, time)`` except ``baseline``.
    """
    cfg = cfg or SensorConfig()
    rng = np.random.default_rng(cfg.seed)
    T, nl = frames.shape[0], net.n_links
    L, k, alpha = net.length_km, net.k, net.alpha
    out = {name: np.zeros((nl, T)) for name in ("R_path_true", "A_rain", "A_uniform", "A_waa")}
    waa = np.zeros(nl)
    for t in range(T):
        s = sample_along_paths(frames[t], grid, net, n_samples)
        r = s.mean(axis=1)
        out["R_path_true"][:, t] = r
        out["A_rain"][:, t] = k * (s ** alpha[:, None]).mean(axis=1) * L
        out["A_uniform"][:, t] = k * r**alpha * L
        if wet_antenna is None:
            waa = wet_antenna_db(r, cfg.waa_max_db, cfg.waa_rate_per_mm_h)
        else:
            waa = wet_antenna.step(waa, r, dt_min)
        out["A_waa"][:, t] = waa
    baseline = rng.normal(0.0, cfg.baseline_sigma_db, nl)
    noise = rng.normal(0.0, cfg.noise_sigma_db, (nl, T))
    A_total = out["A_rain"] + out["A_waa"] + baseline[:, None] + noise
    if cfg.quantization_db > 0:
        A_total = np.round(A_total / cfg.quantization_db) * cfg.quantization_db
    out["A_observed"] = np.clip(A_total, 0.0, None)
    out["baseline"] = baseline
    out["R_retrieved"] = retrieve_rain(out["A_observed"].T, net, cfg).T
    return out


def _invert_power_law(A_rain_db: np.ndarray, net: CMLNetwork) -> np.ndarray:
    """Solve ``A = k R**alpha L`` for R."""
    specific = np.clip(A_rain_db, 0.0, None) / (net.k * net.length_km)
    return specific ** (1.0 / net.alpha)


def retrieve_rain(A_observed: np.ndarray, net: CMLNetwork,
                  cfg: SensorConfig, n_iter: int = 12) -> np.ndarray:
    """Recover path-averaged rain rate from observed attenuation.

    The wet-antenna term depends on the rain rate we are trying to find, so the
    correction is applied by fixed-point iteration - the standard practical
    approach. The assumed coefficients differ from the true ones, which is the
    realistic case.
    """
    R = _invert_power_law(A_observed, net)
    for _ in range(n_iter):
        waa = wet_antenna_db(R, cfg.waa_max_db_assumed, cfg.waa_rate_assumed)
        R = _invert_power_law(A_observed - waa, net)
    return np.clip(R, 0.0, None)


def path_averaging_bias(result: dict, net: CMLNetwork) -> dict:
    """Quantify the Jensen-inequality bias, split by the sign of alpha - 1."""
    true_avg = result["R_path_true"]
    clean = result["R_retrieved_clean"]
    wet = true_avg > 0.1
    if not wet.any():
        return {"n_wet": 0}

    rel = np.zeros_like(true_avg)
    rel[wet] = (clean[wet] - true_avg[wet]) / true_avg[wet]

    hi = wet & (net.alpha > 1.0)
    lo = wet & (net.alpha < 1.0)
    return {
        "n_wet": int(wet.sum()),
        "median_rel_bias": float(np.median(rel[wet])),
        "median_rel_bias_alpha_gt1": float(np.median(rel[hi])) if hi.any() else float("nan"),
        "median_rel_bias_alpha_lt1": float(np.median(rel[lo])) if lo.any() else float("nan"),
        "n_alpha_gt1": int(hi.sum()),
        "n_alpha_lt1": int(lo.sum()),
    }
