"""
Rank the merge methods against a *known* rainfall field.

The problem this solves: on real data there is no truth. Scoring a merged
field against gauges compares it to ten points, each with its own error, and
those same gauges are often an input to the merge. Any ranking built that way
is partly circular and mostly blind to what happens between the gauges.

So the methods are first scored where truth is exact. The rainfall fields come
from ``projects/simulation/regimes`` - stratiform, convective and frontal, with
very different wet-area ratios and decorrelation lengths. The *sampling* is
real: the CML endpoints, lengths, frequencies and polarizations are the actual
OpenMRG network, and the grid is the actual OpenMRG radar grid. Only the
rainfall is synthetic.

Each method then sees:

* CML observations - the synthetic field integrated along each real link path
  and pushed through the ITU-R forward model and back, so the path-averaging
  bias and the retrieval error are present.
* A radar field - the synthetic truth degraded the way real radar is: smoothed
  to the beam, biased by a Z-R power law, and noisy.
* Gauges - point samples at the real municipal gauge positions.

The resulting ranking is what justifies the method choice on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
from core.data_paths import REPO_ROOT  # noqa: E402
sys.path.insert(0, str(HERE))

from core.itu_p838 import get_k_alpha  # noqa: E402
from merging import METHODS, run, score            # noqa: E402
from core.simulation.rain_fields import Grid  # noqa: E402

N_PATH_SAMPLES = 96


def grid_from_radar(ds_rad: xr.Dataset) -> tuple[np.ndarray, np.ndarray, Grid]:
    """A simulation grid matching the real radar footprint.

    The OpenMRG radar grid is ~1 km and 37x48 cells. The synthetic fields are
    generated on a square power-of-two grid for the FFT, then cropped, so the
    spectral generator stays exact.
    """
    xg = np.asarray(ds_rad.x_grid)
    yg = np.asarray(ds_rad.y_grid)
    ny, nx = xg.shape

    dx_km = float(np.abs(np.diff(xg[ny // 2, :])).mean()) / 1000.0
    n = int(2 ** np.ceil(np.log2(max(ny, nx))))
    return xg, yg, Grid(n=n, dx_km=dx_km)


def sample_paths(field: np.ndarray, sim: Grid, x0, y0, x1, y1,
                 origin_x: float, origin_y: float) -> np.ndarray:
    """Sample a synthetic field along every real link path.

    Link endpoints are in UTM metres; they are shifted to the simulation grid's
    own origin and converted to fractional cell indices.
    """
    from scipy.ndimage import map_coordinates

    s = np.linspace(0.0, 1.0, N_PATH_SAMPLES)[None, :]
    xs = (x0[:, None] + s * (x1 - x0)[:, None] - origin_x) / 1000.0
    ys = (y0[:, None] + s * (y1 - y0)[:, None] - origin_y) / 1000.0

    col = xs / sim.dx_km - 0.5
    row = ys / sim.dx_km - 0.5
    out = map_coordinates(field, [row.ravel(), col.ravel()], order=1,
                          mode="grid-wrap")
    return np.clip(out.reshape(xs.shape), 0.0, None)


def synth_cml_observations(field, sim, ds_cml, origin_x, origin_y, rng,
                           noise_db=0.15, quant_db=0.3,
                           waa_max_db=0.5, waa_rate=0.28) -> np.ndarray:
    """Forward-model then retrieve rain on the real link geometry.

    Uses the real length, frequency and polarization of every link, so the
    ITU-R coefficients and the path-averaging non-linearity are the network's
    own.
    """
    x0 = np.asarray(ds_cml.site_0_x)
    y0 = np.asarray(ds_cml.site_0_y)
    x1 = np.asarray(ds_cml.site_1_x)
    y1 = np.asarray(ds_cml.site_1_y)
    length = np.asarray(ds_cml.length)
    freq = np.asarray(ds_cml.frequency)

    pol = ds_cml.polarization.values if "polarization" in ds_cml.coords \
        else np.array(["vertical"] * len(freq))
    pol = np.array([str(p).lower() for p in np.atleast_1d(pol)])
    if pol.size != freq.size:
        pol = np.array(["vertical"] * freq.size)

    ka = np.array([get_k_alpha(float(f), p if p in ("vertical", "horizontal")
                               else "vertical")
                   for f, p in zip(freq, pol)])
    k, alpha = ka[:, 0], ka[:, 1]

    samples = sample_paths(field, sim, x0, y0, x1, y1, origin_x, origin_y)

    # Forward: attenuation integrates R^alpha along the path, not the mean.
    a_rain = k * (samples ** alpha[:, None]).mean(axis=1) * length
    r_path = samples.mean(axis=1)
    a_total = a_rain + waa_max_db * (1.0 - np.exp(-waa_rate * r_path))
    a_total = a_total + rng.normal(0.0, noise_db, a_total.size)
    a_obs = np.round(np.clip(a_total, 0.0, None) / quant_db) * quant_db

    # Retrieve: the same inversion the real pipeline uses.
    r = np.zeros_like(a_obs)
    for _ in range(8):
        waa = waa_max_db * (1.0 - np.exp(-waa_rate * r))
        r = (np.clip(a_obs - waa, 0.0, None) / (k * length)) ** (1.0 / alpha)
    return np.nan_to_num(np.clip(r, 0.0, None))


def synth_radar(field_on_radar_grid: np.ndarray, rng, xg=None, yg=None,
                smooth_cells: float = 1.2, zr_bias: float = 0.65,
                range_bias_per_100km: float = 0.25,
                rel_noise: float = 0.25) -> np.ndarray:
    """Degrade a truth field the way weather radar degrades it.

    Radar QPE is not an unbiased view of the rain field, and that is the whole
    reason for merging it with ground sensors. The degradations applied here,
    in the order they physically arise:

    ``smooth_cells``
        beam volume averaging, which blunts convective peaks.
    ``zr_bias``
        a multiplicative bias from the Z-R relation, calibration drift and
        attenuation. 0.65 means the radar reads ~35% low, which is within the
        range routinely reported for uncorrected C-band QPE.
    ``range_bias_per_100km``
        the bias worsens with distance as the beam rises and broadens.
    ``rel_noise``
        multiplicative, not additive - radar error scales with the signal,
        which is exactly why an additive and a multiplicative merge behave
        so differently.

    An earlier version of this function defaulted to ``zr_bias=1.0``. With an
    unbiased radar there is nothing for a merge to correct, so radar-only won
    every regime by construction and the benchmark measured nothing. Keeping
    the bias realistic is what makes the comparison meaningful.
    """
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(field_on_radar_grid, smooth_cells)

    bias = np.full_like(smoothed, zr_bias)
    if xg is not None and yg is not None:
        # Range measured from the domain's south-west corner, standing in for
        # a radar site off one edge.
        r_km = np.hypot(xg - xg.min(), yg - yg.min()) / 1000.0
        bias = bias * (1.0 - range_bias_per_100km * r_km / 100.0)
        bias = np.clip(bias, 0.15, 1.5)

    noise = np.exp(rng.normal(0.0, rel_noise, smoothed.shape))
    return np.clip(smoothed * bias * noise, 0.0, None)


def build_case(model, ds_rad, ds_cml, ds_gauge, seed: int = 7) -> dict:
    """One synthetic case: truth on the radar grid plus all three sensor views."""
    xg, yg, sim = grid_from_radar(ds_rad)
    rng = np.random.default_rng(seed)

    field = model.build(sim)
    origin_x, origin_y = float(xg.min()), float(yg.min())

    # Crop the square simulation field onto the radar footprint.
    col = ((xg - origin_x) / 1000.0 / sim.dx_km).round().astype(int)
    row = ((yg - origin_y) / 1000.0 / sim.dx_km).round().astype(int)
    col = np.clip(col, 0, sim.n - 1)
    row = np.clip(row, 0, sim.n - 1)
    truth = field[row, col]

    cml_obs = synth_cml_observations(field, sim, ds_cml, origin_x, origin_y, rng)
    radar_obs = synth_radar(truth, rng, xg=xg, yg=yg)

    gx = np.asarray(ds_gauge.x)
    gy = np.asarray(ds_gauge.y)
    gi = [np.unravel_index(np.argmin((xg - x) ** 2 + (yg - y) ** 2), xg.shape)
          for x, y in zip(gx, gy)]
    gauge_obs = np.array([truth[i, j] for i, j in gi])

    da_rad = xr.DataArray(radar_obs, dims=("y", "x"), coords=dict(
        x=ds_rad.x, y=ds_rad.y, x_grid=ds_rad.x_grid, y_grid=ds_rad.y_grid))
    da_cml = xr.DataArray(cml_obs, dims=("cml_id",),
                          coords={c: ds_cml[c] for c in
                                  ("cml_id", "site_0_x", "site_0_y",
                                   "site_1_x", "site_1_y", "x", "y")})
    da_gauge = xr.DataArray(gauge_obs, dims=("id",), coords=dict(
        id=ds_gauge.id, x=ds_gauge.x, y=ds_gauge.y))

    return {"truth": truth, "da_rad": da_rad, "da_cml": da_cml,
            "da_gauge": da_gauge, "sim": sim, "model": model}


def run_benchmark(ds_rad, ds_cml, ds_gauge, n_seeds: int = 3) -> list[dict]:
    """Score every method on every synthetic regime, averaged over seeds."""
    _, _, sim = grid_from_radar(ds_rad)
    rows = []
    for model in benchmark_models(sim):
        for method in METHODS:
            per_seed = []
            for seed in range(n_seeds):
                case = build_case(model, ds_rad, ds_cml, ds_gauge,
                                  seed=7 + 13 * seed)
                est = run(method, case["da_rad"], case["da_cml"],
                          case["da_gauge"])
                per_seed.append(score(case["truth"], np.asarray(est)))
            agg = {k: float(np.nanmean([s[k] for s in per_seed]))
                   for k in per_seed[0]}
            rows.append({"regime": model.name, "regime_key": model.key,
                         "method": method.label, "method_key": method.key,
                         "family": method.family, **agg})
    return rows


# --------------------------------------------------------------------------
# domain-scaled regimes
# --------------------------------------------------------------------------
def benchmark_models(sim: Grid) -> tuple:
    """Rainfall regimes rescaled to the radar's domain and resolution.

    ``projects/simulation/regimes`` tunes its three models for a 32 x 32 km
    domain at 250 m. The OpenMRG radar grid is ~2 km over ~126 km, so the
    defaults transplant badly: 12 convective cells of ~1.5 km radius cover 2%
    of a mesoscale domain and are sub-pixel at 2 km, which makes the benchmark
    a test of nothing.

    These variants keep each regime's *character* - wet fraction, intensity
    distribution, anisotropy - at sizes a 2 km grid can actually resolve,
    which is also the size real systems have at this scale: convective cells
    5-15 km across, a squall line with a several-km convective ribbon and a
    tens-of-km stratiform tail.
    """
    from core.simulation.rain_fields import ConvectiveField, FrontalBandField, StratiformField

    # The spectral model is scale-free, so stratiform needs no adjustment
    # beyond the grid it is generated on.
    stratiform = StratiformField()

    convective = ConvectiveField(
        n_cells=14,
        radius_floor_km=1.6,
        radius_scale=1.9,
        cutoff_radii=1.7,
        background_war=0.05,
        background_mean_mm_h=0.8,
    )

    frontal = FrontalBandField(
        conv_width_km=4.0,
        strat_width_km=11.0,
        lead_edge_km=2.5,
        centre_frac=0.5,
    )
    return (stratiform, convective, frontal)
