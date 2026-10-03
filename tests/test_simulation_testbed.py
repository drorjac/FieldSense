"""The field simulator, its sensors and the bridge to the mapping and nowcasting code."""

from __future__ import annotations

import numpy as np
import pytest

from core.simulation import fields_1d as f1
from core.simulation import flows as fl
from core.simulation import generators as gen
from core.simulation import random_fields as rf
from core.simulation import sensors as sn
from core.simulation import spacetime as st
from core.simulation.rain_fields import Grid, field_stats

GRID = Grid(n=64, dx_km=0.5)


# --------------------------------------------------------------------------- random fields
def test_grf_is_standardised_and_anisotropic():
    rng = np.random.default_rng(0)
    g = rf.grf(128, 0.5, rng, "matern", length_km=4.0, nu=1.0, anisotropy=4.0, angle_deg=0.0)
    assert abs(g.mean()) < 1e-9 and abs(g.std() - 1) < 1e-9
    # stretched along x: neighbours along x are more alike than along y
    rx = np.corrcoef(g[:, :-4].ravel(), g[:, 4:].ravel())[0, 1]
    ry = np.corrcoef(g[:-4].ravel(), g[4:].ravel())[0, 1]
    assert rx > ry + 0.1


@pytest.mark.parametrize("dist", rf.WET_DISTRIBUTIONS)
def test_to_rain_has_exact_wet_fraction_and_mean(dist):
    g = rf.grf(128, 0.5, np.random.default_rng(1), "exponential", 5.0)
    r = rf.to_rain(g, war=0.3, dist=dist, mean_wet=4.0, cv_wet=1.0, threshold_mm_h=0.0)
    assert abs((r > 0).mean() - 0.3) < 0.002
    assert abs(r[r > 0].mean() - 4.0) / 4.0 < 0.1


# --------------------------------------------------------------------------- generators
@pytest.mark.parametrize("key", sorted(set(gen.REGISTRY) | set(gen.PRESETS)))
def test_every_generator_builds_rain(key):
    r = gen.make(key, seed=2).build(GRID)
    assert r.shape == GRID.shape and np.isfinite(r).all() and (r >= 0).all()
    assert 0.0 < (r > 0.1).mean() <= 1.0
    # same seed, same field
    assert np.array_equal(r, gen.make(key, seed=2).build(GRID))


def test_hycell_skirt_widens_cells():
    common = dict(n_cells=5, poisson=False, seed=4)
    # HyCell is the Gaussian core or the skirt, whichever is larger: more area above 1 mm/h
    g = gen.make("gaussian_cells", profile="gaussian", **common).build(GRID)
    h = gen.make("gaussian_cells", profile="hycell", **common).build(GRID)
    assert (h >= 1).mean() > (g >= 1).mean()


def test_cascade_is_intermittent_and_mean_preserving_on_average():
    means = [gen.make("cascade", beta=0.3, smooth_cells=0, seed=s).build(GRID).mean() for s in range(20)]
    assert 1.0 < np.mean(means) < 3.0
    r = gen.make("cascade", beta=0.6, smooth_cells=0, seed=0).build(GRID)
    assert (r == 0).mean() > 0.5


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        gen.make("no such model")


# --------------------------------------------------------------------------- flows
@pytest.mark.parametrize("kind", ["rotation", "shear", "deformation", "random"])
def test_flows_are_divergence_free(kind):
    flow = fl.make_flow(kind).prepare(32.0, Grid(n=64, dx_km=0.5))
    v = flow.field(Grid(n=64, dx_km=0.5))
    div = np.gradient(v[0], 0.5, axis=1) + np.gradient(v[1], 0.5, axis=0)
    assert np.abs(div[2:-2, 2:-2]).max() < 0.05 * np.abs(v).max() / 0.5


# --------------------------------------------------------------------------- space-time
def test_frozen_fields_are_exactly_predictable_by_their_flow():
    for flow in (fl.UniformFlow(mean=(24, 6)), fl.RotationFlow(omega_deg_h=40, mean=(10, 0))):
        s = st.simulate(gen.make("stratiform_matern", seed=1), GRID, 7, 5, flow=flow, evolution="frozen")
        assert s.predictability(3) > 0.97


def test_evolution_lowers_predictability_and_tau_controls_it():
    m = gen.make("scale_free", seed=1)
    fast = st.simulate(m, GRID, 9, 5, evolution="ar1", tau_min=20, seed=1).predictability(6)
    slow = st.simulate(m, GRID, 9, 5, evolution="ar1", tau_min=400, seed=1).predictability(6)
    assert fast < slow < 1.0


def test_ar1_keeps_the_intensity_distribution():
    s = st.simulate(gen.make("convective_cells", seed=3), GRID, 6, 5, evolution="cascade", seed=3)
    q0, q5 = np.percentile(s.frames[0], [50, 99]), np.percentile(s.frames[5], [50, 99])
    assert np.allclose(q0, q5, rtol=0.35, atol=0.5)


def test_frontal_band_moves_with_the_flow_under_evolution():
    s = st.simulate(gen.make("frontal", seed=1), GRID, 7, 5, evolution="ar1", tau_min=60)
    assert s.predictability(6) > 0.8


def test_lifecycle_needs_cells_and_produces_them():
    with pytest.raises(TypeError):
        st.simulate(gen.make("metagaussian"), GRID, 3, 5, evolution="lifecycle")
    s = st.simulate(gen.make("convective_cells", seed=1), GRID, 5, 5, evolution="lifecycle")
    assert (s.frames > 0.1).any(axis=(1, 2)).all()
    assert s.extras["cells_in_view"].min() > 0


def test_cloud_model_rains():
    from core.simulation.cloud_model import WarmRainModel
    s = WarmRainModel(seed=1, spinup_min=40).simulate(Grid(n=48, dx_km=0.5), 4, 5)
    assert s.frames.max() > 5 and (s.frames >= 0).all()
    assert set(s.extras) == {"cloud_water", "vapour", "updraft"}


# --------------------------------------------------------------------------- 1-D
def test_point_process_series():
    rng = np.random.default_rng(0)
    for fn in (f1.bartlett_lewis, f1.neyman_scott):
        r = fn(240, 5, rng, storm_rate_per_h=0.05)
        assert r.shape == (240 * 12,) and (r >= 0).all() and r.max() > 0
    d, v = f1.transect(np.ones((10, 10)), 1.0, 0, 0, 6, 8)
    assert d[-1] == pytest.approx(10.0) and np.allclose(v, 1.0)


# --------------------------------------------------------------------------- sensors
def _frames(n=11):
    return st.simulate(gen.make("stratiform_matern", seed=2), GRID, n, 1.0, evolution="frozen").frames


def test_ideal_radar_recovers_the_rain():
    cfg = sn.RadarConfig(resolution_km=0.5, dsd_sigma_db=0, noise_db=0, quantization_db=0, attenuation=False,
                         beam_width_deg=0.0, mds_dbz_1km=-99, echo_top_km=99)
    f = _frames()
    r = sn.Radar(cfg, GRID).observe(f, 1.0, 5.0)
    truth = f[5] * 5 / 60
    assert np.allclose(r[0], truth, rtol=1e-4, atol=1e-4)


def test_x_band_attenuation_makes_the_radar_underestimate():
    f = _frames() * 10                                    # heavy rain
    kw = dict(dsd_sigma_db=0, noise_db=0, quantization_db=0, mds_dbz_1km=-99, echo_top_km=99)
    s = sn.Radar(sn.RadarConfig(band="S", **kw), GRID).observe(f, 1.0, 5.0).sum()
    x = sn.Radar(sn.RadarConfig(band="X", **kw), GRID).observe(f, 1.0, 5.0).sum()
    assert x < 0.9 * s


def test_ideal_gauges_and_links_measure_the_truth():
    f = _frames()
    g = sn.Gauges(sn.GaugeConfig(n=10, tip_mm=0, undercatch_mean=1, undercatch_sd=0), GRID).observe(f, 1, 5)
    assert np.allclose(g["rain_mm"], g["true_mm"])
    cfg = sn.CMLConfig(n_links=20, waa_max_db=0, waa_max_db_assumed=0, baseline_sigma_db=0,
                       baseline_drift_db_per_sqrt_h=0, noise_sigma_db=0, quantization_db=0, outage_frac=0,
                       wet_dry=False)
    c = sn.CMLs(cfg, GRID).observe(f, 1, 5)
    wet = c["true_mm"] > 0.05
    # only the path-averaging non-linearity is left
    assert np.median(np.abs(c["rain_mm"][wet] / c["true_mm"][wet] - 1)) < 0.05


def test_faulty_gauges():
    f = _frames()
    cfg = sn.GaugeConfig(n=200, dead_frac=0.2, stuck_zero_frac=0.2, seed=1)
    o = sn.Gauges(cfg, GRID).observe(f, 1, 5)["rain_mm"]
    assert 0.1 < np.isnan(o[:, 0]).mean() < 0.3


# --------------------------------------------------------------------------- scenario + methods
@pytest.fixture(scope="module")
def case():
    from core.simulation.scenario import Scenario
    return Scenario(model="convective_cells", n=64, dx_km=1.0, analysis_factor=1, duration_min=60,
                    truth_dt_min=2.5, n_links=30, n_gauges=12, n_pws=8, seed=3).run()


def test_scenario_produces_the_formats_of_the_real_data(case):
    T = case.truth.sizes["time"]
    assert case.truth.dims == ("time", "lat", "lon") and case.radar.shape == case.truth.shape
    assert case.links.dims == ("link", "time") and case.links.sizes["time"] == T
    for c in ("site_0_lat", "site_1_lon", "mid_lat", "frequency", "polarization", "true_mm"):
        assert c in case.links.coords
    assert case.gauges.dims == ("station", "time") and "lat" in case.gauges.coords
    assert case.velocity.shape == (2,) + case.truth.shape[1:]
    # the analysis grid is 1 km square
    from core.nowcast.grid import pixel_km
    assert pixel_km(case.geo_grid) == pytest.approx(1.0, rel=0.01)


def test_every_map_method_runs_on_a_synthetic_case(case):
    from core.simulation import benchmark as bm
    timings = {}
    products = bm.maps(case, with_mergeplg=False, timings=timings)
    assert not [k for k in timings if k.endswith(":error")]
    df = bm.score_maps(case, products)
    assert {"radar", "idw links", "gmz links", "idw gauges", "mfb [links+gauges]"} <= set(df.method)
    assert df.set_index("method").loc["radar", "corr"] > 0.5


def test_motion_score_is_zero_for_the_truth(case):
    from core.simulation import benchmark as bm
    s = bm.score_motion(case.velocity, case.velocity)
    assert s["epe_kmh"] == 0 and s["dir_err_deg"] == 0


# --------------------------------------------------------------------------- learned motion
def test_rotation_augmentation_rotates_the_motion():
    torch = pytest.importorskip("torch")
    from core.nowcast import learned_motion as lm
    base = rf.grf(32, 1.0, np.random.default_rng(0), "gaussian", 3.0)
    frames = np.stack([np.roll(np.roll(base, i * 1, axis=0), i * 2, axis=1) for i in range(2)])  # u=2, v=1
    y = np.zeros((1, 2, 32, 32), np.float32)
    y[0, 0], y[0, 1] = 2, 1
    for k in range(4):
        for flip in (False, True):
            xa, ya = lm._augment(torch.from_numpy(frames[None].astype(np.float32)), torch.from_numpy(y), k, flip)
            a, b = xa[0, 0].numpy(), xa[0, 1].numpy()
            u, v = int(ya[0, 0, 0, 0]), int(ya[0, 1, 0, 0])
            assert np.allclose(np.roll(np.roll(a, v, axis=0), u, axis=1), b, atol=1e-5)


def test_wet_dry_processing_removes_the_false_rain_of_a_baseline_offset():
    f = np.zeros((61,) + GRID.shape, np.float32)          # an hour without rain
    cfg = dict(n_links=20, baseline_sigma_db=0.5, outage_frac=0, seed=2)
    raw = sn.CMLs(sn.CMLConfig(wet_dry=False, **cfg), GRID).observe(f, 1, 5)["rain_mm"]
    processed = sn.CMLs(sn.CMLConfig(**cfg), GRID).observe(f, 1, 5)["rain_mm"]
    assert np.nansum(raw) > 1.0
    assert np.nansum(processed) < 0.05 * np.nansum(raw)


# --------------------------------------------------------------------------- resolution tools
def test_pia_correction_restores_x_band_rain():
    f = _frames() * 10
    kw = dict(band="X", dsd_sigma_db=0, noise_db=0, quantization_db=0, mds_dbz_1km=-99, echo_top_km=99)
    raw = sn.Radar(sn.RadarConfig(**kw), GRID).observe(f, 1.0, 5.0).sum()
    fixed = sn.Radar(sn.RadarConfig(pia_correction=1.0, **kw), GRID).observe(f, 1.0, 5.0).sum()
    s = sn.Radar(sn.RadarConfig(**{**kw, "band": "S"}), GRID).observe(f, 1.0, 5.0).sum()
    assert raw < 0.9 * fixed and abs(fixed / s - 1) < 0.05


def test_texture_adds_small_scale_variance_and_keeps_the_mean():
    from core.simulation.sensors import block_mean
    g = Grid(n=128, dx_km=0.1)
    plain = gen.make("convective_cells", seed=1, n_cells=4, poisson=False).build(g)
    rough = gen.make("convective_cells", seed=1, n_cells=4, poisson=False, texture_strength=0.4).build(g)

    def sub(r):
        return (r - np.kron(block_mean(r[None], 8)[0], np.ones((8, 8)))).var() / r.var()
    assert sub(rough) > 2 * sub(plain)
    assert 0.7 < rough.sum() / plain.sum() < 1.3


def test_evolve_every_keeps_the_marginal_and_moves_exactly_between_updates():
    s = st.simulate(gen.make("scale_free", seed=1), GRID, 6, 1.0, evolution="ar1", tau_min=1e9,
                    evolve_every=5)
    assert s.predictability(1) > 0.98       # only fractional-shift interpolation is lost


def test_score_scales_and_link_distance(case):
    from core.simulation import benchmark as bm
    prods = {"radar": case.radar, "truth": case.truth}
    d = bm.score_scales(case, prods, space_km=(1.0, 4.0), time_min=(5, 15))
    t = d[d.method == "truth"]
    assert np.allclose(t.nrmse, 0, atol=1e-5)          # the truth scores perfectly at every scale
    r = d[d.method == "radar"].set_index(["time_min", "space_km"]).nrmse
    assert r[(15, 4.0)] < r[(5, 1.0)]                  # coarser is easier
    dist = bm.link_distance_km(case)
    assert dist.shape == case.truth_fine.shape[1:] and dist.min() < 0.5


def test_a_case_carries_every_radar():
    from core.simulation.scenario import Scenario
    c = Scenario(model="scale_free", n=64, dx_km=0.5, analysis_factor=2, duration_min=10, truth_dt_min=5,
                 n_links=0, n_gauges=0, extra_radars={"xband": sn.RadarConfig(band="X", resolution_km=0.5)},
                 seed=1).run()
    assert set(c.radars) == {"radar", "xband"} and c.radars["xband"].shape == c.truth.shape
