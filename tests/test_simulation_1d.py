"""The link simulator (cml_network, reconstruct), 1-D fields and the wet antenna with memory.

All synthetic: every case has an answer known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.itu_p838 import get_k_alpha
from core.simulation import cml_network as cn
from core.simulation import fields_1d as f1
from core.simulation import reconstruct as rc
from core.simulation import wet_antenna as wa
from core.simulation.rain_fields import Grid

GRID = Grid(n=64, dx_km=0.5)

# no wet antenna, no baseline error, no noise, no quantization
CLEAN = cn.SensorConfig(waa_max_db=0.0, waa_max_db_assumed=0.0, baseline_sigma_db=0.0,
                        noise_sigma_db=0.0, quantization_db=0.0)


def links(freqs, pol="horizontal", y_km=None, length_km=6.0):
    """Horizontal links (along x), one per frequency, at rows ``y_km``."""
    n = len(freqs)
    y = np.linspace(4.0, GRID.size_km - 4.0, n) if y_km is None else np.asarray(y_km, float)
    xa = np.full(n, 3.0)
    ka = np.array([get_k_alpha(f, pol) for f in freqs])
    return cn.CMLNetwork(xa=xa, ya=y, xb=xa + length_km, yb=y.copy(),
                         freq_ghz=np.asarray(freqs, float), pol=np.array([pol] * n),
                         k=ka[:, 0], alpha=ka[:, 1])


@pytest.fixture(scope="module")
def net():
    return cn.synthesize_network(GRID, n_links=30, n_nodes=24, seed=3)


# --------------------------------------------------------------------------- cml_network
def test_sample_along_paths_of_uniform_rain_is_the_rain(net):
    s = cn.sample_along_paths(np.full(GRID.shape, 4.2), GRID, net)
    assert s.shape == (net.n_links, cn.N_PATH_SAMPLES)
    np.testing.assert_allclose(s, 4.2, rtol=1e-12)


@pytest.mark.parametrize("rate", [0.5, 5.0, 40.0])
def test_forward_then_retrieve_recovers_uniform_rain(net, rate):
    out = cn.forward_model(np.full(GRID.shape, rate), GRID, net, CLEAN)
    np.testing.assert_allclose(out["A_rain"], out["A_uniform"], rtol=1e-12)
    np.testing.assert_allclose(out["A_observed"], out["A_rain"], rtol=1e-12)
    np.testing.assert_allclose(out["R_retrieved"], rate, rtol=1e-9)
    np.testing.assert_allclose(out["R_retrieved_clean"], rate, rtol=1e-9)
    # and the attenuation is the ITU law k R^alpha L
    np.testing.assert_allclose(out["A_rain"], net.k * rate**net.alpha * net.length_km, rtol=1e-12)


def test_quantization_step_is_respected(net):
    cfg = cn.SensorConfig(quantization_db=0.3)
    rain = np.random.default_rng(0).gamma(1.0, 3.0, GRID.shape)
    a = cn.forward_model(rain, GRID, net, cfg)["A_observed"]
    np.testing.assert_allclose(a / 0.3, np.round(a / 0.3), atol=1e-9)
    assert len(np.unique(np.round(a, 9))) > 3


def test_path_averaging_bias_is_zero_for_uniform_rain(net):
    out = cn.forward_model(np.full(GRID.shape, 6.0), GRID, net, CLEAN)
    b = cn.path_averaging_bias(out, net)
    assert b["n_wet"] == net.n_links
    assert abs(b["median_rel_bias"]) < 1e-9


def test_path_averaging_bias_has_the_sign_of_alpha_minus_one():
    # rain varying along x, so every horizontal link sees a non-uniform profile
    xx, _ = GRID.meshgrid()
    rain = 5.0 * (1.0 + 0.9 * np.sin(2 * np.pi * xx / 8.0))
    net = links([10, 10, 15, 38, 38, 60])                 # alpha 1.26, 1.26, 1.12, 0.88, 0.88, 0.77
    out = cn.forward_model(rain, GRID, net, CLEAN)
    assert np.all((out["A_rain"] > out["A_uniform"]) == (net.alpha > 1))   # Jensen
    b = cn.path_averaging_bias(out, net)
    assert b["n_alpha_gt1"] == 3 and b["n_alpha_lt1"] == 3
    assert b["median_rel_bias_alpha_gt1"] > 0.005
    assert b["median_rel_bias_alpha_lt1"] < -0.005


# --------------------------------------------------------------------------- reconstruct
@pytest.mark.parametrize("method", [rc.idw_midpoint, rc.idw_path])
def test_idw_is_exact_on_uniform_rain(net, method):
    field = method(net, np.full(net.n_links, 3.3), GRID)
    assert field.shape == GRID.shape
    np.testing.assert_allclose(field, 3.3, rtol=1e-12)


def test_score_of_identical_fields():
    t = np.random.default_rng(1).gamma(0.6, 4.0, GRID.shape)
    s = rc.score(t, t.copy())
    assert s["rmse"] == 0 and s["mae"] == 0 and s["bias"] == 0
    assert s["corr"] == pytest.approx(1.0)
    assert s["peak_ratio"] == 1.0 and s["war_true"] == s["war_est"]
    d = rc.decompose(t, t, t)
    assert d["rmse_total"] == 0 and d["rmse_sensor_increment"] == 0


# --------------------------------------------------------------------------- fields_1d
@pytest.mark.parametrize("war", [0.2, 0.5, 0.8])
def test_metagaussian_1d_hits_the_wet_fraction(war):
    r = f1.metagaussian_1d(4096, 0.1, np.random.default_rng(2), war=war, mean_wet=3.0)
    assert r.shape == (4096,) and (r >= 0).all()
    assert abs((r > 0).mean() - war) < 0.01


def test_cascade_1d_has_the_right_length_and_conserves_the_mean_in_expectation():
    r = f1.cascade_1d(9, np.random.default_rng(0), mean=2.0)
    assert r.shape == (2**9,) and (r >= 0).all() and (r == 0).any()
    means = [f1.cascade_1d(8, np.random.default_rng(s), mean=2.0).mean() for s in range(400)]
    assert np.mean(means) == pytest.approx(2.0, rel=0.1)


def test_series_stats_on_a_known_series():
    r = np.array([0, 0, 2, 2, 2, 0, 0, 0, 1, 0], dtype=float)
    s = f1.series_stats(r, dt_min=5.0)
    assert set(s) == {"wet_fraction", "mean_mm_h", "acf1", "mean_wet_spell_min",
                      "mean_dry_spell_min"}
    assert s["wet_fraction"] == pytest.approx(0.4)
    assert s["mean_mm_h"] == pytest.approx(0.7)
    assert s["mean_wet_spell_min"] == pytest.approx((15 + 5) / 2)
    assert s["mean_dry_spell_min"] == pytest.approx((10 + 15 + 5) / 3)
    assert -1 <= s["acf1"] <= 1


# --------------------------------------------------------------------------- wet antenna
def test_static_models():
    wet = np.array([0, 1, 1, 0, 1], dtype=bool)
    np.testing.assert_array_equal(wa.waa_constant(wet, 1.5), [0, 1.5, 1.5, 0, 1.5])
    r = np.array([0.0, 0.5, 5.0, 50.0])
    # zeta = 1 is the saturating form the simulator already uses
    np.testing.assert_allclose(wa.waa_pastorek_2021(r, 2.3, 0.28, 1.0),
                               cn.wet_antenna_db(r, 2.3, 0.28), rtol=1e-12)
    assert wa.waa_pastorek_2021(np.array([0.0]))[0] == 0.0


def test_schleiss_builds_up_to_95_percent_after_tau_and_resets_when_dry():
    wet = np.r_[np.zeros(1, bool), np.ones(300, bool), np.zeros(5, bool)]
    w = wa.waa_schleiss_2013(wet, waa_max_db=2.0, tau_min=60.0, dt_min=1.0)
    assert np.all(np.diff(w[1:301]) >= 0)
    assert w[60] == pytest.approx(2.0 * 0.95, abs=0.02)
    assert w[300] == pytest.approx(2.0, abs=1e-3) and (w[301:] == 0).all()


def test_schleiss_with_a_obs_matches_pycomlink():
    pcw = pytest.importorskip("pycomlink.processing.wet_antenna")
    rng = np.random.default_rng(4)
    a_obs = np.clip(rng.normal(1.0, 1.0, 400), 0, None)
    wet = rng.random(400) < 0.6
    ours = wa.waa_schleiss_2013(wet, 2.0, 15.0, 1.0, a_obs=a_obs)
    theirs = pcw.waa_schleiss_2013(rsl=a_obs, baseline=np.zeros(400), wet=wet.astype(float),
                                   waa_max=2.0, delta_t=1.0, tau=15.0)
    np.testing.assert_allclose(ours, np.asarray(theirs), rtol=1e-12)


def test_pastorek_matches_pycomlink():
    pcw = pytest.importorskip("pycomlink.processing.wet_antenna")
    r = np.array([0.0, 0.3, 2.0, 20.0, 100.0])
    np.testing.assert_allclose(wa.waa_pastorek_2021(r), pcw.waa_pastorek_2021(r), rtol=1e-12)


MODEL = wa.DynamicWetAntenna(a_max_db=2.0, gamma_per_mm_h=0.3, tau_wet_min=5.0, tau_dry_min=30.0)


def test_dynamic_builds_up_monotonically_to_the_equilibrium():
    d = MODEL.simulate(np.full(200, 4.0), dt_min=1.0)
    eq = float(MODEL.equilibrium(4.0))
    assert np.all(np.diff(d[:60]) > 0) and np.all(np.diff(d) >= 0) and np.all(d <= eq)
    assert d[-1] == pytest.approx(eq, rel=1e-9)                           # steady state
    assert d[4] == pytest.approx(eq * (1 - np.exp(-1.0)), rel=1e-9)       # tau_wet, exactly
    assert float(MODEL.rate(eq, 4.0)) == 0.0


def test_dynamic_dries_with_an_exponential_tail_of_tau_dry():
    rain = np.r_[np.full(120, 8.0), np.zeros(180)]
    d = MODEL.simulate(rain, dt_min=1.0)
    tail = d[120:]
    assert np.all(np.diff(tail) < 0) and tail[-1] > 0
    slope = np.polyfit(np.arange(tail.size), np.log(tail), 1)[0]
    assert -1.0 / slope == pytest.approx(MODEL.tau_dry_min, rel=1e-6)
    # lighter rain: dries towards the lower equilibrium with tau_dry as well
    d2 = MODEL.simulate(np.r_[np.full(120, 8.0), np.full(60, 1.0)], dt_min=1.0)
    eq1 = float(MODEL.equilibrium(1.0))
    np.testing.assert_allclose(d2[120:] - eq1, (d2[119] - eq1) * np.exp(-np.arange(1, 61) / 30.0),
                               rtol=1e-9)


def test_dynamic_step_is_exact_for_any_time_step():
    fine = MODEL.simulate(np.full(60, 3.0), dt_min=0.5)[-1]
    coarse = MODEL.simulate(np.full(3, 3.0), dt_min=10.0)[-1]
    assert fine == pytest.approx(coarse, rel=1e-12)
    # a per-link state along the last axis
    many = MODEL.simulate(np.array([[0.0] * 10, [5.0] * 10]), dt_min=1.0, delta0=[1.0, 0.0])
    assert many.shape == (2, 10) and many[0, -1] < 1.0 and many[1, -1] > 0


def test_forward_model_default_is_unchanged(net):
    rain = np.random.default_rng(5).gamma(0.8, 3.0, GRID.shape)
    out = cn.forward_model(rain, GRID, net)
    same = cn.forward_model(rain, GRID, net, wet_antenna=None, waa_state=np.ones(net.n_links))
    for key in out:
        np.testing.assert_array_equal(out[key], same[key])
    # rebuilt from the documented chain: static wet antenna, baseline, noise, quantization
    cfg = cn.SensorConfig()
    rng = np.random.default_rng(cfg.seed)
    s = cn.sample_along_paths(rain, GRID, net)
    a_rain = net.k * (s ** net.alpha[:, None]).mean(axis=1) * net.length_km
    total = (a_rain + cn.wet_antenna_db(s.mean(axis=1), cfg.waa_max_db, cfg.waa_rate_per_mm_h)
             + rng.normal(0.0, cfg.baseline_sigma_db, net.n_links)
             + rng.normal(0.0, cfg.noise_sigma_db, net.n_links))
    expected = np.clip(np.round(total / cfg.quantization_db) * cfg.quantization_db, 0.0, None)
    np.testing.assert_array_equal(out["A_observed"], expected)
    np.testing.assert_array_equal(out["A_rain"], a_rain)


def test_forward_model_with_a_dynamic_wet_antenna_carries_state(net):
    rain = np.full(GRID.shape, 5.0)
    first = cn.forward_model(rain, GRID, net, wet_antenna=MODEL, dt_min=5.0)
    np.testing.assert_allclose(first["A_waa"], MODEL.equilibrium(5.0) * (1 - np.exp(-1.0)))
    second = cn.forward_model(rain, GRID, net, wet_antenna=MODEL, waa_state=first["A_waa"],
                              dt_min=5.0)
    assert np.all(second["A_waa"] > first["A_waa"])
    dry = cn.forward_model(np.zeros(GRID.shape), GRID, net, wet_antenna=MODEL,
                           waa_state=second["A_waa"], dt_min=30.0)
    np.testing.assert_allclose(dry["A_waa"], second["A_waa"] * np.exp(-1.0))
    assert (dry["A_rain"] == 0).all()


def test_forward_series_static_and_dynamic(net):
    frames = np.concatenate([np.full((20,) + GRID.shape, 6.0), np.zeros((40,) + GRID.shape)])
    static = cn.forward_series(frames, GRID, net, dt_min=1.0, cfg=CLEAN)
    assert static["A_observed"].shape == (net.n_links, 60)
    np.testing.assert_allclose(static["R_retrieved"][:, :20], 6.0, rtol=1e-9)
    assert (static["A_waa"] == 0).all()
    dyn = cn.forward_series(frames, GRID, net, dt_min=1.0, cfg=CLEAN, wet_antenna=MODEL)
    np.testing.assert_allclose(dyn["A_waa"][0], MODEL.simulate(frames[:, 0, 0], 1.0), rtol=1e-12)
    # after the rain, everything left in the observed attenuation is the drying antenna
    np.testing.assert_allclose(dyn["A_observed"][:, 20:], dyn["A_waa"][:, 20:], rtol=1e-12)
