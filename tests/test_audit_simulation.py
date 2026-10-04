"""Audit of core/simulation: each generator, flow and sensor against its reference.

Every test is a closed form or a hand calculation the code must reproduce:
the covariance a Gaussian random field was asked for, the wet fraction and
marginal of the meta-Gaussian transform, the displacement of a blob by a known
flow, the Z-R relation, the ITU-R forward model of a link, the wet-antenna
models of pycomlink. Small grids, no downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from core.itu_p838 import get_k_alpha
from core.simulation import cml_network as cn
from core.simulation import fields_1d as f1
from core.simulation import flows as fl
from core.simulation import generators as gen
from core.simulation import random_fields as rf
from core.simulation import sensors as sn
from core.simulation import spacetime as st
from core.simulation import wet_antenna as wa
from core.simulation.moving_fields import shift
from core.simulation.rain_fields import Grid


# --------------------------------------------------------------------------- random fields
def _matern(r, length, nu):
    """Matern correlation in the convention of ``random_fields`` (scale sqrt(2 nu) r / L)."""
    x = np.sqrt(2 * nu) * np.asarray(r, dtype=float) / length
    with np.errstate(invalid="ignore"):
        c = 2 ** (1 - nu) / special.gamma(nu) * x ** nu * special.kv(nu, x)
    return np.where(x == 0, 1.0, c)


def _expected_corr_along_x(n, dx, **kw):
    """The correlation the spectrum implies on the periodic grid: inverse FFT of S(k)."""
    k = rf._wavenumbers(n, dx, 1.0, 0.0)
    s = rf.spectral_density(k, dim=2, **kw)
    s[0, 0] = 0.0
    c = np.fft.ifft2(s).real
    return c[0] / c[0, 0]


@pytest.mark.parametrize("cov, closed_form", [
    ("exponential", lambda r, L: np.exp(-r / L)),
    ("gaussian", lambda r, L: np.exp(-(r / L) ** 2)),
    ("matern", lambda r, L: _matern(r, L, 1.5)),
])
def test_spectral_density_is_the_documented_covariance(cov, closed_form):
    n, dx, L = 256, 0.25, 3.0
    corr = _expected_corr_along_x(n, dx, cov=cov, length_km=L, nu=1.5)
    lags = np.array([4, 12, 24])                     # 1, 3, 6 km
    # removing the mean (S(0) = 0) lowers the correlation by a constant on a finite domain
    want = closed_form(lags * dx, L)
    assert np.allclose(corr[lags], want, atol=0.03), (corr[lags], want)


def test_grf_realisations_have_the_requested_correlation_at_lag():
    rng = np.random.default_rng(1)
    n, dx, L = 128, 0.5, 3.0
    acc = np.zeros(n)
    for _ in range(12):
        g = rf.grf(n, dx, rng, "exponential", length_km=L)
        F = np.fft.fft2(g)
        acc += np.fft.ifft2(np.abs(F) ** 2).real[0] / g.size
    corr = acc / acc[0]
    for lag in (2, 6, 12):                           # 1, 3, 6 km
        assert abs(corr[lag] - np.exp(-lag * dx / L)) < 0.08


def test_powerlaw_grf_has_the_requested_spectral_slope():
    rng = np.random.default_rng(2)
    n, beta = 128, 3.0
    k = rf._wavenumbers(n, 1.0, 1.0, 0.0)
    psd = np.zeros_like(k)
    for _ in range(8):
        psd += np.abs(np.fft.fft2(rf.grf(n, 1.0, rng, "powerlaw", beta=beta))) ** 2
    band = (k > 2 / n) & (k < 0.4)
    slope = np.polyfit(np.log(k[band]), np.log(psd[band]), 1)[0]
    assert abs(slope + beta) < 0.15


def test_grf_anisotropy_stretches_along_the_requested_angle():
    rng = np.random.default_rng(3)
    g = rf.grf(128, 0.5, rng, "gaussian", length_km=3.0, anisotropy=3.0, angle_deg=90.0)
    rx = np.corrcoef(g[:, :-4].ravel(), g[:, 4:].ravel())[0, 1]
    ry = np.corrcoef(g[:-4].ravel(), g[4:].ravel())[0, 1]
    assert ry > rx + 0.2                             # 90 deg: long along y (rows)


@pytest.mark.parametrize("dist", rf.WET_DISTRIBUTIONS)
def test_wet_distribution_has_the_requested_mean_and_cv(dist):
    d = rf.wet_distribution(dist, 3.0, 0.9)
    assert d.mean() == pytest.approx(3.0, rel=1e-6)
    assert d.std() / d.mean() == pytest.approx(0.9 if dist != "exponential" else 1.0, rel=1e-4)


@pytest.mark.parametrize("dist", ["gamma", "lognormal", "weibull"])
def test_to_rain_wet_fraction_and_marginal(dist):
    g = np.random.default_rng(4).standard_normal((200, 200))
    rain = rf.to_rain(g, war=0.3, dist=dist, mean_wet=4.0, cv_wet=0.8)
    wet = rain[rain > 0]
    assert abs((rain > 0).mean() - 0.3) < 0.003      # minus the few wet values under 0.1 mm/h
    assert wet.mean() == pytest.approx(4.0, rel=0.02)
    assert wet.std() / wet.mean() == pytest.approx(0.8, rel=0.05)
    # the wettest Gaussian values carry the heaviest rain: the transform is monotone
    assert stats.spearmanr(g[rain > 0], wet).correlation > 0.999


def test_to_rain_threshold_eats_into_war_for_a_very_skewed_marginal():
    """RISK: the 0.1 mm/h cut makes 'exact' war only approximately exact (here 4 % low)."""
    g = np.random.default_rng(5).standard_normal((200, 200))
    rain = rf.to_rain(g, war=0.5, dist="gamma", mean_wet=1.0, cv_wet=2.0)
    lost = 0.5 - (rain > 0).mean()
    assert lost == pytest.approx(0.5 * rf.wet_distribution("gamma", 1.0, 2.0).cdf(0.1), abs=0.005)
    assert lost > 0.1


def test_metagaussian_generator_hits_war_and_mean():
    m = gen.MetaGaussian(war=0.4, dist="gamma", mean_wet_mm_h=3.0, cv_wet=0.7, seed=1)
    rain = m.build(Grid(n=96, dx_km=0.5))
    assert abs((rain > 0).mean() - 0.4) < 0.003
    assert rain[rain > 0].mean() == pytest.approx(3.0, rel=0.02)


# --------------------------------------------------------------------------- other generators
def test_beta_lognormal_cascade_wet_fraction_and_mean():
    """Over & Gupta (1996): P(W > 0) = 2^-beta per level, E[W] = 1."""
    n, beta = 64, 0.15
    levels = 6
    wet, means = [], []
    for s in range(30):
        m = gen.BetaLognormalCascade(beta=beta, sigma2=0.25, mean_mm_h=1000.0, smooth_cells=0.0, seed=s)
        f = m.build(Grid(n=n, dx_km=1.0))
        wet.append((f > 0).mean())
        means.append(f.mean())
    assert np.mean(wet) == pytest.approx(2.0 ** (-beta * levels), abs=0.02)
    assert np.mean(means) == pytest.approx(1000.0, rel=0.15)


def test_cascade_1d_is_mean_preserving():
    rng = np.random.default_rng(6)
    means = [f1.cascade_1d(8, rng, beta=0.2, sigma2=0.3, mean=2.0).mean() for _ in range(400)]
    assert np.mean(means) == pytest.approx(2.0, rel=0.06)


def test_rainfarm_wet_fraction_and_mean():
    r = gen.RainFARM(war=0.6, mean_mm_h=2.0, seed=2).build(Grid(n=64, dx_km=1.0))
    assert abs((r > 0).mean() - 0.6) < 0.02
    assert r.mean() == pytest.approx(2.0, rel=0.03)  # minus the values cut under 0.1 mm/h


def test_downscale_to_reproduces_the_coarse_field():
    fine = np.random.default_rng(7).gamma(2.0, 5.0, (32, 32)) + 1.0
    coarse = np.arange(16, dtype=float).reshape(4, 4) + 1.0
    out = gen.downscale_to(fine, coarse)
    assert np.allclose(out.reshape(4, 8, 4, 8).mean(axis=(1, 3)), coarse)


def _moment_scaling(field, q):
    n = field.shape[0]
    pts = []
    for b in (1, 2, 4, 8, 16):
        c = field.reshape(n // b, b, n // b, b).mean(axis=(1, 3))
        pts.append((np.log(n / b), np.log(np.mean(c ** q))))
    x, y = np.array(pts).T
    return np.polyfit(x, y, 1)[0]


def test_universal_multifractal_moment_scaling_function():
    alpha, c1, q = 1.6, 0.1, 2.0
    ks = []
    for s in range(3):
        r = gen.UniversalMultifractal(alpha=alpha, C1=c1, H=0.0, war=1.0, seed=s).build(Grid(n=128, dx_km=1.0))
        ks.append(_moment_scaling(r, q))
    want = c1 / (alpha - 1) * (q ** alpha - q)
    assert np.mean(ks) == pytest.approx(want, rel=0.25)


def test_pulses_to_series_conserves_depth():
    starts, durs, xs = np.array([0.1, 1.33, 2.9]), np.array([0.5, 0.07, 0.6]), np.array([3.0, 10.0, 1.0])
    out = f1._pulses_to_series(starts, durs, xs, 4.0, 0.25)
    assert out.sum() * 0.25 == pytest.approx((durs * xs).sum())


# --------------------------------------------------------------------------- flows
def _blob(grid, x0, y0, s=1.5):
    xx, yy = grid.meshgrid()
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s ** 2))


def _centroid(f, grid):
    xx, yy = grid.meshgrid()
    return (f * xx).sum() / f.sum(), (f * yy).sum() / f.sum()


def test_spectral_shift_moves_a_blob_by_the_known_vector():
    g = Grid(n=64, dx_km=0.5)
    f = _blob(g, 10.0, 12.0)
    out = shift(f, g.dx_km, (6.0, -3.0), 1.5)        # +9 km east, -4.5 km north
    cx, cy = _centroid(np.clip(out, 0, None), g)
    assert (cx, cy) == pytest.approx((19.0, 7.5), abs=0.02)
    assert out.sum() == pytest.approx(f.sum(), rel=1e-9)


def test_uniform_advect_matches_shift_and_departure_points():
    g = Grid(n=32, dx_km=1.0)
    flow = fl.UniformFlow(mean=(4.0, 2.0)).prepare(g.size_km)
    x, y = fl.departure_points(flow, g, (0.0, 0.0), 0.5)
    xx, yy = g.meshgrid()
    assert np.allclose(x, xx - 2.0) and np.allclose(y, yy - 1.0)
    f = _blob(g, 10.0, 10.0)
    assert np.allclose(fl.advect(f, flow, g, (0.0, 0.0), 0.5), shift(f, 1.0, (4.0, 2.0), 0.5))


def test_rotation_flow_turns_a_blob_counter_clockwise_and_keeps_its_mass():
    g = Grid(n=96, dx_km=0.5)
    flow = fl.RotationFlow(omega_deg_h=90.0).prepare(g.size_km)     # centre (24, 24)
    f = _blob(g, 34.0, 24.0)
    for _ in range(12):                                             # 1 h in 5-min steps
        f = fl.advect(f, flow, g, (0.0, 0.0), 1.0 / 12)
    cx, cy = _centroid(np.clip(f, 0, None), g)
    assert (cx, cy) == pytest.approx((24.0, 34.0), abs=0.3)
    assert f.sum() == pytest.approx(_blob(g, 34.0, 24.0).sum(), rel=0.02)


@pytest.mark.parametrize("flow", [fl.RotationFlow(omega_deg_h=40.0), fl.ShearFlow(),
                                  fl.DeformationFlow(), fl.RandomFlow(rms_kmh=10.0, length_km=10.0)])
def test_flows_are_divergence_free(flow):
    g = Grid(n=64, dx_km=0.5)
    flow.prepare(g.size_km, g)
    u, v = flow.field(g)
    du_dx = np.gradient(u, g.dx_km, axis=1)
    dv_dy = np.gradient(v, g.dx_km, axis=0)
    div = (du_dx + dv_dy)[2:-2, 2:-2]
    scale = np.abs(np.gradient(u, g.dx_km, axis=0)).mean() + np.abs(du_dx).mean() + 1e-12
    assert np.abs(div).mean() < 0.05 * scale


def test_velocity_px_units():
    assert fl.velocity_px(np.array([12.0, -6.0]), 0.5, 5.0) == pytest.approx([2.0, -1.0])


# --------------------------------------------------------------------------- space-time
def test_frozen_uniform_sequence_is_an_exact_shift_and_fully_predictable():
    g = Grid(n=64, dx_km=0.5)
    seq = st.simulate(gen.MetaGaussian(seed=3), g, n_steps=4, dt_min=6.0,
                      flow=fl.UniformFlow(mean=(10.0, 5.0)), evolution="frozen")
    want = np.clip(shift(seq.frames[0].astype(float), g.dx_km, (10.0, 5.0), 0.3), 0, None)
    want[want < 0.1] = 0.0
    assert np.allclose(seq.frames[3], want, atol=1e-4)
    assert seq.predictability(2) > 0.97


def test_frozen_rotation_sequence_follows_the_flow():
    g = Grid(n=64, dx_km=0.5)
    seq = st.simulate(gen.MetaGaussian(length_km=4.0, war=0.6, seed=4), g, n_steps=4, dt_min=10.0,
                      flow=fl.RotationFlow(omega_deg_h=30.0), evolution="frozen", pad_frac=0.25)
    assert seq.predictability(3) > 0.9


def test_ar1_keeps_the_marginal_and_decorrelates_at_rho():
    g = Grid(n=64, dx_km=0.5)
    seq = st.simulate(gen.MetaGaussian(seed=5), g, n_steps=6, dt_min=10.0,
                      flow=fl.UniformFlow(mean=(0.0, 0.0)), evolution="ar1", tau_min=30.0)
    assert seq.extras["rho"] == pytest.approx(np.exp(-10.0 / 30.0))
    wet = [(f > 0).mean() for f in seq.frames]
    assert np.ptp(wet) < 0.01
    assert seq.predictability(1) > seq.predictability(4)


def test_octave_bands_partition_unity():
    w, scales = st._octave_bands(Grid(n=64, dx_km=0.5), 5)
    assert np.allclose(w.sum(0), 1.0)
    assert np.all(np.diff(scales) > 0)                 # finest first


# --------------------------------------------------------------------------- sensors: radar
def _clean_radar(**kw):
    base = dict(site_km=(0.0, 0.0), band="C", resolution_km=0.5, beam_width_deg=1e-3, dsd_sigma_db=0.0,
                noise_db=0.0, quantization_db=0.0, attenuation=False, mds_dbz_1km=-60.0,
                echo_top_km=50.0)
    base.update(kw)
    return sn.RadarConfig(**base)


def test_radar_zr_round_trip_without_impairments():
    g = Grid(n=32, dx_km=0.5)
    rain = np.full(g.shape, 5.0)
    r = sn.Radar(_clean_radar(), g).scan(rain, 5.0)
    assert np.allclose(r, 5.0, rtol=1e-9)


def test_radar_wrong_assumed_zr_gives_the_closed_form_bias():
    g = Grid(n=16, dx_km=0.5)
    rain = np.full(g.shape, 10.0)
    r = sn.Radar(_clean_radar(zr_true=(300.0, 1.4), zr_assumed=(200.0, 1.6)), g).scan(rain, 5.0)
    assert np.allclose(r, (300.0 * 10.0 ** 1.4 / 200.0) ** (1 / 1.6), rtol=1e-9)


def test_radar_two_way_pia_is_itu_times_twice_the_range():
    g = Grid(n=32, dx_km=0.5)
    radar = sn.Radar(_clean_radar(band="X"), g)
    rain = np.full(g.shape, 20.0)
    k, a = get_k_alpha(9.4, "horizontal")
    want = 2 * k * 20.0 ** a * radar.range_km
    assert np.allclose(radar.pia_db(rain, n_samples=256), want, rtol=0.02)


def test_radar_azimuth_is_clockwise_from_north():
    g = Grid(n=4, dx_km=1.0)
    radar = sn.Radar(_clean_radar(site_km=(2.0, 2.0)), g)
    # pixel (row 3, col 2) sits at (2.5, 3.5): north-north-east; (row 2, col 3) at (3.5, 2.5)
    assert radar.azimuth[3, 2] == pytest.approx(np.degrees(np.arctan2(0.5, 1.5)))
    assert radar.azimuth[2, 3] == pytest.approx(np.degrees(np.arctan2(1.5, 0.5)))


def test_radar_observe_scans_at_interval_end():
    g = Grid(n=8, dx_km=0.5)
    frames = np.stack([np.full(g.shape, float(t + 1)) for t in range(5)])   # 0, 5, ... 20 min
    out = sn.Radar(_clean_radar(), g).observe(frames, dt_min=5.0, product_min=10.0)
    assert out.shape[0] == 2
    assert np.allclose(out[:, 0, 0], [3.0 * 10 / 60, 5.0 * 10 / 60])


# --------------------------------------------------------------------------- sensors: gauges
def test_gauges_accumulate_interval_ending_depths():
    g = Grid(n=16, dx_km=1.0)
    gauges = sn.Gauges(sn.GaugeConfig(n=5, tip_mm=0.0, undercatch_mean=1.0, undercatch_sd=0.0), g)
    rates = np.array([0.0, 6.0, 12.0, 0.0, 3.0])                     # mm/h at t = 0, 5, ... 20
    frames = rates[:, None, None] * np.ones((5,) + g.shape)
    obs = gauges.observe(frames, dt_min=5.0, product_min=10.0)
    assert np.allclose(obs["rain_mm"], [[(6 + 12) * 5 / 60, (0 + 3) * 5 / 60]] * 5)
    assert np.allclose(obs["true_mm"], obs["rain_mm"])


# --------------------------------------------------------------------------- CML forward model
def _net(freqs, pols, length_km=5.0, n=1):
    freqs = np.asarray(freqs, dtype=float)
    k, a = zip(*[get_k_alpha(f, p) for f, p in zip(freqs, pols)])
    m = freqs.size
    y = np.linspace(4.0, 28.0, m)
    return cn.CMLNetwork(xa=np.full(m, 4.0), ya=y, xb=np.full(m, 4.0 + length_km), yb=y,
                         freq_ghz=freqs, pol=np.array(pols), k=np.array(k), alpha=np.array(a))


NO_IMPAIRMENT = cn.SensorConfig(waa_max_db=0.0, waa_max_db_assumed=0.0, baseline_sigma_db=0.0,
                                noise_sigma_db=0.0, quantization_db=0.0)


def test_forward_model_uniform_rain_is_itu_and_inverts_exactly():
    g = Grid(n=64, dx_km=0.5)
    net = _net([10.0, 23.0, 38.0], ["horizontal", "vertical", "horizontal"])
    res = cn.forward_model(np.full(g.shape, 12.0), g, net, NO_IMPAIRMENT)
    want = np.array([get_k_alpha(f, p)[0] * 12.0 ** get_k_alpha(f, p)[1] * 5.0
                     for f, p in zip(net.freq_ghz, net.pol)])
    assert np.allclose(res["A_rain"], want, rtol=1e-9)
    assert np.allclose(res["A_uniform"], want, rtol=1e-9)
    assert np.allclose(res["R_retrieved_clean"], 12.0, rtol=1e-9)
    assert np.allclose(res["R_retrieved"], 12.0, rtol=1e-9)


def test_path_averaging_bias_follows_jensen():
    g = Grid(n=64, dx_km=0.5)
    net = _net([10.0, 38.0], ["horizontal", "horizontal"], length_km=10.0)
    xx, _ = g.meshgrid()
    rain = np.where(xx < 9.0, 30.0, 1.0)                             # half the path wet
    res = cn.forward_model(rain, g, net, NO_IMPAIRMENT)
    rel = res["R_retrieved_clean"] / res["R_path_true"] - 1
    assert net.alpha[0] > 1 and rel[0] > 0.05
    assert net.alpha[1] < 1 and rel[1] < -0.02


def test_retrieval_with_the_true_wet_antenna_recovers_the_rain():
    g = Grid(n=64, dx_km=0.5)
    net = _net([23.0], ["vertical"], length_km=5.0)
    cfg = cn.SensorConfig(waa_max_db=2.0, waa_rate_per_mm_h=0.3, waa_max_db_assumed=2.0,
                          waa_rate_assumed=0.3, baseline_sigma_db=0.0, noise_sigma_db=0.0,
                          quantization_db=0.0)
    res = cn.forward_model(np.full(g.shape, 8.0), g, net, cfg)
    assert res["R_retrieved"][0] == pytest.approx(8.0, rel=1e-3)


def test_static_wet_antenna_is_pastorek_with_zeta_one():
    r = np.array([0.0, 0.5, 3.0, 30.0])
    assert np.allclose(cn.wet_antenna_db(r, 2.3, 0.28), wa.waa_pastorek_2021(r, 2.3, 0.28, 1.0))


# --------------------------------------------------------------------------- wet antenna
def test_waa_schleiss_matches_pycomlink():
    wet_antenna = pytest.importorskip("pycomlink.processing.wet_antenna")
    rng = np.random.default_rng(8)
    a_obs = np.clip(rng.normal(1.0, 1.0, 300), 0, None)
    wet = rng.random(300) < 0.6
    ours = wa.waa_schleiss_2013(wet, 2.3, 15.0, 1.0, a_obs=a_obs)
    theirs = wet_antenna.waa_schleiss_2013(rsl=a_obs, baseline=np.zeros(300), wet=wet,
                                           waa_max=2.3, delta_t=1.0, tau=15.0)
    assert np.allclose(ours, np.asarray(theirs))


def test_waa_schleiss_reaches_95_percent_after_tau():
    wet = np.ones(601, dtype=bool)
    w = wa.waa_schleiss_2013(wet, 2.0, tau_min=60.0, dt_min=0.1)
    assert w[600] / 2.0 == pytest.approx(1 - np.exp(-3), abs=0.005)


def test_waa_pastorek_matches_pycomlink():
    wet_antenna = pytest.importorskip("pycomlink.processing.wet_antenna")
    r = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    assert np.allclose(wa.waa_pastorek_2021(r), wet_antenna.waa_pastorek_2021(R=r))


def test_dynamic_wet_antenna_is_the_exact_relaxation():
    m = wa.DynamicWetAntenna()
    # wetting under constant rain: one long step == many short ones == closed form
    one = m.step(0.0, 5.0, 12.0)
    many = m.simulate(np.full(120, 5.0), 0.1)[-1]
    eq = m.equilibrium(5.0)
    assert one == pytest.approx(many, rel=1e-9)
    assert one == pytest.approx(eq * (1 - np.exp(-12.0 / m.tau_wet_min)), rel=1e-9)
    # drying after the rain: exp(-t / tau_dry)
    dry = m.simulate(np.zeros(30), 1.0, delta0=1.5)
    assert dry[-1] == pytest.approx(1.5 * np.exp(-30.0 / m.tau_dry_min), rel=1e-9)
    # the rate is the right-hand side the step integrates
    assert m.rate(0.5, 5.0) == pytest.approx((eq - 0.5) / m.tau_wet_min)
