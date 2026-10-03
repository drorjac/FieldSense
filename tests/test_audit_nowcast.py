"""Audit of core/nowcast against hand calculations and closed forms.

Advection: a known blob moved by a known vector; mass conservation of the
interpolated frames. Verification: CSI and FSS on hand-made cases, CRPS against
the closed form for a small ensemble (Hersbach 2000 = Gneiting and Raftery 2007
with the empirical CDF), SAL on cases whose S, A and L are known. Grid: square
pixels, unit conversion, dB transform. Learned motion: the rotation and flip
of the target vectors follow the arrays, and the shapes are (2, y, x).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.geo import Domain, haversine_m
from core.nowcast import advection, grid as G, scores, verify


def blob(shape=(40, 40), cy=20.0, cx=20.0, sigma=2.5, peak=10.0):
    yy, xx = np.indices(shape, dtype=float)
    return peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------
# advection
# --------------------------------------------------------------------------
def test_interpolate_pair_moves_blob_halfway_along_velocity():
    """V[0] is along x (columns), V[1] along y (rows): pysteps' convention."""
    u, v = 4.0, -2.0
    r1 = blob(cx=15, cy=20)
    r2 = blob(cx=15 + u, cy=20 + v)
    vel = np.stack([np.full(r1.shape, u), np.full(r1.shape, v)])
    (mid,) = advection.interpolate_pair(r1, r2, vel, n_sub=2)
    assert np.allclose(mid, blob(cx=15 + u / 2, cy=20 + v / 2), atol=1e-6)


def test_interpolate_pair_conserves_mass_at_fractional_shifts():
    """Bilinear sampling of an interior blob at a sub-pixel shift keeps its sum."""
    u, v = 3.0, 1.0
    r1 = blob(cx=12, cy=15)
    r2 = blob(cx=12 + u, cy=15 + v)
    vel = np.stack([np.full(r1.shape, u), np.full(r1.shape, v)])
    frames = advection.interpolate_pair(r1, r2, vel, n_sub=5)
    assert len(frames) == 4
    for f in frames:
        assert f.sum() == pytest.approx(r1.sum(), rel=1e-5)


def test_hourly_total_plain_mean_and_swath():
    """Without motion: the mean of the scans. With motion: the same mass, a smoother swath."""
    n = 13
    frames = np.stack([blob((40, 96), cx=10 + 6 * i, cy=20, sigma=1.5) for i in range(n)])
    plain = advection.hourly_total(frames)
    assert np.allclose(plain, frames.mean(axis=0))
    vel = np.stack([np.full(frames.shape[1:], 6.0), np.zeros(frames.shape[1:])])
    adv = advection.hourly_total(frames, vel, n_sub=6)
    assert adv.sum() == pytest.approx(plain.sum(), rel=1e-5)
    # along the swath the advected total is nearly uniform; the plain one beads
    row = adv[20, 20:72]
    assert row.std() / row.mean() < 0.02
    assert plain[20, 20:72].std() / plain[20, 20:72].mean() > 0.2


def test_hourly_total_linear_in_time_is_exact():
    """Equally spaced scans, both ends included: the mean is the exact time mean of a ramp."""
    t = np.linspace(0, 1, 13)
    rates = (2.0 + 6.0 * t)[:, None, None] * np.ones((1, 5, 5))
    vel = np.zeros((2, 5, 5))
    assert np.allclose(advection.hourly_total(rates), 5.0)
    assert np.allclose(advection.hourly_total(rates, vel), 5.0)


def test_pysteps_extrapolation_uses_the_same_vector_convention():
    from core.nowcast.methods import extrapolate
    u, v = 3.0, 2.0
    r = blob(cx=12, cy=12)
    vel = np.stack([np.full(r.shape, u), np.full(r.shape, v)])
    out = extrapolate(r, vel, 2)
    assert out.shape == (2,) + r.shape
    ok = np.isfinite(out[1])           # cells advected in from outside are NaN
    assert ok.mean() > 0.7
    assert np.allclose(out[1][ok], blob(cx=12 + 2 * u, cy=12 + 2 * v)[ok], atol=1e-6)


def test_reachable_masks_cells_advected_in_from_outside():
    from core.nowcast.methods import reachable
    vel = np.stack([np.full((10, 12), 2.0), np.zeros((10, 12))])
    m = reachable(vel, (10, 12), 3)
    assert m.shape == (3, 10, 12)
    for lead in range(3):
        # backward trajectory x - 2 (lead + 1) leaves the domain for the first columns
        n_out = 2 * (lead + 1)
        assert not m[lead, :, :n_out].any()
        assert m[lead, :, n_out:].all()


# --------------------------------------------------------------------------
# deterministic verification
# --------------------------------------------------------------------------
def test_categorical_scores_hand_case_and_nan_cells_excluded():
    f = np.array([[2.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, np.nan]])
    o = np.array([[2.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 2.0]])
    d = verify.Deterministic(1.0, gridded=False, thresholds=(1.0,))
    d.add(f, o)
    r = d.compute()
    # hits 2, false alarms 1, misses 1, correct negatives 3 (the NaN pair dropped)
    assert r["POD_1"] == pytest.approx(2 / 3)
    assert r["FAR_1"] == pytest.approx(1 / 3)
    assert r["CSI_1"] == pytest.approx(2 / 4)
    assert r["BIAS_1"] == pytest.approx(1.0)
    n = 7
    hr = (2 + 1) * (2 + 1) / n
    assert r["ETS_1"] == pytest.approx((2 - hr) / (2 + 1 + 1 - hr))
    ok = np.isfinite(f)
    assert r["MAE"] == pytest.approx(np.abs(f[ok] - o[ok]).mean())
    assert r["ME"] == pytest.approx((f[ok] - o[ok]).mean())


def test_categorical_scores_are_pooled_not_averaged():
    d = verify.Deterministic(1.0, gridded=False, thresholds=(1.0,))
    d.add(np.array([2.0, 2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0, 2.0]))   # CSI 1
    d.add(np.array([2.0, 0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0, 0.0]))   # CSI 0
    assert d.compute()["CSI_1"] == pytest.approx(4 / 6)


def _fss_by_hand(f, o, thr, n):
    """Roberts and Lean (2008): fractions in an n x n window, zero outside the domain."""
    bf, bo = (f >= thr).astype(float), (o >= thr).astype(float)
    pad = n // 2
    def frac(b):
        p = np.pad(b, pad)
        out = np.zeros_like(b)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                out[i, j] = p[i:i + n, j:j + n].mean()
        return out
    pf, po = frac(bf), frac(bo)
    return 1 - np.sum((pf - po) ** 2) / (np.sum(pf ** 2) + np.sum(po ** 2))


def test_fss_matches_hand_case():
    o = np.zeros((9, 9))
    f = np.zeros((9, 9))
    o[3:5, 3:5] = 2.0
    f[3:5, 4:6] = 2.0                 # shifted one pixel: 2 of 4 overlap
    d = verify.Deterministic(1.0, thresholds=(1.0,), fss_scales_km=(1, 3), fss_threshold=1.0)
    d.add(f, o)
    r = d.compute()
    assert r["FSS_1km"] == pytest.approx(1 - 4 / 8)          # point scale: 1 - mismatches/(nf+no)
    assert r["FSS_3km"] == pytest.approx(_fss_by_hand(f, o, 1.0, 3), rel=1e-6)
    assert r["FSS_3km"] > r["FSS_1km"]


def test_fss_window_from_scale_and_pixel_size():
    """A 10 km scale on 2 km pixels is a 5-pixel window."""
    d = verify.Deterministic(2.0, fss_scales_km=(10,))
    assert d.fss[10]["scale"] == 5


# --------------------------------------------------------------------------
# ensemble verification
# --------------------------------------------------------------------------
def crps_closed_form(members, y):
    """CRPS of the empirical CDF: E|X - y| - E|X - X'| / 2 (Gneiting and Raftery 2007)."""
    x = np.asarray(members, float)
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))


@pytest.mark.parametrize("y", [1.5, -1.0, 7.0])
def test_crps_matches_closed_form(y):
    members = np.array([0.0, 1.0, 2.0, 3.0])
    e = verify.Ensemble(4)
    e.add(members[:, None], np.array([y]))
    assert e.compute()["CRPS"] == pytest.approx(crps_closed_form(members, y), abs=1e-12)
    if y == 1.5:
        assert e.compute()["CRPS"] == pytest.approx(0.375)


@pytest.mark.xfail(strict=True, reason=(
    "pysteps 1.21 CRPS_accum (Hersbach 2000) uses strict inequalities only, so an "
    "observation equal to a member drops that bin's contribution: CRPS is biased low "
    "whenever obs and members tie, i.e. at every dry pixel with some dry members "
    "(1-5 % low on synthetic rain with 30-90 % dry cells). Fixing it in "
    "core.nowcast.verify would change the published CRPS of the nowcasting projects."))
@pytest.mark.parametrize("members, y", [([0.0, 1.0, 2.0, 3.0], 2.0), ([0.0, 0.0, 1.0, 2.0], 0.0)])
def test_crps_with_ties_matches_closed_form(members, y):
    members = np.asarray(members)
    e = verify.Ensemble(4)
    e.add(members[:, None], np.array([y]))
    assert e.compute()["CRPS"] == pytest.approx(crps_closed_form(members, y), abs=1e-12)


def test_crps_of_a_degenerate_ensemble_is_the_absolute_error():
    rng = np.random.default_rng(0)
    obs = rng.gamma(1.0, 2.0, 50)
    fc = rng.gamma(1.0, 2.0, 50)
    e = verify.Ensemble(3)
    e.add(np.stack([fc] * 3), obs)
    assert e.compute()["CRPS"] == pytest.approx(np.mean(np.abs(fc - obs)))


def test_ensemble_perfect_forecast_roc_and_mean():
    rng = np.random.default_rng(1)
    obs = rng.gamma(0.6, 2.0, (20, 20))
    e = verify.Ensemble(5)
    e.add(np.stack([obs] * 5), obs)
    r = e.compute()
    assert r["CRPS"] == pytest.approx(0.0, abs=1e-12)
    assert r["ROC_area"] == pytest.approx(1.0, abs=1e-6)
    assert r["ensmean_MAE"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# SAL (Wernli et al. 2008)
# --------------------------------------------------------------------------
def test_sal_identical_scaled_and_shifted():
    o = blob((60, 60), cy=30, cx=20, sigma=3)
    assert scores.sal_score(o, o) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    # doubling the rain: the scaled volume R_n / R_max is invariant, so S = 0; A = 1 / 1.5
    S, A, L = scores.sal_score(2 * o, o)
    assert S == pytest.approx(0.0, abs=1e-12)
    assert A == pytest.approx(2 / 3)
    assert L == pytest.approx(0.0, abs=1e-12)
    # a shift of 10 pixels: L1 = 10 / diagonal, L2 = 0 (single object)
    S, A, L = scores.sal_score(blob((60, 60), cy=30, cx=30, sigma=3), o)
    assert S == pytest.approx(0.0, abs=1e-6)
    assert A == pytest.approx(0.0, abs=1e-6)
    assert L == pytest.approx(10 / np.hypot(60, 60), rel=1e-6)


def test_sal_structure_sign_and_dry_fields():
    o = blob((60, 60), cy=30, cx=30, sigma=3)
    wide = blob((60, 60), cy=30, cx=30, sigma=8) * o.sum() / blob((60, 60), cy=30, cx=30, sigma=8).sum()
    S, A, _ = scores.sal_score(wide, o)
    assert S > 0.5                       # too wide / flat a forecast: S positive (Wernli)
    assert A == pytest.approx(0.0, abs=1e-6)
    assert scores.sal_score(np.zeros((60, 60)), o) is None


def test_sal_two_objects_location_second_term():
    """Two equal objects 20 px apart vs one in the middle: same centre of mass, L = L2 only."""
    two = blob((60, 60), cy=30, cx=20, sigma=2) + blob((60, 60), cy=30, cx=40, sigma=2)
    one = blob((60, 60), cy=30, cx=30, sigma=2) * 2
    _, _, L = scores.sal_score(two, one)
    d = np.hypot(60, 60)
    assert L == pytest.approx(2 * 10 / d, rel=1e-3)


# --------------------------------------------------------------------------
# grid and the bridge to pysteps
# --------------------------------------------------------------------------
def test_square_grid_pixels_are_square_and_pixel_km_correct():
    dom = Domain(57.2, 58.0, 11.4, 12.6, "t")
    g = G.square_grid(dom, 2.0)
    assert G.pixel_km(g) == pytest.approx(2.0, rel=1e-4)       # lat rounded to 1e-6 deg
    i, j = len(g.lat) // 2, len(g.lon) // 2
    dy = haversine_m(g.lat[i], g.lon[j], g.lat[i + 1], g.lon[j]) / 1000
    dx = haversine_m(g.lat[i], g.lon[j], g.lat[i], g.lon[j + 1]) / 1000
    assert dy == pytest.approx(2.0, rel=1e-4)
    assert dx == pytest.approx(2.0, rel=2e-3)
    assert np.all(np.diff(g.lat) > 0)            # row 0 south, as metadata's yorigin="lower" says
    m = G.metadata(g, 15)
    assert m["yorigin"] == "lower"
    assert m["x2"] - m["x1"] == pytest.approx(len(g.lon) * m["xpixelsize"])
    assert "R=6371008.8" in m["projection"]


def test_to_pysteps_converts_depth_per_step_to_rate():
    g = G.square_grid(Domain(44.0, 44.2, 10.0, 10.3, "t"), 2.0)
    times = pd.date_range("2024-01-01", periods=3, freq="15min")
    data = np.full((3,) + g.shape, 0.5)
    data[1, 0, 0] = np.nan
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": g.lat, "lon": g.lon})
    rate, meta = G.to_pysteps(da.transpose("lon", "time", "lat"), 15)
    assert rate.shape == (3,) + g.shape
    assert np.allclose(rate[0], 2.0)               # 0.5 mm per 15 min = 2 mm/h
    assert rate[1, 0, 0] == 0.0
    assert meta["accutime"] == 15.0 and meta["unit"] == "mm/h"
    assert np.isnan(G.to_pysteps(da, 15, fill=None)[0][1, 0, 0])


def test_pixel_km_of_a_descending_lat_grid_is_positive():
    """A north-up (descending lat) grid must not give a negative pixel size to pysteps."""
    from core.geo import Grid
    g = G.square_grid(Domain(44.0, 44.2, 10.0, 10.3, "t"), 2.0)
    flipped = Grid(g.lat[::-1], g.lon)
    assert G.pixel_km(flipped) == pytest.approx(G.pixel_km(g), rel=1e-4)
    assert G.metadata(flipped, 15)["xpixelsize"] > 0
    assert G.metadata(flipped, 15)["yorigin"] == "upper"


def test_dbr_round_trip():
    r = np.array([[0.0, 0.05, 0.1, 1.0], [5.0, 20.0, 100.0, 0.3]])
    meta = G.metadata(G.square_grid(Domain(44.0, 44.1, 10.0, 10.1, "t"), 2.0), 5)
    dbr, mdb = G.to_dbr(r, meta)
    assert dbr[0, 0] == G.DB_ZEROVALUE and dbr[0, 1] == G.DB_ZEROVALUE
    assert dbr[0, 3] == pytest.approx(0.0) and dbr[1, 0] == pytest.approx(10 * np.log10(5))
    back = G.from_dbr(dbr, mdb)
    keep = r >= 0.1
    assert np.allclose(back[keep], r[keep])
    assert np.all(back[~keep] == 0.0)


# --------------------------------------------------------------------------
# learned motion: vector conventions under augmentation, shapes
# --------------------------------------------------------------------------
def _shift_of(a, b):
    """Integer (dx, dy) displacement from the peak of ``a`` to the peak of ``b``."""
    ia = np.unravel_index(np.argmax(a), a.shape)
    ib = np.unravel_index(np.argmax(b), b.shape)
    return ib[1] - ia[1], ib[0] - ia[0]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_learned_motion_augmentation_rotates_vectors_with_arrays(k, flip):
    torch = pytest.importorskip("torch")
    from core.nowcast import learned_motion as lm
    u, v = 3, -2
    a = blob((32, 32), cy=14, cx=11, sigma=1.5)
    b = blob((32, 32), cy=14 + v, cx=11 + u, sigma=1.5)
    x = torch.from_numpy(np.stack([a, b])[None].astype(np.float32))
    y = torch.from_numpy(np.stack([np.full(a.shape, u), np.full(a.shape, v)])[None].astype(np.float32))
    xa, ya = lm._augment(x, y, k, flip)
    want = _shift_of(xa[0, 0].numpy(), xa[0, 1].numpy())
    assert (float(ya[0, 0, 5, 5]), float(ya[0, 1, 5, 5])) == pytest.approx(want)


def test_learned_motion_shapes_and_transform():
    pytest.importorskip("torch")
    from core.nowcast import learned_motion as lm
    assert lm.transform(np.array(0.0)) == pytest.approx(0.0)
    assert lm.transform(np.array(-1.0)) == pytest.approx(0.0)      # negative rates clipped
    assert np.all(np.diff(lm.transform(np.array([0.0, 0.1, 1.0, 10.0]))) > 0)
    net = lm._net()
    net.eval()
    rate = np.random.default_rng(0).gamma(0.5, 1.0, (6, 23, 30))
    v = lm.motion(rate, net)
    assert v.shape == (2, 23, 30)
    from core.nowcast.methods import N_PAST
    assert lm.N_PAST == N_PAST["LK"]


def test_simulated_velocity_px_matches_pysteps_advection():
    """The training target (``FieldSequence.velocity_px``) is in pysteps' convention."""
    from core.nowcast.methods import extrapolate
    from core.simulation import flows as fl, generators as gen, spacetime as st
    from core.simulation.rain_fields import Grid
    seq = st.simulate(gen.make("convective_cells", seed=3), Grid(n=48, dx_km=1.0), 3, 5.0,
                      flow=fl.make_flow("uniform", mean=(24.0, -12.0)), evolution="frozen", seed=3)
    vel = seq.velocity_px()
    assert vel.shape[0] == 2
    assert np.allclose(vel[:, 0, 0], [2.0, -1.0])
    pred = extrapolate(seq.frames[0], vel, 1)[0]
    ok = np.isfinite(pred)
    obs = seq.frames[1]
    swapped = extrapolate(seq.frames[0], vel[::-1], 1)[0]
    ok &= np.isfinite(obs) & np.isfinite(swapped)
    assert ok.mean() > 0.5
    err = np.abs(pred[ok] - obs[ok]).mean()
    assert err < 0.25 * np.abs(swapped[ok] - obs[ok]).mean()
