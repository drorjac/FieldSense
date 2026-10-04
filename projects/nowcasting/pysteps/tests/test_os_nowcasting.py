"""Synthetic checks of the nowcasting chain: a Gaussian rain cell moving at a known speed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("pysteps")

from core.geo import Domain, to_local_xy  # noqa: E402
from os_nowcasting import grid as G, interp, nowcast, verify  # noqa: E402

U, V = 2.0, 1.0                     # pixels per step (x, y)


def moving_cell(n=10, ny=64, nx=80, peak=20.0, sigma=6.0):
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    frames = [peak * np.exp(-((x - 20 - U * t) ** 2 + (y - 25 - V * t) ** 2) / (2 * sigma ** 2))
              for t in range(n)]
    return np.array(frames)


def meta(shape=(64, 80), step=15):
    g = G.square_grid(Domain(44.0, 44.0 + shape[0] * 0.018, 10.0, 10.0 + shape[1] * 0.0251, "t"), 2.0)
    return G.metadata(g, step)


def test_square_grid_has_square_pixels():
    g = G.square_grid(Domain(57.2, 58.0, 11.4, 12.6, "t"), 2.0)
    lat0, lon0 = g.lat.mean(), g.lon.mean()
    x, y = to_local_xy(np.array([g.lat[10], g.lat[11], g.lat[10]]),
                       np.array([g.lon[10], g.lon[10], g.lon[11]]), lat0, lon0)
    assert abs((y[1] - y[0]) - 2000) < 5
    assert abs((x[2] - x[0]) - 2000) < 60          # exact at the centre latitude


def test_to_pysteps_converts_amounts_to_rates():
    g = G.square_grid(Domain(44.0, 44.2, 10.0, 10.3, "t"), 2.0)
    da = xr.DataArray(np.full((3,) + g.shape, 0.5), dims=("time", "lat", "lon"),
                      coords={"time": pd.date_range("2022-08-18", periods=3, freq="15min"),
                              "lat": g.lat, "lon": g.lon})
    rate, m = G.to_pysteps(da, 15)
    assert np.allclose(rate, 2.0)
    assert m["xpixelsize"] == pytest.approx(2000, rel=1e-3) and m["yorigin"] == "lower"


@pytest.mark.parametrize("method", ["LK", "VET", "proesmans"])
def test_motion_recovers_the_cell_speed(method):
    frames = moving_cell()
    v = nowcast.motion(frames, meta(), method)
    core = frames[-1] > 5
    assert np.median(v[0][core]) == pytest.approx(U, abs=0.6)
    assert np.median(v[1][core]) == pytest.approx(V, abs=0.6)


def test_extrapolation_beats_persistence():
    frames = moving_cell(n=14)
    past, future = frames[:8], frames[8:]
    m = meta()
    v = nowcast.motion(past, m, "LK")
    ext = nowcast.deterministic("extrapolation", past, m, v, 6)
    per = nowcast.deterministic("persistence", past, m, v, 6)
    mae = lambda f: np.abs(f - future).mean()                     # noqa: E731
    assert mae(ext) < 0.5 * mae(per)


@pytest.mark.parametrize("method", ["sprog", "anvil"])
def test_cascade_methods_run_and_keep_rain(method):
    # a textured cell: on a perfectly smooth one ANVIL's Lagrangian differences vanish and it
    # returns no rain
    frames = moving_cell(n=8) * (1 + 0.3 * np.random.default_rng(0).random((8, 64, 80)))
    m = meta()
    f = nowcast.deterministic(method, frames, m, nowcast.motion(frames, m, "LK"), 4)
    assert f.shape == (4,) + frames.shape[1:] and np.isfinite(f).all() and f.max() > 1


def test_steps_ensemble_shape():
    frames = moving_cell(n=8)
    m = meta()
    e = nowcast.ensemble("steps", frames, m, nowcast.motion(frames, m, "LK"), 3, n_members=4)
    assert e.shape == (4, 3) + frames.shape[1:] and np.isfinite(e).all()


def test_perfect_forecast_scores():
    obs = moving_cell(n=1)[0]
    d = verify.Deterministic(2.0)
    d.add(obs, obs, sal=True)
    r = d.compute()
    assert r["CSI_1"] == pytest.approx(1) and r["MAE"] == pytest.approx(0)
    assert r["FSS_10km"] == pytest.approx(1)
    assert abs(r["SAL_S"]) < 1e-9 and abs(r["SAL_A"]) < 1e-9 and abs(r["SAL_L"]) < 1e-9


def test_sal_signs():
    obs = moving_cell(n=1)[0]
    s, a, l = verify.sal_score(2 * obs, obs)
    assert a > 0 and abs(s) < 0.05 and l < 0.01
    s, a, l = verify.sal_score(np.roll(obs, 20, axis=1), obs)
    assert l > 0.05


def test_ensemble_scores():
    obs = moving_cell(n=1)[0]
    e = verify.Ensemble(5)
    e.add(np.repeat(obs[None], 5, axis=0), obs)
    r = e.compute()
    assert r["CRPS"] == pytest.approx(0, abs=1e-9) and r["ROC_area"] > 0.95


def test_nan_cells_are_ignored():
    obs = moving_cell(n=1)[0]
    f = obs.copy()
    f[:10] = 99.0
    obs_nan = obs.copy()
    obs_nan[:10] = np.nan
    d = verify.Deterministic(2.0)
    d.add(f, obs_nan)
    assert d.compute()["MAE"] == pytest.approx(0)


def test_advection_interpolation_smooths_a_moving_cell():
    from scipy.ndimage import map_coordinates
    frames = moving_cell(n=13, sigma=1.0)              # a small fast cell leaves beads
    m = meta(step=5)
    plain = interp.hourly_total(frames, m, interpolate=False)
    adv = interp.hourly_total(frames, m, interpolate=True)
    assert adv.sum() == pytest.approx(plain.sum(), rel=0.05)        # mass kept
    t = np.linspace(2, 10, 200)                                      # along the track
    ridge = lambda f: map_coordinates(f, (25 + V * t, 20 + U * t), order=1)   # noqa: E731
    cv = lambda r: r.std() / r.mean()                                # noqa: E731
    assert cv(ridge(adv)) < 0.7 * cv(ridge(plain))                   # beads become a swath
    still = np.repeat(frames[:1], 13, axis=0)
    assert np.allclose(interp.hourly_total(still, m), frames[0], atol=1e-6)
