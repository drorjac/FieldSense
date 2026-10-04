"""core.nowcast grid, methods and verification on a synthetic moving rain cell."""

import numpy as np
import pytest

pytest.importorskip("pysteps")

from core.geo import Domain  # noqa: E402
from core.nowcast import grid as G, methods as M, verify as V  # noqa: E402

U, V_ = 2.0, 1.0


def cell(n=8, ny=48, nx=64, peak=20.0, sigma=5.0):
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    return np.array([peak * np.exp(-((x - 15 - U * t) ** 2 + (y - 18 - V_ * t) ** 2) / (2 * sigma ** 2))
                     for t in range(n)])


def meta(shape=(48, 64)):
    g = G.square_grid(Domain(44.0, 44.0 + shape[0] * 0.018, 10.0, 10.0 + shape[1] * 0.0251, "t"), 2.0)
    return G.metadata(g, 5, product="test")


def test_dbr_roundtrip():
    r = np.array([[0.0, 0.05, 1.0, 10.0]])
    dbr, m = G.to_dbr(r, meta())
    back = G.from_dbr(dbr, m)
    assert np.allclose(back[0, 2:], r[0, 2:], rtol=1e-6) and back[0, 0] == 0


def test_lk_and_extrapolation_beat_persistence():
    f = cell(12)
    past, future = f[:8], f[8:]
    v = M.motion(past, meta(), "LK")
    core = past[-1] > 5
    assert np.median(v[0][core]) == pytest.approx(U, abs=0.6)
    ext = M.deterministic("extrapolation", past, meta(), v, 4)
    per = M.deterministic("persistence", past, meta(), v, 4)
    assert np.nanmean(np.abs(ext - future)) < np.nanmean(np.abs(per - future))


def test_reachable_mask_loses_the_upwind_edge():
    v = np.zeros((2, 48, 64))
    v[0] = 3.0
    reach = M.reachable(v, (48, 64), 4)
    assert not reach[-1][:, :10].any() and reach[-1][:, 20:].all()


def test_pooled_scores_perfect_forecast():
    f = cell(1)[0]
    d = V.Deterministic(2.0)
    d.add(f, f)
    out = d.compute()
    assert out["CSI_1"] == pytest.approx(1.0) and out["MAE"] == pytest.approx(0.0)
    assert out["FSS_10km"] == pytest.approx(1.0)
    e = V.Ensemble(3)
    e.add(np.stack([f, f, f]), f)
    assert e.compute()["CRPS"] == pytest.approx(0.0, abs=1e-9)
