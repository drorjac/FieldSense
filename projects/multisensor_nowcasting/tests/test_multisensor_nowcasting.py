"""Synthetic checks: additive scores, CRPS, link-rain disaggregation, mapping, the U-Net."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multisensor_nowcasting import scoring  # noqa: E402
from multisensor_nowcasting.cube import _interp, cell_index  # noqa: E402
from core.geo import Grid  # noqa: E402


def _fields(seed=0, n=12, shape=(20, 24)):
    rng = np.random.default_rng(seed)
    o = rng.gamma(0.6, 2.0, (n,) + shape)
    f = o * rng.uniform(0.5, 1.5, o.shape)
    return f, o


def _masks(n=12, shape=(20, 24)):
    m = np.ones((n,) + shape, bool)
    a = m.copy()
    a[:, :, :12] = False
    return {"domain": m, "area": a}


def test_stats_are_additive_and_match_direct_scores():
    f1, o1 = _fields(1)
    f2, o2 = _fields(2)
    m = _masks()
    S = np.stack([scoring.stats(f1, o1, m), scoring.stats(f2, o2, m)])
    pooled = scoring.pooled(S)
    f, o = np.concatenate([f1[0].ravel(), f2[0].ravel()]), np.concatenate([o1[0].ravel(), o2[0].ravel()])
    assert pooled["MAE"][0, 0] == pytest.approx(np.mean(np.abs(f - o)))
    assert pooled["corr"][0, 0] == pytest.approx(np.corrcoef(f, o)[0, 1])
    h = np.sum((f >= 1) & (o >= 1))
    assert pooled["CSI_1"][0, 0] == pytest.approx(h / (h + np.sum((f < 1) & (o >= 1)) + np.sum((f >= 1) & (o < 1))))


def test_perfect_forecast_scores():
    _, o = _fields(3)
    sc = scoring.scores_from(scoring.stats(o, o, _masks()))
    assert np.allclose(sc["CSI_1"], 1) and np.allclose(sc["MAE"], 0) and np.allclose(sc["FSS_10km"], 1)


def test_bootstrap_interval_contains_estimate_and_paired_zero_for_identical():
    S = np.stack([scoring.stats(*_fields(s), _masks()) for s in range(8)])
    days = np.repeat(np.arange(4), 2)
    est, lo, hi = scoring.bootstrap(S, days, "MAE", n_boot=200)
    assert np.all(lo <= est + 1e-12) and np.all(est <= hi + 1e-12)
    d, dlo, dhi = scoring.bootstrap(S, days, "MAE", S_ref=S, n_boot=50)
    assert np.allclose(d, 0) and np.allclose(dlo, 0) and np.allclose(dhi, 0)


def test_crps_members_matches_brute_force():
    rng = np.random.default_rng(0)
    m = rng.gamma(1.0, 2.0, (7, 30))
    o = rng.gamma(1.0, 2.0, 30)
    brute = np.abs(m - o).mean(0) - 0.5 * np.abs(m[:, None] - m[None]).mean((0, 1))
    assert np.allclose(scoring.crps_members(m, o), brute)


def test_interp_skips_missing_sources():
    W = np.array([[1.0, 1.0], [0.0, 1.0]], dtype="float32")
    V = np.array([[2.0, np.nan], [4.0, 4.0]], dtype="float32")
    out = _interp(W, V)
    assert np.allclose(out[:, 0], [3.0, 4.0]) and np.allclose(out[:, 1], [4.0, 4.0])


def test_cell_index_inside_and_outside():
    g = Grid(np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0]))
    assert list(cell_index(g, [0.0, 2.0, 5.0], [10.0, 11.0, 10.0])) == [0, 5, -1]


def test_link_disaggregation_keeps_the_rnn_hourly_total():
    from multisensor_nowcasting import links
    # one link, two hours: a rain burst in minutes 10-19 of the second hour
    X = np.zeros((1, 2, 64), "float32")
    X[0, 1, 10:20] = 5.0
    M = np.array([[23.0, 2.0, 1.0, np.log10(0.12), 1.0]], "float32")
    ds = xr.Dataset({"X": (("link", "time", "feature"), X), "M": (("link", "meta"), M)})
    pat = links.pattern_5min(ds)
    assert pat[0, 0].sum() == 0 and pat[0, 1, 2:4].min() > 0 and pat[0, 1, 4:].sum() == 0
    H = np.array([[0.0, 3.0]])
    pm = pat.mean(-1, keepdims=True)
    scaled = np.where(pm > 0.01, pat * (H[..., None] / pm), H[..., None])
    assert scaled[0, 1].mean() == pytest.approx(3.0)


def test_split_rule_cycles_weeks():
    from multisensor_nowcasting import links
    t = pd.date_range("2015-06-01", periods=24 * 7 * 4, freq="1h") + pd.Timedelta("1h")
    s = links.split(t, "none")
    assert set(np.unique(s)) == {0, 1, 2}


def test_unet_shapes():
    torch = pytest.importorskip("torch")
    from multisensor_nowcasting import learn
    m = learn.build_model(10)
    x = np.zeros((2, 10, 44, 35), "float32")
    out = learn._forward(m, x, None, learn._pad((44, 35)))
    assert tuple(out.shape) == (2, 12, 44, 35)
    _ = torch
