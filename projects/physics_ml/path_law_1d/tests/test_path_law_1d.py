"""The path-law starter helpers, on synthetic inputs with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.itu_p838 import get_k_alpha  # noqa: E402
from core.simulation import wet_antenna as wa  # noqa: E402
from core.simulation.rain_fields import Grid  # noqa: E402
from path_law_1d import events as pe  # noqa: E402
from path_law_1d import simulate as sm  # noqa: E402
from path_law_1d import tails as tl  # noqa: E402

GRID = Grid(n=64, dx_km=0.5)


# --------------------------------------------------------------------------- simulate
def test_random_links_lie_inside_the_domain_with_itu_coefficients():
    net = sm.random_links(GRID, 200, length_km=(0.5, 12.0), seed=1)
    L = net.length_km
    assert L.min() >= 0.5 - 1e-9 and L.max() <= 12.0 + 1e-9
    for v in (net.xa, net.xb, net.ya, net.yb):
        assert v.min() > 0 and v.max() < GRID.size_km
    k, a = get_k_alpha(float(net.freq_ghz[0]), str(net.pol[0]))
    assert (net.k[0], net.alpha[0]) == (k, a)


def test_path_law_table_obeys_jensen():
    net = sm.random_links(GRID, 60, length_km=(1.0, 10.0), seed=2)
    t = sm.path_law_table(net, GRID, regimes=("convective_cells",), n_fields=1)
    assert len(t) > 5 and (t.R_bar >= 0.1).all()
    hi, lo = t[t.b > 1.001], t[t.b < 0.999]
    assert (hi.A_exact >= hi.A_linear * (1 - 1e-12)).all()
    assert (lo.A_exact <= lo.A_linear * (1 + 1e-12)).all()
    # R_linear is the linear law inverted on the exact attenuation
    np.testing.assert_allclose(t.a * t.R_linear**t.b * t.L_km, t.A_exact, rtol=1e-9)


def test_gap_and_score_of_the_exact_law_are_zero():
    t = pd.DataFrame({"regime": "x", "L_km": [0.5, 3.0, 10.0], "R_bar": [1.0, 2.0, 4.0],
                      "R_linear": [1.0, 2.2, 3.0], "A_exact": [1.0, 2.0, 3.0],
                      "A_linear": [1.0, 2.0, 3.3]})
    g = sm.gap_summary(t)
    assert g["median"].tolist() == pytest.approx([0.0, 0.1, -0.25])
    s = sm.score_path_law(t, t.A_exact)
    assert (s.rmse_db == 0).all() and (s.within_5pct == 1).all()
    assert sm.score_path_law(t, t.A_linear).loc["8-12 km", "within_5pct"] == 0


# --------------------------------------------------------------------------- events
def test_radar_events_split_and_mask():
    t = pd.date_range("2020-01-01", periods=72 * 12, freq="5min")
    rate = np.zeros((t.size, 4, 4))
    rate[(t >= "2020-01-01 06:00") & (t < "2020-01-01 09:00")] = 3.0
    rate[(t >= "2020-01-02 12:00") & (t < "2020-01-02 13:00")] = 5.0
    rate[(t >= "2020-01-03 03:00") & (t < "2020-01-03 05:00")] = 2.0
    da = xr.DataArray(rate, dims=("time", "y", "x"),
                      coords={"time": t, "y": np.arange(4.0), "x": np.arange(4.0)})
    ev = pe.split_events(pe.radar_events(da))
    assert len(ev) == 3
    assert ev.start.iloc[0] == pd.Timestamp("2020-01-01 06:00")
    assert ev.end.iloc[0] == pd.Timestamp("2020-01-01 09:00")
    assert ev.split.tolist() == ["train", "test", "train"]
    m = pe.event_mask(t, ev, "test", pad_before="0h", pad_after="2h")
    assert m.sum() == 3 * 12          # one hour of rain plus two of tail


# --------------------------------------------------------------------------- scoring
def test_normalized_scores():
    pytest.importorskip("poligrain")
    from path_law_1d import scoring as ps
    ref = np.r_[np.zeros(20), np.linspace(0.5, 5, 40)]
    s = ps.normalized_scores(ref, ref)
    assert s["nbias"] == pytest.approx(0) and s["nrmse"] == pytest.approx(0)
    assert ps.normalized_scores(ref, 2 * ref)["nbias"] == pytest.approx(1.0)
    t = pd.date_range("2020-01-01", periods=60, freq="15min")
    est = xr.DataArray(np.c_[ref, 2 * ref], dims=("time", "cml_id"),
                       coords={"time": t, "cml_id": [1, 2]})
    ref_da = xr.DataArray(np.c_[ref, ref], dims=("time", "cml_id"),
                          coords={"time": t, "cml_id": [1, 2]})
    table = ps.scores_by_length(est, {"radar": ref_da}, [0.5, 5.0])
    assert table.loc["0-1 km", ("radar", "nbias")] == pytest.approx(0)
    assert table.loc["4-8 km", ("radar", "nbias")] == pytest.approx(1.0)
    assert table.loc["all", ("links", "n")] == 2


# --------------------------------------------------------------------------- tails
MODEL = wa.DynamicWetAntenna(a_max_db=1.5, gamma_per_mm_h=0.4, tau_wet_min=4.0, tau_dry_min=25.0)


def rain_series(n=1440, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for start in rng.choice(np.arange(0, n - 200, 240), 5, replace=False):
        r[start:start + rng.integers(20, 60)] = rng.gamma(2.0, 3.0)
    return r


def test_minutes_since_rain():
    r = np.array([0, 1, 0, 0, 2, 0], float)
    np.testing.assert_array_equal(tl.minutes_since_rain(r, 2.0), [np.inf, 0, 2, 4, 0, 2])


def test_tail_fit_recovers_the_drying_time():
    rain = np.stack([rain_series(seed=s) for s in range(3)])
    delta = MODEL.simulate(rain, 1.0)
    tails = tl.tail_table(delta, rain, 1.0)
    assert len(tails) >= 10
    assert tails.tau_min.median() == pytest.approx(25.0, rel=0.02)
    assert (tails.delta0_db > 0).all()


def test_literature_fit_recovers_its_own_model_and_scores():
    rain = np.stack([rain_series(seed=s) for s in range(4)])
    truth = wa.waa_pastorek_2021(rain, 1.2, 0.2, 0.7)
    params, preds = tl.fit_literature(truth, rain, 1.0)
    assert params.pastorek_a_max_db.to_numpy() == pytest.approx(1.2, rel=1e-6)
    assert (params.pastorek_d == 0.2).all() and (params.pastorek_zeta == 0.7).all()
    scores = tl.waa_scores(truth, preds, rain, 1.0)
    assert scores.loc["pastorek", "rmse_rain_db"] == pytest.approx(0, abs=1e-9)
    assert scores.loc["constant", "rmse_rain_db"] > 0
    # a memoryless model predicts nothing in the tail of a film with memory
    dyn = MODEL.simulate(rain, 1.0)
    _, p = tl.fit_literature(dyn, rain, 1.0)
    s = tl.waa_scores(dyn, {**p, "truth": dyn}, rain, 1.0)
    assert s.loc["truth"].filter(like="rmse").max() == 0
    assert (s.drop("truth").rmse_tail_db > 0.1).all()
