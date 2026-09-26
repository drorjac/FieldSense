"""Alignment, metrics and maps of the nowcasting pipeline, on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

import evaluation as ev  # noqa: E402
import forecasting as fc  # noqa: E402
import maps  # noqa: E402


def _moving(n=40, shape=(4, 5)):
    """Frame t is filled with the value t: easy to see which frame is which."""
    return np.arange(n, dtype=np.float32)[:, None, None] * np.ones((n, *shape), np.float32)


def test_windows_target_is_h_steps_after_the_last_input():
    x, y = fc.windows(_moving(), lookback=8, h=3)
    assert x[0, -1, 0, 0] == 7 and y[0, 0, 0] == 7 + 3
    xm, ym = fc.multi_windows(_moving(), lookback=8, horizons=(1, 2, 3, 4))
    assert ym[0, :, 0, 0].tolist() == [8, 9, 10, 11]
    assert len(xm) == len(_moving()) - 8 - 4 + 1


def test_every_forecast_is_scored_against_the_frame_it_forecast():
    """A perfect oracle scores RMSE 0 through ``evaluation.truth_frames``."""
    test, lb = _moving(), 8
    for h in (1, 2, 3, 4):
        n = len(test) - lb - 4 + 1
        oracle = np.stack([test[i + lb - 1 + h] for i in range(n)])
        assert np.array_equal(ev.truth_frames(test, lb, h, n), oracle)


def test_persistence_and_the_one_step_leak():
    p = fc.persistence(_moving(), lookback=8, horizons=(1, 2))
    assert p[1][0, 0, 0] == 7 and p[2][0, 0, 0] == 7              # last observed
    late = fc.persistence(_moving(), lookback=8, horizons=(1,), one_step_late=True)
    assert late[1][0, 0, 0] == 8                                   # the h=1 truth itself


def test_pysteps_aligned_by_default_and_leaky_when_faithful():
    seen = []

    def spy(frames, n_steps):
        seen.append(frames[-1, 0, 0])
        return np.repeat(frames[-1:], n_steps, axis=0)

    fc.pysteps_forecasts(_moving(), lookback=8, horizons=(1, 2), extrapolate=spy)
    assert seen[0] == 7                                   # same last frame as every model
    seen.clear()
    fc.pysteps_forecasts(_moving(), lookback=8, horizons=(1, 2), faithful=True, extrapolate=spy)
    assert seen[0] == 8                                   # the notebook: one frame late


def test_models_output_the_right_shapes():
    x = torch.zeros(3, 8, 4, 5)
    assert fc.RainTransformer((4, 5), d_model=16, n_heads=2, n_layers=1)(x).shape == (3, 4, 5)
    assert fc.GRUForecaster((4, 5), hidden_dim=8, n_horizons=4)(x).shape == (3, 4, 4, 5)
    assert fc.SimpleMamba is fc.GRUForecaster


def test_metrics_perfect_and_hss_on_a_known_table():
    t = np.array([0, 0, 1, 1, 0, 2.0])
    m = ev.metrics(t, t)
    assert m["RMSE"] == 0 and m["HSS"] == pytest.approx(1.0) and m["CC"] == pytest.approx(1.0)
    p = np.array([0, 1, 1, 0, 0, 2.0])                    # tp=2 tn=2 fp=1 fn=1
    assert ev.metrics(t, p)["HSS"] == pytest.approx(2 * (4 - 1) / (3 * 3 + 3 * 3))


def test_idw_in_km_corrects_the_degree_anisotropy():
    """Two sources, equally far in km (east vs north): km weights them equally."""
    lat0 = 57.7
    dlat = 1.0 / 111.32                                   # 1 km north
    dlon = dlat / np.cos(np.deg2rad(lat0))                # 1 km east
    src_lon, src_lat = np.array([11.9 + dlon, 11.9]), np.array([lat0, lat0 + dlat])
    lon, lat = np.array([[11.9]]), np.array([[lat0]])
    w_km = maps.idw_weights(src_lon, src_lat, lon, lat, metric="km")[0]
    w_deg = maps.idw_weights(src_lon, src_lat, lon, lat, metric="degrees")[0]
    assert w_km == pytest.approx([0.5, 0.5], abs=1e-3)
    assert w_deg[1] > 0.7                                 # degrees favour the northern one


def test_moving_benchmark_brackets_models_between_persistence_and_oracle():
    from synthetic import moving_benchmark

    table, _ = moving_benchmark(n=16, n_frames=(60, 20, 30), lookback=4, horizons=(1, 2),
                                  epochs=1, evolve_tau_min=None)
    cc = table.pivot(index="model", columns="horizon_min", values="CC")
    assert cc.loc["oracle (true motion)"].min() > 0.99          # frozen: motion is everything
    assert (cc.loc["oracle (true motion)"] >= cc.loc["persistence"]).all()
    assert set(cc.index) >= {"transformer (multi)", "gru (multi)", "pod-sindy"}
