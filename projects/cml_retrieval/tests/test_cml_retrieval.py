"""The RNN training helpers and PyNNcml workarounds, on tiny synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

import rnn_retrieval as rr  # noqa: E402


def test_regression_loss_weights_wet_samples_up():
    loss = rr.RegressionLoss(gamma=1.0, gamma_s=0.9)
    zero = torch.zeros(1, 1)
    # the same absolute error costs ~10x less on a dry sample than a wet one
    dry = loss(torch.ones(1, 1), zero)
    wet = loss(torch.full((1, 1), 9.0), torch.full((1, 1), 10.0))
    assert float(dry) == pytest.approx(0.1)
    assert float(wet) == pytest.approx(1 - 0.9 * np.exp(-10.0))


class _Echo(torch.nn.Module):
    """Returns the first signal channel as rain and 1 as wet probability."""

    def init_state(self, batch_size):
        return torch.zeros(batch_size, 1)

    def forward(self, signal, metadata, state):
        out = torch.stack([signal[..., 0], torch.ones_like(signal[..., 0])], dim=-1)
        return out, state + 1


def test_windows_cover_the_series_once_and_carry_state():
    cfg = rr.TrainConfig(window_size=4)
    n_t = 10                                   # 2 full windows, 2 samples dropped
    rain = torch.arange(n_t, dtype=torch.float32).reshape(1, n_t, 1)
    rsl = torch.arange(n_t, dtype=torch.float32).reshape(1, n_t, 1)
    batch = (rain, rsl, rsl.clone(), torch.zeros(1, 2))
    got = list(rr.windows(_Echo(), batch, cfg))
    assert len(got) == 2
    np.testing.assert_array_equal(torch.cat([g[0] for g in got], 1).numpy(),
                                  np.arange(8)[None])
    np.testing.assert_array_equal(torch.cat([g[2] for g in got], 1).numpy(),
                                  np.arange(8)[None])


def test_detection_scores_put_the_gauge_on_the_rows():
    ref = np.array([0.0, 0.0, 0.0, 5.0])
    det = np.array([0, 0, 1, 1])               # one false alarm, one hit
    s = rr.detection_scores(ref, det)
    np.testing.assert_array_equal(s["confusion"], [[2, 1], [0, 1]])
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["f1"] == pytest.approx(2 / 3)


def test_numpy2_patch_finds_nearest_gauge():
    pytest.importorskip("pynncml")
    import pynncml_compat
    from pynncml.datasets import sensors_set

    pynncml_compat.patch_numpy2()

    class G:                                   # PointSensor stores 1-element arrays
        def __init__(self, x, y):
            self.x, self.y = np.array([x]), np.array([y])

    ps = sensors_set.PointSet.__new__(sensors_set.PointSet)
    ps.point_set = [G(0, 0), G(3, 4), G(10, 0)]
    d, [g] = ps.find_near_gauge((3.0, 3.0))
    assert d == pytest.approx(1.0) and g is ps.point_set[1]
    dists, gauges = ps.find_near_gauges((0.0, 0.0), range_limit=6.0)
    assert dists == pytest.approx([0.0, 5.0]) and len(gauges) == 2
