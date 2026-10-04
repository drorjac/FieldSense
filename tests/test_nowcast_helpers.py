"""core.nowcast: advection interpolation and SAL on synthetic fields."""

import numpy as np

from core.nowcast.advection import hourly_total
from core.nowcast.scores import sal_score


def _blob(cx, cy, n=60, width=20.0):
    yy, xx = np.indices((n, n))
    return 10 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / width)


def test_advection_fills_gaps_between_scans():
    # a narrow blob moving 8 px per scan: the plain mean leaves beads, interpolation a swath
    frames = np.stack([_blob(10 + 8 * i, 30, width=4.0) for i in range(6)])
    velocity = np.zeros((2, 60, 60))
    velocity[0] = 8.0
    plain = hourly_total(frames)
    smooth = hourly_total(frames, velocity, n_sub=8)
    swath = smooth[30, 10:50]
    beads = plain[30, 10:50]
    assert np.std(swath) < np.std(beads)
    assert np.isclose(smooth.sum(), plain.sum(), rtol=0.05)


def test_advection_without_motion_is_identity_for_still_rain():
    frames = np.stack([_blob(30, 30)] * 4)
    assert np.allclose(hourly_total(frames, np.zeros((2, 60, 60)), n_sub=4), frames[0])


def test_sal_perfect_amplitude_and_shift():
    obs = _blob(20, 30)
    assert np.allclose(sal_score(obs, obs), (0, 0, 0), atol=1e-12)
    s, a, l = sal_score(2 * obs, obs)
    assert a > 0.6 and abs(s) < 1e-9
    _, _, l = sal_score(np.roll(obs, 15, axis=1), obs)
    assert l > 0.1
    assert sal_score(np.zeros_like(obs), obs) is None
