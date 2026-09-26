"""Moving rain fields: the shift is exact, and each sequence keeps its promises."""

from __future__ import annotations

import numpy as np
import pytest

from core.simulation import moving_fields as mf
from core.simulation.rain_fields import Grid, StratiformField, advect

GRID = Grid(n=32, dx_km=1.0)


def test_spectral_shift_by_whole_cells_is_a_roll():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(32, 32))
    out = mf.shift(f, 1.0, (3.0, -2.0), 1.0)            # 3 cells east, 2 south
    assert np.allclose(out, np.roll(np.roll(f, -2, axis=0), 3, axis=1), atol=1e-10)


def test_spectral_shift_composes():
    # below Nyquist; the Nyquist mode of an even grid has no fractional shift
    f = StratiformField().build(Grid(n=33, dx_km=1.0))
    once = mf.shift(f, 1.0, (7.3, 2.1), 1.0)
    twice = mf.shift(mf.shift(f, 1.0, (7.3, 2.1), 0.5), 1.0, (7.3, 2.1), 0.5)
    assert np.allclose(once, twice, atol=1e-8)


def test_integer_shift_matches_legacy_advect():
    f = StratiformField().build(GRID)
    assert np.array_equal(advect(f, GRID, (14.0, 5.0), 30.0),
                          mf.shift(f, 1.0, (14.0, 5.0), 0.5, method="integer"))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        mf.shift(np.zeros((4, 4)), 1.0, (1.0, 1.0), 1.0, method="bilinear")


def test_frozen_sequence_is_perfectly_predictable():
    seq = mf.sequence(StratiformField(), GRID, 6, dt_min=15, velocity_kmh=(8.0, 4.0))
    assert seq.frames.shape == (6, 32, 32)
    assert seq.predictability(2) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(seq.times_min, [0, 15, 30, 45, 60, 75])


def test_evolution_decorrelates_but_keeps_the_marginals():
    seq = mf.sequence(StratiformField(), GRID, 30, dt_min=15, velocity_kmh=(8.0, 4.0),
                      evolve_tau_min=60)
    p1, p4 = seq.predictability(1), seq.predictability(4)
    assert 1.0 > p1 > p4 > 0.0
    wet = mf.wet_fraction(seq.frames)
    assert np.allclose(wet, wet[0], atol=0.02)
    assert np.allclose(seq.frames.mean(axis=(1, 2)), seq.frames[0].mean(), rtol=0.02)


def test_growth_scales_intensity_exponentially():
    seq = mf.sequence("smooth", GRID, 5, dt_min=30, velocity_kmh=(0.0, 0.0), growth_per_h=-0.5)
    means = seq.frames.mean(axis=(1, 2))
    assert np.allclose(means[1:] / means[:-1], np.exp(-0.5 * 0.5), rtol=1e-4)


def test_velocity_defaults_to_the_models_own():
    seq = mf.sequence(StratiformField(), GRID, 2)
    assert seq.velocity_kmh == StratiformField().advection_kmh


def test_non_periodic_front_moves_without_wrapping():
    from core.simulation.rain_fields import FrontalBandField

    seq = mf.sequence(FrontalBandField(), GRID, 4, dt_min=15, velocity_kmh=(12.0, 0.0))
    assert seq.frames.shape == (4, 32, 32) and seq.pad > 0
    # 3 km per step on 1 km cells: each frame is the previous one moved 3 columns
    assert np.allclose(seq.frames[1][:, 3:], seq.frames[0][:, :-3], atol=1e-3)
    assert seq.predictability(2) == pytest.approx(1.0, abs=1e-3)
