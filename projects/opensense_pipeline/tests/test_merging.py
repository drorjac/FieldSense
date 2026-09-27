"""Merge methods on a synthetic field: every method leaves its inputs untouched."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("mergeplg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import merging  # noqa: E402


def _case(seed=0, n=30, n_links=25, step=1000.0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * step
    xg, yg = np.meshgrid(x, x)
    truth = 5 + 4 * np.sin(xg / 7000) * np.cos(yg / 9000)
    truth[:, : n // 3] = 0.0                     # a dry third: real radar is mostly zeros
    da_rad = xr.DataArray(truth * rng.uniform(0.6, 1.4, truth.shape), dims=("y", "x"),
                          coords=dict(x=x, y=x, x_grid=(("y", "x"), xg),
                                      y_grid=(("y", "x"), yg)))
    x0, y0 = rng.uniform(2e3, (n - 3) * step, (2, n_links))
    ang = rng.uniform(0, np.pi, n_links)
    x1, y1 = x0 + 2e3 * np.cos(ang), y0 + 2e3 * np.sin(ang)
    mid_i = np.clip(((y0 + y1) / 2 / step).astype(int), 0, n - 1)
    mid_j = np.clip(((x0 + x1) / 2 / step).astype(int), 0, n - 1)
    da_cml = xr.DataArray(truth[mid_i, mid_j], dims=("cml_id",), coords=dict(
        cml_id=np.arange(n_links), site_0_x=("cml_id", x0), site_0_y=("cml_id", y0),
        site_1_x=("cml_id", x1), site_1_y=("cml_id", y1),
        x=("cml_id", (x0 + x1) / 2), y=("cml_id", (y0 + y1) / 2)))
    return da_rad, da_cml


@pytest.mark.parametrize("method", merging.METHODS, ids=lambda m: m.key)
def test_method_does_not_modify_the_radar_it_is_given(method):
    # mergeplg 0.1.0's kriging with external drift set every zero-rain cell of
    # its radar input to NaN; "radar only" returned that same array and was
    # scored on the damage
    da_rad, da_cml = _case()
    before = da_rad.values.copy()
    merging.run(method, da_rad, da_cml)
    np.testing.assert_array_equal(da_rad.values, before)


def test_radar_only_is_the_radar_field_even_after_every_other_method():
    da_rad, da_cml = _case(seed=1)
    fields = {m.key: merging.run(m, da_rad, da_cml) for m in merging.METHODS}
    np.testing.assert_array_equal(fields["radar_only"].values, da_rad.values)
    assert np.isfinite(fields["radar_only"].values).all()
