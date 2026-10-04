"""learned_2d helpers on synthetic data: the kriging baseline and the minimal U-Net."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.geo import Grid  # noqa: E402
from learned_2d.baselines import baseline_maps, ordinary_kriging_map  # noqa: E402

VARIOGRAM = {"range": 8000.0, "nugget": 0.1}


def _links(values, n=12, seed=0):
    rng = np.random.default_rng(seed)
    la0, lo0 = rng.uniform(57.62, 57.70, n), rng.uniform(11.92, 12.05, n)
    la1, lo1 = la0 + rng.uniform(-0.02, 0.02, n), lo0 + rng.uniform(-0.03, 0.03, n)
    t = pd.date_range("2015-07-01 01:00", periods=values.shape[1], freq="1h")
    return xr.DataArray(values, dims=("link", "time"),
                        coords={"link": [f"l{i}" for i in range(n)], "time": t,
                                "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                                "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                                "mid_lat": ("link", (la0 + la1) / 2), "mid_lon": ("link", (lo0 + lo1) / 2),
                                "frequency": ("link", np.full(n, 38.0)),
                                "polarization": ("link", np.array(["v"] * n)),
                                "length": ("link", np.full(n, 2.0))})


GRID = Grid(np.round(57.55 + 0.01 * np.arange(25), 6), np.round(11.85 + 0.01 * np.arange(30), 6))


def test_kriging_of_a_constant_field_is_constant_and_stays_near_the_links():
    v = np.full((12, 2), 3.0)
    v[0, 1] = np.nan
    m = ordinary_kriging_map(_links(v), GRID, VARIOGRAM, radius_m=5000)
    vals = m.values[np.isfinite(m.values)]
    np.testing.assert_allclose(vals, 3.0, rtol=1e-6)
    assert np.isnan(m.values).any()                  # cells far from every link are left empty


def test_kriging_follows_a_gradient_and_needs_enough_links():
    rng = np.random.default_rng(1)
    lk = _links(np.zeros((12, 2)))
    v = 10 * (lk.mid_lon.values - 11.9)[:, None] + np.array([0.0, 0.0])
    v[3:, 1] = np.nan                                 # hour 2: only three links
    lk = lk.copy(data=v + rng.normal(0, 0.01, v.shape))
    m = ordinary_kriging_map(lk, GRID, VARIOGRAM, min_links=4)
    west, east = m.isel(time=0).sel(lon=11.95, method="nearest"), m.isel(time=0).sel(lon=12.05, method="nearest")
    assert float(east.mean()) > float(west.mean())
    assert np.isnan(m.isel(time=1).values).all()


def test_baseline_maps_on_a_learning_dataset():
    from core.maps import learning as ml
    lk = _links(np.random.default_rng(2).gamma(1, 2, (12, 4)))
    target = xr.DataArray(np.ones((4,) + GRID.shape, "float32"), dims=("time", "lat", "lon"),
                          coords={"time": lk.time, "lat": GRID.lat, "lon": GRID.lon})
    events = pd.DataFrame({"event_id": ["a", "b"], "start": lk.time.values[[0, 2]], "end": lk.time.values[[1, 3]]})
    ds = ml.assemble(lk, target, ["a", "a", "b", "b"], events, pd.Series({"a": "train", "b": "test"}))
    maps = baseline_maps(ds, VARIOGRAM)
    assert set(maps) == {"idw", "ok", "gmz"}
    for m in maps.values():
        assert m.shape == (4,) + GRID.shape
        assert np.isfinite(m.values[:, ds.observable.values.astype(bool)]).all()


def test_tiny_unet_shapes_and_learning():
    torch = pytest.importorskip("torch")
    from learned_2d.unet import TinyUNet, fit, predict
    rng = np.random.default_rng(0)
    x = rng.gamma(1, 1, (24, 4, 13, 18)).astype("float32")
    y = 2.0 * x[:, 0]                                 # the target is a simple function of channel 0
    mask = np.ones((13, 18), bool)
    model = TinyUNet(n_in=4, c=(8, 16, 16))
    assert model(torch.zeros(2, 4, 13, 18)).shape == (2, 13, 18)
    hist = fit(model, x[:16], y[:16], mask, x[16:], y[16:], epochs=15, batch=8, lr=5e-3)
    assert hist[-1]["train"] < hist[0]["train"]
    p = predict(model, x[16:])
    assert p.shape == (8, 13, 18) and (p >= 0).all()
