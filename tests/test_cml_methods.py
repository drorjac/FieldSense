"""core.cml and core.maps: power law, baselines, estimators, IDW, scores - no data, no network."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.cml import baseline as bl
from core.cml.power_law import attenuation_from_rain, itu_ab, rain_from_attenuation
from core.geo import Grid
from core.maps.idw import accumulate, idw_map, idw_weights
from core.maps.scores import scores


def test_itu_coefficients_match_both_libraries():
    a, b = itu_ab(68.04, "v", "ITU_2003")          # PyNNcml / implementation_1
    assert a[0] == pytest.approx(0.7577, abs=1e-4) and b[0] == pytest.approx(0.7985, abs=1e-4)
    a, b = itu_ab(68.04, "v", "ITU_2005")          # pycomlink / implementation_2
    assert a[0] == pytest.approx(0.9939, abs=1e-4) and b[0] == pytest.approx(0.7261, abs=1e-4)


def test_power_law_roundtrip_and_conventions():
    R = np.array([0.5, 2.0, 10.0, 50.0])
    A = attenuation_from_rain(R, 2.0, 68.04, "v")
    assert np.allclose(rain_from_attenuation(A, 2.0, 68.04, "v", r_min=0), R)
    out = rain_from_attenuation(np.array([-1.0, 0.0, np.nan, 1e-4]), 2.0, 68.04, "v", r_min=0.1)
    assert out[0] == 0 and out[1] == 0 and np.isnan(out[2]) and out[3] == 0


def test_power_law_per_link_broadcast():
    A = np.ones((2, 5))
    R = rain_from_attenuation(A, [1.0, 2.0], [68.04, 24.1], ["v", "h"], r_min=0)
    assert R.shape == (2, 5) and not np.allclose(R[0], R[1])


def test_frequency_out_of_table_raises():
    with pytest.raises(ValueError):
        itu_ab(0.5, "v")


def test_trailing_min_matches_naive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 50))
    w = 7
    naive = np.array([[x[i, max(0, t - w + 1):t + 1].min() for t in range(50)] for i in range(3)])
    assert np.allclose(bl.trailing_min(x, w), naive)


def test_trailing_min_propagates_nan():
    x = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
    out = bl.trailing_min(x, 2)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2]) and out[0, 3] == 3.0


def test_dynamic_baseline_removes_constant_offset():
    A = np.full((1, 300), 40.0)
    A[0, 250:260] += 10
    A_rain, _ = bl.dynamic_baseline(A, window=200, quantization_delta=1.0)
    assert np.allclose(A_rain[0, :250], -1.0)            # qd bias correction
    assert np.allclose(A_rain[0, 250:260], 9.0)


def test_constant_baseline_holds_while_wet():
    A = np.array([[1.0, 2.0, 5.0, 6.0, 3.0]])
    wet = np.array([[0, 0, 1, 1, 0]])
    assert np.allclose(bl.constant_baseline(A, wet), [[1, 2, 2, 2, 3]])


def test_std_wet_dry_edges_are_dry():
    A = np.random.default_rng(1).normal(scale=5, size=(1, 100))
    wet, sig = bl.std_wet_dry(A, window=20, threshold=1.0)
    assert (wet[0, :9] == 0).all() and (wet[0, -10:] == 0).all() and wet[0, 20:80].max() == 1


def test_estimators_match_pynncml():
    torch = pytest.importorskip("torch")
    pnc = pytest.importorskip("pynncml")
    rng = np.random.default_rng(2)
    att = 40 + np.round(np.cumsum(rng.normal(0, 0.3, 1200)))
    att[500:560] += np.linspace(0, 12, 60)
    A_rain, _ = bl.dynamic_baseline(att[None], 200, 1.0)
    ours = rain_from_attenuation(A_rain, 2.0, 68.04, "v", table="ITU_2003", r_min=0.5)
    m = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.INSTANCE, 0.5, 200,
                                                          quantization_delta=1)
    ref, _ = m(torch.tensor(att[None], dtype=torch.float32), pnc.datasets.MetaData(68.04, True, 2.0, 10, 10))
    assert np.allclose(ours, ref.numpy(), atol=1e-4)


def _links(lat, lon, vals):
    return xr.DataArray(np.asarray(vals, float)[:, None], dims=("link", "time"),
                        coords={"link": [f"l{i}" for i in range(len(lat))],
                                "time": pd.to_datetime(["2024-01-01"]),
                                "mid_lat": ("link", lat), "mid_lon": ("link", lon)})


def test_idw_exact_at_source_and_bounded():
    grid = Grid(np.array([40.70, 40.71]), np.array([-74.00, -73.99]))
    da = _links([40.70, 40.71], [-74.00, -73.99], [2.0, 8.0])
    m = idw_map(da, grid)
    assert m.sel(lat=40.70, lon=-74.00).item() == pytest.approx(2.0)
    assert m.sel(lat=40.71, lon=-73.99).item() == pytest.approx(8.0)
    assert 2.0 < m.sel(lat=40.70, lon=-73.99).item() < 8.0


def test_idw_nan_policies_and_radius():
    grid = Grid(np.array([40.70]), np.array([-73.995]))
    da = _links([40.70, 40.70], [-74.00, -73.99], [np.nan, 4.0])
    assert idw_map(da, grid, nan_policy="exclude").item() == pytest.approx(4.0)
    assert idw_map(da, grid, nan_policy="zero").item() == pytest.approx(2.0)
    far = idw_map(da, Grid(np.array([41.5]), np.array([-73.99])), radius_m=10_000)
    assert np.isnan(far.item())


def test_idw_nnear():
    W = idw_weights(np.array([0.0, 100.0, 1000.0]), np.zeros(3), np.array([50.0]), np.zeros(1), nnear=2)
    assert W[0, 2] == 0 and W[0, 0] > 0 and W[0, 1] > 0


def test_accumulate_hour_ending():
    t = pd.date_range("2024-01-01 00:01", "2024-01-01 02:00", freq="1min")
    da = xr.DataArray(np.full((1, t.size), 6.0), dims=("link", "time"), coords={"time": t})
    acc = accumulate(da, "1h")
    assert list(pd.DatetimeIndex(acc.time.values)) == list(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00"]))
    assert np.allclose(acc.values, 6.0)


def test_scores():
    s = scores([1, 2, 3, np.nan], [1, 2, 5, 1])
    assert s["n"] == 3 and s["bias"] == pytest.approx(-2 / 3)
    assert s["nrmse"] == pytest.approx(np.sqrt(4 / 3) / (8 / 3))
    assert s["csi"] == 1.0


def test_accumulate_coverage_and_infill():
    t = pd.date_range("2024-01-01 00:01", "2024-01-01 02:00", freq="1min")
    v = np.full(t.size, 6.0)
    v[:30] = np.nan                         # first hour half missing -> NaN
    v[60:66] = np.nan                       # second hour 90 % present -> infilled
    acc = accumulate(xr.DataArray(v, dims="time", coords={"time": t}), "1h")
    assert np.isnan(acc.values[0]) and acc.values[1] == pytest.approx(6.0)


def test_trailing_min_skipna():
    x = np.array([[5.0, np.nan, 3.0, 4.0]])
    assert np.allclose(bl.trailing_min(x, 2, skipna=True), [[5, 5, 3, 3]])


# ------------------------------------------------------------------ GMZ

def _gmz_links(values, lats=(40.70, 40.70), lons=((-74.00, -73.96), (-73.96, -73.92))):
    t = pd.to_datetime(["2024-01-01 01:00"])
    return xr.DataArray(np.asarray(values, float)[:, None], dims=("link", "time"),
                        coords={"link": ["a", "b"], "time": t,
                                "site_0_lat": ("link", list(lats)), "site_1_lat": ("link", list(lats)),
                                "site_0_lon": ("link", [lo[0] for lo in lons]),
                                "site_1_lon": ("link", [lo[1] for lo in lons]),
                                "frequency": ("link", [38.0, 38.0]), "polarization": ("link", ["v", "v"])})


def test_gmz_keeps_uniform_rain_and_the_link_average():
    from core.maps.gmz import gmz_map, virtual_gauges
    grid = Grid(np.round(np.arange(40.66, 40.74, 0.005), 4), np.round(np.arange(-74.02, -73.90, 0.005), 4))
    uniform = gmz_map(_gmz_links([4.0, 4.0]), grid, n_iter=10, radius_m=5000)
    assert np.nanmax(np.abs(uniform.values - 4.0)) < 1e-6
    # unequal links: the field along each path leans to its neighbour, but its mean in the
    # R**b domain stays the link's value (what GMZ preserves)
    links = _gmz_links([2.0, 10.0])
    field_ = gmz_map(links, grid, n_iter=10, radius_m=5000)
    lat, lon, owner = virtual_gauges(links, per_km=1.0)
    at = field_.isel(time=0).interp(lat=xr.DataArray(lat), lon=xr.DataArray(lon)).values
    b = 0.8816
    for i, v in enumerate([2.0, 10.0]):
        got = np.mean(np.clip(at[owner == i], 0, None) ** b) ** (1 / b)
        assert got == pytest.approx(v, rel=0.15)
    line = gmz_map(links, grid, n_iter=0, radius_m=5000)          # line IDW: no correction
    assert np.isfinite(line.values).any() and not np.allclose(line.values, field_.values, equal_nan=True)


def test_rnn_physics_features_invert_the_power_law():
    from core.cml.power_law import itu_ab
    from core.cml.rnn import FEATURES, N_MIN, physics_features
    a, b = itu_ab(38.0, "v", "ITU_2005")
    L, R = 2.0, 10.0
    A = float(a[0] * R ** b[0] * L)                  # attenuation of 10 mm/h over 2 km
    X = np.zeros((1, 2, len(FEATURES)), "float32")
    X[0, 0, :N_MIN] = A                              # hour 1: steady rain
    X[0, 0, N_MIN + 1] = A
    X[0, 1, :N_MIN] = -3.0                           # hour 2: below the baseline -> no rain
    M = np.array([[38.0, L, 1.0, np.log10(a[0]), b[0]]], "float32")
    p = physics_features(X, M)
    assert p[0, 0] == pytest.approx([R, R, R], rel=1e-3)
    assert p[0, 1].tolist() == [0.0, 0.0, 0.0]


def test_rnn_neighbour_features_see_the_surrounding_links():
    from core.cml.rnn import FEATURES, N_MIN, neighbour_features
    X = np.zeros((3, 1, len(FEATURES)), "float32")
    X[1, 0, :N_MIN] = X[1, 0, N_MIN + 1] = 5.0       # the neighbour sees rain
    M = np.tile(np.array([[38.0, 2.0, 1.0, np.log10(0.4), 0.88]], "float32"), (3, 1))
    lat, lon = [40.70, 40.71, 41.50], [-74.0, -74.0, -74.0]   # the third is 90 km away
    f = neighbour_features(X, M, lat, lon, radius_km=15)
    assert f[0, 0, 0] > 0 and f[0, 0, 1] == 1.0         # link 0's only neighbour is wet
    assert f[1, 0].tolist() == [0.0, 0.0]               # link 1's neighbour (0) is dry
    assert f[2, 0].tolist() == [0.0, 0.0]               # link 2 has no neighbour
