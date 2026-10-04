"""core.maps.learning and core.maps.map_skill on synthetic links and fields."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.geo import Grid
from core.maps import learning as ml
from core.maps import map_skill as sk


def _grid(n_lat=12, n_lon=16, res=0.01):
    return Grid(np.round(57.6 + res * np.arange(n_lat), 6), np.round(11.9 + res * np.arange(n_lon), 6))


def _links(grid, times, seed=0, n=6):
    """Links inside the grid, each with a random value per step."""
    rng = np.random.default_rng(seed)
    la0 = rng.uniform(grid.lat[1], grid.lat[-2], n)
    lo0 = rng.uniform(grid.lon[1], grid.lon[-2], n)
    la1 = np.clip(la0 + rng.uniform(-0.04, 0.04, n), grid.lat[0], grid.lat[-1])
    lo1 = np.clip(lo0 + rng.uniform(-0.06, 0.06, n), grid.lon[0], grid.lon[-1])
    v = rng.gamma(1.0, 2.0, (n, len(times)))
    return xr.DataArray(v, dims=("link", "time"),
                        coords={"link": [f"l{i}" for i in range(n)], "time": times,
                                "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                                "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                                "frequency": ("link", np.full(n, 38.0)),
                                "polarization": ("link", np.array(["v"] * n, dtype=object))},
                        attrs={"units": "mm"})


def _times(n_events=5, hours=4):
    starts = pd.date_range("2015-07-01", periods=n_events, freq="2D")
    return [pd.date_range(s + pd.Timedelta("1h"), periods=hours, freq="1h") for s in starts]


def _dataset(seed=0, n_events=5):
    grid = _grid()
    spans = _times(n_events)
    times = spans[0].append(spans[1:])
    links = _links(grid, times, seed)
    rng = np.random.default_rng(seed + 1)
    target = xr.DataArray(rng.gamma(1.0, 1.0, (len(times),) + grid.shape).astype("float32"),
                          dims=("time", "lat", "lon"), coords={"time": times, "lat": grid.lat, "lon": grid.lon})
    ids = [f"ev{k}" for k in range(n_events)]
    ev = np.repeat(ids, [len(s) for s in spans])
    events = pd.DataFrame({"event_id": ids, "start": [s[0] - pd.Timedelta("1h") for s in spans],
                           "end": [s[-1] for s in spans]})
    gauges = xr.DataArray(rng.gamma(1.0, 1.0, (3, len(times))), dims=("station", "time"),
                          coords={"station": ["g0", "g1", "g2"], "time": times,
                                  "lat": ("station", grid.lat[[2, 5, 8]]),
                                  "lon": ("station", grid.lon[[3, 7, 11]])})
    splits = ml.split_events(ids, seed=seed)
    att = links * 0.3
    return ml.assemble(links, target, ev, events, splits, link_attenuation=att, gauges=gauges), links, grid


# ---------------------------------------------------------------- rasterization
def test_path_weights_sum_to_one_inside_the_grid():
    grid = _grid()
    links = _links(grid, pd.date_range("2015-07-01", periods=2, freq="1h"))
    W = ml.path_weights(links, grid)
    assert W.shape == (grid.lat.size * grid.lon.size, links.sizes["link"])
    np.testing.assert_allclose(W.sum(0), 1.0, atol=1e-12)


def test_rasterization_conserves_the_path_mean_of_a_single_link():
    grid = _grid()
    links = _links(grid, pd.date_range("2015-07-01", periods=3, freq="1h"), n=1)
    W = ml.path_weights(links, grid)
    L = ml.link_lengths_km(links)
    r = ml.rasterize(links.values, W, L, grid.shape)
    flat = r["mean"].reshape(3, -1)
    # the path-weighted mean of the raster along the link is the link's value
    back = (W[:, 0][None, :] * np.nan_to_num(flat)).sum(1)
    np.testing.assert_allclose(back, links.values[0], rtol=1e-10)
    np.testing.assert_allclose(ml.sample_paths(r["mean"], W)[0], links.values[0], rtol=1e-10)
    # coverage: the link's length, spread over its cells
    np.testing.assert_allclose(r["coverage"].reshape(3, -1).sum(1), L[0], rtol=1e-10)


def test_rasterization_conserves_value_times_length_for_many_links():
    grid = _grid()
    links = _links(grid, pd.date_range("2015-07-01", periods=4, freq="1h"), n=8, seed=3)
    vals = links.values.copy()
    vals[2, 1] = np.nan                                     # a missing link drops out of that step
    W = ml.path_weights(links, grid)
    L = ml.link_lengths_km(links)
    r = ml.rasterize(vals, W, L, grid.shape)
    lhs = np.nansum(r["mean"] * r["coverage"], axis=(1, 2))
    rhs = np.nansum(vals * L[:, None], axis=0)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)
    assert np.all(np.isnan(r["mean"]) == (r["coverage"] == 0))


def test_distance_to_links_is_zero_on_a_path():
    grid = _grid()
    links = _links(grid, pd.date_range("2015-07-01", periods=1, freq="1h"), n=3)
    W = ml.path_weights(links, grid)
    d = ml.distance_to_links(grid, links).values.ravel()
    on_path = W.sum(1) > 0
    assert d[on_path].max() < 1.0                            # within a cell of the path
    assert d.min() >= 0 and d[~on_path].min() >= 0


# ---------------------------------------------------------------- splits
@pytest.mark.parametrize("how", ["random", "time"])
def test_split_by_event_never_shares_an_event(how):
    ids = [f"e{k}" for k in range(11)]
    s = ml.split_events(ids, (0.6, 0.2, 0.2), seed=4, how=how)
    assert list(s.index) == ids and set(s) == set(ml.SPLITS)
    assert s.index.is_unique
    if how == "time":
        assert list(s) == sorted(s, key=ml.SPLITS.index)


def test_dataset_steps_follow_their_event_split():
    ds, _, _ = _dataset()
    per_event = pd.Series(ds.split.values).groupby(ds.event_id.values).nunique()
    assert (per_event == 1).all()                            # no event in two splits
    for e, sp in zip(ds.event.values, ds.event_split.values):
        assert set(ds.split.values[ds.event_id.values == e]) == {sp}
    for a in ml.SPLITS:
        for b in ml.SPLITS:
            if a < b:
                ea = set(ds.event_id.values[ds.split.values == a])
                eb = set(ds.event_id.values[ds.split.values == b])
                assert not ea & eb


# ---------------------------------------------------------------- dataset
def test_dataset_shapes_alignment_and_round_trip(tmp_path):
    ds, links, grid = _dataset()
    nt = links.sizes["time"]
    assert ds.inputs.shape == (nt, len(ml.CHANNELS), *grid.shape)
    assert ds.target.shape == (nt, *grid.shape)
    assert ds.link_rain.shape == links.shape and ds.gauges.shape == (3, nt)
    assert np.isfinite(ds.inputs.values).all()
    np.testing.assert_array_equal(ds.time.values, links.time.values)
    # the rain channel at a cell crossed by one link only is that link's value
    W = ml.path_weights(links, grid)
    only = np.flatnonzero((W > 0).sum(1) == 1)
    c = only[0]
    k = int(np.flatnonzero(W[c])[0])
    i, j = divmod(c, grid.lon.size)
    np.testing.assert_allclose(ds.inputs.sel(channel="rain").values[:, i, j], links.values[k], rtol=1e-6)
    np.testing.assert_allclose(ds.inputs.sel(channel="attenuation").values[:, i, j],
                               0.3 * links.values[k] / ds.length.values[k], rtol=1e-5)
    path = ml.save_dataset(ds, tmp_path / "d.nc")
    back = ml.load_dataset(path)
    np.testing.assert_allclose(back.inputs.values, ds.inputs.values)
    assert list(back.split.values) == list(ds.split.values)
    a = ml.arrays(back, "train")
    assert a["x"].shape[1:] == (len(ml.CHANNELS), *grid.shape)
    assert set(a["event_id"]) == set(back.event.values[back.event_split.values == "train"])
    lr = ml.link_rain_of(back)
    assert {"mid_lat", "frequency", "polarization", "length"} <= set(lr.coords)


def test_from_synthetic_builds_one_event_per_case():
    from core.simulation.scenario import Scenario
    cases = [Scenario(n=32, dx_km=1.0, duration_min=30, interval_min=10, n_links=8, n_gauges=3,
                      start=f"2026-07-0{k + 1} 12:00", seed=k).run() for k in range(3)]
    ds = ml.from_synthetic(cases)
    assert ds.sizes["event"] == 3 and set(ds.split.values) == set(ml.SPLITS)
    assert ds.target.shape == (9, 16, 16) and "radar" in ds
    assert ds.sizes["link"] == sum(c.links.sizes["link"] for c in cases) and ds.sizes["station"] == 9
    with pytest.raises(ValueError):
        ml.from_synthetic([cases[0], cases[0]])


# ---------------------------------------------------------------- metrics
def _fields(nt=6, shape=(5, 7), seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-07-01 01:00", periods=nt, freq="1h")
    coords = {"time": t, "lat": np.arange(shape[0]) * 0.01 + 57.0, "lon": np.arange(shape[1]) * 0.01 + 12.0}
    ref = xr.DataArray(rng.gamma(1.0, 1.0, (nt, *shape)), dims=("time", "lat", "lon"), coords=coords)
    return ref, coords


def test_pcc_per_step_and_pooled():
    ref, _ = _fields()
    est = 2 * ref + 1
    np.testing.assert_allclose(sk.pcc_per_step(est, ref).values, 1.0)
    assert sk.pcc_pooled(est, ref) == pytest.approx(1.0)
    flat = ref.copy(data=np.ones(ref.shape))
    assert np.isnan(sk.pcc_per_step(flat, ref).values).all()
    assert isinstance(sk.pcc_per_step(est, ref).index, pd.DatetimeIndex)


def test_cumulative_event_error():
    ref, _ = _fields()
    event = np.array(["a"] * 3 + ["b"] * 3)
    est = ref * 1.5
    t = sk.event_totals(est, ref, event)
    np.testing.assert_allclose(t.rel_error, 0.5)
    expect_a = float(ref.values[:3].sum(0).mean())
    assert t.loc["a", "total_ref"] == pytest.approx(expect_a)
    c = sk.cumulative_event_error(est, ref, event)
    assert c["events"] == 2 and c["event_abs_rel_error"] == pytest.approx(0.5)
    # a NaN cell-step drops out of both totals
    e2 = est.copy()
    e2[0, 0, 0] = np.nan
    t2 = sk.event_totals(e2, ref, event)
    assert t2.loc["a", "rel_error"] == pytest.approx(0.5)


def test_detection_counts():
    ref = np.array([[0.0, 0.5, 2.0, 3.0]])[:, None, :]
    est = np.array([[1.0, 0.0, 2.0, 3.0]])[:, None, :]
    d = sk.detection(est, ref, threshold=1.0)
    # wet ref: 2, 3 -> both hit; est wet at 0 (false alarm)
    assert d["pod"] == 1.0 and d["far"] == pytest.approx(1 / 3) and d["csi"] == pytest.approx(2 / 3)


def test_gauge_point_error_and_skill_table():
    ref, coords = _fields()
    g = ref.isel(lat=[1, 3], lon=[2, 5]).stack(station=("lat", "lon")).reset_index("station")
    gauges = xr.DataArray(g.transpose("station", "time").values, dims=("station", "time"),
                          coords={"station": ["s0", "s1", "s2", "s3"], "time": ref.time,
                                  "lat": ("station", g.lat.values), "lon": ("station", g.lon.values)})
    s = sk.gauge_point_error(ref, gauges)
    assert s["rmse"] == pytest.approx(0.0) and s["n"] == 4 * ref.sizes["time"]
    gappy = ref.where(ref.lat > 57.0)                      # first row missing
    mask = np.ones(ref.shape[1:], bool)
    tab = sk.skill_table({"perfect": ref, "double": 2 * ref, "gappy": gappy}, ref,
                         event=np.array(["a"] * 3 + ["b"] * 3), mask=mask, gauges=gauges)
    assert tab.loc["perfect", "rmse"] == 0 and tab.loc["perfect", "pcc"] == pytest.approx(1)
    assert tab.loc["double", "event_abs_rel_error"] == pytest.approx(1.0)
    # common sample: every map scored on the same cells
    assert tab["n"].nunique() == 1 and tab.loc["perfect", "n"] == 6 * 4 * 7
    assert tab["gauge_n"].nunique() == 1
    assert {"pod", "far", "csi", "pcc_step", "gauge_pcc"} <= set(tab.columns)
