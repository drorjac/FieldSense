"""The CML retrieval chain recovers known rain, and fails loudly when misused."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import attenuation_for, make_cml, rain_event
from core.opensense import example_data
from core.opensense import retrieval as rt

NO_WAA = dict(waa_model="none")


def _retrieve(ds, wet=None, **cfg):
    return rt.retrieve_dataset(ds, rt.RetrievalConfig.for_interval(60, **cfg),
                               wet=wet)


def _perfect_wet(ds, rain):
    """The true wet/dry state per link, so a test exercises the chain alone."""
    return xr.DataArray(rain > 0, dims=("time", "cml_id"),
                        coords={"time": ds.time, "cml_id": ds.cml_id})


def test_recovers_steady_rain_without_wet_antenna(steady_event):
    rain, freqs, lengths = steady_event
    ds = example_data.normalize_cml(make_cml(rain, freqs, lengths), "EPSG:32632")
    out = _retrieve(ds, _perfect_wet(ds, rain), **NO_WAA)

    wet = out.R.isel(time=slice(630, 690))           # well inside the event
    np.testing.assert_allclose(wet.values, 10.0, rtol=0.02)
    dry = out.R.isel(time=slice(0, 500))
    assert float(dry.max()) == 0.0


def test_metadata_is_matched_by_dimension_name(steady_event):
    """Frequency stored (sublink_id, cml_id) must reach the right sublink.

    Regression: flattening the signals as (cml_id, sublink_id) and the
    metadata as stored paired sublinks with other links' frequencies. The
    retrieval then disagrees between the two sublinks of the same link.
    """
    rain, freqs, lengths = steady_event
    for dims in (("sublink_id", "cml_id"), ("cml_id", "sublink_id")):
        ds = example_data.normalize_cml(
            make_cml(rain, freqs, lengths, freq_dims=dims), "EPSG:32632")
        r = _retrieve(ds, _perfect_wet(ds, rain), **NO_WAA).R.isel(time=660)
        np.testing.assert_allclose(r.values, 10.0, rtol=0.02, err_msg=str(dims))


def test_rsl_only_equals_tsl_minus_rsl_when_tsl_is_constant(steady_event):
    rain, freqs, lengths = steady_event
    with_tsl = example_data.normalize_cml(make_cml(rain, freqs, lengths), "EPSG:32632")
    rsl_only = with_tsl.drop_vars("tsl")
    a = _retrieve(with_tsl).R
    b = _retrieve(rsl_only).R
    np.testing.assert_allclose(a.values, b.values)
    assert "no TSL" in _retrieve(rsl_only).attrs["loss"]


@pytest.mark.parametrize("model", ["saturating", "pastorek2021", "leijnse2008"])
def test_wet_antenna_models_reduce_rain_and_keep_dry_dry(steady_event, model):
    rain, freqs, lengths = steady_event
    ds = example_data.normalize_cml(make_cml(rain, freqs, lengths), "EPSG:32632")
    wet = _perfect_wet(ds, rain)
    none = _retrieve(ds, wet, **NO_WAA)
    corr = _retrieve(ds, wet, waa_model=model)

    inside = slice(630, 690)
    assert float(corr.R.isel(time=inside).mean()) < float(none.R.isel(time=inside).mean())
    assert float(corr.waa.isel(time=inside).min()) > 0.0
    assert float(corr.R.isel(time=slice(0, 500)).max()) == 0.0
    # WAA can never exceed the attenuation it is removed from
    assert bool((corr.waa <= corr.A_obs + 1e-9).all())


def test_short_links_and_gaps_are_nan_not_zero():
    rain = rain_event(n_links=2)
    ds = make_cml(rain, [[23.0, 23.0], [23.0, 23.0]], np.array([0.3, 4.0]))
    ds["rsl"][dict(time=slice(650, 655))] = np.nan
    out = _retrieve(example_data.normalize_cml(ds, "EPSG:32632"))
    assert bool(out.R.sel(cml_id=100).isnull().all())          # 0.3 km link
    assert bool(out.R.sel(cml_id=101).isel(time=slice(650, 655)).isnull().all())
    assert bool(out.R.sel(cml_id=101).isel(time=slice(660, 680)).notnull().all())


def test_external_wet_mask_and_zero_when_dry(steady_event):
    rain, freqs, lengths = steady_event
    ds = example_data.normalize_cml(make_cml(rain, freqs, lengths, noise_db=0.05),
                                    "EPSG:32632")
    wet = _perfect_wet(ds, rain)          # broadcast over sublinks
    masked = rt.retrieve_dataset(
        ds, rt.RetrievalConfig.for_interval(60, zero_when_dry=True, **NO_WAA),
        wet=wet)
    dry = slice(0, 500)
    assert float(masked.R.isel(time=dry).max()) == 0.0
    np.testing.assert_allclose(masked.R.isel(time=660).values, 10.0, rtol=0.05)


def test_wet_mask_shape_is_checked():
    with pytest.raises(ValueError, match="wet mask shape"):
        rt.retrieve(np.zeros((10, 2)), [4, 4], [23, 23], ["v", "v"],
                    rt.RetrievalConfig(), wet=np.zeros((10, 3), bool))


def test_config_validation_and_interval_scaling():
    with pytest.raises(ValueError, match="waa_model"):
        rt.RetrievalConfig(waa_model="magic")
    cfg = rt.RetrievalConfig.for_interval(60)
    assert (cfg.wet_window, cfg.baseline_window) == (5, 180)
    # explicit window overrides win, and no duplicate-keyword error
    assert rt.RetrievalConfig.for_interval(60, wet_window=9).wet_window == 9
    assert cfg.with_(waa_max_db=2.3).waa_max_db == 2.3


def test_sampling_interval_ignores_a_gap():
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:30:00",
                  "2020-01-01T00:30:10", "2020-01-01T00:30:20",
                  "2020-01-01T00:30:30"], dtype="datetime64[ns]")
    assert rt.sampling_interval_s(t) == 10.0


def test_combine_sublinks_keeps_one_geometry_per_link(steady_event):
    rain, freqs, lengths = steady_event
    ds = example_data.normalize_cml(make_cml(rain, freqs, lengths), "EPSG:32632")
    per_link = rt.combine_sublinks(_retrieve(ds, **NO_WAA))
    assert per_link.R.dims == ("time", "cml_id")
    for c in ("site_0_x", "site_1_y", "length_km", "frequency_ghz"):
        assert per_link[c].dims == ("cml_id",)
    np.testing.assert_allclose(per_link.frequency_ghz.values, [18.0, 32.0])


def test_forward_model_helper_matches_itu():
    # 23 GHz V: k = 0.128, alpha = 0.963 -> 10 mm/h over 5 km is ~5.9 dB
    assert attenuation_for(10.0, 23.0, 5.0) == pytest.approx(5.90, abs=0.05)


def test_rolling_std_classifier_misses_steady_rain(steady_event):
    """The documented blind spot, pinned so it stays documented.

    The rolling-std test keys on fluctuation. Perfectly steady rain does not
    fluctuate, so it is classified dry, the baseline learns it, and the
    retrieval returns nothing. Real stratiform rain is noisy enough to be
    partly caught, which is why the chain never zeroes rain on this flag -
    and why the nearby-link mask (``core.opensense.wet_dry``) helps.
    """
    rain, freqs, lengths = steady_event
    ds = example_data.normalize_cml(make_cml(rain, freqs, lengths), "EPSG:32632")
    blind = _retrieve(ds, **NO_WAA).R.isel(time=slice(630, 690))
    seeing = _retrieve(ds, _perfect_wet(ds, rain), **NO_WAA).R.isel(time=slice(630, 690))
    assert float(blind.mean()) < 0.1 * float(seeing.mean())
