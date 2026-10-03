"""Audit of the link chains: core/cml (baselines, wet/dry, preprocessing, QC, RNN
features) and the wet-antenna step of core/opensense/retrieval.

References are the libraries the code says it mirrors (PyNNcml, pycomlink) run on
the same synthetic input, or a hand calculation. Window placement (trailing vs
centred) and time labels (interval start vs end) are pinned explicitly, because
those are the mistakes that survive every round-trip test.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.cml import baseline as bl
from core.cml import preprocess as pp
from core.cml.power_law import itu_ab, rain_from_attenuation
from core.itu_p838 import get_k_alpha
from core.opensense import retrieval as rt


# ------------------------------------------------------------------ baselines


def test_dynamic_baseline_is_trailing_min_minus_quantization_delta():
    """Rain attenuation = A - min(A over the last `window` samples) - qd, causally."""
    rng = np.random.default_rng(0)
    A = np.round(40 + rng.normal(0, 1, (2, 120)))
    w, qd = 15, 1.0
    rain, base = bl.dynamic_baseline(A, w, qd)
    for t in range(A.shape[1]):
        lo = max(0, t - w + 1)
        assert np.allclose(rain[:, t], A[:, t] - A[:, lo:t + 1].min(axis=1) - qd)
    # causal: changing the future leaves the past untouched
    B = A.copy()
    B[:, 80:] -= 30
    assert np.array_equal(bl.dynamic_baseline(B, w, qd)[0][:, :80], rain[:, :80])


def test_std_wet_dry_matches_pynncml_and_is_centred():
    """Population std over a window placed (nearly) centrally; wet iff sigma > threshold."""
    rng = np.random.default_rng(1)
    A = 40 + rng.normal(0, 0.3, (1, 300))
    A[0, 120:170] += rng.normal(0, 3.0, 50)
    for window in (21, 20):
        wet, sig = bl.std_wet_dry(A, window, 1.0)
        lead = (window - 1) // 2
        s = pd.Series(A[0])
        # sigma at i is the std of A[i - lead .. i - lead + window - 1]
        ref = s.rolling(window).std(ddof=0).shift(-(window - 1 - lead)).to_numpy()
        inner = slice(lead, A.shape[1] - (window - 1 - lead))
        assert np.allclose(sig[0, inner], ref[inner])
        assert np.array_equal(wet[0, inner], (ref[inner] > 1.0).astype(float))
        if window % 2:                       # odd windows are exactly pandas' centred window
            cen = s.rolling(window, center=True).std(ddof=0).to_numpy()
            assert np.allclose(sig[0, inner], cen[inner])

    torch = pytest.importorskip("torch")
    pnc = pytest.importorskip("pynncml")
    wet_ref, sig_ref = pnc.scm.wet_dry.STDWetDry(1.0, 20)(torch.tensor(A, dtype=torch.float32))
    wet, sig = bl.std_wet_dry(A, 20, 1.0)
    assert np.allclose(sig, sig_ref.numpy(), atol=1e-4)
    assert np.array_equal(wet, wet_ref.numpy())


def test_constant_baseline_chain_matches_pynncml_two_step():
    """ConstantBaselineSTD's numpy chain = PyNNcml's two-step constant model (wa 3 dB)."""
    torch = pytest.importorskip("torch")
    pnc = pytest.importorskip("pynncml")
    rng = np.random.default_rng(2)
    att = 40 + np.round(rng.normal(0, 0.3, 1000))
    att[400:520] += 8 + 4 * rng.normal(0, 1, 120)
    wet, _ = bl.std_wet_dry(att[None], 240, 1.0)
    base = bl.constant_baseline(att[None], wet)
    ours = rain_from_attenuation(att[None] - base - 3.0, 2.0, 38.0, "v", table="ITU_2003", r_min=0.5)
    m = pnc.scm.rain_estimation.two_step_constant_baseline(
        pnc.scm.power_law.PowerLawType.INSTANCE, 0.5, 240, 1.0, wa_factor=3.0)
    ref, ref_wet, ref_bl = m(torch.tensor(att[None], dtype=torch.float32),
                             pnc.datasets.MetaData(38.0, True, 2.0, 10, 10))
    assert np.array_equal(wet, ref_wet.numpy())
    assert np.allclose(base, ref_bl.numpy(), atol=1e-4)
    assert np.allclose(ours, ref.numpy(), atol=1e-3)


def test_constant_baseline_treats_nan_wet_as_wet():
    A = np.array([[1.0, 2.0, 7.0, 8.0, 3.0]])
    W = np.array([[0, 0, np.nan, 1, 0]])
    assert np.array_equal(bl.constant_baseline(A, W), [[1, 2, 2, 2, 3]])


# -------------------------------------------------------------- preprocessing


def _linkset(rsl, tsl=None, start="2024-01-01 00:00", freq="1min"):
    rsl = np.atleast_2d(np.asarray(rsl, float))
    t = pd.date_range(start, periods=rsl.shape[1], freq=freq)
    n = rsl.shape[0]
    ds = xr.Dataset({"rsl": (("link", "time"), rsl)},
                    coords={"link": [f"l{i}" for i in range(n)], "time": t,
                            "length": ("link", np.full(n, 2.0)),
                            "frequency": ("link", np.full(n, 38.0)),
                            "polarization": ("link", np.array(["v"] * n)),
                            "mid_lat": ("link", np.full(n, 40.7)),
                            "mid_lon": ("link", np.full(n, -74.0))})
    if tsl is not None:
        ds["tsl"] = (("link", "time"), np.atleast_2d(np.asarray(tsl, float)))
    return ds


def test_total_loss_is_tsl_minus_rsl_in_db():
    ds = _linkset([[-50.0, -52.0, -55.0]], tsl=[[10.0, np.nan, 12.0]])
    tl = pp.total_loss(ds)
    assert np.allclose(tl.values, [[60.0, 62.0, 67.0]])        # missing TSL bridged from before
    assert np.allclose(pp.total_loss(_linkset([[-50.0, -52.0]]), tsl=3.0).values, [[53.0, 55.0]])


def test_min_max_is_interval_ending_and_right_closed():
    rsl = -50.0 - np.arange(31.0)                      # 00:00 .. 00:30, falling 1 dB/min
    out = pp.min_max(_linkset([rsl]), "15min", interpolate_gap=None)
    t = pd.DatetimeIndex(out.time.values)
    # the 00:15 stamp covers (00:00, 00:15]: samples 1..15
    i = t.get_loc(pd.Timestamp("2024-01-01 00:15"))
    assert out.rsl_max.values[0, i] == -51.0 and out.rsl_min.values[0, i] == -65.0
    assert t[0] == pd.Timestamp("2024-01-01 00:00")      # sample 0 alone ends at 00:00


def test_gauge_wet_reference_bin_start_interpolation():
    """RISK, pinned: the 15-min gauge mean is placed at its bin START and interpolated.

    So a shower in [00:15, 00:30) ramps up from 00:00 and back down by 00:30, and
    minutes after the last bin's start are always 0 (``right=0``). This mirrors
    implementation_1; it leads true rain by up to one bin.
    """
    ds = _linkset(np.full((1, 61), -50.0))
    t5 = pd.date_range("2024-01-01 00:00", "2024-01-01 01:00", freq="5min")
    rain = np.where((t5 >= "2024-01-01 00:15") & (t5 < "2024-01-01 00:30"), 1.0, 0.0)
    pws = xr.Dataset({"rain": (("station", "time"), rain[None])},
                     coords={"station": ["s"], "time": t5, "lat": ("station", [40.7]),
                             "lon": ("station", [-74.0])})
    ref = pp.gauge_wet_reference(ds, pws).values[0]
    assert ref[15] == 1.0 and ref[30] == 0.0
    assert 0 < ref[7] < 1                                 # wet before the shower started
    assert ref[22] == pytest.approx(8 / 15)               # drying while it still rains


def test_fill_gaps_gauge_gated_only_fills_wet_gaps():
    att = xr.DataArray([[1.0, np.nan, 5.0, np.nan]], dims=("link", "time"))
    ref = xr.DataArray([[0.0, 1.0, 1.0, 0.0]], dims=("link", "time"))
    out = pp.fill_gaps_gauge_gated(att, ref, stat="max").values[0]
    assert out[1] == 5.0 and np.isnan(out[3])


# ---------------------------------------------------------------------- QC


def test_metadata_qc_flags_reversed_duplicate():
    from core.cml.link_qc import metadata_qc

    rows = {"a": (40.70, -74.00, 40.72, -73.98, 2.7, 23.0, 1.0),
            "b": (40.72, -73.98, 40.70, -74.00, 2.7, 23.1, 0.9),    # same path, other direction
            "c": (40.70, -74.00, 40.72, -73.98, 2.7, 80.0, 1.0)}    # same path, other band
    table = pd.DataFrame.from_dict(rows, orient="index", columns=[
        "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon", "length", "frequency", "availability"])
    kept, log = metadata_qc(table)
    assert kept == ["a", "c"] and log.link.tolist() == ["b"]


def test_metadata_qc_duplicate_of_a_falsy_label():
    """A duplicate of the link labelled 0 is still found (the check was ``if dup_of:``)."""
    from core.cml.link_qc import metadata_qc

    table = pd.DataFrame([(40.70, -74.00, 40.72, -73.98, 2.7, 23.0),
                          (40.72, -73.98, 40.70, -74.00, 2.7, 23.0)],
                         columns=["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
                                  "length", "frequency"])
    kept, log = metadata_qc(table)
    assert kept == ["0"] and log.link.tolist() == [1]


def test_retrieval_qc_totals_use_the_sampling_interval():
    from core.cml.link_qc import QCConfig, retrieval_qc

    t = pd.date_range("2024-01-01", periods=8, freq="15min")
    rain = np.zeros((4, 8))
    rain[0] = 4.0                                        # 4 mm/h for 2 h = 8 mm
    da = xr.DataArray(rain, dims=("link", "time"),
                      coords={"link": list("abcd"), "time": t,
                              "mid_lat": ("link", [40.70, 40.701, 40.702, 40.703]),
                              "mid_lon": ("link", [-74.0] * 4)})
    log = retrieval_qc(da, QCConfig(wet_alone_mm=7.9))
    assert log.loc[log.link == "a", "value"].item() == pytest.approx(8.0)
    assert QCConfig(wet_alone_mm=8.1) and retrieval_qc(da, QCConfig(wet_alone_mm=8.1)).empty


# --------------------------------------------------------------- RNN features


def test_rnn_features_are_hour_ending_and_causal():
    from core.cml.rnn import N_MIN, features

    n = 2 * 1440 + 60
    rsl = np.full((1, n), -50.0)
    T = pd.Timestamp("2024-01-02 12:00")
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    p = t.get_loc(T)
    rsl[0, p - 4:p + 1] = -56.0                            # 6 dB over the last 5 minutes
    ds = _linkset(rsl)
    X = features(ds, pd.DatetimeIndex([T]))[0, 0]
    assert np.allclose(X[:N_MIN - 5], 0.0) and np.allclose(X[N_MIN - 5:N_MIN], 6.0)
    assert X[N_MIN] == 0.0 and X[N_MIN + 2] == pytest.approx(6.0)
    # the baseline must not see the future: a later shift changes nothing
    rsl2 = rsl.copy()
    rsl2[0, p + 1:] -= 20.0
    assert np.array_equal(features(_linkset(rsl2), pd.DatetimeIndex([T]))[0, 0], X)


def test_rnn_metadata_uses_itu_2005():
    from core.cml.rnn import metadata

    M = metadata(_linkset(np.full((1, 3), -50.0)))
    a, b = itu_ab(38.0, "v", "ITU_2005")
    assert M[0, 3] == pytest.approx(np.log10(a[0])) and M[0, 4] == pytest.approx(b[0])


# ------------------------------------------------- core/opensense/retrieval WAA


def _waa(model, a_obs, f, pol, L, **cfg):
    k, alpha = get_k_alpha(f, pol)
    c = rt.RetrievalConfig(waa_model=model, **cfg)
    w = rt.wet_antenna_attenuation(np.asarray(a_obs, float)[:, None], np.array([k]), np.array([alpha]),
                                   np.array([L]), np.array([float(f)]), c)[:, 0]
    return w, k, alpha


@pytest.mark.parametrize("model", ["pastorek2021", "leijnse2008"])
def test_lookup_waa_matches_pycomlink_from_A_obs(model):
    """At a tabulated frequency both use the same k-R law, so the WAA must agree."""
    waa = pytest.importorskip("pycomlink.processing.wet_antenna")
    A_obs = np.array([0.0, 0.2, 1.0, 3.0, 8.0, 20.0])
    ours, _, _ = _waa(model, A_obs, 23.0, "vertical", 3.0)
    fn = waa.waa_pastorek_2021_from_A_obs if model == "pastorek2021" else waa.waa_leijnse_2008_from_A_obs
    ref = fn(A_obs=A_obs, f_Hz=23e9, pol="V", L_km=3.0)
    assert np.allclose(ours, ref, rtol=5e-3, atol=1e-3)


@pytest.mark.parametrize("model", ["saturating", "pastorek2021", "leijnse2008"])
def test_waa_inversion_is_consistent(model):
    """A_obs = A_rain + WAA(R(A_rain)) holds for the returned WAA (38 GHz, 2 km)."""
    pytest.importorskip("pycomlink")
    from pycomlink.processing import wet_antenna as pw

    R_true = np.array([0.5, 2.0, 10.0, 40.0])
    k, alpha = get_k_alpha(38.0, "vertical")
    cfg = rt.RetrievalConfig()
    fwd = {"saturating": lambda r: cfg.waa_max_db * (1 - np.exp(-cfg.waa_rate_per_mm_h * r)),
           "pastorek2021": lambda r: pw.waa_pastorek_2021(R=r),
           "leijnse2008": lambda r: pw.waa_leijnse_2008(R=r, f_Hz=38e9)}[model]
    A_obs = k * R_true ** alpha * 2.0 + fwd(R_true)
    w, _, _ = _waa(model, A_obs, 38.0, "vertical", 2.0)
    R = ((A_obs - w) / (k * 2.0)) ** (1 / alpha)
    assert np.allclose(R, R_true, rtol=0.02)


@pytest.mark.xfail(strict=True, reason=(
    "BUG: the saturating wet-antenna model is inverted by 8 fixed-point steps that do not "
    "converge when W'(R) * dR/dA > 1, i.e. for links with k*L below ~0.15 (<= 15 GHz, or "
    "short 20-30 GHz links); light rain then oscillates and comes back as 0. Not fixed: "
    "it is the default retrieval and its waa parameters were calibrated with this solver"))
def test_saturating_waa_recovers_light_rain_on_low_k_links():
    cfg = rt.RetrievalConfig()
    R_true = np.array([0.5, 1.0, 2.0])
    for f, pol, L in [(15.0, "horizontal", 2.0), (28.0, "vertical", 0.6)]:
        k, alpha = get_k_alpha(f, pol)
        A_obs = k * R_true ** alpha * L + cfg.waa_max_db * (1 - np.exp(-cfg.waa_rate_per_mm_h * R_true))
        w, _, _ = _waa("saturating", A_obs, f, pol, L)
        R = (np.clip(A_obs - w, 0, None) / (k * L)) ** (1 / alpha)
        assert np.allclose(R, R_true, rtol=0.05), (f, L, R)


def test_retrieve_known_step_without_wet_antenna():
    """A clean 4 dB step over the baseline gives R = (4 / (k L))**(1/alpha)."""
    n = 2000
    loss = np.full((n, 1), 60.0)
    loss[1000:1060, 0] += 4.0
    cfg = rt.RetrievalConfig(waa_model="none", wet_window=30, baseline_window=600)
    out = rt.retrieve(loss, [3.0], [23.0], ["vertical"], cfg)
    k, alpha = get_k_alpha(23.0, "vertical")
    assert np.allclose(out["R"][1010:1050, 0], (4.0 / (k * 3.0)) ** (1 / alpha))
    assert np.all(out["R"][:900, 0] == 0)


def test_rolling_std_wet_dry_is_centred_sample_std():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (200, 1))
    wet = rt.wet_dry_rolling_std(a, 11, 0.9)[:, 0]
    ref = pd.Series(a[:, 0]).rolling(11, center=True, min_periods=2).std().to_numpy() > 0.9
    assert np.array_equal(wet, ref)
