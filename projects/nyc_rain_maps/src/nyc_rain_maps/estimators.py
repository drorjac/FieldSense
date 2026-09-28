"""Rain-rate estimators: one class per method used in implementation_1 / implementation_2.

Every estimator has the same interface::

    est = DynamicBaseline(window=200)
    out = est.estimate(links, pws=pws)      # links: link set from core.opensense.openmesh
    out["rain"]                             # (link, time) path-averaged rain rate, mm/h

and returns an ``xarray.Dataset`` with ``rain`` plus method diagnostics
(``attenuation``, ``baseline``, ``wet`` where meaningful). Time stamps are the input's
1-min stamps (instantaneous rate) unless the method aggregates (then interval-ENDING).

Gap handling (``gap_fill``) is a parameter of every estimator, because it is the main
practical difference between the two implementations - see
:mod:`nyc_rain_maps.preprocess` and ``docs/METHODS.md``:

* ``"none"`` - gaps stay NaN (rain is NaN there);
* ``"min_rsl"`` - implementation_2 (all gaps -> deepest fade);
* ``"gauge_q99"`` / ``"gauge_max"`` - implementation_1 (only gaps while PWS gauges are
  wet -> 99th percentile / max attenuation; needs ``pws=``).
"""

from __future__ import annotations

import contextlib
import io
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from . import baseline as bl
from . import preprocess as pp
from .power_law import rain_from_attenuation

GAP_FILLS = ("none", "min_rsl", "gauge_q99", "gauge_max")


def _link_arrays(links: xr.Dataset):
    return (links.length.values.astype(float), links.frequency.values.astype(float),
            links.polarization.values.astype(str))


def _dataset(links: xr.Dataset, rain: np.ndarray, time=None, **extra) -> xr.Dataset:
    time = links.time.values if time is None else time
    coords = {"link": links.link.values, "time": time}
    coords.update({c: ("link", links[c].values) for c in links.coords
                   if links[c].dims == ("link",) and c != "link"})
    data = {"rain": (("link", "time"), np.asarray(rain, dtype="float32"))}
    for k, v in extra.items():
        if v is not None:
            data[k] = (("link", "time"), np.asarray(v, dtype="float32"))
    ds = xr.Dataset(data, coords=coords)
    ds["rain"].attrs = {"units": "mm/h", "long_name": "path-averaged rain rate"}
    return ds


@dataclass
class RainEstimator(ABC):
    """Base class. Subclasses define ``name``/``implementation`` and ``_estimate``."""

    gap_fill: str = "none"
    tsl: float = 0.0

    name = "base"
    implementation = "common"

    def estimate(self, links: xr.Dataset, pws: xr.Dataset | None = None) -> xr.Dataset:
        if self.gap_fill not in GAP_FILLS:
            raise ValueError(f"gap_fill must be one of {GAP_FILLS}")
        att = self._attenuation(links, pws)
        out = self._estimate(links, att)
        out.attrs.update({"method": self.name, "implementation": self.implementation,
                          "params": repr(self.params())})
        return out

    def params(self) -> dict:
        return asdict(self)

    def _attenuation(self, links: xr.Dataset, pws: xr.Dataset | None) -> xr.DataArray:
        """Total loss after the requested gap handling."""
        if self.gap_fill == "min_rsl":
            links = pp.fill_gaps_min_rsl(links)
        att = pp.total_loss(links, self.tsl)
        if self.gap_fill.startswith("gauge"):
            if pws is None:
                raise ValueError(f"gap_fill={self.gap_fill!r} needs pws= (PWS gauges)")
            ref = pp.gauge_wet_reference(links, pws)
            att = pp.fill_gaps_gauge_gated(att, ref, stat=self.gap_fill.split("_")[1])
        return att

    @abstractmethod
    def _estimate(self, links: xr.Dataset, att: xr.DataArray) -> xr.Dataset: ...


# --------------------------------------------------------------- PyNNcml-style


@dataclass
class DynamicBaseline(RainEstimator):
    """One-step dynamic baseline + ITU power law on 1-min instantaneous attenuation.

    implementation_1 (Gabriela): ``DynamicBaseline(gap_fill="gauge_q99", nan_to_zero=True)``.
    implementation_2 ``pynncml_classical`` dynamic: ``DynamicBaseline(gap_fill="min_rsl")``.
    Both: window 200 min, quantization delta 1 dB, r_min 0.5 mm/h, ITU 2003 table.
    """

    window: int = 200
    quantization_delta: float = 1.0
    r_min: float = 0.5
    table: str = "ITU_2003"
    nan_to_zero: bool = False
    skipna_baseline: bool = False     # True: a gap does not blank the next `window` minutes

    name = "dynamic_baseline"

    def _estimate(self, links, att):
        A = att.values
        A_rain, base = bl.dynamic_baseline(A, self.window, self.quantization_delta,
                                           skipna=self.skipna_baseline)
        L, f, pol = _link_arrays(links)
        R = rain_from_attenuation(A_rain, L, f, pol, table=self.table, r_min=self.r_min)
        if self.nan_to_zero:
            R = np.nan_to_num(R, nan=0.0)
        return _dataset(links, R, attenuation=A_rain, baseline=base)


@dataclass
class ConstantBaselineSTD(RainEstimator):
    """Two-step: rolling-std wet/dry -> constant baseline -> fixed wet-antenna offset.

    implementation_2 ``pynncml_classical`` constant: window 240, threshold 1 dB,
    wa_factor 3 dB, r_min 0.5, ITU 2003, ``gap_fill="min_rsl"``.
    """

    window: int = 240
    threshold: float = 1.0
    wa_factor: float = 3.0
    r_min: float = 0.5
    table: str = "ITU_2003"

    name = "constant_baseline_std"

    def _estimate(self, links, att):
        A = att.values
        wet, _ = bl.std_wet_dry(A, self.window, self.threshold)
        base = bl.constant_baseline(A, wet)
        A_rain = A - base - self.wa_factor
        L, f, pol = _link_arrays(links)
        R = rain_from_attenuation(A_rain, L, f, pol, table=self.table, r_min=self.r_min)
        return _dataset(links, R, attenuation=A_rain, baseline=base, wet=wet)


# ------------------------------------------------------------------- manual


@dataclass
class ManualWindows(RainEstimator):
    """Manual method (implementation_2 ``manual.ipynb``).

    Baseline = median total loss over a hand-picked dry window; A = clip(TL - baseline,
    0) inside the hand-picked rain window, zero outside; ITU 2005, r_min 0.1, no WAA.
    ``dry`` and ``rain`` are ``(start, end)`` pairs.
    """

    dry: tuple = ("", "")
    rain: tuple = ("", "")
    r_min: float = 0.1
    table: str = "ITU_2005"

    name = "manual"

    def _estimate(self, links, att):
        t = pd.DatetimeIndex(links.time.values)
        dry = (t >= pd.Timestamp(self.dry[0])) & (t <= pd.Timestamp(self.dry[1]))
        wet = (t >= pd.Timestamp(self.rain[0])) & (t <= pd.Timestamp(self.rain[1]))
        if not dry.any() or not wet.any():
            raise ValueError("dry/rain windows do not overlap the link time axis")
        TL = att.values
        base = bl.median_dry_baseline(TL, np.broadcast_to(dry, TL.shape))[:, None]
        A = np.where(wet, np.clip(TL - base, 0, None), 0.0)
        L, f, pol = _link_arrays(links)
        R = rain_from_attenuation(A, L, f, pol, table=self.table, r_min=self.r_min)
        return _dataset(links, R, attenuation=A, baseline=np.broadcast_to(base, TL.shape),
                        wet=np.broadcast_to(wet, TL.shape).astype(float))


# ---------------------------------------------------------------- pycomlink


@dataclass
class PycomlinkRSD(RainEstimator):
    """pycomlink: rolling-std wet/dry -> constant|linear baseline -> WAA -> ITU 2005.

    implementation_2 ``pycomlink_basic``: window 240 min (centred), wet if the rolling
    std exceeds the per-link ``quantile`` (0.90) of the rolling std over a *reference
    record* (the full 8-month record, raw, short gaps interpolated), OR the sample was a
    gap; baseline constant with ``n_average_last_dry=5``; Leijnse 2008 WAA; r_min 0.1.

    Pass the long record as ``threshold_links`` (else the event window itself is used,
    which makes the threshold event-dependent).
    """

    window: int = 240
    quantile: float = 0.90
    baseline: str = "constant"          # "constant" | "linear"
    waa: str | None = "leijnse"         # "leijnse" | "pastorek" | None
    n_average_last_dry: int = 5
    r_min: float = 0.1
    table: str = "ITU_2005"
    gap_fill: str = "min_rsl"
    threshold_links: xr.Dataset | None = field(default=None, repr=False)
    threshold_values: np.ndarray | None = field(default=None, repr=False)   # precomputed thresholds()

    name = "pycomlink_rsd"
    implementation = "implementation_2"

    def params(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("threshold_links", "threshold_values")}
        d["threshold_record"] = (None if self.threshold_links is None else
                                 f"{self.threshold_links.attrs.get('start')}..{self.threshold_links.attrs.get('end')}")
        return d

    def thresholds(self, links: xr.Dataset) -> np.ndarray:
        if self.threshold_values is not None:
            return np.asarray(self.threshold_values)
        ref = self.threshold_links if self.threshold_links is not None else links
        ref = ref.sel(link=links.link.values)
        trsl = (-ref["rsl"]).interpolate_na("time", method="linear", max_gap=pd.Timedelta("5min"))
        rsd = trsl.rolling(time=self.window, center=True, min_periods=1).std()
        return rsd.quantile(self.quantile, dim="time", skipna=True).values

    def _estimate(self, links, att):
        import pycomlink as pycml

        gaps = links["rsl"].isnull().values
        thr = self.thresholds(links)[:, None]
        rsd = att.rolling(time=self.window, center=True, min_periods=1).std().values
        wet = (rsd > thr) | gaps
        trsl = att.values
        L, f, pol = _link_arrays(links)

        A_rain, base_all, waa_all = [], [], []
        for i in range(trsl.shape[0]):
            if self.baseline == "constant":
                base = pycml.processing.baseline.baseline_constant(
                    trsl=trsl[i], wet=wet[i], n_average_last_dry=self.n_average_last_dry)
            elif self.baseline == "linear":
                base = pycml.processing.baseline.baseline_linear(rsl=trsl[i], wet=wet[i])
            else:
                raise ValueError("baseline must be 'constant' or 'linear'")
            A_obs = np.clip(trsl[i] - base, 0, None)
            A_obs_safe = np.nan_to_num(A_obs, nan=0.0)
            if self.waa == "leijnse":
                w = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
                    A_obs=A_obs_safe, f_Hz=f[i] * 1e9, pol=pol[i].upper(), L_km=L[i])
            elif self.waa == "pastorek":
                w = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
                    A_obs=A_obs_safe, f_Hz=f[i] * 1e9, pol=pol[i].upper(), L_km=L[i])
            elif self.waa is None:
                w = np.zeros_like(A_obs_safe)
            else:
                raise ValueError("waa must be 'leijnse', 'pastorek' or None")
            w = np.where(np.isnan(A_obs), np.nan, w)
            A_rain.append(np.clip(A_obs - w, 0, None))
            base_all.append(base)
            waa_all.append(w)
        A_rain = np.array(A_rain)
        R = rain_from_attenuation(A_rain, L, f, pol, table=self.table, r_min=self.r_min)
        return _dataset(links, R, attenuation=A_rain, baseline=np.array(base_all),
                        wet=wet.astype(float), waa=np.array(waa_all))


@dataclass
class NearbyLinks(RainEstimator):
    """Nearby-link approach (Overeem et al. 2016) via pycomlink, on 15-min min/max RSL.

    implementation_2 ``pycomlink_basic`` section 3: radius 15 km, median thresholds
    -1.4 / -0.7 dB (and dB/km), >= 3 links, 24 h period, >= 6 h; reference level over
    96 dry intervals (24 h); WAA_max 2.3 dB, alpha 0.33; F-threshold -32.5; ITU 2005.
    Needs a long record (days) and several links to work; the result is cut back to
    ``[start, end]`` if given. Output time stamps are interval-ENDING (a fix of the
    left-labelled stamps in implementation_2).
    """

    radius_km: float = 15.0
    thresh_median_P: float = -1.4
    thresh_median_PL: float = -0.7
    min_links: int = 3
    interval_min: int = 15
    timeperiod_h: int = 24
    min_hours: int = 6
    n_average_dry: int = 96
    min_periods: int = 10
    waa_max: float = 2.3
    alpha: float = 0.33
    F_value_threshold: float = -32.5
    table: str = "ITU_2005"
    start: str | None = None
    end: str | None = None

    name = "nearby_links"
    implementation = "implementation_2"

    def _attenuation(self, links, pws):
        return pp.total_loss(links, self.tsl)          # works on RSL directly

    def _estimate(self, links, att):
        from pycomlink.processing import nearby_rain_retrival as nr
        from pycomlink.processing.wet_dry import nearby_wetdry as nw

        ids = [str(x) for x in links.link.values]
        rsl = links["rsl"].interpolate_na("time", method="linear", max_gap=pd.Timedelta("5min"))
        rsl = xr.DataArray(rsl.values, dims=("cml_id", "time"),
                           coords={"cml_id": ids, "time": links.time.values})
        # Same windows as implementation_2 (left-closed); stamps moved to interval END.
        step = pd.Timedelta(minutes=self.interval_min)
        freq = f"{self.interval_min}min"          # xarray < 2024.9 takes a string here
        pmin = rsl.resample(time=freq).min()
        pmax = rsl.resample(time=freq).max()
        length = xr.DataArray(links.length.values, dims="cml_id", coords={"cml_id": ids})
        pmin, pmax = pmin.assign_coords(length=length), pmax.assign_coords(length=length)
        quiet = contextlib.redirect_stderr(io.StringIO())      # pycomlink prints tqdm bars
        with quiet:
            dist = nw.calc_distance_between_cml_endpoints(
                cml_ids=ids, site_a_latitude=list(links.site_0_lat.values),
                site_a_longitude=list(links.site_0_lon.values),
                site_b_latitude=list(links.site_1_lat.values),
                site_b_longitude=list(links.site_1_lon.values))
            wet, F = nw.nearby_wetdry(
                pmin=pmin, ds_dist=dist, radius=self.radius_km, thresh_median_P=self.thresh_median_P,
                thresh_median_PL=self.thresh_median_PL, min_links=self.min_links,
                interval=self.interval_min, timeperiod=self.timeperiod_h, min_hours=self.min_hours)
            pref = nr.nearby_determine_reference_level(pmin, pmax, wet, n_average_dry=self.n_average_dry,
                                                       min_periods=self.min_periods)
            p_c_min, p_c_max = nr.nearby_correct_received_signals(pmin, pmax, wet, pref)
        f = xr.DataArray(links.frequency.values, dims="cml_id", coords={"cml_id": ids})
        pol = xr.DataArray(np.char.upper(links.polarization.values.astype(str)), dims="cml_id",
                           coords={"cml_id": ids})
        R = nr.nearby_rainfall_retrival(
            pref=pref, p_c_min=p_c_min, p_c_max=p_c_max, F=F, length=pmin.length, f_GHz=f, pol=pol,
            a=None, b=None, a_b_approximation=self.table, waa_max=self.waa_max, alpha=self.alpha,
            F_value_threshold=self.F_value_threshold).clip(min=0)
        R = R.transpose("cml_id", "time").assign_coords(time=R.time.values + step)
        wet = wet.transpose("cml_id", "time").assign_coords(time=wet.time.values + step)
        if self.start or self.end:
            R = R.sel(time=slice(self.start, self.end))
            wet = wet.sel(time=R.time)
        return _dataset(links, R.values, time=R.time.values, wet=wet.values.astype(float))


# ------------------------------------------------------------------ PyNNcml


@dataclass
class PyNNcmlGRU(RainEstimator):
    """PyNNcml pretrained two-step GRU on 15-min min/max RSL (implementation_2).

    Caveats (documented in docs/METHODS.md): the network was trained on OpenMRG
    links at 18-25 GHz, and its length normalisation suggests metres, while
    implementation_2 passes km (``length_unit`` reproduces either). Output is clipped at
    0 and stamped interval-ENDING.
    """

    length_unit: str = "km"
    constant_tsl: float = 10.0
    gap_fill: str = "min_rsl"

    name = "pynncml_gru"
    implementation = "implementation_2"

    def _estimate(self, links, att):
        import pynncml as pnc
        import torch

        nn_mod = pnc.neural_networks
        enum = getattr(nn_mod, "RNNType", None) or getattr(nn_mod, "DNNType")   # renamed across versions
        model = pnc.scm.rain_estimation.two_step_network(1, enum.GRU)
        model.eval()
        rsl = -att.values          # gap-filled RSL
        t_unix = (links.time.values.astype("datetime64[s]").astype(np.int64)).astype(float)
        L, f, pol = _link_arrays(links)
        rain, wet, stamps = [], [], None
        for i in range(rsl.shape[0]):
            meta = pnc.datasets.MetaData(
                frequency=float(f[i]), polarization=pol[i].lower() == "v",
                length=float(L[i] * (1000.0 if self.length_unit == "m" else 1.0)),
                height_far=10, height_near=10)
            link = pnc.datasets.Link(rsl[i].astype("float32"), t_unix, meta)
            mm = link.create_min_max_link(900)
            data = mm.as_tensor(constant_tsl=self.constant_tsl).unsqueeze(0)
            with torch.no_grad():
                res, _ = model(data, link.meta_data.as_tensor(), model.init_state())
            rain.append(res[0, :, 0].clip(min=0).numpy())
            wet.append(res[0, :, 1].numpy())
            if stamps is None:
                stamps = mm.time()
        n = min(len(r) for r in rain)
        # PyNNcml stamps each ~15-min window with its START (linspace, drifting seconds);
        # relabel as interval END on the 15-min grid, like every other product here.
        t = (pd.to_datetime(np.asarray(stamps[:n]), unit="s") + pd.Timedelta("15min")).round("15min")
        return _dataset(links, np.stack([r[:n] for r in rain]), time=t.values,
                        wet=np.stack([w[:n] for w in wet]))    # wet = GRU wet probability


ESTIMATORS = {cls.name: cls for cls in
              [DynamicBaseline, ConstantBaselineSTD, ManualWindows, PycomlinkRSD, NearbyLinks,
               PyNNcmlGRU]}
