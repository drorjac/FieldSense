"""
The processing chain of the OpenSense radar-adjustment intercomparison.

OpenSense's ``radar_adjustment_intercomparison`` repository (OpenSenseAction, 2025-26)
compares radar-CML merging methods of ``mergeplg`` on OpenMRG (JJA 2015) and OpenRainER
(JJA 2022). This module reproduces its data preparation (``1_data_preparation.ipynb``)
and CML processing (``2_cml_processing.ipynb``) step by step, on the same raw files, so
the merging study in ``projects/maps/radar_adjustment`` starts from the same inputs.

CML chain, per sublink at 1 min (:func:`preprocess`, :func:`process`):

1. total loss ``tl = tsl - rsl``;
2. spikes: where the two channels' total loss (linearly gap-filled) differ by more than
   30 dB, the higher channel is set to NaN;
3. link QC on the whole period, three flags from rolling standard deviations of ``tl``
   (diurnal cycle: 5-h std > 2 dB more than 10% of the time; noisy: 1-h std > 0.8 dB
   more than 35% of the time; flat: 1-h std > 0.8 dB less than 1% of the time). As in
   the notebook, ``Dataset.where(~flag, drop=True)`` drops a link only when both of its
   channels are flagged; a single flagged channel is set to NaN;
4. wet/dry from the radar along the path: a radar step with more than 0.01 mm, stamped
   at the end of its interval t, marks t - (rad_freq - 1) ... t + 5 min wet - the
   interval and 5 minutes after (:func:`wet_from_radar`, the notebook's
   ``binary_dilation`` structuring element and origin);
5. constant baseline over wet periods (pycomlink, mean of the last 5 dry minutes);
6. wet-antenna attenuation of Pastorek et al. (2021), ``A_max=6, zeta=0.7, d=0.15``;
7. ``A = tl - baseline - waa`` floored at 0, ``R`` from the ITU k-R relation;
8. hourly totals: mean of the 1-min depths ``R/60`` in each hour (label and closed right)
   times 60, then the mean of the two channels.

Radar and gauges (:func:`openmrg_radar_rate`, :func:`hourly_from_mean`):

* OpenMRG's SMHI composite is stored in dBZ. The notebook computes ``dBZ = 0.4*data - 30``
  but never uses it: the rain rate is ``(10**(data/10) / 200) ** (5/8)``, i.e. the stored
  values are taken as dBZ (correctly) with Z = 200 R^1.6, whereas the file's attributes
  give Z = 200 R^1.5. Kept as written; ``core.opensense.networks`` uses the attributes.
* Hourly radar and gauge totals are the mean of the sub-hourly depths in each hour
  (label and closed right) times the number of steps per hour, then (radar) values of
  0.01 mm or less set to 0. An hour with some missing steps is filled in by the mean.
* Every coordinate is projected to UTM 32N (EPSG:32632), as in the notebook, also for
  Gothenburg (zone 33); ``mergeplg`` only needs a metric plane.

The code mirrors the notebook's operations and order rather than ``core.opensense.
retrieval``; differences from FieldSense's own chain are deliberate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

UTM = "EPSG:32632"
SUBLINKS = ("channel1", "channel2")
RADAR_ZERO_MM = 0.01


# ---------------------------------------------------------------------------
# radar and gauges
# ---------------------------------------------------------------------------
def openmrg_radar_rate(data: xr.DataArray | np.ndarray):
    """Rain rate (mm/h) from the OpenMRG composite as the notebook computes it (Z = 200 R^1.6)."""
    return (10 ** (data / 10) / 200) ** (5 / 8)


def hourly_from_mean(da: xr.DataArray, steps_per_hour: int) -> xr.DataArray:
    """Sub-hourly depths -> hour-ending totals: mean in the hour (closed right) x steps."""
    return da.resample(time="1h", label="right", closed="right").mean(skipna=True) * steps_per_hour


def threshold_radar(da: xr.DataArray, zero: float = RADAR_ZERO_MM) -> xr.DataArray:
    """Values of ``zero`` or less set to 0 (NaN too, as ``xr.where`` does in the notebook)."""
    return xr.where(da > zero, da, 0)


def project(lon, lat):
    """lon/lat -> UTM 32N x/y (m)."""
    import poligrain as plg
    return plg.spatial.project_point_coordinates(lon, lat, UTM)


def add_link_xy(ds: xr.Dataset) -> xr.Dataset:
    """Site and midpoint coordinates in UTM 32N, as ``mergeplg`` expects them."""
    ds = ds.copy()
    ds.coords["site_0_x"], ds.coords["site_0_y"] = project(ds.site_0_lon, ds.site_0_lat)
    ds.coords["site_1_x"], ds.coords["site_1_y"] = project(ds.site_1_lon, ds.site_1_lat)
    ds.coords["x"] = (ds.site_0_x + ds.site_1_x) / 2
    ds.coords["y"] = (ds.site_0_y + ds.site_1_y) / 2
    return ds


def radar_along_links(rain: xr.DataArray, ds_cml: xr.Dataset, lon2d, lat2d) -> xr.DataArray:
    """Radar averaged along each path with poligrain's intersect weights on the lon/lat grid."""
    from core.maps.geometry import path_average_intersect
    return path_average_intersect(rain, ds_cml, plane="lonlat", lon2d=lon2d, lat2d=lat2d)


# ---------------------------------------------------------------------------
# CML chain
# ---------------------------------------------------------------------------
def _flags(tl: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    std5h = tl.rolling(time=60 * 5, center=True).std()
    std1h = tl.rolling(time=60, center=True).std()
    diurnal = (std5h > 2).mean(dim="time") > 0.1
    noisy = (std1h > 0.8).mean(dim="time") > 0.35
    flat = (std1h > 0.8).mean(dim="time") < 0.01
    return diurnal, noisy, flat


def preprocess(ds_input: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Total loss, channel-difference spike filter and link QC (``cml_preprocessing``).

    ``ds_input``: ``tsl``, ``rsl`` (cml_id, sublink_id, time) at 1 min with two sublinks.
    Returns the kept links and those removed.
    """
    ds = ds_input.transpose("cml_id", "sublink_id", "time").copy()
    ds["tl"] = ds.tsl - ds.rsl
    interp = ds.tl.interpolate_na(dim="time", method="linear", max_gap=None)
    diff = interp.diff(dim="sublink_id")                     # channel2 - channel1
    mask = np.concatenate([diff.values < -30, diff.values > 30], axis=1)
    ds["tl"] = xr.where(~mask, ds.tl, np.nan)
    diurnal, noisy, flat = _flags(ds.tl)
    removed = ds.where(diurnal | noisy | flat, drop=True)
    ds = ds.where(~diurnal, drop=True)
    ds = ds.where(~noisy, drop=True)
    ds = ds.where(~flat, drop=True)
    return ds, removed


def wet_from_radar(ds: xr.Dataset, radar_along: xr.DataArray, rad_freq: int) -> xr.DataArray:
    """Wet minutes from the radar along each path (``wetdry_radarbased``).

    ``radar_along``: depth per radar step (cml_id, time), stamped as in the source.
    A stamp with more than 0.01 mm is wet; the mask is dilated over ``rad_freq + 5``
    minutes with the notebook's structuring element and origin, then copied to both
    channels. Returns ``wet`` (cml_id, sublink_id, time) on the minutes of ``ds``.
    """
    from scipy.ndimage import binary_dilation

    timeseq = pd.date_range(ds.time.values[0], ds.time.values[-1], freq="min")
    ds = ds.reindex({"time": timeseq})
    rad = radar_along.transpose("cml_id", "time").reindex({"time": timeseq})
    rad = rad.reindex({"cml_id": ds.cml_id})
    wet = xr.zeros_like(ds["tl"], dtype=bool)
    mask = (rad > 0.01).astype("int").values
    n = rad_freq + 5
    expanded = np.array([binary_dilation(row, structure=np.ones(n), origin=int(n / 2 - 6))
                         for row in mask])
    wet[:] = np.broadcast_to(expanded.astype(bool)[:, None, :], wet.shape)
    return wet


def process(ds_input: xr.Dataset, radar_along: xr.DataArray, rad_freq: int) -> xr.Dataset:
    """Wet/dry, baseline, wet antenna and rain rate (``cml_processing``).

    ``ds_input`` is the output of :func:`preprocess`, with ``frequency`` in MHz and
    ``length`` in m (the OpenSense data format), ``polarization`` "v"/"h".
    """
    import pycomlink as pycml

    ds = ds_input.copy()
    ds["wet"] = wet_from_radar(ds, radar_along, rad_freq)
    ds["baseline"] = pycml.processing.baseline.baseline_constant(
        trsl=ds.tl, wet=ds.wet, n_average_last_dry=5)
    ds["A_obs"] = ds.tl - ds.baseline
    ds["A_obs"] = ds.A_obs.where(ds.A_obs >= 0, 0)
    ds["waa"] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
        A_obs=ds.A_obs, f_Hz=ds.frequency * 1e6, pol=ds.polarization.data,
        L_km=ds.length / 1000, A_max=6, zeta=0.7, d=0.15)
    ds["A"] = ds.tl - ds.baseline - ds.waa
    ds["A"].data[ds.A < 0] = 0
    ds["R"] = pycml.processing.k_R_relation.calc_R_from_A(
        A=ds.A, L_km=ds.length.astype(float) / 1000, f_GHz=ds.frequency / 1000,
        pol=ds.polarization)
    ds["R_acc"] = ds["R"] / 60
    return ds


def hourly_links(ds: xr.Dataset) -> xr.DataArray:
    """1-min depths -> hour-ending totals, then the mean of the channels."""
    h = ds.R_acc.resample(time="1h", label="right", closed="right").mean(skipna=True) * 60
    return h.reduce(np.nanmean, dim="sublink_id")


def chain(ds_cml: xr.Dataset, radar_along: xr.DataArray, rad_freq: int) -> tuple[xr.DataArray, list]:
    """The whole chain: raw 1-min TSL/RSL -> hourly link totals (mm) and the removed ids."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kept, removed = preprocess(ds_cml)
        out = process(kept, radar_along, rad_freq)
        hourly = hourly_links(out)
    return hourly, list(removed.cml_id.values)
