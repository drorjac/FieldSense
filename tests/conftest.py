"""Synthetic OpenSense-1.0 datasets whose rain is known exactly.

Nothing here downloads anything: every test runs on a few kilobytes built in
memory, so the suite checks the code, not the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.itu_p838 import get_k_alpha

# Gothenburg-ish, so EPSG:32632 is a sensible projection.
LON0, LAT0 = 11.97, 57.70
CRS = "EPSG:32632"


def attenuation_for(rain_mm_h, freq_ghz, length_km, pol="vertical"):
    """ITU-R forward model: path attenuation (dB) of uniform rain."""
    k, alpha = get_k_alpha(freq_ghz, pol)
    return k * np.asarray(rain_mm_h, dtype=float) ** alpha * length_km


def make_cml(rain_by_link, freqs, lengths_km, interval_s=60, with_tsl=True,
             freq_dims=("sublink_id", "cml_id"), noise_db=0.0, seed=0):
    """A (time, sublink_id, cml_id) CML dataset carrying known rain.

    ``rain_by_link`` is (time, cml_id) in mm/h, the same on both sublinks.
    ``freqs`` is (cml_id, sublink_id) in GHz; it is *stored* in the order
    ``freq_dims`` and in MHz with a declared unit, the way the OpenSense
    files store it - which is what catches metadata-ordering bugs.
    """
    rain = np.asarray(rain_by_link, dtype=float)
    n_t, n_c = rain.shape
    freqs = np.asarray(freqs, dtype=float)
    n_s = freqs.shape[1]
    rng = np.random.default_rng(seed)

    base = 60.0 + 5.0 * np.arange(n_c)[None, :, None]           # dry loss, dB
    loss = np.repeat(base, n_t, axis=0) + np.zeros((n_t, n_c, n_s))
    for j in range(n_c):
        for s in range(n_s):
            loss[:, j, s] += attenuation_for(rain[:, j], freqs[j, s], lengths_km[j])
    loss += rng.normal(0.0, noise_db, loss.shape)

    tsl = np.full_like(loss, 10.0)
    rsl = tsl - loss
    time = pd.date_range("2020-06-01", periods=n_t, freq=f"{interval_s}s").as_unit("ns")

    # links laid out west-east, each 0.01 deg of longitude per km (roughly)
    lon0 = LON0 + 0.05 * np.arange(n_c)
    lon1 = lon0 + np.asarray(lengths_km) / 59.6
    ds = xr.Dataset(
        {"rsl": (("time", "cml_id", "sublink_id"), rsl)},
        coords={"time": time, "cml_id": np.arange(n_c) + 100,
                "sublink_id": [f"sublink_{i + 1}" for i in range(n_s)],
                "site_0_lon": ("cml_id", lon0),
                "site_0_lat": ("cml_id", np.full(n_c, LAT0)),
                "site_1_lon": ("cml_id", lon1),
                "site_1_lat": ("cml_id", np.full(n_c, LAT0)),
                "length": ("cml_id", np.asarray(lengths_km) * 1000.0,
                           {"units": "m"}),
                "polarization": (("cml_id", "sublink_id"),
                                 np.full((n_c, n_s), "v"))})
    stored = freqs * 1000.0 if freq_dims == ("cml_id", "sublink_id") \
        else freqs.T * 1000.0
    ds.coords["frequency"] = (freq_dims, stored, {"units": "MHz"})
    if with_tsl:
        ds["tsl"] = (("time", "cml_id", "sublink_id"), tsl)
    return ds.transpose("time", "sublink_id", "cml_id")


def rain_event(n_t=1440, start=600, end=720, rate=10.0, n_links=1):
    """A day at 1 min: dry, then two hours of steady rain, then dry."""
    r = np.zeros((n_t, n_links))
    r[start:end, :] = rate
    return r


@pytest.fixture
def steady_event():
    """Two links, two sublinks each at different frequencies, 2 h of 10 mm/h."""
    rain = rain_event(n_links=2)
    freqs = np.array([[18.0, 23.0], [32.0, 38.0]])
    return rain, freqs, np.array([4.0, 6.0])
