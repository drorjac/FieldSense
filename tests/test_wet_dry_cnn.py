"""wet_dry.cnn: window alignment, channels, masks - with a stand-in network.

The real model is downloaded on first use; these tests pass a tiny network
that returns one input value, so they check the plumbing exactly and offline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_cml, rain_event

from core.opensense import wet_dry


class Probe(torch.nn.Module):
    """P(wet) = the window's normalized loss at the target minute, squashed."""

    def forward(self, x):                       # x: (batch, 180, 2)
        assert x.shape[1:] == (wet_dry.CNN_WINDOW, 2)
        return torch.sigmoid(x[:, wet_dry.CNN_TARGET, :1] - 0.5)


@pytest.fixture
def event():
    return make_cml(rain_event(n_t=600, start=300, end=360, n_links=3), freqs=[[23, 25]] * 3,
                    lengths_km=[3.0, 4.0, 5.0], interval_s=60)


def test_prediction_belongs_to_the_minute_the_window_targets(event):
    p = wet_dry.cnn(event, threshold=None, net=Probe())
    assert p.dims == ("time", "cml_id") and p.sizes["time"] == event.sizes["time"]
    assert p.isel(time=slice(0, wet_dry.CNN_TARGET)).isnull().all()      # no full window yet
    assert p.isel(time=slice(-(wet_dry.CNN_WINDOW - wet_dry.CNN_TARGET - 1), None)).isnull().all()
    wet = (p > 0.5).any("cml_id")
    minutes = np.flatnonzero(wet.values)
    assert minutes.min() == 300 and minutes.max() == 359                  # exactly the rain


def test_threshold_gives_a_mask_and_small_chunks_change_nothing(event):
    full = wet_dry.cnn(event, threshold=None, net=Probe())
    chunked = wet_dry.cnn(event, threshold=None, net=Probe(), links_per_chunk=1, batch_size=7)
    np.testing.assert_allclose(full, chunked)
    mask = wet_dry.cnn(event, threshold=0.5, net=Probe())
    assert set(np.unique(mask.values[np.isfinite(mask.values)])) <= {0.0, 1.0}


def test_one_sublink_is_fed_twice_and_gaps_stay_nan(event):
    one = event.isel(sublink_id=[0])
    one["rsl"][{"time": slice(400, 420), "cml_id": 0}] = np.nan
    p = wet_dry.cnn(one, threshold=None, net=Probe())
    assert p.isel(time=slice(400, 420), cml_id=0).isnull().all()
    assert p.isel(time=slice(400, 420), cml_id=1).notnull().all()


def test_ten_second_data_maps_back_onto_its_own_axis():
    ds = make_cml(rain_event(n_t=6 * 400, start=6 * 250, end=6 * 300), freqs=[[23, 25]],
                  lengths_km=[3.0], interval_s=10)
    p = wet_dry.cnn(ds, threshold=0.5, net=Probe())
    assert p.sizes["time"] == ds.sizes["time"]
    wet_minutes = np.flatnonzero(p.isel(cml_id=0).values == 1.0) // 6
    assert wet_minutes.min() == 250 and wet_minutes.max() == 299
