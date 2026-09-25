"""
The CML retrieval chain, independent of where the data came from.

``ingest_openmrg`` and ``ingest_openrainer`` each used to carry their own copy
of this, which meant the chain could only be tested through a full ingest of a
4.6 GB archive. Pulled out here so the same code runs on the raw archives, on
the curated example subsets, and in ``validate_retrieval.py`` against the
reference retrieval OpenSense publishes.

The chain, per sublink:

1. total loss ``A = TSL - RSL``, or ``-RSL`` where no TSL is published
2. wet/dry classification by rolling standard deviation (Schleiss & Berne 2010)
3. dry-weather baseline, a long rolling median over samples classified dry
4. ``A_rain = A - baseline``, floored
5. wet-antenna attenuation removed by fixed-point iteration
6. ``R = (A_rain / (k L))**(1/alpha)`` with ITU-R P.838-3 coefficients
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.opensense import conventions as cv


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval parameters.

    ``wet_window`` and ``baseline_window`` are in **samples**, so they must be
    scaled to the sampling interval of the source: OpenMRG is 10 s, OpenRainER
    and OpenMesh are 1 min. Use :meth:`for_interval` rather than hardcoding.

    ``baseline_window`` and ``waa_max_db`` are calibration choices, not
    physics, and both were checked against the municipal gauges on the
    25 August OpenMRG event - see the project README for the table. A
    one-hour baseline absorbs a multi-hour event into its own dry reference;
    the Schleiss et al. (2013) 2.3 dB wet-antenna value removes most of the
    signal on a 2 km link at 23 GHz.
    """

    wet_window: int = 30
    wet_threshold_db: float = 0.8
    baseline_window: int = 1080
    min_attenuation_db: float = 0.1
    waa_max_db: float = 0.5
    waa_rate_per_mm_h: float = 0.28
    min_length_km: float = 0.5

    @classmethod
    def for_interval(cls, seconds: float, **overrides) -> "RetrievalConfig":
        """Config with the windows scaled to a sampling interval.

        Defaults are expressed for OpenMRG's 10 s sampling: a 5-minute
        wet/dry window and a 3-hour baseline.
        """
        per_5min = max(2, int(round(300.0 / seconds)))
        per_3h = max(4, int(round(3 * 3600.0 / seconds)))
        return cls(wet_window=per_5min, baseline_window=per_3h, **overrides)


def wet_dry_rolling_std(attenuation: np.ndarray, window: int,
                        threshold_db: float) -> np.ndarray:
    """Wet flag per sample from the rolling standard deviation of total loss.

    Rain makes the signal fluctuate, so a window whose spread exceeds a
    threshold is classified wet. Operates along axis 0 (time).
    """
    roll = pd.DataFrame(attenuation).rolling(
        window=window, center=True, min_periods=max(2, window // 4))
    return roll.std().to_numpy() > threshold_db


def baseline_from_dry(attenuation: np.ndarray, wet: np.ndarray,
                      window: int) -> np.ndarray:
    """Dry-weather reference level per sublink.

    A long rolling median over samples classified dry, carried across wet
    spells. A rolling median beats holding the single last dry value, which
    would inherit that sample's noise into the whole wet spell.
    """
    dry_only = pd.DataFrame(np.where(wet, np.nan, attenuation))
    base = (dry_only.rolling(window=window, center=True, min_periods=1)
            .median().ffill().bfill())

    all_wet = base.isna().all(axis=0)
    if all_wet.any():
        fallback = pd.DataFrame(attenuation).quantile(0.1)
        base.loc[:, all_wet] = fallback[all_wet].to_numpy()
    return base.to_numpy()


def total_loss_from(rsl: np.ndarray, tsl: np.ndarray | None = None) -> np.ndarray:
    """Path loss in dB from received, and transmitted power if it exists.

    With both channels the loss is ``TSL - RSL``, and a drift in transmit
    power cancels. OpenMesh publishes **RSL only**, so there ``-RSL`` is used
    instead, which assumes the transmitter holds a constant level.

    That assumption is not free. Anything that moves TSL - automatic transmit
    power control, a hardware change, a temperature drift in the amplifier -
    enters the retrieval as if it were rain. The saving grace is that the
    baseline step removes whatever part of it is slow; what survives is the
    fast component, and there is no way to separate that from real
    attenuation without the second channel.
    """
    rsl = np.asarray(rsl, dtype=float)
    if tsl is None:
        return -rsl
    return np.asarray(tsl, dtype=float) - rsl


def retrieve(total_loss: np.ndarray, length_km: np.ndarray,
             freq_ghz: np.ndarray, polarization,
             cfg: RetrievalConfig | None = None) -> dict:
    """Total path loss (time, link) -> rain rate (time, link) in mm/h.

    ``length_km``, ``freq_ghz`` and ``polarization`` are per-link, length
    matching ``total_loss.shape[1]``. Returns a dict with ``R``, ``A_rain``,
    ``baseline`` and ``wet`` so callers can inspect the intermediate steps.
    """
    cfg = cfg or RetrievalConfig()

    loss = np.asarray(total_loss, dtype=float)
    gaps = ~np.isfinite(loss)
    loss = pd.DataFrame(loss).ffill().bfill().to_numpy()

    wet = wet_dry_rolling_std(loss, cfg.wet_window, cfg.wet_threshold_db)
    baseline = baseline_from_dry(loss, wet, cfg.baseline_window)

    # Rain attenuation is the excess over the dry reference, everywhere.
    #
    # It is tempting to also force a_rain to zero wherever the classifier says
    # dry, but that silently deletes steady rain: the rolling-standard-
    # deviation test keys on fluctuation, and widespread stratiform rain
    # attenuates steadily. Subtracting the baseline already drives genuinely
    # dry periods to ~0.
    a_rain = np.clip(loss - baseline, 0.0, None)
    a_rain[a_rain < cfg.min_attenuation_db] = 0.0

    length_km = np.asarray(length_km, dtype=float).ravel()
    k, alpha = cv.itu_coefficients(
        np.asarray(freq_ghz, dtype=float).ravel(),
        cv.normalize_polarization(polarization))
    k = k[None, :]
    alpha = alpha[None, :]
    lengths = length_km[None, :]

    rain = np.zeros_like(a_rain)
    for _ in range(8):
        waa = cfg.waa_max_db * (1.0 - np.exp(-cfg.waa_rate_per_mm_h * rain))
        rain = (np.clip(a_rain - waa, 0.0, None)
                / (k * lengths)) ** (1.0 / alpha)
    rain[~np.isfinite(rain)] = 0.0
    rain[gaps] = np.nan

    # Short links cannot resolve rain: their path attenuation sits below the
    # quantization floor, so the retrieval is pure noise amplification.
    rain[:, length_km < cfg.min_length_km] = np.nan

    return {"R": rain, "A_rain": a_rain, "baseline": baseline, "wet": wet}
