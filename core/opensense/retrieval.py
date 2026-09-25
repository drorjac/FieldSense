"""
The CML retrieval chain, independent of where the data came from.

The same code runs on the raw Zenodo archives, on the curated example subsets,
in ``validate_retrieval.py`` against the reference retrieval OpenSense
publishes, and in ``retrieval_benchmark.py`` where its variants are ranked.

The chain, per sublink:

1. total loss ``A = TSL - RSL``, or ``-RSL`` where no TSL is published
2. wet/dry classification - rolling standard deviation (Schleiss & Berne 2010)
   by default, or any externally supplied mask (radar along the path,
   nearby-link, a neural classifier)
3. dry-weather baseline, a long rolling median over samples classified dry
4. ``A_obs = A - baseline``, floored
5. wet-antenna attenuation removed from ``A_obs``, with one of several models
6. ``R = (A_rain / (k L))**(1/alpha)`` with ITU-R P.838-3 coefficients

Two entry points:

``retrieve(total_loss, length_km, freq_ghz, polarization, cfg)``
    plain arrays, (time, sublink) in, dict of arrays out.
``retrieve_dataset(ds_cml, cfg)``
    an OpenSense-1.0 CML dataset in, an xarray Dataset out. Handles TSL
    presence, units, the sublink broadcast of per-link metadata, and the
    sampling interval. Prefer this one.

The steps are public functions too, so an experiment that varies only one
step (``retrieval_benchmark.py``) can compute the others once.

Example
-------
>>> from core.opensense import example_data, retrieval
>>> cml = example_data.load("openmrg", "8d",
...                         time=slice("2015-07-28", "2015-07-28"))["cml"]
>>> out = retrieval.retrieve_dataset(cml)                  # (time, cml_id, sublink_id)
>>> per_link = retrieval.combine_sublinks(out)             # (time, cml_id)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr

from core.opensense import conventions as cv

# Wet-antenna models ``rain_from_attenuation`` knows about.
#
# ``saturating``   WAA = waa_max (1 - exp(-rate R)), solved by fixed point.
#                  The form of Schleiss et al. (2013) and Garcia-Rubia (2011),
#                  here parameterized in rain rate.
# ``pastorek2021`` WAA = A_max (1 - exp(-d R**zeta)), the "KR-alt" model of
#                  Pastorek et al. (2021), via pycomlink.
# ``leijnse2008``  a physically based thin water film on the radome, whose
#                  thickness grows with R, via pycomlink. Frequency-dependent.
# ``none``         no correction.
WAA_MODELS = ("none", "saturating", "pastorek2021", "leijnse2008")


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval parameters.

    ``wet_window`` and ``baseline_window`` are in **samples**, so they must be
    scaled to the sampling interval of the source: OpenMRG is 10 s, OpenRainER
    and OpenMesh are 1 min. Use :meth:`for_interval` rather than hardcoding.

    ``baseline_window`` and the wet-antenna parameters are calibration
    choices, not physics. The defaults were checked against the municipal
    gauges on the 25 August OpenMRG event - see the project README for the
    table, and ``retrieval_benchmark.py`` for how they compare against the
    alternatives over the full 8-day reference period.
    """

    wet_window: int = 30
    wet_threshold_db: float = 0.8
    baseline_window: int = 1080
    min_attenuation_db: float = 0.1
    min_length_km: float = 0.5

    waa_model: str = "saturating"
    # saturating
    waa_max_db: float = 0.5
    waa_rate_per_mm_h: float = 0.28
    # pastorek2021 (pycomlink defaults)
    waa_pastorek_a_max_db: float = 14.0
    waa_pastorek_zeta: float = 0.55
    waa_pastorek_d: float = 0.1

    # With a trustworthy external wet mask (radar, nearby links), rain can be
    # forced to zero where the mask says dry. Off by default, because the
    # rolling-std classifier misses steady stratiform rain - see ``retrieve``.
    zero_when_dry: bool = False

    def __post_init__(self):
        if self.waa_model not in WAA_MODELS:
            raise ValueError(f"waa_model must be one of {WAA_MODELS}, "
                             f"got {self.waa_model!r}")
        if self.wet_window < 2 or self.baseline_window < 2:
            raise ValueError("wet_window and baseline_window are sample counts "
                             "and must be >= 2")

    @classmethod
    def for_interval(cls, seconds: float, **overrides) -> "RetrievalConfig":
        """Config with the windows scaled to a sampling interval.

        Defaults are expressed for OpenMRG's 10 s sampling: a 5-minute
        wet/dry window and a 3-hour baseline. Explicit ``wet_window`` or
        ``baseline_window`` overrides win over the scaled values.
        """
        scaled = dict(wet_window=max(2, int(round(300.0 / seconds))),
                      baseline_window=max(4, int(round(3 * 3600.0 / seconds))))
        return cls(**{**scaled, **overrides})

    def with_(self, **changes) -> "RetrievalConfig":
        """A copy with some fields changed; the frozen-dataclass ``replace``."""
        return replace(self, **changes)

    def describe(self) -> str:
        """One-line summary, for dataset attributes and logs."""
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
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

    A sublink wet for the entire window has no dry reference; its own lower
    decile is the best available proxy.
    """
    dry_only = pd.DataFrame(np.where(wet, np.nan, attenuation))
    base = (dry_only.rolling(window=window, center=True, min_periods=1)
            .median().ffill().bfill())

    all_wet = base.isna().all(axis=0)
    if all_wet.any():
        fallback = pd.DataFrame(attenuation).quantile(0.1)
        base.loc[:, all_wet] = fallback[all_wet].to_numpy()
    return base.to_numpy()


def wet_antenna_attenuation(a_obs: np.ndarray, k: np.ndarray, alpha: np.ndarray,
                            length_km: np.ndarray, freq_ghz: np.ndarray,
                            cfg: RetrievalConfig) -> np.ndarray:
    """Wet-antenna attenuation (dB) contained in the observed excess ``a_obs``.

    ``a_obs`` is (time, sublink); the per-sublink arrays are 1-D.

    Every model here is a function of the rain rate, which is what is being
    solved for. The ``saturating`` model is inverted by fixed-point
    iteration, which is how this chain has always done it. The pycomlink
    models are inverted with a per-sublink lookup table from
    ``A_obs = A_rain + WAA(R(A_rain))`` to ``WAA`` - the same construction as
    pycomlink's ``*_from_A_obs`` functions, but using this repository's
    ITU-R P.838-3 coefficients so every model shares one k-R relation.
    """
    if cfg.waa_model == "none":
        return np.zeros_like(a_obs)

    kl = (k * length_km)[None, :]
    inv_alpha = (1.0 / alpha)[None, :]

    if cfg.waa_model == "saturating":
        rain = np.zeros_like(a_obs)
        waa = np.zeros_like(a_obs)
        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(8):
                waa = cfg.waa_max_db * (1.0 - np.exp(-cfg.waa_rate_per_mm_h * rain))
                rain = (np.clip(a_obs - waa, 0.0, None) / kl) ** inv_alpha
                rain[~np.isfinite(rain)] = 0.0
        return waa

    from pycomlink.processing import wet_antenna as pcm_waa

    if cfg.waa_model == "pastorek2021":
        def waa_of_rain(r, _f_ghz):
            return pcm_waa.waa_pastorek_2021(
                R=r, A_max=cfg.waa_pastorek_a_max_db,
                zeta=cfg.waa_pastorek_zeta, d=cfg.waa_pastorek_d)
    else:  # leijnse2008
        def waa_of_rain(r, f_ghz):
            return pcm_waa.waa_leijnse_2008(R=r, f_Hz=f_ghz * 1e9)

    # A_rain grid wide enough for anything a terrestrial link reports.
    a_grid = np.concatenate([[0.0], np.logspace(-4, 2.5, 240)])
    waa = np.zeros_like(a_obs)
    for j in range(a_obs.shape[1]):
        if not (np.isfinite(kl[0, j]) and kl[0, j] > 0):
            continue
        r = (a_grid / kl[0, j]) ** inv_alpha[0, j]
        w = np.nan_to_num(np.asarray(waa_of_rain(r, float(freq_ghz[j])),
                                     dtype=float))
        # A_obs = A_rain + WAA is monotone in R for every model here, so the
        # table inverts with a plain 1-D interpolation.
        waa[:, j] = np.interp(a_obs[:, j], a_grid + w, w)
    return waa


def rain_from_attenuation(a_obs: np.ndarray, length_km: np.ndarray,
                          freq_ghz: np.ndarray, polarization,
                          cfg: RetrievalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Excess attenuation over baseline (time, sublink) -> (rain rate, WAA).

    Removes wet-antenna attenuation, then inverts the ITU-R P.838-3 power
    law. Short links are masked: their path attenuation sits below the
    quantization floor, so the inversion is pure noise amplification.
    """
    length_km = np.asarray(length_km, dtype=float).ravel()
    freq_ghz = np.asarray(freq_ghz, dtype=float).ravel()
    k, alpha = cv.itu_coefficients(freq_ghz, cv.normalize_polarization(polarization))

    waa = wet_antenna_attenuation(a_obs, k, alpha, length_km, freq_ghz, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        rain = (np.clip(a_obs - waa, 0.0, None)
                / (k * length_km)[None, :]) ** (1.0 / alpha)[None, :]
    rain[~np.isfinite(rain)] = 0.0
    rain[:, ~(length_km >= cfg.min_length_km)] = np.nan
    return rain, waa


def retrieve(total_loss: np.ndarray, length_km: np.ndarray,
             freq_ghz: np.ndarray, polarization,
             cfg: RetrievalConfig | None = None,
             wet: np.ndarray | None = None) -> dict:
    """Total path loss (time, sublink) -> rain rate (time, sublink) in mm/h.

    ``length_km``, ``freq_ghz`` and ``polarization`` are per sublink, length
    matching ``total_loss.shape[1]``.

    ``wet`` optionally replaces the rolling-std classifier with an external
    boolean mask of the same shape. It then drives the baseline, and - with
    ``cfg.zero_when_dry`` - also forces dry samples to zero rain.

    Returns a dict with ``R``, ``A_obs``, ``A_rain`` (``A_obs`` minus
    wet-antenna attenuation), ``waa``, ``baseline`` and ``wet`` so callers
    can inspect the intermediate steps.
    """
    cfg = cfg or RetrievalConfig()

    loss = np.asarray(total_loss, dtype=float)
    gaps = ~np.isfinite(loss)
    loss = pd.DataFrame(loss).ffill().bfill().to_numpy()

    if wet is None:
        wet = wet_dry_rolling_std(loss, cfg.wet_window, cfg.wet_threshold_db)
    else:
        wet = np.asarray(wet, dtype=bool)
        if wet.shape != loss.shape:
            raise ValueError(f"wet mask shape {wet.shape} does not match "
                             f"total_loss {loss.shape}")
    baseline = baseline_from_dry(loss, wet, cfg.baseline_window)

    # Rain attenuation is the excess over the dry reference, everywhere.
    #
    # It is tempting to also force it to zero wherever the rolling-std
    # classifier says dry, but that silently deletes steady rain: the test
    # keys on fluctuation, and widespread stratiform rain attenuates steadily.
    # Subtracting the baseline already drives genuinely dry periods to ~0.
    # ``zero_when_dry`` exists for masks that do not share that blind spot.
    a_obs = np.clip(loss - baseline, 0.0, None)
    a_obs[a_obs < cfg.min_attenuation_db] = 0.0
    if cfg.zero_when_dry:
        a_obs[~wet] = 0.0

    rain, waa = rain_from_attenuation(a_obs, length_km, freq_ghz,
                                      polarization, cfg)
    rain[gaps] = np.nan

    return {"R": rain, "A_obs": a_obs, "A_rain": np.clip(a_obs - waa, 0.0, None),
            "waa": waa, "baseline": baseline, "wet": wet}


# --------------------------------------------------------------------------
# xarray front end
# --------------------------------------------------------------------------
def sampling_interval_s(time) -> float:
    """Median spacing of a time coordinate, in seconds.

    The median, not the first difference: raw archives have gaps, and a
    single gap at the start would scale every window wrongly.
    """
    t = np.asarray(time, dtype="datetime64[ns]")
    if t.size < 2:
        raise ValueError("need at least two timestamps to infer the interval")
    return float(np.median(np.diff(t)).astype("timedelta64[ms]").astype(float)) / 1e3


def link_metadata(ds: xr.Dataset, like: xr.DataArray) -> tuple:
    """Length (km), frequency (GHz), polarization broadcast to ``like``'s shape.

    Per-link metadata is sometimes per ``cml_id`` and sometimes per
    (``sublink_id``, ``cml_id``), in either order. Broadcasting against a
    (cml_id, sublink_id) template by name handles every layout.
    """
    def coord(normalized, raw, convert):
        if normalized in ds.coords:
            return ds[normalized]
        if raw in ds.variables:
            return xr.DataArray(convert(ds[raw]), dims=ds[raw].dims)
        raise ValueError(f"CML dataset has neither {normalized!r} nor {raw!r}")

    length = coord("length_km", "length", cv.to_km)
    freq = coord("frequency_ghz", "frequency", cv.to_ghz)
    if "polarization" in ds.variables:
        pol = xr.DataArray(
            cv.normalize_polarization(ds.polarization.values)
            .reshape(ds.polarization.shape), dims=ds.polarization.dims)
    else:
        pol = xr.full_like(freq, "vertical", dtype=object)

    out = []
    for da in (length, freq, pol):
        da = da.reset_coords(drop=True) if hasattr(da, "reset_coords") else da
        out.append(np.asarray(da.broadcast_like(like).transpose(*like.dims)))
    return tuple(out)


def retrieve_dataset(ds: xr.Dataset, cfg: RetrievalConfig | None = None,
                     wet: xr.DataArray | np.ndarray | None = None,
                     use_tsl: bool | None = None) -> xr.Dataset:
    """Run the chain on an OpenSense-1.0 CML dataset.

    ``ds`` needs ``rsl`` (and ``tsl`` if present) on (time, cml_id,
    sublink_id), plus ``length``/``frequency``/``polarization`` in any of the
    unit conventions ``conventions.py`` understands. A dataset without a
    ``sublink_id`` dimension is treated as one sublink per link.

    ``use_tsl`` defaults to using TSL whenever the dataset has it and it is
    not all-NaN. ``cfg`` defaults to the standard windows scaled to the
    dataset's own sampling interval. ``wet`` is an optional external wet mask
    on the same (time, cml_id[, sublink_id]) axes.

    Returns a Dataset with ``R`` (mm/h), ``A_obs``, ``A_rain``, ``waa``,
    ``baseline`` and ``wet`` on (time, cml_id, sublink_id), carrying the input
    coordinates. Use :func:`combine_sublinks` to get one series per link.
    """
    squeeze_sublink = "sublink_id" not in ds.dims
    if squeeze_sublink:
        ds = ds.expand_dims(sublink_id=["sublink_1"])

    dims = ("time", "cml_id", "sublink_id")
    rsl = ds.rsl.transpose(*dims)
    if use_tsl is None:
        use_tsl = "tsl" in ds and bool(np.isfinite(ds.tsl).any())
    tsl = ds.tsl.transpose(*dims) if use_tsl else None
    loss = total_loss_from(rsl.values, None if tsl is None else tsl.values)
    n_t, n_c, n_s = loss.shape

    length, freq, pol = link_metadata(ds, rsl.isel(time=0, drop=True))
    cfg = cfg or RetrievalConfig.for_interval(sampling_interval_s(ds.time))

    wet_flat = None
    if wet is not None:
        if isinstance(wet, xr.DataArray):
            wet = wet.broadcast_like(rsl).transpose(*dims).values
        wet_flat = np.asarray(wet, dtype=bool).reshape(n_t, n_c * n_s)

    res = retrieve(loss.reshape(n_t, n_c * n_s), length.ravel(), freq.ravel(),
                   pol.ravel(), cfg, wet=wet_flat)

    out = xr.Dataset(
        {name: (dims, np.asarray(res[name]).reshape(n_t, n_c, n_s))
         for name in ("R", "A_obs", "A_rain", "waa", "baseline", "wet")},
        coords={c: v for c, v in rsl.coords.items()})
    out.R.attrs.update(units="mm h-1", long_name="rain rate")
    for name in ("A_obs", "A_rain", "waa", "baseline"):
        out[name].attrs["units"] = "dB"
    out.attrs.update(retrieval=cfg.describe(),
                     loss="TSL - RSL" if use_tsl else "-RSL (no TSL used)")
    if squeeze_sublink:
        out = out.isel(sublink_id=0, drop=True)
    return out


def combine_sublinks(ds: xr.Dataset, var: str = "R") -> xr.Dataset:
    """Average the sublinks of each link into one series per ``cml_id``.

    Both directions of a link traverse the same path, so averaging halves the
    retrieval noise. Per-sublink coordinates (``frequency``, ``polarization``)
    are reduced to the first sublink's value, which is what mergeplg and
    poligrain need: a single geometry per link.

    Not appropriate for multi-band networks such as OpenMesh, whose sublinks
    differ in sensitivity by orders of magnitude - see ``ingest_openmesh``.
    """
    if "sublink_id" not in ds.dims:
        return ds[[var]]
    with np.errstate(invalid="ignore"):
        da = ds[var].mean("sublink_id", skipna=True, keep_attrs=True)
    out = da.to_dataset(name=var)
    for name, c in ds.coords.items():
        if name in out.coords or name in ds.dims or "time" in c.dims:
            continue
        if "sublink_id" in c.dims:
            out.coords[name] = c.isel(sublink_id=0, drop=True)
        elif "cml_id" in c.dims:
            out.coords[name] = c
    return out
