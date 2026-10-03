"""The non-rain term: drying tails, and the literature wet-antenna models fitted per link.

With rain along the path known (simulation truth, or radar along the path), the
non-rain term is what the path law does not explain,

    delta_hat = A_obs - f(R, L),     f = a R^b L (the linear law) by default.

Two views of it here, both baselines for a learned ``d delta / dt = F(delta, R)``:

``tail_table``        every post-event drying tail per link, fitted with
                      ``amp exp(-t / tau) + offset``: the drying time read directly
                      off the data, which a learned model's drying time should match
``fit_literature``    the static models of ``core.simulation.wet_antenna`` (constant,
                      Schleiss 2013, Pastorek 2021) fitted per link by least squares.
                      None of them has memory, so all of them predict zero in a tail.
``waa_scores``        RMSE of any prediction of ``delta_hat`` in rain and in the tails

Arrays are ``(links, time)``, rain in mm/h, attenuation in dB, time steps of ``dt_min``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from core.simulation import wet_antenna as wa

WET_MM_H = 0.1


def minutes_since_rain(rain: np.ndarray, dt_min: float, wet_mm_h: float = WET_MM_H) -> np.ndarray:
    """Minutes since the last wet step along the last axis (0 when wet, inf before any)."""
    wet = np.asarray(rain) >= wet_mm_h
    out = np.full(wet.shape, np.inf)
    last = np.full(wet.shape[:-1], -np.inf)
    for i in range(wet.shape[-1]):
        last = np.where(wet[..., i], i, last)
        out[..., i] = (i - last) * dt_min
    return out


def _tail_starts(wet: np.ndarray, min_wet: int):
    """Indices where a wet run of at least ``min_wet`` steps ends (first dry step)."""
    edges = np.diff(np.r_[0, wet.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [e for s, e in zip(starts, ends) if e - s >= min_wet and e < wet.size]


def fit_tail(y: np.ndarray, dt_min: float) -> dict:
    """Fit ``amp exp(-t / tau) + offset`` to one tail; ``t = 0`` at its first sample."""
    t = np.arange(y.size) * dt_min
    p0 = (max(y[0] - y[-1], 0.01), 20.0, y[-1])
    try:
        p, _ = curve_fit(lambda t, a, tau, c: a * np.exp(-t / tau) + c, t, y, p0=p0,
                         bounds=([0.0, 0.5, -np.inf], [np.inf, 2000.0, np.inf]), maxfev=4000)
    except (RuntimeError, ValueError):
        return {"amp_db": np.nan, "tau_min": np.nan, "offset_db": np.nan, "rmse_db": np.nan}
    resid = y - (p[0] * np.exp(-t / p[1]) + p[2])
    return {"amp_db": p[0], "tau_min": p[1], "offset_db": p[2],
            "rmse_db": float(np.sqrt(np.mean(resid**2)))}


def tail_table(delta: np.ndarray, rain: np.ndarray, dt_min: float, link_ids=None,
               times=None, min_wet_min: float = 10.0, min_len_min: float = 30.0,
               max_len_min: float = 180.0, wet_mm_h: float = WET_MM_H) -> pd.DataFrame:
    """Every drying tail of ``delta`` after rain, fitted (one row per tail).

    A tail starts at the first dry step after at least ``min_wet_min`` of rain and
    runs until it rains again or ``max_len_min``; tails shorter than ``min_len_min``
    or with missing values are skipped. Columns: ``link``, ``start``, ``n_steps``,
    ``delta0_db`` (the first value), and the fit of :func:`fit_tail`.
    """
    delta, rain = np.atleast_2d(delta), np.atleast_2d(rain)
    link_ids = np.arange(delta.shape[0]) if link_ids is None else np.asarray(link_ids)
    rows = []
    for j in range(delta.shape[0]):
        wet = rain[j] >= wet_mm_h
        for s in _tail_starts(wet, int(np.ceil(min_wet_min / dt_min))):
            nxt = np.flatnonzero(wet[s:])
            stop = s + (nxt[0] if nxt.size else wet.size - s)
            stop = min(stop, s + int(max_len_min / dt_min))
            y = delta[j, s:stop]
            if y.size < min_len_min / dt_min or not np.isfinite(y).all():
                continue
            rows.append({"link": link_ids[j], "start": s if times is None else times[s],
                         "n_steps": y.size, "delta0_db": float(y[0]), **fit_tail(y, dt_min)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# literature models, per link
# --------------------------------------------------------------------------
def _scale(target, shape, mask):
    """Least-squares amplitude ``c >= 0`` of ``c * shape`` per link, and the SSE."""
    t = np.where(mask, target, 0.0)
    s = np.where(mask, shape, 0.0)
    den = (s * s).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip(np.where(den > 0, (t * s).sum(axis=1) / den, 0.0), 0.0, None)
    sse = ((t - c[:, None] * s) ** 2).sum(axis=1)
    return c, sse


def fit_literature(delta: np.ndarray, rain: np.ndarray, dt_min: float, fit_mask=None,
                   wet_mm_h: float = WET_MM_H, taus_min=(5, 10, 20, 30, 45, 60, 90, 120, 180),
                   d_grid=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5),
                   zeta_grid=(0.3, 0.5, 0.7, 0.9, 1.1)) -> tuple[pd.DataFrame, dict]:
    """Fit the constant, Schleiss (2013) and Pastorek (2021) models to ``delta`` per link.

    Least squares over the finite samples in ``fit_mask`` (``(time,)`` or
    ``(links, time)``; default all), dry samples included, so a model is charged for
    what it predicts after the rain as well. The amplitude is solved in closed form;
    the shape parameters (``tau`` for Schleiss, ``d`` and ``zeta`` for Pastorek) by
    grid search. Returns ``(params, predictions)``: one row per link with each
    model's parameters, and ``{model: (links, time)}`` predictions over all samples.
    """
    delta, rain = np.atleast_2d(delta), np.atleast_2d(rain)
    wet = rain >= wet_mm_h
    mask = np.isfinite(delta) & np.isfinite(rain)
    if fit_mask is not None:
        mask &= np.broadcast_to(np.asarray(fit_mask, bool), delta.shape)
    target = np.nan_to_num(delta)
    params = pd.DataFrame(index=pd.RangeIndex(delta.shape[0], name="link"))
    preds = {}

    c, _ = _scale(target, wet.astype(float), mask)
    params["constant_db"] = c
    preds["constant"] = wa.waa_constant(wet, 1.0) * c[:, None]

    best = (np.full(len(c), np.inf), np.zeros(len(c)), np.zeros(len(c)))
    for tau in taus_min:
        g = wa.waa_schleiss_2013(wet, 1.0, tau, dt_min)          # linear in waa_max
        amp, sse = _scale(target, g, mask)
        better = sse < best[0]
        best = (np.where(better, sse, best[0]), np.where(better, amp, best[1]),
                np.where(better, tau, best[2]))
    params["schleiss_waa_max_db"], params["schleiss_tau_min"] = best[1], best[2]
    preds["schleiss"] = np.zeros(delta.shape)
    for tau in np.unique(best[2]):
        sel = best[2] == tau
        preds["schleiss"][sel] = wa.waa_schleiss_2013(wet[sel], 1.0, tau, dt_min) * best[1][sel, None]

    best = (np.full(len(c), np.inf), np.zeros(len(c)), np.zeros(len(c)), np.zeros(len(c)))
    for d in d_grid:
        for z in zeta_grid:
            s = wa.waa_pastorek_2021(np.nan_to_num(rain), 1.0, d, z)
            amp, sse = _scale(target, s, mask)
            better = sse < best[0]
            best = (np.where(better, sse, best[0]), np.where(better, amp, best[1]),
                    np.where(better, d, best[2]), np.where(better, z, best[3]))
    params["pastorek_a_max_db"], params["pastorek_d"], params["pastorek_zeta"] = best[1:]
    preds["pastorek"] = np.stack([wa.waa_pastorek_2021(np.nan_to_num(rain[j]), a, d, z)
                                  for j, (a, d, z) in enumerate(zip(*best[1:]))])
    return params, preds


def waa_scores(delta: np.ndarray, predictions: dict, rain: np.ndarray, dt_min: float,
               eval_mask=None, tail_min: float = 180.0, wet_mm_h: float = WET_MM_H) -> pd.DataFrame:
    """RMSE (dB) of each prediction of ``delta``, pooled over links, in three regimes.

    ``rain``: wet samples; ``tail``: dry samples within ``tail_min`` after rain (where
    a model without memory predicts zero); ``dry``: the rest. ``eval_mask`` restricts
    the samples (``(time,)`` or ``(links, time)``), e.g. to test events.
    """
    delta, rain = np.atleast_2d(delta), np.atleast_2d(rain)
    since = minutes_since_rain(rain, dt_min, wet_mm_h)
    ok = np.isfinite(delta) & np.isfinite(rain)
    if eval_mask is not None:
        ok &= np.broadcast_to(np.asarray(eval_mask, bool), delta.shape)
    regimes = {"rain": since == 0, "tail": (since > 0) & (since <= tail_min),
               "dry": since > tail_min}
    rows = {}
    for name, pred in predictions.items():
        err = np.asarray(pred, float) - delta
        rows[name] = {f"rmse_{k}_db": float(np.sqrt(np.mean(err[ok & m] ** 2))) if (ok & m).any()
                      else np.nan for k, m in regimes.items()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "model"
    for k, m in regimes.items():
        out[f"n_{k}"] = int((ok & m).sum())
    return out
