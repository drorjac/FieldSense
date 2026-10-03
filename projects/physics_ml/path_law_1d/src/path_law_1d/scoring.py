"""Scores of a per-link rain estimate against radar and gauges, by link length.

The proposal judges every arm by its normalized bias and normalized RMSE against
both references, broken down by link length: the path law matters on long links,
the non-rain term on short ones. The metrics come from
``core.opensense.evaluation.rainfall_metrics`` (poligrain), normalized here:

    nbias = sum(est - ref) / sum(ref)        nrmse = rmse / mean(ref)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.opensense.evaluation import rainfall_metrics

LINK_LENGTH_BINS_KM = (0.0, 1.0, 2.0, 4.0, 8.0, 20.0)


def normalized_scores(reference, estimate) -> dict:
    """``nbias``, ``nrmse``, ``r`` and ``n`` (valid pairs) of ``estimate`` against ``reference``."""
    m = rainfall_metrics(reference, estimate)
    mean_ref = m["mean_ref"]
    ok = np.isfinite(mean_ref) and mean_ref > 0
    return {"nbias": m["pbias"] / 100.0 if ok else np.nan,
            "nrmse": m["rmse"] / mean_ref if ok else np.nan,
            "r": m["r"], "n": m["n"]}


def scores_by_length(estimate: xr.DataArray, references: dict, length_km,
                     time_mask=None, bins=LINK_LENGTH_BINS_KM) -> pd.DataFrame:
    """Score a ``(time, cml_id)`` estimate against each ``(time, cml_id)`` reference.

    ``references`` maps a name to a reference on the same time steps and links (NaN
    where it has none: a link with no gauge nearby). ``length_km`` is per ``cml_id``.
    ``time_mask`` (boolean over time) restricts the scoring, e.g. to test events.
    Returns one row per length bin (and ``"all"``), columns ``(reference, metric)``
    plus the number of links per bin.
    """
    L = np.asarray(length_km, float)
    labels = [f"{lo:g}-{hi:g} km" for lo, hi in zip(bins[:-1], bins[1:])]
    which = pd.cut(L, bins, labels=labels)
    groups = [(lab, np.asarray(which == lab)) for lab in labels] + [("all", np.ones(L.size, bool))]
    rows = {}
    for lab, sel in groups:
        row = {("links", "n"): int(sel.sum())}
        for name, ref in references.items():
            est_, ref_ = xr.align(estimate.transpose("time", "cml_id"),
                                  ref.transpose("time", "cml_id"), join="inner")
            e, r = est_.values[:, sel], ref_.values[:, sel]
            if time_mask is not None:
                tm = pd.Series(np.asarray(time_mask), index=pd.DatetimeIndex(estimate.time.values))
                keep = tm.reindex(pd.DatetimeIndex(est_.time.values)).fillna(False).to_numpy(bool)
                e, r = e[keep], r[keep]
            s = normalized_scores(r.ravel(), e.ravel()) if sel.any() else {}
            row.update({(name, k): s.get(k, np.nan) for k in ("nbias", "nrmse", "r", "n")})
        rows[lab] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "length"
    return out
