"""Test-week scores: the RNN against every power-law method, on the same link-hours.

Scored against the averaged target the RNN was trained on, and separately against each
source (radar along the path, gauges near the link), so a model cannot win only by
copying the average it was taught. The sample is the test hours where the reference and
every full-coverage estimate have a value: estimates within 80% of the best-covered one.
The RNN answers even during link outages, where the power-law methods have nothing, so
this keeps it to the hours the power-law methods can be scored on; a method with more
gaps than that (the nearby-link one) is scored on its share, with its coverage shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.maps.scores import scores

from .settings import TARGET_POINTS
from .train import NetData, load, predict

PL = ["pl_dynamic", "pl_constant", "pl_pycomlink", "pl_nearby"]


def estimates(name: str, network: str) -> tuple:
    """``(NetData, {method: (link, time) mm})`` with the RNN's prediction as ``rnn``."""
    model, cfg, meta = load(name)
    d = NetData.load(network, "target", cfg.min_link_valid, cfg.physics, cfg.neighbours)
    est = {"rnn": predict(model, cfg, d, meta["history"].get("calibration"))}
    for m in PL:
        if m in d.ds:
            est[m.removeprefix("pl_")] = d.ds[m].values
    return d, est


def score_network(d: NetData, est: dict, part: int = 2, wet: float = 0.1) -> pd.DataFrame:
    refs = ["target", "radar"] + [p for p in TARGET_POINTS[d.network] if p in d.ds]
    rows = []
    in_part = np.broadcast_to(d.part == part, d.y.shape)
    for ref_name in refs:
        ref = d.ds[ref_name].values
        base = in_part & np.isfinite(ref)
        n = {k: int((base & np.isfinite(v)).sum()) for k, v in est.items()}
        full = [k for k in est if n[k] >= 0.8 * max(n.values())]
        common = base & np.all([np.isfinite(est[k]) for k in full], axis=0)
        for k, v in est.items():
            sel = common & np.isfinite(v)
            rows.append({"network": d.network, "reference": ref_name, "method": k, "sample": "common",
                         "coverage": sel.sum() / max(common.sum(), 1), **scores(v[sel], ref[sel], wet)})
        # head to head: the RNN on exactly each power-law method's own valid hours
        for k, v in est.items():
            if k == "rnn":
                continue
            sel = base & np.isfinite(v) & np.isfinite(est["rnn"])
            for who, x in (("rnn", est["rnn"]), (k, v)):
                rows.append({"network": d.network, "reference": ref_name, "method": who,
                             "sample": f"paired with {k}", "coverage": 1.0, **scores(x[sel], ref[sel], wet)})
    return pd.DataFrame(rows)


def evaluate(name: str, networks) -> pd.DataFrame:
    return pd.concat([score_network(*estimates(name, n)) for n in networks], ignore_index=True)


def verdict(table: pd.DataFrame, reference: str = "target") -> pd.DataFrame:
    """Per network and power-law method: the RNN and the method on that method's own valid
    hours. The RNN "wins" a network only if it has lower RMSE *and* higher correlation than
    every power-law method, each on its own sample."""
    rows = []
    t = table[(table.reference == reference) & table["sample"].str.startswith("paired")]
    for (n, smp), g in t.groupby(["network", "sample"], sort=False):
        pl = smp.removeprefix("paired with ")
        r, p = g[g.method == "rnn"].iloc[0], g[g.method == pl].iloc[0]
        rows.append({"network": n, "power_law": pl, "n": int(r["n"]),
                     "rmse_rnn": r["rmse"], "rmse_pl": p["rmse"], "corr_rnn": r["corr"], "corr_pl": p["corr"],
                     "bias_rnn": r["rel_bias"], "bias_pl": p["rel_bias"], "csi_rnn": r["csi"], "csi_pl": p["csi"],
                     "rnn_better": bool(r["rmse"] < p["rmse"] and r["corr"] > p["corr"])})
    return pd.DataFrame(rows)


def wins(table: pd.DataFrame, reference: str = "target") -> pd.Series:
    """Per network: does the RNN beat every power-law method head to head?"""
    v = verdict(table, reference)
    return v.groupby("network").rnn_better.all()
