"""Motion learned from simulations, and whether it holds on real radar.

``train``      the U-Net of ``core.nowcast.learned_motion`` on simulated sequences with their
               true flow: from the fields alone (``learned``) or as a correction to
               Lucas-Kanade (``learned+LK``)
``real_radar`` the same networks on the real radar of two OpenSense networks, where the true
               motion is unknown: each motion field is judged by the nowcast it makes -
               extrapolation along it, scored against the radar that followed (CSI, FSS,
               RMSE, pooled by ``core.nowcast.verify``)

The real radar is the native-step radar ``os_nowcasting`` caches for its events (OpenMRG:
SMHI, 5 min; OpenRainER: ARPAE, 15 min; both on its 2 km grid), read from those files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .settings import MODELS, REAL_RADAR_CACHE, REAL_STEP_MIN

NETS = {"learned": None, "learned+LK": "LK"}


def model_path(prior=None) -> Path:
    return MODELS / ("learned_motion.pt" if not prior else f"learned_motion_{prior}.pt")


def train(n_sequences: int = 700, epochs: int = 20, retrain: bool = False, prior=None):
    """The learned motion network: pure (``prior=None``) or a correction to LK (``prior="LK"``).

    Both are trained on the same simulated sequences, cached in ``results/models``.
    """
    from core.nowcast import learned_motion as lm
    path = model_path(prior)
    if path.exists() and not retrain:
        return lm.load(path)
    t = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    cache = MODELS / f"learned_motion_data_{n_sequences}.npz"
    if cache.exists():
        d = np.load(cache)
        data = (d["X"], d["Y"])
    else:
        data = lm.simulate_dataset(n_sequences, seed=0)
        np.savez_compressed(cache, X=data[0], Y=data[1])
    if prior:
        pcache = MODELS / f"learned_motion_data_{n_sequences}_{prior}.npz"
        if pcache.exists():
            P = np.load(pcache)["P"]
        else:
            P = lm.prior_fields(data[0], prior)
            np.savez_compressed(pcache, P=P)
        data = (*data, P)
    print(f"training on {len(data[0])} samples ({(time.time() - t) / 60:.1f} min of data)", flush=True)
    net = lm.train(epochs=epochs, seed=0, data=data, prior=prior)
    lm.save(net, path)
    path.with_suffix(".json").write_text(json.dumps(
        {"n_sequences": n_sequences, "n_samples": len(data[0]), "epochs": epochs, "prior": prior,
         "val_epe_px": net.val_loss, "minutes": (time.time() - t) / 60}, indent=1))
    return net


def available() -> dict:
    """``{label: network}`` for every trained network in ``results/models``."""
    return {label: train(prior=prior) for label, prior in NETS.items() if model_path(prior).exists()}


# --------------------------------------------------------------------------
# real radar
# --------------------------------------------------------------------------
def real_events(network: str) -> list[xr.DataArray]:
    """The event windows ``os_nowcasting`` cached (mm per step); windows inside another dropped."""
    files = sorted((REAL_RADAR_CACHE / network).glob("radar_*.nc"))
    spans = [tuple(f.stem.split("_")[1:3]) for f in files]
    keep = [f for f, (a, b) in zip(files, spans)
            if not any((c <= a and b <= d) and (c, d) != (a, b) for c, d in spans)]
    out = []
    for f in keep:
        with xr.open_dataarray(f) as da:
            out.append(da.load())
    return out


def real_radar(network: str = "openmrg", issue_every: int = 3, n_leads: int | None = None,
               min_wet: float = 0.05, nets=None, methods=("LK", "VET", "DARTS", "proesmans"),
               log=print) -> pd.DataFrame:
    """Extrapolation along each motion field from every issue time of every cached event.

    Lead times up to an hour (``n_leads`` steps). Issue times every ``issue_every`` steps with
    more than ``min_wet`` of the domain wet; only
    cells every method's backward trajectory keeps inside the domain are scored, for all.
    """
    from core.geo import Grid
    from core.nowcast import learned_motion as lm
    from core.nowcast import methods as nm
    from core.nowcast.grid import metadata, pixel_km
    from core.nowcast.verify import Collector, Deterministic
    nets = available() if nets is None else nets
    step = REAL_STEP_MIN[network]
    n_leads = n_leads or 60 // step                     # an hour ahead
    col = None
    n_issues = 0
    for ev in real_events(network):
        grid = Grid(ev.lat.values, ev.lon.values)
        meta = metadata(grid, step, ev.time.values)
        rate = np.nan_to_num(ev.transpose("time", "lat", "lon").values.astype(float)) * 60.0 / step
        col = col or Collector(lambda: Deterministic(pixel_km(grid)))
        for t in range(8, rate.shape[0] - n_leads, issue_every):
            if (rate[t] > 0.1).mean() < min_wet:
                continue
            hist, obs = rate[:t + 1], rate[t + 1:t + 1 + n_leads]
            vel = {m: nm.motion(hist, meta, m) for m in methods}
            for label, net in nets.items():
                vel[label] = lm.motion(hist, net)
            reach = np.all([nm.reachable(v, rate.shape[1:], n_leads) for v in vel.values()], axis=0)
            o = np.where(reach, obs, np.nan)
            for m, v in vel.items():
                f = np.nan_to_num(nm.deterministic("extrapolation", hist, meta, v, n_leads))
                for k in range(n_leads):
                    col[(m, (k + 1) * step)].add(f[k], o[k])
            f = nm.deterministic("persistence", hist, meta, None, n_leads)
            for k in range(n_leads):
                col[("persistence", (k + 1) * step)].add(f[k], o[k])
            n_issues += 1
        log(f"{network} event {str(ev.time.values[0])[:16]}: {n_issues} issue times so far")
    out = pd.DataFrame(col.rows(("method", "lead_min")))
    return out.assign(network=network, n_issues=n_issues)
