"""
The model-driven PyNNcml chain on OpenMRG: wet/dry, baseline, power law, maps.

The counterpart of ``notebooks/model_driven_tutorial.ipynb`` (kept as it was)
without its per-cell boilerplate - in particular the 25-line block that
guesses how a given PyNNcml version exposes a link's gauge reference, which
is ``gauge_reference`` here.

    from model_driven import ModelDrivenConfig, load, single_link, rain_maps
    cfg = ModelDrivenConfig()
    link_set = load(slice("2015-06-01", "2015-06-10"))
    one = single_link(link_set.get_link(0), cfg)
    maps = rain_maps(link_set, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelDrivenConfig:
    """Hyper-parameters, defaulting to the upstream tutorial's."""

    wet_dry_threshold: float = 0.1          # statistical test threshold
    wet_dry_window: int = 8                 # samples (15-min min/max)
    wa_factor: float = 1.6                  # wet-antenna factor, two-step model
    r_min: float = 0.3                      # mm/h floor of the power law
    dynamic_baseline_window: int = 4
    roi_km: float = 3.0                     # IDW / GMZ region of interest
    gmz_points_per_link: int = 2


def load(time_slice: slice):
    """OpenMRG as PyNNcml min/max links, from the repository's archive."""
    import pynncml as pnc
    from core.scientific_packages import pynncml_compat

    pynncml_compat.apply()
    link_set, _, _ = pynncml_compat.quietly(
        pnc.datasets.load_open_mrg, data_path=pynncml_compat.openmrg_data_path(),
        time_slice=time_slice, change2min_max=True)
    return link_set


def gauge_reference(link) -> np.ndarray:
    """A link's nearest-gauge rain rate, whichever way this PyNNcml exposes it."""
    ref = getattr(link, "gauge_ref", None)
    if isinstance(ref, (list, tuple)) and ref:
        ref = ref[0]
    if hasattr(ref, "data_array"):
        return np.asarray(ref.data_array)
    if hasattr(link, "gauge_data"):
        return np.asarray(link.gauge_data())
    if ref is None:
        raise AttributeError("link carries no gauge reference")
    return np.asarray(ref)


def single_link(link, cfg: ModelDrivenConfig) -> dict:
    """Wet/dry and two rain estimates (constant vs dynamic baseline) for one link."""
    import pynncml as pnc

    att = link.attenuation()
    swd = pnc.scm.wet_dry.statistics_wet_dry(cfg.wet_dry_threshold,
                                             cfg.wet_dry_window, is_min_max=True)
    detection, sigma = swd(att)
    constant = pnc.scm.rain_estimation.two_step_constant_baseline(
        pnc.scm.power_law.PowerLawType.MAX, cfg.r_min, cfg.wet_dry_window,
        cfg.wet_dry_threshold, wa_factor=cfg.wa_factor)
    dynamic = pnc.scm.rain_estimation.one_step_dynamic_baseline(
        pnc.scm.power_law.PowerLawType.MAX, cfg.r_min, cfg.dynamic_baseline_window, 1)
    rain_c, _, base_c = constant(att, link.meta_data)
    rain_d, base_d = dynamic(att, link.meta_data)

    def flat(t):
        return t.detach().numpy().ravel() if hasattr(t, "detach") else np.ravel(t)

    return {"attenuation": att[0].detach().numpy(),       # (time, [max, min])
            "detection": flat(detection), "sigma": flat(sigma),
            "reference": gauge_reference(link),
            "rain": {"constant baseline": flat(rain_c), "dynamic baseline": flat(rain_d)},
            "baseline": {"constant baseline": flat(base_c), "dynamic baseline": flat(base_d)}}


def rain_maps(link_set, cfg: ModelDrivenConfig) -> dict:
    """IDW and GMZ fields at the timestep where IDW varies most across space.

    Returns maps in the ``plots.map_panels`` shape plus GMZ's loss curve.
    """
    import pynncml as pnc
    from core.scientific_packages import pynncml_compat

    pynncml_compat.patch_pynncml_gmz()
    dynamic = pnc.scm.rain_estimation.one_step_dynamic_baseline(
        pnc.scm.power_law.PowerLawType.MAX, cfg.r_min, cfg.dynamic_baseline_window, 1)
    rain = pynncml_compat.quietly(pnc.mcm.InferMultipleCMLs(dynamic), link_set)
    idw = pnc.mcm.generate_link_set_idw(link_set, roi=cfg.roi_km)
    gmz = pnc.mcm.generate_link_set_gmz(link_set, roi=cfg.roi_km,
                                        point_per_link=cfg.gmz_points_per_link)
    idw_maps = idw(rain).numpy()
    gmz_maps, losses = gmz(rain)
    gmz_maps = gmz_maps.numpy()
    t = int(np.argmax(np.std(idw_maps, axis=(1, 2))))
    x, y = np.asarray(idw.x_grid_vector), np.asarray(idw.y_grid_vector)
    return {"t": t, "gmz_loss": np.asarray(losses)[:, t],
            "maps": {"idw": {"grid": idw_maps[t].T, "x": x, "y": y, "label": "IDW"},
                     "gmz": {"grid": gmz_maps[t].T, "x": x, "y": y,
                             "label": f"GMZ ({cfg.gmz_points_per_link} pts/link)"}}}
