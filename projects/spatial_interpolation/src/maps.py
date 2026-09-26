"""
Rain maps on the radar grid from per-link rain: IDW and GMZ.

Part 2 of ``advanced_models_colab_v2.ipynb``, with each of its choices kept
behind a flag (``faithful``) so the corrected version can be compared:

IDW
    inverse-distance weights from link midpoints to every radar pixel. The
    notebook measured distance in raw degrees; at 57.7 N a degree of
    longitude is 0.53 of a degree of latitude, so east-west distances
    counted half. ``faithful=False`` uses kilometres.
GMZ
    PyNNcml's Goldshtein-Messer-Zinevich reconstruction, which returns a
    field on its own grid normalized to ``(utm - min) / scale`` over the link
    set. The notebook ran pynncml 0.3.7 unpatched - its bilinear step reads
    the wrong cell for one corner almost everywhere, see
    ``core.scientific_packages.pynncml_compat`` - and mapped the normalized
    grid onto the bounding box of link *midpoints*, which misregisters it.
    ``faithful=False`` patches GMZ and inverts PyNNcml's normalization
    exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

KM_PER_DEG_LAT = 111.32


def idw_weights(src_lon, src_lat, lon, lat, power: float = 2.0,
                metric: str = "degrees", eps: float = 1e-6) -> np.ndarray:
    """(n_pixels, n_sources) row-normalized inverse-distance weights."""
    px, py = np.ravel(lon)[:, None], np.ravel(lat)[:, None]
    dx, dy = px - np.asarray(src_lon)[None, :], py - np.asarray(src_lat)[None, :]
    if metric == "km":
        dx = dx * KM_PER_DEG_LAT * np.cos(np.deg2rad(np.mean(lat)))
        dy = dy * KM_PER_DEG_LAT
    elif metric != "degrees":
        raise ValueError(f"metric must be 'degrees' or 'km', got {metric!r}")
    w = 1.0 / np.maximum(np.hypot(dx, dy), eps) ** power
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def idw(values: np.ndarray, src_lon, src_lat, lon, lat, power: float = 2.0,
        metric: str = "degrees") -> np.ndarray:
    """(T, n_sources) values -> (T, *lon.shape) maps."""
    w = idw_weights(src_lon, src_lat, lon, lat, power, metric)
    return (np.asarray(values, dtype=np.float32) @ w.T).reshape(len(values), *np.shape(lon))


def gmz(link_set, values: np.ndarray, lon, lat, roi: float = 3.0,
        points_per_link: int = 2, faithful: bool = True,
        link_lon=None, link_lat=None) -> np.ndarray:
    """GMZ maps on the radar grid from (T, n_links) per-link rain.

    ``link_lon``/``link_lat`` (midpoints) are needed only for the faithful
    registration, which spreads the normalized grid over their bounding box.
    """
    import pynncml as pnc
    import torch

    from core.scientific_packages import pynncml_compat
    if faithful:
        pynncml_compat.unpatch_pynncml_gmz()     # the patch is sticky in-process
    else:
        pynncml_compat.patch_pynncml_gmz()
    model = pnc.mcm.generate_link_set_gmz(link_set, roi=roi, point_per_link=points_per_link)
    out = model(torch.tensor(np.asarray(values, dtype=np.float32)).T)
    field = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()   # (T, X, Y)
    xg = np.asarray(model.base_idw.x_grid_vector)
    yg = np.asarray(model.base_idw.y_grid_vector)

    if faithful:
        # what the notebook did: the normalized vectors are not lon/lat, so
        # it fell back to a linear map over the midpoints' bounding box
        xs = np.linspace(np.min(link_lon), np.max(link_lon), xg.size)
        ys = np.linspace(np.min(link_lat), np.max(link_lat), yg.size)
        qx, qy = np.ravel(lon), np.ravel(lat)
    else:
        import utm
        east, north, _, _ = utm.from_latlon(np.ravel(lat), np.ravel(lon),
                                            force_zone_number=32, force_zone_letter="V")
        xs, ys = xg, yg
        qx = (east - link_set.x_min) / link_set.scale
        qy = (north - link_set.y_min) / link_set.scale

    pts = np.column_stack([qx, qy])
    maps = np.empty((field.shape[0], *np.shape(lon)), dtype=np.float32)
    for t in range(field.shape[0]):
        f = RegularGridInterpolator((xs, ys), field[t], bounds_error=False, fill_value=0.0)
        maps[t] = f(pts).reshape(np.shape(lon))
    return maps
