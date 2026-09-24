"""
Rainfall maps from CML data, four ways.

The same links, the same timestep, four reconstruction backends from the
OpenSense ecosystem, so the maps are directly comparable:

``pycomlink_idw``
    ``pycomlink.spatial.interpolator.IdwKdtreeInterpolator``. Inverse distance
    weighting from each link's **midpoint**, on a lon/lat grid. This is what
    the OpenSense sandbox's ``openMRG_use_case`` notebook uses.
``pynncml_idw``
    ``pynncml.mcm.generate_link_set_idw``. Also midpoint IDW, but on a
    projected (UTM) grid with a region-of-interest radius.
``pynncml_gmz``
    ``pynncml.mcm.generate_link_set_gmz``. Goldshtein-Messer-Zinevich: solves
    for the field treating each link as a *line* of several points rather than
    one midpoint, so link length enters the reconstruction.
``mergeplg_idw`` / ``mergeplg_kriging``
    ``mergeplg.interpolate.InterpolateIDW`` and
    ``InterpolateOrdinaryKriging``. Line-aware, with the kriging variant doing
    block kriging along the path.

The distinction that matters: **midpoint methods collapse a 5 km path to a
point; line-aware methods do not.** GMZ and the mergeplg pair are the two
families that use the geometry, and they disagree with the midpoint methods
most where links are long and rain is patchy.

    python -m rain_maps --list
    python -m rain_maps --method all
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "projects/opensense_pipeline/src"))

# UTM zone covering Gothenburg, which OpenMRG sits in.
UTM_ZONE_NUMBER, UTM_ZONE_LETTER = 32, "V"


# --------------------------------------------------------------------------
def load_cml_rain(subset: str = "8d", time_index: int | None = None,
                  min_wet_links: int = 30) -> xr.Dataset:
    """OpenMRG links with a rain rate per link, at one wet timestep.

    Uses the reference retrieval the example subset ships, so the maps
    compare *reconstruction* methods rather than retrieval choices.
    """
    import example_data

    ds = example_data.load("openmrg", subset)
    cml = ds["cml"]

    rain = cml.R.transpose("time", "cml_id", "sublink_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_link = np.nanmean(np.asarray(rain), axis=2)

    if time_index is None:
        wet = (per_link >= 0.1).sum(axis=1)
        if wet.max() < min_wet_links:
            time_index = int(np.nanargmax(np.nansum(per_link, axis=1)))
        else:
            time_index = int(np.nanargmax(np.nansum(per_link, axis=1)))

    out = xr.Dataset(
        {"R": (("cml_id",), per_link[time_index])},
        coords={"cml_id": cml.cml_id},
    )
    for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
              "site_0_x", "site_0_y", "site_1_x", "site_1_y", "x", "y",
              "length_km", "frequency_ghz"):
        if c in cml.coords:
            da = cml[c]
            if "sublink_id" in da.dims:
                da = da.isel(sublink_id=0, drop=True)
            out.coords[c] = ("cml_id", np.asarray(da))
    out.attrs["time"] = str(cml.time.values[time_index])
    out.attrs["time_index"] = time_index
    out.coords["lat_center"] = (out.site_0_lat + out.site_1_lat) / 2
    out.coords["lon_center"] = (out.site_0_lon + out.site_1_lon) / 2
    return out


# --------------------------------------------------------------------------
def map_pycomlink_idw(cml: xr.Dataset, resolution: float = 0.005,
                      nnear: int = 15, p: float = 2.0,
                      max_distance: float = 0.3) -> dict:
    """Midpoint IDW on a lon/lat grid, as the OpenSense sandbox does it."""
    import pycomlink as pycml

    interp = pycml.spatial.interpolator.IdwKdtreeInterpolator(
        nnear=nnear, p=p, exclude_nan=True, max_distance=max_distance)
    grid = interp(x=cml.lon_center, y=cml.lat_center, z=cml.R,
                  resolution=resolution)
    return {"grid": np.asarray(grid), "x": interp.xgrid, "y": interp.ygrid,
            "crs": "lonlat", "label": "pycomlink IDW (midpoint)"}


def _pynncml_link_set(cml: xr.Dataset):
    """Build a pynncml ``LinkSet`` carrying only geometry.

    The reconstruction methods need link endpoints, length, frequency and
    polarization; the RSL series is required by the constructor but is not
    used once rain rates are supplied directly, so a single dummy sample is
    passed rather than the full record.
    """
    from pynncml.datasets.link_data import Link, MetaData
    from pynncml.datasets.sensors_set import LinkSet

    links = []
    t = np.array([0.0, 1.0])
    for i in range(cml.sizes["cml_id"]):
        meta = MetaData(
            frequency=float(cml.frequency_ghz[i]),
            polarization=True,
            length=float(cml.length_km[i]),
            height_far=0.0, height_near=0.0,
            lon_lat_site_zero=[float(cml.site_0_lon[i]),
                               float(cml.site_0_lat[i])],
            lon_lat_site_one=[float(cml.site_1_lon[i]),
                              float(cml.site_1_lat[i])],
            force_zone_number=UTM_ZONE_NUMBER,
            force_zone_letter=UTM_ZONE_LETTER,
        )
        links.append(Link(link_rsl=np.zeros(2), time_array=t, meta_data=meta,
                          link_tsl=np.zeros(2)))
    return LinkSet(links)


def _pynncml_reconstruct(cml: xr.Dataset, kind: str, roi: float,
                         pixel_area: float, point_per_link: int) -> dict:
    import torch
    import pynncml as pnc

    link_set = _pynncml_link_set(cml)
    rain = torch.tensor(np.nan_to_num(np.asarray(cml.R, dtype=np.float32))
                        ).reshape(-1, 1)

    if kind == "idw":
        model = pnc.mcm.generate_link_set_idw(link_set, roi=roi,
                                              pixel_area=pixel_area)
        grid = model(rain)
        label = "pynncml IDW (midpoint)"
    else:
        model = pnc.mcm.generate_link_set_gmz(link_set, roi=roi,
                                              pixel_area=pixel_area,
                                              point_per_link=point_per_link)
        grid = model(rain)
        if isinstance(grid, (tuple, list)):
            grid = grid[0]
        label = f"pynncml GMZ ({point_per_link} pts/link)"

    arr = grid.detach().numpy() if hasattr(grid, "detach") else np.asarray(grid)
    arr = np.squeeze(arr)
    # GMZ keeps the grid on its inner IDW solver; plain IDW exposes it directly.
    holder = getattr(model, "base_idw", model)
    return {"grid": arr, "x": np.asarray(holder.x_grid_vector),
            "y": np.asarray(holder.y_grid_vector), "crs": "utm",
            "label": label}


def map_pynncml_idw(cml, roi: float = 2.0, pixel_area: float = 1.0) -> dict:
    return _pynncml_reconstruct(cml, "idw", roi, pixel_area, 0)


def map_pynncml_gmz(cml, roi: float = 2.0, pixel_area: float = 1.0,
                    point_per_link: int = 3) -> dict:
    return _pynncml_reconstruct(cml, "gmz", roi, pixel_area, point_per_link)


def map_mergeplg(cml: xr.Dataset, kind: str = "idw",
                 resolution_m: float = 500.0) -> dict:
    """Line-aware interpolation on a projected grid."""
    from mergeplg import interpolate

    x0, x1 = np.asarray(cml.site_0_x), np.asarray(cml.site_1_x)
    y0, y1 = np.asarray(cml.site_0_y), np.asarray(cml.site_1_y)
    pad = 2000.0
    xs = np.arange(min(x0.min(), x1.min()) - pad,
                   max(x0.max(), x1.max()) + pad, resolution_m)
    ys = np.arange(min(y0.min(), y1.min()) - pad,
                   max(y0.max(), y1.max()) + pad, resolution_m)
    xg, yg = np.meshgrid(xs, ys)

    da_grid = xr.DataArray(
        np.zeros(xg.shape), dims=("y", "x"),
        coords={"x": xs, "y": ys,
                "x_grid": (("y", "x"), xg), "y_grid": (("y", "x"), yg)})
    da_cml = xr.DataArray(
        np.nan_to_num(np.asarray(cml.R)), dims=("cml_id",),
        coords={c: cml[c] for c in ("cml_id", "site_0_x", "site_0_y",
                                    "site_1_x", "site_1_y", "x", "y")})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "idw":
            model = interpolate.InterpolateIDW(min_observations=1)
            model.update(da_cml=da_cml)
            grid = model.interpolate(da_grid, da_cml=da_cml, p=2,
                                     idw_method="radolan", nnear=8,
                                     max_distance=30000)
            label = "mergeplg IDW (line-aware)"
        else:
            model = interpolate.InterpolateOrdinaryKriging(
                discretization=8, min_observations=1)
            model.update(da_cml=da_cml)
            grid = model.interpolate(da_grid, da_cml=da_cml,
                                     variogram_model="spherical",
                                     nnear=8, full_line=True)
            label = "mergeplg block kriging (line-aware)"

    return {"grid": np.asarray(grid), "x": xs, "y": ys, "crs": "utm",
            "label": label}


METHODS = {
    "pycomlink_idw": map_pycomlink_idw,
    "pynncml_idw": map_pynncml_idw,
    "pynncml_gmz": map_pynncml_gmz,
    "mergeplg_idw": lambda c: map_mergeplg(c, "idw"),
    "mergeplg_kriging": lambda c: map_mergeplg(c, "kriging"),
}


# --------------------------------------------------------------------------
# upstream patch
# --------------------------------------------------------------------------
def patch_pynncml_gmz() -> list[str]:
    """Fix two defects in ``pynncml``'s GMZ bilinear interpolation.

    Both are in ``multiple_cmls_methods/rain_field_reconstruction/gmz.py``,
    in ``compute_rain_point_from_field``, at pynncml 0.3.7:

    **1. The ceiling index is unclamped.**
    ``i_ceiling = i * (1 - is_pos) + (i + 1) * is_pos`` reaches ``len(grid)``
    when a link point falls on or beyond the last grid node, and the call
    dies with ``IndexError: index 56 is out of bounds for dimension 0 with
    size 56``. Any link touching the edge of the bounding box built from the
    links themselves triggers it, which is why it fires on a full network.

    **2. The ceiling-ceiling corner reads the wrong axis.**
    ``cc = in_rain_map[:, j_ceiling, j_ceiling]`` uses the *y* index for both
    dimensions where it should be ``[:, i_ceiling, j_ceiling]``. Unlike the
    first, this does not raise - it silently returns a value from the wrong
    grid cell for one of the four bilinear corners, so GMZ output is wrong
    wherever ``i_ceiling != j_ceiling``, which is almost everywhere.

    This patches the function in memory. It is a local workaround, not a
    fix: both belong upstream at github.com/haihabi/PyNNcml.

    Returns the list of patches applied.
    """
    import torch
    from pynncml.multiple_cmls_methods.rain_field_reconstruction import gmz

    if getattr(gmz, "_fieldsense_patched", False):
        return []

    def compute_rain_point_from_field(self, in_rain_map):
        """Bilinear sample of the rain field at each link discretization point."""
        point_set = self.base_idw.point_set
        point_set_x, point_set_y = point_set[:, 0], point_set[:, 1]
        x_grid_vector = self.base_idw.x_grid_vector
        y_grid_vector = self.base_idw.y_grid_vector
        n_x, n_y = x_grid_vector.shape[0], y_grid_vector.shape[0]

        delta_x = point_set_x.unsqueeze(-1) - x_grid_vector.unsqueeze(0)
        delta_y = point_set_y.unsqueeze(-1) - y_grid_vector.unsqueeze(0)

        i = torch.argmin(torch.abs(delta_x), dim=1)
        is_pos = (torch.gather(delta_x, 1, i.reshape(-1, 1)) > 0).long().flatten()
        i_floor = (i * is_pos + (i - 1) * (1 - is_pos)).clamp(0, n_x - 2)
        i_ceiling = (i_floor + 1).clamp(1, n_x - 1)

        j = torch.argmin(torch.abs(delta_y), dim=1)
        js_pos = (torch.gather(delta_y, 1, j.reshape(-1, 1)) > 0).long().flatten()
        j_floor = (j * js_pos + (j - 1) * (1 - js_pos)).clamp(0, n_y - 2)
        j_ceiling = (j_floor + 1).clamp(1, n_y - 1)

        ff = in_rain_map[:, i_floor, j_floor]
        cf = in_rain_map[:, i_ceiling, j_floor]
        fc = in_rain_map[:, i_floor, j_ceiling]
        cc = in_rain_map[:, i_ceiling, j_ceiling]      # was j_ceiling twice

        dx = x_grid_vector[i_ceiling] - x_grid_vector[i_floor]
        dy = y_grid_vector[j_ceiling] - y_grid_vector[j_floor]
        w_x_floor = (point_set_x - x_grid_vector[i_floor]) / dx
        w_x_ceil = (x_grid_vector[i_ceiling] - point_set_x) / dx
        w_y_floor = (point_set_y - y_grid_vector[j_floor]) / dy
        w_y_ceil = (y_grid_vector[j_ceiling] - point_set_y) / dy

        rain_point = (w_x_floor * w_y_floor * ff + w_x_ceil * w_y_floor * cf
                      + w_x_floor * w_y_ceil * fc + w_x_ceil * w_y_ceil * cc)
        return rain_point.T

    gmz.GMZInterpolation.compute_rain_point_from_field = compute_rain_point_from_field
    gmz._fieldsense_patched = True
    return ["clamped i_ceiling/j_ceiling to the grid",
            "cc corner now reads [i_ceiling, j_ceiling], was [j_ceiling, j_ceiling]"]
