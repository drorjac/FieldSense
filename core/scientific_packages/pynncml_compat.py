"""
Workarounds for PyNNcml 0.3.7 in this environment, in one place.

Each is a local patch applied in memory, not a fix: all three belong
upstream at github.com/haihabi/PyNNcml.

``openmrg_data_path``   point the OpenMRG loader at the repository's archive
                        instead of re-downloading and re-extracting 4.6 GB
``patch_numpy2``        gauge distances call ``math.sqrt`` on one-element
                        arrays, an error under NumPy 2
``patch_pynncml_gmz``   two defects in GMZ's bilinear interpolation

``apply()`` applies the last two; every function is idempotent.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENMRG_RAW = REPO_ROOT / "dataset/open_datasets/OpenMRG_Sweden/raw"
OPENMRG_EXTRACTED = OPENMRG_RAW / "extracted"


def quietly(fn, *args, **kwargs):
    """Call a PyNNcml loader without its chatter.

    It prints "File already exists" on stdout and a progress bar per link on
    stderr - about 700 lines of notebook output for one call.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def apply() -> list[str]:
    """Apply every in-memory patch; returns what was patched this call."""
    return patch_numpy2() + patch_pynncml_gmz()


def openmrg_data_path() -> str:
    """Make PyNNcml read the repository's OpenMRG archive in place.

    ``pynncml.datasets.load_open_mrg`` downloads to ``./data/`` relative to
    the working directory, reads the gauge CSVs from that same folder, and
    extracts the whole 4.6 GB archive again on every call. Here it is given
    the already-extracted folder (``python -m core.opensense.fetch --dataset
    openmrg``), finds the zip there through a symlink, and skips the
    re-extraction. Nothing is copied.
    """
    from pynncml.datasets import loaders

    if not (OPENMRG_EXTRACTED / "cml" / "cml.nc").exists():
        raise FileNotFoundError(
            f"{OPENMRG_EXTRACTED}/cml/cml.nc missing - run "
            f"python -m core.opensense.fetch --dataset openmrg and extract it")
    zip_link = OPENMRG_EXTRACTED / "OpenMRG.zip"
    if not zip_link.exists():
        zip_link.symlink_to(OPENMRG_RAW / "OpenMRG.zip")

    if not getattr(loaders, "_fieldsense_repo_archive", False):
        original = loaders.transform_open_mrg

        def transform(fn, path):
            with mock.patch.object(zipfile.ZipFile, "extractall", lambda *a, **k: None):
                return original(fn, path)

        loaders.transform_open_mrg = transform
        loaders._fieldsense_repo_archive = True
    return str(OPENMRG_EXTRACTED) + "/"


def patch_numpy2() -> list[str]:
    """Make ``PointSet.find_near_gauge(s)`` work under NumPy 2.

    ``PointSensor`` stores its UTM ``x``/``y`` as one-element arrays, and the
    two nearest-gauge searches pass them to ``math.sqrt``, which NumPy 2
    refuses ("only 0-dimensional arrays can be converted to Python
    scalars"). ``loader_open_mrg_dataset`` fails on its first link.
    """
    from pynncml.datasets import sensors_set

    if getattr(sensors_set, "_fieldsense_numpy2", False):
        return []

    def _distances(self, xy_center):
        x = np.array([float(np.ravel(g.x)[0]) for g in self.point_set])
        y = np.array([float(np.ravel(g.y)[0]) for g in self.point_set])
        return np.hypot(float(xy_center[0]) - x, float(xy_center[1]) - y)

    def find_near_gauge(self, xy_center):
        d = _distances(self, xy_center)
        return float(d.min()), [self.point_set[int(np.argmin(d))]]

    def find_near_gauges(self, xy_center, range_limit):
        d = _distances(self, xy_center)
        order = [i for i in np.argsort(d) if d[i] < range_limit]
        return [float(d[i]) for i in order], [self.point_set[i] for i in order]

    sensors_set.PointSet.find_near_gauge = find_near_gauge
    sensors_set.PointSet.find_near_gauges = find_near_gauges
    sensors_set._fieldsense_numpy2 = True
    return ["PointSet.find_near_gauge(s): vectorized, NumPy 2 safe"]


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

    gmz._fieldsense_original = gmz.GMZInterpolation.compute_rain_point_from_field
    gmz.GMZInterpolation.compute_rain_point_from_field = compute_rain_point_from_field
    gmz._fieldsense_patched = True
    return ["clamped i_ceiling/j_ceiling to the grid",
            "cc corner now reads [i_ceiling, j_ceiling], was [j_ceiling, j_ceiling]"]


def unpatch_pynncml_gmz() -> bool:
    """Restore pynncml's original GMZ interpolation; True if it was patched.

    For reproducing results computed with the unpatched library in the same
    process as corrected ones - the patch is otherwise sticky.
    """
    from pynncml.multiple_cmls_methods.rain_field_reconstruction import gmz

    if not getattr(gmz, "_fieldsense_patched", False):
        return False
    gmz.GMZInterpolation.compute_rain_point_from_field = gmz._fieldsense_original
    gmz._fieldsense_patched = False
    return True
