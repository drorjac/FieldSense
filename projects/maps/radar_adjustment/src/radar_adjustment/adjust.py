"""
The intercomparison: every radar adjustment of ``3_adjust_radar`` / ``3b_adjust_radar``.

Each task is one (mergeplg version, network, range-check set, variant, month): the merger
is built on the month's radar and links, as in the notebooks, and called hour by hour.
The radar is adjusted with the links only; the gauges are never used, so every gauge is
an independent check. A task saves (``FIELDS/<version>/<network>/<checks>/<variant>_<month>.nc``):

``at_gauges``  the adjusted field at each gauge (poligrain ``GridAtPoints``, nearest
               cell, as ``4_analysis``), (id, time)
``total``      the month's total of the adjusted field (y, x), for maps

Full hourly fields of OpenRainER's 290 x 373 grid would be ~1 GB per variant and month;
they are not kept, and the grid is cropped to the box around links and gauges plus 0.1 deg
(:func:`crop`), which leaves the values at the gauges unchanged. Tasks already on disk are skipped, so a run can be resumed.

Two versions of ``mergeplg``:

``pinned``  commit 9894b9c, the submodule the OpenSense repository records
``main``    the commit pinned in ``pyproject.toml`` (``mergeplg-main``). Its block kriging and
            block KED use a different right-hand side (``0.5 (var_within + nugget)`` in place
            of ``var_within``; identical for points) and its IDW default is ``standard``
            (1/d^2) where 9894b9c's is ``radolan``. The notebook passes no ``idw_method``, so
            each version runs with its own default. ``main`` reproduces the table printed in
            ``4_analysis`` (all eight methods within 0.001 in correlation and RMSE): the
            published results were made with a mergeplg newer than the recorded submodule.

Both are run in ``.venv-mergeplg-main``; ``pinned`` puts ``MERGEPLG_PINNED`` first on
``sys.path`` in its worker processes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from radar_adjustment.settings import CHECKS, FIELDS, MERGEPLG_PINNED, MONTHS, NNEAR, VARIANTS, VARIOGRAM

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    version: str          # "pinned" | "main"
    network: str
    checks: str           # key of CHECKS, or a free label for extension runs
    variant: str          # key of VARIANTS, or an extension product
    month: str
    kwargs: tuple = ()    # extra/overriding constructor arguments, as sorted items
    adjusters: str = "cml"  # "cml" | "pws" | "cml+pws"

    @property
    def path(self):
        tag = self.variant if self.adjusters == "cml" else f"{self.variant}@{self.adjusters}"
        return FIELDS / self.version / self.network / self.checks / f"{tag}_{self.month}.nc"


def checks_for(variant: str, checks: str) -> dict:
    """The notebook's choice: difference check for additive and KED, ratio check for multiplicative."""
    c = CHECKS[checks]
    if variant.startswith("mul"):
        return {"ratio_check": c["ratio_check"]} if "ratio_check" in c else {}
    return {"diff_check": c["diff_check"]} if "diff_check" in c else {}


def merger_kwargs(variant: str, checks: str | None) -> tuple[str, dict]:
    cls, method, full_line = VARIANTS[variant]
    kw = {"nnear": NNEAR}
    if checks is not None:
        kw["range_checks"] = checks_for(variant, checks)
    if cls == "MergeDifferenceIDW":
        kw["method"] = method            # idw_method: each version's default, as the notebook
    else:
        kw.update(variogram_parameters=dict(VARIOGRAM), full_line=full_line)
        if method:
            kw["method"] = method
    return cls, kw


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
def _init_worker(version: str):
    if version == "pinned":
        sys.path.insert(0, str(MERGEPLG_PINNED))
    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")


def _mergeplg_version() -> str:
    import mergeplg
    return getattr(mergeplg, "__version__", "?") + " @ " + mergeplg.__file__


def crop(rad: xr.DataArray, cml: xr.DataArray, points: xr.DataArray, pad_deg: float = 0.1):
    """The radar's rows and columns that hold any cell within the links' and points' box + pad.

    Every adjusted value at a cell depends only on the observations and that cell's radar
    value (kriging and IDW weights are computed cell by cell), so cropping the grid leaves
    the values at the gauges unchanged; it only shortens the per-cell loops.
    """
    lats = np.concatenate([cml.site_0_lat.values, cml.site_1_lat.values, points.lat.values])
    lons = np.concatenate([cml.site_0_lon.values, cml.site_1_lon.values, points.lon.values])
    inside = ((rad.lat >= lats.min() - pad_deg) & (rad.lat <= lats.max() + pad_deg)
              & (rad.lon >= lons.min() - pad_deg) & (rad.lon <= lons.max() + pad_deg)).values
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    return rad.isel({rad.dims[1]: slice(rows[0], rows[-1] + 1),
                     rad.dims[2]: slice(cols[0], cols[-1] + 1)})


def build_merger(cls_name: str, kwargs: dict, rad, cmls=None, gauges=None):
    from mergeplg import merge
    cls = getattr(merge, cls_name)
    return cls(ds_rad=rad, ds_cmls=cmls, ds_gauges=gauges, **kwargs)


def adjust_series(merger, rad, cmls=None, gauges=None, call_kwargs=None, radar_if_empty=False) -> list:
    """The merger called hour by hour, as in the notebooks (one adjusted field per hour).

    ``radar_if_empty``: an hour with no valid observation keeps the radar, as mergeplg's
    other mergers do below ``min_observations``; its RADOLAN code fails on such an hour.
    """
    out = []
    for t in rad.time.values:
        kw = dict(call_kwargs or {})
        if cmls is not None:
            kw["da_cmls"] = cmls.sel(time=t)
        if gauges is not None:
            kw["da_gauges"] = gauges.sel(time=t)
        if radar_if_empty and not any(np.isfinite(v.values).any() for k, v in kw.items() if k.startswith("da_")):
            out.append(np.asarray(rad.sel(time=t), dtype="float32"))
            continue
        f = merger(da_rad=rad.sel(time=t), **kw)
        if isinstance(f, xr.Dataset):                 # RADOLAN returns its products
            f = f["RW"] if "RW" in f else f[list(f.data_vars)[0]]
        out.append(np.asarray(f, dtype="float32").reshape(rad.shape[1:]))
    return out


def at_points(fields: np.ndarray, rad: xr.DataArray, points: xr.DataArray) -> np.ndarray:
    """Fields (time, y, x) at each point's nearest cell: poligrain GridAtPoints(nnear=1, "best")."""
    import poligrain as plg
    grid = rad.isel(time=slice(0, fields.shape[0])).copy(data=fields)
    pts = points.isel(time=slice(0, fields.shape[0]))
    g = plg.spatial.GridAtPoints(da_gridded_data=grid.isel(time=0), da_point_data=pts.isel(time=0),
                                 nnear=1, stat="best")
    return g(da_gridded_data=grid, da_point_data=pts).transpose("id", "time").values


def run_task(task: Task, product=None) -> str:
    """Compute and save one task. ``product`` overrides the variant lookup (extensions)."""
    from radar_adjustment.prepare import load
    if task.path.exists():
        return f"skip {task.path.name}"
    t0 = time.time()
    rad, cml_m, g_m = load(task.network, task.month)
    rad = crop(rad, cml_m, g_m)
    if product is not None:
        fields = product(task, rad, cml_m, g_m)
    else:
        cls, kw = merger_kwargs(task.variant, task.checks if task.checks in CHECKS else "default")
        kw.update(dict(task.kwargs))
        merger = build_merger(cls, kw, rad, cml_m)
        fields = np.stack(adjust_series(merger, rad, cml_m))
    vals = at_points(fields, rad, g_m)
    ds = xr.Dataset(
        {"at_gauges": (("id", "time"), vals.astype("float32")),
         "total": (rad.dims[1:], np.nansum(fields, axis=0).astype("float32")),
         "n_nan": ("time", np.isnan(fields).reshape(fields.shape[0], -1).sum(axis=1))},
        coords={"id": g_m.id.values, "time": rad.time.values})
    ds.attrs.update(mergeplg=_mergeplg_version(), seconds=round(time.time() - t0, 1),
                    kwargs=repr(dict(task.kwargs)), checks=task.checks, adjusters=task.adjusters)
    task.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.path.with_suffix(".part")
    ds.to_netcdf(tmp)
    tmp.rename(task.path)
    return f"done {task.path.relative_to(FIELDS)} in {time.time() - t0:.0f} s"


def _run(task):
    try:
        return run_task(task)
    except Exception as e:                       # noqa: BLE001 - one bad task must not stop the run
        log.exception("task failed: %s", task)
        return f"FAILED {task}: {type(e).__name__}: {e}"


def run_tasks(tasks: list[Task], workers: int = 6, func=_run) -> list[str]:
    """Tasks grouped by mergeplg version, each group in its own spawned pool."""
    results = []
    for version in ("pinned", "main"):
        todo = [t for t in tasks if t.version == version and not t.path.exists()]
        if not todo:
            continue
        log.info("%s: %d tasks", version, len(todo))
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(version,), maxtasksperchild=4) as pool:
            for msg in pool.imap_unordered(func, todo):
                log.info(msg)
                results.append(msg)
    return results


POINT_VARIANTS = ("add_p_ok", "mul_p_ok", "ked_p")


def intercomparison_tasks(networks, versions=("pinned", "main"), checks=tuple(CHECKS)) -> list[Task]:
    tasks = []
    for version in versions:
        for net in networks:
            for c in checks:
                for v in VARIANTS:
                    for m in MONTHS[net]:
                        tasks.append(Task(version, net, c, v, m))
    # OpenRainER's pinned run, which only the replication needs, is limited to the published
    # checks plus the point variants (identical on both versions; the main run reuses them)
    tasks = [t for t in tasks if not (t.version == "pinned" and t.network == "openrainer" and t.checks != "default"
                                      and t.variant not in POINT_VARIANTS)]
    # slowest first: OpenRainER's kriging
    return sorted(tasks, key=lambda t: (t.version != "pinned", t.network != "openrainer",
                                        "idw" in t.variant))


def run_intercomparison(networks, workers: int = 6, versions=("pinned", "main")):
    return run_tasks(intercomparison_tasks(networks, versions), workers)
