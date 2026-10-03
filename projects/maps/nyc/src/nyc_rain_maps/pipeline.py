"""End-to-end event pipeline: links -> rain rates -> maps -> radar comparison.

    from nyc_rain_maps.pipeline import run_event
    res = run_event("2023-12-27 12:00", "2023-12-28 18:00")
    res.map_scores            # NRMSE etc. per method, map vs MRMS
    res.link_scores           # per method and link, link vs radar along the path
    res.save("outputs/dec27") # NetCDF + CSV + figures

Everything is on one common, MRMS-aligned 0.01 deg grid and hourly (hour-ENDING)
accumulations, as the project README's scoring rules require, so every method is scored the same
way. Method definitions live in :data:`METHODS`; see ``docs/METHODS.md`` for
what each one is and where it came from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import xarray as xr

from core.cml.estimators import (ConstantBaselineSTD, DynamicBaseline, ManualWindows, NearbyLinks,
                             PycomlinkRSD, PyNNcmlGRU, RainEstimator)
from core.cml.link_qc import QCConfig, retrieval_qc, run_qc
from core.geo import NYC, OPENMESH, Domain, Grid
from core.maps.geometry import distance_to_links_km
from core.maps.scores import compare_links, compare_maps
from core.opensense import openmesh as om
from core.maps.idw import accumulate, idw_map
from core.radar.mrms import hourly_rainfall, to_grid
from core.radar.mrms import MRMSClient

log = logging.getLogger(__name__)

# The 8 sublinks both implementations selected (by eye) for their events.
SHARED8 = ["3/sublink_3", "8/sublink_1", "12/sublink_2", "20/sublink_2", "31/sublink_1",
           "40/sublink_1", "45/sublink_1", "34/sublink_1"]


@dataclass
class MethodSpec:
    """How to build and run one method."""

    factory: Callable[..., RainEstimator]
    implementation: str
    description: str
    needs_full_record: bool = False     # nearby-link / RSD thresholds use the long record
    needs_pws: bool = False
    optional: bool = False              # skipped (not an error) if its dependency is missing


METHODS: dict[str, MethodSpec] = {
    "impl1_dynamic": MethodSpec(
        lambda **_: DynamicBaseline(gap_fill="gauge_q99", nan_to_zero=True), "implementation_1",
        "Dynamic baseline (200 min, qd 1 dB), gaps filled with q99 attenuation while PWS wet, ITU 2003",
        needs_pws=True),
    "impl1_dynamic_maxfill": MethodSpec(
        lambda **_: DynamicBaseline(gap_fill="gauge_max", nan_to_zero=True), "implementation_1",
        "As impl1_dynamic with max-attenuation fill (what produced the committed map1/map2.pkl)",
        needs_pws=True),
    "impl2_classical_dynamic": MethodSpec(
        lambda **_: DynamicBaseline(gap_fill="min_rsl"), "implementation_2",
        "PyNNcml one-step dynamic baseline on min-RSL-filled series, ITU 2003"),
    "impl2_classical_constant": MethodSpec(
        lambda **_: ConstantBaselineSTD(gap_fill="min_rsl"), "implementation_2",
        "PyNNcml two-step: STD wet/dry (240, 1 dB) + constant baseline + 3 dB WAA, ITU 2003"),
    "impl2_pycomlink": MethodSpec(
        lambda full=None, **_: PycomlinkRSD(threshold_links=full), "implementation_2",
        "pycomlink RSD wet/dry (240 min, q90 of 8-month record) + constant baseline + Leijnse WAA, ITU 2005",
        needs_full_record=True),
    "impl2_pycomlink_linear": MethodSpec(
        lambda full=None, **_: PycomlinkRSD(threshold_links=full, baseline="linear"), "implementation_2",
        "As impl2_pycomlink with linear baseline", needs_full_record=True),
    "impl2_nearby": MethodSpec(
        lambda start=None, end=None, **_: NearbyLinks(start=start, end=end), "implementation_2",
        "Nearby-link wet/dry + reference level (Overeem 2016) on 15-min min/max, ITU 2005",
        needs_full_record=True),
    "impl2_gru": MethodSpec(
        lambda **_: PyNNcmlGRU(), "implementation_2",
        "PyNNcml pretrained two-step GRU on 15-min min/max (trained on 18-25 GHz OpenMRG)",
        optional=True),
}

DEFAULT_METHODS = ["impl1_dynamic", "impl2_classical_dynamic", "impl2_classical_constant",
                   "impl2_pycomlink", "impl2_nearby"]


@dataclass
class EventResult:
    start: pd.Timestamp
    end: pd.Timestamp
    links: xr.Dataset
    link_rain: dict = field(default_factory=dict)       # method -> Dataset (native step)
    link_hourly: dict = field(default_factory=dict)     # method -> DataArray (link, time) mm
    maps_hourly: dict = field(default_factory=dict)     # method -> DataArray (time, lat, lon) mm
    radar_hourly: xr.DataArray | None = None
    distance_to_link_km: xr.DataArray | None = None
    map_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    link_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    qc_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    qc_summary: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def event_totals(self) -> xr.Dataset:
        """Event accumulation maps (mm) for every method plus ``radar``."""
        data = {m: v.sum("time", min_count=1) for m, v in self.maps_hourly.items()}
        if self.radar_hourly is not None:
            data["radar"] = self.radar_hourly.sum("time", min_count=1)
        return xr.Dataset(data)

    def save(self, out_dir: str | Path, figures: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        maps = xr.Dataset({m: v for m, v in self.maps_hourly.items()})
        if self.radar_hourly is not None:
            maps["radar"] = self.radar_hourly
        if self.distance_to_link_km is not None:
            maps["distance_to_link_km"] = self.distance_to_link_km
        maps.attrs = {k: str(v) for k, v in self.config.items()}
        maps.to_netcdf(out / "maps_hourly.nc")
        xr.Dataset({m: v for m, v in self.link_hourly.items()}).drop_vars(
            [c for c in ["sublink_id", "polarization"] if c in self.links.coords], errors="ignore"
        ).to_netcdf(out / "links_hourly.nc")
        self.map_scores.to_csv(out / "map_scores.csv")
        self.link_scores.to_csv(out / "link_scores.csv")
        self.qc_log.to_csv(out / "qc_log.csv", index=False)
        (out / "summary.json").write_text(json.dumps(
            {"start": str(self.start), "end": str(self.end), "config": self.config,
             "qc": self.qc_summary, "errors": self.errors}, indent=2, default=str))
        if figures:
            from . import plots as plotting
            plotting.event_comparison_figure(self, out / "event_totals.png")
        return out


def _distance_to_links(grid: Grid, links: xr.Dataset) -> xr.DataArray:
    """Distance (km) from each grid cell to the nearest link path (sampled every 100 m)."""
    glat, glon = grid.mesh()
    return xr.DataArray(distance_to_links_km(glat, glon, links, step_m=100), dims=("lat", "lon"),
                        coords={"lat": grid.lat, "lon": grid.lon}, name="distance_to_link_km")


def select_links(link_set, start, end, qc: QCConfig | None = None):
    """Resolve a link-set spec to labels. Returns (labels, qc_log, qc_summary)."""
    if isinstance(link_set, (list, tuple)):
        return list(link_set), pd.DataFrame(), {}
    if link_set == "shared8":
        return list(SHARED8), pd.DataFrame(), {}
    if link_set == "all":
        return list(om.sublinks_table().index), pd.DataFrame(), {}
    if link_set == "selected":
        from .link_selection import load_or_select
        sel = load_or_select()
        return list(sel.selected), pd.DataFrame(), {"selection": sel.summary()}
    if link_set == "qc":
        table = om.sublinks_table()
        links = om.load_links(start, end, list(table.index))
        keep, qlog, summary = run_qc(table, links, qc or QCConfig())
        return keep, qlog, summary
    raise ValueError(f"unknown link_set {link_set!r} (list, 'shared8', 'selected', 'qc', 'all')")


def run_event(start, end, methods=None, link_set="shared8", domain: Domain = OPENMESH,
              res_deg: float = 0.01, idw: dict | None = None, spinup: str = "6h",
              near_km: float = 2.0, manual_windows: dict | None = None,
              client: MRMSClient | None = None, qc: QCConfig | None = None) -> EventResult:
    """Run every requested method for one event and score it against MRMS.

    Parameters
    ----------
    start, end:
        Event window (UTC). Hourly scores cover hours ending in ``(start, end]``.
    methods:
        Names from :data:`METHODS` (default :data:`DEFAULT_METHODS`); add
        ``"impl2_manual"`` together with ``manual_windows={"dry": (t0, t1), "rain": (t0, t1)}``.
    link_set:
        ``"shared8"`` (default; the implementations' links), ``"selected"`` (the
        cherry-picking pipeline, :mod:`nyc_rain_maps.link_selection`), ``"qc"`` (automatic
        metadata/time-series QC on all 103 sublinks), ``"all"``, or a list of labels.
    spinup:
        Extra signal loaded before ``start`` so baselines are established (the dynamic
        baseline needs 200 min of history). The implementations used none.
    idw:
        IDW options for :func:`~nyc_rain_maps.idw.idw_map` (default power 2,
        radius 10 km, all links, NaN excluded).
    near_km:
        Map scores are also reported for cells within this distance of a link.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    methods = list(methods or DEFAULT_METHODS)
    idw = {"power": 2.0, "radius_m": 10_000.0, "nnear": None, "nan_policy": "exclude", **(idw or {})}
    client = client or MRMSClient()

    labels, qc_log, qc_summary = select_links(link_set, start, end, qc)
    t_load = start - pd.Timedelta(spinup)
    links = om.load_links(t_load, end, labels)
    full = None
    if any(METHODS[m].needs_full_record for m in methods if m in METHODS):
        full = om.load_links(*om.open_openmesh().time.values[[0, -1]], labels)
    pws = om.load_pws(t_load - pd.Timedelta("1h"), end) if any(
        METHODS[m].needs_pws for m in methods if m in METHODS) else None

    grid = Grid.from_domain(domain, res_deg)
    res = EventResult(start=start, end=end, links=links, qc_log=qc_log, qc_summary=qc_summary,
                      config={"methods": methods, "link_set": link_set if isinstance(link_set, str) else "custom",
                              "n_links": len(labels), "domain": domain.name, "res_deg": res_deg,
                              "idw": idw, "spinup": spinup, "near_km": near_km})

    for m in methods:
        try:
            if m == "impl2_manual":
                if not manual_windows:
                    raise ValueError("impl2_manual needs manual_windows={'dry':..., 'rain':...}")
                est, src = ManualWindows(gap_fill="min_rsl", **manual_windows), links
            else:
                spec = METHODS[m]
                est = spec.factory(full=full, start=str(t_load), end=str(end))
                src = full if isinstance(est, NearbyLinks) else links
            out = est.estimate(src, pws=pws)
            out = out.sel(time=slice(start + pd.Timedelta("1min"), end))
            res.link_rain[m] = out
            hourly = accumulate(out["rain"], "1h").sel(time=slice(start + pd.Timedelta("1h"), end))
            res.link_hourly[m] = hourly
            res.maps_hourly[m] = idw_map(hourly, grid, **idw)
        except ImportError as exc:
            res.errors[m] = f"skipped, missing dependency: {exc}"
            log.warning("%s skipped: %s", m, exc)
        except Exception as exc:  # keep the other methods running; report the failure
            if m in METHODS and METHODS[m].optional:
                res.errors[m] = f"skipped: {exc}"
            else:
                res.errors[m] = f"failed: {exc!r}"
            log.exception("method %s failed", m)

    # Radar is always cut from the NYC crop when the domain fits in it, so one cache
    # (e.g. the event-catalog record) serves every sub-domain.
    inside = (domain.lat_min >= NYC.lat_min and domain.lat_max <= NYC.lat_max
              and domain.lon_min >= NYC.lon_min and domain.lon_max <= NYC.lon_max)
    radar = hourly_rainfall(start, end, NYC if inside else domain, client=client)
    # exact cell match (grid is MRMS-aligned at 0.01 deg); cells outside the crop stay NaN
    if res_deg <= 0.0101:
        res.radar_hourly = radar.reindex(lat=grid.lat, lon=grid.lon, method="nearest",
                                         tolerance=res_deg / 2)
    else:
        res.radar_hourly = to_grid(radar, grid)
    res.distance_to_link_km = _distance_to_links(grid, links)
    near = res.distance_to_link_km <= near_km

    rows, lrows = [], []
    table = om.links_frame(links)
    for m, mp in res.maps_hourly.items():
        _, pooled = compare_maps(mp, res.radar_hourly)
        _, pooled_near = compare_maps(mp.where(near), res.radar_hourly.where(near))
        row = {"method": m, "implementation": METHODS[m].implementation if m in METHODS else "implementation_2",
               **{k: v for k, v in pooled.items() if not isinstance(v, dict)},
               **{f"near_{k}": v for k, v in pooled_near.items() if not isinstance(v, dict)},
               **{f"total_{k}": v for k, v in pooled["event_total_scores"].items()}}
        rows.append(row)
        ls = compare_links(res.link_hourly[m], res.radar_hourly, table)
        ls.insert(0, "method", m)
        lrows.append(ls)
        flags = retrieval_qc(res.link_rain[m]["rain"])
        if len(flags):
            flags.insert(0, "method", m)
            res.qc_log = pd.concat([res.qc_log, flags], ignore_index=True)
    res.map_scores = pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()
    res.link_scores = pd.concat(lrows) if lrows else pd.DataFrame()
    return res
