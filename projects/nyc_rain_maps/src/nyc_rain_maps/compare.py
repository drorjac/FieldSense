"""Multi-sensor rain-map comparison: CML vs MRMS radar vs PWS gauges vs ASOS.

Follows what the original implementations compared, on one footing:

* implementation_1 ``compare_cml_pws_radar_5row`` / ``_ogc``: CML map, PWS-IDW map and
  radar side by side, plus difference maps and a three-way MAE/bias table
  (CML-PWS, CML-radar, PWS-radar); gauges filtered by ``filter_gauges_by_accumulation``
  (keep the 5th-95th percentile of event totals); ``extract_at_gauges`` for point checks.
* implementation_2 ``CML_maps.ipynb``: a snapshot grid - rows = sensors/methods including a
  PWS row with its own interpolator, columns = hours around the event peak.

Every sensor is turned into hourly (hour-ENDING) accumulations on the same MRMS-aligned grid
with the same IDW, then compared pairwise on jointly valid cells. ASOS is kept *out* of every
map and used as the independent point reference.

    from nyc_rain_maps.compare import compare_sensors
    comp = compare_sensors("2024-01-09 16:00", "2024-01-10 11:00")
    comp.pairwise()           # every map vs every other map
    comp.points()             # every map at the ASOS stations vs ASOS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from .scores import scores
from core.opensense import openmesh as om
from core.asos import NYC_STATIONS, fetch_asos, hourly_precip
from .idw import idw_map

log = logging.getLogger(__name__)

RADAR = "MRMS"
PWS = "PWS"


def pws_hourly(start, end, domain=None, qc: str = "percentile", lo_pct: float = 5,
               hi_pct: float = 95, min_coverage: float = 0.8, pws: xr.Dataset | None = None
               ) -> xr.DataArray:
    """Hourly PWS accumulations (mm, hour-ending) for hours ending in ``(start, end]``.

    An hour needs ``min_coverage`` of its twelve 5-min samples. ``qc="percentile"``
    applies implementation_1's ``filter_gauges_by_accumulation``: keep stations whose
    event total lies within the ``lo_pct``-``hi_pct`` percentiles (stuck, dead and
    runaway gauges drop out). Returns ``(station, time)`` with ``lat``/``lon`` coords.
    """
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    pws = pws if pws is not None else om.load_pws(t0, t1)
    r = pws["rain"].resample(time="1h", label="right", closed="right")
    h = r.sum(min_count=1).where(r.count() >= min_coverage * 12)
    h = h.sel(time=slice(t0 + pd.Timedelta("1h"), t1))
    if domain is not None:
        h = h.isel(station=domain.contains(h.lat.values, h.lon.values))
    kept_note = "none"
    if qc == "percentile" and h.sizes["station"] > 4:
        tot = h.sum("time", min_count=1)
        lo, hi = np.nanpercentile(tot, [lo_pct, hi_pct])
        keep = (tot >= lo) & (tot <= hi)
        kept_note = f"event totals within p{lo_pct:g}-p{hi_pct:g} ({lo:.1f}-{hi:.1f} mm)"
        h = h.isel(station=keep.values)
    h.attrs = {"units": "mm", "qc": kept_note, "time_label": "end of hour (UTC)"}
    return h.rename("PWS")


def points_to_map(values: xr.DataArray, grid: Grid, **idw) -> xr.DataArray:
    """IDW map from point sensors ``values(station, time)`` (implementation_2: own interpolator)."""
    da = values.rename(station="link").assign_coords(mid_lat=("link", values.lat.values),
                                                    mid_lon=("link", values.lon.values))
    return idw_map(da, grid, **idw)


def asos_hourly_points(start, end, domain=None, stations=NYC_STATIONS) -> xr.DataArray:
    """ASOS hourly precipitation (mm) as ``(station, time)`` with lat/lon, hour-ending."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    a = fetch_asos(t0 - pd.Timedelta("1h"), t1 + pd.Timedelta("1h"), stations)
    h = hourly_precip(a).loc[t0 + pd.Timedelta("1h"):t1]
    coords = a.groupby("station")[["lat", "lon"]].first().loc[h.columns]
    da = xr.DataArray(h.values.T.astype("float32"), dims=("station", "time"),
                      # ns: pandas 3 reads CSV times in us, which xarray converts with a warning
                      coords={"station": list(h.columns), "time": h.index.values.astype("datetime64[ns]"),
                              "lat": ("station", coords.lat.values), "lon": ("station", coords.lon.values)},
                      name="ASOS", attrs={"units": "mm"})
    if domain is not None:
        da = da.isel(station=domain.contains(da.lat.values, da.lon.values))
    return da


@dataclass
class SensorComparison:
    """Hourly maps of every sensor on one grid, plus point sensors, for one event."""

    start: pd.Timestamp
    end: pd.Timestamp
    maps: dict                                  # name -> (time, lat, lon) mm
    pws_points: xr.DataArray                    # (station, time) mm - used in the PWS map
    asos_points: xr.DataArray                   # (station, time) mm - independent reference
    links: xr.Dataset
    distance_to_link_km: xr.DataArray | None = None
    info: dict = field(default_factory=dict)

    @property
    def names(self) -> list:
        return list(self.maps)

    def _joint(self):
        times = None
        for m in self.maps.values():
            times = m.time.values if times is None else np.intersect1d(times, m.time.values)
        return times

    def pairwise(self, near_km: float | None = None, wet_threshold: float = 0.1) -> pd.DataFrame:
        """Scores of every ordered pair (estimate vs reference), on jointly valid cell-hours.

        Pairs follow implementation_1's table: each CML map vs MRMS and vs PWS, and PWS vs
        MRMS. ``near_km`` restricts to cells within that distance of a link.
        """
        times = self._joint()
        mask = None
        if near_km is not None and self.distance_to_link_km is not None:
            mask = (self.distance_to_link_km <= near_km).values
        rows = []
        cml = [n for n in self.names if n not in (RADAR, PWS)]
        pairs = [(c, RADAR) for c in cml] + [(c, PWS) for c in cml] + [(PWS, RADAR)]
        for est, ref in pairs:
            if est not in self.maps or ref not in self.maps:
                continue
            e = self.maps[est].sel(time=times).values
            r = self.maps[ref].sel(time=times).values
            if mask is not None:
                e, r = e[:, mask], r[:, mask]
            rows.append({"estimate": est, "reference": ref, **scores(e, r, wet_threshold)})
        return pd.DataFrame(rows)

    def points(self, wet_threshold: float = 0.1) -> pd.DataFrame:
        """Every map sampled at the ASOS stations vs ASOS hourly totals (independent check)."""
        rows = []
        a = self.asos_points
        for name, m in self.maps.items():
            if a.sizes.get("station", 0) == 0:
                break
            smp = m.sel(lat=xr.DataArray(a.lat.values, dims="station"),
                        lon=xr.DataArray(a.lon.values, dims="station"), method="nearest")
            times = np.intersect1d(smp.time.values, a.time.values)
            est = smp.sel(time=times).transpose("station", "time").values
            ref = a.sel(time=times).values
            rows.append({"map": name, "stations": ",".join(map(str, a.station.values)),
                         **scores(est, ref, wet_threshold)})
        return pd.DataFrame(rows)

    def totals(self) -> xr.Dataset:
        times = self._joint()
        return xr.Dataset({n: m.sel(time=times).sum("time", min_count=1) for n, m in self.maps.items()})

    def peak_time(self) -> pd.Timestamp:
        r = self.maps[RADAR]
        return pd.Timestamp(r.mean(("lat", "lon")).idxmax("time").values)

    def save(self, out_dir: str | Path, figures: bool = True) -> Path:
        from . import plots as plotting
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.pairwise().round(4).to_csv(out / "sensor_pairwise.csv", index=False)
        self.pairwise(near_km=2.0).round(4).to_csv(out / "sensor_pairwise_near_links.csv", index=False)
        self.points().round(4).to_csv(out / "sensor_points_asos.csv", index=False)
        if figures:
            plotting.sensor_snapshot_figure(self, path=out / "sensor_snapshots.png")
            plotting.sensor_totals_figure(self, path=out / "sensor_totals.png")
            plotting.sensor_series_figure(self, path=out / "sensor_series.png")
        return out


def compare_sensors(start, end, methods=("impl1_dynamic", "impl2_pycomlink"), link_set="selected",
                    res=None, pws_qc: str = "percentile", **run_kw) -> SensorComparison:
    """Build a :class:`SensorComparison` for one event.

    ``res``: an existing :class:`~nyc_rain_maps.pipeline.EventResult` to reuse (its methods
    are taken as the CML maps); otherwise :func:`~nyc_rain_maps.pipeline.run_event` is run
    with ``methods`` and ``link_set``.
    """
    from .pipeline import run_event

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if res is None:
        res = run_event(t0, t1, methods=list(methods), link_set=link_set, **run_kw)
    radar = res.radar_hourly
    grid = Grid(radar.lat.values, radar.lon.values)
    idw = res.config.get("idw", {"power": 2.0, "radius_m": 10_000.0})
    from core.geo import Domain
    dom = Domain(float(grid.lat.min()), float(grid.lat.max()), float(grid.lon.min()), float(grid.lon.max()))

    ph = pws_hourly(t0, t1, domain=dom.pad(0.05), qc=pws_qc)
    maps = {RADAR: radar, PWS: points_to_map(ph, grid, **idw)}
    maps.update({f"CML {m}": v for m, v in res.maps_hourly.items()})
    try:
        asos = asos_hourly_points(t0, t1, domain=dom.pad(0.02))
    except Exception as exc:  # IEM hiccup must not break the comparison
        log.warning("ASOS unavailable: %s", exc)
        asos = xr.DataArray(np.empty((0, 0)), dims=("station", "time"))
    return SensorComparison(t0, t1, maps, ph, asos, res.links, res.distance_to_link_km,
                            info={"link_set": res.config.get("link_set"), "n_links": res.config.get("n_links"),
                                  "pws_stations": int(ph.sizes["station"]), "pws_qc": ph.attrs.get("qc"),
                                  "asos_stations": list(map(str, asos.station.values)) if asos.size else []})


def pairwise_matrix(pairwise: pd.DataFrame, metric: str = "nrmse") -> pd.DataFrame:
    """Pairwise table as an estimate x reference matrix of ``metric``."""
    return pairwise.pivot(index="estimate", columns="reference", values=metric)


__all__ = ["SensorComparison", "compare_sensors", "pws_hourly", "points_to_map",
           "asos_hourly_points", "pairwise_matrix", "RADAR", "PWS"]
