"""One event on one network: every sensor mapped on one grid, then compared.

    from multisensor_maps.event import run_event
    res = run_event("openrainer", "2021-09-26 09:00", "2021-09-27 06:00")
    res.pairwise()        # every map against every other, jointly valid cell-hours
    res.point_check()     # radar, link maps and a leave-one-out gauge map at each gauge

Common footing (as in ``projects/maps/nyc``): hour-ending accumulations; links and
point gauges interpolated with the same IDW (power 2, 10 km) onto the network's regular
lat/lon grid, which the radar is also on; scores only where both maps have a value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from core.cml.estimators import study_estimator as make_estimator
from core.cml.link_qc import QCConfig, metadata_qc, timeseries_qc
from core.geo import Grid, haversine_m
from core.maps.gmz import gmz_map
from core.maps.idw import accumulate, idw_map
from core.maps.scores import scores
from core.opensense.networks import NETWORKS, points_near_links, radar_along_links

from .settings import CHECK_POINTS, INTERPOLATORS, MAP_POINTS, METHODS

log = logging.getLogger(__name__)

RADAR = "radar"
IDW = {"power": 2.0, "radius_m": 10_000.0}


def interpolate(name: str, link_hourly: xr.DataArray, grid: Grid) -> xr.DataArray:
    """Link totals to a field: ``idw`` from each midpoint; ``line`` from virtual gauges along
    each path; ``gmz`` the same gauges corrected to each link's average (core.maps.gmz). All
    with power 2 within 10 km, so the difference is the geometry, not the radius."""
    if name == "idw":
        return idw_map(link_hourly, grid, **IDW)
    if name == "line":
        return gmz_map(link_hourly, grid, n_iter=0, **IDW)
    if name == "gmz":
        return gmz_map(link_hourly, grid, n_iter=10, **IDW)
    raise ValueError(name)


def select_links(net, start, end, spinup: str = "24h", qc: QCConfig | None = None):
    """Metadata QC on the network table, then time-series QC over the event window."""
    cfg = qc or QCConfig(domain=net.domain.pad(0.05))
    keep, _ = metadata_qc(net.links_table(), cfg)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    links = net.links(t0 - pd.Timedelta(spinup), t1, keep)
    keep2, _ = timeseries_qc(links.sel(time=slice(t0, t1)), cfg)
    return links.sel(link=keep2)


def distance_to_links(grid: Grid, links: xr.Dataset | pd.DataFrame, step_m: float = 200.0) -> xr.DataArray:
    """Distance (km) from each grid cell to the nearest link path."""
    glat, glon = grid.mesh()
    best = np.full(glat.shape, np.inf)
    get = (lambda c: links[c].values) if isinstance(links, xr.Dataset) else (lambda c: links[c].to_numpy())
    for la0, lo0, la1, lo1 in zip(get("site_0_lat"), get("site_0_lon"), get("site_1_lat"), get("site_1_lon")):
        n = max(2, int(haversine_m(la0, lo0, la1, lo1) / step_m) + 1)
        for s in np.linspace(0, 1, n):
            best = np.minimum(best, haversine_m(glat, glon, la0 + s * (la1 - la0), lo0 + s * (lo1 - lo0)))
    return xr.DataArray(best / 1000.0, dims=("lat", "lon"), coords={"lat": grid.lat, "lon": grid.lon})


def points_map(values: xr.DataArray, grid: Grid) -> xr.DataArray:
    """IDW map from point gauges ``(station, time)`` with the links' IDW settings."""
    da = values.rename(station="link").assign_coords(mid_lat=("link", values.lat.values),
                                                    mid_lon=("link", values.lon.values))
    return idw_map(da, grid, **IDW)


def loo_at_points(values: xr.DataArray) -> xr.DataArray:
    """Leave-one-out IDW: each station estimated from all the others, ``(station, time)``."""
    lat0, lon0 = float(values.lat.mean()), float(values.lon.mean())
    from core.geo import to_local_xy
    x, y = to_local_xy(values.lat.values, values.lon.values, lat0, lon0)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)                     # the station itself never counts
    with np.errstate(divide="ignore"):
        W = 1.0 / d ** IDW["power"]
    W[d > IDW["radius_m"]] = 0.0
    V = values.transpose("station", "time").values
    ok = np.isfinite(V)
    num, den = W @ np.where(ok, V, 0.0), W @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = np.where(den > 0, num / den, np.nan)
    return values.copy(data=est.astype("float32"))


def sample_at(field_: xr.DataArray, lat, lon) -> xr.DataArray:
    """Map value in the grid cell containing each point, ``(station, time)``."""
    s = field_.sel(lat=xr.DataArray(np.asarray(lat), dims="station"),
                   lon=xr.DataArray(np.asarray(lon), dims="station"), method="nearest")
    return s.transpose("station", "time")


@dataclass
class EventResult:
    network: str
    start: pd.Timestamp
    end: pd.Timestamp
    links: xr.Dataset
    maps: dict                                  # name -> (time, lat, lon) mm
    points: dict                                # point set -> (station, time) mm
    link_hourly: dict                           # method -> (link, time) mm
    distance_km: xr.DataArray
    errors: dict = field(default_factory=dict)

    @property
    def cml_names(self) -> list:
        """Link maps, named ``"<retrieval> <interpolation>"``."""
        return [m for m in self.maps if m not in (RADAR, *self.points)]

    def _times(self):
        times = None
        for m in self.maps.values():
            times = m.time.values if times is None else np.intersect1d(times, m.time.values)
        return times

    def pairwise(self, near_km: float | None = 2.0, wet: float = 0.1) -> pd.DataFrame:
        """Every map against the radar and every gauge map; the gauge maps against the radar."""
        t = self._times()
        mask = (self.distance_km <= near_km).values if near_km is not None else None
        refs = [RADAR] + [p for p in self.points if p in self.maps]
        rows = []
        for ref in refs:
            for est in [n for n in self.maps if n != ref and not (n in self.points and ref in self.points
                                                                 and refs.index(n) < refs.index(ref))]:
                if est == RADAR:
                    continue
                e = self.maps[est].sel(time=t).values
                r = self.maps[ref].sel(time=t).values
                if mask is not None:
                    e, r = e[:, mask], r[:, mask]
                rows.append({"network": self.network, "estimate": est, "reference": ref,
                             "near_km": near_km, **scores(e, r, wet)})
        return pd.DataFrame(rows)

    def point_check(self, check: str | None = None, near_km: float | None = None,
                    wet: float = 0.1) -> pd.DataFrame:
        """At each gauge of ``check``: radar, link maps, other gauge maps and the
        leave-one-out gauge map, against the gauge itself (hourly)."""
        check = check or CHECK_POINTS[self.network]
        g = self.points[check]
        if near_km is not None:
            d = self.distance_km.sel(lat=xr.DataArray(g.lat.values, dims="station"),
                                     lon=xr.DataArray(g.lon.values, dims="station"), method="nearest")
            g = g.isel(station=np.flatnonzero(d.values <= near_km))
        if g.sizes["station"] == 0:
            return pd.DataFrame()
        t = np.intersect1d(self._times(), g.time.values)
        ref = g.sel(time=t).values
        est = {f"{check} (leave-one-out)": loo_at_points(self.points[check]).sel(station=g.station, time=t)}
        for name, m in self.maps.items():
            if name == check:
                continue
            est[name] = sample_at(m.sel(time=t), g.lat.values, g.lon.values)
        return pd.DataFrame([{"network": self.network, "check": check, "estimate": k,
                              "stations": int(g.sizes["station"]), **scores(v.values, ref, wet)}
                             for k, v in est.items()])

    def link_check(self, wet: float = 0.1) -> pd.DataFrame:
        """Each method's hourly link totals against radar along the path and gauges near it."""
        radar_path = radar_along_links(self.maps[RADAR], self.links)
        near = {p: points_near_links(v, self.links) for p, v in self.points.items()}
        rows = []
        for m, lh in self.link_hourly.items():
            for ref_name, ref in [(RADAR, radar_path), *near.items()]:
                t = np.intersect1d(lh.time.values, ref.time.values)
                rows.append({"network": self.network, "method": m, "reference": ref_name,
                             **scores(lh.sel(time=t).values, ref.sel(link=lh.link, time=t).values, wet)})
        return pd.DataFrame(rows)


def run_event(network: str, start, end, methods=METHODS, spinup: str = "24h",
              extra: dict | None = None, interpolators=INTERPOLATORS) -> EventResult:
    """Everything for one event. ``extra``: ``{name: f(links) -> rain(link, time)}`` for other
    methods - mm/h at the links' 1-minute step, or hour-ending mm with ``attrs["hourly"]``
    (``core.cml.rnn.HourlyRNN``)."""
    net = NETWORKS[network]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    grid = net.grid
    links = select_links(net, t0, t1, spinup)
    log.info("%s %s: %d links", network, t0, links.sizes["link"])

    maps, link_hourly, errors = {}, {}, {}
    radar = net.radar_hourly(t0, t1)
    maps[RADAR] = radar
    points = net.points_hourly(t0, t1)
    for p in MAP_POINTS[network]:
        if p in points and points[p].sizes["station"] >= 3:
            maps[p] = points_map(points[p], grid)

    if links.sizes["link"] < 2:                # network outage: radar and gauges only
        errors["links"] = f"{links.sizes['link']} link(s) pass QC - no link maps"
        methods, extra = (), None
    record = None
    if "pycomlink" in methods:
        record = net.links(t0 - pd.Timedelta("10D"), t1, list(links.link.values))
        record.attrs.update(start=str(t0 - pd.Timedelta("10D")), end=str(t1))
    estimators = {m: (lambda L, m=m: make_estimator(m, t0 - pd.Timedelta(spinup), t1, record)
                      .estimate(L)["rain"]) for m in methods}
    estimators.update(extra or {})
    for m, f in estimators.items():
        try:
            rain = f(links)
            if rain.attrs.get("hourly"):              # already hour-ending totals (the RNN)
                hourly = rain.sel(time=slice(t0 + pd.Timedelta("1h"), t1))
            else:
                rain = rain.sel(time=slice(t0 + pd.Timedelta("1min"), t1))
                hourly = accumulate(rain, "1h").sel(time=slice(t0 + pd.Timedelta("1h"), t1))
            link_hourly[m] = hourly
            for interp in interpolators:
                maps[f"{m} {interp}"] = interpolate(interp, hourly, grid)
        except Exception as exc:              # keep the other methods running
            errors[m] = repr(exc)
            log.exception("%s failed on %s %s", m, network, t0)
    return EventResult(network, t0, t1, links, maps, points, link_hourly,
                       distance_to_links(grid, links), errors)
