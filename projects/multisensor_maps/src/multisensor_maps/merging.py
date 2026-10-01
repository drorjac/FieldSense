"""Links, gauges and radar merged into one map: every combination, scored at held-out gauges.

    from multisensor_maps.merging import run_merging
    run_merging()                  # -> results/merging/report.md

The question ``pcpn_maps`` asked of New York's extreme events (``reports/extremes`` there),
asked on all three networks and with the merging methods of the OpenSense package
``mergeplg`` next to its own: what is the best hourly rain map that can be made from the
links, the gauges and the radar together, and what does each sensor add?

For each of the study's events (``study.network_events``):

1. **Inputs** - the QC'd links and every retrieval (``event.run_event``: four power laws
   and the RNN), the gauges, the radar; in New York also MRMS *radar-only*, which unlike
   Pass 2 has no gauge correction of its own. Cached under ``DATA_DIR/merging``.
2. **Products**, hourly on the network's grid, named by what goes in:

   ====================  ==========================================================
   ``R``                 the radar (``radar``; New York also ``radar only``)
   ``L``                 links alone: ``links <retrieval> idw`` / ``line``
   ``G``                 gauges alone: ``gauges`` (IDW)
   ``L+G``               ``links <retrieval> + gauges``: one IDW over both
   ``R+G``, ``R+L``,     ``<radar> <method> [<source>]``: the radar adjusted with the
   ``R+L+G``             gauges, the links, or both, by each method below
   ====================  ==========================================================

   Adjustment methods: ``mfb``, ``add``, ``mul`` of ``core.maps.merge`` (``pcpn_maps``:
   mean-field bias, additive and multiplicative IDW of the residuals) and ``idw_add``,
   ``idw_mul``, ``okrig_add``, ``ked`` of ``core.maps.mergeplg_methods`` (``mergeplg``:
   difference IDW, difference block kriging, block kriging with external drift). The
   kriging variogram is fitted to each network's radar fields, never to a gauge.
3. **Scores**. The check gauges (``settings.CHECK_POINTS``) are split into ``FOLDS``
   groups by station; each is held out in turn, every product that uses gauges is rebuilt
   from the other gauges - links and radar do not change - and scored at the held-out
   ones, hour by hour. Every product is scored on the same station-hours (see
   :func:`pooled`). Gauges no product uses - Gothenburg's SMHI gauge and New York's ASOS
   stations - give a second, fully independent check with every gauge in.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid, haversine_m, to_local_xy
from core.maps import merge
from core.maps.mergeplg_methods import DEFAULT_VARIOGRAM, METHODS as MERGEPLG, Merger, fit_radar_variogram
from core.maps.scores import scores
from core.opensense.networks import NETWORKS

from .event import interpolate, points_map, run_event
from .settings import CHECK_POINTS, DATA_DIR, MAP_POINTS, RESULTS_DIR

log = logging.getLogger(__name__)

MERGE_DIR = DATA_DIR / "merging"
OUT_DIR = RESULTS_DIR / "merging"
FOLDS = 5
PCPN = merge.ADJUSTMENTS                      # mfb, add, mul
ADJUSTMENTS = PCPN + MERGEPLG
PACKAGE = {**{m: "pcpn_maps" for m in PCPN}, **{m: "mergeplg" for m in MERGEPLG}}
LINK_MAPS = ("idw", "line")
INDEPENDENT = {"openmrg": "smhi", "openmesh": "asos"}
FAMILIES = ("R", "L", "G", "L+G", "R+G", "R+L", "R+L+G")
NEAR_KM = 5.0
KRIGING = ("okrig_add", "ked")
DEFAULT_TAG = " (5 km variogram)"            # kriging again with mergeplg's default variogram


# ---------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    network: str
    start: pd.Timestamp
    end: pd.Timestamp
    links: dict                   # retrieval -> (link, time) hour-ending mm, with geometry
    points: dict                  # point set -> (station, time) hour-ending mm
    radars: dict                  # name -> (time, lat, lon) mm
    errors: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.network} {self.start:%Y-%m-%d %H}"


def _folder(network, start) -> Path:
    return MERGE_DIR / "inputs" / f"{network}_{pd.Timestamp(start):%Y%m%dT%H}"


def radar_only(start, end, grid: Grid, times) -> xr.DataArray | None:
    """MRMS RadarOnly QPE (no gauge correction) on the OpenMesh grid, or None if missing."""
    from core.geo import NYC
    from core.radar.mrms import hourly_rainfall
    try:
        q = hourly_rainfall(start, end, NYC, product="RadarOnly_QPE_01H")
    except Exception as exc:                     # archive gap: the product is skipped
        log.warning("radar-only unavailable for %s..%s: %s", start, end, exc)
        return None
    q = q.reindex(lat=grid.lat, lon=grid.lon, method="nearest", tolerance=0.005).reindex(time=times)
    return q.astype("float32").rename("radar only").assign_attrs(units="mm", source="MRMS RadarOnly QPE 01H")


def load_inputs(network: str, start, end, extra: dict | None = None, refresh: bool = False) -> Inputs:
    """The event's links (every retrieval), gauges and radar; cached after the first run."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    folder = _folder(network, t0)
    if (folder / "meta.json").exists() and not refresh:
        meta = json.loads((folder / "meta.json").read_text())
        links = {m: xr.open_dataarray(folder / f"links_{m}.nc").load() for m in meta["links"]}
        points = {p: xr.open_dataarray(folder / f"points_{p}.nc").load() for p in meta["points"]}
        radars = {r: xr.open_dataarray(folder / f"radar_{i}.nc").load() for i, r in enumerate(meta["radars"])}
        return Inputs(network, t0, t1, links, points, radars, meta.get("errors", {}))
    res = run_event(network, t0, t1, extra=extra, interpolators=())
    radars = {"radar": res.maps["radar"]}
    if network == "openmesh":
        ro = radar_only(t0, t1, NETWORKS[network].grid, res.maps["radar"].time.values)
        if ro is not None and np.isfinite(ro.values).mean() > 0.5:
            radars["radar only"] = ro
    geo = ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon", "mid_lat", "mid_lon"]
    links = {}
    for m, lh in res.link_hourly.items():
        lh = lh.assign_coords({c: ("link", res.links[c].sel(link=lh.link).values) for c in geo})
        meta = ("frequency", "length", "polarization")         # the line/GMZ maps need these
        lh = lh.assign_coords({c: ("link", res.links[c].sel(link=lh.link).values.astype(str if c == "polarization"
                                                                                       else float)) for c in meta})
        links[m] = lh.reset_coords([c for c in lh.coords if c not in ("link", "time", *geo, *meta)], drop=True)
    points = {p: v.reset_coords([c for c in v.coords if c not in ("station", "time", "lat", "lon")], drop=True)
              for p, v in res.points.items()}
    folder.mkdir(parents=True, exist_ok=True)

    def _nc(v):         # netCDF attributes: numbers and strings only
        return v.assign_attrs({k: (a if isinstance(a, (int, float, str)) and not isinstance(a, bool) else str(a))
                               for k, a in v.attrs.items()})
    links = {m: _nc(v) for m, v in links.items()}
    points = {p: _nc(v) for p, v in points.items()}
    radars = {r: _nc(v) for r, v in radars.items()}
    for m, v in links.items():
        v.astype("float32").rename("rain").assign_coords(link=v.link.values.astype(str)).to_netcdf(folder / f"links_{m}.nc")
    for p, v in points.items():
        v.astype("float32").rename("rain").assign_coords(station=v.station.values.astype(str)).to_netcdf(folder / f"points_{p}.nc")
    for i, (r, v) in enumerate(radars.items()):
        v.astype("float32").rename("radar").to_netcdf(folder / f"radar_{i}.nc")
    (folder / "meta.json").write_text(json.dumps({"links": list(links), "points": list(points),
                                                  "radars": list(radars), "errors": res.errors,
                                                  "start": str(t0), "end": str(t1)}, indent=2))
    return load_inputs(network, t0, t1)


# -------------------------------------------------------------------------- products
def fold_of(station: str, folds: int = FOLDS) -> int:
    return int(hashlib.md5(str(station).encode()).hexdigest(), 16) % folds


def cells_at(grid: Grid, lat, lon) -> np.ndarray:
    """Flat index of the grid cell containing each point."""
    i = np.abs(grid.lat[None, :] - np.asarray(lat)[:, None]).argmin(1)
    j = np.abs(grid.lon[None, :] - np.asarray(lon)[:, None]).argmin(1)
    return i * grid.lon.size + j


def km_to_links(lat, lon, links: xr.DataArray, step_m: float = 200.0) -> np.ndarray:
    """Distance (km) from each point to the nearest link path."""
    best = np.full(np.shape(lat), np.inf)
    for la0, lo0, la1, lo1 in zip(links.site_0_lat.values, links.site_0_lon.values,
                                  links.site_1_lat.values, links.site_1_lon.values):
        n = max(2, int(haversine_m(la0, lo0, la1, lo1) / step_m) + 1)
        for s in np.linspace(0, 1, n):
            best = np.minimum(best, haversine_m(lat, lon, la0 + s * (la1 - la0), lo0 + s * (lo1 - lo0)))
    return best / 1000.0


class EventMerging:
    """Every product of one event, at any cells, with any gauges held out."""

    def __init__(self, inp: Inputs, variogram: dict | None = None, retrievals=None):
        self.inp = inp
        self.net = NETWORKS[inp.network]
        self.grid = self.net.grid
        first = next(iter(inp.radars.values()))
        self.times = first.time.values
        self.radars = {k: v.reindex(time=self.times) for k, v in inp.radars.items()}
        self.links = {m: v.reindex(time=self.times) for m, v in inp.links.items()
                      if retrievals is None or m in retrievals}
        self.links = {m: v for m, v in self.links.items() if v.sizes["link"] >= 2}
        sets = [inp.points[p].reindex(time=self.times) for p in MAP_POINTS[inp.network]
                if p in inp.points and inp.points[p].sizes["station"]]
        self.gauges = xr.concat(sets, "station") if sets else None
        self.variogram = variogram
        self._cache: dict = {}

    # which products exist
    def specs(self) -> pd.DataFrame:
        rows = [dict(product=r, family="R", radar=r) for r in self.radars]
        srcs = []
        for m in self.links:
            rows += [dict(product=f"links {m} {i}", family="L", retrieval=m, interp=i) for i in LINK_MAPS]
            srcs.append(("R+L", m, m))
        if self.gauges is not None:
            rows.append(dict(product="gauges", family="G"))
            srcs.insert(0, ("R+G", "gauges", None))
            for m in self.links:
                rows.append(dict(product=f"links {m} + gauges", family="L+G", retrieval=m))
                srcs.append(("R+L+G", f"{m} + gauges", m))
        for r in self.radars:
            for fam, src, m in srcs:
                for a in ADJUSTMENTS:
                    rows.append(dict(product=f"{r} {a} [{src}]", family=fam, radar=r, retrieval=m, source=src,
                                     method=a, package=PACKAGE[a]))
        df = pd.DataFrame(rows)
        for c in ("radar", "retrieval", "interp", "source", "method", "package"):
            if c not in df:
                df[c] = None
        df["uses_gauges"] = df.family.str.contains("G")
        return df.set_index("product")

    # the pieces, cached
    def _obs(self, radar: str, src: str):
        """pcpn_maps observations ``(obs, radar_at_obs)`` for a source, all gauges in."""
        key = ("obs", radar, src)
        if key not in self._cache:
            parts = {}
            m = src.removesuffix(" + gauges")
            if m in self.links:
                parts["cml"] = self.links[m]
            if src == "gauges" or src.endswith("+ gauges"):
                parts["gauges"] = self.gauges
            self._cache[key] = merge.observations(self.radars[radar], **parts)
        return self._cache[key]

    def _merger(self, radar: str, src: str) -> Merger:
        key = ("mergeplg", radar, src)
        if key not in self._cache:
            m = src.removesuffix(" + gauges")
            L = self.links.get(m)
            G = self.gauges if (src == "gauges" or src.endswith("+ gauges")) else None
            self._cache[key] = Merger(self.radars[radar], L, G, variogram=self.variogram)
        return self._cache[key]

    def _link_map(self, m: str, interp: str) -> np.ndarray:
        key = ("L", m, interp)
        if key not in self._cache:
            f = interpolate(interp, self.links[m], self.grid).reindex(time=self.times)
            self._cache[key] = f.transpose("time", "lat", "lon").values.reshape(len(self.times), -1)
        return self._cache[key]

    @staticmethod
    def _flat(f: xr.DataArray, times) -> np.ndarray:
        return f.reindex(time=times).transpose("time", "lat", "lon").values.reshape(len(times), -1)

    def values(self, product: str, spec: pd.Series, cells: np.ndarray, held_out=()) -> np.ndarray:
        """``(time, cells)`` of a product, with the gauges in ``held_out`` left out."""
        fam = spec.family
        held = set(map(str, held_out))
        gmask = None if self.gauges is None else ~np.isin(self.gauges.station.values.astype(str), list(held))
        if fam == "R":
            return self._flat(self.radars[spec.radar], self.times)[:, cells]
        if fam == "L":
            return self._link_map(spec.retrieval, spec.interp)[:, cells]
        if fam == "G":
            f = points_map(self.gauges.isel(station=gmask), self.grid)
            return self._flat(f, self.times)[:, cells]
        if fam == "L+G":
            obs = merge.observations(None, cml=self.links[spec.retrieval],
                                     gauges=self.gauges.isel(station=gmask))[0]
            return self._flat(merge.merge_idw(obs, self.grid), self.times)[:, cells]
        # radar adjusted
        if spec.method in PCPN:
            obs, rad = self._obs(spec.radar, spec.source)
            keep = ~((obs.kind.values == "gauges") & np.isin(obs.point.values, list(held)))
            f = merge.adjust(self.radars[spec.radar], obs.isel(point=keep), rad.isel(point=keep), spec.method)
            return self._flat(f, self.times)[:, cells]
        mg = self._merger(spec.radar, spec.source)
        mask = None
        if held and "gauges" in spec.source:
            n_links = mg.obs.shape[0] - self.gauges.sizes["station"]
            mask = np.concatenate([np.ones(n_links, bool), gmask])
        return mg.adjust(spec.method, cells=cells, obs_mask=mask)

    def field(self, product: str, spec: pd.Series) -> xr.DataArray:
        """The product's map on the whole grid, every gauge in."""
        v = self.values(product, spec, np.arange(self.grid.lat.size * self.grid.lon.size))
        return xr.DataArray(v.reshape((len(self.times),) + self.grid.shape).astype("float32"),
                            dims=("time", "lat", "lon"),
                            coords={"time": self.times, "lat": self.grid.lat, "lon": self.grid.lon}, name=product)


# --------------------------------------------------------------------------- scoring
def score_event(em: EventMerging, folds: int = FOLDS, methods=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide table: one row per (gauge, hour) - held-out check gauges and independent gauges -
    one column per product; and the product specs. ``methods`` keeps only the radar
    adjustments by those methods."""
    specs = em.specs()
    if methods is not None:
        specs = specs[specs.method.isin(methods)]
    inp = em.inp
    frames = []
    check = inp.points.get(CHECK_POINTS[inp.network])
    any_links = next(iter(em.links.values()), None)
    sets = [("check", check)]
    ind = INDEPENDENT.get(inp.network)
    if ind and ind in inp.points and inp.points[ind].sizes["station"]:
        sets.append(("independent", inp.points[ind]))
    for subset, pts in sets:
        if pts is None or pts.sizes["station"] == 0:
            continue
        pts = pts.reindex(time=em.times)
        # only gauges inside the grid
        inside = ((pts.lat >= em.grid.lat.min()) & (pts.lat <= em.grid.lat.max())
                  & (pts.lon >= em.grid.lon.min()) & (pts.lon <= em.grid.lon.max())).values
        pts = pts.isel(station=inside)
        if pts.sizes["station"] == 0:
            continue
        st = pts.station.values.astype(str)
        cells = cells_at(em.grid, pts.lat.values, pts.lon.values)
        fold = np.array([fold_of(s, folds) for s in st]) if subset == "check" else np.full(st.size, -1)
        near = (km_to_links(pts.lat.values, pts.lon.values, any_links) if any_links is not None
                else np.full(st.size, np.inf))
        T, S = len(em.times), st.size
        base = pd.DataFrame({"event": inp.label, "network": inp.network, "subset": subset,
                             "station": np.tile(st, T), "time": np.repeat(em.times, S),
                             "fold": np.tile(fold, T), "km_to_link": np.tile(near, T),
                             "obs": pts.transpose("time", "station").values.ravel()})
        cols = {}
        for name, spec in specs.iterrows():
            out = np.full((T, S), np.nan)
            try:
                if subset == "check" and spec.uses_gauges:
                    for k in np.unique(fold):
                        sel = fold == k
                        out[:, sel] = em.values(name, spec, cells[sel], held_out=st[sel])
                else:
                    out[:] = em.values(name, spec, cells)
            except Exception as exc:                        # keep the other products
                log.warning("%s: %s failed: %r", inp.label, name, exc)
            cols[name] = out.ravel().astype("float32")
        frames.append(pd.concat([base, pd.DataFrame(cols)], axis=1))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), specs


def pooled(table: pd.DataFrame, specs: pd.DataFrame, by=(), min_coverage: float = 0.9,
           wet: float = 0.1) -> pd.DataFrame:
    """Scores of every product on the station-hours all full-coverage products share.

    A product with gaps (under ``min_coverage`` of the best-covered one) does not shrink
    the common sample; it is scored on its share of it and its coverage is shown."""
    prods = [p for p in specs.index if p in table]
    ok = table.obs.notna().values
    n = {p: int((ok & table[p].notna().values).sum()) for p in prods}
    top = max(n.values()) if n else 0
    full = [p for p in prods if n[p] >= min_coverage * top and top > 0]
    common = ok & np.all([table[p].notna().values for p in full], axis=0) if full else ok
    rows = []
    groups = table[common].groupby(list(by)) if by else [((), table[common])]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for p in prods:
            s = scores(g[p].values, g.obs.values, wet)
            rows.append({**dict(zip(by, key)), "product": p, "coverage": s.get("n", 0) / max(len(g), 1), **s})
    out = pd.DataFrame(rows)
    return out.join(specs, on="product")


def rank(table: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Pooled held-out scores per product, best first, with the median per-event NRMSE and
    the number of events each product is best on."""
    tab = pooled(table, specs).set_index("product")
    per = pooled(table, specs, by=("event",)).pivot(index="product", columns="event", values="nrmse")
    tab["median_event_nrmse"] = per.median(axis=1)
    tab["event_wins"] = (per.rank(axis=0, method="min") == 1).sum(axis=1)
    return tab.sort_values("nrmse")


# ----------------------------------------------------------------------------- study
@dataclass
class MergingStudy:
    events: pd.DataFrame
    tables: dict                   # network -> wide station-hour table
    specs: dict                    # network -> product specs
    variograms: dict
    rankings: dict = field(default_factory=dict)      # (network, subset) -> ranking
    examples: dict = field(default_factory=dict)      # network -> (EventMerging, {product: field})
    errors: dict = field(default_factory=dict)

    def save(self, out: Path = OUT_DIR) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        self.events.to_csv(out / "events.csv", index=False)
        for (n, subset), r in self.rankings.items():
            tag = subset.replace("<= ", "le").replace(" ", "_").replace("-", "")
            r.round(4).to_csv(out / f"ranking_{n}_{tag}.csv")
        per_event = []
        for n, t in self.tables.items():
            chk = t[t.subset == "check"]
            per_event.append(pooled(chk, self.specs[n], by=("event",)).assign(network=n))
        pd.concat(per_event, ignore_index=True)[["network", "event", "product", "family", "n", "rel_bias",
                                                  "nrmse", "corr", "csi"]].round(4).to_csv(
            out / "scores_per_event.csv", index=False)
        (out / "variograms.json").write_text(json.dumps(self.variograms, indent=2))
        (out / "errors.json").write_text(json.dumps(self.errors, indent=2))
        return out


def network_variogram(inputs: list[Inputs]) -> dict:
    """Spherical variogram shape fitted to the network's radar over its study events."""
    radar = xr.concat([next(iter(i.radars.values())) for i in inputs], "time")
    return fit_radar_variogram(radar)


def subsets(table: pd.DataFrame) -> dict:
    """The station-hour sets the report scores on."""
    out = {"held-out gauges": table[table.subset == "check"],
           f"held-out gauges <= {NEAR_KM:g} km of a link": table[(table.subset == "check")
                                                               & (table.km_to_link <= NEAR_KM)]}
    if (table.subset == "independent").any():
        out["independent gauges"] = table[table.subset == "independent"]
    return out


def scored_event(inp: Inputs, variogram: dict, refresh: bool = False):
    """:func:`score_event` for one event, cached (pickle) with the variogram it used."""
    path = MERGE_DIR / "scores" / f"{_folder(inp.network, inp.start).name}.pkl"
    if path.exists() and not refresh:
        cached = pd.read_pickle(path)
        if cached["variogram"] == variogram:
            return cached["table"], cached["specs"]
    em = EventMerging(inp, variogram)
    if not em.links:
        return None, None
    log.info("merging: %s (%d links, %d gauges, %d hours)", inp.label,
             next(iter(em.links.values())).sizes["link"],
             0 if em.gauges is None else em.gauges.sizes["station"], len(em.times))
    table, specs = score_event(em)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"table": table, "specs": specs, "variogram": variogram}, path)
    return table, specs


def with_default_variogram(inp: Inputs, table: pd.DataFrame, specs: pd.DataFrame, refresh: bool = False):
    """Add the kriging products rebuilt with mergeplg's default variogram (sill 0.9, range
    5 km, nugget 0.1), named ``"<product> (5 km variogram)"``: the sensitivity to the
    variogram that is otherwise fitted to the radar."""
    path = MERGE_DIR / "scores" / f"{_folder(inp.network, inp.start).name}_default_variogram.pkl"
    if path.exists() and not refresh:
        extra, xspecs = pd.read_pickle(path)
    else:
        extra, xspecs = score_event(EventMerging(inp, dict(DEFAULT_VARIOGRAM)), methods=KRIGING)
        pd.to_pickle((extra, xspecs), path)
    if not (extra[["station", "time", "subset"]].values == table[["station", "time", "subset"]].values).all():
        raise ValueError(f"{inp.label}: default-variogram rows do not match")
    cols = [c for c in xspecs.index if c in extra]
    table = pd.concat([table, extra[cols].rename(columns=lambda c: c + DEFAULT_TAG)], axis=1)
    specs = pd.concat([specs.assign(variogram=np.where(specs.method.isin(KRIGING), "radar fit", None)),
                       xspecs.rename(index=lambda c: c + DEFAULT_TAG).assign(variogram="mergeplg default")])
    return table, specs


def run_merging(networks=("openmrg", "openrainer", "openmesh"), extra_factory=None,
                events: pd.DataFrame | None = None, examples: bool = True,
                refresh: bool = False, sensitivity: bool = True) -> MergingStudy:
    """Every event of every network: inputs, variogram, products, held-out scores, ranking.
    Inputs and scores are cached per event under ``DATA_DIR/merging``; ``refresh`` rescores."""
    from .study import network_events

    ev_rows, tables, specs, variograms, errors = [], {}, {}, {}, {}
    for network in networks:
        evs = network_events(network) if events is None else events[events.network == network]
        ev_rows.append(evs)
        extra = extra_factory(network) if extra_factory else None
        inputs = []
        for ev in evs.itertuples():
            try:
                inputs.append(load_inputs(network, ev.start, ev.end, extra=extra))
            except Exception as exc:
                errors[f"{network} {pd.Timestamp(ev.start):%Y-%m-%d %H}"] = repr(exc)
                log.exception("inputs failed: %s %s", network, ev.start)
        variograms[network] = network_variogram(inputs)
        log.info("%s variogram: %s", network, variograms[network])
        frames, sp = [], None
        for inp in inputs:
            for k, v in inp.errors.items():
                errors[f"{inp.label} / {k}"] = v
            try:
                t, s = scored_event(inp, variograms[network], refresh)
            except Exception as exc:
                errors[f"{inp.label} / merging"] = repr(exc)
                log.exception("merging failed: %s", inp.label)
                continue
            if t is None:
                errors[f"{inp.label} / merging"] = "fewer than 2 links - left out of the merging study"
                continue
            if sensitivity:
                t, s = with_default_variogram(inp, t, s, refresh)
            frames.append(t)
            sp = s if sp is None else sp.combine_first(s)
        tables[network] = pd.concat(frames, ignore_index=True)
        specs[network] = sp
    study = MergingStudy(pd.concat(ev_rows, ignore_index=True), tables, specs, variograms, errors=errors)
    for n, t in tables.items():
        for name, sub in subsets(t).items():
            if len(sub):
                study.rankings[(n, name)] = rank(sub, specs[n])
    if examples:
        for n in tables:
            evs = study.events[study.events.network == n]
            big = evs.loc[evs.total_mm.idxmax()]
            inp = load_inputs(n, big.start, big.end)
            em = EventMerging(inp, variograms[n])
            if em.links:
                study.examples[n] = (em, example_fields(em, study.rankings[(n, "held-out gauges")]))
    return study


def best_per_family(ranking: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """The best product of each input combination (R, L, G, L+G, R+G, R+L, R+L+G)."""
    r = ranking[ranking.coverage >= min_coverage]
    if "variogram" in r:
        r = r[r.variogram != "mergeplg default"]
    rows = [r[r.family == f].iloc[0].rename(r[r.family == f].index[0]) for f in FAMILIES if (r.family == f).any()]
    return pd.DataFrame(rows)


def example_fields(em: EventMerging, ranking: pd.DataFrame) -> dict:
    """Whole-grid maps of the radar and of the best product of each input combination."""
    names = list(dict.fromkeys(list(em.radars) + list(best_per_family(ranking).index)))
    specs = em.specs()
    return {p: em.field(p, specs.loc[p]) for p in names if p in specs.index}
