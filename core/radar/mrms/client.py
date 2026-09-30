"""Fetch, decode, crop and cache MRMS radar products.

Sources, tried in order for every file:

* ``aws`` - NOAA Open Data Dissemination bucket ``noaa-mrms-pds`` (official NOAA
  distribution, anonymous HTTPS, archive from Oct 2020);
* ``iem`` - Iowa Environmental Mesonet mtarchive mirror (longer archive, same bytes).

Each file is downloaded into memory, integrity-checked (gzip + GRIB2 magic), decoded with
ecCodes, cropped to the requested :class:`~core.geo.Domain` by index, and stored
in a per-product / per-domain / per-day NetCDF cache. Only the crop is kept (a CONUS
field is ~100 MB in memory, a NYC crop is a few kB), so an 8-month hourly record of the
city costs a few MB on disk. Files confirmed absent on every source (HTTP 404) are
remembered, so repeated runs do not re-request gaps.

Typical use::

    from core.radar.mrms import MRMSClient
    from core.geo import NYC
    client = MRMSClient()
    qpe = client.load("MultiSensor_QPE_01H_Pass2", "2024-01-09 12:00", "2024-01-10 12:00", NYC)
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import multiprocessing
import os
import sys
import threading
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr

from core.geo import Domain
from core import data_paths as dp
from .products import MRMSProduct, get_product

log = logging.getLogger(__name__)

AWS_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"
IEM_BASE = "https://mtarchive.geol.iastate.edu"
SOURCES = ("aws", "iem")
USER_AGENT = "FieldSense (+https://github.com/drorjac/FieldSense)"
MRMS_CACHE = dp.data_path(dp.MRMS_CACHE)   # crops cached in ~/data/cml (see core/data_paths.py)


class MRMSError(RuntimeError):
    pass


class MRMSNotFound(MRMSError):
    """The file does not exist on any source (every source answered 404)."""


# --------------------------------------------------------------------------- URLs


def file_url(product: MRMSProduct | str, t: pd.Timestamp, source: str = "aws") -> str:
    """URL of one MRMS file valid at ``t`` (UTC) on ``source``."""
    p = get_product(product) if isinstance(product, str) else product
    t = pd.Timestamp(t)
    stamp = t.strftime("%Y%m%d-%H%M%S")
    if source == "aws":
        return f"{AWS_BASE}/CONUS/{p.name}/{t:%Y%m%d}/MRMS_{p.name}_{stamp}.grib2.gz"
    if source == "iem":
        return f"{IEM_BASE}/{t:%Y/%m/%d}/mrms/ncep/{p.iem_name}/{p.name}_{stamp}.grib2.gz"
    raise ValueError(f"unknown source {source!r}; use one of {SOURCES}")


def valid_times(product: MRMSProduct | str, start, end, freq: str | pd.Timedelta | None = None
                ) -> pd.DatetimeIndex:
    """Nominal file valid times in ``[start, end]``, on the product cadence or a coarser ``freq``.

    ``freq`` must be a multiple of the product cadence (e.g. ``"10min"`` for the 2-min
    PrecipFlag). Times are aligned to multiples of ``freq`` from midnight.
    """
    p = get_product(product) if isinstance(product, str) else product
    step = pd.Timedelta(freq) if freq is not None else pd.Timedelta(p.cadence)
    if step % pd.Timedelta(p.cadence):
        raise ValueError(f"freq {step} is not a multiple of {p.name} cadence {p.cadence}")
    start, end = _utc_naive(start), _utc_naive(end)
    return pd.date_range(start.ceil(step), end.floor(step), freq=step)


def _utc_naive(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


# ------------------------------------------------------------------------- decoding


@dataclass(frozen=True)
class _GridSpec:
    ni: int
    nj: int
    lat0: float      # latitude of first row
    lon0: float      # longitude of first column, [-180, 180)
    dlat: float      # signed row step (negative when north-up)
    dlon: float

    def window(self, domain: Domain) -> tuple[slice, slice]:
        """Row/column slices covering all cell centres inside ``domain``."""
        rows = (np.array([domain.lat_max, domain.lat_min]) - self.lat0) / self.dlat
        cols = (np.array([domain.lon_min, domain.lon_max]) - self.lon0) / self.dlon
        r0, r1 = int(np.ceil(rows.min() - 1e-6)), int(np.floor(rows.max() + 1e-6))
        c0, c1 = int(np.ceil(cols.min() - 1e-6)), int(np.floor(cols.max() + 1e-6))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.nj - 1), min(c1, self.ni - 1)
        if r1 < r0 or c1 < c0:
            raise MRMSError(f"domain {domain} is outside the MRMS grid")
        return slice(r0, r1 + 1), slice(c0, c1 + 1)


def decode_grib(raw: bytes, domain: Domain) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    """Decode one (uncompressed) MRMS GRIB2 message and crop it to ``domain``.

    Returns ``(values[lat, lon], lat_ascending, lon, valid_time)``; negative codes
    (no coverage / missing) and the GRIB missing value are NaN.
    """
    import eccodes  # optional dependency: pip install eccodes

    gid = eccodes.codes_new_from_message(raw)
    try:
        g = lambda k: eccodes.codes_get(gid, k)  # noqa: E731
        ni, nj = int(g("Ni")), int(g("Nj"))
        lat0 = float(g("latitudeOfFirstGridPointInDegrees"))
        lon0 = float(g("longitudeOfFirstGridPointInDegrees"))
        dlat = float(g("jDirectionIncrementInDegrees")) * (1 if int(g("jScansPositively")) else -1)
        dlon = float(g("iDirectionIncrementInDegrees")) * (-1 if int(g("iScansNegatively")) else 1)
        missing = float(g("missingValue"))
        valid = pd.Timestamp(f"{int(g('validityDate')):08d}{int(g('validityTime')):04d}")
        values = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)

    lon0 = ((lon0 + 180.0) % 360.0) - 180.0
    spec = _GridSpec(ni, nj, lat0, lon0, dlat, dlon)
    rs, cs = spec.window(domain)
    field = values.reshape(nj, ni)[rs, cs].astype("float32")
    field[(field < 0) | (field == missing)] = np.nan

    lat = np.round(lat0 + dlat * np.arange(rs.start, rs.stop), 4)
    lon = np.round(lon0 + dlon * np.arange(cs.start, cs.stop), 4)
    if dlat < 0:                       # store north-up files as ascending latitude
        field, lat = field[::-1], lat[::-1]
    return field, lat, lon, valid


def _check_payload(content: bytes) -> bytes:
    try:
        raw = gzip.decompress(content)
    except (OSError, EOFError) as exc:
        raise MRMSError(f"corrupt gzip payload ({len(content)} bytes): {exc}") from exc
    if raw[:4] != b"GRIB" or raw[-4:] != b"7777":
        raise MRMSError("payload is not a complete GRIB message")
    return raw


# --------------------------------------------------------------------------- client


class MRMSClient:
    """Download, decode, crop and cache MRMS products.

    Parameters
    ----------
    cache_dir:
        Root of the NetCDF crop cache (default ``<data dir>/mrms/cache``).
    sources:
        Ordered list of archives to try (``"aws"``, ``"iem"``).
    keep_raw:
        Also keep the original ``.grib2.gz`` files under ``<data dir>/mrms/raw``.
    max_workers:
        Parallel downloads (thread mode, ``processes=0``).
    processes:
        Worker processes that download + decode in parallel (default: CPUs - 1;
        0 = threads for download and serial decoding).
    retries, timeout:
        Per-request retry count (exponential back-off) and timeout in seconds.
    """

    def __init__(self, cache_dir: str | Path | None = None, sources: Sequence[str] = SOURCES,
                 keep_raw: bool = False, max_workers: int = 8, retries: int = 4,
                 timeout: float = 60.0, session: requests.Session | None = None,
                 processes: int | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else MRMS_CACHE
        self.raw_dir = MRMS_CACHE.parent / "mrms_raw"
        for s in sources:
            if s not in SOURCES:
                raise ValueError(f"unknown source {s!r}")
        self.sources = tuple(sources)
        self.keep_raw = keep_raw
        self.max_workers = max_workers
        self._pool = None
        self.processes = (max(1, (os.cpu_count() or 2) - 1) if processes is None else processes)
        self.retries = retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        self.session.mount("https://", adapter)

    # ---------------------------------------------------------------- single file

    def download(self, product: str | MRMSProduct, t) -> bytes:
        """Return the uncompressed GRIB2 bytes valid at ``t``.

        Raises :class:`MRMSNotFound` when every source answers 404, and
        :class:`MRMSError` after exhausting retries on other failures.
        """
        p = get_product(product) if isinstance(product, str) else product
        t = _utc_naive(t)
        raw_path = self.raw_dir / p.name / f"{t:%Y%m%d}" / f"{p.name}_{t:%Y%m%d-%H%M%S}.grib2.gz"
        if raw_path.exists():
            try:
                return _check_payload(raw_path.read_bytes())
            except MRMSError:
                raw_path.unlink()          # corrupt leftover, fetch again

        not_found, errors = 0, []
        for source in self.sources:
            url = file_url(p, t, source)
            for attempt in range(self.retries):
                try:
                    r = self.session.get(url, timeout=self.timeout)
                    if r.status_code == 404 or (source == "aws" and r.status_code == 403):
                        not_found += 1     # S3 answers 403 for absent keys on some paths
                        break
                    r.raise_for_status()
                    raw = _check_payload(r.content)
                    if self.keep_raw:
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_write_bytes(raw_path, r.content)
                    return raw
                except (requests.RequestException, MRMSError) as exc:
                    errors.append(f"{source}: {exc}")
                    time.sleep(min(2 ** attempt, 30) * 0.5)
        if not_found == len(self.sources):
            raise MRMSNotFound(f"{p.name} {t} not found on {', '.join(self.sources)}")
        raise MRMSError(f"{p.name} {t}: all sources failed: {errors[-3:]}")

    def read(self, product: str | MRMSProduct, t, domain: Domain) -> xr.DataArray:
        """One field valid at ``t``, cropped to ``domain`` (no caching)."""
        p = get_product(product) if isinstance(product, str) else product
        field, lat, lon, valid = decode_grib(self.download(p, t), domain)
        return _to_dataarray(field[None], lat, lon, [valid], p)

    # ------------------------------------------------------------- time series

    def load(self, product: str | MRMSProduct, start, end, domain: Domain,
             freq: str | None = None, progress: bool = False) -> xr.DataArray:
        """Fields for every valid time in ``[start, end]`` as ``(time, lat, lon)``.

        Uses and fills the per-day cache. Times missing from every archive are dropped
        from the result (``attrs['missing_times']`` lists them), never zero-filled.
        """
        p = get_product(product) if isinstance(product, str) else product
        times = valid_times(p, start, end, freq)
        if times.empty:
            raise ValueError(f"no {p.name} valid times between {start} and {end}")

        pieces, missing = [], []
        days = times.normalize().unique()
        with self._worker_pool():
            for i, day in enumerate(days):
                want = times[times.normalize() == day]
                da, gone = self._load_day(p, day, want, domain)
                if da is not None:
                    pieces.append(da)
                missing += gone
                if progress:
                    log.info("%s %s: %d/%d days", p.name, day.date(), i + 1, len(days))

        if not pieces:
            raise MRMSNotFound(f"no {p.name} data between {start} and {end}")
        out = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
        out.attrs["missing_times"] = [str(t) for t in missing]
        return out

    def _cache_path(self, p: MRMSProduct, domain: Domain, day: pd.Timestamp) -> Path:
        return self.cache_dir / p.name / domain.key / f"{day:%Y%m%d}.nc"

    def _load_day(self, p: MRMSProduct, day: pd.Timestamp, want: pd.DatetimeIndex, domain: Domain
                  ) -> tuple[xr.DataArray | None, list[pd.Timestamp]]:
        path = self._cache_path(p, domain, day)
        cached, known_missing = None, set()
        if path.exists():
            try:
                with xr.open_dataarray(path, engine="netcdf4") as da:
                    cached = da.load()
                cached.attrs["missing_times"] = json.loads(cached.attrs.get("missing_times", "[]"))
                known_missing = {pd.Timestamp(s) for s in cached.attrs["missing_times"]}
                if cached.sizes.get("time", 0) == 0:     # empty marker: only gaps are known
                    cached = None
            except (OSError, ValueError) as exc:
                log.warning("unreadable cache %s (%s); rebuilding", path, exc)
                path.unlink(missing_ok=True)

        have = set(pd.DatetimeIndex(cached.time.values)) if cached is not None else set()
        todo = [t for t in want if t not in have and t not in known_missing]

        if todo:
            new, gone, failed = self._fetch_many(p, todo, domain)
            # Only remember 404s for times safely in the past; recent files may still arrive.
            settled = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=2)
            known_missing |= {t for t in gone if t < settled}
            if new is not None:
                cached = new if cached is None else xr.concat([cached, new], dim="time")
                cached = cached.sortby("time")
                _, idx = np.unique(cached.time.values, return_index=True)
                cached = cached.isel(time=idx)
            if cached is not None:
                cached.attrs["missing_times"] = sorted(str(t) for t in known_missing)
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_to_netcdf(cached, path)
            elif known_missing:
                # Nothing downloaded but gaps are confirmed: persist an empty marker.
                empty = xr.DataArray(np.empty((0, 0, 0), "float32"), dims=("time", "lat", "lon"),
                                     coords={"time": pd.DatetimeIndex([])}, name=_var_name(p),
                                     attrs={"missing_times": sorted(str(t) for t in known_missing)})
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_to_netcdf(empty, path)
            missing_now = [t for t in todo if t in gone or t in failed]
            if failed:
                log.warning("%s: %d file(s) failed on every source and will be retried next time: %s",
                            p.name, len(failed), [str(t) for t in failed[:5]])
        else:
            missing_now = []

        missing = sorted(set(missing_now) | {t for t in want if t in known_missing})
        if cached is None or cached.sizes.get("time", 0) == 0:
            return None, missing
        sel = cached.sel(time=cached.time.isin(want.values))
        return (sel if sel.sizes["time"] else None), missing

    def _fetch_many(self, p: MRMSProduct, times: Iterable[pd.Timestamp], domain: Domain
                    ) -> tuple[xr.DataArray | None, list[pd.Timestamp]]:
        """Download + decode + crop ``times``; returns (fields, confirmed-missing times).

        With ``processes > 0`` each worker process downloads, decodes and crops a file
        and ships back only the small crop (decoding a CONUS field is CPU-bound, ~0.3 s).
        """
        times = list(times)
        fields, gone, failed, lat, lon = {}, [], [], None, None
        results = None
        if self._pool is not None and len(times) > 1:
            try:
                futures = {self._pool.submit(_worker_fetch, p.name, t, domain): t for t in times}
                results = [(futures[f], _result_or_error(f)) for f in as_completed(futures)]
            except BrokenProcessPool as exc:          # e.g. killed worker: continue with threads
                log.warning("process pool broke (%s); falling back to threads", exc)
                self.processes, results = 0, None
        if results is None:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._fetch_one, p, t, domain): t for t in times}
                results = [(futures[f], _result_or_error(f)) for f in as_completed(futures)]
        for t, res in results:
            if res is None:
                gone.append(t)
                continue
            if isinstance(res, MRMSError):        # transient: not remembered as missing
                failed.append(t)
                continue
            field, lat, lon, valid = res
            if valid != t:
                log.warning("%s: file for %s reports valid time %s", p.name, t, valid)
            fields[t] = field
        if not fields:
            return None, gone, failed
        order = sorted(fields)
        return _to_dataarray(np.stack([fields[t] for t in order]), lat, lon, order, p), gone, failed

    @contextlib.contextmanager
    def _worker_pool(self):
        """Keep one process pool alive for a whole ``load`` call (spawning is slow)."""
        if self.processes <= 0 or self._pool is not None or not _spawn_safe():
            yield
            return
        cfg = dict(sources=self.sources, keep_raw=self.keep_raw, retries=self.retries,
                   timeout=self.timeout, max_workers=1, cache_dir=self.cache_dir)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.processes, mp_context=ctx,
                                 initializer=_init_worker, initargs=(cfg,)) as pool:
            self._pool = pool
            try:
                yield
            finally:
                self._pool = None

    def _fetch_one(self, p: MRMSProduct, t: pd.Timestamp, domain: Domain):
        try:
            raw = self.download(p, t)
        except MRMSNotFound:
            return None
        with _DECODE_LOCK:                 # ecCodes thread-safety is build-dependent
            return decode_grib(raw, domain)


_DECODE_LOCK = threading.Lock()


def _result_or_error(fut):
    """Future result, or the MRMSError it raised (other exceptions propagate)."""
    try:
        return fut.result()
    except MRMSError as exc:
        return exc


def _spawn_safe() -> bool:
    """Spawned workers re-import ``__main__``; impossible for stdin/``-c`` scripts."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return path is None or os.path.exists(path)     # None = interactive/Jupyter: fine
_WORKER_CLIENT: "MRMSClient | None" = None


def _init_worker(cfg: dict) -> None:
    global _WORKER_CLIENT
    _WORKER_CLIENT = MRMSClient(processes=0, **cfg)


def _worker_fetch(product: str, t: pd.Timestamp, domain: Domain):
    return _WORKER_CLIENT._fetch_one(get_product(product), t, domain)


# ------------------------------------------------------------------------- helpers


def _var_name(p: MRMSProduct) -> str:
    return p.name.split("_00.00")[0]


def _to_dataarray(data, lat, lon, times, p: MRMSProduct) -> xr.DataArray:
    attrs = {"product": p.name, "units": p.units, "kind": p.kind,
             "source": "NOAA/NSSL MRMS", "missing_times": []}
    if p.accumulation is not None:
        attrs["accumulation"] = str(pd.Timedelta(p.accumulation))
        attrs["time_label"] = "end of accumulation window (UTC)"
    else:
        attrs["time_label"] = "valid time (UTC)"
    return xr.DataArray(data.astype("float32"), dims=("time", "lat", "lon"),
                        coords={"time": pd.DatetimeIndex(times), "lat": lat, "lon": lon},
                        name=_var_name(p), attrs=attrs)


def _atomic_to_netcdf(da: xr.DataArray, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
    os.close(fd)
    try:
        enc = {da.name: {"zlib": True, "complevel": 4}} if da.size else {}
        out = da.copy()
        out.attrs = {k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in da.attrs.items()}
        out.to_netcdf(tmp, engine="netcdf4", encoding=enc)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    os.replace(tmp, path)
