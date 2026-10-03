"""The Dutch commercial microwave link dataset, and KNMI hourly gauges to score it.

Overeem, Walraven, Leijnse and Uijlenhoet (2024), "Four-year commercial microwave link
dataset for the Netherlands", 4TU.ResearchData, doi:10.4121/be252844-b672-471e-8d69-
27269a862ec1.v1, CC BY 4.0. One text file, ``IDRawCMLdata.dat`` inside
``IDRawCMLdata.zip`` (9.5 GB zipped, ~37 GB of text, ~440 million rows), in RAINLINK's
input format, one row per sub-link and 15-min interval::

    YStart XStart YEnd XEnd Frequency DateTime ES SES Pmin Pmax PathLength Vendor ID
    52.21836 6.925286 52.229751 6.921736 37.912 201101122315 0 0 -50 -49 1.29 NOKIA 4066

Coordinates in degrees (Y = lat, X = lon), frequency in GHz, ``DateTime`` the END of the
15-min interval in UTC, ``Pmin``/``Pmax`` the minimum and maximum received signal level
over the interval (dBm; Nokia 1 dB, NEC 0.1 dB resolution), path length in km, ``ID``
one sub-link (one direction of a path). There is no transmitted level: it is constant.
The file is sorted by time, which :func:`iter_rainlink` uses to skip and stop early.

Three layers here:

1. **Reading** - :func:`iter_rainlink` streams the text out of the zip in blocks and
   yields the rows of a period as DataFrames; it never holds the file in memory.
2. **OpenSense format** - :func:`to_dataset` turns rows into ``rsl_min``/``rsl_max``
   ``(cml_id, sublink_id, time)``; :func:`convert_months` writes one netCDF per month to
   ``~/data/cml/netherlands/monthly/``; :func:`open_months` reads them back.
   A path (``cml_id``) is an unordered pair of endpoints, named by their coordinates so
   the name is the same in every month; ``sublink_0``, ``sublink_1``, ... are its
   RAINLINK IDs ordered by direction (site 0 to site 1 first), frequency and ID.
3. **Retrieval** - :func:`rainlink_retrieval` runs RAINLINK's chain (Overeem, Leijnse
   and Uijlenhoet 2016, AMT 9, 2425-2444, doi:10.5194/amt-9-2425-2016) through
   pycomlink's ports of it, with RAINLINK's default parameters.

KNMI's hourly automatic-station precipitation (:func:`knmi_hourly`) is the reference.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import IO, Iterator

import numpy as np
import pandas as pd
import xarray as xr

from core import data_paths as dp

log = logging.getLogger(__name__)

ZIP_NAME = "IDRawCMLdata.zip"
DAT_NAME = "IDRawCMLdata.dat"
COLUMNS = ["YStart", "XStart", "YEnd", "XEnd", "Frequency", "DateTime", "ES", "SES",
           "Pmin", "Pmax", "PathLength", "Vendor", "ID"]
_USE = {"YStart": "float64", "XStart": "float64", "YEnd": "float64", "XEnd": "float64",
        "Frequency": "float64", "DateTime": "int64", "Pmin": "float32", "Pmax": "float32",
        "PathLength": "float64", "Vendor": "category", "ID": "int64"}
STEP = pd.Timedelta("15min")
BLOCK_BYTES = 32 << 20          # ~400k rows: about a day of the network

# RAINLINK defaults (Overeem et al. 2016, and RAINLINK's run script), as named in pycomlink
WETDRY = dict(radius=15, thresh_median_P=-1.4, thresh_median_PL=-0.7, min_links=3,
              interval=15, timeperiod=24, min_hours=6)
REFERENCE = dict(n_average_dry=96, min_periods=10)       # 24 h window, >= 2.5 h dry
RETRIEVAL = dict(waa_max=2.3, alpha=0.33, F_value_threshold=-32.5)
SPINUP = pd.Timedelta("24h")    # history the wet/dry and reference level need


def raw_zip() -> Path:
    """``~/data/cml/netherlands/_download/IDRawCMLdata.zip`` (fetch it with
    ``python -m core.opensense.fetch --dataset netherlands``)."""
    return dp.data_path(dp.NETHERLANDS_DOWNLOAD) / ZIP_NAME


def monthly_dir() -> Path:
    return dp.data_path(dp.NETHERLANDS_MONTHLY)


# ------------------------------------------------------------------ reading


def _stamp(t) -> int:
    """A time as RAINLINK's ``YYYYMMDDhhmm`` integer."""
    return int(pd.Timestamp(t).strftime("%Y%m%d%H%M"))


def _open_text(source) -> IO[bytes]:
    """A binary stream of the RAINLINK text: from the zip, a ``.dat`` file or a stream."""
    if hasattr(source, "read"):
        return source
    source = Path(source)
    if source.suffix == ".zip":
        zf = zipfile.ZipFile(source)
        names = [n for n in zf.namelist() if n.endswith(".dat")]
        return zf.open(names[0])
    return source.open("rb")


def _first_stamp(line: bytes) -> int | None:
    parts = line.split()
    if len(parts) != len(COLUMNS) or not parts[5].isdigit():
        return None
    return int(parts[5])


def _parse(block: bytes) -> pd.DataFrame:
    if block.startswith(b"YStart"):                     # the header line
        block = block.split(b"\n", 1)[1] if b"\n" in block else b""
    return pd.read_csv(io.BytesIO(block), sep=" ", header=None, names=COLUMNS,
                       usecols=list(_USE), dtype=_USE, on_bad_lines="skip", engine="c")


def iter_rainlink(source=None, start=None, end=None,
                  block_bytes: int = BLOCK_BYTES) -> Iterator[pd.DataFrame]:
    """Rows of the RAINLINK text with ``start < DateTime <= end``, block by block.

    ``source``: the zip (default :func:`raw_zip`), the extracted ``.dat`` or a binary
    stream. Times are interval ends, as in the file, so ``start`` = 00:00 and ``end`` =
    24:00 of a day select exactly that day's 96 intervals. Blocks that end before
    ``start`` are skipped without parsing, and reading stops at the first block that
    begins after ``end`` (the file is sorted by time; a block that is not is still
    filtered row by row, so the result is right either way, only slower).
    """
    lo = _stamp(start) if start is not None else -1
    hi = _stamp(end) if end is not None else np.iinfo(np.int64).max
    fh = _open_text(source if source is not None else raw_zip())
    rest = b""
    try:
        while True:
            chunk = fh.read(block_bytes)
            block = rest + chunk
            if not chunk:
                rest = b""
            else:
                cut = block.rfind(b"\n") + 1
                block, rest = block[:cut], block[cut:]
            if block:
                lines = block.split(b"\n", 2)
                first = _first_stamp(lines[0]) or (_first_stamp(lines[1]) if len(lines) > 1 else None)
                last = _first_stamp(block.rstrip(b"\n").rsplit(b"\n", 1)[-1])
                if first is not None and first > hi:
                    break
                if last is None or last > lo:
                    df = _parse(block)
                    df = df[(df.DateTime > lo) & (df.DateTime <= hi)]
                    if len(df):
                        yield df
            if not chunk:
                break
    finally:
        fh.close()


def read_rainlink(source=None, start=None, end=None) -> pd.DataFrame:
    """All rows of a period in one DataFrame (small periods and tests)."""
    parts = list(iter_rainlink(source, start, end))
    return pd.concat(parts, ignore_index=True) if parts else _parse(b"")


def rows_per_day(source=None, block_bytes: int = 8 << 20) -> pd.Series:
    """Rows per day over the whole file, without parsing it into a table.

    A block whose first and last rows fall on the same day is counted by its newlines;
    only the few blocks that span midnight are split into lines. One pass over the
    37 GB of text, limited by decompression (~6 min).
    """
    fh = _open_text(source if source is not None else raw_zip())
    counts: dict[int, int] = {}
    rest = b""
    try:
        while chunk := fh.read(block_bytes):
            block = rest + chunk
            cut = block.rfind(b"\n") + 1
            block, rest = block[:cut], block[cut:]
            if not block:                                  # no complete line yet
                continue
            lines = block.split(b"\n", 2)
            first = _first_stamp(lines[0]) or _first_stamp(lines[1])
            last = _first_stamp(block.rstrip(b"\n").rsplit(b"\n", 1)[-1])
            if first is not None and last is not None and first // 10_000 == last // 10_000:
                day = first // 10_000
                header = int(block.startswith(b"YStart"))
                counts[day] = counts.get(day, 0) + block.count(b"\n") - header
                continue
            for ln in block.split(b"\n"):
                stamp = _first_stamp(ln)
                if stamp is not None:
                    counts[stamp // 10_000] = counts.get(stamp // 10_000, 0) + 1
    finally:
        fh.close()
    s = pd.Series(counts, name="rows").sort_index()
    s.index = pd.to_datetime(s.index.astype(str), format="%Y%m%d")
    s.index.name = "date"
    return s


# ------------------------------------------------------------------ OpenSense format


def _times(stamps: np.ndarray) -> np.ndarray:
    u, inv = np.unique(stamps, return_inverse=True)
    return pd.to_datetime(u.astype(str), format="%Y%m%d%H%M").values[inv]


def sublink_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per RAINLINK ID: path, direction, frequency, length, vendor, endpoints.

    An ID's metadata is its most frequent value over the rows given. Endpoints are
    rounded to 1e-5 deg (~1 m) before pairing the two directions of a path.
    """
    def mode(s):
        return s.mode().iloc[0]
    meta = df.groupby("ID").agg(lat_s=("YStart", mode), lon_s=("XStart", mode),
                                lat_e=("YEnd", mode), lon_e=("XEnd", mode),
                                frequency=("Frequency", "median"),
                                length=("PathLength", "median"),
                                vendor=("Vendor", mode)).round({"lat_s": 5, "lon_s": 5,
                                                               "lat_e": 5, "lon_e": 5})
    s = meta[["lat_s", "lon_s"]].to_numpy()
    e = meta[["lat_e", "lon_e"]].to_numpy()
    start_first = (s[:, 0] < e[:, 0]) | ((s[:, 0] == e[:, 0]) & (s[:, 1] <= e[:, 1]))
    site0 = np.where(start_first[:, None], s, e)
    site1 = np.where(start_first[:, None], e, s)
    meta["site_0_lat"], meta["site_0_lon"] = site0[:, 0], site0[:, 1]
    meta["site_1_lat"], meta["site_1_lon"] = site1[:, 0], site1[:, 1]
    meta["direction"] = np.where(start_first, 0, 1)      # 0: transmitted from site 0
    meta["cml_id"] = [f"{a:.5f}_{b:.5f}_{c:.5f}_{d:.5f}"
                      for a, b, c, d in np.hstack([site0, site1])]
    return meta


_META = ["cml_id", "direction", "frequency", "length", "vendor",
         "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]


def to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """RAINLINK rows -> OpenSense-format ``rsl_min``/``rsl_max(cml_id, sublink_id, time)``.

    ``time`` is the end of each 15-min interval (UTC), on a regular 15-min axis from the
    first to the last row; missing intervals are NaN. Rows with ``Pmin > Pmax`` are kept
    as given (the retrieval drops them, as RAINLINK's preprocessing does).
    """
    meta = sublink_table(df)
    t = _times(df.DateTime.to_numpy())
    time = pd.date_range(t.min(), t.max(), freq=STEP).as_unit("ns")
    iid = meta.index.get_indexer(df.ID)
    it = time.get_indexer(t)
    flat = {}
    for col in ("Pmin", "Pmax"):
        a = np.full((len(meta), time.size), np.nan, dtype="float32")
        a[iid, it] = df[col].to_numpy()
        flat[col] = a
    return _assemble(meta, time, flat["Pmin"], flat["Pmax"])


def _assemble(meta: pd.DataFrame, time: pd.DatetimeIndex, pmin: np.ndarray,
              pmax: np.ndarray) -> xr.Dataset:
    """Per-ID arrays ``(id, time)`` (rows in ``meta.index`` order) -> the path form.

    ``sublink_0``, ``sublink_1``, ... of a path are its IDs ordered by direction,
    frequency and ID, so the same set of IDs always gets the same slots.
    """
    meta = meta.copy()
    meta.index.name = "ID"
    order = meta.reset_index().sort_values(["cml_id", "direction", "frequency", "ID"])
    slot = order.groupby("cml_id").cumcount()
    meta["slot"] = pd.Series(slot.to_numpy(), index=order.ID.to_numpy()).loc[meta.index]
    paths = np.array(sorted(meta.cml_id.unique()))
    subs = np.array([f"sublink_{k}" for k in range(int(slot.max()) + 1)])
    ip = pd.Index(paths).get_indexer(meta.cml_id)
    isub = meta.slot.to_numpy()
    shape = (paths.size, subs.size, time.size)
    out = {}
    for name, col, v in (("rsl_min", "Pmin", pmin), ("rsl_max", "Pmax", pmax)):
        a = np.full(shape, np.nan, dtype="float32")
        a[ip, isub] = v
        out[name] = (("cml_id", "sublink_id", "time"), a,
                     {"units": "dBm", "long_name": f"{col} over the 15-min interval"})

    def sub_coord(values, fill, dtype):
        a = np.full(shape[:2], fill, dtype=dtype)
        a[ip, isub] = values
        return a

    per_path = meta.groupby("cml_id").agg(site_0_lat=("site_0_lat", "first"),
                                          site_0_lon=("site_0_lon", "first"),
                                          site_1_lat=("site_1_lat", "first"),
                                          site_1_lon=("site_1_lon", "first"),
                                          length=("length", "mean"),
                                          vendor=("vendor", "first")).loc[paths]
    ds = xr.Dataset(out, coords={"cml_id": paths, "sublink_id": subs, "time": time})
    for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"):
        ds.coords[c] = ("cml_id", per_path[c].to_numpy(), {"units": "degrees"})
    ds.coords["length"] = ("cml_id", per_path.length.to_numpy(), {"units": "km"})
    ds.coords["vendor"] = ("cml_id", np.array(per_path.vendor.astype(str).tolist(), dtype=str))
    ds.coords["frequency"] = (("cml_id", "sublink_id"),
                              sub_coord(meta.frequency.to_numpy(), np.nan, float), {"units": "GHz"})
    ds.coords["rainlink_id"] = (("cml_id", "sublink_id"),
                                sub_coord(meta.index.to_numpy(), -1, "int64"),
                                {"long_name": "RAINLINK sub-link ID", "missing": -1})
    ds.coords["direction"] = (("cml_id", "sublink_id"),
                              sub_coord(meta.direction.to_numpy(), -1, "int8"),
                              {"long_name": "0: site 0 to site 1, 1: reverse"})
    ds.time.attrs["long_name"] = "end of the 15-min interval (UTC)"
    ds.attrs.update(title="Netherlands CML (Overeem et al. 2024), RAINLINK min/max format",
                    source="doi:10.4121/be252844-b672-471e-8d69-27269a862ec1.v1",
                    license="CC-BY-4.0", polarization="not given (mostly vertical)",
                    time_label="end of 15-min interval")
    return ds


def sublinks(ds: xr.Dataset) -> xr.Dataset:
    """Path form -> one entry per RAINLINK ID, dim ``cml_id`` = ``"<path>/<sublink>"``.

    The flat form pycomlink's nearby (RAINLINK) functions take; empty slots are dropped.
    Variables are renamed ``pmin``/``pmax`` and ``length``, ``frequency``, site and
    midpoint coordinates are attached per entry.
    """
    flat = ds.stack(link=("cml_id", "sublink_id"))
    flat = flat.isel(link=np.flatnonzero(flat.rainlink_id.values >= 0))
    label = [f"{c}/{s}" for c, s in zip(flat.cml_id.values, flat.sublink_id.values)]
    keep = {c: ("link", flat[c].values) for c in
            ("rainlink_id", "frequency", "length", "direction", "vendor",
             "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon")}
    keep["path"] = ("link", flat.cml_id.values)
    out = xr.Dataset({"pmin": (("link", "time"), flat.rsl_min.transpose("link", "time").values),
                      "pmax": (("link", "time"), flat.rsl_max.transpose("link", "time").values)},
                     coords={"link": label, "time": ds.time.values, **keep})
    out.coords["mid_lat"] = (out.site_0_lat + out.site_1_lat) / 2
    out.coords["mid_lon"] = (out.site_0_lon + out.site_1_lon) / 2
    return out.rename(link="cml_id")


def _months(start, end) -> list[pd.Period]:
    """Months holding any interval end in (start, end].

    An interval ending at 00:00 on the 1st belongs to the month before, so an ``end``
    of 1 September does not touch September.
    """
    first = pd.Timestamp(start).to_period("M")
    last = (pd.Timestamp(end) - STEP).to_period("M")
    return list(pd.period_range(first, last, freq="M"))


def month_file(month, out_dir: Path | None = None) -> Path:
    return Path(out_dir or monthly_dir()) / f"cml_{pd.Period(month, 'M')}.nc"


def convert_months(start, end, source=None, out_dir: Path | None = None,
                   overwrite: bool = False) -> list[Path]:
    """Write ``cml_YYYY-MM.nc`` for every month touching (start, end]; return their paths.

    Whole months are written (a month file is complete or absent), already present
    ones are skipped unless ``overwrite``. One streaming pass over the source: rows are
    collected for the months still to write and a month is written as soon as the
    sorted file has moved past it. Months with no rows are reported and not written.
    """
    out_dir = Path(out_dir or monthly_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    months = _months(start, end)
    todo = [m for m in months if overwrite or not month_file(m, out_dir).exists()]
    paths = [month_file(m, out_dir) for m in months]
    if not todo:
        return paths
    lo = todo[0].start_time
    hi = todo[-1].end_time.floor("D") + pd.Timedelta("1D")       # 24:00 of the last day
    pending: dict[pd.Period, list[pd.DataFrame]] = {m: [] for m in todo}
    written: set[pd.Period] = set()

    def flush(m):
        parts = pending.pop(m)
        if not parts:
            log.warning("no rows for %s; not written", m)
            return
        ds = to_dataset(pd.concat(parts, ignore_index=True))
        tmp = month_file(m, out_dir).with_suffix(".nc.part")
        enc = {v: {"zlib": True, "complevel": 4} for v in ("rsl_min", "rsl_max")}
        ds.to_netcdf(tmp, encoding=enc)
        tmp.replace(month_file(m, out_dir))
        written.add(m)
        log.info("wrote %s: %d paths, %d sub-links, %d steps", month_file(m, out_dir).name,
                 ds.sizes["cml_id"], int((ds.rainlink_id >= 0).sum()), ds.sizes["time"])

    for df in iter_rainlink(source, lo, hi):
        # the 00:00 row of the 1st belongs (interval end) to the previous month
        per = pd.Series(pd.DatetimeIndex(_times(df.DateTime.to_numpy())) - STEP,
                        index=df.index).dt.to_period("M")
        for m, part in df.groupby(per.values, observed=True):
            m = pd.Period(m, "M")
            if m in written:
                raise RuntimeError(f"rows for {m} after it was written: source not sorted")
            if m in pending:
                pending[m].append(part)
        newest = per.max()
        for m in [m for m in pending if m < newest]:
            flush(m)
    for m in list(pending):
        flush(m)
    return paths


def open_months(start, end, out_dir: Path | None = None, convert: bool = True,
                source=None) -> xr.Dataset:
    """The OpenSense dataset for (start, end], from the monthly files (made if missing).

    Months are joined on the RAINLINK IDs and the sub-link slots assigned again over all
    of them: a path that gains or loses a sub-link between months would otherwise hold
    different IDs in the same slot.
    """
    months = _months(start, end)
    files = [month_file(m, out_dir) for m in months]
    if convert and not all(f.exists() for f in files):
        convert_months(start, end, source=source, out_dir=out_dir)
    dss = [xr.open_dataset(f) for f in files if f.exists()]
    if not dss:
        raise FileNotFoundError(f"no monthly files for {start}..{end} in {out_dir or monthly_dir()}")
    time = pd.DatetimeIndex(np.unique(np.concatenate([d.time.values for d in dss])))
    metas, values = [], {}
    for d in dss:
        flat = d.stack(link=("cml_id", "sublink_id"))
        flat = flat.isel(link=np.flatnonzero(flat.rainlink_id.values >= 0))
        ids = flat.rainlink_id.values
        m = pd.DataFrame({c: flat[c].values for c in _META}, index=ids)
        metas.append(m)
        it = time.get_indexer(d.time.values)
        for v in ("rsl_min", "rsl_max"):
            x = flat[v].transpose("link", "time").values
            for k, i in enumerate(ids):
                values.setdefault((v, i), np.full(time.size, np.nan, dtype="float32"))[it] = x[k]
        d.close()
    meta = pd.concat(metas)
    meta = meta[~meta.index.duplicated(keep="last")]
    ds = _assemble(meta, time, np.stack([values[("rsl_min", i)] for i in meta.index]),
                   np.stack([values[("rsl_max", i)] for i in meta.index]))
    return ds.sel(time=slice(pd.Timestamp(start) + STEP, pd.Timestamp(end)))


# ------------------------------------------------------------------ retrieval


MAX_GHZ = 60.0


def usable(sub: xr.Dataset) -> xr.Dataset:
    """Sub-links with a path length above 0 km and a frequency below 60 GHz.

    The dataset README drops the same 0.08 % of the data before its figures: a few
    sub-links have identical endpoints (0 km), and some report 254.135 GHz.
    """
    keep = (sub.length > 0) & (sub.frequency < MAX_GHZ) & sub.frequency.notnull()
    return sub.isel(cml_id=np.flatnonzero(keep.values))


def preprocess(sub: xr.Dataset) -> xr.Dataset:
    """RAINLINK's ``PreprocessingMinMaxRSL``: an interval with ``Pmin > Pmax`` is dropped."""
    bad = sub.pmin > sub.pmax
    return sub.assign(pmin=sub.pmin.where(~bad), pmax=sub.pmax.where(~bad))


def nearby_within(site_0_lat, site_0_lon, site_1_lat, site_1_lon,
                  radius: float = WETDRY["radius"]) -> np.ndarray:
    """``(n, n)`` bool: both ends of link j within ``radius`` km of both ends of link i.

    pycomlink's ``calc_distance_between_cml_endpoints`` + ``within_r``, vectorised: same
    haversine (pycomlink's, R = 6367 km), same rule, a link always its own neighbour.
    """
    from pycomlink.spatial.helper import haversine

    a = (np.asarray(site_0_lon, float), np.asarray(site_0_lat, float))
    b = (np.asarray(site_1_lon, float), np.asarray(site_1_lat, float))
    within = np.ones((a[0].size, a[0].size), bool)
    for p in (a, b):
        for q in (a, b):
            within &= haversine(p[0][:, None], p[1][:, None], q[0][None, :], q[1][None, :]) < radius
    np.fill_diagonal(within, True)
    return within


def _nanmedian0(x: np.ndarray) -> np.ndarray:
    """``np.nanmedian(x, axis=0)``, by one sort (NaN sorts last); ~10x faster here."""
    s = np.sort(x, axis=0)
    k = np.isfinite(x).sum(axis=0)
    lo = np.take_along_axis(s, np.maximum(k - 1, 0)[None] // 2, axis=0)[0]
    hi = np.take_along_axis(s, (k // 2)[None], axis=0)[0]
    return np.where(k > 0, (lo + hi) / 2, np.nan)


def nearby_wetdry_fast(pmin: xr.DataArray, within: np.ndarray, radius=None,
                       thresh_median_P=-1.4, thresh_median_PL=-0.7, min_links=3,
                       interval=15, timeperiod=24, min_hours=6):
    """pycomlink's ``nearby_wetdry`` (RAINLINK's nearby-link wet/dry), vectorised.

    Same arguments (``within`` from :func:`nearby_within` replaces the distance
    Dataset and ``radius``) and the same result, element for element - see
    ``tests/test_netherlands.py``. pycomlink's version loops over links with xarray
    selections and copies the whole wet array per link, which is hours for the ~5000
    Dutch sub-links over a summer; here only the neighbourhood medians loop, in numpy,
    and every other step is pycomlink's own xarray expression applied to all links at
    once. Returns ``(wet, F)`` as pycomlink does.
    """
    period = int(timeperiod * 60 / interval)
    hours_needed = int(min_hours * 60 / interval)
    max_pmin = pmin.rolling(time=period, min_periods=hours_needed).max(skipna=False)
    deltaP = pmin - max_pmin
    deltaPL = deltaP / pmin.length

    dP = deltaP.transpose("cml_id", "time").values
    dPL = deltaPL.transpose("cml_id", "time").values
    n, nt = dP.shape
    medianP = np.full((n, nt), np.nan)
    medianPL = np.full((n, nt), np.nan)
    active = within.sum(axis=1) > min_links
    for i in np.flatnonzero(active):
        nb = np.flatnonzero(within[i])
        enough = np.isfinite(dP[nb]).sum(axis=0) > min_links
        medianP[i] = np.where(enough, _nanmedian0(dP[nb]), np.nan)
        medianPL[i] = np.where(enough, _nanmedian0(dPL[nb]), np.nan)

    dims, coords = ("cml_id", "time"), {"cml_id": pmin.cml_id, "time": pmin.time}
    mP = xr.DataArray(medianP, dims=dims, coords=coords)
    mPL = xr.DataArray(medianPL, dims=dims, coords=coords)
    wet = xr.where(np.isnan(mP) | np.isnan(mPL), np.nan,
                   (mP < thresh_median_P) & (mPL < thresh_median_PL))
    act = xr.DataArray(active, dims="cml_id", coords={"cml_id": pmin.cml_id})
    wet = wet.where(act)
    F = (deltaPL.transpose(*dims) - mPL).rolling(time=timeperiod * 4).sum(skipna=True) \
        * (interval / 60)
    F = F.where(act)

    wet_tmp = wet.copy()
    cond = (wet_tmp == 1) & (deltaP.transpose(*dims) < -2)
    for shift in [1, -1, -2]:
        wet = xr.where(cond.shift(time=shift), 1, wet)
    wet = xr.where(np.isnan(wet_tmp), np.nan, wet)
    return wet.transpose(*dims), F.transpose(*dims)


def rainlink_retrieval(sub: xr.Dataset, pol: str = "v", wetdry: dict | None = None,
                       reference: dict | None = None, retrieval: dict | None = None,
                       fast: bool = True) -> xr.Dataset:
    """RAINLINK's chain on 15-min min/max levels: rain rate (mm/h) per sub-link and step.

    ``sub`` is the flat form of :func:`sublinks` (``pmin``/``pmax(cml_id, time)`` with
    ``length`` in km, ``frequency`` in GHz, site coordinates). Steps, all pycomlink
    ports of RAINLINK (Overeem et al. 2016) with its defaults (module constants):

    1. preprocessing: drop intervals with Pmin > Pmax;
    2. ``nearby_wetdry``: an interval is wet when the median drop of Pmin below its
       24-h maximum, over the links within 15 km, is below -1.4 dB and below
       -0.7 dB/km; also returns the outlier filter F;
    3. ``nearby_determine_reference_level``: median of (Pmin+Pmax)/2 over the dry
       intervals of the previous 24 h (at least 2.5 h of them);
    4. ``nearby_correct_received_signals``: Pmin/Pmax set to the reference outside wet
       intervals;
    5. ``nearby_rainfall_retrival``: Amin, Amax minus the wet-antenna attenuation
       Aa = 2.3 dB, through ``k = a R^b`` with ITU-R P.838-3 coefficients
       (``core.cml.power_law.itu_ab``; RAINLINK uses its own DSD-derived ones),
       R = 0.33 Rmax + 0.67 Rmin, and NaN where F <= -32.5 dB h/km.

    Returns a Dataset with ``rain`` (mm/h), ``wet`` (1/0/NaN), ``pref`` and ``F``.
    Sub-links with fewer than ``min_links + 1`` links within the radius get NaN, as in
    RAINLINK. A sub-link's frequency outside the ITU table (1-1000 GHz) raises; pass
    :func:`usable` sub-links. ``fast=False`` runs pycomlink's own ``nearby_wetdry``
    in step 2 instead of :func:`nearby_wetdry_fast` (identical, much slower).
    """
    from pycomlink.processing.nearby_rain_retrival import (
        nearby_correct_received_signals, nearby_determine_reference_level,
        nearby_rainfall_retrival)
    from pycomlink.processing.wet_dry.nearby_wetdry import (
        calc_distance_between_cml_endpoints, nearby_wetdry)

    from core.cml.power_law import itu_ab

    wetdry = {**WETDRY, **(wetdry or {})}
    reference = {**REFERENCE, **(reference or {})}
    retrieval = {**RETRIEVAL, **(retrieval or {})}
    sub = preprocess(sub)
    # only what pycomlink reads: extra per-link coordinates make every xarray step slower
    bare = xr.Dataset({"pmin": sub.pmin.astype("float64"), "pmax": sub.pmax.astype("float64")},
                      coords={"cml_id": sub.cml_id.values, "time": sub.time.values,
                              "length": ("cml_id", sub.length.values.astype(float))})
    pmin, pmax = bare.pmin, bare.pmax
    if fast:
        within = nearby_within(sub.site_0_lat.values, sub.site_0_lon.values,
                               sub.site_1_lat.values, sub.site_1_lon.values, wetdry["radius"])
        wet, F = nearby_wetdry_fast(pmin, within, **wetdry)
    else:
        dist = calc_distance_between_cml_endpoints(
            cml_ids=sub.cml_id.values,
            site_a_latitude=sub.site_0_lat.values, site_a_longitude=sub.site_0_lon.values,
            site_b_latitude=sub.site_1_lat.values, site_b_longitude=sub.site_1_lon.values)
        wet, F = nearby_wetdry(pmin, dist, **wetdry)
    pref = nearby_determine_reference_level(pmin, pmax, wet, **reference)
    p_c_min, p_c_max = nearby_correct_received_signals(pmin, pmax, wet, pref)
    a, b = itu_ab(sub.frequency.values, np.full(sub.sizes["cml_id"], pol))
    a = xr.DataArray(a, dims="cml_id", coords={"cml_id": bare.cml_id})
    b = xr.DataArray(b, dims="cml_id", coords={"cml_id": bare.cml_id})
    length = xr.DataArray(sub.length.values.astype(float), dims="cml_id",
                          coords={"cml_id": bare.cml_id})
    rain = nearby_rainfall_retrival(pref, p_c_min, p_c_max, F, length=length, a=a, b=b,
                                    **retrieval)
    rain = rain.transpose("cml_id", "time")
    out = xr.Dataset({"rain": rain, "wet": wet, "pref": pref, "F": F})
    out = out.assign_coords({c: sub[c] for c in sub.coords if sub[c].dims == ("cml_id",)})
    out.rain.attrs = {"units": "mm/h", "long_name": "path-averaged rain rate (RAINLINK)",
                      "polarization": pol, "power_law": "ITU-R P.838-3", **retrieval}
    return out


def path_rain(rain: xr.DataArray) -> xr.DataArray:
    """Sub-link rain ``(cml_id, time)`` -> path mean ``(link, time)`` with midpoints.

    RAINLINK averages the sub-links of a full-duplex path before mapping; a path's value
    is the mean of its sub-links with data.
    """
    out = rain.groupby("path").mean().rename(path="link")
    out = out.assign_coords(link=out.link.values.astype(str))
    for c in ("mid_lat", "mid_lon", "length", "site_0_lat", "site_0_lon",
              "site_1_lat", "site_1_lon"):
        if c in rain.coords:
            v = rain[c].to_series().groupby(rain.path.values).first()
            out.coords[c] = ("link", v.loc[out.link.values].to_numpy())
    return out


# ------------------------------------------------------------------ KNMI gauges

KNMI_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"


def parse_knmi_hourly(text: str) -> tuple[pd.DataFrame, xr.DataArray]:
    """KNMI ``uurgegevens`` text -> (stations, ``rain(station, time)`` in mm per hour).

    The header lists the stations (``STN LON(east) LAT(north) ALT(m) NAME``); rows are
    ``STN,YYYYMMDD,HH,RH``. ``HH`` = 1..24 is the hour ENDING at HH UT (``uurvak 05``
    runs 04-05 UT), so ``time`` is the end of the hour. ``RH`` is in 0.1 mm; KNMI
    documents ``-1`` as "< 0.05 mm", which is taken as 0 (it is below the 0.1 mm
    resolution, and an amount that rounds to zero). An empty field is a missing value.
    """
    lines = text.splitlines()
    stations, rows, in_stations = [], [], False
    for ln in lines:
        if ln.startswith("# STN") and "LON" in ln:
            in_stations = True
            continue
        if in_stations:
            parts = ln[1:].split() if ln.startswith("#") else []
            if len(parts) >= 5 and parts[0].isdigit():
                stations.append({"station": int(parts[0]), "lon": float(parts[1]),
                                 "lat": float(parts[2]), "alt": float(parts[3]),
                                 "name": " ".join(parts[4:])})
                continue
            in_stations = False
        if ln and not ln.startswith("#"):
            rows.append(ln)
    st = pd.DataFrame(stations, columns=["station", "lon", "lat", "alt", "name"]).set_index("station")
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     names=["station", "date", "hh", "RH"], skipinitialspace=True,
                     dtype={"station": int, "date": str, "hh": int}, na_values=[""])
    rh = pd.to_numeric(df.RH, errors="coerce")
    mm = np.where(rh == -1, 0.0, rh / 10.0)
    t = pd.to_datetime(df.date, format="%Y%m%d") + pd.to_timedelta(df.hh, unit="h")
    wide = pd.DataFrame({"station": df.station.values, "time": t.values, "mm": mm}) \
        .pivot_table(index="station", columns="time", values="mm", aggfunc="first", dropna=False)
    da = xr.DataArray(wide.to_numpy(float), dims=("station", "time"),
                      coords={"station": wide.index.to_numpy(),
                              "time": pd.DatetimeIndex(wide.columns).as_unit("ns")}, name="rain")
    da.attrs = {"units": "mm", "long_name": "KNMI hourly precipitation (RH)",
                "time_label": "end of the hour (UTC)", "below_0.05mm": "set to 0"}
    st = st.reindex(da.station.values)
    names = np.array(st["name"].astype(str).tolist(), dtype=str)
    da = da.assign_coords(lat=("station", st.lat.to_numpy()), lon=("station", st.lon.to_numpy()),
                          station_name=("station", names))
    return st, da


def knmi_hourly(start, end, cache_dir: Path | None = None) -> xr.DataArray:
    """KNMI hourly gauge precipitation, all automatic stations, hours ending in (start, end].

    Fetched month by month from KNMI's open climatology endpoint (no key) and cached as
    the raw response text under ``~/data/cml/netherlands/knmi_hourly/RH_YYYY-MM.txt``.
    """
    import requests

    cache = Path(cache_dir or dp.data_path(dp.NETHERLANDS_KNMI))
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for m in _months(pd.Timestamp(start) + pd.Timedelta("1h") - STEP, end):
        f = cache / f"RH_{m}.txt"
        if not f.exists():
            day0 = m.start_time
            day1 = m.end_time.floor("D")
            r = requests.post(KNMI_URL, timeout=300, data={
                "start": day0.strftime("%Y%m%d") + "01", "end": day1.strftime("%Y%m%d") + "24",
                "vars": "RH", "stns": "ALL"})
            r.raise_for_status()
            f.with_suffix(".part").write_text(r.text)
            f.with_suffix(".part").replace(f)
        da = parse_knmi_hourly(f.read_text())[1]
        # the hours ending in this month only: a response can run past the end date
        parts.append(da.sel(time=slice(m.start_time + pd.Timedelta("1h"),
                                       (m + 1).start_time)))
    da = xr.concat(parts, dim="time", join="outer") if len(parts) > 1 else parts[0]
    da = da.sel(time=slice(pd.Timestamp(start) + pd.Timedelta("1h"), pd.Timestamp(end)))
    # stations in the header without an RH sensor (offshore, coastal) are all missing
    return da.isel(station=np.flatnonzero(da.notnull().any("time").values))
