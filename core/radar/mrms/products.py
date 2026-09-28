"""MRMS product registry.

MRMS (Multi-Radar/Multi-Sensor, NOAA/NSSL) CONUS products are 0.01 deg lat/lon grids of
7000 x 3500 cells (lon 230.005..299.995 E, lat 54.995..20.005 N, north-up), GRIB2 with
PNG packing, one gzipped file per valid time.

Time convention: the file time stamp is the *valid time*. For accumulations
(``*_QPE_*``) this is the END of the accumulation window, e.g. the 01H file stamped
12:00 holds rain from 11:00 to 12:00 UTC.

Missing data: ``-3`` = no radar coverage; other negative codes (``-1``, ``-999``) =
missing. All negatives become NaN when read (see ``PrecipFlag`` for its categories).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class MRMSProduct:
    name: str               # AWS directory name, e.g. "MultiSensor_QPE_01H_Pass2_00.00"
    iem_name: str           # IEM mtarchive directory name
    units: str
    kind: str               # "accumulation" | "rate" | "category" | "index"
    cadence: timedelta      # spacing between files
    accumulation: timedelta | None = None
    description: str = ""

    @property
    def is_categorical(self) -> bool:
        return self.kind == "category"


_H = timedelta(hours=1)
_2M = timedelta(minutes=2)

PRODUCTS: dict[str, MRMSProduct] = {p.name.split("_00.00")[0]: p for p in [
    MRMSProduct(
        "MultiSensor_QPE_01H_Pass2_00.00", "MultiSensor_QPE_01H_Pass2", "mm", "accumulation", _H, _H,
        "1-h gauge-corrected radar QPE, second pass (~60 min latency, more gauges). "
        "The reference rainfall product of this package."),
    MRMSProduct(
        "MultiSensor_QPE_01H_Pass1_00.00", "MultiSensor_QPE_01H_Pass1", "mm", "accumulation", _H, _H,
        "1-h gauge-corrected radar QPE, first pass (~20 min latency)."),
    MRMSProduct(
        "RadarOnly_QPE_01H_00.00", "RadarOnly_QPE_01H", "mm", "accumulation", _H, _H,
        "1-h radar-only QPE (no gauge correction)."),
    MRMSProduct(
        "MultiSensor_QPE_24H_Pass2_00.00", "MultiSensor_QPE_24H_Pass2", "mm", "accumulation",
        _H, timedelta(hours=24),
        "24-h gauge-corrected QPE, second pass. Used to screen days for the event catalog."),
    MRMSProduct(
        "PrecipRate_00.00", "PrecipRate", "mm/h", "rate", _2M, None,
        "Instantaneous surface precipitation rate, every 2 min."),
    MRMSProduct(
        "PrecipFlag_00.00", "PrecipFlag", "category", "category", _2M, None,
        "Surface precipitation type, every 2 min (see PRECIP_FLAG)."),
    MRMSProduct(
        "RadarQualityIndex_00.00", "RadarQualityIndex", "1", "index", _2M, None,
        "Radar quality index 0..1 (beam blockage, height, bright band). "
        "Useful to mask cells where the radar baseline itself is unreliable."),
]}

# MRMS surface precipitation type (PrecipFlag) categories, per NOAA WDTD documentation.
# -3 (no coverage) is converted to NaN on read.
PRECIP_FLAG = {
    0: "none",
    1: "warm stratiform rain",
    3: "snow",
    6: "convective rain",
    7: "rain mixed with hail",
    10: "cool stratiform rain",
    91: "tropical/stratiform rain mix",
    96: "tropical/convective rain mix",
}
SNOW_FLAGS = (3,)
RAIN_FLAGS = (1, 6, 7, 10, 91, 96)
# "Cool stratiform" is stratiform rain with surface temperature below ~5 C; it is where
# rain/snow transitions and mixed precipitation live, so it is tracked separately.
COOL_RAIN_FLAGS = (10,)


def get_product(name: str) -> MRMSProduct:
    key = name.split("_00.00")[0]
    try:
        return PRODUCTS[key]
    except KeyError:
        raise KeyError(f"unknown MRMS product {name!r}; known: {sorted(PRODUCTS)}") from None
