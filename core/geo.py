"""Spatial domains, target grids and small geodesy helpers.

All coordinates are WGS84 degrees with longitudes in [-180, 180). Grids are regular in
latitude/longitude, which matches MRMS (0.01 deg) and keeps resampling exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class Domain:
    """A lat/lon bounding box (edges, inclusive)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: str = "custom"

    def __post_init__(self):
        if not (self.lat_min < self.lat_max and self.lon_min < self.lon_max):
            raise ValueError(f"degenerate domain: {self}")

    @property
    def key(self) -> str:
        """Stable short identifier, used in cache paths."""
        spec = f"{self.lat_min:.4f},{self.lat_max:.4f},{self.lon_min:.4f},{self.lon_max:.4f}"
        return f"{self.name}-{hashlib.sha1(spec.encode()).hexdigest()[:8]}"

    def pad(self, deg: float) -> "Domain":
        return Domain(self.lat_min - deg, self.lat_max + deg,
                      self.lon_min - deg, self.lon_max + deg, self.name)

    def contains(self, lat, lon) -> np.ndarray:
        lat, lon = np.asarray(lat), np.asarray(lon)
        return ((lat >= self.lat_min) & (lat <= self.lat_max)
                & (lon >= self.lon_min) & (lon <= self.lon_max))

    @classmethod
    def around(cls, lats, lons, pad_deg: float = 0.02, name: str = "custom") -> "Domain":
        lats, lons = np.asarray(lats, float), np.asarray(lons, float)
        return cls(np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons), name).pad(pad_deg)


# New York City, five boroughs plus a small margin: the event-catalog domain.
NYC = Domain(40.48, 40.93, -74.27, -73.68, name="nyc")

# Bounding box of the 103 OpenMesh sublinks (Brooklyn / lower Manhattan / Queens edge),
# padded by 0.03 deg (~3 km) so the interpolated map is not truncated at the links.
OPENMESH = Domain(40.5719, 40.8604, -74.0437, -73.8764, name="openmesh")

DOMAINS = {"nyc": NYC, "openmesh": OPENMESH}


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid defined by cell centres."""

    lat: np.ndarray
    lon: np.ndarray

    @classmethod
    def from_domain(cls, domain: Domain, res_deg: float = 0.01) -> "Grid":
        """Cell centres spaced ``res_deg`` apart covering ``domain``.

        With the default 0.01 deg the centres coincide with MRMS cell centres
        (which sit at odd multiples of 0.005 deg), so radar needs no interpolation.
        """
        def axis(lo, hi):
            # centres at odd multiples of res/2 that lie inside [lo, hi]
            first = (np.ceil(round((lo - res_deg / 2) / res_deg, 9)) + 0.5) * res_deg
            n = int(np.floor(round((hi - first) / res_deg, 9))) + 1
            return np.round(first + res_deg * np.arange(n), 6)
        return cls(axis(domain.lat_min, domain.lat_max), axis(domain.lon_min, domain.lon_max))

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.size, self.lon.size

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat2d, lon2d) arrays of shape ``(n_lat, n_lon)``."""
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        return lat2d, lon2d

    def empty(self, name: str = "rain", **attrs) -> xr.DataArray:
        return xr.DataArray(np.full(self.shape, np.nan), dims=("lat", "lon"),
                            coords={"lat": self.lat, "lon": self.lon}, name=name, attrs=attrs)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres (broadcasts)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def to_local_xy(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to metres around (lat0, lon0).

    Error is well under 0.1% across a city-sized domain, which is far below any
    other error source here and keeps interpolation distances in plain metres.
    """
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return x, y
