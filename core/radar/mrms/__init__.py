"""MRMS radar (NOAA/NSSL Multi-Radar/Multi-Sensor): products, a fetch/decode/cache client,
and the rainfall-map products built from it.

``products``  the product registry and PrecipFlag categories
``client``    download from the NOAA AWS bucket (IEM mirror as fallback), decode GRIB2 with
              ecCodes, crop to a :class:`core.geo.Domain`, cache one NetCDF per day
``maps``      hourly and event accumulations, rain rate, quality masking, regridding,
              radar at points and along link paths

Decoding needs ``eccodes`` (``pip install -e ".[mrms]"``). Crops are cached under
``~/data/cml/openmesh/weather/radar/mrms_cache/`` (see core/data_paths.py).

    from core.geo import NYC
    from core.radar.mrms import MRMSClient, hourly_rainfall
    qpe = hourly_rainfall("2024-01-09 12:00", "2024-01-10 12:00", NYC)
"""

from .client import MRMSClient, MRMSError, MRMSNotFound, decode_grib, file_url, valid_times
from .maps import (QPE_1H, domain_mean_series, event_accumulation, hourly_rainfall,
                   mask_low_quality, path_average, rain_rate, sample_points, to_grid)
from .products import (COOL_RAIN_FLAGS, PRECIP_FLAG, PRODUCTS, RAIN_FLAGS, SNOW_FLAGS,
                       MRMSProduct, get_product)

__all__ = ["MRMSClient", "MRMSError", "MRMSNotFound", "MRMSProduct", "PRODUCTS", "PRECIP_FLAG",
           "RAIN_FLAGS", "SNOW_FLAGS", "COOL_RAIN_FLAGS", "QPE_1H", "get_product", "file_url",
           "valid_times", "decode_grib", "hourly_rainfall", "event_accumulation", "rain_rate",
           "to_grid", "sample_points", "path_average", "mask_low_quality", "domain_mean_series"]
