# MRMS radar: from archive to rainfall map

`core.radar.mrms` fetches NOAA's Multi-Radar/Multi-Sensor (MRMS) products, decodes and crops
them, caches the crops, and turns them into rainfall maps on any grid. It replaces the two
per-implementation MRMS loaders (see `docs/METHODS.md` for their issues).

## Products

| Name (`get_product`) | Content | Units | Cadence | Use here |
|---|---|---|---|---|
| `MultiSensor_QPE_01H_Pass2` | 1-h gauge-corrected radar QPE, 2nd pass | mm | hourly | **reference rainfall** |
| `MultiSensor_QPE_01H_Pass1` | same, 1st pass (fewer gauges, lower latency) | mm | hourly | near-real-time |
| `RadarOnly_QPE_01H` | 1-h radar-only QPE | mm | hourly | radar vs gauge-corrected |
| `MultiSensor_QPE_24H_Pass2` | 24-h QPE | mm | hourly | daily screening |
| `PrecipRate` | instantaneous surface rate | mm/h | 2 min | sub-hourly maps, animations |
| `PrecipFlag` | surface precipitation type | category | 2 min | snow / rain / mix typing |
| `RadarQualityIndex` | 0–1 radar quality | – | 2 min | masking unreliable cells |

All are CONUS grids, 0.01° (7000 × 3500), lon 230.005–299.995 °E, north-up, GRIB2 with PNG
packing. **Time stamps are valid times; for accumulations the stamp is the END of the window**
(the 01H file stamped 12:00 holds 11:00–12:00 UTC). This is the convention for every hourly
product in the package (CML accumulations included).

PrecipFlag categories (NOAA WDTD): 0 none, 1 warm stratiform rain, 3 snow, 6 convective,
7 rain + hail, 10 cool stratiform rain (surface < ~5 °C — where transitions live), 91/96
tropical stratiform/convective; −3 = no coverage.

## Sources

1. **`aws`** — NOAA Open Data Dissemination bucket `noaa-mrms-pds` (the official NOAA
   distribution; anonymous HTTPS; archive from Oct 2020).
2. **`iem`** — Iowa Environmental Mesonet `mtarchive` mirror (longer archive, same files).

Every file is tried on each source in order, with retries and exponential back-off; S3's 403
for absent keys counts as "not found". A file absent on every source (404) is recorded as
missing in the cache (only if older than 2 days, so late files are retried) and never
zero-filled.

## Processing

For each file: download to memory → gunzip → integrity check (GRIB magic + `7777` end marker)
→ ecCodes decode (grid geometry read from the file, not assumed) → crop by index to the
requested `Domain` → negative codes (−3 no coverage, −1/−999 missing) and the GRIB missing
value → NaN → latitudes flipped to ascending, longitudes to −180..180.

Decoding a CONUS field takes ~0.3 s of CPU, so files are fetched and decoded in a process pool
(one worker per core − 1; threads as fallback), and only the kB-sized crop is returned.

## Cache

`dataset/open_datasets/MRMS/cache/<product>/<domain key>/<YYYYMMDD>.nc` — one NetCDF per day, zlib
compressed, containing whichever valid times were requested so far plus the list of confirmed
missing times. The domain key contains a hash of the bounding box, so different crops never
collide; writes are atomic (temp file + rename). A full year of hourly NYC QPE is ~20 MB.

The cache is git-ignored; `MRMSClient(cache_dir=...)` puts it elsewhere.

## Products built on top (`core.radar.mrms`)

- `hourly_rainfall(start, end, domain)` — hourly QPE for hours ending in (start, end].
- `event_accumulation(start, end, domain, min_coverage=1.0)` — event total; cells missing
  any hour are NaN (or rescaled with `min_coverage < 1`, documented in the attributes).
- `rain_rate(start, end, domain, freq="10min", average="15min")` — PrecipRate, subsampled
  and block-averaged interval-ending.
- `to_grid(field, grid, method="auto")` — nearest / block-mean / linear resampling to any
  target grid (block mean when the target is coarser).
- `sample_points`, `path_average` — radar at gauges, or averaged along each link path (the
  fairest CML-vs-radar comparison: no interpolation involved).
- `mask_low_quality(field, rqi, threshold=0.5)` — drop cells with a poor radar quality index.

## Use

```python
from core.geo import NYC
from core.radar.mrms import MRMSClient, event_accumulation, hourly_rainfall

qpe = hourly_rainfall("2024-01-09 12:00", "2024-01-10 12:00", NYC)          # hourly QPE, NYC
flags = MRMSClient().load("PrecipFlag", "2024-01-09 12:00", "2024-01-09 18:00", NYC, freq="10min")
total = event_accumulation("2024-01-09 12:00", "2024-01-10 12:00", NYC)
```

Decoding needs `eccodes`: `pip install -e ".[mrms]"`.

## Known limitations of the reference

MRMS is a radar-based estimate, not ground truth: the beam overshoots low precipitation far
from the radar (NYC is ~50 km from KOKX, Upton NY), tall buildings block and clutter the beam,
reflectivity–rain conversion is empirical, and snow QPE is much less reliable than rain QPE.
Pass 2 gauge correction reduces but does not remove these.

## Validation (`python projects/nyc_rain_maps/src/run.py validate` → [`projects/nyc_rain_maps/results/validation/`](../../projects/nyc_rain_maps/results/validation/README.md))

The fetching chain is audited against independent paths, and the product against the official NWS
ASOS gauges (Central Park, LaGuardia, JFK, Newark):

- **Same bytes from both archives:** files fetched separately from NOAA AWS and IEM are
  byte-identical after decompression.
- **Decoder:** our ecCodes crop equals an independent full-file cfgrib decode value for value,
  with the same missing-data mask and valid time.
- **Georeferencing:** the value we sample at each official gauge equals ecCodes' own
  nearest-grid-point lookup in the raw GRIB.
- **Time labels:** 24 of our hourly Pass-2 fields sum exactly to MRMS's 24-h Pass-2 product, which
  also proves the hour-ending labels (a one-hour shift would break the match).
- **Completeness:** 5,899 of 5,904 hours in the OpenMesh period (5 missing on every archive).
- **Against the official gauges (8 months, same station-hours):** in rain MRMS Pass 2 has bias
  -4%, NRMSE 0.78, correlation 0.90; in snow bias
  -7%, NRMSE 0.65. Pass 1 performs the same; radar-only QPE reads ~20 % low in rain,
  which is what the gauge correction of Pass 1/2 removes.
