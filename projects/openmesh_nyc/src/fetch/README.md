# Data Fetching

Collection pipelines for the weather observations that sit alongside the
OpenMesh links. The notebooks in `../../notebooks/` drive them. KOKX radar is
used by other projects too, so its fetcher lives in `core/radar/nexrad.py`.

## 📁 Files

```
src/fetch/
├── asos_functions.py    # NOAA ASOS 1-minute fetch, processing, save_data()
├── asos_plotting.py     # ASOS figures
├── wu_functions.py      # Weather Underground PWS fetch, cleaning, run_wu_pipeline()
└── wu_plotting.py       # PWS figures

../../notebooks/
├── asos_pipeline.ipynb                 # ASOS end to end
├── wu_pipeline.ipynb                   # Weather Underground end to end
└── download_and_read_openmesh.ipynb    # the published OpenMesh dataset from Zenodo
```

Output goes wherever you point `output_dir` in the notebook. Data files are
git-ignored by the root `.gitignore` (`*.csv`, `*.nc`, ...).

## 📊 Data Sources

### 1. NOAA ASOS (`asos_*.py`)
**Source:** Iowa Environmental Mesonet (IEM), 1-minute ASOS archive
**Data:** Airport weather stations: temperature, wind, visibility, precipitation type and amount
**Resolution:** 1 minute
**API Key:** Not required ✓

This uses the **1-minute archive** (`asos1min.py`), not METAR (`asos.py`), which
reports only hourly. NCEI collects the 1-minute data directly from the stations
twice a day; IEM parses NCEI's raw format into a clean download. Data therefore
lags 18-36 hours and isn't real-time. Coverage goes back to 2000 for US sites.

| Variable | Unit | Description |
|----------|------|-------------|
| `temp_c` | °C | Temperature |
| `dewpoint_c` | °C | Dewpoint |
| `wind_speed_ms` | m/s | Wind speed |
| `wind_gust_ms` | m/s | Wind gust |
| `wind_dir_deg` | ° | Wind direction |
| `visibility_km` | km | Visibility |
| `precip_type` | - | Precipitation type (rain, snow, etc.) |
| `precip_mm` | mm | Precipitation |

**Quick Start:** open `notebooks/asos_pipeline.ipynb`, set the date period and
stations, and run all cells. Stations use 4-letter ICAO codes (e.g. KJFK,
KLGA, KNYC). The NYC list is `ASOS_stations.csv` in the published dataset's
`weather stations/` folder. You can also browse the
[IEM network list](https://mesonet.agron.iastate.edu/sites/networks.php?network=ASOS).

- **Endpoint:** https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py
- **Docs:** https://mesonet.agron.iastate.edu/request/asos/1min.phtml
- **IEM parser:** https://github.com/akrherz/iem/blob/main/scripts/ingestors/asos_1minute/parse_ncei_asos1minute.py

### 2. Weather Underground (`wu_*.py`)
**Source:** Weather Underground Personal Weather Stations API
**Data:** Community weather stations
**Resolution:** 5 minutes (historical)
**API Key:** Required ⚠️. Get one at https://www.wunderground.com/member/api-keys

**Quick Start:**
1. `export WU_API_KEY="your_key_here"`
2. Open `notebooks/wu_pipeline.ipynb`, set the date period and station IDs
   (e.g. `KNYNEWYO1805`), and run all cells.

The NYC station list is `pws_metadata.csv` in the published dataset's
`weather stations/` folder. You can also search the
[WunderMap](https://www.wunderground.com/wundermap).

### 3. OpenMesh (Zenodo)
**Source:** https://zenodo.org/records/15287692 (links, PWS and metadata, 2023-2024)
**API Key:** Not required ✓

Open `notebooks/download_and_read_openmesh.ipynb` and run all cells. It downloads
and extracts the dataset.

## 📋 Requirements

pandas, numpy, matplotlib, requests (see `../../requirements.txt`). You only need
an API key for Weather Underground.
