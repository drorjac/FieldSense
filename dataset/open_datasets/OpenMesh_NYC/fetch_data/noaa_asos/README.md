# NOAA ASOS 1-Minute Data Pipeline

Fetch TRUE per-minute precipitation and weather data from ASOS stations via IEM.

## 📁 Files

- **`asos_pipeline.ipynb`** - Complete workflow notebook
- **`asos_functions.py`** - Fetching, processing, and plotting functions

## 🔄 Change from Previous Version

Previous version used METAR data (`asos.py` endpoint) which reports hourly observations.

This version uses the **1-minute ASOS archive** which NCEI (National Centers for Environmental Information) collects directly from ASOS stations via phone twice daily. The raw NCEI format is difficult to work with and mostly undocumented. IEM processes this data and provides a clean, accessible download.

**Note:** Data is delayed 18-36 hours (not real-time) due to NCEI collection method.

## 📊 Variables

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

## 🌐 Data Source

- **Endpoint:** https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py
- **Docs:** https://mesonet.agron.iastate.edu/request/asos/1min.phtml
- **IEM Processing:** https://github.com/akrherz/iem/blob/main/scripts/ingestors/asos_1minute/parse_ncei_asos1minute.py
- **Archive:** Available for US ASOS sites back to 2000

Part of the OpenMesh project.
