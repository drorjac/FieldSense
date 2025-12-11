-------
The OpenMesh Sample Dataset
-------
Version: 1.0 (Sample Package)
DOI: 10.5281/zenodo.15287692
License: https://creativecommons.org/licenses/by/4.0/

--------
Description of files in the folder "links"

The wireless link data from NYC Community Mesh Network (https://www.nycmesh.net/) is provided in NetCDF files formatted according to the OpenSense-1.0 convention (https://github.com/OpenSenseAction). This data covers the full observation period from October 2023 through July 2024 and represents opportunistic sensing using the mesh network's microwave communication infrastructure.

**ds_openmesh.nc** - Complete Link Dataset
Organized as a grouped NetCDF4 file where each link (cml_id) is stored as a separate group. Within each link group, data is structured with two dimensions (sublink_id and time) and contains the following variables:

- rsl: Received Signal Level in dBm for each sublink and time step
- tsl: Transmitted Signal Level in dBm for each sublink and time step
- frequency: Operating frequency in MHz
- length: Physical path length in meters
- polarization: Antenna polarization (v=vertical, h=horizontal)

Time is in UTC (seconds since 1970-01-01 00:00:00) and represents the timestamp when signal levels were recorded. Each link typically consists of multiple sublinks representing different frequency bands or bidirectional paths between the same two sites.

**links_metadata.csv** - Link Metadata
Contains comprehensive metadata for all sublinks:
- cml_id: Link identifier
- sublink_id: Sublink identifier within each link (sublink_1, sublink_2, etc.)
- site_0_lat, site_0_lon: Coordinates of first endpoint (decimal degrees, WGS84)
- site_1_lat, site_1_lon: Coordinates of second endpoint (decimal degrees, WGS84)
- frequency: Operating frequency in MHz
- length: Link path length in meters
- polarization: Antenna polarization

**openmesh_dataset_example.ipynb** - Example Notebook
Jupyter notebook demonstrating how to load the grouped NetCDF structure, access link data, and create basic visualizations. The grouped NetCDF structure can be read using standard NetCDF libraries (e.g., netCDF4 in Python, ncdf4 in R).

------
Description of files in the folder "weather stations"

This folder contains weather station observations from two sources: Personal Weather Stations (PWS) from Weather Underground and ASOS (Automated Surface Observing System) stations from NOAA.

**IMPORTANT: PWS data contains only a TWO-WEEK SAMPLE (January 15-30, 2024) for demonstration purposes. This is NOT the complete PWS dataset.**

**pws_opensense_sample_jan.nc** - PWS Sample Data
Personal Weather Station observations from Weather Underground's community network (https://www.wunderground.com/) provided in NetCDF format following the OpenSense-1.0 convention. The complete PWS dataset spans the same period as the links data (October 2023 - July 2024) but only this two-week subset is included in this sample package.

Organized as a grouped NetCDF4 file where each weather station is a separate group. Each station group contains:

- time dimension with measurements at 5-minute intervals
- Scalar coordinates: latitude, longitude, altitude (in meters)
- Data variables:
  - rainfall_rate: Rainfall intensity in mm/h
  - rainfall_amount: Accumulated rainfall in mm per 5-minute interval
  - temperature: Air temperature in °C
  - relative_humidity: Relative humidity in %
  - wind_velocity: Wind speed in m/s
  - wind_direction: Wind direction in degrees
  - air_pressure: Atmospheric pressure in hPa

**pws_metadata.csv** - PWS Station Metadata
Station information for all Weather Underground PWS stations:
- Station ID: Weather Underground station identifier (e.g., KNYNEWYO1805)
- Latitude, Longitude: Station coordinates in decimal degrees (WGS84)
- Elevation: Station elevation in meters
- Description: Station source (WU PWS = Weather Underground Personal Weather Station)

**ASOS_stations.csv** - ASOS Station Metadata
Metadata for NOAA ASOS (Automated Surface Observing System) stations used for rainfall validation and comparison. Data were obtained via the NCEI Data Access Application (https://www.ncei.noaa.gov/access/), which provides access to official NOAA/NCEI archives including Local Climatological Data (LCD) and Integrated Surface Dataset (ISD).

Contains:
- Station ID: ASOS/AWOS station identifier
- Station Name: Full station name
- Latitude, Longitude: Station coordinates in decimal degrees (WGS84)
- Elevation: Station elevation in meters

**read_pws_sample.ipynb** - Example Notebook
Jupyter notebook demonstrating how to read the grouped NetCDF structure and visualize weather station data for all stations in the sample period.

**Data Source Acknowledgments**:
- Weather Underground (https://www.wunderground.com/) and the community of personal weather station operators who generously share their observations publicly
- NOAA National Centers for Environmental Information (NCEI) for ASOS/AWOS station data


------
Description of folder "maps"

This folder contains interactive web-based visualizations of the NYC Community Mesh Network topology. Both maps can be viewed by opening the HTML files in any modern web browser (Chrome, Firefox, Safari, Edge) - no additional software or installation required.
- Link metadata accessible by clicking on individual links

**frequency_map.html** - Frequency Band Distribution
Displays the complete network with links colored by operating frequency band  with different frequency allocations (5 GHz, 24 GHz, 60 GHz, 70Ghz.)

**directional_map.html** - Link Directionality and Configuration
Shows the directional characteristics and configuration of network links.


**Technical Details:**
The visualizations are built using Folium (Python library) and Leaflet.js (JavaScript mapping library). Base maps are provided by OpenStreetMap contributors. No API keys or credentials are required to view the maps.

**Map Credits:**
- Base map data © OpenStreetMap contributors
- Interactive mapping library: Leaflet.js (https://leafletjs.com/)
- Python interface: Folium (https://python-visualization.github.io/folium/)
------
------
Example Code and Data Processing Pipeline

This sample package includes example code demonstrating how to read and work with the OpenMesh data formats:

- openmesh_dataset_example.ipynb: Jupyter notebook showing how to load grouped NetCDF link data
- read_pws_sample.ipynb: Jupyter notebook demonstrating PWS data loading and visualization

**Complete Data Processing Pipeline:**

For the full code pipeline including:
- Weather data API and processing workflows
- Data formatting to OpenSense-1.0 standard
- Analysis scripts and visualization tools
- Integration with corresponding grid products

Visit the official OpenMesh GitHub repository:
https://github.com/drorjac/OpenMesh

The repository contains code used for data collection, processing, and analysis, enabling reproducibility and providing templates for working with similar opportunistic sensing datasets.

------
Dataset Citation

When using OpenMesh dataset, please cite:

@article{jacoby2025openmesh,
  title={OpenMesh: Wireless Signal Dataset for Opportunistic Urban Weather Sensing in New York City},
  author={Jacoby, Dror and Yu, Shuyue and Hu, Qianfei and Hine, Zachary and Johnson, Rob and Ostrometzky, Jonatan and Kadota, Igor and Zussman, Gil and Messer, Hagit},
  journal={Earth System Science Data Discussions},
  volume={2025},
  pages={1--27},
  year={2025},
  publisher={G{\"o}ttingen, Germany}
}

Jacoby, D., et al. (2025). OpenMesh: Wireless Signal Dataset for Opportunistic Urban Weather Sensing in New York City. Earth System Science Data.

GitHub Repository: https://github.com/drorjac/OpenMesh

And acknowledge:
- NYC Community Mesh Network (https://www.nycmesh.net/) for providing access to the wireless infrastructure
- Weather Underground (https://www.wunderground.com/) and community weather station operators for PWS observations

------
Complete Dataset Access

This package contains:
- **Links data**: COMPLETE dataset (October 2023 - July 2024)
- **PWS data**: SAMPLE ONLY (January 15-30, 2024)

For access to the complete PWS dataset covering the full observation period, please refer to the GitHub Repository

------
Contact Information

For questions about this dataset or to request the complete PWS data:

Dror Jacoby
Cellular Environmental Monitoring (CellEnMon) Lab, Tel Aviv University
Wireless and Mobile Networking (WiMNet) Lab, Columbia University
Email: drorjacoby@mail.tau.ac.il
------
References

OpenMesh GitHub Repository: https://github.com/drorjac/OpenMesh
OpenSense data format: https://github.com/OpenSenseAction
NYC Community Mesh Network: https://www.nycmesh.net/
Weather Underground: https://www.wunderground.com/