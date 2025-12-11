-------
The OpenMRG data set
-------
Version: 1.1
DOI: https://doi.org/10.5281/zenodo.6673750
Licence: https://creativecommons.org/licenses/by-sa/4.0

--------
Description of files in the folder "cml"

The Commercial Microwave Link (CML) data is provided in a NetCDF file based on the CF convention (http://cfconventions.org/). The file is called cml.nc and consists two dimensions (sublink and time) and two data variables (rsl and tsl). The rsl and tsl variables contain the recorded signal levels in dBm for each sub-link and time step. The time is in UTC (seconds since 1970-01-01 00:00:00) and represents start of the sampling window during which the RSL or TSL was sampled. Sublink is a numeric ID of each sub-link, and it corresponds to the Sublink column in the file cml_metadata.csv, which contains additional metadata about the sub-links: polarisation, frequency (in GHz), length (in km) between the nodes, and coordinates (in decimal degrees, WGS84) of the transmitting (Near) and receiving (Far) node of each sub-link. The file cml_metadata.csv also contain a Link identifier, to facilitate matching sub-link pairs, as well as a Direction code for each sub-link. The cml.nc file can be read by standard software and libraries supporting NetCDF (e.g. cdo, ncdf4 in R, netCDF4 in python), an example script is included (example_read_cml.nc.py). 

------
Description of files in the folder "gauges"

The SMHI gauge data is provided for the station GbgA (full name Göteborg A) in the comma-separated CSV files GbgA-71420-2015JJA.csv and GbgA-71420-metadata.csv within subfolder "smhi". GbgA-71420-metadata.csv contains metadata for the station (e.g. station ID, coordinates) and metadata for each measured parameter (e.g. Short name, full name, unit). The file also provides quality code (QC) definitions. The file GbgA-71420-2015JJA.csv contains time step, measured values and quality codes for each parameter. Time_UTC defines the date and time in UTC, formatted according to https://en.wikipedia.org/wiki/ISO_8601. The measured values are provided in the column named "[Short name]_[unit]", and the corresponding quality codes in the column named "[Short name]_QC". The measurement interval differs between parameters, as defined in GbgA-71420-metadata.csv. For example, accumulated precipitation (Pvol) the measurement interval starts 15min before this timestamp and ends at the given Time_UTC. Pvol_mm provides the accumulated precipitation recorded during the measurement interval in millimetre. Pvol_QC provides a quality code for each measurement. The included GbgA data covers June-August 2015; while data from 1996 onwards is available at https://opendata.smhi.se/ (specifically, https://opendata-download-metobs.smhi.se/explore).

The City network data is provided in the comma-separated CSV files CityGauges-2015JJA.csv and CityGauges-metadata.csv within the subfolder "city". CityGauges-2015JJA.csv contains 11 columns: Time_UTC, and then one column for each station (e.g. Jarn). Time_UTC defines the date and time in UTC at the end of the measurement interval (each measurement interval ends at the given timestamp and starts on second 01 of the preceding minute). Each station column contains the accumulated rainfall recorded during the measurement interval in millimetre. The tipping-bucket records were converted to a regular time series assuming the rain fell during the preceding minute. Some known errors in the data were removed, indicated by NA. The file CityGauges-metadata.csv provides essential metadata for each station, including Name, Location, coordinates (decimal degrees, WGS84), gauge type, unit and measurement resolution. 

------
Description of files in the folder "radar"

The Radar data is provided in a NetCDF file and is called radar.nc. It consists of three dimensions (x, y and time), one data variable (data) and a variable containing the projection (crs). These projection settings can also be found in Table 2 of the reference article. Additionally, the file contains the two variables lat and lon that provide the centre coordinates for the grid cells. The data variable contains the recorded reflectivity in pseudo dBZ for each x, y grid cell and time step. Pseudo-dBZ means dbZ converted to values between 0 and 255 for efficient storage (255 represents missing data). The data values are linked to the scale_factor (0.4) and add_offset (-30) attributes. Therefore, the data will be automatically converted to dBZ (=0.4*data_value-30) during reading (for libraries that supports this e.g. netCDF4 in python). The time is in UTC (seconds since 1970-01-01 00:00:00) and represents the timestamp on which the reflectivity was sampled. The radar.nc file can be read by standard software and libraries supporting NetCDF (e.g. cdo, ncdf4 in R, netCDF4 in python), an example script is included (example_read_radar_nc.py). 

------
Description of files in the folder "Torslanda-event-2015-07-28"

In addition to the data we also include some illustrations of the Torslanda event observed on 28th July 2015: 1) an interactive HTML file showing time series of the Torslanda gauge, nearby CMLs and overlying radar pixel (Torslanda-interactive-time-series.zip). 2) a video animation showing the evolution of the radar field during the event (Torslanda-radar-animation.mp4). 
