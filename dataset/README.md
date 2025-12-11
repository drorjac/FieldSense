# FieldSense Dataset Collection

This directory contains a collection of open datasets for opportunistic sensing, environmental monitoring, and multi-sensor data fusion. All datasets follow the [OpenSense-1.0 naming convention](https://github.com/OpenSenseAction/opensense_example_data) and are formatted as NetCDF files for interoperability.

## 📁 Directory Structure

```
dataset/
├── open_datasets/          # Raw/open datasets from various sources
│   ├── OpenMesh_NYC/       # NYC Community Mesh Network data
│   ├── OpenMRG_Sweden/     # Swedish microwave link, radar, and gauge data
│   ├── CML_Netherlands/    # Netherlands commercial microwave link data
│   └── OpenRainER_Italy/   # Italian precipitation dataset
├── our_datasets/           # Processed datasets (ready for analysis)
├── processing/             # Data processing scripts and pipelines
└── examples/               # Example notebooks and tutorials
```

## 🌐 Available Datasets

### 1. OpenMesh (NYC Community Mesh Network)

**Location:** `open_datasets/OpenMesh_NYC/`

**Description:** Wireless signal dataset from the NYC Community Mesh Network for opportunistic urban weather sensing. Contains microwave link data, personal weather stations (PWS), and ASOS station observations.

**Data Coverage:**
- **Links Data:** October 2023 - July 2024 (complete)
- **PWS Data:** January 15-30, 2024 (sample)
- **Network:** 75 links with 3 sublinks each
- **Time Resolution:** Variable (5-minute to hourly)

**Contents:**
- `links/` - Microwave link signal level data (RSL/TSL) in NetCDF format
- `weather_stations/` - Personal weather stations and ASOS station data
- `maps/` - Interactive network topology visualizations
- `fetch_data/` - Data collection and processing pipelines

**Dataset Links:**
- **Zenodo:** [10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692)
- **GitHub Repository:** [github.com/drorjac/OpenMesh](https://github.com/drorjac/OpenMesh)
- **Network Source:** [NYC Mesh Network](https://www.nycmesh.net/)

**Citation:**
```bibtex
@article{jacoby2025openmesh,
  title={OpenMesh: Wireless Signal Dataset for Opportunistic Urban Weather Sensing in New York City},
  author={Jacoby, Dror and Yu, Shuyue and Hu, Qianfei and Hine, Zachary and Johnson, Rob and Ostrometzky, Jonatan and Kadota, Igor and Zussman, Gil and Messer, Hagit},
  journal={Earth System Science Data},
  year={2025}
}
```

**Format:** OpenSense-1.0 (grouped NetCDF4)

---

### 2. OpenMRG (Sweden)

**Location:** `open_datasets/OpenMRG_Sweden/`

**Description:** Open data from Sweden combining Commercial Microwave Links (CML), weather radar, and rain gauge measurements for hydrological and meteorological analysis. Covers the Gothenburg region during summer 2015.

**Data Coverage:**
- **Period:** June-August 2015 (JJA)
- **Region:** Gothenburg, Sweden
- **Sensors:** CML links, SMHI rain gauges, weather radar

**Contents:**
- `cml/` - Commercial microwave link data (NetCDF format)
- `gauges/` - Rain gauge data (SMHI and City gauges)
- `radar/` - Weather radar data
- Example reading scripts and metadata

**Dataset Links:**
- **OpenSense Action:** [OpenMRG Dataset](https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/)
- **Zenodo:** [10.5281/zenodo.6673750](https://doi.org/10.5281/zenodo.6673750)
- **License:** CC BY-SA 4.0

**Format:** NetCDF (CF conventions) and CSV

---

### 3. CML Netherlands

**Location:** `open_datasets/CML_Netherlands/`

**Description:** Four-year comprehensive dataset of commercial microwave link data from the Netherlands (2011-2015), used for rainfall estimation and related studies.

**Data Coverage:**
- **Period:** 2011-2015 (4 years)
- **Region:** Netherlands
- **Network:** Commercial microwave link infrastructure

**Dataset Links:**
- **TU Delft Data Repository:** [10.4121/be252844-b672-471e-8d69-27269a862ec1.v1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1.v1)
- **OpenSense Action:** [Four-year CML Dataset](https://opensenseaction.eu/datasets/four-year-commercial-microwave-link-dataset-for-the-netherlands/)

**Format:** NetCDF

---

### 4. OpenRainER (Italy)

**Location:** `open_datasets/OpenRainER_Italy/`

**Description:** Precipitation dataset from the Emilia-Romagna region in Italy containing two years of multi-sensor precipitation data including weather radar, rain gauge, and Commercial Microwave Link (CML) measurements.

**Data Coverage:**
- **Period:** 2021-2022 (2 years)
- **Region:** Emilia-Romagna, Italy
- **Sensors:** Weather radar, rain gauges, CML links

**Dataset Links:**
- **Zenodo:** [10.5281/zenodo.10593848](https://zenodo.org/record/10593848)
- **OpenSense Action:** [OpenRainER Dataset](https://opensenseaction.eu/news/new-open-cml-dataset-from-italy-openrainer/)

**Format:** NetCDF

---

## 📊 Data Format Standards

All datasets in this collection follow standardized formats for interoperability:

### OpenSense-1.0 Convention

The [OpenSense-1.0 naming convention](https://github.com/OpenSenseAction/opensense_example_data) provides:
- Standardized NetCDF structure for CML data
- Consistent variable naming and metadata
- Grouped NetCDF4 format for multi-link datasets
- CF conventions compliance where applicable

**Key Features:**
- **Dimensions:** `time`, `cml_id`, `sublink_id`
- **Coordinates:** Spatial coordinates, frequency, polarization, link length
- **Variables:** `rsl` (Received Signal Level), `tsl` (Transmitted Signal Level)
- **Metadata:** Comprehensive global attributes following OpenSense standards

### Example Reading Code

```python
import xarray as xr

# Load OpenSense-1.0 formatted dataset
ds = xr.open_dataset('path/to/dataset.nc')

# Access link data
rsl = ds.rsl  # Received Signal Level
time = ds.time  # Time coordinates
cml_ids = ds.cml_id  # Link identifiers
```

See individual dataset folders for specific example notebooks.

---

## 🔄 Data Processing Workflow

1. **Raw Data** → Stored in `open_datasets/` with original format
2. **Processing** → Scripts in `processing/` convert to standardized format
3. **Processed Data** → Saved in `our_datasets/` ready for analysis

### Processing Scripts

- Dataset-specific conversion scripts
- Format standardization tools
- Quality control and validation
- Metadata extraction and enrichment

---

## 📚 Example Notebooks

Each dataset folder contains example notebooks demonstrating:
- Data loading and reading
- Basic visualization
- Data quality assessment
- Format compliance checking

**Notable Examples:**
- `OpenMesh_NYC/links/openmesh_dataset_example.ipynb` - Link data exploration
- `OpenMesh_NYC/weather_stations/read_pws_sample.ipynb` - Weather station data
- `OpenMRG_Sweden/cml/example_read_cml.nc.py` - CML data reading

---

## 🔗 Related Resources

### OpenSense Action
- **Website:** [opensenseaction.eu](https://opensenseaction.eu/)
- **GitHub:** [github.com/OpenSenseAction](https://github.com/OpenSenseAction)
- **Example Data Repository:** [github.com/OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data)

### Standards and Conventions
- **OpenSense-1.0:** [OpenSense naming convention](https://github.com/OpenSenseAction)
- **CF Conventions:** [cfconventions.org](http://cfconventions.org/)
- **NetCDF:** [unidata.ucar.edu/software/netcdf](https://www.unidata.ucar.edu/software/netcdf/)

### Tools and Libraries
- **xarray:** Multi-dimensional arrays with labeled dimensions
- **netCDF4:** Python interface to NetCDF files
- **PyNNcml:** Neural network tools for CML data analysis

---

## 📝 Dataset Usage Guidelines

### Citation Requirements

When using datasets from this collection:
1. **Cite the original dataset** using the provided DOI/citation
2. **Acknowledge data sources** (e.g., NYC Mesh Network, SMHI, NOAA)
3. **Reference this repository** if using processed versions

### License Information

Each dataset maintains its original license:
- **OpenMesh:** CC BY 4.0
- **OpenMRG:** CC BY-SA 4.0
- **CML Netherlands:** Check original repository
- **OpenRainER:** Check Zenodo record

### Data Access

- **Sample data** is included in this repository
- **Complete datasets** may require download from original sources
- **Processing scripts** are provided for data conversion

---

## 🛠️ Contributing

To add a new dataset:

1. Create a folder in `open_datasets/` following naming convention
2. Include a `README.md` with:
   - Dataset description and coverage
   - Original source links and citations
   - Data format and structure
   - Example usage code
3. Add processing scripts in `processing/` if needed
4. Update this main README with dataset information

---

## 📧 Contact

For questions about datasets or data processing:
- Check individual dataset README files
- Refer to original dataset repositories
- Open an issue in the project repository

---

## 🔄 Version History

- **2025-01:** Initial dataset collection structure
- Datasets follow OpenSense-1.0 conventions
- Standardized NetCDF format for interoperability

---

## 📖 Additional Documentation

- **Processing Guide:** See `processing/` directory
- **Examples:** See `examples/` directory
- **Dataset-Specific Docs:** See individual dataset folders

---

*Last Updated: January 2025*
