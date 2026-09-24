# CML Retrieval — PyNNcml Tutorials

Rain retrieval from commercial microwave links using **PyNNcml**, worked
end to end against the OpenMRG dataset. These were previously filed under
`core/examples/`; they are a topic in their own right, not a shared utility.

```
cml_retrieval/
├── notebooks/
│   ├── model_driven_tutorial.ipynb    # wet-dry, baseline models, IDW/GMZ
│   ├── data_driven_tutorial.ipynb     # neural-network retrieval
│   └── rainfall_maps_idw_gmz.ipynb    # five reconstructions, side by side
├── src/
│   └── rain_maps.py                   # one API over four packages
└── README.md
```

## Rainfall maps, five ways

`src/rain_maps.py` runs five reconstructions over the same links at the same
timestep, so the differences are the methods rather than the input:

| | package | geometry | peak (mm/h) | grid mean |
|---|---|---|---|---|
| IDW | `pycomlink` | midpoint | 85.9 | 2.63 |
| IDW | `pynncml` | midpoint | 84.6 | 1.30 |
| **GMZ** | `pynncml` | **line**, 3 pts/link | 66.6 | 1.68 |
| IDW | `mergeplg` | **line** | 72.9 | 2.13 |
| block kriging | `mergeplg` | **line** | 80.5 | 2.19 |

```bash
jupyter lab projects/cml_retrieval/notebooks/rainfall_maps_idw_gmz.ipynb
```

The split that matters is midpoint versus line. A CML measures a path average
over kilometres; collapsing it to its midpoint throws that away. GMZ
(Goldshtein–Messer–Zinevich) and both `mergeplg` methods keep it, and it
shows: GMZ has the highest 99th percentile of the five (42.8 against
30–35 mm/h) while having the *lowest* peak, because it spreads intensity
along the paths instead of piling it onto midpoints.

The grid means span a factor of two, which is mostly a coverage difference
rather than a physical one — `pynncml` only fills a radius around each link
and leaves the rest masked, while the others interpolate across the whole
domain.

### Two bugs in pynncml's GMZ

Found while building this, both in `compute_rain_point_from_field`
(pynncml 0.3.7). `rain_maps.patch_pynncml_gmz()` fixes them in memory; they
belong upstream at [haihabi/PyNNcml](https://github.com/haihabi/PyNNcml).

**The ceiling index is unclamped.** `i_ceiling` reaches `len(grid)` whenever a
link point lands on the edge of the bounding box built from the links
themselves — so it raises `IndexError` on any full network, which is how it
surfaced.

**The ceiling-ceiling corner reads the wrong axis.**
`cc = in_rain_map[:, j_ceiling, j_ceiling]` uses the *y* index for both
dimensions where it should be `[i_ceiling, j_ceiling]`. This one does not
raise — it silently samples the wrong grid cell for one of the four bilinear
corners, wherever `i_ceiling != j_ceiling`, which is almost everywhere. Any
GMZ result produced with this version is affected.

## Related work in this repository

| | |
|---|---|
| `projects/opensense_pipeline/` | the same problem at scale — full retrieval chain on OpenMRG and OpenRainER, then `mergeplg` merging |
| `projects/rainfall_field_sim/` | synthetic rain fields and CML sampling, with known ground truth |
| `core/scientific_packages/PYNNcml.md` | PyNNcml reference notes |
| `core/scientific_packages/pynncml_wrapper.py` | thin wrapper used by these tutorials |

## Overview

These tutorials demonstrate how to **PyNNcml** (Neural Network-based Commercial Microwave Link rain estimation) with the OpenMRG dataset.

## Overview

PyNNcml is a Python toolbox based on PyTorch that utilizes neural networks for rain estimation and classification from commercial microwave link (CML) data. These tutorials demonstrate how to:

- Load and process CML data from the OpenMRG dataset
- Perform wet-dry classification using statistical methods
- Estimate rainfall intensity using baseline models
- Reconstruct rainfall fields using spatial interpolation methods

## Tutorials

### 1. Model-Driven Tutorial (`model_driven_tutorial.ipynb`)

**End-to-end model-driven rainfall detection, estimation, and rain field reconstruction.**

This tutorial covers:
- Loading the OpenMRG dataset
- Wet-dry classification using statistics test
- Rainfall intensity estimation using:
  - Two-steps constant baseline model
  - One-step dynamic baseline model
- Rain field reconstruction using:
  - Inverse Distance Weighting (IDW)
  - Goldshtein, Messer, Zinevich (GMZ) method

### 2. Data-Driven Tutorial (`data_driven_tutorial.ipynb`)

**Neural network-based approaches for CML rain estimation.**

This tutorial demonstrates:
- Wet-dry classification using LSTM networks
- Rain estimation using RNN models
- Training custom neural network models
- Model evaluation and robustness testing

## Prerequisites

1. **Activate the conda environment:**
   ```bash
   conda activate fieldsense
   ```

2. **Install PyNNcml** (if not already installed):
   ```bash
   pip install pynncml
   ```

3. **Required packages:**
   - PyTorch
   - NumPy
   - Pandas
   - Matplotlib
   - netCDF4
   - xarray

## Dataset

The tutorials use the **OpenMRG dataset** (Open data from microwave links, radar, and gauges):
- **Location**: Data is automatically downloaded to `./data/OpenMRG.zip` on first run. This
  `data/` directory is git-ignored — it is a download target, not tracked content.
- **Source**: [OpenSense Action - OpenMRG Dataset](https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/)
- **Already in this repo**: a copy of OpenMRG (metadata, gauges, radar, reader scripts and
  the dataset `readme.txt`) lives at `dataset/open_datasets/OpenMRG_Sweden/`. Point the
  notebooks there, or symlink it, to skip the download:
  ```bash
  ln -s ../../dataset/open_datasets/OpenMRG_Sweden projects/cml_retrieval/data
  ```
- **Alternative**: Place `cml.nc` and `cml_metadata.csv` in `data/cml/` directory

### Data Structure

```
data/                        # git-ignored; mirrors dataset/open_datasets/OpenMRG_Sweden/
├── cml/
│   ├── cml.nc              # NetCDF file with RSL/TSL measurements
│   ├── cml_metadata.csv    # Link metadata (coordinates, frequencies, etc.)
│   └── example_read_cml.nc.py
├── gauges/
│   ├── city/               # City gauge data
│   └── smhi/               # SMHI gauge data
└── radar/                  # Radar data
```

## Quick Start

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open a tutorial:**
   - Start with `model_driven_tutorial.ipynb` for model-driven approaches
   - Or `data_driven_tutorial.ipynb` for neural network methods

3. **Run the cells** - The notebook will automatically:
   - Check if PyNNcml is installed
   - Download the OpenMRG dataset if needed
   - Load and process the data

## Loading Fewer Links

To reduce computation time, you can limit the number of links loaded:

```python
import pynncml as pnc

# Option 1: Load only links near rain gauges (fewer links, default)
link_set, ps, _ = pnc.datasets.load_open_mrg(
    time_slice=slice("2015-06-01", "2015-06-10"),
    change2min_max=True,
    link_selection=pnc.datasets.xarray_processing.LinkSelection.GAUGEONLY
)

# Option 2: Limit to smaller geographic region
link_set, ps, _ = pnc.datasets.load_open_mrg(
    time_slice=slice("2015-06-01", "2015-06-10"),
    change2min_max=True,
    xy_min=[11.9, 57.7],  # [lon_min, lat_min]
    xy_max=[12.0, 57.8],  # [lon_max, lat_max]
    link_selection=pnc.datasets.xarray_processing.LinkSelection.GAUGEONLY
)

# Option 3: Reduce link-to-gauge distance
link_set, ps, _ = pnc.datasets.load_open_mrg(
    time_slice=slice("2015-06-01", "2015-06-10"),
    change2min_max=True,
    link2gauge_distance=1000,  # Smaller = fewer links (default: 2000m)
    link_selection=pnc.datasets.xarray_processing.LinkSelection.GAUGEONLY
)
```

## FieldSense Integration

These tutorials are integrated with the FieldSense project's data loading utilities:

```python
from core.scientific_packages.pynncml_wrapper import load_openmrg_from_local, find_openmrg_data

# Automatically searches for data files
data_path = find_openmrg_data()
link_set, ps, _ = load_openmrg_from_local(time_slice=slice("2015-06-01", "2015-06-10"))
```

## Official PyNNcml Resources

For more examples, documentation, and advanced usage:

- **GitHub Repository**: [https://github.com/haihabi/pynncml](https://github.com/haihabi/pynncml)
- **PyPI Package**: [https://pypi.org/project/pynncml/](https://pypi.org/project/pynncml/)
- **Examples**: [https://github.com/haihabi/pynncml/tree/main/examples](https://github.com/haihabi/pynncml/tree/main/examples)
- **Tutorials**: [https://github.com/haihabi/pynncml/tree/main/examples/tutorials](https://github.com/haihabi/pynncml/tree/main/examples/tutorials)

## Key Papers

If using PyNNcml in research, consider citing:

1. **Habi, H. V., & Messer, H.** (2020). "Recurrent Neural Network for Rain Estimation Using Commercial Microwave Links." *IEEE Transactions on Geoscience and Remote Sensing*, 59(5), 3672-3681.

2. **Habi, H. V., & Messer, H.** (2020). "Wet-Dry Classification Using LSTM and Commercial Microwave Links."

3. **Habi, H. V.** (2020). "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links."

## Troubleshooting

### AttributeError: 'list' object has no attribute 'data_array'

If you encounter this error when accessing `gauge_ref`, the notebook includes code to handle both list and object formats. The gauge reference data structure may vary depending on the PyNNcml version.

### Data Not Found

If data files are not found:
1. Check that `./data/OpenMRG.zip` exists (will be downloaded automatically)
2. Or place `cml.nc` in `data/cml/` directory
3. The notebook will search common locations automatically

### Memory Issues

If you run out of memory:
- Use `link_selection=LinkSelection.GAUGEONLY` to load fewer links
- Reduce the time slice range
- Use spatial filtering with `xy_min` and `xy_max`

## Related Resources

- [PyNNcml Documentation](../scientific_packages/PYNNcml.md)
- [FieldSense Data Loaders](../../scientific_packages/pynncml_wrapper.py)
- [OpenMRG Dataset README](../../../dataset/open_datasets/OpenMRG_Sweden/README.md)

