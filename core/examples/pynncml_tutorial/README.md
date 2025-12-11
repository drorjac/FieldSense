# PyNNcml Tutorials

This directory contains tutorials for using **PyNNcml** (Neural Network-based Commercial Microwave Link rain estimation) with the OpenMRG dataset.

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
- **Location**: Data is automatically downloaded to `./data/OpenMRG.zip` on first run
- **Source**: [OpenSense Action - OpenMRG Dataset](https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/)
- **Alternative**: Place `cml.nc` and `cml_metadata.csv` in `data/cml/` directory

### Data Structure

```
data/
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

