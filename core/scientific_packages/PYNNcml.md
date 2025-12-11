# PyNNcml Resources

**Neural Network-based Commercial Microwave Link Rain Estimation**

PyNNcml is a Python toolbox based on PyTorch that utilizes neural networks for rain estimation and classification from commercial microwave link (CML) data.

## Overview

Commercial Microwave Links (CMLs) are radio links used for cellular communication. When rain occurs, the signal attenuation increases, which can be used to estimate rainfall intensity. PyNNcml implements neural network-based methods to:

- Classify wet/dry conditions from CML data
- Estimate rain rates from CML attenuation measurements
- Establish dynamic baselines for signal levels
- Evaluate model performance and robustness

## Installation

```bash
pip install pynncml
```

**Note:** Requires PyTorch. If not already installed:

```bash
pip install torch torchvision
```

## Official Resources

| Resource | Link |
|----------|------|
| GitHub | https://github.com/haihabi/pynncml |
| PyPI | https://pypi.org/project/pynncml/ |
| Author | Hai Victor Habi |

## Key Features

### 1. Wet-Dry Classification
- Uses LSTM (Long Short-Term Memory) networks
- Distinguishes between wet and dry periods from CML data
- Essential preprocessing step for rain estimation

### 2. Baseline Estimation
- Dynamic baseline determination
- Adapts to varying signal conditions
- Handles signal drift and variations

### 3. Rain Estimation
- Recurrent Neural Network (RNN) models
- Power-law relationships
- Converts attenuation to rain rate (mm/h)

### 4. Metrics and Evaluation
- Performance assessment tools
- Robustness evaluation
- Comparison with ground truth (rain gauges)

## Dependencies

```
torch
torchvision
numpy
pandas
scipy
matplotlib
netcdf4
xarray
huggingface_hub
poligrain
prettytable
utm
tqdm
```

## Quick Example

```python
import pynncml
import numpy as np

# Load CML data (example structure)
# data should contain RSL (Received Signal Level) measurements
rsl_data = np.load('cml_data.npy')  # Shape: (n_timesteps, n_links)

# Wet-dry classification
classifier = pynncml.WetDryClassifier()
wet_dry_labels = classifier.classify(rsl_data)

# Rain estimation
estimator = pynncml.RainEstimator()
rain_rates = estimator.estimate(rsl_data, wet_dry_labels)
```

## Dataset Integration

### OpenMRG Dataset (Sweden)
- Available in `dataset/open_datasets/OpenMRG_Sweden/`
- Includes CML data in NetCDF format (`cml.nc`)
- Contains RSL and TSL (Transmitted Signal Level) measurements
- Metadata in `cml_metadata.csv`

### Reading CML Data from NetCDF

```python
import netCDF4
import numpy as np

# Open NetCDF file
nc = netCDF4.Dataset('cml.nc', 'r')

# Read RSL data
rsl = np.array(nc.variables['rsl'][:])  # Shape: (time, sublink)

# Handle missing values
rsl[rsl == 1e10] = np.nan

# Read time
time = nc.variables['time'][:]
time_units = nc.variables['time'].units

nc.close()
```

## Applications in FieldSense

PyNNcml is relevant for:
- **Multi-sensor data fusion**: Combining CML with rain gauges and radar
- **Spatio-temporal field reconstruction**: Using CML network for spatial rain mapping
- **Wireless sensing**: Leveraging existing communication infrastructure for environmental sensing
- **Physics-informed ML**: Integrating physical attenuation models with neural networks

## Example Workflows

### 1. Basic Rain Estimation Pipeline

```python
# 1. Load CML data
cml_data = load_cml_data('path/to/cml.nc')

# 2. Wet-dry classification
wet_dry = classify_wet_dry(cml_data)

# 3. Baseline correction
baseline = compute_baseline(cml_data, wet_dry)

# 4. Attenuation calculation
attenuation = baseline - cml_data

# 5. Rain rate estimation
rain_rate = estimate_rain(attenuation, link_length, frequency)
```

### 2. Training Custom Models

```python
from pynncml import train_rnn_model

# Prepare training data
X_train, y_train = prepare_training_data(cml_train, gauge_train)

# Train RNN model
model = train_rnn_model(
    X_train, y_train,
    model_type='lstm',
    hidden_size=64,
    num_layers=2,
    epochs=50
)
```

## Key Papers

If using PyNNcml in research, consider citing:

1. **Habi, H. V., & Messer, H.** (2020). "Wet-Dry Classification Using LSTM and Commercial Microwave Links."

2. **Habi, H. V., & Messer, H.** (2020). "RNN Models for Rain Detection."

3. **Habi, H. V.** (2020). "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links."

4. **Habi, H. V., & Messer, H.** (2020). "Recurrent Neural Network for Rain Estimation Using Commercial Microwave Links." *IEEE Transactions on Geoscience and Remote Sensing*, 59(5), 3672-3681.

## Tips

1. **Data quality**: CML data often contains missing values and outliers. Preprocess carefully.
2. **Baseline stability**: Dynamic baselines help handle signal drift over time.
3. **Link characteristics**: Use link-specific parameters (length, frequency, polarization).
4. **Validation**: Always validate with independent rain gauge data.
5. **Spatial interpolation**: Combine multiple CML links for spatial rain mapping.

## Related Datasets in FieldSense

- **OpenMRG (Sweden)**: `dataset/open_datasets/OpenMRG_Sweden/`
- **CML Netherlands**: `dataset/open_datasets/CML_Netherlands/`
- **OpenRainER (Italy)**: `dataset/open_datasets/OpenRainER_Italy/`

## Integration with Other Tools

PyNNcml can be combined with:
- **SINDy**: Discover governing equations from CML-derived rain fields
- **PySR**: Symbolic regression for rain rate relationships
- **Spatial interpolation**: Kriging, IDW, etc. for spatial mapping

