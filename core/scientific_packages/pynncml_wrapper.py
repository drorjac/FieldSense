"""
PyNNcml Wrapper Module

Wrapper functions and utilities for working with PyNNcml (Neural Network-based
Commercial Microwave Link rain estimation) in the FieldSense project.

This module provides convenient interfaces for:
- Loading and preprocessing CML data
- Wet-dry classification
- Rain rate estimation
- Integration with FieldSense data loaders
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import warnings

try:
    import pynncml
    PYNNcML_AVAILABLE = True
except ImportError:
    PYNNcML_AVAILABLE = False
    warnings.warn(
        "pynncml is not installed. Install it with: pip install pynncml",
        ImportWarning
    )

try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

import glob
from pathlib import Path
import zipfile
import shutil


def load_cml_netcdf(
    filepath: str,
    variable: str = 'rsl',
    handle_missing: float = 1e10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CML data from NetCDF file.
    
    Parameters
    ----------
    filepath : str
        Path to NetCDF file (e.g., 'cml.nc')
    variable : str, optional
        Variable to load ('rsl' or 'tsl'), default 'rsl'
    handle_missing : float, optional
        Missing value indicator, default 1e10
        
    Returns
    -------
    data : np.ndarray
        CML data array, shape (n_time, n_sublinks)
    time : np.ndarray
        Time array in seconds since epoch
    """
    if not NETCDF4_AVAILABLE:
        raise ImportError("netCDF4 is required to load NetCDF files")
    
    nc = netCDF4.Dataset(filepath, 'r')
    
    # Load data
    data = np.array(nc.variables[variable][:])
    
    # Handle missing values
    data[data == handle_missing] = np.nan
    
    # Load time
    time = np.array(nc.variables['time'][:])
    
    nc.close()
    
    return data, time


def preprocess_cml_data(
    data: np.ndarray,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    fill_nan: bool = False,
    method: str = 'forward_fill'
) -> np.ndarray:
    """
    Preprocess CML data: remove outliers, handle NaN values.
    
    Parameters
    ----------
    data : np.ndarray
        CML data array, shape (n_time, n_links)
    remove_outliers : bool
        Whether to remove outliers using z-score
    outlier_threshold : float
        Z-score threshold for outlier detection
    fill_nan : bool
        Whether to fill NaN values
    method : str
        Method for filling NaN: 'forward_fill', 'linear', 'zero'
        
    Returns
    -------
    processed_data : np.ndarray
        Preprocessed CML data
    """
    processed = data.copy()
    
    # Remove outliers using z-score
    if remove_outliers:
        for i in range(processed.shape[1]):
            link_data = processed[:, i]
            valid_mask = ~np.isnan(link_data)
            if valid_mask.sum() > 0:
                z_scores = np.abs((link_data[valid_mask] - link_data[valid_mask].mean()) 
                                 / link_data[valid_mask].std())
                outlier_mask = np.zeros_like(valid_mask)
                outlier_mask[valid_mask] = z_scores > outlier_threshold
                processed[outlier_mask, i] = np.nan
    
    # Fill NaN values
    if fill_nan:
        if method == 'forward_fill':
            processed = pd.DataFrame(processed).fillna(method='ffill').fillna(method='bfill').values
        elif method == 'linear':
            processed = pd.DataFrame(processed).interpolate(method='linear').values
        elif method == 'zero':
            processed = np.nan_to_num(processed, nan=0.0)
    
    return processed


def compute_attenuation(
    rsl: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    method: str = 'dynamic'
) -> np.ndarray:
    """
    Compute attenuation from Received Signal Level (RSL).
    
    Parameters
    ----------
    rsl : np.ndarray
        Received Signal Level in dBm, shape (n_time, n_links)
    baseline : np.ndarray, optional
        Baseline signal level. If None, computed from data
    method : str
        Baseline method: 'dynamic', 'quantile', 'mean'
        
    Returns
    -------
    attenuation : np.ndarray
        Attenuation in dB, shape (n_time, n_links)
    """
    if baseline is None:
        if method == 'dynamic':
            # Use maximum (dry period) as baseline
            baseline = np.nanmax(rsl, axis=0, keepdims=True)
        elif method == 'quantile':
            # Use 95th percentile as baseline
            baseline = np.nanpercentile(rsl, 95, axis=0, keepdims=True)
        elif method == 'mean':
            # Use mean as baseline
            baseline = np.nanmean(rsl, axis=0, keepdims=True)
    
    attenuation = baseline - rsl
    attenuation[attenuation < 0] = 0  # No negative attenuation
    
    return attenuation


def estimate_rain_from_attenuation(
    attenuation: np.ndarray,
    link_lengths: np.ndarray,
    frequencies: np.ndarray,
    method: str = 'power_law'
) -> np.ndarray:
    """
    Convert attenuation to rain rate using power-law relationship.
    
    Parameters
    ----------
    attenuation : np.ndarray
        Attenuation in dB, shape (n_time, n_links)
    link_lengths : np.ndarray
        Link lengths in km, shape (n_links,)
    frequencies : np.ndarray
        Frequencies in GHz, shape (n_links,)
    method : str
        Estimation method: 'power_law'
        
    Returns
    -------
    rain_rate : np.ndarray
        Rain rate in mm/h, shape (n_time, n_links)
    """
    # Power-law relationship: A = a * R^b * L
    # Where A = attenuation (dB), R = rain rate (mm/h), L = length (km)
    # Typical values: a = 0.091, b = 1.0 (frequency-dependent)
    
    # Frequency-dependent coefficients (simplified)
    # More accurate: use ITU-R models
    a = 0.091  # dB/(mm/h)/km
    b = 1.0
    
    # Reshape for broadcasting
    lengths = link_lengths.reshape(1, -1)
    
    # Solve for R: R = (A / (a * L))^(1/b)
    rain_rate = np.power(attenuation / (a * lengths + 1e-10), 1.0 / b)
    
    # Handle invalid values
    rain_rate[rain_rate < 0] = 0
    rain_rate[np.isnan(rain_rate)] = 0
    
    return rain_rate


# Check if PyNNcml is available for advanced features
if PYNNcML_AVAILABLE:
    def classify_wet_dry_pynncml(
        rsl_data: np.ndarray,
        model_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Classify wet/dry conditions using PyNNcml.
        
        Parameters
        ----------
        rsl_data : np.ndarray
            RSL data, shape (n_time, n_links)
        model_path : str, optional
            Path to pre-trained model
            
        Returns
        -------
        labels : np.ndarray
            Binary labels (1=wet, 0=dry), shape (n_time, n_links)
        """
        # Placeholder - implement using PyNNcml API
        # This requires knowledge of PyNNcml's actual API
        raise NotImplementedError(
            "PyNNcml integration needs to be implemented based on actual API"
        )
    
    def estimate_rain_pynncml(
        rsl_data: np.ndarray,
        wet_dry_labels: Optional[np.ndarray] = None,
        model_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Estimate rain rates using PyNNcml neural network models.
        
        Parameters
        ----------
        rsl_data : np.ndarray
            RSL data, shape (n_time, n_links)
        wet_dry_labels : np.ndarray, optional
            Pre-computed wet/dry labels
        model_path : str, optional
            Path to pre-trained model
            
        Returns
        -------
        rain_rates : np.ndarray
            Rain rates in mm/h, shape (n_time, n_links)
        """
        # Placeholder - implement using PyNNcml API
        raise NotImplementedError(
            "PyNNcml integration needs to be implemented based on actual API"
        )


def find_openmrg_data(
    search_paths: Optional[list] = None,
    project_root: Optional[str] = None,
    include_zip: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Search for OpenMRG dataset files (cml.nc or OpenMRG.zip) in common locations.
    
    Parameters
    ----------
    search_paths : list, optional
        Custom paths to search. If None, uses default search locations.
    project_root : str, optional
        Root of FieldSense project. If None, tries to auto-detect.
    include_zip : bool
        Whether to also search for zip files
        
    Returns
    -------
    data_path : str or None
        Path to cml.nc file if found, None otherwise
    zip_path : str or None
        Path to OpenMRG.zip file if found, None otherwise
    """
    zip_path = None
    
    # Default search locations
    if search_paths is None:
        search_paths = []
        zip_paths = []
        
        # 1. FieldSense project location (relative to this file)
        if project_root is None:
            # Try to find FieldSense root (go up from core/scientific_packages)
            current_file = Path(__file__).resolve()
            fieldsense_root = current_file.parent.parent.parent
            project_root = str(fieldsense_root)
        
        fieldsense_data = Path(project_root) / "dataset" / "open_datasets" / "OpenMRG_Sweden" / "cml" / "cml.nc"
        search_paths.append(str(fieldsense_data))
        
        # Also check tutorial data location
        tutorial_data = Path(project_root) / "core" / "examples" / "pynncml_tutorial" / "data" / "cml" / "cml.nc"
        search_paths.append(str(tutorial_data))
        
        # 2. Common user data locations
        home = Path.home()
        common_locations = [
            home / "data" / "OpenMRG" / "cml.nc",
            home / "Downloads" / "OpenMRG" / "cml.nc",
            home / "Documents" / "OpenMRG" / "cml.nc",
            home / "Desktop" / "OpenMRG" / "cml.nc",
        ]
        search_paths.extend([str(p) for p in common_locations])
        
        # 3. Search for zip files if requested
        if include_zip:
            # Check common zip locations
            zip_candidates = [
                Path("./data/OpenMRG.zip"),  # Current directory (notebook location)
                Path(project_root) / "data" / "OpenMRG.zip",
                home / "data" / "OpenMRG.zip",
                home / "Downloads" / "OpenMRG.zip",
                home / "Downloads" / "openmrg.zip",
            ]
            zip_paths.extend([str(p) for p in zip_candidates])
            
            # Search in Downloads folder recursively
            downloads = home / "Downloads"
            if downloads.exists():
                for pattern in ["**/OpenMRG.zip", "**/openmrg.zip"]:
                    matches = list(downloads.glob(pattern))
                    if matches:
                        zip_paths.extend([str(m) for m in matches[:3]])  # Limit to 3 matches
        
        # 4. Search in Downloads folder recursively for cml.nc (max depth 3)
        downloads = home / "Downloads"
        if downloads.exists():
            for pattern in ["**/cml.nc", "**/OpenMRG*/cml.nc", "**/openmrg*/cml.nc"]:
                matches = list(downloads.glob(pattern))
                if matches:
                    search_paths.extend([str(m) for m in matches[:5]])  # Limit to 5 matches
    
    # Search for zip files first
    if include_zip:
        for path_str in zip_paths:
            path = Path(path_str)
            if path.exists() and path.is_file() and path.suffix.lower() == '.zip':
                zip_path = str(path.resolve())
                break
    
    # Search through all paths for cml.nc
    data_path = None
    for path_str in search_paths:
        path = Path(path_str)
        if path.exists() and path.is_file():
            data_path = str(path.resolve())
            break
    
    return data_path, zip_path


def load_openmrg_from_local(
    time_slice=None,
    data_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    use_pynncml_loader: bool = True
):
    """
    Load OpenMRG dataset from local files, with automatic file discovery.
    
    This function searches for OpenMRG data files and loads them. If PyNNcml
    is available, it uses PyNNcml's loader. Otherwise, uses basic NetCDF loading.
    
    Parameters
    ----------
    time_slice : slice or str, optional
        Time slice to filter data (e.g., slice("2015-06-01", "2015-06-10"))
    data_path : str, optional
        Direct path to cml.nc file. If None, searches automatically.
    metadata_path : str, optional
        Direct path to cml_metadata.csv. If None, searches automatically.
    use_pynncml_loader : bool
        If True and PyNNcml is available, use PyNNcml's load_open_mrg().
        If False, use basic NetCDF loading.
        
    Returns
    -------
    link_set : object
        PyNNcml LinkSet object (if PyNNcml available) or dict with data
    ps : object or None
        PyNNcml processing object or None
    metadata : dict or None
        Additional metadata
    """
    # Try to use PyNNcml's loader if available and requested
    if use_pynncml_loader and PYNNcML_AVAILABLE:
        try:
            import pynncml as pnc
            # If data_path is provided, PyNNcml might need it configured
            # Otherwise, let it download/search on its own
            if data_path is None:
                # Use PyNNcml's built-in loader (will download if needed)
                return pnc.datasets.load_open_mrg(time_slice=time_slice, change2min_max=True)
            else:
                # TODO: Configure PyNNcml to use local path
                # For now, fall back to manual loading
                pass
        except Exception as e:
            warnings.warn(f"PyNNcml loader failed: {e}. Falling back to manual loading.")
    
    # Manual loading using NetCDF
    if data_path is None:
        data_path = find_openmrg_data()
    
    if data_path is None:
        raise FileNotFoundError(
            "OpenMRG data file (cml.nc) not found. "
            "Please download it from: https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/ "
            "or provide the path using data_path parameter."
        )
    
    if not NETCDF4_AVAILABLE:
        raise ImportError("netCDF4 is required to load NetCDF files")
    
    # Load data using our wrapper function
    data, time = load_cml_netcdf(data_path, variable='rsl')
    
    # Load metadata if available
    metadata = None
    if metadata_path is None:
        # Try to find metadata file
        data_dir = Path(data_path).parent
        metadata_candidates = [
            data_dir / "cml_metadata.csv",
            data_dir.parent / "cml" / "cml_metadata.csv",
        ]
        for candidate in metadata_candidates:
            if candidate.exists():
                metadata_path = str(candidate)
                break
    
    if metadata_path and Path(metadata_path).exists():
        metadata = pd.read_csv(metadata_path)
    
    # Return basic data structure
    # Note: This is simplified - PyNNcml's LinkSet has more features
    return {
        'data': data,
        'time': time,
        'metadata': metadata,
        'data_path': data_path
    }, None, None


# ============================================================================
# Test/Example Section - Only runs if file is executed directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyNNcml Wrapper - Module Test")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing imports...")
    print(f"   ✓ numpy available: {np.__version__}")
    print(f"   ✓ pandas available: {pd.__version__}")
    
    # Test optional dependencies
    print("\n2. Testing optional dependencies...")
    print(f"   - PyNNcml: {'✓ Available' if PYNNcML_AVAILABLE else '✗ Not installed (optional)'}")
    print(f"   - netCDF4: {'✓ Available' if NETCDF4_AVAILABLE else '✗ Not installed (optional)'}")
    
    # Test basic functionality
    print("\n3. Testing basic functions...")
    
    # Test data preprocessing
    test_data = np.random.randn(100, 5)
    test_data[10:15, 0] = np.nan  # Add some NaN values
    processed = preprocess_cml_data(test_data, fill_nan=False)
    print(f"   ✓ preprocess_cml_data() works: {processed.shape}")
    
    # Test attenuation computation
    rsl_test = np.random.randn(100, 3) * 10 + 50  # Simulated RSL values
    attenuation = compute_attenuation(rsl_test, method='mean')
    print(f"   ✓ compute_attenuation() works: {attenuation.shape}")
    
    # Test rain estimation
    link_lengths = np.array([5.0, 10.0, 15.0])  # km
    frequencies = np.array([23.0, 38.0, 23.0])  # GHz
    rain = estimate_rain_from_attenuation(attenuation, link_lengths, frequencies)
    print(f"   ✓ estimate_rain_from_attenuation() works: {rain.shape}")
    
    # Test data finding
    print("\n4. Testing data file discovery...")
    data_path = find_openmrg_data()
    if data_path:
        print(f"   ✓ Found OpenMRG data: {data_path}")
    else:
        print("   ℹ OpenMRG data not found (this is OK if you haven't downloaded it)")
    
    print("\n" + "=" * 60)
    print("✓ All basic functions working!")
    print("=" * 60)
    print("\nNote: This is a module - import it in your code:")
    print("  from core.scientific_packages import pynncml_wrapper")
    print("\nOr use individual functions:")
    print("  from core.scientific_packages.pynncml_wrapper import load_openmrg_from_local")

