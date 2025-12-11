"""
Scientific Packages Module

This module contains implementations and wrappers for scientific computing packages
used in the FieldSense project, including:
- SINDy (Sparse Identification of Nonlinear Dynamics)
- PySR (PySymbolic Regression)
- PyNNcml (Neural Network-based CML Rain Estimation)
- Other physics-informed machine learning tools
"""

__version__ = "0.1.0"

# Import wrappers if available
__all__ = []

try:
    from . import pynncml_wrapper
    __all__.append('pynncml_wrapper')
except ImportError:
    pass

try:
    from . import sindy_wrapper
    __all__.append('sindy_wrapper')
except ImportError:
    pass

