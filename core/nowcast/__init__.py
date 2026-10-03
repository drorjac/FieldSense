"""Nowcasting shared by ``projects/os_nowcasting``, ``projects/multisensor_nowcasting`` and
the tutorials.

``grid``       square-pixel grids, xarray fields -> pysteps arrays and metadata, dB transform
``methods``    motion (LK, VET, DARTS, Proesmans), extrapolation, S-PROG, ANVIL, LINDA, STEPS,
               the reachable-cell mask
``verify``     pooled deterministic and ensemble scores on pysteps' accumulators
``advection``  advection-based temporal interpolation of rain fields (accumulations)
``scores``     verification scores pysteps does not provide (SAL on scipy.ndimage)
``learned_motion``  a U-Net motion field trained on simulations with a known true flow
"""
