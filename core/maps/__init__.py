"""Rain maps on a lat/lon grid, and how to score them.

``idw``      inverse-distance weighting of link or point values to a ``core.geo.Grid``;
             rates to hour-ending accumulations
``merge``    links and gauges in one IDW map; radar adjusted to them (mean-field bias,
             additive, multiplicative)
``mergeplg_methods``  mergeplg's radar merging (difference IDW / kriging, KED) at any cells
``geometry`` a field along each link path; distance to the nearest path
``scores``   NRMSE, bias, correlation, POD/FAR/CSI; map-vs-radar and link-vs-radar-path
"""
