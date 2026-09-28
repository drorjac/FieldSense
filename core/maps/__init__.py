"""Rain maps on a lat/lon grid, and how to score them.

``idw``      inverse-distance weighting of link or point values to a ``core.geo.Grid``;
             rates to hour-ending accumulations
``scores``   NRMSE, bias, correlation, POD/FAR/CSI; map-vs-radar and link-vs-radar-path
"""
