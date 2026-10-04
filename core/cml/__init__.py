"""Rain retrieval methods for commercial microwave links, one class per method.

The methods of the two student implementations unified in ``pcpn_maps`` (see
``projects/maps/nyc/docs/METHODS.md``), on a flat *link set*: ``rsl(link, time)``
at 1 min, optional ``tsl(link, time)``, and per-link ``frequency`` (GHz), ``length`` (km),
``polarization`` ("v"/"h"), site coordinates and ``mid_lat``/``mid_lon``.
``core.opensense.openmesh`` and ``core.opensense.networks`` build link sets.

``power_law``    ITU-R P.838 k-R power law, 2003 (PyNNcml) and 2005 (pycomlink) tables
``baseline``     dynamic / constant baselines and rolling-std wet/dry, in numpy
``preprocess``   total loss, gap filling, min/max aggregation
``estimators``   DynamicBaseline, ConstantBaselineSTD, ManualWindows, PycomlinkRSD,
                 NearbyLinks, PyNNcmlGRU - ``est.estimate(links)["rain"]`` in mm/h
``link_qc``      metadata, time-series and retrieval checks of a link set

``core.opensense.retrieval`` is FieldSense's own chain; these are kept exact to the
libraries and implementations they come from.
"""
