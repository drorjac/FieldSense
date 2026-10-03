"""Verification, pooled over issue times with pysteps' own accumulators.

The session's scores (blocks 04 and 04a), each accumulated over every issue time of a
study and computed once at the end, so a score is a pooled one (one contingency table,
one set of sums), not an average of per-forecast scores:

* categorical, per threshold (0.5, 1, 5 mm/h): POD, FAR, CSI, ETS, frequency bias
  (``pysteps.verification.detcatscores``);
* continuous: ME, MAE, RMSE, Pearson correlation (``detcontscores``);
* spatial: fractions skill score at 2-50 km for 1 mm/h (``spatialscores.fss``), and
  SAL (Wernli et al. 2008) - :func:`core.nowcast.scores.sal_score`, on ``scipy.ndimage`` with pysteps' object
  definition (threshold 1/15 of the 95th percentile of wet pixels), because pysteps' SAL
  needs scikit-image;
* ensemble: CRPS, ROC area and reliability for 1 mm/h, rank histogram
  (``probscores``, ``ensscores``).

Only cells where both fields are finite count; categorical and continuous scores take the
valid cells as flat arrays (pysteps counts a NaN pair as a correct negative otherwise).

The accumulators are :mod:`core.nowcast.verify`, with this project's thresholds as defaults.
"""

from __future__ import annotations

from core.nowcast.verify import (CAT_SCORES, CONT_SCORES, Collector, Deterministic,  # noqa: F401
                                 Ensemble, at_points)
from core.nowcast.scores import sal_score  # noqa: F401
