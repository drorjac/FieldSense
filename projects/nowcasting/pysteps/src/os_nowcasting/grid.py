"""The nowcasting grid and the bridge to pysteps: :mod:`core.nowcast.grid`.

    grid = square_grid(domain, 2.0)
    precip, meta = to_pysteps(field_mm, step_min=15)      # (time, y, x) mm/h + metadata
    dbr, meta_db = to_dbr(precip, meta)
"""

from __future__ import annotations

from core.nowcast.grid import (KM_PER_DEG, from_dbr, metadata, pixel_km,  # noqa: F401
                               square_grid, to_dbr, to_pysteps)
