"""
Command line for maps/netherlands.

    python projects/maps/netherlands/src/run.py availability  # rows per month, 2011-2015 (~6 min)
    python projects/maps/netherlands/src/run.py convert    # zip -> monthly netCDF, May-Aug 2012 (~11 min)
    python projects/maps/netherlands/src/run.py retrieve   # RAINLINK on every sub-link (~10 min)
    python projects/maps/netherlands/src/run.py score      # hourly series, maps, -> results/*.csv
    python projects/maps/netherlands/src/run.py figures    # -> results/figures/
    python projects/maps/netherlands/src/run.py report     # -> results/report.md
    python projects/maps/netherlands/src/run.py all

Needs ``~/data/cml/netherlands/_download/IDRawCMLdata.zip``
(``python -m core.opensense.fetch --dataset netherlands``, 9.5 GB) for ``convert``; the
KNMI gauges are fetched on first use.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = ["availability", "convert", "retrieve", "score", "figures", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import matplotlib
    matplotlib.use("Agg")
    from core.opensense import netherlands as nl
    from netherlands import figures, report, study
    from netherlands.settings import END, SPINUP, START
    for step in STEPS if a.step == "all" else [a.step]:
        if step == "availability":
            from netherlands.settings import RESULTS_DIR
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            study.availability().to_csv(RESULTS_DIR / "availability_monthly.csv", index=False)
        elif step == "convert":
            nl.convert_months(START - SPINUP, END, overwrite=a.force)
        elif step == "retrieve":
            study.retrieve(force=a.force)
        elif step == "score":
            study.hourly(force=a.force)
            study.maps_at_gauges(force=a.force)
            study.run_all()
        elif step == "figures":
            figures.storm_day(force=a.force)
            figures.write()
        else:
            report.write()


if __name__ == "__main__":
    main()
