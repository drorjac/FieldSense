"""
Command line for wet_area.

    python projects/maps/wet_area/src/run.py synthetic      # -> results/synthetic.csv
    python projects/maps/wet_area/src/run.py real           # -> results/real.csv
    python projects/maps/wet_area/src/run.py report         # -> results/report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["synthetic", "real", "report"])
    ap.add_argument("--cases", type=int, default=None, help="synthetic cases (default: settings)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from wet_area import report, study
    from wet_area.settings import N_SYNTHETIC, RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if a.step == "synthetic":
        study.synthetic(a.cases or N_SYNTHETIC).to_csv(RESULTS_DIR / "synthetic.csv", index=False)
    elif a.step == "real":
        study.real().to_csv(RESULTS_DIR / "real.csv", index=False)
    else:
        report.write()


if __name__ == "__main__":
    main()
