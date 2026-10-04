"""
Command line for link_weights.

    python projects/maps/link_weights/src/run.py      # -> results/error_model.csv, test_scores.csv, report.md
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from link_weights import study
    from link_weights.settings import RESULTS_DIR
    res = study.run(RESULTS_DIR)
    s = study.summary(res["scores"])
    s.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(res["model"].round(3).to_string())
    print(s.to_string())


if __name__ == "__main__":
    main()
