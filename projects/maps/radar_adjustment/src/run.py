"""
Command line for radar_adjustment.

    python projects/maps/radar_adjustment/src/run.py prepare             # inputs, as the OpenSense notebooks build them
    python projects/maps/radar_adjustment/src/run.py adjust              # the intercomparison (needs mergeplg's new API)
    python projects/maps/radar_adjustment/src/run.py extend              # weather stations, mapping, RADOLAN, variogram
    python projects/maps/radar_adjustment/src/run.py arpae               # ARPAE's gauge-adjusted radar at the same gauges
    python projects/maps/radar_adjustment/src/run.py report              # -> results/report.md and figures

``adjust`` and ``extend`` run in ``.venv-mergeplg-main`` (see README): the replication uses
the mergeplg commit the OpenSense repository pins, the extensions mergeplg main.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "adjust", "extend", "arpae", "report"])
    p.add_argument("--network", choices=["openmrg", "openrainer", "openmesh"], action="append")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if a.command == "prepare":
        from radar_adjustment.prepare import PREPARE
        for net in a.network or ["openmrg", "openrainer"]:
            print(PREPARE[net](force=a.force))
    elif a.command == "adjust":
        from radar_adjustment.adjust import run_intercomparison
        run_intercomparison(a.network or ["openmrg", "openrainer"], workers=a.workers)
    elif a.command == "extend":
        from radar_adjustment.extend import run_extensions
        run_extensions(a.network, workers=a.workers)
    elif a.command == "arpae":
        from radar_adjustment.arpae import score
        print(score()["overall"].to_string())
    elif a.command == "report":
        from radar_adjustment.report import write_report
        print(write_report())


if __name__ == "__main__":
    main()
