"""
Command line for os_nowcasting.

    python projects/nowcasting/pysteps/src/run.py study --network openrainer     # nowcast every event
    python projects/nowcasting/pysteps/src/run.py study --network openmrg
    python projects/nowcasting/pysteps/src/run.py report                         # -> results/report.md
    python projects/nowcasting/pysteps/src/run.py study --network openmrg --event 20150827T01 \\
        --start "2015-08-27 06:00" --end "2015-08-27 09:00" --fresh          # a quick look

``study`` resumes from the last finished event; ``--fresh`` starts over.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study")
    s.add_argument("--network", required=True, choices=["openrainer", "openmrg"])
    s.add_argument("--event")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--fresh", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    warnings.filterwarnings("ignore")
    if a.cmd == "study":
        from os_nowcasting.study import run_network
        events = [(a.event, a.start, a.end)] if a.event else None
        st = run_network(a.network, events, fresh=a.fresh)
        print(f"{a.network}: {len(st.issues)} issue times in events {st.done}")
    elif a.cmd == "report":
        from os_nowcasting.report import write_report
        print(write_report())


if __name__ == "__main__":
    main()
