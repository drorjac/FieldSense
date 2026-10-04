"""Command line for multisensor_nowcasting.

    python projects/nowcasting/multisensor/src/run.py build --network openmrg
    python projects/nowcasting/multisensor/src/run.py physics --network openmrg
    python projects/nowcasting/multisensor/src/run.py train --network openmrg
    python projects/nowcasting/multisensor/src/run.py evaluate --network openmrg
    python projects/nowcasting/multisensor/src/run.py report
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # torch + pysteps share a process on macOS

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "physics", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--network", choices=("openmrg", "openmesh"), required=True)
        p.add_argument("--refresh", action="store_true", help="recompute instead of reading the cache")
        if name in ("physics",):
            p.add_argument("--workers", type=int, default=7)
        if name == "train":
            p.add_argument("--only", nargs="*", default=None, help="model names to train (default: all)")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "build":
        from multisensor_nowcasting import cube
        cube.build(a.network, refresh=a.refresh)
    elif a.cmd == "physics":
        from multisensor_nowcasting import physics
        physics.run(a.network, workers=a.workers, refresh=a.refresh)
    elif a.cmd == "train":
        from multisensor_nowcasting import learn
        learn.train_all(a.network, only=a.only, refresh=a.refresh)
    elif a.cmd == "evaluate":
        from multisensor_nowcasting import evaluate
        evaluate.run(a.network, refresh=a.refresh)
    elif a.cmd == "report":
        from multisensor_nowcasting import report
        print(report.write())


if __name__ == "__main__":
    main()
