"""
Command line for cml_rnn.

    python projects/retrieval/rnn_three_networks/src/run.py build --network openrainer     # hourly dataset -> DATA_DIR
    python projects/retrieval/rnn_three_networks/src/run.py train --name all                 # all three networks
    python projects/retrieval/rnn_three_networks/src/run.py train --name all_lstm --set rnn_type=LSTM epochs=80
    python projects/retrieval/rnn_three_networks/src/run.py evaluate --name all            # -> results/all/
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--network", nargs="+", default=["openmrg", "openrainer", "openmesh"])
    b.add_argument("--overwrite", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--networks", nargs="+", default=["openmrg", "openrainer", "openmesh"])
    t.add_argument("--name", required=True)
    t.add_argument("--set", nargs="*", default=[], help="Config overrides, key=value")
    e = sub.add_parser("evaluate")
    e.add_argument("--name", required=True)
    e.add_argument("--networks", nargs="+", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if args.cmd == "build":
        from cml_rnn.build import build
        for n in args.network:
            ds = build(n, overwrite=args.overwrite)
            print(n, dict(ds.sizes))
    elif args.cmd == "train":
        from cml_rnn.train import Config, train
        over = {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            typ = type(getattr(Config(), k))
            over[k] = (v.lower() in ("1", "true", "yes")) if typ is bool else typ(v)
        train(args.networks, Config(**over), args.name)
    elif args.cmd == "evaluate":
        evaluate(args.name, args.networks)


def evaluate(name: str, networks=None):
    import json
    from cml_rnn import report
    from cml_rnn.evaluate import estimates, score_network, verdict
    from cml_rnn.settings import DATA_DIR, RESULTS_DIR
    import pandas as pd
    networks = networks or json.loads((DATA_DIR / "models" / name / "config.json").read_text())["networks"]
    tables = []
    for n in networks:
        d, est = estimates(name, n)
        tables.append(score_network(d, est))
        report.scatter_figure(d, est, RESULTS_DIR / name / "figures" / f"scatter_{n}.png")
        report.series_figure(d, est, RESULTS_DIR / name / "figures" / f"series_{n}.png")
    table = pd.concat(tables, ignore_index=True)
    report.scores_figure(table, RESULTS_DIR / name / "figures" / "scores_target.png")
    refs = ["target", "radar"] + sorted(set(table.reference) - {"target", "radar"})
    verdicts = {r: verdict(table, r) for r in refs}
    out = report.write(name, table, verdicts)
    v = verdicts["target"]
    print(v.round(3).to_string())
    print("RNN beats every power-law method:", v.groupby("network").rnn_better.all().to_dict())
    print(out / "report.md")


if __name__ == "__main__":
    main()
