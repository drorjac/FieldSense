"""
Command line for multisensor_maps.

    python projects/multisensor_maps/src/run.py events                  # the study events per network
    python projects/multisensor_maps/src/run.py event --network openrainer --start "2021-09-26 06:00" --end "2021-09-26 19:00"
    python projects/multisensor_maps/src/run.py study                   # -> results/report.md
    python projects/multisensor_maps/src/run.py merge                   # -> results/merging/report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))



def rnn_factory():
    """The RNN of projects/cml_rnn as an extra retrieval, if a trained model is there."""
    from multisensor_maps.settings import RNN_MODEL
    if not (RNN_MODEL / "model.pt").exists():
        return None
    from core.cml.rnn import HourlyRNN
    return lambda network: {"rnn": HourlyRNN(RNN_MODEL, network).estimate}


def study(extra_factory=None) -> Path:
    from core.opensense.networks import NETWORKS
    from multisensor_maps import plots
    from multisensor_maps.report import write_report
    from multisensor_maps.study import run_study

    extra_factory = extra_factory or rnn_factory()

    s = run_study(extra_factory=extra_factory)
    out = s.save()
    figs = {"pooled": "figures/pooled_scores.png", "events": {}}
    plots.pooled_figure(s.pooled_maps, s.pooled_points, out / figs["pooled"])
    for n, res in s.examples.items():
        figs["events"][n] = f"figures/largest_event_{n}.png"
        plots.event_totals_figure(res, out / figs["events"][n])
    pts = {k: NETWORKS[k].points(*_first_day(s, k)) for k in NETWORKS}
    plots.networks_figure(pts, out / "figures/networks.png")
    write_report(s, figs, out)
    return out / "report.md"


def merge(networks=("openmrg", "openrainer", "openmesh"), refresh: bool = False) -> Path:
    """Links, gauges and radar merged every way, scored at held-out gauges."""
    from multisensor_maps.merging import run_merging
    from multisensor_maps.merging_report import write_report

    s = run_merging(networks, extra_factory=rnn_factory(), refresh=refresh)
    s.save()
    return write_report(s)


def _first_day(s, key):
    ev = s.events[s.events.network == key].iloc[0]
    return ev.start, ev.end


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("events")
    p = sub.add_parser("event")
    p.add_argument("--network", required=True, choices=["openmrg", "openrainer", "openmesh"])
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    sub.add_parser("study")
    p = sub.add_parser("merge")
    p.add_argument("--network", action="append", choices=["openmrg", "openrainer", "openmesh"])
    p.add_argument("--refresh", action="store_true", help="rescore every event (inputs stay cached)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.cmd == "events":
        from multisensor_maps.study import network_events
        for n in ("openmrg", "openrainer", "openmesh"):
            print(network_events(n)[["network", "start", "end", "total_mm", "peak_hourly_mm"]].to_string())
    elif args.cmd == "event":
        from multisensor_maps.event import run_event
        res = run_event(args.network, args.start, args.end)
        print(res.pairwise()[["estimate", "reference", "rel_bias", "nrmse", "corr"]].round(2).to_string())
        print(res.point_check()[["estimate", "stations", "rel_bias", "nrmse", "corr"]].round(2).to_string())
    elif args.cmd == "study":
        print(study())
    elif args.cmd == "merge":
        for name in ("pycomlink", "core.cml", "multisensor_maps.event"):
            logging.getLogger(name).setLevel(logging.WARNING)
        print(merge(tuple(args.network or ("openmrg", "openrainer", "openmesh")), args.refresh))


if __name__ == "__main__":
    main()
