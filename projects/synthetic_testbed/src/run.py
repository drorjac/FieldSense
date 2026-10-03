"""
Command line for synthetic_testbed. Each study writes its tables to ``results/tables`` and
``figures`` redraws every figure from them.

    python projects/synthetic_testbed/src/run.py gallery              # generators, evolutions, fields (2 min)
    python projects/synthetic_testbed/src/run.py learn [--prior LK]   # learned motion (~25 min each)
    python projects/synthetic_testbed/src/run.py regional --n 24      # 64 km, every method (~1 h)
    python projects/synthetic_testbed/src/run.py regional --set lifecycle --n 12
    python projects/synthetic_testbed/src/run.py city --seeds 0:12 --workers 3   # street level (~1.5 h)
    python projects/synthetic_testbed/src/run.py real --network openmrg           # learned motion on real radar
    python projects/synthetic_testbed/src/run.py real --network openrainer --issue-every 4 --skip DARTS
    python projects/synthetic_testbed/src/run.py links --n 8          # what the links' errors cost
    python projects/synthetic_testbed/src/run.py figures
"""

from __future__ import annotations

import os

# PyTorch and pysteps both start OpenMP pools; on some macOS builds two pools in one process crash
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def gallery():
    from synthetic_testbed import gallery as g, figures as fg
    from synthetic_testbed.settings import TABLES, ensure_dirs
    ensure_dirs()
    fields, stats = g.fields()
    stats.to_csv(TABLES / "generators.csv", index=False)
    seqs, pred = g.evolutions()
    pred.to_csv(TABLES / "evolution_predictability.csv", index=False)
    g.predictability().to_csv(TABLES / "predictability.csv", index=False)
    fg.fig_gallery(fields, stats)
    fg.fig_evolution(seqs, pred)
    fg.fig_other_fields()
    print(stats.round(2).to_string(index=False))


def learn(n_sequences, epochs, prior):
    from synthetic_testbed import learned
    learned.train(n_sequences, epochs, retrain=True, prior=prior)


def regional(n, first_seed, scenario_set, quick):
    from synthetic_testbed import learned, regional as rg
    from synthetic_testbed.settings import TABLES, ensure_dirs
    ensure_dirs()
    out = TABLES / ("regional" if scenario_set == "mixed" else f"regional_{scenario_set}")
    out.mkdir(exist_ok=True)

    def save(maps, motion, now, scen):
        for name, df in (("maps", maps), ("motion", motion), ("nowcast", now), ("scenarios", scen)):
            df.to_csv(out / f"{name}.csv", index=False)

    res = rg.study(n, first_seed=first_seed, nets=learned.available(), steps_members=0 if quick else 8,
                   scenario_set=scenario_set, log=lambda m: print(m, flush=True), checkpoint=save)
    save(*res)


def _city_worker(seeds):
    import warnings as w
    w.filterwarnings("ignore")
    from synthetic_testbed import city
    city.study(seeds, log=lambda m: print(m, flush=True))


def city_study(seeds, workers):
    if workers <= 1:
        _city_worker(seeds)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_city_worker, [seeds[i::workers] for i in range(workers)]))


def real(network, issue_every, skip):
    from synthetic_testbed import learned
    from synthetic_testbed.settings import TABLES, ensure_dirs
    ensure_dirs()
    methods = tuple(m for m in ("LK", "VET", "DARTS", "proesmans") if m not in skip)
    d = learned.real_radar(network, issue_every=issue_every, methods=methods,
                           log=lambda m: print(m, flush=True))
    d.to_csv(TABLES / f"real_radar_{network}.csv", index=False)
    print(d[d.lead_min.isin([15, 30, 60])].pivot_table(index="method", columns="lead_min",
                                                        values="CSI_1").round(3))


def links(n, first_seed):
    from synthetic_testbed import links as lk
    from synthetic_testbed.settings import TABLES, ensure_dirs
    ensure_dirs()
    d = lk.study(n, first_seed, log=lambda m: print(m, flush=True))
    d.to_csv(TABLES / "link_counterfactual.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gallery")
    s = sub.add_parser("learn")
    s.add_argument("--sequences", type=int, default=700)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--prior", choices=["LK"], help="learn a correction to this motion field")
    s = sub.add_parser("regional")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--first-seed", type=int, default=100)
    s.add_argument("--set", default="mixed", choices=["mixed", "lifecycle"])
    s.add_argument("--quick", action="store_true", help="no STEPS ensembles")
    s = sub.add_parser("city")
    s.add_argument("--seeds", default="0:12", help="first:last (exclusive)")
    s.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("real")
    s.add_argument("--network", default="openmrg", choices=["openmrg", "openrainer"])
    s.add_argument("--issue-every", type=int, default=6, help="steps between issue times")
    s.add_argument("--skip", nargs="*", default=[], help="motion methods to leave out (e.g. DARTS)")
    s = sub.add_parser("links")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--first-seed", type=int, default=100)
    sub.add_parser("figures")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    if a.cmd == "gallery":
        gallery()
    elif a.cmd == "learn":
        learn(a.sequences, a.epochs, a.prior)
    elif a.cmd == "regional":
        regional(a.n, a.first_seed, a.set, a.quick)
    elif a.cmd == "city":
        lo, hi = (int(x) for x in a.seeds.split(":"))
        city_study(list(range(lo, hi)), a.workers)
    elif a.cmd == "real":
        real(a.network, a.issue_every, a.skip)
    elif a.cmd == "links":
        links(a.n, a.first_seed)
    else:
        from synthetic_testbed import figures as fg
        fg.all_from_tables()


if __name__ == "__main__":
    main()
