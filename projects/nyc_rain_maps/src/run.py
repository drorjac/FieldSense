"""
Command line for the nyc_rain_maps project. Run from the repository root or anywhere:

    python projects/nyc_rain_maps/src/run.py select       # gauge-calibrated link selection -> results/link_selection
    python projects/nyc_rain_maps/src/run.py event   --start "2024-01-09 16:00" --end "2024-01-10 11:00"
    python projects/nyc_rain_maps/src/run.py compare --start "2024-01-09 16:00" --end "2024-01-10 11:00"
    python projects/nyc_rain_maps/src/run.py study        # 10 catalog events x methods x link sets -> results/study
    python projects/nyc_rain_maps/src/run.py all-events   # all 52 events of the record -> results/all_events
    python projects/nyc_rain_maps/src/run.py validate     # radar audit; every source vs ASOS -> results/validation
    python projects/nyc_rain_maps/src/run.py events       # rebuild the event catalog (hours of MRMS downloads)

``event`` and ``compare`` write NetCDF, CSV and figures under
``dataset/open_datasets/OpenMesh_NYC/nyc_rain_maps/outputs/`` unless ``-o`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nyc_rain_maps import settings  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select", help="gauge-calibrated link selection")
    for name in ("event", "compare"):
        p = sub.add_parser(name, help="one event: CML methods vs MRMS" if name == "event"
                           else "one event: CML, PWS, MRMS and ASOS on one grid")
        p.add_argument("--start", required=True)
        p.add_argument("--end", required=True)
        p.add_argument("--link-set", default="selected" if name == "compare" else "shared8",
                       help="shared8 | selected | qc | all")
        p.add_argument("-o", "--out")
    sub.add_parser("study", help="the catalog events x methods x link sets")
    sub.add_parser("all-events", help="every event of the OpenMesh record")
    sub.add_parser("validate", help="radar audit and every source vs ASOS")
    sub.add_parser("events", help="rebuild the event catalog from MRMS and ASOS")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out_root = settings.DATA_DIR / "outputs"

    if args.cmd == "select":
        from nyc_rain_maps.link_selection import default_selection_dir, select_links
        res = select_links()
        print(res.save(default_selection_dir()).parent / "README.md")
    elif args.cmd == "event":
        from nyc_rain_maps.pipeline import run_event
        res = run_event(args.start, args.end, link_set=args.link_set)
        out = Path(args.out or out_root / f"event_{args.start[:10]}")
        res.save(out)
        print(res.map_scores[["nrmse", "rel_bias", "corr", "csi"]].round(3))
        print(out)
    elif args.cmd == "compare":
        from nyc_rain_maps.compare import compare_sensors
        comp = compare_sensors(args.start, args.end, link_set=args.link_set)
        out = Path(args.out or out_root / f"sensors_{args.start[:10]}")
        comp.save(out)
        print(comp.pairwise(near_km=2.0)[["estimate", "reference", "rel_bias", "nrmse", "corr"]].round(3))
        print(out)
    elif args.cmd == "study":
        from nyc_rain_maps.study import StudyConfig, run_study
        study = run_study(StudyConfig())
        print(study.write_report(settings.RESULTS_DIR / "study") / "report.md")
    elif args.cmd == "all-events":
        from nyc_rain_maps.all_events import run_all_events, write_report
        study, cat = run_all_events()
        print(write_report(study, cat) / "README.md")
    elif args.cmd == "validate":
        from nyc_rain_maps.validation import run_validation
        print(run_validation().write_report(settings.RESULTS_DIR / "validation") / "README.md")
    elif args.cmd == "events":
        from nyc_rain_maps.events import build_catalog, write_catalog
        print(write_catalog(build_catalog()))


if __name__ == "__main__":
    main()
