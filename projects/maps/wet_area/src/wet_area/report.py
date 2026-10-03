"""results/report.md from results/synthetic.csv and results/real.csv."""

from __future__ import annotations

import pandas as pd

from .settings import RESULTS_DIR
from .study import COLS, summary

ORDER = ["idw", "masked p0.3", "masked p0.5", "masked p0.7",
         "conditional p0.3", "conditional p0.5", "conditional p0.7"]


def _table(df: pd.DataFrame) -> str:
    df = df.copy()
    df["map"] = pd.Categorical(df["map"], ORDER, ordered=True)
    df = df.sort_values([c for c in df.columns if c not in COLS and c != "map"] + ["map"])
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write() -> None:
    syn = pd.read_csv(RESULTS_DIR / "synthetic.csv")
    real = pd.read_csv(RESULTS_DIR / "real.csv")
    s_syn = summary(syn, [])
    s_real = summary(real, ["reference", "network", "retrieval"])
    s_syn.to_csv(RESULTS_DIR / "summary_synthetic.csv", index=False)
    s_real.to_csv(RESULTS_DIR / "summary_real.csv", index=False)
    L = ["# wet_area: results", "",
         "Computed by `python projects/maps/wet_area/src/run.py report` from `synthetic.csv` and "
         "`real.csv`. Every map is of hourly totals from links alone; wet = 0.1 mm. Medians over "
         f"{syn.case.nunique()} synthetic cases and {real.event.nunique()} real events.",
         "", "`war_ratio`: the map's wet fraction over the reference's (1 is right). "
         "`peak_ratio`: 99th percentile of the map over the reference's.", "",
         "## Against the true field (synthetic)", "", _table(s_syn), "",
         "## Against the radar and held-out gauges (real events)", "", _table(s_real), ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(L))
