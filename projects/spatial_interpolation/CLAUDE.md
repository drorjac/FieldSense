# projects/spatial_interpolation/ — working rules

## Interaction style
- **Talk only by default.** Do not edit any file in this directory unless the user gives an explicit command to edit (e.g. "edit X", "rewrite the abstract", "add this paragraph to section Y").
- Reading, grepping, running the notebook, and quoting numbers are all fine without permission.
- Do not propose changes by writing them into files. Propose in chat, wait for the go-ahead, then edit.
- Do not jump to "next steps" or assume the next action. After answering, stop and wait.

## The three files
- `notebooks/advanced_models_colab_v2.ipynb` — main working notebook. **Source of truth for all numbers, plots, and the actual experiment that was run.** Do not edit.
- `paper/full_report.tex` — long-form report. Describes an **older/different experiment scope** (IDW vs GMZ map inputs, pySTEPS baseline, full-period training). Its result tables do **not** match the notebook. Treat it as background prose / methodology reference, not as a source of numbers.
- `paper/paper.tex` — the 5-page IEEE conference paper being written. This is the target artifact. It matches what the notebook actually tested: Transformer / POD-SINDy / Mamba-style SSM, single-step vs multi-horizon, self-supervised (CML-target) vs gauge-supervised. Edits only on explicit request, section by section.

## Paper target
- Venue: IEEE conference (IEEEtran, `conference` option).
- **Hard limit: 5 pages.** Every addition needs to be justified against the page budget.
- Authors: Ben Yehoshua S., Jacoby D., Salganik Y. (Tel Aviv University) — per `paper/full_report.tex`.

## Numbers
- Pull results from notebook cells 72 (point-to-pixel vs gauges), 74 (grid-to-grid vs radar), 76 (bar charts), 78 (peak-event map grid at 2015-08-25 06:45), 80 (best-per-(target, horizon, metric) summary).
- Do not invent numbers and do not lift numbers from `paper/full_report.tex` — they describe a different experiment.

## Editing protocol
- One section at a time, in the order the user asks for.
- Show the proposed change in chat first when it is short; for longer rewrites, ask whether to show inline or write directly.
- Preserve existing `\tbd{}` markers until the user confirms the replacement.
