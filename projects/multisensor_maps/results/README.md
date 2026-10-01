# multisensor_maps results

Everything here is written by `src/run.py`; nothing is edited by hand. The 10 largest storms of
each network (`events.csv`), hour-ending accumulations throughout.

## Maps (`run.py study`)

| file | what it holds |
|---|---|
| `report.md` | the study's report: findings, retrieval x interpolation tables, per-network scores |
| `events.csv` | the storms: network, start, end, domain-mean total, peak hour |
| `pooled_map_scores.csv` | every map against the radar and each gauge map, pooled per network (cells <= 2 km of a link, and all cells) |
| `pooled_point_scores.csv` | at the check gauges: radar, every map, and the gauge map rebuilt without the gauge |
| `event_scores.csv` | the same map comparisons per storm |
| `link_scores.csv` | each retrieval's hourly link totals against the radar along the path and gauges near it |
| `errors.json` | what could not be scored, and why |
| `figures/networks.png` | the three networks: links, gauges, radar grid |
| `figures/pooled_scores.png` | pooled NRMSE of every map, against the radar and at the gauges |
| `figures/largest_event_<network>.png` | event totals of every map on the largest storm |

## Merging (`run.py merge`, in `merging/`)

| file | what it holds |
|---|---|
| `merging/report.md` | the merging report: best product of each sensor combination, method x source tables, retrieval after merging, variogram sensitivity, rankings |
| `merging/ranking_<network>_heldout_gauges.csv` | every product (radar, links, gauges, links + gauges, radar adjusted by 7 methods with each source), pooled at held-out check gauges, best first; with family (inputs), method, package, median per-storm NRMSE and storms won |
| `merging/ranking_<network>_heldout_gauges_le5_km_of_a_link.csv` | the same on check gauges within 5 km of a link |
| `merging/ranking_<network>_independent_gauges.csv` | at gauges no product uses (Gothenburg SMHI, New York ASOS), every gauge in |
| `merging/scores_per_event.csv` | every product's scores per storm at the held-out gauges |
| `merging/variograms.json` | the spherical variogram fitted to each network's radar for the kriging methods |
| `merging/events.csv`, `merging/errors.json` | the storms; what was left out and why |
| `merging/figures/families.png` | best product of each sensor combination, per network |
| `merging/figures/families_near_links.png` | the same at gauges within 5 km of a link |
| `merging/figures/methods.png` | NRMSE of each adjustment method with gauges, links, and both |
| `merging/figures/largest_event_<network>.png` | event totals of the radar and the best product of each combination |

Product names in the merging files: `radar`, `radar only` (MRMS without gauge correction, New
York); `links <retrieval> idw|line`; `gauges`; `links <retrieval> + gauges`; and
`<radar> <method> [<source>]` for the radar adjusted by `mfb`, `add`, `mul` (`pcpn_maps`) or
`idw_add`, `idw_mul`, `okrig_add`, `ked` (`mergeplg`) with `gauges`, `<retrieval>` links, or
`<retrieval> + gauges`. A ` (5 km variogram)` suffix marks the kriging products rebuilt with
`mergeplg`'s default variogram.
