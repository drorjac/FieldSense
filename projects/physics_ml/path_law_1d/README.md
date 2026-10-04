# path_law_1d: the laws of microwave-link attenuation (proposed project: starter)

A starting point for the B.Sc. proposal
[*Discovering the Laws of Microwave-Link Attenuation*](../../../docs/pre_projects/1d_project.md).
After removing the baseline, a link measures

```
A(t) = integral over the path of a R(x, t)^b dx  +  delta(t)  +  n(t)
```

and the standard retrieval assumes uniform rain and `delta = 0`, so `A = a R^b L`. The
proposal learns readable equations for both parts: the **path law** `f(R, L)` with
symbolic regression (PySR), and the **non-rain term** `delta` - the wet antenna, which
builds up and dries - as `d delta / dt = F(delta, R)` with weak-form SINDy.

**This folder does not do that research.** Following
[`docs/pre_projects/README.md`](../../../docs/pre_projects/README.md), it hands the
project a working base: the data load, the chain runs, the baseline is scored the way
the proposal will be judged, the problem exists in simulation with a known answer, and
each notebook stops at a "your method goes here" cell.

## What FieldSense provides

| the proposal needs | where it is |
|---|---|
| OpenMRG links, radar and gauges | `core.opensense.example_data` (the 8-day subset used here), `core.opensense.networks` (the full record) |
| The processing chain: wet/dry, baseline, wet antenna, power law | `core.opensense.retrieval`, `core.cml.power_law` |
| Radar along each link, nearest gauge | `core.opensense.evaluation.radar_along_links`, `closest_gauges`, `gauge_series_at_links` |
| Storm events, split by event | `core.events.detect_events`, wrapped in `src/path_law_1d/events.py` |
| Simulated rain, links of any length and frequency, the exact path integral | `core.simulation` (`generators`, `spacetime`, `cml_network.forward_model`, `forward_series`) |
| A wet antenna with memory, and the literature models | `core.simulation.wet_antenna` (`DynamicWetAntenna`; constant, Schleiss 2013, Pastorek 2021) |
| Scores by link length; drying tails; literature models fitted per link | `src/path_law_1d/` (`scoring`, `tails`, `simulate`), tested in `tests/` |
| SINDy and PySR set-up | [`../discovery`](../discovery/) (`01_sindy_basics`, `02_pysr_basics`) |

## Notebooks

| notebook | what it hands over | runtime |
|---|---|---|
| [`01_simulated_path_law.ipynb`](notebooks/01_simulated_path_law.ipynb) | 800 simulated links (0.3-20 km, 10-80 GHz) over six rain regimes; exact integral vs `a R^b L`; the gap by length and regime; the tidy table `(R_bar, L, f, regime, A_exact, A_linear)` a PySR run would learn from, and `score_path_law` | ~10 s |
| [`02_real_links_and_radar.ipynb`](notebooks/02_real_links_and_radar.ipynb) | OpenMRG, 364 links, 8 days; radar along each path and the nearest gauge; seven storm events split train/test; the linear law (no correction, and with the Pastorek correction) scored by link length on the test events | ~1 min |
| [`03_wet_antenna_tails.ipynb`](notebooks/03_wet_antenna_tails.ipynb) | the dynamic wet antenna in simulation, then OpenMRG: `delta_hat = A - a R^b L` with radar as `R`; post-event drying tails and their fitted drying time; the literature models fitted per link and scored in rain, in the tail and when dry | ~1.5 min |

Runtimes on a laptop CPU, with the example subset already downloaded. Run from the
repository root with FieldSense installed (`pip install -e ".[opensense,notebooks]"`).

## The baselines, as the notebooks compute them

**The path law in simulation** (notebook 01). The linear law gets the rain within 5 % for
99 % of (field, link) pairs below 1 km, 74 % at 4-8 km and 52 % at 12-20 km; it holds for
at least 90 % of cases only up to 2 km. The error grows with how cellular the rain is:
at 12-20 km the median absolute rain error is 1 % in stratiform rain and 9 % in convective
cells. Its sign follows `b - 1` (Jensen's inequality).

**The linear law on OpenMRG** (notebook 02, test events, 15-minute means, against the
radar along the path):

| link length | 0-1 km | 1-2 km | 2-4 km | 4-8 km | 8-20 km |
|---|---|---|---|---|---|
| normalized bias, no correction | +1.64 | +0.55 | +0.06 | -0.06 | +0.16 |
| NRMSE, no correction | 6.9 | 4.0 | 2.6 | 2.1 | 2.5 |
| normalized bias, Pastorek 2021 | -0.38 | -0.45 | -0.46 | -0.50 | -0.58 |
| NRMSE, Pastorek 2021 | 2.6 | 2.5 | 2.1 | 1.9 | 1.9 |

Without a correction the links under 1 km read 2.6 times the radar's rain - a term that
does not grow with length; the fixed literature correction removes it on short links and then
under-reads every length. The gauge columns are in the notebook.

**The non-rain term** (notebook 03). In simulation the drying time read off the tails is
28 min against the model's 30. On OpenMRG, 3,116 tails on 354 links start above 0.5 dB,
at about 1.1 dB whatever the link length, with median fitted drying times of 9-11 min by
length bin (and some tails that last hours). The literature models, fitted per link, predict nothing in the
tail: the constant and Schleiss models score exactly as no model there (RMSE 1.06 dB on
the test events), Pastorek barely better (1.01 dB).

## What is yours to do

- **Task A.** Learn `f(R, L)` from the table of notebook 01 (fit on fields 0-3, score on
  4-5 with `score_path_law`); find the range of lengths where the linear law holds; apply
  the learned law to OpenMRG with the radar along the path as reference (notebook 02).
- **Task B.** Learn `d delta / dt = F(delta, R)` per link with weak-form SINDy: first on
  the simulated `delta_hat` of notebook 03, where the answer is `DynamicWetAntenna`
  (`tau_wet` 5 min, `tau_dry` 30 min), then on OpenMRG. Compare its drying time with the
  tails, and decide which links have a wet antenna.
- **The rain estimate.** Invert the combined model on the test events and score it with
  `scoring.scores_by_length` against radar and gauges: the four arms of the proposal.
- Choices left open on purpose: the wet/dry threshold, the radar as `R` (it is a 2 km
  volume aloft, not the rain at the antenna), the 6-hour baseline hold in notebook 03,
  and OpenRainER as a second network (`core.opensense.networks.OpenRainER`).

## Layout

```
path_law_1d/
├── notebooks/       01_simulated_path_law, 02_real_links_and_radar, 03_wet_antenna_tails
├── src/path_law_1d/ simulate.py, events.py, scoring.py, tails.py
└── tests/           test_path_law_1d.py
```

The notebooks write one generated file, `dataset/open_datasets/path_law_1d/simulated_path_law.csv`.

## References

1. Berne, A., and Uijlenhoet, R. (2007). Path-averaged rainfall estimation using microwave links:
   Uncertainty due to spatial rainfall variability. *Geophys. Res. Lett.*, 34, L07403.
2. Schleiss, M., Rieckermann, J., and Berne, A. (2013). Quantification and modeling of wet-antenna
   attenuation for commercial microwave links. *IEEE Geosci. Remote Sens. Lett.*, 10(5), 1195-1199.
3. Pastorek, J., Fencl, M., Rieckermann, J., and Bares, V. (2021). Precipitation estimates from
   commercial microwave links: Practical approaches to wet-antenna correction. *IEEE Trans. Geosci.
   Remote Sens.*, 60, 1-9.
4. Andersson, J. C. M., et al. (2022). OpenMRG: Open data from Microwave links, Radar, and Gauges for
   rainfall quantification in Gothenburg, Sweden. *Earth Syst. Sci. Data*, 14, 5411-5426.
5. Messenger, D. A., and Bortz, D. M. (2021). Weak SINDy: Galerkin-based data-driven model selection.
   *Multiscale Model. Simul.*, 19(3), 1474-1497.
6. Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl.
   arXiv:2305.01582.
