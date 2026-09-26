# Rainfall Field Simulation & CML Sensor Modelling

Simulates three rainfall regimes over a region, measures how differently the
water is distributed in each, then puts a commercial microwave link (CML)
network over them and reconstructs the field from what the links report.

The question it answers: **when you build a rainfall field from CML
measurements, what actually limits you — the sensors, or where the sensors
happen to be?**

## Quick start

```bash
pip install -r requirements.txt
pip install -r projects/rainfall_field_sim/requirements.txt

python projects/rainfall_field_sim/src/run_demo.py
```

Runs in about 2 min and writes seven figures plus summary tables to `results/`.
`--no-sweep` skips the network-density sweep. `--links`,
`--grid` and `--dx` change the network size and domain resolution.

## The three rainfall models

All three live on a 32 x 32 km periodic domain at 250 m resolution, and all
three are tuned to comparable *mean* rainfall so that the differences below are
about structure, not amount.

| | Stratiform | Convective cells | Frontal band |
|---|---|---|---|
| construction | spectral GRF, `P(k) ~ k^-3.2`, threshold-and-power transform | 12 compact anisotropic cells + weak GRF background | analytic cross-band profile, GRF along-band modulation |
| wet area | 86% | 20% | 52% |
| domain mean | 1.89 mm/h | 2.47 mm/h | 3.99 mm/h |
| mean where wet | 2.20 mm/h | 12.61 mm/h | 7.74 mm/h |
| peak | 8 mm/h | 97 mm/h | 77 mm/h |
| decorrelation length | 5.9 km | 1.9 km | 2.8 km |
| water in wettest 5% of area | 15% | 69% | 51% |

That last row is the one that matters downstream. Stratiform rain spreads its
water evenly; convective rain puts nearly 70% of it into a twentieth of the
map. A sparse network samples those two situations very differently.

## Moving fields

`core/simulation/moving_fields.py` puts the three regimes in motion, each at
its own `advection_kmh`, either frozen (pure translation) or evolving: each
step the pattern is mixed with a fresh realization of the same regime, with
an e-folding time `tau`, and mapped back onto the original values, so wet
area and intensities stay fixed while the cells reshape. Stratiform and
convective fields are periodic and shift exactly; the frontal band is not (a
straight band at 35 degrees cannot tile the domain), so it is generated on a
padded domain and cropped, and nothing wraps into view.

`fig7` and the demo's last table ask how much of the field the true motion
alone predicts - the ceiling for any advection-based nowcast:

| correlation with the field moved at the true velocity, +30 / +60 min | frozen | tau 120 min | tau 60 min |
|---|---|---|---|
| Stratiform | 1.00 / 1.00 | 0.74 / 0.53 | 0.59 / 0.37 |
| Convective cells | 1.00 / 1.00 | 0.60 / 0.37 | 0.36 / 0.18 |
| Frontal band | 1.00 / 1.00 | 0.98 / 0.96 | 0.97 / 0.95 |

At the same `tau`, convective rain loses its predictability twice as fast as
stratiform: its water sits in a few small cells, and when a cell is replaced
the correlation goes with it. The front stays predictable because its
geometry is fixed and only its texture evolves. `spatial_interpolation`
uses the same sequences to test its forecasters against a known future.

## The CML sensor chain

Links are drawn as a plausible backhaul topology: towers scattered over the
domain, each connected to its nearest few neighbours, with frequency assigned
from path length the way real link planning does it (short hops get 38 GHz,
long hops 15–18 GHz). The default network is 90 links over 414 km of path.

The forward model, in order:

1. **Line integral** of specific attenuation, `A = k · mean(R^α) · L`, using
   ITU-R P.838-3 coefficients from `core/itu_p838.py`.
2. **Wet-antenna attenuation** — a saturating term, after Schleiss et al. (2013).
3. **Baseline error** — uncertainty in the dry-weather reference, σ = 0.25 dB.
4. **Receiver noise** — σ = 0.12 dB.
5. **Quantization** — 0.3 dB reporting steps.

Retrieval inverts steps 1–2 and is blind to 3–5. The wet-antenna correction is
applied by fixed-point iteration with *deliberately mismatched* coefficients,
because in practice you never have the true ones.

### What the sensor model shows

**Path-averaging bias is real but small, and it changes sign.** Attenuation
integrates `R^α` while the retrieval solves for a single `R`, so by Jensen's
inequality the retrieved path mean overshoots when `α > 1` and undershoots when
`α < 1`. Since α crosses 1 near 23 GHz, both happen inside one network. The
size of the bias tracks how variable the rain is along the path:

| field | median along-path CV | median bias, α > 1 | median bias, α < 1 |
|---|---|---|---|
| Stratiform | 0.26 | +0.3% | −0.1% |
| Convective | 1.27 | +4.8% | −1.3% |
| Frontal | 0.29 | +0.3% | −0.2% |

Worst case across the convective field is about +12%. At normal CML
frequencies α sits within 0.11 of 1, so the power law is nearly linear — which
is *why* path-averaged CML rainfall works as well as it does.

**Wet-antenna attenuation is the dominant sensor error, by a wide margin.** On
wet convective links the median rain signal is 2.47 dB and the wet-antenna term
is 1.62 dB — 66% of the signal. Baseline error (0.25 dB), receiver noise
(0.12 dB) and quantization (0.09 dB) together are under a fifth of it.

## The headline result

Reconstructing with path-aware IDW, at 90 links:

| field | RMSE, geometry alone | RMSE, full sensor chain | sensor share | correlation | peak retained |
|---|---|---|---|---|---|
| Stratiform | 1.08 mm/h | 1.08 mm/h | 0.1% | 0.83 | 60% |
| Convective | 7.73 mm/h | 7.76 mm/h | 0.4% | 0.47 | 17% |
| Frontal | 10.06 mm/h | 9.96 mm/h | −1.1% | 0.38 | 13% |

"Geometry alone" reconstructs from the *true* path averages, so it isolates
what is lost purely to sparse line-integral sampling. "Full sensor chain"
reconstructs from retrieved values. The gap between them is what modelling the
sensor physics actually costs you.

**At operational density, the sensor chain is a rounding error.** Almost all of
the reconstruction error comes from the network geometry — from cells that fall
between links and gradients that no link crosses. The small negative share for
the frontal field is not the sensors helping; it is two error fields partially
cancelling, and it is within realization noise.

Two caveats worth stating plainly:

- **IDW cannot reproduce intermittency at all.** Estimated wet area is 99–100%
  for every field, against true values of 86 / 20 / 52%. Inverse-distance
  weighting produces a smooth field with no dry pixels, so wet-area ratio —
  the statistic that most distinguishes these three regimes — does not survive
  reconstruction. A method that can produce dry areas is needed for that.
- **Peaks are destroyed.** Only 13–17% of the convective and frontal peak
  intensity survives.

### Where the sensor chain does start to matter

Averaging over six independent topologies per density:

| links | stratiform sensor share | convective | frontal |
|---|---|---|---|
| 25 | 0.5% | 1.2% | ~0% |
| 90 | 6.2% | 1.2% | ~0% |
| 280 | **25.8%** | ~0% | ~0% |

For the smooth field, densifying the network drives geometry error down
(1.44 → 0.91 mm/h) while sensor error stays put, so its share grows — and past
about 160 links the *total* error starts rising again. There is an optimal
density beyond which better sampling cannot buy anything, because the sensor
bias sets a floor.

For the convective and frontal fields no such crossover appears even at 280
links: the structure is finer than the link spacing at every density tested,
so geometry keeps dominating.

## Layout

```
rainfall_field_sim/
├── src/
│   ├── figures.py        # figure builders
│   └── run_demo.py       # entry point
├── results/              # generated figures
├── requirements.txt
└── README.md
```

The models are shared by `opensense_pipeline`, `physics_ml` and
`spatial_interpolation`, so they live in `core/`:

```
core/
├── simulation/
│   ├── rain_fields.py    # the three models and their statistics
│   ├── moving_fields.py  # translation, growth and evolution in time
│   ├── cml_network.py    # topology, forward model, impairments, retrieval
│   └── reconstruct.py    # IDW variants, scoring, error decomposition
└── viz_style.py          # palette and matplotlib defaults
```

Figures: `fig1` the three fields, `fig2` their proportions, `fig3` the network
over each field, `fig4` the sensor chain, `fig5` the reconstructions, `fig6`
the error budget and density sweep, `fig7` the fields in motion.

## Notes

- ITU-R P.838-3 coefficients come from `core/itu_p838.py`, added with this
  project. `projects/physics_ml/src/rain_simulator.py` carries an older copy of
  the same table that interpolates slightly differently; it was left untouched.
- Every number in this README is produced by `run_demo.py` with default
  settings. Re-run it rather than trusting the table if you change anything.
- All randomness is seeded, so the figures are reproducible.
