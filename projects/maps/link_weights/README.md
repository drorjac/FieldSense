# link_weights: weighting each link by its expected error

**Question.** A link's error depends on its length: the wet-antenna offset does not grow
with the path, so on a short link it is a large share of the signal. If each link's
expected error is known, does weighting the links by it make better maps?

**Method.** On the 29 cached storms of [`multisensor`](../multisensor/) (OpenMRG,
OpenRainER, OpenMesh), every other storm of a network trains, the rest test. On training
storms, hourly link totals are compared with the radar averaged along each path
(`core.maps.geometry.path_average_points`); the mean squared error per length bin (0-1,
1-2, 2-4, 4-8, > 8 km) is the error model. On test storms links are mapped three ways with
`core.maps.idw.idw_map` (power 2, 10 km): plain; weighted by the inverse error variance of
their bin (`weights=`); and plain without the links under 1 km. Maps are scored against the
radar on the grid and at held-out gauges. Two retrievals: the dynamic-baseline power law
and the RNN.

## Findings

Full tables: `results/error_model.csv`, `results/summary.csv`, `results/test_scores.csv`.

| held-out gauges, NRMSE (median of test storms) | plain IDW | weighted | no links < 1 km |
|---|---|---|---|
| Gothenburg, power law | 3.62 | 2.33 | 2.21 |
| Emilia-Romagna, power law | 2.61 | 2.36 | 2.57 |
| New York, power law | 1.56 | 0.95 | 0.93 |
| Gothenburg, RNN | 0.65 | 0.66 | 0.64 |
| Emilia-Romagna, RNN | 1.51 | 1.51 | 1.52 |
| New York, RNN | 0.74 | 0.72 | 0.76 |

- **Power-law links read far too high on short paths.** Below 1 km the dynamic-baseline
  retrieval is +434% against the radar in Gothenburg and about +190% in Emilia-Romagna and
  New York; above 4 km the bias is a fraction of that.
- **Weighting by the learned error repairs most of the power-law map**, against the gauges
  and against the radar. Dropping the short links does about as well where short links are
  the whole problem (Gothenburg, New York), not in Emilia-Romagna, where the 1-4 km links
  are also biased.
- **The RNN's error is flat across lengths**, so weighting changes nothing for it: it has
  learned the length-dependent offset the power law misses (as in
  [`retrieval/rnn_three_networks`](../../retrieval/rnn_three_networks/)).

**Caveats.** 5 test storms per network; the error model is learned against the radar,
which has its own errors (in Emilia-Romagna it reads about double the gauges). The weights
are per length bin only; a per-link model (frequency, a wet-antenna fit) is the natural
next step, and the subject of [`physics_ml/path_law_1d`](../../physics_ml/path_law_1d/).

## Running it

```bash
python projects/maps/link_weights/src/run.py    # ~1 min; needs maps/multisensor's event inputs
```

| notebook | what |
|---|---|
| [`01_link_weights.ipynb`](notebooks/01_link_weights.ipynb) | error by length, and the maps on the test storms |
