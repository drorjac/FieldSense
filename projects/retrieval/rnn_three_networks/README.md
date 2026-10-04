# cml_rnn: PyNNcml's RNN, trained on three networks, against the power law

Can a recurrent network turn microwave-link signals into rain better than the ITU-R power
law? This project trains PyNNcml's two-step RNN (Habi & Messer) on three networks at once -
Gothenburg (OpenMRG), Emilia-Romagna (OpenRainER) and New York City (OpenMesh) - against the
*average of several references* along each link, and scores it on held-out weeks head to head
with the four power-law retrievals of `core.cml`.

**Result: yes.** Per link and hour on the test weeks, the RNN (`gru_physics_nbr`) has lower
RMSE *and* higher correlation than every power-law method, each on its own valid hours, on all
three networks - against the averaged target, the radar alone and every gauge network alone.
The strongest power-law method is the nearby-link one (Overeem 2016):

| network | RNN RMSE / corr / bias / CSI | nearby-link RMSE / corr / bias / CSI |
|---|---|---|
| Gothenburg | 0.44 / 0.84 / +18% / 0.67 | 0.56 / 0.76 / -16% / 0.46 |
| Emilia-Romagna | 0.62 / 0.82 / +2% / 0.55 | 1.21 / 0.47 / -46% / 0.31 |
| New York City | 0.78 / 0.78 / +9% / 0.68 | 1.27 / 0.50 / -30% / 0.49 |

(mm/h, hourly link rain against the averaged target on nearby-link's valid test hours;
`results/gru_physics_nbr/report.md` has every method against every reference. The other
power-law methods are further behind; the dynamic baseline over-reads by +100% to +650%.)

**In maps too.** Used as a retrieval in `projects/maps/multisensor` - on the ten largest storms
of each network, all of them test for the RNN - its maps are the best link maps against the
radar on all three networks, and at held-out gauges in Emilia-Romagna and New York. At
Gothenburg's ten municipal gauges it has the lower RMSE, bias and false-alarm rate, while the
nearby-link map keeps a slightly higher correlation (0.87 vs 0.85).

## What made the difference

Validation RMSE picked each step (test weeks never used); test weeks shown for the record
(`results/ablation.csv`, same calibration for all):

| inputs | validation RMSE | Gothenburg corr | Emilia-Romagna corr | New York RMSE / corr |
|---|---|---|---|---|
| 60 one-minute excess-loss values + 4 hourly summaries | 0.526 | 0.79 | 0.75 | 1.31 / 0.54 |
| + the power-law rate of that excess loss | 0.480 | 0.81 | 0.78 | 1.11 / 0.59 |
| + what the links within 15 km see | **0.408** | **0.84** | **0.82** | **0.78 / 0.78** |
| *nearby-link power law* | | *0.76* | *0.47* | *1.27 / 0.50* |

- **Physics as an input.** A network trained mostly on light rain shrinks heavy rain toward the
  mean and cannot extrapolate. Giving it the power-law rate R = (A / aL)^(1/b) of each minute's
  excess loss supplies a quantity that already scales with heavy rain; the network learns how
  far to trust it.
- **Neighbours.** The nearby-link method is the only power-law method that looks at the other
  links - to decide wet or dry and to set each link's reference level - and it was the one the
  RNN could not beat in dense networks. The median rate of the links within 15 km and the share
  of them seeing excess loss gave the RNN the same information; New York's links, few and noisy,
  gained most. The features use other links' signals only, never a reference.
- **Tried and dropped**, by validation: an LSTM backbone (0.526 vs 0.480 for the GRU), 7-day
  training windows (0.505), and oversampling wet windows (0.466-0.474 - it taught the network to
  over-read ordinary hours).

## Data

| network | links | references along each link | source |
|---|---|---|---|
| Gothenburg (OpenMRG), JJA 2015 | TSL + RSL, 10 s -> 1 min | SMHI radar, Netatmo PWS, municipal gauges | [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689); Andersson et al. (2022) |
| Emilia-Romagna (OpenRainER), 2021-22 | TSL + RSL, 1 min | ARPAE radar (15-min totals), ARPAE gauges | [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808) |
| New York City (OpenMesh), 2023-24 | RSL only, 1 min | MRMS radar, WU PWS | [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692), [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286); Jacoby et al. (2026) |

All are read through `core/opensense/networks.py`, hour-ending in UTC; file locations are in
[`DATA.md`](../../../DATA.md). The time-label convention of each source (interval start or end)
was established by lagging it against the links (see [`multisensor_maps`](../../maps/multisensor/)).

## What is trained on what

**Inputs** (`core/cml/rnn.py`), per link and hour: 60 one-minute values of *excess loss* - total
loss (TSL - RSL, or -RSL where TSL is not recorded) minus a causal baseline, the median of the
previous day's 15-minute medians - and four hourly summaries (share of missing minutes, mean and
maximum excess, standard deviation of the loss); the power-law rate of the excess loss (hour's
mean and maximum, rate of the mean); the neighbours' median rate and wet share. Metadata:
frequency, length, polarization, ITU-R P.838-3 log10(a) and b.

**Target**: the mean of the references available along the link - radar averaged along the path,
and the gauges within 3 km of it (Netatmo PWS and municipal gauges in Gothenburg, ARPAE gauges
in Emilia-Romagna, WU PWS in New York).

**Network**: PyNNcml's `two_step_network` unchanged - a 2-layer GRU (128 features), a metadata
branch, a rain head and a wet-probability head; the paper's loss (rain-weighted MSE plus
wet/dry cross-entropy). Random 96-hour windows (24 hours of context not scored) from the three
networks in equal shares; early stopping on validation. About 5 minutes on 8 CPU cores.

**Split** (`settings.split`): ISO weeks cycle train, train, validation, test; the ten largest
storms of each network are always test, so the test weeks are wetter and heavier than the rest.

**Calibration** (training and validation weeks only): the wet-probability threshold that
minimizes validation RMSE, and one scale per network making the estimate's total equal the
target's over the *wet* hours (target >= 1 mm) - fitted on all hours, drizzle set the scale and
it pulled heavy rain down.

| network | links | hours | train / validation / test hours | wet test link-hours |
|---|---|---|---|---|
| Gothenburg, JJA 2015 | 716 | 2,208 | 833 / 412 / 963 | 114,378 |
| Emilia-Romagna, May-Nov 2021 and 2022 | 288 | 10,272 | 4,503 / 2,512 / 3,257 | 105,472 |
| New York City, Nov 2023 - Jun 2024 | 39 | 5,832 | 2,749 / 1,182 / 1,901 | 11,619 |

## How to run

```bash
pip install -e ".[opensense,mrms,notebooks]" && pip install -r projects/retrieval/rnn_three_networks/requirements.txt
python projects/retrieval/rnn_three_networks/src/run.py build                    # the three hourly datasets (~40 min)
python projects/retrieval/rnn_three_networks/src/run.py train --name gru_physics_nbr --set physics=true neighbours=true
python projects/retrieval/rnn_three_networks/src/run.py evaluate --name gru_physics_nbr    # -> results/gru_physics_nbr/
```

Datasets and models are written to `dataset/open_datasets/_cml_rnn/` (not tracked).
`core.cml.rnn.HourlyRNN` runs a trained model on any link set; `projects/maps/multisensor` uses
it as a map retrieval.

| notebook | |
|---|---|
| `notebooks/01_data.ipynb` | inputs, references and the split |
| `notebooks/02_rnn_vs_power_law.ipynb` | the trained model against every power-law method, per reference |

## Layout

```
cml_rnn/
├── src/run.py
├── src/cml_rnn/
│   ├── build.py      # per network: inputs, references, power-law estimates -> one hourly dataset
│   ├── train.py      # PyNNcml two-step network, windows, loss, early stopping, calibration
│   ├── evaluate.py   # test-week scores; head to head against each power-law method
│   ├── report.py, settings.py
├── notebooks/
└── results/          # gru_physics_nbr/ (report, scores, figures), ablation.csv
```

## Caveats

- The references disagree with each other (radar against gauges, hourly NRMSE 1-1.6 on these
  networks); the RNN is trained on their mean and scored against each alone. Against the ARPAE
  gauges alone it reads high (+47%) where the radar reads low - the mean sits between them.
- Gothenburg's record is one summer (13 weeks), so its validation weeks hold few storms; the
  per-network scale is therefore fitted on training and validation weeks together.
- One model for three networks: the network is known where it is used, and enters only through
  the per-network scale.

## Related

- [`tutorials/03_training_a_retrieval_network.ipynb`](../../../tutorials/): training a small
  retrieval network step by step, with the data preparation and split explained.
- [`cml_retrieval`](../openmrg/): the same PyNNcml network trained against gauges on
  OpenMRG alone.
- [`multisensor_maps`](../../maps/multisensor/) and [`radar_adjustment`](../../maps/radar_adjustment/):
  the trained model as a map retrieval and as a radar adjuster in New York.

## References

1. Habi, H. V., and Messer, H. (2021). Recurrent neural network for rain estimation using
   commercial microwave links. *IEEE TGRS*, 59(5), 3672-3681.
   [doi:10.1109/TGRS.2020.3010305](https://doi.org/10.1109/TGRS.2020.3010305);
   code: [PyNNcml](https://github.com/haihabi/PyNNcml)
2. Overeem, A., Leijnse, H., and Uijlenhoet, R. (2016). Retrieval algorithm for rainfall
   mapping from microwave links in a cellular communication network. *Atmospheric Measurement
   Techniques*, 9, 2425-2444. [doi:10.5194/amt-9-2425-2016](https://doi.org/10.5194/amt-9-2425-2016)
3. Andersson, J. C. M., et al. (2022). OpenMRG. *Earth System Science Data*, 14, 5411-5426.
   [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)
4. Jacoby, D., et al. (2026). OpenMesh: wireless signal dataset for opportunistic urban
   weather sensing in New York City. *Earth System Science Data*, 18, 5817-5836.
   [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)
5. ITU-R P.838-3 (2005). <https://www.itu.int/rec/R-REC-P.838-3-200503-I/en>
