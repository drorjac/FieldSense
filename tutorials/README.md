# Tutorials

Ten notebooks that teach the methods behind FieldSense, in the order of the chain: open
data, link signal to rain, a trained retrieval, weather radar, 2D rain maps, radar
adjustment, nowcasting with pysteps and with a neural network, nowcasting with every
sensor together, and a simulated rain field on which every method can be checked against the
truth. Each method is written
out by hand first, then run with the library that implements it, on real data, with
labelled figures and a list of references and links at the end.

Tutorials 1-9 run on the small OpenSense example subsets, which download on first use (about
60 MB, into `~/data/cml/`; see [DATA.md](../DATA.md)). They are committed with their
outputs, so they can be read on GitHub without running anything.

| # | notebook | you will learn | data | runtime |
|---|---|---|---|---|
| 1 | [Open data](01_open_data.ipynb) | the five datasets (sensors, resolution, licence, DOI); the OpenSense data format; opening each one; interval-start vs interval-end time stamps | all example subsets | < 1 min |
| 2 | [Link signal to rain rate](02_cml_signal_to_rain.ipynb) | ITU-R P.838-3 attenuation; total loss, quality control, wet/dry, baseline, wet antenna, k-R power law; checks against radar and a gauge | OpenMRG | ~1 min |
| 3 | [Training a retrieval network](03_training_a_retrieval_network.ipynb) | inputs and targets from link signals; a leakage-free split by time; a small GRU trained with early stopping; comparison with the power law | OpenMRG | ~1.5 min |
| 4 | [Weather radar](04_weather_radar.ipynb) | reflectivity and Z-R relations; the SMHI, ARPAE and MRMS products; grids, rates and depths; radar along a link and at gauges; why radar is not ground truth | OpenMRG, OpenRainER | < 1 min |
| 5 | [2D rain maps](05_2d_rain_maps.ipynb) | IDW, line IDW, GMZ, ordinary kriging (variogram and kriging system by hand), block kriging for lines; leave-one-out scoring; parameter sensitivity | OpenMRG | ~1.5 min |
| 6 | [Radar adjustment](06_radar_adjustment.ipynb) | mean-field bias, residual IDW, difference kriging, kriging with external drift, range checks, RADOLAN; links and weather stations as adjusters; evaluation at held-out gauges | OpenMRG, OpenMRG2 PWS | ~2.5 min |
| 7 | [Nowcasting with pysteps](07_nowcasting_with_pysteps.ipynb) | optical flow (by hand, LK, VET, DARTS, Proesmans); semi-Lagrangian extrapolation; the cascade, S-PROG, ANVIL, LINDA, STEPS; CSI, FSS, SAL, CRPS, ROC, reliability; nowcasting link maps; advection interpolation | OpenRainER, OpenMRG | ~1.5 min |
| 8 | [Learned nowcasting](08_learned_nowcasting.ipynb) | a nowcasting dataset without leakage; a U-Net in PyTorch; training with validation and early stopping; a like-for-like comparison with pysteps; why networks trained on squared error blur | OpenRainER | ~4-6 min |
| 9 | [Links, radar and weather stations together](09_multisensor_nowcasting.ipynb) | every sensor on one 5-minute grid; link and PWS maps with gaps; radar adjusted with both; nowcasting a sensor map along its own motion or the radar's; scoring on the same cells and gauges; a U-Net with and without the sensors | OpenMRG, OpenMRG2 PWS | ~8 min |
| 10 | [A simulated rain field and every sensor](10_simulated_rain_testbed.ipynb) | a Gaussian random field from its spectrum; the meta-Gaussian transform to rain; rain cells (Gaussian, HyCell); exact advection; what a link, a gauge and a radar measure, written out; a map's accuracy from street level to city scale; motion estimates against the true motion | simulated (no download) | ~1 min |

Runtimes are for a laptop CPU, after the data have downloaded.

## Running them

From the repository root, with the root install (see [GETTING_STARTED.md](../GETTING_STARTED.md)):

```bash
pip install -e ".[opensense,notebooks]"
jupyter lab tutorials/
```

or execute one from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/05_2d_rain_maps.ipynb
```

Tutorial 6 runs RADOLAN only in an environment with the development version of `mergeplg`
(`.venv-mergeplg-main`); elsewhere it explains the method and points to
[`projects/radar_adjustment`](../projects/radar_adjustment/README.md). Tutorials 7, 8 and 9 set
`OMP_NUM_THREADS=1` before importing anything: on macOS, torch and pysteps' multi-threaded
code crash when they share a process.

## From tutorial to project

Each tutorial shows a method on a few days of data; the projects run it on full records and
report what holds up.

| tutorial | project |
|---|---|
| 1 | [`openmesh_nyc`](../projects/openmesh_nyc/), [`examples/`](../examples/) |
| 2 | [`cml_retrieval`](../projects/cml_retrieval/), [`opensense_pipeline`](../projects/opensense_pipeline/) |
| 3 | [`cml_rnn`](../projects/cml_rnn/), [`physics_ml`](../projects/physics_ml/) |
| 4 | [`nyc_rain_maps`](../projects/nyc_rain_maps/), [`multisensor_maps`](../projects/multisensor_maps/) |
| 5 | [`multisensor_maps`](../projects/multisensor_maps/), [`rainfall_field_sim`](../projects/rainfall_field_sim/) |
| 6 | [`radar_adjustment`](../projects/radar_adjustment/), [`multisensor_maps`](../projects/multisensor_maps/), [`opensense_pipeline`](../projects/opensense_pipeline/) |
| 7 | [`os_nowcasting`](../projects/os_nowcasting/) |
| 8 | [`spatial_interpolation`](../projects/spatial_interpolation/), [`multisensor_nowcasting`](../projects/multisensor_nowcasting/) |
| 9 | [`multisensor_nowcasting`](../projects/multisensor_nowcasting/) |

Every paper, dataset and package cited in the tutorials, with links:
[docs/references.md](../docs/references.md).
