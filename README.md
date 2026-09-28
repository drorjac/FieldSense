# FieldSense

Rainfall sensing with the networks that already exist.

Rain gauges are sparse and weather radar measures rain indirectly, high above
the ground. Networks that are already everywhere can fill the gap: the
microwave links between mobile-phone towers (commercial microwave links, CMLs)
lose signal in rain, and personal weather stations (PWS) are dense and free.
FieldSense turns the signal loss into rain rates and rainfall maps, merges
them with gauges and radar, forecasts the field a short time ahead, and asks
where physics-informed machine learning helps.

**New here?** Start with [GETTING_STARTED.md](GETTING_STARTED.md): forking,
installing, and opening your first dataset.

## The chain

```
 open data  ──►  signal loss to rain rate  ──►  rainfall map, merged with radar  ──►  forecast
 examples/       cml_retrieval                  opensense_pipeline                    spatial_interpolation
 openmesh_nyc    physics_ml

 and why the results look the way they do:  rainfall_field_sim (simulation), physics_ml
```

| project | question | start here |
|---|---|---|
| [`examples/`](examples/) | what is in each open dataset, and how to read it | `examples/00_overview.ipynb` |
| [`openmesh_nyc`](projects/openmesh_nyc/) | the OpenMesh NYC dataset: links, PWS, ASOS, radar; collection pipelines and paper | `notebooks/openmesh_data.ipynb` |
| [`cml_retrieval`](projects/cml_retrieval/) | signal loss to rain rate with PyNNcml on OpenMRG: model-driven chain vs a two-step RNN; five map methods | `notebooks/model_driven_retrieval.ipynb` |
| [`opensense_pipeline`](projects/opensense_pipeline/) | raw open data (OpenMRG, OpenRainER) to merged rainfall maps, scored against radar and gauges | `notebooks/02_end_to_end.ipynb` |
| [`spatial_interpolation`](projects/spatial_interpolation/) | nowcasting 15-60 min ahead from CML maps: Transformer, GRU, POD-SINDy vs persistence | `notebooks/nowcasting.ipynb` |
| [`rainfall_field_sim`](projects/rainfall_field_sim/) | on simulated rain: does the error come from the sensors or from where the links are? | `python src/run_demo.py` |
| [`physics_ml`](projects/physics_ml/) | hybrid physics + neural retrieval; rediscovering the ITU-R rain law (PySR) and advection (SINDy) | `notebooks/hybrid_retrieval.ipynb` |

## Findings so far

Each is computed in the project named, where the details and caveats are.

- **Merging helps only where the links are weak.** On the dense Swedish
  network the links alone beat radar and every merge against held-out gauges
  (RMSE 4.45 vs 6.16 mm/h); on the sparse Italian network merging only edges
  radar (7.90 vs 8.00), and all of the gain is within 5 km of a link.
  *(opensense_pipeline)*
- **The wet-antenna correction sets the magnitude of retrieved rain,** and
  its default does not transfer between networks: the ratio to the OpenSense
  reference moves from 1.93 to 0.74 across plausible settings.
  *(opensense_pipeline)*
- **Radar is not always the reference.** Over Manhattan the links correlate
  +0.53 with PWS and about 0 with the KOKX radar, whose beam passes far above
  the city; in snow the links carry no precipitation signal.
  *(opensense_pipeline)*
- **Geometry, not sensor physics, limits the map.** At 90 links the sensor
  chain changes the reconstruction error by about 1% or less; almost all of
  it comes from rain that falls between links. *(rainfall_field_sim)*
- **Machine learning recovers part of the physics.** PySR finds the ITU-R
  exponent from real data (0.911 vs 0.913) but not the prefactor (83% high,
  the wet-antenna offset again); SINDy finds the advection velocity from
  exact fields but not through a CML network; the hybrid retrieval does not
  beat its own physics branch. *(physics_ml)*

## Example datasets

[`examples/`](examples/) has one notebook per dataset of
[OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data):
OpenMRG (Gothenburg), OpenRainER (Emilia-Romagna), OpenMesh (New York City)
and the Amsterdam PWS. Files download on first use, about 60 MB in all.

```python
from core.opensense import example_data
data = example_data.load("openmrg", "8d")    # {"cml": ..., "radar": ..., "gauge_municipal": ..., ...}
```

[`dataset/README.md`](dataset/README.md) lists every dataset with its
official source, license and citation, and how to fetch the full records.

## Layout

```
FieldSense/
├── GETTING_STARTED.md    # start here
├── examples/             # one notebook per OpenSense example dataset
├── projects/             # the research projects above; each has src/, notebooks/, README
├── core/                 # code more than one project imports (see core/README.md)
│   ├── opensense/            # data: fetch, example subsets, conventions, retrieval, wet/dry, PWS QC, scoring
│   ├── simulation/           # synthetic rain fields, moving fields, CML network, reconstruction
│   ├── scientific_packages/  # PyNNcml compatibility and RNN training; PySINDy/PySR notes
│   ├── radar/                # KOKX NEXRAD for New York
│   ├── itu_p838.py           # ITU-R P.838-3 rain attenuation
│   └── viz_style.py          # shared palette and matplotlib defaults
├── dataset/              # dataset documentation; data is downloaded, never committed
├── tests/                # pytest for core/ (synthetic data, no downloads)
└── pyproject.toml        # installs core/ as a package
```

Projects import shared code as `from core... import` and never from each
other; code moves into `core/` when a second project needs it. Notebooks are
short: each cell calls into the project's `src/` or `core/`, and every number
shown is computed.

## Installation

```bash
git clone https://github.com/<your-username>/FieldSense.git && cd FieldSense   # your fork
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[opensense,notebooks,dev]"
pip install -r projects/<project>/requirements.txt    # the project you work on
```

## Tests

```bash
python -m pytest        # core and every project's tests, offline, about 30 s
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT License, see [LICENSE](LICENSE).
