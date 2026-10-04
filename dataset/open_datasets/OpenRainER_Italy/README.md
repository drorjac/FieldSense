# OpenRainER - precipitation data from Emilia-Romagna, Italy

Two years (2021-2022) of commercial microwave links from the Lepida ScpA
network, with Arpae-SIMC weather radar and rain gauges.

- **Official source:** Covi and Roversi, [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808) (v2.0.3), CC BY 4.0
- **OpenSense page:** <https://opensenseaction.eu/news/new-open-cml-dataset-from-italy-openrainer/>
- **Example subset:** [opensense_example_data/OpenRainER](https://github.com/OpenSenseAction/opensense_example_data/tree/main/OpenRainER), read in `examples/02_openrainer.ipynb`

## Getting the data

```bash
python -m core.opensense.fetch --dataset openrainer   # CML, gauges and radar rain, 1.4 GB
python projects/maps/archive_pipeline/src/ingest_openrainer.py
```

The two other radar archives (`RADadj.tar`, `RADref.tar`, 1.6 GB each) are
fetched only on request: `--files RADadj.tar`.
