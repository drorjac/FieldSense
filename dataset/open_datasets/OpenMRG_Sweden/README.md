# OpenMRG - open data from microwave links, radar and gauges

Commercial microwave links, weather radar and rain gauges over Gothenburg,
Sweden, June-August 2015, published by SMHI.

- **Official source:** Andersson et al. (2022), [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689) (v1.1), CC BY-SA 4.0
- **OpenSense page:** <https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/>
- **Example subsets:** [opensense_example_data/OpenMRG](https://github.com/OpenSenseAction/opensense_example_data/tree/main/OpenMRG), read in `examples/01_openmrg.ipynb`

## Getting the data

```bash
python -m core.opensense.fetch --dataset openmrg      # 318 MB zip into raw/
python projects/maps/archive_pipeline/src/ingest_openmrg.py   # OpenSense NetCDF into processed/
```

`readme.txt` is SMHI's description of every file in the archive, and
`cml/example_read_cml.nc.py` and `radar/example_read_radar_nc.py` are SMHI's
example readers. The data files themselves are not in git.
