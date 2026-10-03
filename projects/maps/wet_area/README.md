# wet_area: rain maps that can be dry

**Question.** IDW spreads every wet link over its whole radius, so a link map is wet almost
everywhere and its peaks are diluted by the dry links around them. Does deciding wet or dry
first, from the same links, fix it?

**Methods** (`core/maps/wet_area.py`), all of hourly totals from links alone, power 2 within
10 km as in [`multisensor`](../multisensor/):

- `idw`: plain IDW of every link, the baseline;
- `masked`: the same map, zero where the distance-weighted share of wet links is below a cut `p`;
- `conditional`: IDW of the wet links only, with the same mask (the indicator approach of
  geostatistics, with IDW weights instead of kriging).

**Data.** 16 simulated cases (`core.simulation.scenario.random_scenario`, links only, scored
against the true field) and the 29 storms of [`multisensor`](../multisensor/) on OpenMRG,
OpenRainER and OpenMesh, scored against the radar on the same grid and at held-out gauges
(never mapped). Two link retrievals: the RNN and the dynamic-baseline power law.

## Findings

Medians over cases or events; full tables in [`results/report.md`](results/report.md).

- **The mask fixes the wet area.** Plain IDW of power-law links is 12-29% too wet against
  the radar and the gauges; with `p = 0.5` the wet fraction is within a few percent of the
  reference (simulated: 1.14 to 1.02) and false alarms fall (Gothenburg, at the gauges: FAR
  0.24 to 0.06; Emilia-Romagna 0.31 to 0.23). The RNN links are already close to right, so
  the mask changes little there.
- **It does not fix the amounts.** NRMSE and correlation move by about 1% or less on every
  network and both references: the hourly error is in where the heavy rain is, not in the
  drizzle at the edges.
- **Conditional IDW does not bring the peaks back.** On simulated rain the 99th percentile of
  the map is 0.38 of the truth's for IDW and 0.42 for conditional IDW. Peaks fall between
  links, and no interpolation of the same links recovers them (as in
  [`simulation/regimes`](../../simulation/regimes/)).
- `p = 0.5` is the sound default; `p = 0.7` starts to miss real rain (CSI falls).

## Running it

```bash
python projects/maps/wet_area/src/run.py synthetic   # 16 cases, ~3 min -> results/synthetic.csv
python projects/maps/wet_area/src/run.py real        # needs maps/multisensor's event inputs, ~3 min
python projects/maps/wet_area/src/run.py report      # -> results/report.md, summary_*.csv
```

The real-data step reads the per-event inputs `maps/multisensor` caches under
`dataset/open_datasets/_multisensor_maps/merging/inputs/` (`run.py merge` there builds them).

| notebook | what |
|---|---|
| [`01_wet_area.ipynb`](notebooks/01_wet_area.ipynb) | the tables, and one Gothenburg hour mapped four ways |
