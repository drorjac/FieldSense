 notebook # wet_area: rain maps that can be dry
 notebook 
 notebook **Question.** IDW spreads every wet link over its whole radius, so a link map is wet almost
 notebook everywhere and its peaks are diluted by the dry links around them. Does deciding wet or dry
 notebook first, from the same links, fix it?
 notebook 
 notebook **Methods** (`core/maps/wet_area.py`), all of hourly totals from links alone, power 2 within
 notebook 10 km as in [`multisensor`](../multisensor/):
 notebook 
 notebook - `idw`: plain IDW of every link, the baseline;
 notebook - `masked`: the same map, zero where the distance-weighted share of wet links is below a cut `p`;
 notebook - `conditional`: IDW of the wet links only, with the same mask (the indicator approach of
 notebook   geostatistics, with IDW weights instead of kriging).
 notebook 
 notebook **Data.** 16 simulated cases (`core.simulation.scenario.random_scenario`, links only, scored
 notebook against the true field) and the 29 storms of [`multisensor`](../multisensor/) on OpenMRG,
 notebook OpenRainER and OpenMesh, scored against the radar on the same grid and at held-out gauges
 notebook (never mapped). Two link retrievals: the RNN and the dynamic-baseline power law.
 notebook 
 notebook ## Findings
 notebook 
 notebook Medians over cases or events; full tables in [`results/report.md`](results/report.md).
 notebook 
 notebook - **The mask fixes the wet area.** Plain IDW of power-law links is 12-29% too wet against
 notebook   the radar and the gauges; with `p = 0.5` the wet fraction is within a few percent of the
 notebook   reference (simulated: 1.09 to 1.02) and false alarms fall (Gothenburg, at the gauges: FAR
 notebook   0.24 to 0.06; Emilia-Romagna 0.31 to 0.23). The RNN links are already close to right, so
 notebook   the mask changes little there.
 notebook - **It does not fix the amounts.** NRMSE and correlation move by about 1% or less on every
 notebook   network and both references: the hourly error is in where the heavy rain is, not in the
 notebook   drizzle at the edges.
 notebook - **Conditional IDW does not bring the peaks back.** On simulated rain the 99th percentile of
 notebook   the map is 0.38 of the truth's for IDW and 0.42 for conditional IDW. Peaks fall between
 notebook   links, and no interpolation of the same links recovers them (as in
 notebook   [`simulation/regimes`](../../simulation/regimes/)).
 notebook - `p = 0.5` is the sound default; `p = 0.7` starts to miss real rain (CSI falls).
 notebook 
 notebook ## Running it

Tutorial [11](../../../tutorials/11_dry_maps_and_link_weights.ipynb) teaches the method by hand on the OpenMRG subset.
 notebook 
 notebook ```bash
 notebook python projects/maps/wet_area/src/run.py synthetic   # 16 cases, ~3 min -> results/synthetic.csv
 notebook python projects/maps/wet_area/src/run.py real        # needs maps/multisensor's event inputs, ~3 min
 notebook python projects/maps/wet_area/src/run.py report      # -> results/report.md, summary_*.csv
 notebook ```
 notebook 
 notebook The real-data step reads the per-event inputs `maps/multisensor` caches under
 notebook `dataset/open_datasets/_multisensor_maps/merging/inputs/` (`run.py merge` there builds them).
 notebook 
 notebook | notebook | what |
 notebook |---|---|
 notebook | [`01_wet_area.ipynb`](notebooks/01_wet_area.ipynb) | the tables, and one Gothenburg hour mapped four ways |
