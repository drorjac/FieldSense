# maps: links, gauges and radar to a rainfall map

Stage 3 of the chain. Which interpolation turns point and path measurements into a field,
and what does each sensor add when links, gauges, weather stations and radar are merged?

| subproject | question | networks | start here |
|---|---|---|---|
| [`multisensor`](multisensor/) | retrieval x interpolation (IDW, line IDW, GMZ) on the 10 largest storms of each network, then radar, links and gauges merged every way | OpenMRG, OpenRainER, OpenMesh | `notebooks/01_three_networks.ipynb` |
| [`radar_adjustment`](radar_adjustment/) | the OpenSense radar-adjustment intercomparison reproduced from the raw archives, extended to weather stations, RADOLAN and New York | OpenMRG, OpenRainER, NYC | `notebooks/01_intercomparison.ipynb` |
| [`nyc`](nyc/) | NYC Mesh link maps against MRMS radar, PWS and ASOS over 52 storms of rain, snow and mix | OpenMesh | `notebooks/01_data.ipynb` |
| [`wet_area`](wet_area/) | maps that can be dry: a wet/dry mask from the links before IDW, on simulated truth and 29 storms | OpenMRG, OpenRainER, OpenMesh | `notebooks/01_wet_area.ipynb` |
| [`link_weights`](link_weights/) | each link weighted by its expected error (learned by length on half the storms) when links are mapped | OpenMRG, OpenRainER, OpenMesh | `notebooks/01_link_weights.ipynb` |
| [`netherlands`](netherlands/) | RAINLINK over the whole Dutch network (15-min Pmin/Pmax), summer 2012: paths and IDW maps against KNMI's hourly gauges | Netherlands CML | `notebooks/01_netherlands.ipynb` |
| [`archive_pipeline`](archive_pipeline/) | (frozen) the first end-to-end version, raw data to merged maps; kept for its results, superseded by the three above | OpenMRG, OpenRainER | `notebooks/02_end_to_end.ipynb` |
| [`learned_2d`](learned_2d/) | proposed project: starter. Learned link-to-map models (`docs/pre_projects/2d_project.md`): the dataset (link channels, link table, radar target, gauges, split by storm event), IDW / kriging / GMZ scored with the proposal's metrics, a minimal U-Net, simulated truth | OpenMRG | `notebooks/01_dataset.ipynb` |

**Headlines.**
- Retrieval matters more than interpolation: the spread between retrievals is several times
  the spread between interpolations; line IDW is the best interpolation for power-law links.
- Merged maps at held-out gauges: Gothenburg radar + RNN links 1.01 (radar 1.47);
  Emilia-Romagna KED with gauges 2.21 to 1.27; New York PWS with gauge-corrected radar 0.88.
- Radar adjustment reproduces the intercomparison within 0.0006 RMSE; every adjustment beats
  the radar (Gothenburg 1.44 to 1.27 mm, Emilia-Romagna 3.58 to 3.08), almost all of it near
  the links. Weather stations beat links as adjusters wherever they exist.
- A wet/dry mask before IDW fixes the wet area of link maps (power-law links 12-29% too wet,
  then within a few percent) but not their amounts or peaks.
- ARPAE's own gauge-adjusted radar (RMSE 2.26 mm) beats every link adjustment (3.08) at
  every distance from the links; it is not independent of the gauges it is scored at.
- Power-law links under 1 km read 2-5 times too high; weighting links by their learned error
  by length cuts power-law map error at held-out gauges (Gothenburg NRMSE 3.62 to 2.33, New
  York 1.56 to 0.95). The RNN's error is flat with length.
- Over New York the best link map differs from MRMS by NRMSE 0.57, against 0.33 for the PWS.
- Across the Netherlands (summer 2012, ~2800 paths, RAINLINK), IDW link maps at KNMI's
  gauges reach hourly correlation 0.77 and daily r² 0.72 with -3% bias; single paths are
  noisier (median correlation 0.69), and a few paths under 1 km read several times the
  gauge's rain.

Shared code: `core/maps/` (IDW, GMZ, merging, mergeplg methods, scores, map skill, learning datasets).
