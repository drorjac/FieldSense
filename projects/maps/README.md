# maps: links, gauges and radar to a rainfall map

Stage 3 of the chain. Which interpolation turns point and path measurements into a field,
and what does each sensor add when links, gauges, weather stations and radar are merged?

| subproject | question | networks | start here |
|---|---|---|---|
| [`multisensor`](multisensor/) | retrieval x interpolation (IDW, line IDW, GMZ) on the 10 largest storms of each network, then radar, links and gauges merged every way | OpenMRG, OpenRainER, OpenMesh | `notebooks/01_three_networks.ipynb` |
| [`radar_adjustment`](radar_adjustment/) | the OpenSense radar-adjustment intercomparison reproduced from the raw archives, extended to weather stations, RADOLAN and New York | OpenMRG, OpenRainER, NYC | `notebooks/01_intercomparison.ipynb` |
| [`nyc`](nyc/) | NYC Mesh link maps against MRMS radar, PWS and ASOS over 52 storms of rain, snow and mix | OpenMesh | `notebooks/01_data.ipynb` |
| [`archive_pipeline`](archive_pipeline/) | (frozen) the first end-to-end version, raw data to merged maps; kept for its results, superseded by the three above | OpenMRG, OpenRainER | `notebooks/02_end_to_end.ipynb` |

**Headlines.**
- Retrieval matters more than interpolation: the spread between retrievals is several times
  the spread between interpolations; line IDW is the best interpolation for power-law links.
- Merged maps at held-out gauges: Gothenburg radar + RNN links 1.01 (radar 1.47);
  Emilia-Romagna KED with gauges 2.21 to 1.27; New York PWS with gauge-corrected radar 0.88.
- Radar adjustment reproduces the intercomparison within 0.0006 RMSE; every adjustment beats
  the radar (Gothenburg 1.44 to 1.27 mm, Emilia-Romagna 3.58 to 3.08), almost all of it near
  the links. Weather stations beat links as adjusters wherever they exist.
- Over New York the best link map differs from MRMS by NRMSE 0.57, against 0.33 for the PWS.

Shared code: `core/maps/` (IDW, GMZ, merging, mergeplg methods, scores).
