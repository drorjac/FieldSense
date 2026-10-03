# nowcasting: the next hour

Stage 4 of the chain. Can a rain map made from links, weather stations and radar be
forecast a short time ahead, by pysteps or by a neural network?

| subproject | question | networks | start here |
|---|---|---|---|
| [`pysteps`](pysteps/) | the OpenSense pysteps session as a study: radar, link and merged maps nowcast with every pysteps method | OpenRainER, OpenMRG | `notebooks/05_results.ipynb` |
| [`multisensor`](multisensor/) | links, radar and weather stations nowcast together, by pysteps and by 15 U-Nets: what each sensor adds | OpenMRG, NYC | `notebooks/04_results.ipynb` |
| [`spatial_interpolation`](../spatial_interpolation/) | (paper, kept at `projects/spatial_interpolation`) nowcasting 15-60 min from CML maps: Transformer, POD-SINDy, Mamba-style SSM vs persistence | OpenMRG | `notebooks/nowcasting.ipynb` |

**Headlines.**
- A link map cannot be nowcast on its own (CSI 0.16 vs 0.17 for persistence at 60 min); merged
  into the radar, links keep the radar's motion and nowcast as well as the radar.
- Links and weather stations lower the next-hour error at Gothenburg's gauges from 1.20 to
  1.02 mm; no U-Net beats the radar's pysteps extrapolation.

Shared code: `core/nowcast/` (advection, learned motion, methods, scores, verification).
