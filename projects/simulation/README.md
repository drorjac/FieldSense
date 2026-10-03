# simulation: known truth for every stage

On real data the truth is never known. Here the rain, its motion and every sensor are
simulated, so each method of the other stages can be scored against the true field.

| subproject | question | start here |
|---|---|---|
| [`regimes`](regimes/) | stratiform, convective and frontal rain under a link network: does the map error come from the sensors or from where the links are? | `notebooks/01_regimes_and_reconstruction.ipynb` |
| [`testbed`](testbed/) | every map, merging, motion and nowcast method scored against the truth, from street level (100 m, 5 min) to city scale | `notebooks/01_generators_and_motion.ipynb` |

**Headlines.**
- Geometry, not sensor physics, limits the map: at 90 links the sensor chain changes the error
  by about 1% or less; almost all of it is rain falling between links.
- No map resolves a street over five minutes (best NRMSE 1.12, radar 1.60); within 250 m of a
  link the error halves (0.85).

Shared code: `core/simulation/` (generators, flows, 1-D fields, sensors, scenarios, benchmark).
