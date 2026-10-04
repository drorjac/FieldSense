# physics_ml: physics found from data

Does a physics-aware learning method (symbolic regression, SINDy, hybrid networks) find
what the physics says is there, in FieldSense's own quantities?

| subproject | question | start here |
|---|---|---|
| [`discovery`](discovery/) | PySR on the ITU-R rain law, SINDy on advection, and a hybrid physics + neural retrieval | `notebooks/hybrid_retrieval.ipynb` |
| [`path_law_1d`](path_law_1d/) | proposed project: starter. The path law `f(R, L)` and a wet antenna with memory: simulated truth, OpenMRG with radar along each link, the linear-law and literature baselines scored by link length | `notebooks/01_simulated_path_law.ipynb` |

**Headline.** PySR finds the ITU-R exponent from real data (0.911 vs 0.913) but not the
prefactor (83% high, the wet-antenna offset); SINDy finds the advection velocity from exact
fields but not through a link network; the hybrid retrieval does not beat its physics branch.
