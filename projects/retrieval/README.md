# retrieval: link signal to rain rate

Stage 2 of the chain. How should the signal loss of a microwave link be turned into a
rain rate: the ITU-R power law with its processing chain (wet/dry, baseline, wet antenna),
or a recurrent network trained on it?

| subproject | question | networks | start here |
|---|---|---|---|
| [`openmrg`](openmrg/) | PyNNcml's model-driven chain against its two-step RNN, and five map methods on top | OpenMRG | `notebooks/model_driven_retrieval.ipynb` |
| [`rnn_three_networks`](rnn_three_networks/) | the two-step RNN trained on three networks at once, head to head with every power-law variant | OpenMRG, OpenRainER, OpenMesh | `notebooks/02_rnn_vs_power_law.ipynb` |

**Headline.** The RNN beats every power-law method on held-out weeks on all three networks,
against the radar and against each gauge network (New York: correlation 0.78 against 0.50
for the best power law), and its maps are the best link maps downstream
([`maps/multisensor`](../maps/multisensor/)).

Shared code: `core/cml/` (power law, preprocessing, RNN), `core/opensense/` (wet/dry, retrieval chain).
