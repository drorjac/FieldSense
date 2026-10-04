# data: the datasets

Stage 1 of the chain. Datasets FieldSense builds or curates itself, with the pipelines
that collect them. The open datasets it only reads (OpenMRG, OpenRainER, OpenMRG2 PWS,
the OpenSense example subsets) are described in [`DATA.md`](../../DATA.md) and
[`examples/`](../../examples/).

| subproject | what it is | start here |
|---|---|---|
| [`openmesh_nyc`](openmesh_nyc/) | OpenMesh: NYC Mesh microwave links with PWS, ASOS gauges and radar over New York; the collection pipelines and the data paper | `notebooks/openmesh_data.ipynb` |
