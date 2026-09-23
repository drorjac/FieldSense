# estimation_after_detection

> **Status: placeholder.** Scaffolding only — no implementation yet.

## Overview

Rain-rate estimation conditioned on a prior wet/dry detection step, rather
than estimating through the dry periods. The premise is that detection and
estimation have different error structures, and that treating them as one
regression forces a single model to do both jobs badly.

## Structure

```
estimation_after_detection/
├── src/
│   └── main.py           # placeholder entry point
├── requirements.txt      # empty until there are dependencies
└── README.md
```

## Related work in this repository

| | |
|---|---|
| `projects/opensense_pipeline/` | wet/dry classification and retrieval on real OpenMRG and OpenRainER data |
| `projects/cml_retrieval/` | PyNNcml wet-dry classifiers and baseline models |
| `projects/physics_ml/` | hybrid physics + neural retrieval |

## Team

- _unassigned_
