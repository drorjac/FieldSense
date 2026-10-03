# netherlands: RAINLINK over a whole country

**Question.** The Dutch network (Overeem et al. 2024) is the largest open CML dataset: ~2800
paths over 35,000 km², 15-min minimum and maximum received power, no transmitted power.
Run through RAINLINK, the retrieval written for it, how close do path rain and link maps
come to KNMI's automatic gauges, hour by hour and day by day?

**Data.** Summer (June-August) 2012, from the 9.5 GB 4TU archive
([`dataset/open_datasets/CML_Netherlands`](../../../dataset/open_datasets/CML_Netherlands/)).
2012 has data on every summer day and ~4000 reporting sub-links per 15 minutes; 2011 is
as complete, while from 2013 on the network shrinks as fibre replaces links
(`results/availability_monthly.csv`). In the study period: 2793 paths, 4878 sub-links,
of which 2 are dropped (0 km, or a reported 254 GHz, as in the dataset README); a median
of 4418 sub-links carries a rain estimate per 15 minutes. Reference: KNMI's hourly
precipitation (`RH`) from its 32 automatic stations that measure rain, fetched from the
open climatology service; `-1` (< 0.05 mm) is taken as 0. The gauges are never used to
make the link estimates.

**Method.**

- **Retrieval**: RAINLINK (Overeem, Leijnse and Uijlenhoet 2016, AMT 9, 2425-2444) with its
  default parameters, through pycomlink's port (`core.opensense.netherlands.rainlink_retrieval`):
  nearby-link wet/dry (15 km, -1.4 dB, -0.7 dB/km, 24 h), reference level from the dry
  intervals of the previous 24 h, wet-antenna attenuation 2.3 dB, `R = 0.33 R(Pmin) + 0.67 R(Pmax)`,
  outlier filter at -32.5 dB h/km. The power law is ITU-R P.838-3 for vertical polarization
  (RAINLINK ships its own DSD-derived coefficients). The wet/dry step is a vectorised copy of
  pycomlink's, tested equal to it (the original takes hours on this network).
- **Hourly**: hourly totals per sub-link (3 of 4 steps needed), averaged over the sub-links
  of each path, as RAINLINK does before mapping.
- **(a) Path level**: each path against the nearest gauge within 5 km of its midpoint.
- **(b) Map level**: IDW (power 2, 10 km, as in [`multisensor`](../multisensor/)) of the hourly
  path totals on a 0.02° grid, read at the cell holding each gauge; also with the wet mask of
  [`wet_area`](../wet_area/) (`core.maps.wet_area.masked_idw`, p = 0.5).
- Scores from `core.maps.scores` (wet = 0.1 mm), plus CV (std of residuals over the
  reference mean) and r², the two numbers RAINLINK's papers report. Daily totals over the
  hours valid in both series (at least 20).

## Findings

Full tables in [`results/report.md`](results/report.md).

- **Link maps are close to the gauges.** IDW at all 31 gauges with a link within 10 km:
  hourly correlation 0.77 (r² 0.59), daily 0.85 (r² 0.72), bias -3%, CSI 0.58 hourly and
  0.80 daily. Summer totals at the gauges are 0.90 of the gauges' (median; quartiles 0.80
  to 1.11).
- **Single paths are noisier, and a few short ones are wrong.** 264 paths within 5 km of
  27 gauges: median hourly correlation per pair 0.69, median total ratio 0.96. Pooled,
  hourly correlation is only 0.41 and the bias +26%, because 11 paths read more than three
  times the gauge's summer total, 8 of them shorter than 1 km, where a 1 dB step is tens of
  mm/h. Without the paths under 1 km (a check, not part of RAINLINK): hourly correlation
  0.63, daily 0.71, bias -2 to -3%. Mapping averages these paths away.
- **The wet mask does not help here.** It cuts false alarms (hourly FAR 0.18 to 0.10) but
  misses more rain (POD 0.67 to 0.53), lowers CSI (0.58 to 0.50) and adds a -12% bias.
  RAINLINK's own wet/dry classification already sets dry links to zero.
- **Against the published numbers.** Overeem et al. (2016, AMT, Sect. 4.4) report daily IDW
  maps with r² 0.70, CV 0.51 and +8% bias over 12 days of 2011 against gauge-adjusted
  radar. Here daily r² is 0.72 and CV 1.03: similar correlation, twice the scatter, which
  is what a point gauge against a 0.02° cell should add. The references differ, so this is
  a plausibility check, not a reproduction.

**Caveats.** One summer. KNMI's 32 automatic gauges are about 30 km apart, so the map
score rests on 31 points. The gauges are point measurements; the paths and cells average
over kilometres. ITU coefficients, not RAINLINK's own.
KNMI's gauge-adjusted radar would be the better reference (it is what RAINLINK was
validated against), but the KNMI Data Platform requires an API key, so it is not used.
The wettest day (26 August) is mapped about 1.4 times too wet at the gauges.

## Running it

```bash
python -m core.opensense.fetch --dataset netherlands       # 9.5 GB zip, once
python projects/maps/netherlands/src/run.py availability   # rows per month, 2011-2015 (~6 min)
python projects/maps/netherlands/src/run.py convert        # May-Aug 2012 to monthly netCDF (~11 min)
python projects/maps/netherlands/src/run.py retrieve       # RAINLINK, ~4900 sub-links (~11 min)
python projects/maps/netherlands/src/run.py score          # hourly series, maps at the gauges (~20 min)
python projects/maps/netherlands/src/run.py figures        # network and wettest-day maps
python projects/maps/netherlands/src/run.py report         # -> results/report.md
```

`all` runs every step. Caches go to `dataset/open_datasets/_netherlands/` (rain per
sub-link, hourly series, maps at the gauges); the monthly CML files and the KNMI responses
to `~/data/cml/netherlands/` (see `DATA.md`). Settings (period, grid, IDW, distances) are in
`src/netherlands/settings.py`.

| notebook | what |
|---|---|
| [`01_netherlands.ipynb`](notebooks/01_netherlands.ipynb) | availability, the network, paths against gauges, the tables, the wettest day (reads the caches, ~30 s) |

See also [`examples/05_netherlands.ipynb`](../../../examples/05_netherlands.ipynb): the data
format and one link's Pmin/Pmax beside a gauge.
