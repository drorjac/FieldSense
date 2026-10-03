# maps/netherlands: results

Computed by `python projects/maps/netherlands/src/run.py report` from the CSVs here. RAINLINK (pycomlink's port, default parameters, ITU-R P.838-3 vertical) on the Dutch network, summer (JJA) 2012, against KNMI's hourly automatic gauges. Hourly totals in mm; wet = 0.1 mm. A link hour needs 75% of its 15-min steps; a day needs 20 hours valid in both series, summed over the same hours.

`rel_bias` = sum(est)/sum(ref) - 1; `nrmse` = RMSE / mean(ref); `cv` = std(residual) / mean(ref) and `r2` = corr^2, the two numbers RAINLINK's papers report.

## Network

| what | value |
|---|---|
| period | 2012-06-01 .. 2012-08-31 (interval ends in (2012-06-01 00:00:00, 2012-09-01 00:00:00]) |
| paths_in_data | 2793 |
| sublinks_in_data | 4878 |
| sublinks_usable | 4876 |
| paths_usable | 2791 |
| sublinks_with_rain_estimate_median_per_step | 4418 |
| paths_with_hourly_value_median_per_hour | 2567 |
| wet_fraction_of_estimates | 0.089 |
| knmi_gauges_with_data | 32 |
| grid | 0.02 deg, 141 x 196 cells |

## Path level: each path against the nearest KNMI gauge within 5 km of its midpoint

| pairs_used | scale | pairs | gauges | n | mean_ref | mean_est | rel_bias | nrmse | cv | corr | r2 | pod | far | csi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all paths | hourly | 264 | 27 | 546865 | 0.13 | 0.17 | 0.26 | 11.36 | 11.35 | 0.41 | 0.17 | 0.53 | 0.12 | 0.50 |
| all paths | daily | 264 | 27 | 22491 | 3.19 | 3.93 | 0.23 | 4.90 | 4.89 | 0.38 | 0.15 | 0.79 | 0.05 | 0.75 |
| paths >= 1 km | hourly | 205 | 27 | 433997 | 0.13 | 0.13 | -0.02 | 5.08 | 5.08 | 0.63 | 0.40 | 0.54 | 0.13 | 0.50 |
| paths >= 1 km | daily | 205 | 27 | 17969 | 3.18 | 3.08 | -0.03 | 1.59 | 1.59 | 0.71 | 0.50 | 0.79 | 0.06 | 0.75 |

Pairs: 264 paths at 27 gauges, median distance 3.9 km, median path length 2.0 km. Per pair (`path_pairs.csv`): median hourly correlation 0.69; the ratio of link to gauge summer totals has median 0.96, quartiles 0.75 to 1.33, and is above 3 for 11 paths, of which 8 are shorter than 1 km.

## Map level: IDW (power 2, 10 km) at every KNMI gauge

Maps on a 0.02 deg grid, read at the cell holding each gauge; `masked` is `core.maps.wet_area.masked_idw` (p = 0.5). Gauges with no link within the radius have no map value and are not scored (`gauges_covered`).

| map | scale | gauges_covered | n | mean_ref | mean_est | rel_bias | nrmse | cv | corr | r2 | pod | far | csi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idw | hourly | 31 | 68448 | 0.13 | 0.13 | -0.03 | 3.60 | 3.60 | 0.77 | 0.59 | 0.67 | 0.18 | 0.58 |
| idw | daily | 31 | 2852 | 3.11 | 3.01 | -0.03 | 1.03 | 1.03 | 0.85 | 0.72 | 0.87 | 0.09 | 0.80 |
| masked | hourly | 31 | 68448 | 0.13 | 0.11 | -0.12 | 3.71 | 3.71 | 0.76 | 0.57 | 0.53 | 0.10 | 0.50 |
| masked | daily | 31 | 2852 | 3.11 | 2.75 | -0.12 | 1.06 | 1.05 | 0.83 | 0.70 | 0.77 | 0.04 | 0.74 |

Per gauge: `map_stations.csv`.

## Published RAINLINK numbers

Overeem, Leijnse and Uijlenhoet (2016, AMT 9, 2425-2444, Sect. 4.3-4.4), 12 days of 2011 from Nokia links, against gauge-adjusted radar on every land pixel (here: point gauges, a different reference), with RAINLINK's own DSD-based power law:

| source | what | scale | rel_bias | cv | r2 |
|---|---|---|---|---|---|
| Overeem et al. 2016, AMT | path-averaged links | 15 min | 0.105 | 3.840 | 0.540 |
| Overeem et al. 2016, AMT | map, IDW | 15 min | 0.083 | 3.300 | 0.410 |
| Overeem et al. 2016, AMT | map, IDW | daily | 0.082 | 0.510 | 0.700 |
| Overeem et al. 2016, AMT | map, kriging | daily | 0.014 | 0.540 | 0.730 |

## Per gauge (IDW, hourly)

Ratio of the map's summer total to the gauge's, over the 31 covered gauges: median 0.90, quartiles 0.80 to 1.11.
