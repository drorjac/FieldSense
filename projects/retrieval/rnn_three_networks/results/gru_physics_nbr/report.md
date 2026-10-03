# RNN `gru_physics_nbr` against the power law

_Trained on openmrg, openrainer, openmesh in 5 min (best epoch 11); configuration in `config.json`._

## Head to head against the target

Each power-law method against the RNN on that method's own valid test hours (hourly link rain, mm/h).

| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |
|---|---|---|---|---|---|---|---|
| openmrg | dynamic | 666,872 | 0.451 / 2.371 | 0.833 / 0.522 | +16% / +191% | 0.67 / 0.58 | yes |
| openmrg | constant | 666,870 | 0.451 / 1.557 | 0.833 / 0.344 | +16% / -25% | 0.67 / 0.29 | yes |
| openmrg | pycomlink | 666,872 | 0.451 / 1.098 | 0.833 / 0.596 | +16% / +8% | 0.67 / 0.45 | yes |
| openmrg | nearby | 638,910 | 0.440 / 0.563 | 0.837 / 0.758 | +18% / -16% | 0.67 / 0.46 | yes |
| openrainer | dynamic | 829,741 | 0.631 / 3.177 | 0.811 / 0.286 | +3% / +137% | 0.54 / 0.32 | yes |
| openrainer | constant | 828,897 | 0.631 / 3.614 | 0.811 / 0.159 | +3% / -7% | 0.54 / 0.24 | yes |
| openrainer | pycomlink | 829,741 | 0.631 / 2.404 | 0.811 / 0.270 | +3% / -26% | 0.54 / 0.32 | yes |
| openrainer | nearby | 679,047 | 0.623 / 1.209 | 0.817 / 0.472 | +2% / -46% | 0.55 / 0.31 | yes |
| openmesh | dynamic | 66,063 | 0.715 / 4.216 | 0.795 / 0.223 | +9% / +653% | 0.67 / 0.23 | yes |
| openmesh | constant | 65,891 | 0.716 / 1.412 | 0.795 / 0.405 | +9% / -24% | 0.67 / 0.28 | yes |
| openmesh | pycomlink | 66,063 | 0.715 / 1.164 | 0.795 / 0.444 | +9% / -45% | 0.67 / 0.32 | yes |
| openmesh | nearby | 53,551 | 0.780 / 1.271 | 0.784 / 0.498 | +9% / -30% | 0.68 / 0.49 | yes |

## Head to head against the radar

Each power-law method against the RNN on that method's own valid test hours (hourly link rain, mm/h).

| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |
|---|---|---|---|---|---|---|---|
| openmrg | dynamic | 660,406 | 0.511 / 2.395 | 0.789 / 0.490 | +14% / +185% | 0.64 / 0.55 | yes |
| openmrg | constant | 660,404 | 0.511 / 1.586 | 0.789 / 0.321 | +14% / -25% | 0.64 / 0.28 | yes |
| openmrg | pycomlink | 660,406 | 0.511 / 1.137 | 0.789 / 0.557 | +14% / +7% | 0.64 / 0.42 | yes |
| openmrg | nearby | 636,217 | 0.501 / 0.631 | 0.792 / 0.699 | +14% / -19% | 0.64 / 0.43 | yes |
| openrainer | dynamic | 810,125 | 0.731 / 3.234 | 0.784 / 0.269 | -11% / +103% | 0.51 / 0.31 | yes |
| openrainer | constant | 809,281 | 0.731 / 3.689 | 0.784 / 0.146 | -11% / -20% | 0.51 / 0.24 | yes |
| openrainer | pycomlink | 810,125 | 0.731 / 2.479 | 0.784 / 0.250 | -11% / -36% | 0.51 / 0.33 | yes |
| openrainer | nearby | 661,391 | 0.723 / 1.320 | 0.791 / 0.434 | -12% / -53% | 0.53 / 0.31 | yes |
| openmesh | dynamic | 66,063 | 0.679 / 4.209 | 0.813 / 0.227 | +7% / +642% | 0.67 / 0.22 | yes |
| openmesh | constant | 65,891 | 0.679 / 1.395 | 0.813 / 0.415 | +7% / -26% | 0.67 / 0.29 | yes |
| openmesh | pycomlink | 66,063 | 0.679 / 1.149 | 0.813 / 0.453 | +7% / -46% | 0.67 / 0.33 | yes |
| openmesh | nearby | 53,551 | 0.726 / 1.250 | 0.807 / 0.504 | +9% / -30% | 0.68 / 0.50 | yes |

## Head to head against the city

Each power-law method against the RNN on that method's own valid test hours (hourly link rain, mm/h).

| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |
|---|---|---|---|---|---|---|---|
| openmrg | dynamic | 293,183 | 0.427 / 3.287 | 0.852 / 0.490 | +7% / +270% | 0.67 / 0.59 | yes |
| openmrg | constant | 293,181 | 0.427 / 2.225 | 0.852 / 0.278 | +7% / -9% | 0.67 / 0.28 | yes |
| openmrg | pycomlink | 293,183 | 0.427 / 1.442 | 0.852 / 0.572 | +7% / +28% | 0.67 / 0.49 | yes |
| openmrg | nearby | 267,134 | 0.393 / 0.567 | 0.864 / 0.789 | +12% / -8% | 0.67 / 0.51 | yes |

## Head to head against the gauges

Each power-law method against the RNN on that method's own valid test hours (hourly link rain, mm/h).

| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |
|---|---|---|---|---|---|---|---|
| openrainer | dynamic | 582,188 | 0.683 / 2.581 | 0.771 / 0.340 | +47% / +232% | 0.54 / 0.31 | yes |
| openrainer | constant | 581,534 | 0.683 / 2.233 | 0.771 / 0.266 | +47% / +16% | 0.54 / 0.26 | yes |
| openrainer | pycomlink | 582,188 | 0.683 / 2.049 | 0.771 / 0.317 | +47% / +0% | 0.54 / 0.33 | yes |
| openrainer | nearby | 473,040 | 0.681 / 0.863 | 0.768 / 0.609 | +44% / -29% | 0.55 / 0.34 | yes |

## Head to head against the pws

Each power-law method against the RNN on that method's own valid test hours (hourly link rain, mm/h).

| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |
|---|---|---|---|---|---|---|---|
| openmrg | dynamic | 399,392 | 0.516 / 2.897 | 0.764 / 0.435 | +30% / +276% | 0.62 / 0.55 | yes |
| openmrg | constant | 399,390 | 0.516 / 1.961 | 0.764 / 0.263 | +30% / -5% | 0.62 / 0.27 | yes |
| openmrg | pycomlink | 399,392 | 0.516 / 1.358 | 0.764 / 0.504 | +30% / +36% | 0.62 / 0.45 | yes |
| openmrg | nearby | 374,653 | 0.502 / 0.629 | 0.768 / 0.694 | +33% / +1% | 0.63 / 0.47 | yes |
| openmesh | dynamic | 66,061 | 0.793 / 4.230 | 0.759 / 0.215 | +11% / +665% | 0.64 / 0.22 | yes |
| openmesh | constant | 65,889 | 0.793 / 1.452 | 0.759 / 0.385 | +11% / -23% | 0.64 / 0.28 | yes |
| openmesh | pycomlink | 66,061 | 0.793 / 1.207 | 0.759 / 0.426 | +11% / -44% | 0.64 / 0.32 | yes |
| openmesh | nearby | 53,549 | 0.877 / 1.321 | 0.744 / 0.481 | +10% / -30% | 0.64 / 0.49 | yes |
