# wet_area: results

Computed by `python projects/maps/wet_area/src/run.py report` from `synthetic.csv` and `real.csv`. Every map is of hourly totals from links alone; wet = 0.1 mm. Medians over 16 synthetic cases and 29 real events.

`war_ratio`: the map's wet fraction over the reference's (1 is right). `peak_ratio`: 99th percentile of the map over the reference's.

## Against the true field (synthetic)

| map | nrmse | corr | rel_bias | csi | far | war_ratio | peak_ratio |
|---|---|---|---|---|---|---|---|
| idw | 1.333 | 0.195 | -0.276 | 0.862 | 0.137 | 1.14 | 0.384 |
| masked p0.3 | 1.338 | 0.189 | -0.34 | 0.868 | 0.122 | 1.077 | 0.384 |
| masked p0.5 | 1.344 | 0.182 | -0.371 | 0.758 | 0.113 | 1.021 | 0.384 |
| masked p0.7 | 1.344 | 0.164 | -0.398 | 0.593 | 0.071 | 0.776 | 0.384 |
| conditional p0.3 | 1.364 | 0.209 | -0.015 | 0.868 | 0.122 | 1.077 | 0.425 |
| conditional p0.5 | 1.378 | 0.196 | -0.214 | 0.758 | 0.113 | 1.021 | 0.424 |
| conditional p0.7 | 1.378 | 0.163 | -0.361 | 0.593 | 0.071 | 0.776 | 0.387 |

## Against the radar and held-out gauges (real events)

| reference | network | retrieval | map | nrmse | corr | rel_bias | csi | far | war_ratio | peak_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| gauges | openmesh | dynamic | idw | 1.217 | 0.69 | 0.593 | 0.859 | 0.141 | 1.165 | 1.118 |
| gauges | openmesh | dynamic | masked p0.3 | 1.215 | 0.689 | 0.587 | 0.868 | 0.132 | 1.152 | 1.118 |
| gauges | openmesh | dynamic | masked p0.5 | 1.216 | 0.686 | 0.568 | 0.881 | 0.119 | 1.132 | 1.118 |
| gauges | openmesh | dynamic | masked p0.7 | 1.198 | 0.692 | 0.497 | 0.866 | 0.118 | 1.082 | 1.118 |
| gauges | openmesh | dynamic | conditional p0.3 | 1.379 | 0.644 | 0.787 | 0.868 | 0.132 | 1.152 | 1.144 |
| gauges | openmesh | dynamic | conditional p0.5 | 1.355 | 0.646 | 0.745 | 0.881 | 0.119 | 1.132 | 1.144 |
| gauges | openmesh | dynamic | conditional p0.7 | 1.317 | 0.666 | 0.603 | 0.866 | 0.118 | 1.082 | 1.144 |
| gauges | openmesh | rnn | idw | 0.81 | 0.718 | -0.135 | 0.891 | 0.077 | 1.0 | 0.599 |
| gauges | openmesh | rnn | masked p0.3 | 0.811 | 0.718 | -0.139 | 0.891 | 0.077 | 1.0 | 0.599 |
| gauges | openmesh | rnn | masked p0.5 | 0.812 | 0.718 | -0.142 | 0.887 | 0.08 | 0.984 | 0.599 |
| gauges | openmesh | rnn | masked p0.7 | 0.818 | 0.718 | -0.147 | 0.829 | 0.068 | 0.918 | 0.599 |
| gauges | openmesh | rnn | conditional p0.3 | 0.81 | 0.713 | -0.126 | 0.891 | 0.077 | 1.02 | 0.599 |
| gauges | openmesh | rnn | conditional p0.5 | 0.81 | 0.713 | -0.133 | 0.887 | 0.08 | 0.984 | 0.599 |
| gauges | openmesh | rnn | conditional p0.7 | 0.817 | 0.714 | -0.146 | 0.829 | 0.068 | 0.918 | 0.599 |
| gauges | openmrg | dynamic | idw | 3.19 | 0.855 | 1.523 | 0.749 | 0.24 | 1.291 | 2.762 |
| gauges | openmrg | dynamic | masked p0.3 | 3.188 | 0.856 | 1.484 | 0.817 | 0.082 | 1.036 | 2.762 |
| gauges | openmrg | dynamic | masked p0.5 | 3.188 | 0.856 | 1.466 | 0.816 | 0.058 | 0.964 | 2.762 |
| gauges | openmrg | dynamic | masked p0.7 | 3.187 | 0.854 | 1.436 | 0.799 | 0.033 | 0.895 | 2.762 |
| gauges | openmrg | dynamic | conditional p0.3 | 3.195 | 0.855 | 1.555 | 0.817 | 0.082 | 1.036 | 2.762 |
| gauges | openmrg | dynamic | conditional p0.5 | 3.192 | 0.855 | 1.509 | 0.816 | 0.058 | 0.964 | 2.762 |
| gauges | openmrg | dynamic | conditional p0.7 | 3.191 | 0.853 | 1.46 | 0.799 | 0.033 | 0.895 | 2.762 |
| gauges | openmrg | rnn | idw | 0.843 | 0.887 | 0.026 | 0.808 | 0.093 | 0.966 | 0.89 |
| gauges | openmrg | rnn | masked p0.3 | 0.844 | 0.886 | 0.019 | 0.808 | 0.094 | 0.952 | 0.89 |
| gauges | openmrg | rnn | masked p0.5 | 0.844 | 0.885 | 0.01 | 0.782 | 0.094 | 0.929 | 0.89 |
| gauges | openmrg | rnn | masked p0.7 | 0.85 | 0.884 | 0.005 | 0.764 | 0.08 | 0.876 | 0.89 |
| gauges | openmrg | rnn | conditional p0.3 | 0.843 | 0.886 | 0.04 | 0.79 | 0.108 | 0.997 | 0.89 |
| gauges | openmrg | rnn | conditional p0.5 | 0.842 | 0.885 | 0.022 | 0.78 | 0.094 | 0.947 | 0.89 |
| gauges | openmrg | rnn | conditional p0.7 | 0.849 | 0.884 | 0.013 | 0.764 | 0.08 | 0.876 | 0.89 |
| gauges | openrainer | dynamic | idw | 2.91 | 0.51 | 0.811 | 0.602 | 0.309 | 1.231 | 1.772 |
| gauges | openrainer | dynamic | masked p0.3 | 2.897 | 0.517 | 0.762 | 0.613 | 0.275 | 1.147 | 1.762 |
| gauges | openrainer | dynamic | masked p0.5 | 2.865 | 0.521 | 0.68 | 0.619 | 0.226 | 1.048 | 1.726 |
| gauges | openrainer | dynamic | masked p0.7 | 2.797 | 0.537 | 0.575 | 0.592 | 0.181 | 0.961 | 1.678 |
| gauges | openrainer | dynamic | conditional p0.3 | 3.207 | 0.48 | 0.933 | 0.612 | 0.289 | 1.18 | 1.904 |
| gauges | openrainer | dynamic | conditional p0.5 | 3.001 | 0.496 | 0.781 | 0.619 | 0.229 | 1.054 | 1.808 |
| gauges | openrainer | dynamic | conditional p0.7 | 2.823 | 0.533 | 0.603 | 0.593 | 0.182 | 0.962 | 1.685 |
| gauges | openrainer | rnn | idw | 1.715 | 0.739 | 0.342 | 0.663 | 0.239 | 1.092 | 1.275 |
| gauges | openrainer | rnn | masked p0.3 | 1.716 | 0.739 | 0.337 | 0.664 | 0.231 | 1.078 | 1.275 |
| gauges | openrainer | rnn | masked p0.5 | 1.718 | 0.739 | 0.33 | 0.664 | 0.211 | 1.055 | 1.275 |
| gauges | openrainer | rnn | masked p0.7 | 1.719 | 0.739 | 0.314 | 0.655 | 0.19 | 1.011 | 1.275 |
| gauges | openrainer | rnn | conditional p0.3 | 1.723 | 0.737 | 0.362 | 0.659 | 0.244 | 1.103 | 1.275 |
| gauges | openrainer | rnn | conditional p0.5 | 1.722 | 0.737 | 0.346 | 0.664 | 0.214 | 1.06 | 1.275 |
| gauges | openrainer | rnn | conditional p0.7 | 1.721 | 0.739 | 0.32 | 0.655 | 0.19 | 1.012 | 1.275 |
| radar | openmesh | dynamic | idw | 1.227 | 0.608 | 0.632 | 0.895 | 0.103 | 1.111 | 1.666 |
| radar | openmesh | dynamic | masked p0.3 | 1.227 | 0.608 | 0.632 | 0.896 | 0.101 | 1.107 | 1.666 |
| radar | openmesh | dynamic | masked p0.5 | 1.227 | 0.61 | 0.627 | 0.892 | 0.089 | 1.076 | 1.666 |
| radar | openmesh | dynamic | masked p0.7 | 1.221 | 0.622 | 0.587 | 0.879 | 0.062 | 0.979 | 1.666 |
| radar | openmesh | dynamic | conditional p0.3 | 1.281 | 0.566 | 0.682 | 0.896 | 0.1 | 1.107 | 1.666 |
| radar | openmesh | dynamic | conditional p0.5 | 1.275 | 0.57 | 0.672 | 0.892 | 0.089 | 1.076 | 1.666 |
| radar | openmesh | dynamic | conditional p0.7 | 1.25 | 0.6 | 0.612 | 0.879 | 0.062 | 0.979 | 1.666 |
| radar | openmesh | rnn | idw | 0.641 | 0.838 | -0.137 | 0.872 | 0.064 | 1.042 | 0.64 |
| radar | openmesh | rnn | masked p0.3 | 0.641 | 0.838 | -0.138 | 0.876 | 0.064 | 1.042 | 0.64 |
| radar | openmesh | rnn | masked p0.5 | 0.641 | 0.838 | -0.142 | 0.882 | 0.059 | 0.998 | 0.64 |
| radar | openmesh | rnn | masked p0.7 | 0.641 | 0.837 | -0.145 | 0.875 | 0.057 | 0.95 | 0.64 |
| radar | openmesh | rnn | conditional p0.3 | 0.641 | 0.837 | -0.13 | 0.893 | 0.064 | 1.042 | 0.64 |
| radar | openmesh | rnn | conditional p0.5 | 0.641 | 0.837 | -0.138 | 0.882 | 0.059 | 1.007 | 0.64 |
| radar | openmesh | rnn | conditional p0.7 | 0.641 | 0.837 | -0.144 | 0.875 | 0.057 | 0.95 | 0.64 |
| radar | openmrg | dynamic | idw | 1.993 | 0.745 | 0.869 | 0.763 | 0.189 | 1.12 | 1.854 |
| radar | openmrg | dynamic | masked p0.3 | 1.993 | 0.749 | 0.842 | 0.766 | 0.123 | 1.038 | 1.854 |
| radar | openmrg | dynamic | masked p0.5 | 1.993 | 0.751 | 0.815 | 0.749 | 0.101 | 0.967 | 1.854 |
| radar | openmrg | dynamic | masked p0.7 | 1.993 | 0.752 | 0.768 | 0.718 | 0.085 | 0.887 | 1.854 |
| radar | openmrg | dynamic | conditional p0.3 | 2.014 | 0.732 | 0.945 | 0.771 | 0.128 | 1.051 | 1.866 |
| radar | openmrg | dynamic | conditional p0.5 | 2.003 | 0.744 | 0.88 | 0.751 | 0.102 | 0.97 | 1.861 |
| radar | openmrg | dynamic | conditional p0.7 | 1.999 | 0.749 | 0.803 | 0.718 | 0.085 | 0.887 | 1.858 |
| radar | openmrg | rnn | idw | 1.387 | 0.74 | 0.117 | 0.767 | 0.111 | 1.015 | 1.024 |
| radar | openmrg | rnn | masked p0.3 | 1.389 | 0.74 | 0.111 | 0.764 | 0.107 | 1.005 | 1.024 |
| radar | openmrg | rnn | masked p0.5 | 1.39 | 0.74 | 0.105 | 0.752 | 0.107 | 0.98 | 1.024 |
| radar | openmrg | rnn | masked p0.7 | 1.394 | 0.74 | 0.093 | 0.728 | 0.106 | 0.939 | 1.024 |
| radar | openmrg | rnn | conditional p0.3 | 1.395 | 0.739 | 0.138 | 0.77 | 0.11 | 1.028 | 1.027 |
| radar | openmrg | rnn | conditional p0.5 | 1.394 | 0.74 | 0.124 | 0.753 | 0.108 | 0.982 | 1.027 |
| radar | openmrg | rnn | conditional p0.7 | 1.397 | 0.74 | 0.104 | 0.729 | 0.106 | 0.939 | 1.027 |
| radar | openrainer | dynamic | idw | 2.561 | 0.533 | 0.334 | 0.589 | 0.26 | 1.086 | 1.249 |
| radar | openrainer | dynamic | masked p0.3 | 2.558 | 0.537 | 0.252 | 0.585 | 0.236 | 1.05 | 1.249 |
| radar | openrainer | dynamic | masked p0.5 | 2.536 | 0.54 | 0.127 | 0.567 | 0.208 | 0.982 | 1.249 |
| radar | openrainer | dynamic | masked p0.7 | 2.502 | 0.545 | -0.033 | 0.551 | 0.18 | 0.898 | 1.227 |
| radar | openrainer | dynamic | conditional p0.3 | 2.773 | 0.472 | 0.49 | 0.587 | 0.245 | 1.065 | 1.497 |
| radar | openrainer | dynamic | conditional p0.5 | 2.624 | 0.518 | 0.262 | 0.568 | 0.211 | 0.989 | 1.323 |
| radar | openrainer | dynamic | conditional p0.7 | 2.519 | 0.541 | -0.006 | 0.551 | 0.18 | 0.899 | 1.246 |
| radar | openrainer | rnn | idw | 1.45 | 0.704 | -0.215 | 0.623 | 0.21 | 1.009 | 0.815 |
| radar | openrainer | rnn | masked p0.3 | 1.451 | 0.704 | -0.217 | 0.62 | 0.208 | 0.985 | 0.815 |
| radar | openrainer | rnn | masked p0.5 | 1.452 | 0.703 | -0.221 | 0.609 | 0.199 | 0.928 | 0.815 |
| radar | openrainer | rnn | masked p0.7 | 1.455 | 0.702 | -0.226 | 0.594 | 0.187 | 0.872 | 0.814 |
| radar | openrainer | rnn | conditional p0.3 | 1.453 | 0.702 | -0.197 | 0.623 | 0.216 | 1.017 | 0.826 |
| radar | openrainer | rnn | conditional p0.5 | 1.454 | 0.702 | -0.215 | 0.61 | 0.202 | 0.938 | 0.823 |
| radar | openrainer | rnn | conditional p0.7 | 1.455 | 0.702 | -0.224 | 0.594 | 0.188 | 0.873 | 0.818 |
