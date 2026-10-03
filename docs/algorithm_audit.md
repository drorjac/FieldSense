# Algorithm audit of core/

Every algorithm in `core/` was checked against its reference and against physical sanity.
The reference is the paper's equations, the library the code says it mirrors (pycomlink,
PyNNcml, pysteps, poligrain, pyproj, scipy), or a hand calculation. Each check is pinned by
a test in `tests/test_audit_<area>.py`. All of these tests are fast and synthetic, and none
of them downloads anything.

Verdicts:

- **OK**: the code agrees with the reference.
- **RISK**: the code is correct where it is used, but it rests on a fragile or
  undocumented assumption.
- **BUG**: the code gives a wrong result and has not been fixed. A strict `xfail` test
  describes the correct behaviour.
- **FIXED**: the code gave a wrong result and was fixed in its own commit.

Totals: 84 OK, 13 RISK, 4 BUG, 6 FIXED. One of the six fixes is documentation only.

A bug was left unfixed when fixing it would change published numbers, or when the right
behaviour is open to debate. The strict xfail then turns into a failure on the day someone
fixes the code, which is the signal to update the test.

## Table

| area | function | reference | verdict | test | note |
|---|---|---|---|---|---|
| itu | `itu_p838.ITU_P838_TABLE` | ITU-R P.838-3 eqs. 2-3 and Tables 1-4 | FIXED | `test_itu_p838_table_is_the_recommendation`, `test_the_two_copies_of_table_5_agree`, `test_spot_values_from_table_5` | alpha_V(10 GHz) read 1.2157 instead of 1.2156, and alpha_V(86 GHz) read 0.6930 instead of 0.6929 |
| itu | `power_law.ITU_2005`, `ITU_2003`, `itu_ab` | P.838-3 equations; pycomlink `a_b` | OK | `test_power_law_2005_table_is_the_recommendation`, `test_2003_and_2005_tables_against_pycomlink` | identical to pycomlink for both tables and both polarizations |
| itu | `get_k_alpha` vs `itu_ab` between table frequencies | P.838-3 equations | RISK | `test_off_grid_interpolation_against_the_equations`, `test_simulator_and_retrieval_power_laws_agree_within_two_percent` | see below |
| itu | `get_k_alpha` outside the table | P.838-3 | RISK | `test_get_k_alpha_clamps_outside_the_table` | see below |
| itu | `specific_attenuation`, `attenuation_from_rain` | dB/km = k R^alpha; dB = dB/km x km | OK | `test_specific_attenuation_units`, `test_horizontal_attenuates_more_than_vertical_in_rain` | gamma_H > gamma_V at 10-94 GHz, as oblate drops require |
| itu | `rain_from_attenuation` | R = (A/(aL))^(1/b); pycomlink `calc_R_from_A` | OK | `test_k_r_inversion_by_hand`, `test_inversion_matches_pycomlink` | A <= 0 gives 0, NaN stays NaN, R_min applied, per-link broadcasting correct |
| cml | `baseline.dynamic_baseline` | Ostrometzky & Messer 2018; PyNNcml | OK | `test_dynamic_baseline_is_trailing_min_minus_quantization_delta` | trailing (causal) window; A - min - qd |
| cml | `baseline.std_wet_dry` | Schleiss & Berne 2010; PyNNcml `STDWetDry` | OK | `test_std_wet_dry_matches_pynncml_and_is_centred` | population std; centred for odd windows, half a sample early for even ones, as in PyNNcml |
| cml | `baseline.constant_baseline` + `ConstantBaselineSTD` chain | PyNNcml `two_step_constant_baseline` | OK | `test_constant_baseline_chain_matches_pynncml_two_step`, `test_constant_baseline_treats_nan_wet_as_wet` | wet flags, baseline and rain identical |
| cml | `preprocess.total_loss` | TL = TSL - RSL (dB) | OK | `test_total_loss_is_tsl_minus_rsl_in_db` | |
| cml | `preprocess.min_max` | interval-ending, right-closed | OK | `test_min_max_is_interval_ending_and_right_closed` | |
| cml | `preprocess.gauge_wet_reference` | implementation_1 | RISK | `test_gauge_wet_reference_bin_start_interpolation` | see below |
| cml | `preprocess.fill_gaps_gauge_gated` | implementation_1 | OK | `test_fill_gaps_gauge_gated_only_fills_wet_gaps` | |
| cml | `link_qc.metadata_qc` duplicates | README procedure | FIXED | `test_metadata_qc_flags_reversed_duplicate`, `test_metadata_qc_duplicate_of_a_falsy_label` | `if dup_of:` missed a duplicate of the link labelled 0 |
| cml | `link_qc.retrieval_qc` | mm = mm/h x h | OK | `test_retrieval_qc_totals_use_the_sampling_interval` | |
| cml | `rnn.features`, `rnn.metadata` | hour-ending (T-1h, T]; causal baseline | OK | `test_rnn_features_are_hour_ending_and_causal`, `test_rnn_metadata_uses_itu_2005` | a later change in signal leaves the hour unchanged |
| cml | `opensense.retrieval.wet_antenna_attenuation` (pastorek2021, leijnse2008) | Pastorek et al. 2021, Leijnse et al. 2008 via pycomlink `*_from_A_obs` | OK | `test_lookup_waa_matches_pycomlink_from_A_obs`, `test_waa_inversion_is_consistent` | agree within 0.5 % |
| cml | `opensense.retrieval.wet_antenna_attenuation` (saturating) | A_obs = A_rain + W(R(A_rain)) | BUG | `test_saturating_waa_recovers_light_rain_on_low_k_links` (xfail) | see below |
| cml | `opensense.retrieval.retrieve`, `wet_dry_rolling_std` | hand case; pandas centred std | OK | `test_retrieve_known_step_without_wet_antenna`, `test_rolling_std_wet_dry_is_centred_sample_std` | |
| maps | `idw.idw_weights`, `apply_weights`, `points_idw_map` | IDW definition, hand weights | OK | `test_audit_idw_weights_exact_at_source_and_radius`, `test_audit_apply_weights_nan_excluded_and_hand_value`, `test_audit_points_idw_exact_at_station_and_bounded` | exact at sources, bounded, radius respected, NaN excluded per step |
| maps | `idw_weights` with `nnear` and NaN sources | pycomlink KDTree IDW (k nearest valid) | BUG | `test_audit_idw_nnear_counts_valid_sources_only` (xfail) | see below |
| maps | `idw_weights`, NaN source on a cell centre | idw.py docstring | RISK | `test_audit_idw_nan_source_on_cell_centre_drops_out` (xfail) | see below |
| maps | `idw.accumulate` | hour-ending label, mean rate x hours | OK | `test_audit_accumulate_interval_ending_and_coverage` | 1 h and 15 min factors |
| maps | `gmz.gmz_map` | Goldshtein et al. 2009; PyNNcml `GMZInterpolation` | OK | `test_audit_gmz_virtual_gauges_keep_path_average`, `test_audit_gmz_zero_iterations_is_line_idw` | virtual gauges keep mean R^b equal to the link's to 1e-5 |
| maps | `gmz.gmz_map`, the returned map | path average along the link | RISK | `test_audit_gmz_virtual_gauges_keep_path_average` | see below |
| maps | `merge.adjust` (mfb, add, mul), `merge_idw` | sum G / sum R; additive and multiplicative residual IDW | OK | `test_audit_mean_field_bias_hand_calculation`, `test_audit_additive_and_multiplicative_adjustment`, `test_audit_merge_idw_exact_at_gauges` | |
| maps | `mergeplg_methods.Merger` | exact offset, ratio and linear-drift cases | OK | `test_audit_mergeplg_methods_reproduce_exact_relations`, `test_audit_mergeplg_too_few_observations_returns_radar`, `test_audit_kriging_weights_invariant_to_variogram_scale` | |
| maps | `standardised_semivariogram` | white noise gives gamma = 1 | OK | `test_audit_standardised_semivariogram_white_noise_is_one` | |
| maps | `scores.scores` | NRMSE / mean(ref), rel_bias, POD/FAR/CSI | OK | `test_audit_scores_hand_case` | |
| maps | `geometry` segment distance, path averages | hand cases; linear field gives the midpoint value | OK | `test_audit_segment_distance_hand_cases`, `test_audit_distance_to_links_matches_projected_segment_distance`, `test_audit_path_average_points_linear_field_is_midpoint_value`, `test_audit_path_average_intersect_linear_field_is_midpoint_value` | |
| maps | `geometry.path_average_intersect`, link on a cell edge | poligrain | RISK | `test_audit_path_average_intersect_link_on_cell_edge` (xfail) | see below |
| maps | `wet_area` | indicator IDW; contingency table | OK | `test_audit_wet_probability_is_indicator_idw`, `test_audit_masked_and_conditional_idw`, `test_audit_wet_area_scores_hand_case` | |
| nowcast | `advection.interpolate_pair`, `hourly_total` | semi-Lagrangian; known blob and vector | OK | `test_interpolate_pair_moves_blob_halfway_along_velocity`, `test_interpolate_pair_conserves_mass_at_fractional_shifts`, `test_hourly_total_plain_mean_and_swath`, `test_hourly_total_linear_in_time_is_exact` | mass kept to 1e-5 in the interior |
| nowcast | `advection.interpolate_pair` at the edges | `mode="nearest"` | RISK | (none) | see below |
| nowcast | `methods.extrapolate`, `reachable` | pysteps semilagrangian vector convention | OK | `test_pysteps_extrapolation_uses_the_same_vector_convention`, `test_reachable_masks_cells_advected_in_from_outside` | |
| nowcast | `verify.Deterministic` categorical and continuous | contingency table by hand | OK | `test_categorical_scores_hand_case_and_nan_cells_excluded`, `test_categorical_scores_are_pooled_not_averaged` | |
| nowcast | FSS | Roberts & Lean 2008, by hand | OK | `test_fss_matches_hand_case`, `test_fss_window_from_scale_and_pixel_size` | |
| nowcast | `verify.Ensemble` CRPS without ties | E\|X-y\| - E\|X-X'\|/2 | OK | `test_crps_matches_closed_form`, `test_crps_of_a_degenerate_ensemble_is_the_absolute_error`, `test_ensemble_perfect_forecast_roc_and_mean` | |
| nowcast | `verify.Ensemble` CRPS with ties | same closed form | BUG | `test_crps_with_ties_matches_closed_form` (xfail) | see below |
| nowcast | `scores.sal_score` | Wernli et al. 2008 | OK | `test_sal_identical_scaled_and_shifted`, `test_sal_structure_sign_and_dry_fields`, `test_sal_two_objects_location_second_term` | |
| nowcast | `grid.square_grid`, `to_pysteps`, `to_dbr` | haversine; mm per step to mm/h | OK | `test_square_grid_pixels_are_square_and_pixel_km_correct`, `test_to_pysteps_converts_depth_per_step_to_rate`, `test_dbr_round_trip` | |
| nowcast | `grid.pixel_km`, `metadata` yorigin | sign and orientation | FIXED | `test_pixel_km_of_a_descending_lat_grid_is_positive` | a north-to-south grid gave a negative pixel size |
| nowcast | `grid.metadata` square pixels | cos(latitude) | RISK | (none) | see below |
| nowcast | `learned_motion` augmentation, shapes | vectors rotate with the arrays; pysteps (2, y, x) | OK | `test_learned_motion_augmentation_rotates_vectors_with_arrays`, `test_learned_motion_shapes_and_transform`, `test_simulated_velocity_px_matches_pysteps_advection` | |
| simulation | `random_fields.spectral_density`, `grf` | closed-form exponential, Gaussian and Matern covariance; P(k) ~ k^-beta | OK | `test_spectral_density_is_the_documented_covariance`, `test_grf_realisations_have_the_requested_correlation_at_lag`, `test_powerlaw_grf_has_the_requested_spectral_slope`, `test_grf_anisotropy_stretches_along_the_requested_angle` | correlation within 0.03 at 1, 3 and 6 km; slope within 0.15 |
| simulation | `wet_distribution`, `to_rain`, `MetaGaussian` | requested war and wet marginal | OK | `test_wet_distribution_has_the_requested_mean_and_cv`, `test_to_rain_wet_fraction_and_marginal`, `test_metagaussian_generator_hits_war_and_mean` | |
| simulation | `to_rain` 0.1 mm/h floor | "exact" war | RISK | `test_to_rain_threshold_eats_into_war_for_a_very_skewed_marginal` | see below |
| simulation | `BetaLognormalCascade`, `cascade_1d`, `RainFARM`, `downscale_to` | Over & Gupta 1996; mean preservation | OK | `test_beta_lognormal_cascade_wet_fraction_and_mean`, `test_cascade_1d_is_mean_preserving`, `test_rainfarm_wet_fraction_and_mean`, `test_downscale_to_reproduces_the_coarse_field` | |
| simulation | `UniversalMultifractal` | Schertzer & Lovejoy, K(q) = C1/(alpha-1)(q^alpha - q) | BUG | `test_universal_multifractal_moment_scaling_function` (xfail) | see below |
| simulation | `fields_1d`, `moving_fields.shift`, `flows` | depth conserved; known vector; divergence-free | OK | `test_pulses_to_series_conserves_depth`, `test_spectral_shift_moves_a_blob_by_the_known_vector`, `test_uniform_advect_matches_shift_and_departure_points`, `test_rotation_flow_turns_a_blob_counter_clockwise_and_keeps_its_mass`, `test_flows_are_divergence_free`, `test_velocity_px_units` | |
| simulation | `spacetime` | exact shift; rho = exp(-dt/tau); bands sum to one | OK | `test_frozen_uniform_sequence_is_an_exact_shift_and_fully_predictable`, `test_frozen_rotation_sequence_follows_the_flow`, `test_ar1_keeps_the_marginal_and_decorrelates_at_rho`, `test_octave_bands_partition_unity` | |
| simulation | `sensors.Radar`, `Gauges` | Z = aR^b; two-way PIA = 2 k R^alpha r; interval-ending labels | OK | `test_radar_zr_round_trip_without_impairments`, `test_radar_wrong_assumed_zr_gives_the_closed_form_bias`, `test_radar_two_way_pia_is_itu_times_twice_the_range`, `test_radar_azimuth_is_clockwise_from_north`, `test_radar_observe_scans_at_interval_end`, `test_gauges_accumulate_interval_ending_depths` | |
| simulation | `cml_network.forward_model`, `retrieve_rain` | ITU k R^alpha L; Jensen bias of path averaging | OK | `test_forward_model_uniform_rain_is_itu_and_inverts_exactly`, `test_path_averaging_bias_follows_jensen`, `test_retrieval_with_the_true_wet_antenna_recovers_the_rain` | audited read-only |
| simulation | `wet_antenna` (static, Schleiss 2013, Pastorek 2021, dynamic) | pycomlink; exact relaxation | OK | `test_static_wet_antenna_is_pastorek_with_zeta_one`, `test_waa_schleiss_matches_pycomlink`, `test_waa_schleiss_reaches_95_percent_after_tau`, `test_waa_pastorek_matches_pycomlink`, `test_dynamic_wet_antenna_is_the_exact_relaxation` | |
| simulation | `wet_antenna` growth step capped at 1 | pycomlink (no cap) | RISK | `test_waa_schleiss_matches_pycomlink` | identical while 3 dt < tau |
| opensense | `conventions.to_ghz`, `normalize_polarization`, `project_points`, `project_cml` | units; pyproj EPSG:32632 | OK | `test_to_ghz_declared_and_undeclared_units`, `test_normalize_polarization_spellings`, `test_project_points_matches_pyproj_and_preserves_distance`, `test_project_cml_midpoint_and_sites` | |
| opensense | `conventions.to_km` without declared units | magnitude guess | RISK | `test_to_km_declared_and_undeclared_units`, `test_undeclared_length_of_a_short_link_network_is_read_as_km` | see below |
| opensense | `conventions.itu_coefficients` | `get_k_alpha` per link | FIXED | `test_itu_coefficients_per_link_and_scalar_polarization` | a single polarization was replaced by vertical |
| opensense | `evaluation.rainfall_metrics`, `aggregate` | poligrain; hand calculation; label conventions | OK | `test_rainfall_metrics_against_hand_calculation`, `test_rainfall_metrics_pairs_by_coordinates_not_memory_order`, `test_aggregate_label_conventions` | |
| opensense | `intercomparison_chain` hourly, radar and preprocessing | the OpenSense notebook | OK | `test_hourly_from_mean_is_hour_ending_and_closed_right`, `test_hourly_links_mean_of_one_minute_depths`, `test_threshold_radar_zeroes_small_values_and_nan`, `test_openmrg_radar_rate_marshall_palmer_200_16`, `test_preprocess_spike_removes_the_higher_channel_only`, `test_preprocess_flat_link_is_removed` | |
| opensense | `intercomparison_chain.wet_from_radar` | window t-(rad_freq-1) .. t+5 | FIXED | `test_wet_from_radar_window_is_interval_plus_five_minutes` | one minute late for even steps of 8 min or more |
| opensense | `pws_qc.rate_flag` | two-tips rule | OK | `test_rate_flag_two_tips_rule` | |
| opensense | `pws_qc.regularize` | interval labels | RISK | `test_regularize_sums_amounts_and_keeps_gaps_nan` | see below |
| opensense | `pws_qc.neighbour_reference` | median of the neighbours | RISK | `test_neighbour_reference_excludes_self_and_far_stations` | see below |
| geo | `haversine_m` | sphere; pyproj WGS84 geodesic | OK | `test_haversine_known_values`, `test_haversine_vs_wgs84_geodesic_over_nyc` | at most 0.26 % from the ellipsoid over NYC |
| geo | `to_local_xy` | great circle, geodesic | FIXED (doc) | `test_to_local_xy_distance_accuracy_over_nyc`, `test_to_local_xy_axes_and_origin` | the docstring claimed under 0.1 %; the measured error is 0.3 % (0.6 % against WGS84) |
| events | `detect_events` | its documented rules | OK | `test_events_gap_rule_labels_and_totals`, `test_events_gap_of_exactly_min_gap_splits`, `test_events_min_total_and_wet_threshold`, `test_events_low_coverage_and_archive_gaps_are_dry` | hour-ending labels |

## Bugs left unfixed

**Saturating wet-antenna inversion does not converge on low-k links**
(`core/opensense/retrieval.py:216-224`).

- The model is W = waa_max (1 - exp(-c R)), and it is inverted by 8 fixed-point steps.
- The iteration is a contraction only while W'(R) dR/dA < 1. With the defaults
  (0.5 dB, 0.28 per mm/h) that holds only for links with k L above about 0.15.
  Below that are links at 15 GHz or lower, and 20-30 GHz links shorter than about 1 km.
- Below that limit the iteration oscillates, and light rain comes back as 0. Examples:
  - 15 GHz H on 2 km returns 0 at 1 mm/h and below, and 9 % too little at 2 mm/h.
  - 10 GHz V on 5 km returns 0 at 2 mm/h and below.
- 18-40 GHz links of normal length are unaffected.
- The correct solution is unique, because A_obs is monotone in R. The same lookup-table
  inversion that the pastorek and leijnse models already use would find it.
- Why not fixed: `saturating` is the default of `retrieve_dataset`, and its parameters were
  calibrated against gauges with this solver.
- What could change: OpenMesh and OpenRainER sublinks in `projects/maps/archive_pipeline`
  (`ingest_openmesh`, `ingest_openrainer`, `validate_retrieval`, `run_pipeline`), and short
  OpenMRG links.

**`nnear` counts NaN sources** (`core/maps/idw.py:49-52`).

- The k nearest sources are chosen before NaN sources are excluded. A cell whose nearest
  source is missing therefore averages k-1 sources, or comes out NaN.
- pycomlink takes the k nearest valid sources.
- Affected: `projects/nowcasting/pysteps/src/os_nowcasting/products.py` (`IDW_NNEAR = 12`).
- Why not fixed: the fix needs weights per time step and would change those products.

**CRPS is biased low when the observation equals a member**
(pysteps 1.21.5 `probscores.CRPS_accum`, called from `core/nowcast/verify.py:118`).

- pysteps uses strict inequalities only, so a tied bin drops out. For members
  [0, 1, 2, 3] and observation 2 the closed form gives 0.375; pysteps gives 0.0625.
- Ties are routine in rain data (dry pixel, dry members). On synthetic fields with 30-90 %
  dry cells the pooled CRPS is 1-5 % too low.
- Affected: the published CRPS of `projects/nowcasting/pysteps` and
  `projects/nowcasting/multisensor`.
- The fix belongs upstream, or in a local closed-form CRPS.

**UniversalMultifractal does not have universal-multifractal scaling**
(`core/simulation/generators.py:384, 392`).

- The stable noise is filtered with k^(-d/alpha) in Fourier space. Fractionally
  integrated flux needs k^(-d(1-1/alpha)), which is \|x\|^(-d/alpha) in real space.
- The scale also leaves out scipy's \|cos(pi alpha/2)\| factor.
- Measured at alpha = 1.6, C1 = 0.1: K(2) is 0.08 against the theoretical 0.17.
  alpha = 2 is unaffected.
- Affected: the `multifractal` rows of `projects/simulation/testbed`,
  `core/simulation/scenario.py` and `core/nowcast/learned_motion.py`.

## Risks

- **Two interpolations of Table 5** (`core/itu_p838.py:145`, `core/cml/power_law.py:39`).
  - The simulators and the opensense chain use `get_k_alpha`, which interpolates log-log
    linearly. The core/cml retrievals use `itu_ab`, a cubic spline.
  - Both are exact on the 1 GHz grid. Between grid points `get_k_alpha` is up to 2 % off
    the P.838-3 equations above 7 GHz and up to 3.3 % off at 5-7 GHz; `itu_ab` stays within
    0.3 %.
  - A link simulated with one and retrieved with the other at, say, 7.5 GHz is biased by
    about 1.5 % in R.
- **Silent clamping** (`core/itu_p838.py:141`).
  - `get_k_alpha` clamps frequencies outside the table, while `itu_ab` raises.
  - Vertical polarization is tabulated only to 94 GHz, so a 95-100 GHz vertical link gets
    the 94 GHz values (k up to 4 % low).
- **Gauge reference leads the rain** (`core/cml/preprocess.py:90-93`).
  - The 15-min gauge mean is stamped at its bin start and interpolated linearly. The
    reference therefore rises up to a bin before the rain and falls while it still rains.
  - After the last bin's start it is 0.
  - This mirrors implementation_1, which it reproduces on purpose.
- **NaN source on a cell centre** (`core/maps/idw.py:43-46`). A destination that coincides
  with a source puts all its weight there. If that source is NaN, the cell is NaN even
  though other sources are in range.
- **The GMZ map does not keep the link average** (`core/maps/gmz.py:91-100`).
  - The correction keeps each link's virtual-gauge mean of R^b, as in the paper and
    PyNNcml. The returned map, read back along the link, does not: in a toy case a 1 mm link
    reads 1.43 mm.
  - When virtual gauges sit on cell centres, GMZ equals line IDW.
- **Link on a cell edge** (`core/maps/geometry.py:92-101`, poligrain). A link lying exactly
  on a cell edge is counted in both cells, so its path average doubles. The fix belongs in
  poligrain.
- **Rain added at the edge** (`core/nowcast/advection.py:30`). `mode="nearest"` repeats edge
  values outward, which adds rain at the boundary when the flow comes in from outside.
- **Square pixels** (`core/nowcast/grid.py:57`). Pixels are square only at the centre
  latitude, about 0.15 % off over the 0.8 degree Gothenburg domain.
- **Rain floor and wet fraction** (`core/simulation/random_fields.py:163`). `to_rain` zeroes
  wet values under 0.1 mm/h after the transform. The requested war is therefore exact only
  when the wet marginal has little mass below 0.1. With gamma, mean 1 and CV 2, the realised
  war is 0.39 for a requested 0.5.
- **Wet-antenna growth cap** (`core/simulation/wet_antenna.py:82`). The growth step is
  capped at 1. This matches pycomlink whenever 3 dt < tau.
- **Unit guess for lengths** (`core/opensense/conventions.py:46`). With no units declared,
  a network whose median length is under 100 m is read as km.
- **PWS interval labels** (`core/opensense/pws_qc.py:54`). `regularize` uses start-labelled
  bins, while PWS amounts are stamped at interval end. The QC is consistent with itself, but
  the convention is undocumented.
- **Co-located stations** (`core/opensense/pws_qc.py:86`). The test `dist > 0` also drops a
  distinct station that shares a location from the neighbours.

Outside core/: `projects/maps/radar_adjustment/src/prepare.py:121` sums the city gauges with
`resample("15min", label="right")`, but the bins stay closed on the left. Check whether
that is inherited from the notebook.
