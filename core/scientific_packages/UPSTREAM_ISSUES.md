# Upstream issues - drafts, not yet filed

Problems found in OpenSense packages while building FieldSense, each with a
minimal reproduction and the workaround this repository uses. They are
drafts for the maintainers' trackers; filing them posts publicly, so that is
left to a person. Versions: pycomlink 0.6.0, pypwsqc 0.2.1, poligrain 0.3.1,
mergeplg 0.1.0 (PyPI) and `main` at dd380b1, NumPy 2.4.6, Python 3.11.

---

## pycomlink: `cnn_wd` returns all-NaN predictions under NumPy >= 2.4

**Repository:** OpenSenseAction/pycomlink

`pycomlink.processing.pytorch_utils.run_inference.cnn_wd` runs the model
correctly, then loses every prediction in `redistribute_results`. The model
returns an `(N, 1)` array, so each `pred_value` in the loop is a one-element
array, and

```python
pred_array[time_idx, cml_idx] = pred_value
```

raises `ValueError: setting an array element with a sequence` on NumPy 2.4
(deprecated since 1.25). The surrounding `except (IndexError, ValueError):
continue` swallows it, so the function returns a dataset whose
`predictions` are all NaN, with no warning.

Reproduction (any model; here the published Polz et al. 2020 one):

```python
import numpy as np
a = np.full((2, 2), np.nan)
a[0, 0] = np.array([0.5], dtype=np.float32)   # ValueError on NumPy 2.4
```

With `cnn_wd(url, tl_normed)` on one OpenMRG day, `predictions.notnull().mean()`
is 0.0; flattening the predictions before `redistribute_results` gives 0.88.

**Suggested fix:** `predictions = np.asarray(results["predictions"]).ravel()`
before the loop, and catch only the lookup errors the `try` is meant for.
The loop itself (`np.where` per prediction) could be replaced by
`ref_times.get_indexer(times)` / `ref_cml_ids.get_indexer(cml_ids)` and one
fancy-indexed assignment.

Two related notes: `batchify_windows` builds every 180-sample window as a
Python list - about 6 GB for 364 links over 8 days at 1 minute - and NaNs in
the input propagate to every window that touches them, since there is no gap
handling. FieldSense's `core.opensense.wet_dry.cnn` uses `get_model` with its
own chunked `sliding_window_view` windows and matches pycomlink's
predictions to 4e-8.

---

## pypwsqc: `so_filter` needs undocumented pre-created variables; docs and release disagree

**Repository:** OpenSenseAction/pypwsqc

1. `flagging.so_filter` writes into `ds_pws.so_flag` and
   `ds_pws.median_corr_nbrs` but never creates them, so a dataset prepared as
   for `fz_filter`/`hi_filter` fails with
   `AttributeError: 'Dataset' object has no attribute 'median_corr_nbrs'`.
   The docstring does not mention either variable. Suggested fix: initialise
   both (e.g. `xr.full_like(ds_pws.rainfall, -1)` and `np.nan`) at the top of
   the function.

2. The documentation notebooks call the filters with a different signature
   from the 0.2.1 release: `hi_filter(ds_pws, hi_thres_a, hi_thres_b, nint,
   n_stat, distance_matrix, max_distance)` and `so_filter(..., bias_corr=True,
   beta=beta, dbc=dbc)`, while 0.2.1's `hi_filter` takes no distance matrix
   (it reads `ds_pws.reference` and `ds_pws.nbrs_not_nan`, which the caller
   must build) and `so_filter` has no `bias_corr`. Either a release of `main`
   or docs pinned to the released API would resolve it.

3. Minor: the SO default window (8064 five-minute steps, 28 days) silently
   yields all `-1` on shorter records; a warning when
   `ds_pws.sizes["time"] <= evaluation_period` would help.

FieldSense's `core.opensense.pws_qc` builds the neighbour reference with
poligrain, initialises the SO variables, and skips SO on short records.

---

## poligrain: `get_closest_points_to_line` reads `length` in coordinate units

**Repository:** OpenSenseAction/poligrain

The candidate search radius is `length / 2 + max_distance` around each link
midpoint, with `length` taken from `ds_cmls.length`. With projected
coordinates in metres and `length` in km - common after unit normalization,
and the unit several OpenSense files use - the radius around a 4 km link is
1.002 km instead of 3 km, and gauges beside the far half of the path are
silently not found.

```python
links: one link from (0, 0) to (4000, 0) m, length = 4 (km)
gauges: one at (3900, 500) m, 500 m from the path
get_closest_points_to_line(links, gauges, max_distance=1000, n_closest=1)
# -> distance inf (not found); with length in metres it is found at 500 m
```

**Suggested fix:** compute the half length from the endpoint coordinates the
function already reads, or check `length.attrs["units"]` and convert.
FieldSense's `evaluation.closest_gauges` recomputes the length from the
projected endpoints (test: `tests/test_evaluation.py`).

---

## mergeplg: a release of `main`

**Repository:** OpenSenseAction/mergeplg

PyPI's 0.1.0 (March 2025) predates the constructor-plus-call API, the
`MergeRADOLAN` class, `c0_within` (nugget from link geometry) and the KED
memory work on `main`. The API change is breaking (`update()`/`adjust()` ->
constructor + `__call__`, `da_cml` -> `da_cmls`, `da_gauge` -> `da_gauges`,
`n_closest` -> `nnear`, IDW default `radolan` -> `standard`), so
downstream code cannot track `main` without pinning a
commit. A tagged release, with the renames in a changelog, would let users
move together.

**Bug on main with pandas 3:** `MergeRADOLAN` fails on float32 radar with
`TypeError: Invalid value '[...]' for dtype 'float32'` in
`radolan.processing.rh_to_rw` (`df_stations_t.loc[sensor_is_cml,
"radar_RB_rainfall"] = ...`): the station table inherits the radar's dtype
and is then assigned float64 values, a silent upcast in pandas 2 and an
error in pandas 3.0.6. Casting the inputs to float64 avoids it; creating the
column as float64 would fix it.

One behaviour worth documenting: `MergeRADOLAN.__call__` picks its starting
audit station at random by default (`start_index_in_relevant_stations=
"random"`), so repeated runs differ unless an index is passed.
