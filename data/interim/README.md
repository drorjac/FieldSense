# data/interim — local data override

Empty by default and git-ignored (except this file). Every input FieldSense reads is
looked up here first, then in the shared store `~/data/cml/` — see [`../../DATA.md`](../../DATA.md).

To use your own copy of a file, place it here under the same relative path as in
`~/data/cml/`, e.g. `data/interim/openmrg/cml/openmrg_cml_full.nc`.
