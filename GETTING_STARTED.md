# Getting started with FieldSense

A guide for students joining the project: what it is, how to get your own
copy, and how to open your first dataset.

## What the project is

Rain gauges are sparse and weather radar is indirect. But the mobile-phone
network is everywhere, and its microwave links between towers (commercial
microwave links, CMLs) lose signal when it rains: the heavier the rain on the
path, the larger the loss. FieldSense turns that loss into rainfall, and
combines it with other sensors that already exist - personal weather stations
(PWS), rain gauges and radar.

The work follows one chain, and each project in `projects/` takes one step:

```
open data  ->  signal loss to rain rate  ->  rainfall map  ->  merged with radar  ->  forecast
examples/      cml_retrieval                 opensense_pipeline                       spatial_interpolation
openmesh_nyc   physics_ml
```

Two further projects ask *why* results look the way they do:
`rainfall_field_sim` (on simulated rain: is the error from the sensors or
from where the links are?) and `physics_ml` (can machine learning recover, or
beat, the physics?). The root [README](README.md) lists every project with
its starting notebook.

## 1. Fork and clone

You work in **your own fork**, and send work back through pull requests.

1. Sign in to GitHub and open <https://github.com/drorjac/FieldSense>.
2. Click **Fork** (top right). This creates `github.com/<your-username>/FieldSense`.
3. Clone your fork and connect it to the original repository:

```bash
git clone https://github.com/<your-username>/FieldSense.git
cd FieldSense
git remote add upstream https://github.com/drorjac/FieldSense.git
git remote -v        # origin = your fork, upstream = drorjac/FieldSense
```

4. Before starting new work, bring your copy up to date, and work on a branch:

```bash
git fetch upstream
git switch main && git merge upstream/main && git push origin main
git switch -c <your-name>/<topic>
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the rest: syncing, commit style,
and opening a pull request.

## 2. Install

Python 3.11 is what the project is developed and tested on.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[opensense,notebooks,dev]"
python -m pytest                     # offline, about 30 s; everything should pass
```

`-e` installs `core/` (the shared code) so that `from core... import` works
from any notebook. In PyCharm or VS Code, choose `.venv` as the interpreter or
notebook kernel.

### Import errors?

`No module named 'core'`, or `numpy.core.multiarray failed to import` from
`poligrain` or `netCDF4`, means the notebook runs in a different Python than
the `.venv` you installed into - usually an older environment, or one created
with access to the system's packages, whose numpy does not match.

1. Point the editor at the repository's `.venv`:
   - **PyCharm:** Settings → Project → Python Interpreter → Add Interpreter →
     Add Local Interpreter → Select existing → `<repo>/.venv/bin/python`
     (Windows: `<repo>\.venv\Scripts\python.exe`).
   - **VS Code:** Python: Select Interpreter, and in a notebook, Select Kernel.
2. Select the same interpreter as the notebook's kernel, and restart the kernel.
3. Check it from a terminal with `.venv` activated:

```bash
python -c "import core, poligrain, netCDF4; print(core.__file__)"
```

This should print a path inside your clone of FieldSense. If it fails, run
the `pip install` line above again inside `.venv`.

## 3. Open your first dataset: OpenMRG

**OpenMRG** is 364 microwave links, a weather radar and rain gauges over
Gothenburg, Sweden, in summer 2015, published by the Swedish weather service
(SMHI). It is the best place to start: dense, clean and well documented.

The quickest way in is the notebook: `examples/01_openmrg.ipynb`. Open it and
run all cells. The data downloads by itself on the first run.

The same, in your own code:

```python
from core.opensense import example_data

# one day of the 8-day example subset: links, radar and gauges
data = example_data.load("openmrg", "8d", time=slice("2015-07-25", "2015-07-25"))
cml, radar, gauges = data["cml"], data["radar"], data["gauge_municipal"]
print(cml)                               # an xarray Dataset

link = cml.sel(cml_id=10297)
(link.tsl - link.rsl).plot.line(x="time")    # signal loss in dB: it rises in rain
```

Or open the file directly with plain `xarray` after the first download:

```python
import xarray as xr
ds = xr.open_dataset(example_data.sample_dir("openmrg") / "openmrg_cml_8d.nc")   # ~/data/cml/openmrg/_sample_8d/
```

Read the notebook's section "The frequency trap" before using the raw file
this way: it stores frequency in MHz without saying so. `example_data.load()`
fixes this and similar problems for you, which is why it is the recommended
way in.

The other datasets work the same way:

| dataset | where | call | notebook |
|---|---|---|---|
| OpenMRG | Gothenburg, Sweden | `example_data.load("openmrg", "8d")` | `examples/01_openmrg.ipynb` |
| OpenRainER | Emilia-Romagna, Italy | `example_data.load("openrainer", "8d")` | `examples/02_openrainer.ipynb` |
| OpenMesh | New York City | `example_data.load("openmesh", "20d")` | `examples/03_openmesh.ipynb` |
| Amsterdam PWS | Amsterdam, Netherlands | `example_data.load("ams_pws", "full_period")` | `examples/04_ams_pws.ipynb` |

Start with `examples/00_overview.ipynb` for all four side by side.

## 4. Where the data comes from

There are two sizes of each dataset. Both download on request; **data files
are never committed to git** (`.nc`, `.zip`, `.tar` and the data folders are
in `.gitignore`).

**Example subsets** - a few days each, about 60 MB in all. Enough for
learning and for most experiments.

- Source: [OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data),
  published by the [OpenSense](https://opensenseaction.eu/) community.
- Fetched by `core/opensense/example_data.py` into `~/data/cml/<dataset>/_sample*/` (see [`DATA.md`](DATA.md)).
- `python -m core.opensense.example_data --list` shows them all.

**Full records** - months to years, hundreds of MB to several GB. Use them
when a few days is not enough.

- Fetched by `core/opensense/fetch.py` into `~/data/cml/<dataset>/_download/` (see [`DATA.md`](DATA.md)).
- `python -m core.opensense.fetch --list` shows every file and its size.
- For example, all of OpenMRG (a 318 MB zip): `python -m core.opensense.fetch --dataset openmrg`

| dataset | official source | license |
|---|---|---|
| OpenMRG | Andersson et al. (2022), [Zenodo 10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689); [OpenSense page](https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/) | CC BY-SA 4.0 |
| OpenRainER | Covi et al., [Zenodo 10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808) (v2.0.3); [OpenSense page](https://opensenseaction.eu/news/new-open-cml-dataset-from-italy-openrainer/) | CC BY 4.0 |
| OpenMesh (links) | Jacoby et al. (2026), [ESSD 18, 5817-5836](https://doi.org/10.5194/essd-18-5817-2026); data [Zenodo 10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692); code [drorjac/OpenMesh](https://github.com/drorjac/OpenMesh) | CC BY 4.0 |
| OpenMesh (PWS) | [Zenodo 10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) | CC BY-NC 4.0 (non-commercial) |
| Amsterdam PWS | de Vos et al. (2019), [GRL, doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731); data at [4TU](https://data.4tu.nl/articles/dataset/Rainfall_observations_datasets_from_Personal_Weather_Stations/12703250) | CC BY 4.0 |
| CML Netherlands | Overeem et al., [4TU 10.4121/be252844-b672-471e-8d69-27269a862ec1.v1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1.v1) - described in `dataset/`, no loader yet | see source |

**Cite the original dataset** in any report or paper that uses it.

The OpenSense software these notebooks use:
[`poligrain`](https://github.com/OpenSenseAction/poligrain) (plotting and
matching sensors), [`pypwsqc`](https://github.com/OpenSenseAction/pypwsqc)
(PWS quality control), [`mergeplg`](https://github.com/OpenSenseAction/mergeplg)
(merging with radar), [`pycomlink`](https://github.com/pycomlink/pycomlink)
(CML processing).

## 5. First exercises

1. Run `examples/01_openmrg.ipynb`. Pick a different day and a different link:
   does the signal loss rise when the radar above the link shows rain?
2. In `examples/02_openrainer.ipynb`, the radar is compared with gauges
   twice: raw and gauge-adjusted. Why is the adjusted one so much closer?
3. In `examples/03_openmesh.ipynb`, find the snow days in January 2024. Which
   sensors see them, and which do not?
4. Then choose a project from the [README](README.md) and run its starting
   notebook.

## Working rules

- Put your code in the project you work on (`projects/<name>/src/`); keep
  notebooks short and call into `src/`.
- Code that a second project needs moves to `core/`, with a test in `tests/`.
- Never commit data, credentials or large outputs.
- Run `python -m pytest` before opening a pull request.
