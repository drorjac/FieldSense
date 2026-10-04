# Contributing

FieldSense uses a fork workflow. You pull updates from the main repository (`upstream`) into
your fork (`origin`), work on a branch, and send work back through a pull request.

```
upstream (drorjac/FieldSense) ──fetch/merge──► your clone ──push──► origin (your fork) ──pull request──► upstream
```

| direction | when | how often |
|---|---|---|
| pull: upstream to you | before new work; when `core/` or the docs change | often |
| push: you to upstream | to submit a project, a fix or a new shared module | occasionally |

## 1. Set up once

1. On GitHub, open <https://github.com/drorjac/FieldSense> and click **Fork**.
2. Clone your fork and add the main repository as `upstream`:

```bash
git clone https://github.com/<your-username>/FieldSense.git
cd FieldSense
git remote add upstream https://github.com/drorjac/FieldSense.git
git remote -v        # origin = your fork, upstream = drorjac/FieldSense
```

3. Install (Python 3.11):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[opensense,notebooks,dev]"           # core/ in editable mode
pip install -r projects/<your-project>/requirements.txt
python -m pytest                                      # everything should pass
```

`-e` makes `import core` work from any script, notebook or terminal.

## 2. Pull updates

```bash
git fetch upstream
git switch main
git merge upstream/main
git push origin main        # keep your fork's main in step
```

Then start each piece of work on its own branch:

```bash
git switch -c <your-name>/<topic>
```

## 3. Submit work

```bash
git add projects/<your-project>/
git commit -m "<project>: what changed and why"
git push -u origin <your-name>/<topic>
```

On GitHub, open a pull request from `<your-username>/FieldSense:<your-name>/<topic>` into
`drorjac/FieldSense:main`. In the description, say what the change does, how you checked it
(tests, notebooks executed, numbers reproduced) and anything a reviewer should look at first.

**Commit messages:** a short subject in the imperative, prefixed with the project or the
part of `core/` it touches (`cml_rnn: train on OpenRainER`, `core.maps: add block kriging`),
then a body if the reason is not obvious.

## Where to work

| folder | rule |
|---|---|
| `projects/<your-project>/` | yours: add and change anything |
| `core/` | shared by several projects: change through a pull request that keeps every caller working, with tests in `tests/` |
| other projects' folders | do not edit; open an issue or ask the owner |
| `dataset/`, `DATA.md` | documentation of the published data; data files are never committed |

`core/` is for code that more than one project imports. A helper only your project uses
belongs in `projects/<your-project>/src/`. If you need a module from another project, it is
shared by definition: propose moving it into `core/`. Never put another project's folder on
`sys.path`.

```python
from core.itu_p838 import get_k_alpha       # shared code: always through core
import my_module                            # your project's own src/
```

## Project layout

```
projects/<your-project>/
├── src/              # code; a run.py command line for anything that takes more than a minute
├── notebooks/        # short notebooks that call into src/ and show results
├── results/          # small outputs worth keeping: report.md, CSV tables, PNG figures
├── tests/            # fast tests on synthetic data, added to testpaths in pyproject.toml
├── paper/            # manuscript, if any
├── requirements.txt  # dependencies beyond the root install
└── README.md         # question, data, method, findings, how to run, caveats, references
```

A project is a topic and is self-contained: its code, notebooks, results and manuscript live
together. Input data is read through `core.data_paths` (see [DATA.md](DATA.md)) and never
copied into the repository; large intermediate files go under `dataset/open_datasets/`,
which is git-ignored.

Two habits keep results trustworthy:

- **Every number in a README, report or paper is computed by code in the repository**, and
  the README says which command produces it.
- **Scripts that start worker processes have an `if __name__ == "__main__":` guard.**
  Without it, each worker re-runs the script and can corrupt shared caches.

## Notebooks

Committed outputs bloat the repository and make diffs unreadable. Clear them before
committing:

```bash
jupyter nbconvert --clear-output --inplace projects/<your-project>/notebooks/<notebook>.ipynb
```

or install [nbstripout](https://github.com/kynan/nbstripout) once per clone
(`pip install nbstripout && nbstripout --install`). A figure that is a deliverable is saved
to `results/` as a PNG and linked from the README.

The exception is notebooks meant to be read on GitHub (`tutorials/`, `examples/`): they keep
their outputs, as long as each stays within about 1 MB (lower `figure.dpi` if it grows).

## Troubleshooting

**Merge conflict.** Edit the file, keep what is right, remove the `<<<<<<<`, `=======` and
`>>>>>>>` markers, then `git add <file>` and `git commit`. For a shared file you did not mean
to change, take upstream's version: `git checkout --theirs <file>`.

**Undo uncommitted changes to a file:** `git restore <file>`.

**Start over from upstream** (discards all local, uncommitted work):

```bash
git fetch upstream
git switch main
git reset --hard upstream/main
```

## Quick reference

| task | command |
|---|---|
| status | `git status` |
| update from upstream | `git fetch upstream && git merge upstream/main` |
| new branch | `git switch -c <your-name>/<topic>` |
| commit | `git add <files> && git commit -m "<project>: message"` |
| push a branch | `git push -u origin <your-name>/<topic>` |
| run the tests | `python -m pytest` |

Git basics: the [GitHub Git guides](https://github.com/git-guides).
