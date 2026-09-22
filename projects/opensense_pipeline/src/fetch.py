"""
Fetch raw open datasets from Zenodo.

Downloads are resumable (HTTP Range), verified against the checksum Zenodo
publishes, and idempotent - an already-complete, verified file is skipped. Raw
archives land under ``dataset/open_datasets/<dataset>/raw/`` and are excluded
from git by the root ``.gitignore`` (``*.zip``, ``*.tar``, ``*.nc``).

    python -m fetch --dataset openmrg
    python -m fetch --dataset openrainer --files CML.tar AWS.tar RADrain.tar
    python -m fetch --list
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "dataset" / "open_datasets"

CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Source:
    """A Zenodo record backing one of our dataset folders."""

    name: str
    record_id: str          # the concept/latest record id
    folder: str             # folder under dataset/open_datasets/
    license: str
    doi: str
    default_files: tuple    # files to fetch when --files is not given


SOURCES = {
    "openmrg": Source(
        name="OpenMRG",
        record_id="7107689",
        folder="OpenMRG_Sweden",
        license="CC-BY-SA-4.0",
        doi="10.5281/zenodo.7107689",
        default_files=("OpenMRG.zip",),
    ),
    "openrainer": Source(
        name="OpenRainER",
        record_id="22829808",
        folder="OpenRainER_Italy",
        license="CC-BY-4.0",
        doi="10.5281/zenodo.22829808",
        # The three radar archives are ~1.6 GB each; RADrain (15-min accumulated
        # rain maps) is the one the merge stage needs. The other two are
        # reflectivity and an adjusted product, fetched only on request.
        default_files=("README.txt", "AWS.tar", "CML.tar", "RADrain.tar"),
    ),
}


def list_record(source: Source) -> list[dict]:
    """Return Zenodo's file listing for a record."""
    r = requests.get(
        f"https://zenodo.org/api/records/{source.record_id}", timeout=60
    )
    r.raise_for_status()
    payload = r.json()
    return [
        {
            "key": f["key"],
            "size": f["size"],
            "checksum": f.get("checksum", ""),
            "url": f["links"]["self"],
        }
        for f in payload.get("files", [])
    ]


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - Zenodo publishes md5, not our choice
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, checksum: str, size: int) -> bool:
    """True when ``path`` matches the published size and checksum."""
    if not path.exists() or path.stat().st_size != size:
        return False
    if not checksum:
        return True
    algo, _, digest = checksum.partition(":")
    if algo != "md5":
        return True  # nothing else is published today; size check stands
    return _md5(path) == digest


def download(entry: dict, dest_dir: Path, force: bool = False) -> Path:
    """Download one file with resume support. Returns the local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["key"]
    size = entry["size"]

    if not force and verify(dest, entry["checksum"], size):
        print(f"  [skip]  {entry['key']}  ({size/1e6:.1f} MB, already verified)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() and not force else 0
    if force and part.exists():
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    if have:
        print(f"  [resume] {entry['key']} from {have/1e6:.1f} MB")
    else:
        print(f"  [get]   {entry['key']}  ({size/1e6:.1f} MB)")

    with requests.get(entry["url"], stream=True, timeout=120,
                      headers=headers) as r:
        r.raise_for_status()
        done = have
        last_pct = -5
        with part.open(mode) as fh:
            for block in r.iter_content(CHUNK):
                fh.write(block)
                done += len(block)
                pct = int(done * 100 / size) if size else 0
                if pct >= last_pct + 5:
                    print(f"          {pct:3d}%  {done/1e6:8.1f} / "
                          f"{size/1e6:.1f} MB", flush=True)
                    last_pct = pct

    part.replace(dest)
    if not verify(dest, entry["checksum"], size):
        raise RuntimeError(f"checksum mismatch for {entry['key']}")
    print(f"  [ok]    {entry['key']} verified")
    return dest


def fetch(dataset: str, files: list[str] | None = None,
          force: bool = False) -> list[Path]:
    source = SOURCES[dataset]
    entries = list_record(source)
    available = {e["key"]: e for e in entries}

    wanted = files or list(source.default_files)
    missing = [w for w in wanted if w not in available]
    if missing:
        raise SystemExit(
            f"not in record {source.record_id}: {missing}\n"
            f"available: {sorted(available)}"
        )

    dest_dir = DATA_ROOT / source.folder / "raw"
    print(f"{source.name}  (DOI {source.doi}, {source.license})")
    print(f"  -> {dest_dir}")
    return [download(available[w], dest_dir, force) for w in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(SOURCES), help="which dataset")
    ap.add_argument("--files", nargs="*", help="specific files (default: the "
                                               "dataset's default set)")
    ap.add_argument("--force", action="store_true", help="re-download even if "
                                                         "already verified")
    ap.add_argument("--list", action="store_true", help="list record contents "
                                                        "and exit")
    args = ap.parse_args()

    if args.list:
        for key, source in SOURCES.items():
            print(f"\n{source.name}  ({key})  DOI {source.doi}  {source.license}")
            for e in list_record(source):
                mark = "*" if e["key"] in source.default_files else " "
                print(f"  {mark} {e['size']/1e6:9.2f} MB  {e['key']}")
        print("\n* = fetched by default")
        return

    if not args.dataset:
        ap.error("--dataset is required unless --list is given")
    fetch(args.dataset, args.files, args.force)


if __name__ == "__main__":
    sys.exit(main())
