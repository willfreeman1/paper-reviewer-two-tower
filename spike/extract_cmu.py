"""
Unpack the CMU gold zip so the scorer finds:

  spike/data/gold_cmu/data/evaluations.csv

Put any of these in spike/data/ then run this script:

  gold_cmu_data.zip
  goldstandard-reviewer-paper-match-main.zip
  (or another .zip from the CMU gold GitHub repo)

The GitHub zip has an extra top folder; this script looks for
evaluations.csv and copies that `data/` folder into place.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"
OUT = ROOT / "gold_cmu"
EVAL_REL = Path("data") / "evaluations.csv"
ZIP_NAMES = (
    "gold_cmu_data.zip",
    "goldstandard-reviewer-paper-match-main.zip",
    "goldstandard-reviewer-paper-match.zip",
)


def find_zip() -> Path:
    for name in ZIP_NAMES:
        p = ROOT / name
        if p.exists():
            return p
    zips = sorted(ROOT.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    names = ", ".join(ZIP_NAMES)
    raise FileNotFoundError(
        f"No CMU zip in {ROOT}. Download the survey zip from "
        f"https://github.com/niharshah/goldstandard-reviewer-paper-match "
        f"and save it as one of: {names}"
    )


def find_data_dir(tree: Path) -> Path:
    hits = list(tree.rglob("evaluations.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No evaluations.csv inside the zip. Expected a `data/` folder "
            f"from the CMU gold repo."
        )
    hits.sort(key=lambda p: len(p.parts))
    return hits[0].parent


def already_unpacked() -> bool:
    return (OUT / EVAL_REL).exists()


def main() -> None:
    if already_unpacked():
        print(f"already in place: {OUT / EVAL_REL}")
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = find_zip()
    scratch = ROOT / "_cmu_extract_tmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(scratch)
        src_data = find_data_dir(scratch)
        dest_data = OUT / "data"
        if dest_data.exists():
            shutil.rmtree(dest_data)
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_data, dest_data)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)

    if not already_unpacked():
        print(f"unpack finished but {OUT / EVAL_REL} is missing", file=sys.stderr)
        sys.exit(1)
    print(f"extracted {zip_path.name} -> {OUT / EVAL_REL}")


if __name__ == "__main__":
    main()
