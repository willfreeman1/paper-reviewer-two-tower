"""Unpack the CMU gold zip into spike/data/gold_cmu/."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent / "data"
zip_path = root / "gold_cmu_data.zip"
out = root / "gold_cmu"
zipfile.ZipFile(zip_path).extractall(out)
print(f"extracted {zip_path} -> {out}")
