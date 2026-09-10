"""
Train Stage 1 + Stage 2 on each weighted SimCite key and score gold.

Does not overwrite vanilla two_tower.pt / reranker.txt.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)
SPIKE = Path(__file__).parent
DATA = SPIKE / "data"
TOWER = DATA / "tower"

VARIANTS = ("recency", "topic", "method", "all")


def run(args):
    print("\n>>>", " ".join(str(a) for a in args), flush=True)
    subprocess.run(args, check=True, cwd=str(ROOT))


def main():
    only = sys.argv[1:] or list(VARIANTS)
    for name in only:
        pairs = DATA / f"simcite_pairs_{name}.json"
        if not pairs.exists():
            raise SystemExit(f"Missing {pairs}; run build_simcite_weighted.py first")
        ckpt = TOWER / f"two_tower_{name}.pt"
        r_full = f"reranker_{name}.txt"
        r_nc = f"reranker_{name}_nocite.txt"
        run([
            str(PY), str(SPIKE / "train_two_tower.py"),
            "--pairs", str(pairs),
            "--ckpt", str(ckpt),
            "--eval_out", str(DATA / f"two_tower_eval_{name}.json"),
        ])
        run([
            str(PY), str(SPIKE / "train_reranker.py"),
            "--pairs", str(pairs),
            "--tower", str(ckpt),
            "--model_full", r_full,
            "--model_nocite", r_nc,
            "--eval_out", str(DATA / f"reranker_eval_{name}.json"),
        ])
        run([
            str(PY), str(SPIKE / "eval_reranker_gold.py"),
            "--tower", str(ckpt),
            "--reranker", str(TOWER / r_full),
            "--reranker_nocite", str(TOWER / r_nc),
            "--eval_out", str(DATA / f"reranker_gold_eval_{name}.json"),
        ])

    summary = {"vanilla_from_disk": {}, "variants": {}}
    old_tt = DATA / "two_tower_eval.json"
    old_rg = DATA / "reranker_gold_eval.json"
    if old_tt.exists():
        summary["vanilla_from_disk"]["stage1"] = json.loads(old_tt.read_text(encoding="utf-8"))
    if old_rg.exists():
        summary["vanilla_from_disk"]["stage2_gold"] = json.loads(old_rg.read_text(encoding="utf-8"))
    for name in VARIANTS:
        tt = DATA / f"two_tower_eval_{name}.json"
        rg = DATA / f"reranker_gold_eval_{name}.json"
        re_ = DATA / f"reranker_eval_{name}.json"
        if tt.exists() or rg.exists():
            summary["variants"][name] = {
                "stage1": json.loads(tt.read_text(encoding="utf-8")) if tt.exists() else None,
                "stage2_gold": json.loads(rg.read_text(encoding="utf-8")) if rg.exists() else None,
                "stage2_incatalog": json.loads(re_.read_text(encoding="utf-8")) if re_.exists() else None,
            }
    out = DATA / "simcite_ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
