"""
Score Stage 3 (four-persona committee) on CMU + LR-Bench.

Trained-on: nothing — the language model is used as-is.
Scored-on: the same human pairs as Stage 1/2 gold eval.

Does not retrain the Stage 2 tree. Catalog-scale committee labels for
that tree would be thousands of dollars; we only buy votes on gold
until those votes beat frozen SPECTER2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import spearmanr

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from eval_reranker_gold import load_qwen_by_title, tags_for  # noqa: E402
from stage3_committee import (  # noqa: E402
    CACHE_PATH,
    MODEL,
    load_cache,
    score_pair,
)
from tower_features import DATA_DIR  # noqa: E402
from train_two_tower import gold_text  # noqa: E402


def union_tags(papers, title_to_sets, gold_by_text):
    out = {"topics": set(), "methodologies": set(), "applications": set()}
    for p in papers:
        t = tags_for(p, title_to_sets, gold_by_text)
        for k in out:
            out[k].update(t.get(k) or [])
    return {k: sorted(v) for k, v in out.items()}


def pack_tags(p, title_to_sets, gold_by_text):
    t = tags_for(p, title_to_sets, gold_by_text)
    return {k: sorted(t.get(k) or []) for k in ("topics", "methodologies", "applications")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N cases (smoke)")
    ap.add_argument("--skip_lrb", action="store_true")
    ap.add_argument("--skip_cmu", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in .env")

    client = OpenAI()
    cache = load_cache()
    print(f"Model {MODEL}  cache {len(cache)}  file {CACHE_PATH}", flush=True)
    title_to_sets, gold_by_text = load_qwen_by_title()

    out = {"model": MODEL, "cache": str(CACHE_PATH)}

    if not args.skip_cmu:
        cases = cmu.build_cases({})
        if args.limit:
            cases = cases[: args.limit]
        rhos = []
        n_pairs = 0
        t0 = time.time()
        for i, c in enumerate(cases, 1):
            scores, exps = [], []
            rev_tags = union_tags(c["profile_papers"], title_to_sets, gold_by_text)
            for p, exp in c["candidates"]:
                rec = score_pair(
                    client,
                    p,
                    c["profile_papers"],
                    pack_tags(p, title_to_sets, gold_by_text),
                    rev_tags,
                    cache,
                )
                sc = rec["scores"].get("mean")
                if sc is None:
                    continue
                scores.append(sc)
                exps.append(exp)
                n_pairs += 1
            if len(scores) >= 3 and len(set(exps)) >= 2:
                rho, _ = spearmanr(scores, exps)
                if not np.isnan(rho):
                    rhos.append(float(rho))
            if i % 5 == 0 or i == len(cases):
                print(
                    f"  CMU {i}/{len(cases)}  pairs {n_pairs}  "
                    f"mean Spearman {np.mean(rhos) if rhos else None:.3f}  "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
        out["cmu"] = {
            "mean_spearman": float(np.mean(rhos)) if rhos else None,
            "n": len(rhos),
            "n_pairs": n_pairs,
        }
        print("CMU", out["cmu"], flush=True)

    if not args.skip_lrb:
        cases = lrb.load_cases()
        if args.limit:
            cases = cases[: args.limit]
        correct = total = 0
        t0 = time.time()
        for i, c in enumerate(cases, 1):
            if c["kind"] == "pc":
                paper = c["fixed_paper"]
                rec_a = score_pair(
                    client,
                    paper,
                    c["profile_a"],
                    pack_tags(paper, title_to_sets, gold_by_text),
                    union_tags(c["profile_a"], title_to_sets, gold_by_text),
                    cache,
                )
                rec_b = score_pair(
                    client,
                    paper,
                    c["profile_b"],
                    pack_tags(paper, title_to_sets, gold_by_text),
                    union_tags(c["profile_b"], title_to_sets, gold_by_text),
                    cache,
                )
            else:
                rec_a = score_pair(
                    client,
                    c["paper_a"],
                    c["fixed_profile"],
                    pack_tags(c["paper_a"], title_to_sets, gold_by_text),
                    union_tags(c["fixed_profile"], title_to_sets, gold_by_text),
                    cache,
                )
                rec_b = score_pair(
                    client,
                    c["paper_b"],
                    c["fixed_profile"],
                    pack_tags(c["paper_b"], title_to_sets, gold_by_text),
                    union_tags(c["fixed_profile"], title_to_sets, gold_by_text),
                    cache,
                )
            sa, sb = rec_a["scores"].get("mean"), rec_b["scores"].get("mean")
            if sa is None or sb is None or sa == sb:
                continue
            total += 1
            if (sa > sb) == c["a_is_positive"]:
                correct += 1
            if i % 25 == 0 or i == len(cases):
                acc = correct / total if total else None
                print(
                    f"  LRB {i}/{len(cases)}  decided {total}  acc {acc}  "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
        out["lr_bench"] = {
            "pairwise_accuracy": (correct / total) if total else None,
            "n": total,
            "n_cases": len(cases),
        }
        print("LR-Bench", out["lr_bench"], flush=True)

    path = DATA_DIR / "stage3_gold_eval.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
