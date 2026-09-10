"""Dump unique CMU + LR-Bench papers (title/abstract/year/ids) for enrichment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from tower_features import DATA_DIR  # noqa: E402
from train_two_tower import gold_text, paper_year_from_gold  # noqa: E402


def paper_title(p):
    return (p.get("title") or p.get("paper_title") or "").strip()


def paper_abstract(p):
    return (p.get("abstract") or "").strip()


def iter_raw_papers():
    for c in cmu.build_cases({}):
        for p in c["profile_papers"]:
            yield "cmu", p
        for p, _ in c["candidates"]:
            yield "cmu", p
    for c in lrb.load_cases():
        if c["kind"] == "pc":
            papers = [c["fixed_paper"]] + c["profile_a"] + c["profile_b"]
        else:
            papers = c["fixed_profile"] + [c["paper_a"], c["paper_b"]]
        for p in papers:
            yield "lrb", p


def unique_gold_papers():
    seen = {}
    for src, p in iter_raw_papers():
        key = gold_text(p)
        if not key.strip() or key in seen:
            if key in seen and src not in seen[key]["sources"]:
                seen[key]["sources"].append(src)
            continue
        seen[key] = {
            "id": hashlib.sha1(key.encode("utf-8")).hexdigest(),
            "text_key": key,
            "title": paper_title(p),
            "abstract": paper_abstract(p),
            "year": paper_year_from_gold(p),
            "arxiv_id": p.get("arXivId") or p.get("arxiv_id"),
            "ss_id": p.get("ssId") or p.get("paper_id"),
            "sources": [src],
        }
    return list(seen.values())


def main():
    papers = unique_gold_papers()
    out = DATA_DIR / "gold_papers.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for rec in papers:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_abs = sum(1 for p in papers if p["abstract"])
    n_year = sum(1 for p in papers if p["year"] is not None)
    print(f"Wrote {len(papers)} unique gold papers -> {out}")
    print(f"  with abstract={n_abs}  with year={n_year}")


if __name__ == "__main__":
    main()
