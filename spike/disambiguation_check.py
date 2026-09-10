"""
Author disambiguation check on the CS-only catalog.

1. Topic coherence for every author: share of papers in their most common
   subfield. A real person usually concentrates; a mix of unrelated
   subfields can mean a merged identity (or a genuinely broad researcher).
2. Print the worst-coherence authors plus a random sample for a manual look.
"""
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent / "data"
random.seed(7)


def load_data():
    papers = {}
    with open(DATA_DIR / "papers_cs_only.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    author_paper_map = json.loads((DATA_DIR / "author_paper_map_cs_only.json").read_text(encoding="utf-8"))
    return papers, author_paper_map


def subfields_for_paper(p):
    return {t.get("subfield") for t in (p.get("topics") or []) if t.get("subfield")}


def coherence_score(pids, papers):
    """Fraction of papers whose subfield set includes the author's single
    most common subfield. 1.0 = every paper touches their top subfield
    (very coherent); low = scattered across many unrelated subfields."""
    subfield_counts = Counter()
    paper_subfields = []
    for pid in pids:
        p = papers.get(pid)
        if not p:
            paper_subfields.append(set())
            continue
        sf = subfields_for_paper(p)
        paper_subfields.append(sf)
        subfield_counts.update(sf)
    if not subfield_counts:
        return None, None
    top_subfield, _ = subfield_counts.most_common(1)[0]
    n_touching_top = sum(1 for sf in paper_subfields if top_subfield in sf)
    return n_touching_top / len(pids), top_subfield


def print_author(aid, pids, papers, label):
    print(f"\n=== {label}: Author {aid} ({len(pids)} CS papers) ===")
    for pid in pids[:8]:
        p = papers.get(pid, {})
        title = p.get("title") or "(no title)"
        year = p.get("year")
        subfields = sorted(subfields_for_paper(p))
        print(f"  [{year}] {title}  -- subfields: {subfields}")


def main():
    papers, author_paper_map = load_data()
    print(f"Loaded {len(papers)} CS papers, {len(author_paper_map)} authors.\n")

    scores = {}
    for aid, pids in author_paper_map.items():
        score, top_subfield = coherence_score(pids, papers)
        if score is not None:
            scores[aid] = score

    vals = sorted(scores.values())
    n = len(vals)
    print("===== TOPIC COHERENCE DISTRIBUTION (all authors) =====")
    print(f"n = {n}")
    for pct in [5, 10, 25, 50, 75, 90, 100]:
        idx = min(n - 1, int(n * pct / 100))
        print(f"  {pct}th percentile: {vals[idx]:.2f}")
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        n_below = sum(1 for v in vals if v < thresh)
        print(f"  authors below {thresh:.1f} coherence: {n_below} ({100 * n_below / n:.1f}%)")

    # Worst-coherence authors (most likely problem cases)
    worst = sorted(scores.items(), key=lambda kv: kv[1])[:8]
    print("\n\n===== WORST-COHERENCE AUTHORS (manual read) =====")
    for aid, score in worst:
        print_author(aid, author_paper_map[aid], papers, f"coherence={score:.2f}")

    # Random baseline sample for comparison
    random_sample = random.sample(list(author_paper_map.keys()), 8)
    print("\n\n===== RANDOM BASELINE AUTHORS (manual read) =====")
    for aid in random_sample:
        score = scores.get(aid)
        print_author(aid, author_paper_map[aid], papers, f"coherence={score:.2f}" if score else "coherence=N/A")


if __name__ == "__main__":
    main()
