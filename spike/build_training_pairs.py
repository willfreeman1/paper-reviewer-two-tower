"""
Phase 1: build the actual (anchor_paper, positive_paper) training pairs for
both ground-truth variants. Both use the SAME pair FORMAT (two paper texts
that should end up close together) so the exact same training script can
run either one -- only the pair SOURCE differs, which is the whole point of
the ablation.

Authors variant: anchor and positive are two different papers by the same
author. Straightforward -- this is just "co-authorship implies similarity."

SimCite variant: anchor = paper P; positive = a DIFFERENT paper (NOT the
cited paper itself) by an author who wrote one of P's top-similar cited
papers. Important: we deliberately do NOT use (P, cited_paper) as the pair
-- that pairing was chosen BY embedding similarity in build_simcite_pairs.py,
so training on it directly would be circular (telling the model to move
closer two things it already thinks are close, using the same frozen model
that did the choosing -- zero new signal). Using a DIFFERENT paper by that
same author gives a genuinely new, non-circular training signal: "this
paper is relevant to this author's broader body of work," not just to the
one paper we already flagged as similar.

Caps pairs per author to avoid a few prolific authors dominating the set,
and caps total pairs per variant so training stays fast (this phase is
meant to be cheap/fast -- see BRIEF.md "v1 scope").
"""
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
MAX_PAIRS_PER_AUTHOR = 6   # cap so prolific authors don't dominate
MAX_TOTAL_PAIRS = 60000    # cap total per variant -- keeps training fast

random.seed(SEED)


def build_authors_pairs(author_paper_map):
    pairs = []
    for aid, pids in author_paper_map.items():
        if len(pids) < 2:
            continue
        n = min(MAX_PAIRS_PER_AUTHOR, len(pids) * (len(pids) - 1))
        seen = set()
        attempts = 0
        while len(seen) < n and attempts < n * 4:
            attempts += 1
            i, j = random.sample(pids, 2)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j))
    random.shuffle(pairs)
    return pairs[:MAX_TOTAL_PAIRS]


def build_simcite_pairs(simcite_pairs, author_paper_map):
    # For each paper P, collect distinct positive authors (dedup across
    # multiple cited-paper/similarity entries pointing at the same author).
    pairs = []
    per_author_count = defaultdict(int)
    items = list(simcite_pairs.items())
    random.shuffle(items)
    for pid, entries in items:
        seen_authors = set()
        for e in entries:
            aid = e["author_id"]
            if aid in seen_authors:
                continue
            seen_authors.add(aid)
            author_pids = author_paper_map.get(aid, [])
            candidates = [p for p in author_pids if p != e["cited_paper_id"] and p != pid]
            if not candidates:
                continue
            if per_author_count[aid] >= MAX_PAIRS_PER_AUTHOR:
                continue
            positive = random.choice(candidates)
            pairs.append((pid, positive))
            per_author_count[aid] += 1
    random.shuffle(pairs)
    return pairs[:MAX_TOTAL_PAIRS]


def main():
    author_paper_map = json.loads((DATA_DIR / "author_paper_map_final.json").read_text(encoding="utf-8"))
    simcite_pairs = json.loads((DATA_DIR / "simcite_pairs.json").read_text(encoding="utf-8"))

    authors_pairs = build_authors_pairs(author_paper_map)
    simcite_training_pairs = build_simcite_pairs(simcite_pairs, author_paper_map)

    with open(DATA_DIR / "training_pairs_authors.jsonl", "w", encoding="utf-8") as f:
        for a, b in authors_pairs:
            f.write(json.dumps({"anchor_id": a, "positive_id": b}) + "\n")
    with open(DATA_DIR / "training_pairs_simcite.jsonl", "w", encoding="utf-8") as f:
        for a, b in simcite_training_pairs:
            f.write(json.dumps({"anchor_id": a, "positive_id": b}) + "\n")

    print(f"Authors variant: {len(authors_pairs)} training pairs "
          f"-> spike/data/training_pairs_authors.jsonl")
    print(f"SimCite variant: {len(simcite_training_pairs)} training pairs "
          f"-> spike/data/training_pairs_simcite.jsonl")

    # Sanity: overlap between the two pair sets (as unordered id-pairs) --
    # expect low overlap, confirming they're genuinely different signals.
    def norm(pairs):
        return {frozenset(p) for p in pairs}
    overlap = norm(authors_pairs) & norm(simcite_training_pairs)
    print(f"Overlap between the two pair sets (same paper-pair chosen by "
          f"both rules): {len(overlap)} "
          f"({100 * len(overlap) / max(1, min(len(authors_pairs), len(simcite_training_pairs))):.1f}% "
          f"of the smaller set)")


if __name__ == "__main__":
    main()
