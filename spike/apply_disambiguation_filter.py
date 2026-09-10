"""
Feasibility spike, step 2c: apply the coherence-score disambiguation filter
decided on in disambiguation_check.py (drop authors below 0.5 topic
coherence -- likely merged/mixed identities) to produce the final author set
for training/eval.

Note: only the AUTHOR list is trimmed, not the papers pool. A paper by a
"merged identity" author is still a real CS paper -- it's just that specific
author profile's ground-truth reviewer/authorship signal that's unreliable,
so it's fine to keep those papers in the general candidate pool while
dropping them as a reviewer/query profile.
"""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COHERENCE_THRESHOLD = 0.5


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
    subfield_counts = Counter()
    paper_subfields = []
    for pid in pids:
        p = papers.get(pid)
        sf = subfields_for_paper(p) if p else set()
        paper_subfields.append(sf)
        subfield_counts.update(sf)
    if not subfield_counts:
        return None
    top_subfield, _ = subfield_counts.most_common(1)[0]
    n_touching_top = sum(1 for sf in paper_subfields if top_subfield in sf)
    return n_touching_top / len(pids)


def main():
    papers, author_paper_map = load_data()
    print(f"Loaded {len(papers)} CS papers, {len(author_paper_map)} authors.")

    kept = {}
    dropped = 0
    no_score = 0
    for aid, pids in author_paper_map.items():
        score = coherence_score(pids, papers)
        if score is None:
            no_score += 1
            continue
        if score >= COHERENCE_THRESHOLD:
            kept[aid] = pids
        else:
            dropped += 1

    with open(DATA_DIR / "author_paper_map_final.json", "w", encoding="utf-8") as f:
        json.dump(kept, f)

    # papers_final.jsonl: the same CS-only paper pool (unchanged) but written
    # under the "final" name for a stable, self-describing filename pair.
    with open(DATA_DIR / "papers_final.jsonl", "w", encoding="utf-8") as f:
        for p in papers.values():
            f.write(json.dumps(p) + "\n")

    print(f"\nCoherence threshold: {COHERENCE_THRESHOLD}")
    print(f"Authors dropped (coherence < {COHERENCE_THRESHOLD}): {dropped} "
          f"({100 * dropped / len(author_paper_map):.1f}%)")
    if no_score:
        print(f"Authors with no scoreable papers (no topics at all): {no_score}")
    print(f"Authors kept (final): {len(kept)}")
    papers_per_author = [len(v) for v in kept.values()]
    print(f"Avg papers/author (final): {sum(papers_per_author) / max(1, len(kept)):.1f}")
    print(f"Papers pool (unchanged, CS-only): {len(papers)}")
    print(f"\nWrote: author_paper_map_final.json, papers_final.jsonl")


if __name__ == "__main__":
    main()
