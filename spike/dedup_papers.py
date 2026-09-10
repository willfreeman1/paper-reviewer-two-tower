"""
Feasibility spike, step 2 (data cleaning): dedupe papers that are the same
underlying work indexed multiple times (preprint/published/conference
versions) under different OpenAlex IDs. Cheap, no API calls needed.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def normalize_title(title):
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def main():
    papers = []
    with open(DATA_DIR / "papers_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    # Group by normalized title; keep the one with the most complete data
    # (has abstract > longer abstract > has referenced_works) as canonical.
    groups = defaultdict(list)
    for p in papers:
        groups[normalize_title(p["title"])].append(p)

    id_to_canonical = {}
    canonical_papers = {}
    n_dup_groups = 0
    for norm_title, group in groups.items():
        if not norm_title:
            # empty/missing title -- treat each as its own paper, don't dedupe blindly
            for p in group:
                id_to_canonical[p["id"]] = p["id"]
                canonical_papers[p["id"]] = p
            continue
        if len(group) > 1:
            n_dup_groups += 1
        best = max(group, key=lambda p: (
            bool(p.get("abstract")),
            len(p.get("abstract") or ""),
            len(p.get("referenced_works") or []),
        ))
        for p in group:
            id_to_canonical[p["id"]] = best["id"]
        canonical_papers[best["id"]] = best

    author_paper_map = json.loads((DATA_DIR / "author_paper_map.json").read_text(encoding="utf-8"))
    clean_author_map = {}
    dropped_authors = 0
    for aid, pids in author_paper_map.items():
        canon_ids = []
        seen = set()
        for pid in pids:
            cid = id_to_canonical.get(pid, pid)
            if cid not in seen:
                seen.add(cid)
                canon_ids.append(cid)
        if len(canon_ids) >= 5:
            clean_author_map[aid] = canon_ids
        else:
            dropped_authors += 1

    with open(DATA_DIR / "papers_clean.jsonl", "w", encoding="utf-8") as f:
        for p in canonical_papers.values():
            f.write(json.dumps(p) + "\n")
    with open(DATA_DIR / "author_paper_map_clean.json", "w", encoding="utf-8") as f:
        json.dump(clean_author_map, f)

    print(f"Papers before dedup: {len(papers)}")
    print(f"Duplicate title-groups found: {n_dup_groups}")
    print(f"Papers after dedup: {len(canonical_papers)} "
          f"({len(papers) - len(canonical_papers)} removed, "
          f"{100 * (len(papers) - len(canonical_papers)) / len(papers):.1f}%)")
    print(f"Authors before: {len(author_paper_map)}")
    print(f"Authors after dedup + re-applying >=5 papers filter: {len(clean_author_map)} "
          f"({dropped_authors} dropped below threshold after dedup)")


if __name__ == "__main__":
    main()
