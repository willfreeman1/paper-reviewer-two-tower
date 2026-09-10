"""
Keep Computer Science papers only.

The original pull discovered authors from CS works, then fetched each
author's full recent list with no field filter, so non-CS papers leaked in.
This trims the already-pulled files. No new API calls.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def is_cs_paper(p):
    topics = p.get("topics") or []
    if not topics:
        return False
    # Primary topic only. "Any topic tagged CS" lets in papers where CS is
    # a side tag (for example a marketing paper with one loosely related
    # AI subtopic).
    return topics[0].get("field") == "Computer Science"


def main():
    papers = []
    with open(DATA_DIR / "papers_clean.jsonl", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    cs_papers = {p["id"]: p for p in papers if is_cs_paper(p)}

    author_paper_map = json.loads(
        (DATA_DIR / "author_paper_map_clean.json").read_text(encoding="utf-8")
    )
    cs_author_map = {}
    dropped = 0
    for aid, pids in author_paper_map.items():
        kept = [pid for pid in pids if pid in cs_papers]
        if len(kept) >= 5:
            cs_author_map[aid] = kept
        else:
            dropped += 1

    with open(DATA_DIR / "papers_cs_only.jsonl", "w", encoding="utf-8") as f:
        for p in cs_papers.values():
            f.write(json.dumps(p) + "\n")
    with open(DATA_DIR / "author_paper_map_cs_only.json", "w", encoding="utf-8") as f:
        json.dump(cs_author_map, f)

    print(f"Papers before CS filter: {len(papers)}")
    print(f"Papers after CS filter (any topic tagged CS): {len(cs_papers)} "
          f"({100 * len(cs_papers) / len(papers):.1f}%)")
    print(f"Authors before: {len(author_paper_map)}")
    print(f"Authors after CS filter + re-applying >=5 papers: {len(cs_author_map)} "
          f"({dropped} dropped below threshold)")


if __name__ == "__main__":
    main()
