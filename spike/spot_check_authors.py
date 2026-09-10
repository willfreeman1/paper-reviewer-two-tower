"""Print a random sample of authors' paper titles for manual disambiguation review."""
import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent / "data"
random.seed(7)

papers = {}
with open(DATA_DIR / "papers_raw.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        papers[p["id"]] = p

author_paper_map = json.loads((DATA_DIR / "author_paper_map.json").read_text(encoding="utf-8"))
sample_authors = random.sample(list(author_paper_map.keys()), 20)

for aid in sample_authors:
    pids = author_paper_map[aid]
    print(f"\n=== Author {aid} ({len(pids)} papers) ===")
    for pid in pids[:8]:
        p = papers.get(pid, {})
        title = p.get("title") or "(no title)"
        year = p.get("year")
        fields = sorted({t["field"] for t in (p.get("topics") or []) if t.get("field")})
        print(f"  [{year}] {title}  -- fields: {fields}")
