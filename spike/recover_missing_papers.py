"""
Feasibility spike, step 2a (data recovery): recover papers that are missing
from papers_raw.jsonl even though an author's paper list references them.

Root cause: the *old* (pre-perf-fix) checkpoint() function rewrote the
entire papers_checkpoint.jsonl from scratch on every checkpoint, which was
slow. When that run's process was killed (to fix the perf bug), it died
mid-write of that slow rewrite -- checkpoint.json (authors -> paper ID
lists, fast to write, written first) ended up ahead of
papers_checkpoint.jsonl (actual paper content, slow to write, truncated by
the kill). ~132k of the ~327k referenced paper IDs across ~8,600 authors
were affected.

Fix: we know the exact missing IDs (referenced by author_paper_map.json but
absent from papers_raw.jsonl) -- batch-refetch just those by ID directly
from OpenAlex (cheap `ids.openalex:` OR-filter, up to 50 per request) and
merge them into papers_raw.jsonl. No need to re-run the full pull.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pull_openalex import get, BASE, WORK_SELECT_FIELDS, MAX_WORKERS, reconstruct_abstract  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 50  # OpenAlex OR-filter practical batch size


def short_id(full_id):
    return full_id.rsplit("/", 1)[-1]


def build_rec(pid, p):
    """Mirror the exact transformation pull_openalex.py applies when writing
    papers_raw.jsonl -- papers_raw.jsonl stores this transformed record, NOT
    the raw OpenAlex work object, so recovered papers must match the same
    schema or downstream scripts (dedup, CS-filter) will break on them."""
    return {
        "id": pid,
        "title": p.get("title"),
        "abstract": reconstruct_abstract(p.get("abstract_inverted_index")),
        "year": p.get("publication_year"),
        "cited_by_count": p.get("cited_by_count"),
        "topics": [
            {
                "id": t["id"], "name": t["display_name"],
                "field": (t.get("field") or {}).get("display_name"),
                "subfield": (t.get("subfield") or {}).get("display_name"),
            }
            for t in (p.get("topics") or [])
        ],
        "referenced_works": p.get("referenced_works") or [],
        "author_ids": [
            au["author"]["id"]
            for au in (p.get("authorships") or [])
            if au.get("author")
        ],
        "author_orcids": {
            au["author"]["id"]: au["author"].get("orcid")
            for au in (p.get("authorships") or [])
            if au.get("author") and au["author"].get("orcid")
        },
    }


def fetch_batch(short_ids):
    filt = "ids.openalex:" + "|".join(short_ids)
    data = get(f"{BASE}/works", {
        "filter": filt,
        "per-page": len(short_ids),
        "select": WORK_SELECT_FIELDS,
    })
    return data.get("results", [])


def main():
    papers = {}
    with open(DATA_DIR / "papers_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    print(f"Existing papers in papers_raw.jsonl: {len(papers)}")

    author_paper_map = json.loads((DATA_DIR / "author_paper_map.json").read_text(encoding="utf-8"))
    referenced = set()
    for pids in author_paper_map.values():
        referenced.update(pids)
    missing = sorted(referenced - set(papers.keys()))
    print(f"Missing referenced papers to recover: {len(missing)}")

    batches = [missing[i:i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    print(f"Fetching in {len(batches)} batches of up to {BATCH_SIZE}...")

    recovered = {}
    truly_gone = 0
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_batch, [short_id(pid) for pid in b]): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                print(f"  batch failed: {e}")
                results = []
            for w in results:
                recovered[w["id"]] = build_rec(w["id"], w)
            done += 1
            if done % 100 == 0 or done == len(batches):
                print(f"  ...{done}/{len(batches)} batches done, {len(recovered)} papers recovered so far")

    still_missing = [pid for pid in missing if pid not in recovered]
    truly_gone = len(still_missing)
    print(f"\nRecovered: {len(recovered)}/{len(missing)}")
    print(f"Still missing (deleted/merged on OpenAlex's side since original fetch, or ID mismatch): {truly_gone}")
    if still_missing[:5]:
        print(f"  sample still-missing IDs: {still_missing[:5]}")

    # Merge recovered papers into papers_raw.jsonl (append-only; safe to re-run).
    papers.update(recovered)
    with open(DATA_DIR / "papers_raw.jsonl", "w", encoding="utf-8") as f:
        for p in papers.values():
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(papers)} total papers back to papers_raw.jsonl "
          f"(was {len(papers) - len(recovered)}, +{len(recovered)}).")


if __name__ == "__main__":
    main()
