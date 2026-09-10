"""
Pull Computer Science authors and their papers from OpenAlex.

The Authors endpoint cannot filter by field, so this:

  1. Pulls CS works (topics.field.id, has_abstract) to collect author IDs.
  2. Samples those IDs as reviewer candidates.
  3. Fetches each candidate's full works list and keeps people with enough
     papers in the year window.
  4. Rebuilds abstracts from OpenAlex's inverted index and writes JSONL.
"""

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EMAIL = (os.environ.get("OPENALEX_MAILTO") or "").strip()
API_KEY = os.environ.get("OPENALEX_API_KEY")  # get one free at openalex.org/settings/api
BASE = "https://api.openalex.org"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

CS_FIELD_FILTER = "topics.field.id:fields/17"  # Computer Science
POOL_SIZE = 90000           # works pulled just to discover candidate author IDs
TARGET_AUTHORS = 20000
CANDIDATE_BUFFER = 55000    # try more than TARGET_AUTHORS since many will fail the >=MIN_WORKS filter
MIN_WORKS = 5
MAX_PAPERS_PER_AUTHOR = 25  # cap per-author profile size
YEAR_FROM = 2015
SEED = 42
MAX_PER_PAGE = 100          # OpenAlex page-size cap

WORK_SELECT_FIELDS = (
    "id,title,abstract_inverted_index,publication_year,cited_by_count,"
    "topics,referenced_works,authorships"
)

session = requests.Session()
# Raise the per-host pool so concurrent workers are not stuck at 10 connections.
_adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount("https://", _adapter)
session.mount("http://", _adapter)
random.seed(SEED)

# OpenAlex allows up to 100 req/s. Sequential calls are latency-bound (~4/s);
# overlap requests with a thread pool and a shared cap under that limit.
REQUEST_DELAY = 0.0
MAX_WORKERS = 20
TARGET_RPS = 70


class RateLimiter:
    """Thread-safe token-bucket-ish limiter: blocks callers as needed to keep
    the aggregate call rate at/under `rate_per_sec`, shared across threads."""

    def __init__(self, rate_per_sec):
        self.min_interval = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            start = max(now, self.next_slot)
            self.next_slot = start + self.min_interval
            sleep_for = start - now
        if sleep_for > 0:
            time.sleep(sleep_for)


rate_limiter = RateLimiter(TARGET_RPS)


def get(url, params, max_retries=8):
    params = dict(params)
    if EMAIL:
        params["mailto"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    for attempt in range(max_retries):
        rate_limiter.wait()
        r = session.get(url, params=params, timeout=30)
        if REQUEST_DELAY:
            time.sleep(REQUEST_DELAY)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        time.sleep(min(30, 1.5 * (attempt + 1)))
    r.raise_for_status()


def reconstruct_abstract(inv_index):
    """Rebuild plaintext abstract from OpenAlex's inverted-index format."""
    if not inv_index:
        return None
    positioned = []
    for word, idxs in inv_index.items():
        for idx in idxs:
            positioned.append((idx, word))
    positioned.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positioned)


def discover_candidate_author_ids(pool_size=POOL_SIZE, target_candidates=CANDIDATE_BUFFER):
    # OpenAlex "sample" caps at 10,000 IDs, so use cursor pagination instead.
    # Cursor order is not a true random sample.
    print(f"Scanning CS works (cursor-paginated) to discover candidate authors "
          f"(target: {target_candidates} unique IDs, scanning up to {pool_size} works)...")
    author_ids = set()
    cursor = "*"
    scanned = 0
    while cursor and scanned < pool_size and len(author_ids) < target_candidates:
        data = get(f"{BASE}/works", {
            "filter": f"{CS_FIELD_FILTER},has_abstract:true,publication_year:>{YEAR_FROM - 1}",
            "per-page": MAX_PER_PAGE,
            "cursor": cursor,
            "select": "id,authorships",
        })
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            for au in (w.get("authorships") or []):
                a = au.get("author") or {}
                if a.get("id"):
                    author_ids.add(a["id"])
        scanned += len(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if scanned % 5000 == 0:
            print(f"  ...scanned {scanned} works, {len(author_ids)} unique candidate IDs so far")
    author_ids = list(author_ids)
    random.shuffle(author_ids)
    print(f"  -> discovered {len(author_ids)} distinct candidate author IDs (scanned {scanned} works)")
    return author_ids


def fetch_author_works(author_id, max_papers=MAX_PAPERS_PER_AUTHOR):
    # Same CS field filter as discovery, so each author's list stays in-field.
    works = []
    cursor = "*"
    filt = f"author.id:{author_id},{CS_FIELD_FILTER},publication_year:>{YEAR_FROM - 1}"
    while cursor and len(works) < max_papers:
        data = get(f"{BASE}/works", {
            "filter": filt,
            "sort": "publication_year:desc",
            "per-page": min(200, max_papers),
            "cursor": cursor,
            "select": WORK_SELECT_FIELDS,
        })
        results = data.get("results", [])
        works.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not results:
            break
    return works[:max_papers]


CANDIDATES_FILE = DATA_DIR / "candidate_author_ids.json"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
PAPERS_CHECKPOINT_FILE = DATA_DIR / "papers_checkpoint.jsonl"


def main():
    # Reuse the same candidate list across resumed runs so progress is stable.
    if CANDIDATES_FILE.exists():
        candidate_ids = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
        print(f"Reusing {len(candidate_ids)} candidate author IDs from disk.")
    else:
        candidate_ids = discover_candidate_author_ids()[:CANDIDATE_BUFFER]
        CANDIDATES_FILE.write_text(json.dumps(candidate_ids), encoding="utf-8")

    all_papers = {}
    author_paper_ids = {}
    processed_ids = set()
    written_paper_ids = set()  # papers already flushed to PAPERS_CHECKPOINT_FILE on disk

    # Resume from checkpoint if one exists.
    if CHECKPOINT_FILE.exists():
        ckpt = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        author_paper_ids = ckpt["author_paper_ids"]
        processed_ids = set(ckpt["processed_ids"])
        with open(PAPERS_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                all_papers[p["id"]] = p
                written_paper_ids.add(p["id"])
        print(f"Resuming: {len(processed_ids)} candidates already processed, "
              f"{len(author_paper_ids)} authors kept, {len(all_papers)} papers so far.")

    print(f"Fetching full works for {len(candidate_ids)} candidate authors "
          f"(need >= {MIN_WORKS} to keep), {MAX_WORKERS} concurrent workers, "
          f"~{TARGET_RPS} req/s cap...")
    kept = len(author_paper_ids)
    remaining = [aid for aid in candidate_ids if aid not in processed_ids]
    state_lock = threading.Lock()
    t_start = time.time()

    def checkpoint():
        # Append new papers only. Rewriting the whole papers file each
        # checkpoint gets slower as the catalog grows.
        CHECKPOINT_FILE.write_text(json.dumps({
            "author_paper_ids": author_paper_ids,
            "processed_ids": list(processed_ids),
        }), encoding="utf-8")
        new_ids = [pid for pid in all_papers if pid not in written_paper_ids]
        if new_ids:
            with open(PAPERS_CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                for pid in new_ids:
                    f.write(json.dumps(all_papers[pid]) + "\n")
            written_paper_ids.update(new_ids)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        submitted = 0
        remaining_iter = iter(remaining)
        stop = False

        # Prime the pool, then keep submitting one new task per completion
        # (rather than submitting all up front) so we can stop promptly once
        # TARGET_AUTHORS is reached instead of over-fetching.
        def submit_next():
            nonlocal submitted
            for aid in remaining_iter:
                fut = pool.submit(fetch_author_works, aid)
                futures[fut] = aid
                submitted += 1
                return True
            return False

        for _ in range(MAX_WORKERS):
            if not submit_next():
                break

        while futures:
            for fut in as_completed(list(futures.keys())):
                aid = futures.pop(fut)
                works = fut.result()
                with state_lock:
                    processed_ids.add(aid)
                    if len(works) >= MIN_WORKS:
                        author_paper_ids[aid] = [w["id"] for w in works]
                        for w in works:
                            if w["id"] not in all_papers:
                                all_papers[w["id"]] = w
                        kept += 1
                    n_processed = len(processed_ids)
                    if n_processed % 100 == 0:
                        rate = n_processed / (time.time() - t_start)
                        print(f"  ...{n_processed}/{len(candidate_ids)} candidates checked, "
                              f"{kept} kept (>= {MIN_WORKS} papers), "
                              f"{len(all_papers)} unique papers so far "
                              f"({rate:.1f} candidates/s)")
                    if n_processed % 500 == 0:
                        checkpoint()
                if kept >= TARGET_AUTHORS:
                    stop = True
                if not stop:
                    submit_next()
                break  # re-fetch futures.keys() list each loop since it mutates
            if stop:
                break

        # Drain/cancel any still-in-flight futures if we stopped early.
        for fut in list(futures.keys()):
            fut.cancel()

    checkpoint()
    print(f"Done. Kept {kept} authors, {len(all_papers)} unique papers "
          f"({len(processed_ids)} candidates checked in {time.time() - t_start:.0f}s).")

    n_with_abstract = 0
    with open(DATA_DIR / "papers_raw.jsonl", "w", encoding="utf-8") as f:
        for pid, p in all_papers.items():
            abstract = reconstruct_abstract(p.get("abstract_inverted_index"))
            if abstract:
                n_with_abstract += 1
            rec = {
                "id": pid,
                "title": p.get("title"),
                "abstract": abstract,
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
            f.write(json.dumps(rec) + "\n")

    with open(DATA_DIR / "author_paper_map.json", "w", encoding="utf-8") as f:
        json.dump(author_paper_ids, f)

    n_authors = len(author_paper_ids)
    n_papers = len(all_papers)
    papers_per_author = [len(v) for v in author_paper_ids.values()]
    avg_papers = sum(papers_per_author) / max(1, len(papers_per_author))
    n_with_topics = sum(1 for p in all_papers.values() if p.get("topics"))
    n_with_refs = sum(1 for p in all_papers.values() if p.get("referenced_works"))
    n_zero_cite = sum(1 for p in all_papers.values() if (p.get("cited_by_count") or 0) == 0)

    print("\n===== SUMMARY =====")
    print(f"Candidate authors checked:      {len(candidate_ids)}")
    print(f"Authors kept (>= {MIN_WORKS} papers):    {n_authors}")
    print(f"Unique papers pulled:           {n_papers}")
    print(f"Avg papers/author (kept):       {avg_papers:.1f}")
    print(f"Abstract coverage:              {n_with_abstract}/{n_papers} "
          f"({100 * n_with_abstract / max(1, n_papers):.1f}%)")
    print(f"Topics present:                 {n_with_topics}/{n_papers} "
          f"({100 * n_with_topics / max(1, n_papers):.1f}%)")
    print(f"referenced_works present:       {n_with_refs}/{n_papers} "
          f"({100 * n_with_refs / max(1, n_papers):.1f}%)")
    print(f"Papers with 0 citations:        {n_zero_cite}/{n_papers} "
          f"({100 * n_zero_cite / max(1, n_papers):.1f}%)")

    with open(DATA_DIR / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_candidates_checked": len(candidate_ids),
            "n_authors": n_authors,
            "n_papers": n_papers,
            "avg_papers_per_author": avg_papers,
            "abstract_coverage": n_with_abstract / max(1, n_papers),
            "topic_coverage": n_with_topics / max(1, n_papers),
            "referenced_works_coverage": n_with_refs / max(1, n_papers),
            "zero_citation_fraction": n_zero_cite / max(1, n_papers),
        }, f, indent=2)


if __name__ == "__main__":
    main()
