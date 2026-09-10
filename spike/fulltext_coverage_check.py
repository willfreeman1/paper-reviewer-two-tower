"""
Cheap coverage check: look up a sample of our papers by OpenAlex ID
(single-ID lookups are free) and count how many have a DOI, an arXiv ID,
open-access status, or a hosted PDF / parsed XML.
"""
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_SIZE = 1000
SEED = 7
MAX_WORKERS = 20
TARGET_RPS = 40

API_KEY = os.environ.get("OPENALEX_API_KEY")
SELECT = (
    "id,doi,ids,open_access,has_content,best_oa_location,"
    "primary_location,locations,type"
)

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)


class RateLimiter:
    def __init__(self, rate_per_sec):
        self.min_interval = 1.0 / rate_per_sec
        self.lock = __import__("threading").Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            wait = self.last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.last = now


limiter = RateLimiter(TARGET_RPS)


def load_sample():
    ids = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            ids.append(p["id"])
    random.seed(SEED)
    return random.sample(ids, SAMPLE_SIZE)


def fetch_work(openalex_id):
    short_id = openalex_id.rsplit("/", 1)[-1]
    url = f"https://api.openalex.org/works/{short_id}"
    params = {"select": SELECT}
    if API_KEY:
        params["api_key"] = API_KEY
    last_err = None
    for attempt in range(4):
        limiter.wait()
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    return {"id": openalex_id, "_error": str(last_err)}


def summarize(works):
    n = len(works)
    errors = [w for w in works if w.get("_error")]
    ok = [w for w in works if not w.get("_error")]

    def count(pred):
        return sum(1 for w in ok if pred(w))

    has_doi = count(lambda w: bool(w.get("doi") or (w.get("ids") or {}).get("doi")))
    has_arxiv = count(lambda w: bool((w.get("ids") or {}).get("arxiv")))
    is_oa = count(lambda w: bool((w.get("open_access") or {}).get("is_oa")))
    has_oa_url = count(lambda w: bool((w.get("open_access") or {}).get("oa_url")))
    has_content_pdf = count(lambda w: bool((w.get("has_content") or {}).get("pdf")))
    has_content_xml = count(lambda w: bool((w.get("has_content") or {}).get("grobid_xml")))
    best_oa_pdf = count(
        lambda w: bool(((w.get("best_oa_location") or {}).get("pdf_url")))
    )
    primary_pdf = count(
        lambda w: bool(((w.get("primary_location") or {}).get("pdf_url")))
    )

    any_usable = count(
        lambda w: bool(
            (w.get("ids") or {}).get("arxiv")
            or (w.get("has_content") or {}).get("pdf")
            or (w.get("has_content") or {}).get("grobid_xml")
            or ((w.get("best_oa_location") or {}).get("pdf_url"))
            or ((w.get("open_access") or {}).get("oa_url"))
        )
    )

    print(f"\nSampled {n} papers from papers_final.jsonl (seed={SEED})")
    print(f"Successful lookups: {len(ok)}  Errors: {len(errors)}")
    if ok:
        print("\nAmong successful lookups:")
        rows = [
            ("Has a DOI (needed to match other databases)", has_doi),
            ("Has an arXiv ID (free PDF/source download)", has_arxiv),
            ("Marked open-access", is_oa),
            ("Has an open-access URL", has_oa_url),
            ("OpenAlex hosts a PDF (paid $0.01 each if we use them)", has_content_pdf),
            ("OpenAlex hosts parsed XML (same paid download)", has_content_xml),
            ("Best open-access location has a PDF link (often free elsewhere)", best_oa_pdf),
            ("Primary location has a PDF link", primary_pdf),
            ("ANY usable full-text handle (arXiv OR hosted PDF OR OA URL)", any_usable),
        ]
        for label, c in rows:
            print(f"  {100 * c / len(ok):5.1f}%  ({c}/{len(ok)})  {label}")

        # OA status breakdown
        statuses = {}
        for w in ok:
            st = (w.get("open_access") or {}).get("oa_status") or "unknown"
            statuses[st] = statuses.get(st, 0) + 1
        print("\nOpen-access status breakdown:")
        for st, c in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"  {st}: {c} ({100 * c / len(ok):.1f}%)")

    out = DATA_DIR / "fulltext_coverage_sample.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(works, f)
    print(f"\nWrote raw sample to {out}")


def main():
    if not API_KEY:
        print("WARNING: OPENALEX_API_KEY not set")
    sample = load_sample()
    print(f"Looking up {len(sample)} papers by ID (free single-record lookups)...")
    works = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_work, pid): pid for pid in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            works.append(fut.result())
            if i % 200 == 0:
                print(f"  ...{i}/{len(sample)} ({time.time() - start:.0f}s)")
    print(f"Finished in {time.time() - start:.0f}s")
    summarize(works)


if __name__ == "__main__":
    main()
