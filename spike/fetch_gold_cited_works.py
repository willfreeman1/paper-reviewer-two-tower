"""
Fetch OpenAlex records (by ID, not title search) for:

  1. Every work cited by a gold paper we already resolved
  2. The gold papers themselves (so we get authorships for reviewer join)

Resume-safe JSONL. Free API. No GPU.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from pull_openalex import (  # noqa: E402
    BASE,
    EMAIL,
    RateLimiter,
    reconstruct_abstract,
    session,
)
from tower_features import DATA_DIR  # noqa: E402

# Uses the free polite pool (mailto) for ID lookups.
API_KEY = None
OUT_PATH = DATA_DIR / "gold_cited_works.jsonl"
SELECT = (
    "id,title,abstract_inverted_index,publication_year,cited_by_count,"
    "topics,authorships"
)
BATCH = 50
MAX_WORKERS = 12
TARGET_RPS = 15
rate_limiter = RateLimiter(TARGET_RPS)


def short_id(raw):
    if not raw:
        return ""
    return str(raw).rstrip("/").split("/")[-1]


def canon_id(raw):
    sid = short_id(raw)
    return f"https://openalex.org/{sid}" if sid else ""


def oa_get(url, params, max_retries=8):
    """Returns (payload_or_none, status). status is 200, 404, or None on retry exhaustion."""
    params = dict(params)
    if EMAIL:
        params["mailto"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    last_status = None
    for attempt in range(max_retries):
        rate_limiter.wait()
        r = session.get(url, params=params, timeout=45)
        last_status = r.status_code
        if r.status_code == 200:
            return r.json(), 200
        if r.status_code == 404:
            return None, 404
        if r.status_code == 429:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        time.sleep(min(20, 1.5 * (attempt + 1)))
    return None, last_status


def pack_work(w):
    topics = w.get("topics") or []
    subfields = []
    seen = set()
    for t in topics:
        sf = t.get("subfield") or {}
        name = sf.get("display_name") or sf.get("name") if isinstance(sf, dict) else sf
        if name and name not in seen:
            seen.add(name)
            subfields.append(name)
    authors = []
    for a in w.get("authorships") or []:
        au = a.get("author") or {}
        aid = au.get("id")
        if not aid:
            continue
        authors.append({"id": aid, "name": au.get("display_name") or ""})
    return {
        "id": w.get("id"),
        "title": w.get("title") or "",
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")) or "",
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count") or 0,
        "subfields": subfields,
        "author_ids": [a["id"] for a in authors],
        "authors": authors,
    }


def collect_ids():
    ids = set()
    gold_oa_path = DATA_DIR / "gold_openalex.jsonl"
    with open(gold_oa_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            oid = rec.get("openalex_id")
            if oid:
                ids.add(canon_id(oid))
            for ref in rec.get("referenced_works") or []:
                cid = canon_id(ref)
                if cid:
                    ids.add(cid)
    # Catalog papers we already have — still fetch if we want names,
    # but skip IDs we will fill from catalog below only when already written.
    return sorted(ids)


def catalog_pack_by_id(wanted=None):
    out = {}
    path = DATA_DIR / "papers_final.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pid = canon_id(p.get("id"))
            if not pid or (wanted is not None and pid not in wanted):
                continue
            topics = p.get("topics") or []
            subfields = []
            seen = set()
            for t in topics:
                sf = t.get("subfield")
                name = sf.get("display_name") or sf.get("name") if isinstance(sf, dict) else sf
                if name and name not in seen:
                    seen.add(name)
                    subfields.append(name)
            out[pid] = {
                "id": pid,
                "title": p.get("title") or "",
                "abstract": p.get("abstract") or "",
                "year": p.get("year"),
                "cited_by_count": p.get("cited_by_count") or 0,
                "subfields": subfields,
                "author_ids": p.get("author_ids") or [],
                "authors": [{"id": a, "name": ""} for a in (p.get("author_ids") or [])],
                "source": "catalog",
            }
    return out


def load_done():
    done = set()
    if not OUT_PATH.exists():
        return done
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") and not rec.get("missing"):
                done.add(canon_id(rec["id"]))
    return done


def fetch_batch(id_batch):
    shorts = [short_id(i) for i in id_batch if short_id(i)]
    if not shorts:
        return []
    filt = "openalex:" + "|".join(shorts)
    data, _ = oa_get(
        f"{BASE}/works",
        {"filter": filt, "per-page": min(len(shorts), 50), "select": SELECT},
    )
    results = (data or {}).get("results") or []
    packed = [pack_work(w) for w in results if w and w.get("id")]
    got = {canon_id(p["id"]) for p in packed}
    # Single-ID fallback: persist only confirmed 404s. Retry exhaustion is skipped.
    for raw in id_batch:
        cid = canon_id(raw)
        if cid in got:
            continue
        one, status = oa_get(f"{BASE}/works/{short_id(raw)}", {"select": SELECT})
        if one and one.get("id"):
            packed.append(pack_work(one))
        elif status == 404:
            packed.append({"id": cid, "missing": True, "reason": "404"})
    return packed


def main():
    print("Collecting OpenAlex IDs from gold_openalex.jsonl ...", flush=True)
    all_ids = collect_ids()
    print(f"  unique IDs {len(all_ids)}", flush=True)
    done = load_done()
    print(f"  already on disk {len(done)}", flush=True)

    # Seed catalog records we don't already have (saves API + gives abstracts).
    wanted = set(all_ids)
    catalog = catalog_pack_by_id(wanted)
    n_seed = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for pid, rec in catalog.items():
            if pid in done or pid not in wanted:
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(pid)
            n_seed += 1
        out_f.flush()
    print(f"  seeded from catalog {n_seed}", flush=True)

    todo = [i for i in all_ids if i not in done]
    print(f"  todo {len(todo)}", flush=True)
    if not todo:
        print("Nothing to fetch.")
        return

    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    t0 = time.time()
    n_ok = 0
    n_miss = 0
    lock = threading.Lock()
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_batch, b): b for b in batches}
            for i, fut in enumerate(as_completed(futs), 1):
                packed = fut.result() or []
                with lock:
                    for rec in packed:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if rec.get("missing"):
                            n_miss += 1
                        else:
                            n_ok += 1
                    out_f.flush()
                if i % 20 == 0 or i == len(batches):
                    elapsed = time.time() - t0
                    done_n = n_ok + n_miss
                    print(
                        f"  ...{i}/{len(batches)} batches  "
                        f"{done_n / max(elapsed, 1e-6):.1f} works/s  "
                        f"ok {n_ok}  missing {n_miss}",
                        flush=True,
                    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
