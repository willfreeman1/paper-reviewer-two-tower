"""
Look up each gold paper in OpenAlex (free academic database) so Stage 2
can use the same extra facts we have on our catalog: field, topic, year,
citation count, and the list of works it cites.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from collect_gold_papers import unique_gold_papers  # noqa: E402
from pull_openalex import (  # noqa: E402
    BASE,
    EMAIL,
    RateLimiter,
    session,
)
from tower_features import DATA_DIR  # noqa: E402

API_KEY = os.environ.get("OPENALEX_API_KEY")
OUT_PATH = DATA_DIR / "gold_openalex.jsonl"
SELECT = "id,title,publication_year,cited_by_count,topics,referenced_works,ids"
MAX_WORKERS = 16
TARGET_RPS = 40
rate_limiter = RateLimiter(TARGET_RPS)


def norm_title(title):
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return " ".join(t.split())


def titles_match(a, b, min_ratio=0.92):
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 20 and (na in nb or nb in na):
        return True
    return SequenceMatcher(None, na, nb).ratio() >= min_ratio


def oa_get(url, params, max_retries=8):
    params = dict(params)
    if EMAIL:
        params["mailto"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    for attempt in range(max_retries):
        rate_limiter.wait()
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        time.sleep(min(20, 1.5 * (attempt + 1)))
    return None


def subfield_name(raw):
    if isinstance(raw, dict):
        return raw.get("display_name") or raw.get("name")
    return raw


def pack_work(w):
    topics = w.get("topics") or []
    primary = topics[0] if topics else {}
    return {
        "openalex_id": w.get("id"),
        "openalex_title": w.get("title"),
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count") or 0,
        "subfield": subfield_name(primary.get("subfield")),
        "topic_id": primary.get("id"),
        "referenced_works": w.get("referenced_works") or [],
    }


def catalog_by_title():
    out = {}
    path = DATA_DIR / "papers_final.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            nt = norm_title(p.get("title"))
            if not nt:
                continue
            topics = p.get("topics") or []
            primary = topics[0] if topics else {}
            out[nt] = {
                "openalex_id": p.get("id"),
                "openalex_title": p.get("title"),
                "year": p.get("year"),
                "cited_by_count": p.get("cited_by_count") or 0,
                "subfield": subfield_name(primary.get("subfield")),
                "topic_id": primary.get("id"),
                "referenced_works": p.get("referenced_works") or [],
                "match": "catalog_title",
            }
    return out


def lookup_arxiv(arxiv_id):
    if not arxiv_id:
        return None
    aid = str(arxiv_id).strip()
    aid = re.sub(r"v\d+$", "", aid)
    data = oa_get(f"{BASE}/works", {"filter": f"ids.arxiv:{aid}", "per-page": 1, "select": SELECT})
    results = (data or {}).get("results") or []
    return results[0] if results else None


def lookup_title(title):
    title = (title or "").strip()
    if len(title) < 8:
        return None
    # Avoid breaking OpenAlex filter syntax.
    q = re.sub(r"\s+", " ", title).strip()[:280]
    data = oa_get(
        f"{BASE}/works",
        {"search": q, "per-page": 5, "select": SELECT},
    )
    results = (data or {}).get("results") or []
    for w in results:
        if titles_match(title, w.get("title") or ""):
            return w
    return None


def resolve_one(rec, catalog):
    nt = norm_title(rec["title"])
    if nt and nt in catalog:
        packed = dict(catalog[nt])
        packed["id"] = rec["id"]
        packed["text_key"] = rec["text_key"]
        packed["title"] = rec["title"]
        return packed
    work = lookup_arxiv(rec.get("arxiv_id"))
    source = "arxiv" if work else None
    if work is None:
        work = lookup_title(rec.get("title"))
        source = "title_search" if work else None
    if work is None:
        return {
            "id": rec["id"],
            "text_key": rec["text_key"],
            "title": rec["title"],
            "openalex_id": None,
            "match": "none",
        }
    packed = pack_work(work)
    packed.update({"id": rec["id"], "text_key": rec["text_key"], "title": rec["title"], "match": source})
    return packed


def load_done():
    done = {}
    if not OUT_PATH.exists():
        return done
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id"):
                done[rec["id"]] = rec
    return done


def main():
    papers = unique_gold_papers()
    done = load_done()
    todo = [p for p in papers if p["id"] not in done]
    print(f"Gold papers {len(papers)}  already resolved {len(done)}  todo {len(todo)}", flush=True)
    print("Indexing catalog titles ...", flush=True)
    catalog = catalog_by_title()
    print(f"  {len(catalog)} catalog titles", flush=True)

    n_ok = sum(1 for r in done.values() if r.get("openalex_id"))
    t0 = time.time()
    lock = threading.Lock()
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        if not todo:
            print("Nothing to do.")
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = {ex.submit(resolve_one, rec, catalog): rec for rec in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    packed = fut.result()
                    with lock:
                        out_f.write(json.dumps(packed, ensure_ascii=False) + "\n")
                        out_f.flush()
                        if packed.get("openalex_id"):
                            n_ok += 1
                    if i % 200 == 0 or i == len(todo):
                        elapsed = time.time() - t0
                        print(
                            f"  ...{i}/{len(todo)}  {i / max(elapsed, 1e-6):.1f}/s  "
                            f"resolved {n_ok}/{len(done) + i}",
                            flush=True,
                        )
    print(f"Wrote {OUT_PATH}")
    # recount
    hits = {"catalog_title": 0, "arxiv": 0, "title_search": 0, "none": 0}
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            hits[rec.get("match") or "none"] = hits.get(rec.get("match") or "none", 0) + 1
    print("Match sources:", hits)


if __name__ == "__main__":
    main()
