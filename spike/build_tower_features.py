"""
Build the two-tower feature tables (see tower_features.py for what each
column means). Reads papers_final.jsonl, author_paper_map_final.json, and
the existing SPECTER2 matrix. No GPU, no API calls.

Writes spike/data/tower/.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tower_features import (  # noqa: E402
    AS_OF_YEARS,
    DATA_DIR,
    DEFAULT_HALF_LIFE,
    TOWER_DIR,
    l2_normalize,
    recency_weights,
    weighted_mean,
)


def load_papers():
    papers = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    return papers


def build_vocabs(papers):
    subfields = sorted({p["topics"][0]["subfield"] for p in papers.values()})
    topics = sorted({p["topics"][0]["id"] for p in papers.values()})
    # 0 = unknown/padding for the embedding tables in the mixer later
    subfield_to_id = {name: i + 1 for i, name in enumerate(subfields)}
    topic_to_id = {tid: i + 1 for i, tid in enumerate(topics)}
    return subfields, topics, subfield_to_id, topic_to_id


def main():
    t0 = time.time()
    TOWER_DIR.mkdir(parents=True, exist_ok=True)
    half_life = DEFAULT_HALF_LIFE

    print("Loading papers + author map ...", flush=True)
    papers = load_papers()
    author_paper_map = json.loads(
        (DATA_DIR / "author_paper_map_final.json").read_text(encoding="utf-8")
    )
    print(f"  {len(papers)} papers, {len(author_paper_map)} reviewers", flush=True)

    print("Loading SPECTER2 id list + memory-mapping embeddings ...", flush=True)
    full_ids = json.loads((DATA_DIR / "paper_embedding_ids_full.json").read_text(encoding="utf-8"))
    full_id_to_row = {pid: i for i, pid in enumerate(full_ids)}
    full_emb = np.load(DATA_DIR / "paper_embeddings_full.npy", mmap_mode="r")

    paper_ids = sorted(papers.keys())
    missing_emb = [pid for pid in paper_ids if pid not in full_id_to_row]
    if missing_emb:
        raise SystemExit(f"{len(missing_emb)} papers have no SPECTER2 row, e.g. {missing_emb[:3]}")

    print(f"  slicing {len(paper_ids)} embedding rows out of {len(full_ids)} ...", flush=True)
    take = np.fromiter((full_id_to_row[pid] for pid in paper_ids), dtype=np.int64, count=len(paper_ids))
    paper_embeddings = np.array(full_emb[take], dtype=np.float32)
    # L2-normalize so a later unweighted mean is a mean of directions, not magnitudes
    norms = np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
    paper_embeddings = paper_embeddings / np.clip(norms, 1e-8, None)

    subfields, topics, subfield_to_id, topic_to_id = build_vocabs(papers)
    n = len(paper_ids)
    paper_year = np.zeros(n, dtype=np.int16)
    paper_subfield_id = np.zeros(n, dtype=np.int16)
    paper_topic_id = np.zeros(n, dtype=np.int16)
    paper_citations = np.zeros(n, dtype=np.int32)
    for i, pid in enumerate(paper_ids):
        p = papers[pid]
        paper_year[i] = int(p["year"])
        paper_subfield_id[i] = subfield_to_id[p["topics"][0]["subfield"]]
        paper_topic_id[i] = topic_to_id[p["topics"][0]["id"]]
        paper_citations[i] = int(p.get("cited_by_count") or 0)

    paper_id_to_row = {pid: i for i, pid in enumerate(paper_ids)}

    reviewer_ids = sorted(author_paper_map.keys())
    r = len(reviewer_ids)
    offsets = [0]
    paper_idx_chunks = []
    n_papers = np.zeros(r, dtype=np.int16)
    year_min = np.zeros(r, dtype=np.int16)
    year_max = np.zeros(r, dtype=np.int16)
    total_cite = np.zeros(r, dtype=np.int32)
    n_subfields = np.zeros(r, dtype=np.int16)
    top_subfield = np.zeros(r, dtype=np.int16)
    emb_flat = np.zeros((r, paper_embeddings.shape[1]), dtype=np.float32)
    emb_weighted = np.zeros((len(AS_OF_YEARS), r, paper_embeddings.shape[1]), dtype=np.float32)

    print(f"Pooling {r} reviewers (flat mean + half-life={half_life}y snapshots) ...", flush=True)
    skipped_empty = 0
    for j, rid in enumerate(reviewer_ids):
        rows = []
        for pid in author_paper_map[rid]:
            row = paper_id_to_row.get(pid)
            if row is not None:
                rows.append(row)
        if not rows:
            skipped_empty += 1
            offsets.append(offsets[-1])
            continue
        rows = np.array(rows, dtype=np.int32)
        paper_idx_chunks.append(rows)
        offsets.append(offsets[-1] + len(rows))

        years = paper_year[rows]
        embs = paper_embeddings[rows]
        n_papers[j] = len(rows)
        year_min[j] = int(years.min())
        year_max[j] = int(years.max())
        total_cite[j] = int(paper_citations[rows].sum())
        sf = paper_subfield_id[rows]
        n_subfields[j] = int(len(set(sf.tolist())))
        top_subfield[j] = int(Counter(sf.tolist()).most_common(1)[0][0])

        flat = embs.mean(axis=0)
        emb_flat[j] = l2_normalize(flat)
        for yi, as_of in enumerate(AS_OF_YEARS):
            w = recency_weights(years, as_of, half_life)
            pooled = weighted_mean(embs, w, fallback=flat)
            emb_weighted[yi, j] = l2_normalize(pooled)

        if (j + 1) % 2000 == 0 or j + 1 == r:
            print(f"  ...{j + 1}/{r}", flush=True)

    reviewer_paper_idx = (
        np.concatenate(paper_idx_chunks).astype(np.int32) if paper_idx_chunks else np.zeros(0, dtype=np.int32)
    )
    reviewer_offsets = np.array(offsets, dtype=np.int32)

    print("Writing spike/data/tower/ ...", flush=True)
    np.save(TOWER_DIR / "paper_embeddings.npy", paper_embeddings)
    np.save(TOWER_DIR / "paper_year.npy", paper_year)
    np.save(TOWER_DIR / "paper_subfield_id.npy", paper_subfield_id)
    np.save(TOWER_DIR / "paper_topic_id.npy", paper_topic_id)
    np.save(TOWER_DIR / "reviewer_offsets.npy", reviewer_offsets)
    np.save(TOWER_DIR / "reviewer_paper_idx.npy", reviewer_paper_idx)
    np.save(TOWER_DIR / "reviewer_n_papers.npy", n_papers)
    np.save(TOWER_DIR / "reviewer_year_min.npy", year_min)
    np.save(TOWER_DIR / "reviewer_year_max.npy", year_max)
    np.save(TOWER_DIR / "reviewer_total_citations.npy", total_cite)
    np.save(TOWER_DIR / "reviewer_n_subfields.npy", n_subfields)
    np.save(TOWER_DIR / "reviewer_top_subfield_id.npy", top_subfield)
    np.save(TOWER_DIR / "reviewer_emb_flat.npy", emb_flat)
    np.save(TOWER_DIR / "reviewer_emb_weighted.npy", emb_weighted)
    (TOWER_DIR / "paper_ids.json").write_text(json.dumps(paper_ids), encoding="utf-8")
    (TOWER_DIR / "reviewer_ids.json").write_text(json.dumps(reviewer_ids), encoding="utf-8")
    (TOWER_DIR / "subfield_vocab.json").write_text(json.dumps(subfields, indent=2), encoding="utf-8")
    (TOWER_DIR / "topic_vocab.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")
    meta = {
        "n_papers": n,
        "n_reviewers": r,
        "embedding_dim": int(paper_embeddings.shape[1]),
        "n_subfields": len(subfields),
        "n_topics": len(topics),
        "half_life_years": half_life,
        "as_of_years": AS_OF_YEARS,
        "skipped_reviewers_with_no_papers": skipped_empty,
        "notes": {
            "paper_citations": "stored only to sum onto the reviewer (seniority). not a paper-tower input.",
            "time_weight": "weight=0.5**((T-year)/half_life); papers with year>T get 0.",
            "qwen_aspects": "not included; pairwise Stage-2 feature.",
        },
    }
    (TOWER_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Sanity: a reviewer with both old and new papers should move when T changes
    moved = 0
    checked = 0
    for j in range(r):
        if n_papers[j] < 4:
            continue
        if year_max[j] - year_min[j] < 4:
            continue
        a = emb_weighted[0, j]
        b = emb_weighted[-1, j]
        sim = float(np.dot(a, b))
        checked += 1
        if sim < 0.995:
            moved += 1
        if checked >= 500:
            break

    print(
        f"\nDone in {time.time() - t0:.1f}s. "
        f"{n} papers, {r} reviewers, {len(subfields)} subfields, {len(topics)} topics."
    )
    print(f"Wrote {TOWER_DIR}")
    if skipped_empty:
        print(f"WARNING: {skipped_empty} reviewers had zero in-corpus papers.")
    print(
        f"Sanity: of {checked} reviewers with a 4+ year span, "
        f"{moved} have 2015-weighted vs 2026-weighted fingerprints that are not almost identical "
        f"(dot < 0.995). If this is 0, time-weighting is a no-op."
    )


if __name__ == "__main__":
    main()
