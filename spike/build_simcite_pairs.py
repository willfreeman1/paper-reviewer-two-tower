"""
Phase 1: build SimCite ground-truth training pairs.

For each paper P with citations that land on other papers in our own corpus
(checked for viability in simcite_feasibility_check.py -- ~49% of papers
have >=1 usable citation), rank those citations by SPECTER2 embedding
similarity to P and take the top-min(10, available) as "similar cited
papers." The authors of those papers become positive (paper, author) pairs
for P under the SimCite definition -- an alternative to "Authors" (where
P's own authors are the positives).

Requires: spike/data/paper_embeddings_full.npy + paper_embedding_ids_full.json
(built by embed_full_corpus.py, run on GPU -- see work_record.md).

Output: spike/data/simcite_pairs.json -- {paper_id: [{"author_id":...,
"cited_paper_id":..., "similarity":...}, ...]}, one entry per paper that has
>=1 SimCite positive (roughly half the corpus; the other half simply won't
appear as a key -- callers/trainers should treat missing = no SimCite
signal for that paper, and either skip it for this objective or fall back
to Authors labels).
"""
import json
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
TOP_K = 10


def main():
    print("Loading papers, author map, and full-corpus embeddings...")
    papers = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    author_paper_map = json.loads((DATA_DIR / "author_paper_map_final.json").read_text(encoding="utf-8"))
    paper_to_authors = {}
    for aid, pids in author_paper_map.items():
        for pid in pids:
            paper_to_authors.setdefault(pid, []).append(aid)

    emb_ids = json.loads((DATA_DIR / "paper_embedding_ids_full.json").read_text(encoding="utf-8"))
    embs = np.load(DATA_DIR / "paper_embeddings_full.npy")
    assert len(emb_ids) == embs.shape[0], "embedding ids/array length mismatch"
    id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs_normed = (embs / norms).astype(np.float32)
    print(f"Loaded {len(papers)} papers, {len(author_paper_map)} authors, "
          f"{embs.shape[0]} embeddings.")

    simcite_pairs = {}
    n_no_refs = 0
    n_no_incorpus_ref = 0
    n_no_authored_ref = 0
    n_no_embedding = 0
    t0 = time.time()
    done = 0
    for pid, p in papers.items():
        done += 1
        if done % 20000 == 0:
            elapsed = time.time() - t0
            print(f"  ...{done}/{len(papers)} papers processed ({elapsed:.0f}s elapsed)")

        refs = p.get("referenced_works") or []
        if not refs:
            n_no_refs += 1
            continue
        # candidate cited papers: in our corpus, authored by a kept author,
        # and with an embedding available (should be ~all, but be defensive)
        candidates = [r for r in refs if r in papers]
        if not candidates:
            n_no_incorpus_ref += 1
            continue
        candidates = [r for r in candidates if r in paper_to_authors]
        if not candidates:
            n_no_authored_ref += 1
            continue
        candidates = [r for r in candidates if r in id_to_idx]
        if not candidates:
            n_no_embedding += 1
            continue

        if pid not in id_to_idx:
            n_no_embedding += 1
            continue
        qvec = embs_normed[id_to_idx[pid]]
        cand_idxs = [id_to_idx[r] for r in candidates]
        sims = embs_normed[cand_idxs] @ qvec

        order = np.argsort(-sims)[:TOP_K]
        top = [(candidates[i], float(sims[i])) for i in order]

        entries = []
        for cited_pid, sim in top:
            for aid in paper_to_authors[cited_pid]:
                entries.append({
                    "author_id": aid,
                    "cited_paper_id": cited_pid,
                    "similarity": sim,
                })
        simcite_pairs[pid] = entries

    with open(DATA_DIR / "simcite_pairs.json", "w", encoding="utf-8") as f:
        json.dump(simcite_pairs, f)

    n_total = len(papers)
    print(f"\n===== SUMMARY =====")
    print(f"Papers with no referenced_works at all: {n_no_refs} ({100*n_no_refs/n_total:.1f}%)")
    print(f"Papers with refs but none in-corpus: {n_no_incorpus_ref} ({100*n_no_incorpus_ref/n_total:.1f}%)")
    print(f"Papers with in-corpus refs but none authored by a kept author: "
          f"{n_no_authored_ref} ({100*n_no_authored_ref/n_total:.1f}%)")
    print(f"Papers dropped for missing embeddings: {n_no_embedding}")
    print(f"Papers WITH >=1 SimCite positive pair: {len(simcite_pairs)} "
          f"({100*len(simcite_pairs)/n_total:.1f}%)")
    n_author_pairs = sum(len(v) for v in simcite_pairs.values())
    n_distinct_authors = len({e["author_id"] for entries in simcite_pairs.values() for e in entries})
    print(f"Total (paper, author) SimCite positive pairs: {n_author_pairs}")
    print(f"Distinct authors appearing as a SimCite positive: {n_distinct_authors}")
    print(f"\nWrote spike/data/simcite_pairs.json")


if __name__ == "__main__":
    main()
