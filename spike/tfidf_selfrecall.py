"""
Same self-recall + hard-negative test as embed_and_eval.py, using word
overlap (TF-IDF) instead of SPECTER2. Same random seed so the test
authors, held-out papers, and candidate pool match.
"""
import json
import random

import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
N_TEST_AUTHORS = 200
N_HARD_NEG_QUERIES = 20
POOL_SIZE = 3000  # matched to embed_and_eval.py so both methods see the same-size pool

random.seed(SEED)
np.random.seed(SEED)


def load_data():
    papers = {}
    with open(DATA_DIR / "papers_cs_only.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    author_paper_map = json.loads((DATA_DIR / "author_paper_map_cs_only.json").read_text(encoding="utf-8"))
    return papers, author_paper_map


def main():
    papers, author_paper_map = load_data()

    eligible_authors = [aid for aid, pids in author_paper_map.items() if len(pids) >= 6]
    test_authors = random.sample(eligible_authors, min(N_TEST_AUTHORS, len(eligible_authors)))
    held_out_map = {}
    required_ids = set()
    for aid in test_authors:
        pids = author_paper_map[aid]
        held_out = random.choice(pids)
        held_out_map[aid] = held_out
        required_ids.update(pids)

    all_ids = list(papers.keys())
    remaining = [pid for pid in all_ids if pid not in required_ids]
    n_filler = max(0, POOL_SIZE - len(required_ids))
    filler = random.sample(remaining, min(n_filler, len(remaining)))
    pool_ids = list(required_ids) + filler
    papers_list = [papers[pid] for pid in pool_ids]
    print(f"Candidate pool: {len(papers_list)} papers (same selection as SPECTER2 run)")

    texts = [(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers_list]
    ids = pool_ids
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    print("Fitting TF-IDF...")
    vec = TfidfVectorizer(max_features=50000, stop_words="english")
    X = vec.fit_transform(texts)  # sparse matrix, rows are L2-normalizable via cosine_similarity

    print(f"Running self-recall test on {len(test_authors)} authors...")
    ranks = []
    for aid in test_authors:
        held_out = held_out_map[aid]
        profile_pids = [p for p in author_paper_map[aid] if p != held_out and p in id_to_idx]
        if not profile_pids:
            continue
        profile_idx = [id_to_idx[p] for p in profile_pids]
        profile_vec = X[profile_idx].mean(axis=0)
        profile_vec = np.asarray(profile_vec)
        sims = cosine_similarity(X, profile_vec).ravel()
        held_out_idx = id_to_idx[held_out]
        held_out_sim = sims[held_out_idx]
        rank = int((sims > held_out_sim).sum()) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    recall_at = {k: float((ranks <= k).mean()) for k in [1, 5, 10, 50, 100]}
    mrr = float((1.0 / ranks).mean())
    print(f"\n===== TF-IDF SELF-RECALL (n={len(ranks)}, pool={len(papers_list)}) =====")
    for k, v in recall_at.items():
        print(f"  recall@{k}: {v:.3f}")
    print(f"  MRR: {mrr:.3f}")
    print(f"  median rank: {int(np.median(ranks))}")

    # Hard-negative check
    field_index = {}
    for p in papers_list:
        for t in (p.get("topics") or []):
            field = t.get("field")
            if field:
                field_index.setdefault(field, set()).add(p["id"])

    query_papers = random.sample(papers_list, N_HARD_NEG_QUERIES)
    hard_neg_sims, random_neg_sims = [], []
    for qp in query_papers:
        qid = qp["id"]
        qidx = id_to_idx[qid]
        q_authors = set(qp.get("author_ids") or [])
        q_refs = set(qp.get("referenced_works") or [])
        q_fields = {t.get("field") for t in (qp.get("topics") or []) if t.get("field")}
        same_field_ids = set()
        for f in q_fields:
            same_field_ids |= field_index.get(f, set())
        candidates = [pid for pid in same_field_ids
                      if pid != qid and pid not in q_refs
                      and not (set(papers[pid].get("author_ids") or []) & q_authors)]
        if candidates:
            sampled = random.sample(candidates, min(10, len(candidates)))
            idxs = [id_to_idx[pid] for pid in sampled]
            sims = cosine_similarity(X[qidx], X[idxs]).ravel()
            hard_neg_sims.extend(sims.tolist())
        random_ids = [pid for pid in random.sample(ids, 11) if pid != qid][:10]
        idxs = [id_to_idx[pid] for pid in random_ids]
        sims = cosine_similarity(X[qidx], X[idxs]).ravel()
        random_neg_sims.extend(sims.tolist())

    hard_neg_sims = np.array(hard_neg_sims)
    random_neg_sims = np.array(random_neg_sims)
    print(f"\n===== TF-IDF HARD-NEGATIVE CHECK =====")
    print(f"Same-subfield sim: mean={hard_neg_sims.mean():.3f}, n={len(hard_neg_sims)}")
    print(f"Random sim:        mean={random_neg_sims.mean():.3f}, n={len(random_neg_sims)}")

    with open(DATA_DIR / "eval_results_tfidf.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_pool": len(papers_list),
            "n_test_authors": len(ranks),
            "recall_at_k": recall_at,
            "mrr": mrr,
            "median_rank": int(np.median(ranks)),
            "hard_neg_sim_mean": float(hard_neg_sims.mean()),
            "random_neg_sim_mean": float(random_neg_sims.mean()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
