"""
Embed papers with off-the-shelf SPECTER2, then run self-recall@k on
held-out (author, paper) pairs and a hard-negative sanity check.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
N_TEST_AUTHORS = 200
N_HARD_NEG_QUERIES = 20
BATCH_SIZE = 32
MAX_LENGTH = 512
POOL_SIZE = 3000

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


def load_specter2():
    print("Loading SPECTER2 (base + proximity adapter)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return tokenizer, model


def embed_papers(tokenizer, model, papers_list, batch_size=BATCH_SIZE):
    """papers_list: list of paper dicts. Returns (ids, embeddings ndarray)."""
    ids = []
    texts = []
    for p in papers_list:
        title = p.get("title") or ""
        abstract = p.get("abstract") or ""
        text = title + tokenizer.sep_token + abstract if abstract else title
        ids.append(p["id"])
        texts.append(text)

    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False,
                                max_length=MAX_LENGTH)
            output = model(**inputs)
            emb = output.last_hidden_state[:, 0, :].numpy()
            all_embs.append(emb)
            batch_num = i // batch_size + 1
            if batch_num % 5 == 0 or batch_num == n_batches:
                elapsed = time.time() - t0
                rate = batch_num / elapsed
                eta = (n_batches - batch_num) / rate if rate > 0 else 0
                print(f"  batch {batch_num}/{n_batches}  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    embs = np.concatenate(all_embs, axis=0)
    return ids, embs


def main():
    papers, author_paper_map = load_data()
    print(f"Loaded {len(papers)} papers, {len(author_paper_map)} authors.")

    # --- Pick test authors + held-out papers FIRST, so we can guarantee the
    # candidate pool (sized for CPU-inference budget) still contains them ---
    eligible_authors = [aid for aid, pids in author_paper_map.items() if len(pids) >= 6]
    test_authors = random.sample(eligible_authors, min(N_TEST_AUTHORS, len(eligible_authors)))
    held_out_map = {}  # author_id -> held_out_paper_id
    required_ids = set()
    for aid in test_authors:
        pids = author_paper_map[aid]
        held_out = random.choice(pids)
        held_out_map[aid] = held_out
        required_ids.update(pids)  # held-out paper + all profile papers

    all_ids = list(papers.keys())
    remaining = [pid for pid in all_ids if pid not in required_ids]
    n_filler = max(0, POOL_SIZE - len(required_ids))
    filler = random.sample(remaining, min(n_filler, len(remaining)))
    pool_ids = list(required_ids) + filler
    papers_list = [papers[pid] for pid in pool_ids]
    print(f"Candidate pool: {len(papers_list)} papers "
          f"({len(required_ids)} required for test authors + {len(filler)} filler)")

    tokenizer, model = load_specter2()

    print(f"Embedding {len(papers_list)} papers (candidate pool)...")
    ids, embs = embed_papers(tokenizer, model, papers_list)
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    # normalize for cosine similarity via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs_normed = embs / norms

    np.save(DATA_DIR / "paper_embeddings.npy", embs)
    (DATA_DIR / "paper_embedding_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved embeddings: {embs.shape}")

    # --- Self-recall test ---
    print(f"\nRunning self-recall test on {len(test_authors)} held-out authors...")

    ranks = []
    for aid in test_authors:
        held_out = held_out_map[aid]
        profile_pids = [p for p in author_paper_map[aid] if p != held_out and p in id_to_idx]
        if not profile_pids:
            continue
        profile_idx = [id_to_idx[p] for p in profile_pids]
        profile_vec = embs_normed[profile_idx].mean(axis=0)
        profile_vec /= (np.linalg.norm(profile_vec) or 1)

        sims = embs_normed @ profile_vec  # cosine sim to every paper in the pool
        held_out_idx = id_to_idx[held_out]
        held_out_sim = sims[held_out_idx]
        # rank = number of papers strictly better + 1 (ties broken generously)
        rank = int((sims > held_out_sim).sum()) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    recall_at = {k: float((ranks <= k).mean()) for k in [1, 5, 10, 50, 100]}
    mrr = float((1.0 / ranks).mean())
    print(f"\n===== SELF-RECALL RESULTS (n={len(ranks)}, pool size={len(papers_list)}) =====")
    for k, v in recall_at.items():
        print(f"  recall@{k}: {v:.3f}")
    print(f"  MRR: {mrr:.3f}")
    print(f"  median rank: {int(np.median(ranks))}")

    # --- Hard-negative sanity check ---
    print(f"\nRunning hard-negative sanity check on {N_HARD_NEG_QUERIES} query papers...")
    # build field -> paper_ids index
    field_index = {}
    for p in papers_list:
        for t in (p.get("topics") or []):
            field = t.get("field")
            if field:
                field_index.setdefault(field, set()).add(p["id"])

    query_papers = random.sample(papers_list, N_HARD_NEG_QUERIES)
    hard_neg_sims = []
    random_neg_sims = []
    for qp in query_papers:
        qid = qp["id"]
        qidx = id_to_idx[qid]
        qvec = embs_normed[qidx]
        q_authors = set(qp.get("author_ids") or [])
        q_refs = set(qp.get("referenced_works") or [])
        q_fields = {t.get("field") for t in (qp.get("topics") or []) if t.get("field")}

        # same-field candidates NOT authored/cited/referenced by the query paper
        same_field_ids = set()
        for f in q_fields:
            same_field_ids |= field_index.get(f, set())
        candidates = [pid for pid in same_field_ids
                      if pid != qid and pid not in q_refs
                      and not (set(papers[pid].get("author_ids") or []) & q_authors)]
        if candidates:
            sample_n = min(10, len(candidates))
            sampled = random.sample(candidates, sample_n)
            idxs = [id_to_idx[pid] for pid in sampled]
            sims = embs_normed[idxs] @ qvec
            hard_neg_sims.extend(sims.tolist())

        # fully random negatives
        random_ids = random.sample(ids, 10)
        random_ids = [pid for pid in random_ids if pid != qid][:10]
        idxs = [id_to_idx[pid] for pid in random_ids]
        sims = embs_normed[idxs] @ qvec
        random_neg_sims.extend(sims.tolist())

    hard_neg_sims = np.array(hard_neg_sims)
    random_neg_sims = np.array(random_neg_sims)
    print(f"\n===== HARD-NEGATIVE CHECK =====")
    print(f"Same-subfield (not authored/cited) similarity: "
          f"mean={hard_neg_sims.mean():.3f}, median={np.median(hard_neg_sims):.3f}, "
          f"n={len(hard_neg_sims)}")
    print(f"Fully random paper similarity:                  "
          f"mean={random_neg_sims.mean():.3f}, median={np.median(random_neg_sims):.3f}, "
          f"n={len(random_neg_sims)}")
    print(f"Difference (higher = same-subfield genuinely harder to distinguish): "
          f"{hard_neg_sims.mean() - random_neg_sims.mean():.3f}")

    with open(DATA_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_pool": len(papers_list),
            "n_test_authors": len(ranks),
            "recall_at_k": recall_at,
            "mrr": mrr,
            "median_rank": int(np.median(ranks)),
            "hard_neg_sim_mean": float(hard_neg_sims.mean()),
            "hard_neg_sim_median": float(np.median(hard_neg_sims)),
            "random_neg_sim_mean": float(random_neg_sims.mean()),
            "random_neg_sim_median": float(np.median(random_neg_sims)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
