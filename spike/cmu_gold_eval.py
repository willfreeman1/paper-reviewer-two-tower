"""
Proxy-vs-gold check, part 2: run SPECTER2 and TF-IDF against the REAL
human-labeled CMU Gold Standard dataset (self-reported expertise ratings),
and compare the resulting leaderboard to the OpenAlex self-recall leaderboard.

For each participant: build a profile from their own papers (semantic
scholar profile), score their up to 10 rated candidate papers, and compute
the rank correlation (Spearman) between predicted similarity and their
self-reported expertise (1-5). Average across participants.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent / "data" / "gold_cmu" / "data"
OUT_DIR = Path(__file__).parent / "data"


def load_paper(pid, cache):
    if pid in cache:
        return cache[pid]
    fp = DATA_DIR / "papers" / f"{pid}.json"
    if not fp.exists():
        cache[pid] = None
        return None
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        cache[pid] = None
        return None
    cache[pid] = d
    return d


def load_participant_papers(participant_id, cache):
    fp = DATA_DIR / "participants" / f"{participant_id}.json"
    if not fp.exists():
        return []
    d = json.loads(fp.read_text(encoding="utf-8"))
    pids = [p["paperId"] for p in d.get("papers", [])]
    papers = [load_paper(pid, cache) for pid in pids]
    return [p for p in papers if p and p.get("abstract")]


def paper_text(p):
    return (p.get("title") or "") + " " + (p.get("abstract") or "")


def load_evaluations():
    rows = []
    with open(DATA_DIR / "evaluations.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def build_cases(cache):
    """Returns list of dicts: {participant_id, profile_papers, candidates:[(paper, expertise)]}"""
    rows = load_evaluations()
    cases = []
    for row in rows:
        pid = row["ParticipantID"]
        profile_papers = load_participant_papers(pid, cache)
        if len(profile_papers) < 2:
            continue
        candidates = []
        for i in range(1, 11):
            paper_id = row.get(f"Paper{i}")
            expertise = row.get(f"Expertise{i}")
            if not paper_id or not expertise:
                continue
            p = load_paper(paper_id, cache)
            if not p or not p.get("abstract"):
                continue
            candidates.append((p, float(expertise)))
        if len(candidates) >= 3 and len({e for _, e in candidates}) >= 2:
            cases.append({"participant_id": pid, "profile_papers": profile_papers, "candidates": candidates})
    return cases


def eval_tfidf(cases):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Fit one global vectorizer over everything for a shared vocabulary/IDF.
    all_texts = []
    for c in cases:
        all_texts.extend(paper_text(p) for p in c["profile_papers"])
        all_texts.extend(paper_text(p) for p, _ in c["candidates"])
    vec = TfidfVectorizer(max_features=50000, stop_words="english")
    vec.fit(all_texts)

    correlations = []
    for c in cases:
        profile_X = vec.transform([paper_text(p) for p in c["profile_papers"]])
        profile_vec = np.asarray(profile_X.mean(axis=0))
        cand_X = vec.transform([paper_text(p) for p, _ in c["candidates"]])
        sims = cosine_similarity(cand_X, profile_vec).ravel()
        expertise = [e for _, e in c["candidates"]]
        rho, _ = spearmanr(sims, expertise)
        if not np.isnan(rho):
            correlations.append(rho)
    return correlations


def tokenize(text):
    return text.lower().split()


def _build_global_bm25(cases):
    """Shared helper: one global BM25 index (dedup by paper id) over every
    paper that appears anywhere, so IDF statistics reflect a realistic-sized
    collection rather than each case's tiny candidate set."""
    from rank_bm25 import BM25Okapi

    id_to_pos = {}
    corpus_tokens = []

    def add_paper(p):
        pid = p.get("paperId") or p.get("id") or id(p)
        if pid not in id_to_pos:
            id_to_pos[pid] = len(corpus_tokens)
            corpus_tokens.append(tokenize(paper_text(p)))
        return id_to_pos[pid]

    print(f"Fitting BM25 over the shared paper corpus...")
    return add_paper, corpus_tokens, BM25Okapi


def eval_bm25(cases):
    """v1: query = ALL of a reviewer's papers concatenated into one giant
    query, scored against every candidate. This is the naive approach --
    kept for comparison against eval_bm25_maxpaper below, since a long,
    multi-topic query is a poor fit for BM25 (built for short queries)."""
    add_paper, corpus_tokens, BM25Okapi = _build_global_bm25(cases)

    case_positions = []
    for c in cases:
        profile_positions = [add_paper(p) for p in c["profile_papers"]]
        cand_positions = [(add_paper(p), e) for p, e in c["candidates"]]
        case_positions.append((profile_positions, cand_positions))

    print(f"  ({len(corpus_tokens)} unique papers)")
    bm25 = BM25Okapi(corpus_tokens)

    correlations = []
    for profile_positions, cand_positions in case_positions:
        query_tokens = [tok for pos in profile_positions for tok in corpus_tokens[pos]]
        scores = bm25.get_scores(query_tokens)
        sims = [scores[pos] for pos, _ in cand_positions]
        expertise = [e for _, e in cand_positions]
        rho, _ = spearmanr(sims, expertise)
        if not np.isnan(rho):
            correlations.append(rho)
    return correlations


def eval_bm25_maxpaper(cases):
    """v2 (fairer test, closer to how BM25 is actually meant to be used):
    for each CANDIDATE paper, use its own (short, coherent, single-paper)
    text as the query -- scored against the whole shared corpus -- then take
    the MAX score among the reviewer's own profile-paper positions. This
    answers "does the reviewer have at least one paper whose language
    closely echoes this candidate," instead of averaging a whole messy,
    multi-topic history into one query."""
    add_paper, corpus_tokens, BM25Okapi = _build_global_bm25(cases)

    case_positions = []
    for c in cases:
        profile_positions = [add_paper(p) for p in c["profile_papers"]]
        cand_positions = [(add_paper(p), e) for p, e in c["candidates"]]
        case_positions.append((profile_positions, cand_positions))

    print(f"  ({len(corpus_tokens)} unique papers)")
    bm25 = BM25Okapi(corpus_tokens)

    correlations = []
    for profile_positions, cand_positions in case_positions:
        sims = []
        for cand_pos, _ in cand_positions:
            query_tokens = corpus_tokens[cand_pos]
            scores = bm25.get_scores(query_tokens)
            sims.append(max(scores[p] for p in profile_positions))
        expertise = [e for _, e in cand_positions]
        rho, _ = spearmanr(sims, expertise)
        if not np.isnan(rho):
            correlations.append(rho)
    return correlations


def eval_specter2(cases):
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()

    def embed(papers_batch):
        texts = [(p.get("title") or "") + tokenizer.sep_token + (p.get("abstract") or "") for p in papers_batch]
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                                return_token_type_ids=False, max_length=512)
            output = model(**inputs)
            return output.last_hidden_state[:, 0, :].numpy()

    correlations = []
    for i, c in enumerate(cases):
        profile_emb = embed(c["profile_papers"])
        profile_vec = profile_emb.mean(axis=0)
        profile_vec /= (np.linalg.norm(profile_vec) or 1)
        cand_emb = embed([p for p, _ in c["candidates"]])
        norms = np.linalg.norm(cand_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cand_emb_normed = cand_emb / norms
        sims = cand_emb_normed @ profile_vec
        expertise = [e for _, e in c["candidates"]]
        rho, _ = spearmanr(sims, expertise)
        if not np.isnan(rho):
            correlations.append(rho)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(cases)} participants scored")
    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "specter2", "bm25", "bm25max", "both", "all"], default="both")
    args = parser.parse_args()

    cache = {}
    cases = build_cases(cache)
    print(f"Built {len(cases)} usable participant evaluation cases "
          f"(out of {len(load_evaluations())} total participants).")

    results = {}
    if args.method in ("tfidf", "both", "all"):
        corrs = eval_tfidf(cases)
        results["tfidf"] = {"mean_spearman": float(np.mean(corrs)), "n": len(corrs)}
        print(f"\nTF-IDF: mean Spearman correlation = {np.mean(corrs):.3f} (n={len(corrs)} participants)")

    if args.method in ("bm25", "all"):
        corrs = eval_bm25(cases)
        results["bm25"] = {"mean_spearman": float(np.mean(corrs)), "n": len(corrs)}
        print(f"\nBM25 (v1, concatenated-history query): mean Spearman correlation = "
              f"{np.mean(corrs):.3f} (n={len(corrs)} participants)")

    if args.method in ("bm25max", "all"):
        corrs = eval_bm25_maxpaper(cases)
        results["bm25max"] = {"mean_spearman": float(np.mean(corrs)), "n": len(corrs)}
        print(f"\nBM25 (v2, per-paper max query): mean Spearman correlation = "
              f"{np.mean(corrs):.3f} (n={len(corrs)} participants)")

    if args.method in ("specter2", "both", "all"):
        corrs = eval_specter2(cases)
        results["specter2"] = {"mean_spearman": float(np.mean(corrs)), "n": len(corrs)}
        print(f"\nSPECTER2: mean Spearman correlation = {np.mean(corrs):.3f} (n={len(corrs)} participants)")

    out_file = OUT_DIR / f"cmu_gold_results_{args.method}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
