"""
Proxy-vs-gold check, part 3: run TF-IDF / SPECTER2 / BM25 against LR-Bench
(RATE paper, Jan 2026) -- a SECOND, larger, independent real gold-standard
dataset, complementing the CMU Gold Standard check in cmu_gold_eval.py.

Note on format: the publicly released LR-Bench data is PAIRWISE (not the
1-1,055 pointwise 1-5 BARS ratings described in the paper -- the authors'
own README says pointwise data is still being released). Each record is a
(anchor, positive, negative) triple where `positive` has a higher expertise
rating than `negative` for the same anchor:

  - "paper_centric" (pc): anchor = one query paper. positive/negative = two
    candidate REVIEWERS (each with their own paper-list profile). A good
    method should score the query paper more similar to the positive
    reviewer's profile than to the negative reviewer's profile.
  - "reviewer_centric" (rc): anchor = one reviewer profile (paper list).
    positive/negative = two candidate PAPERS. A good method should score the
    positive paper more similar to the reviewer's profile than the negative
    paper.

Both reduce to the same underlying comparison: one multi-paper "profile" vs
two single "candidate" papers -- does the method prefer the higher-rated
one? Metric: pairwise accuracy (% of comparisons where predicted similarity
ranks the positive candidate above the negative one). This is the standard
way to score preference/pairwise data like this, and is what RATE's own
"paper-centric / reviewer-centric preference triplets" training signal is
built from.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def load_cases():
    """Normalize both pc and rc files into one common shape:
    {profile_papers: [...], candidate_a: paper, candidate_b: paper,
     a_is_positive: bool} -- always "does the method prefer candidate_a or
     candidate_b," with a_is_positive telling us the correct answer."""
    cases = []

    pc = json.loads((DATA_DIR / "lr_bench_evaluations_pc.json").read_text(encoding="utf-8"))
    for r in pc:
        anchor = r["anchor"]  # single query paper
        pos_profile = r["positive"]["papers"]
        neg_profile = r["negative"]["papers"]
        if not pos_profile or not neg_profile:
            continue
        # Here the "profile" role alternates: we ask "is the query paper
        # closer to the positive reviewer's profile, or the negative
        # reviewer's profile" -- i.e. two DIFFERENT profiles compared
        # against one fixed candidate (the anchor paper).
        cases.append({
            "kind": "pc",
            "fixed_paper": anchor,
            "profile_a": pos_profile,
            "profile_b": neg_profile,
            "a_is_positive": True,
        })

    rc = json.loads((DATA_DIR / "lr_bench_evaluations_rc.json").read_text(encoding="utf-8"))
    for r in rc:
        anchor_profile = r["anchor"]["papers"]
        pos_paper = r["positive"]
        neg_paper = r["negative"]
        if not anchor_profile:
            continue
        # Here one fixed profile is compared against two candidate papers.
        cases.append({
            "kind": "rc",
            "fixed_profile": anchor_profile,
            "paper_a": pos_paper,
            "paper_b": neg_paper,
            "a_is_positive": True,
        })

    return cases


def paper_text(p):
    return (p.get("title") or p.get("paper_title") or "") + " " + (p.get("abstract") or "")


def all_unique_papers(cases):
    """Dedup every paper object appearing anywhere, by (title, abstract) --
    these records don't share a common ID scheme across pc/rc, so identity
    is by text content."""
    seen = {}

    def add(p):
        key = paper_text(p)
        if key not in seen:
            seen[key] = p
        return key

    for c in cases:
        if c["kind"] == "pc":
            add(c["fixed_paper"])
            for p in c["profile_a"]:
                add(p)
            for p in c["profile_b"]:
                add(p)
        else:
            for p in c["fixed_profile"]:
                add(p)
            add(c["paper_a"])
            add(c["paper_b"])
    return seen  # key (text) -> paper dict


def pairwise_accuracy(cases, sim_fn):
    """sim_fn(profile_papers_list, single_paper) -> float similarity."""
    correct = 0
    total = 0
    for c in cases:
        if c["kind"] == "pc":
            sim_a = sim_fn(c["profile_a"], c["fixed_paper"])
            sim_b = sim_fn(c["profile_b"], c["fixed_paper"])
        else:
            sim_a = sim_fn(c["fixed_profile"], c["paper_a"])
            sim_b = sim_fn(c["fixed_profile"], c["paper_b"])
        if sim_a == sim_b:
            continue  # tie, skip (matches standard pairwise-accuracy convention)
        total += 1
        if (sim_a > sim_b) == c["a_is_positive"]:
            correct += 1
    return correct / total if total else float("nan"), total


def eval_tfidf(cases):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    unique = all_unique_papers(cases)
    keys = list(unique.keys())
    vec = TfidfVectorizer(max_features=50000, stop_words="english")
    X = vec.fit_transform([paper_text(unique[k]) for k in keys])
    key_to_row = {k: i for i, k in enumerate(keys)}

    def sim_fn(profile_papers, single_paper):
        prof_idx = [key_to_row[paper_text(p)] for p in profile_papers]
        prof_vec = np.asarray(X[prof_idx].mean(axis=0))
        cand_idx = key_to_row[paper_text(single_paper)]
        return float(cosine_similarity(X[cand_idx], prof_vec)[0, 0])

    return pairwise_accuracy(cases, sim_fn)


def eval_bm25(cases):
    """Fairer (v2-style, per-candidate-max-paper) BM25 setup -- see the
    note in cmu_gold_eval.py's eval_bm25_maxpaper for why this is used
    instead of concatenating a whole profile into one giant query."""
    from rank_bm25 import BM25Okapi

    unique = all_unique_papers(cases)
    keys = list(unique.keys())
    corpus_tokens = [paper_text(unique[k]).lower().split() for k in keys]
    key_to_row = {k: i for i, k in enumerate(keys)}
    print(f"Fitting BM25 over {len(corpus_tokens)} unique papers...")
    bm25 = BM25Okapi(corpus_tokens)

    score_cache = {}

    def scores_for(single_paper):
        k = paper_text(single_paper)
        if k not in score_cache:
            score_cache[k] = bm25.get_scores(corpus_tokens[key_to_row[k]])
        return score_cache[k]

    def sim_fn(profile_papers, single_paper):
        scores = scores_for(single_paper)  # query = the single candidate paper
        prof_rows = [key_to_row[paper_text(p)] for p in profile_papers]
        return float(max(scores[r] for r in prof_rows))

    return pairwise_accuracy(cases, sim_fn)


def eval_specter2(cases, log_every=200):
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()

    unique = all_unique_papers(cases)
    keys = list(unique.keys())
    print(f"Embedding {len(keys)} unique papers with SPECTER2...")

    embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            texts = [
                (unique[k].get("title") or unique[k].get("paper_title") or "")
                + tokenizer.sep_token + (unique[k].get("abstract") or "")
                for k in batch_keys
            ]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                                return_token_type_ids=False, max_length=512)
            out = model(**inputs)
            embs.append(out.last_hidden_state[:, 0, :].numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"  ...{i + len(batch_keys)}/{len(keys)} papers embedded")
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    key_to_row = {k: i for i, k in enumerate(keys)}

    def sim_fn(profile_papers, single_paper):
        prof_idx = [key_to_row[paper_text(p)] for p in profile_papers]
        prof_vec = embs[prof_idx].mean(axis=0)
        prof_vec /= (np.linalg.norm(prof_vec) or 1)
        cand_vec = embs[key_to_row[paper_text(single_paper)]]
        return float(cand_vec @ prof_vec)

    return pairwise_accuracy(cases, sim_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "bm25", "specter2", "all"], default="all")
    args = parser.parse_args()

    cases = load_cases()
    n_pc = sum(1 for c in cases if c["kind"] == "pc")
    n_rc = sum(1 for c in cases if c["kind"] == "rc")
    print(f"Loaded {len(cases)} pairwise comparisons ({n_pc} paper-centric, {n_rc} reviewer-centric).")

    results = {}
    if args.method in ("tfidf", "all"):
        acc, n = eval_tfidf(cases)
        results["tfidf"] = {"pairwise_accuracy": acc, "n": n}
        print(f"\nTF-IDF: pairwise accuracy = {acc:.3f} (n={n} comparisons)")

    if args.method in ("bm25", "all"):
        acc, n = eval_bm25(cases)
        results["bm25"] = {"pairwise_accuracy": acc, "n": n}
        print(f"\nBM25 (per-candidate-max-paper): pairwise accuracy = {acc:.3f} (n={n} comparisons)")

    if args.method in ("specter2", "all"):
        acc, n = eval_specter2(cases)
        results["specter2"] = {"pairwise_accuracy": acc, "n": n}
        print(f"\nSPECTER2: pairwise accuracy = {acc:.3f} (n={n} comparisons)")

    out_file = DATA_DIR / f"lr_bench_results_{args.method}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
