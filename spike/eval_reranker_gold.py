"""
Score the second-pass tree on CMU + LR-Bench.

These papers are not in the OpenAlex catalog, so most pair clues are
missing. What is available: first-pass closeness, word overlap, recency
when a year is present, and profile size.

The human tests already give a short list (about 10 papers on CMU; two
options on LR-Bench). This scores those pairs; it does not search 15,000
people.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from tower_features import DATA_DIR, TOWER_DIR  # noqa: E402
from train_reranker import (  # noqa: E402
    FEATURE_NAMES,
    load_model,
    tag_coverage,
    tag_jaccard,
)
from train_two_tower import (  # noqa: E402
    collect_gold_texts,
    embed_texts_specter2,
    gold_text,
    paper_year_from_gold,
    specter2_sim_gold,
    tower_sim_gold,
)

EMPTY_TAGS = {
    "topics": frozenset(),
    "methodologies": frozenset(),
    "applications": frozenset(),
}


def norm_title(title):
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return " ".join(t.split())


def tfidf_text(p):
    return ((p.get("title") or p.get("paper_title") or "") + " " + (p.get("abstract") or "")).strip()


def paper_title(p):
    return p.get("title") or p.get("paper_title") or ""


def load_qwen_by_title():
    """Reuse catalog Qwen tags when a gold title matches an OpenAlex paper."""
    id_to_prof = {}
    qwen_path = DATA_DIR / "aspect_profiles_qwen3.jsonl"
    if not qwen_path.exists():
        return {}, {}
    with open(qwen_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("paper_id") and rec.get("profile"):
                id_to_prof[rec["paper_id"]] = rec["profile"]
    extra = DATA_DIR / "aspect_profiles_gold_qwen3.jsonl"
    gold_by_text = {}
    if extra.exists():
        with open(extra, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = rec.get("text_key") or rec.get("title")
                if key and rec.get("profile"):
                    gold_by_text[key] = rec["profile"]
    title_to_sets = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            prof = id_to_prof.get(p["id"])
            if not prof:
                continue
            nt = norm_title(p.get("title"))
            if nt:
                title_to_sets[nt] = _as_sets(prof)
    return title_to_sets, gold_by_text


def _as_sets(prof):
    out = {}
    for key in ("topics", "methodologies", "applications"):
        out[key] = frozenset(str(x).strip().lower() for x in (prof.get(key) or []) if str(x).strip())
    return out


def tags_for(p, title_to_sets, gold_by_text):
    extra = gold_by_text.get(gold_text(p)) or gold_by_text.get(paper_title(p))
    if extra:
        return _as_sets(extra)
    return title_to_sets.get(norm_title(paper_title(p)), EMPTY_TAGS)


def fit_tfidf():
    print("Fitting TF-IDF on the catalog (same settings as second-pass training) ...", flush=True)
    texts = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            texts.append(((p.get("title") or "") + " " + (p.get("abstract") or "")).strip())
    vec = TfidfVectorizer(max_features=20_000, min_df=3, stop_words="english")
    vec.fit(texts)
    print(f"  vocab {len(vec.vocabulary_)}", flush=True)
    return vec


def tfidf_cosine(vec, paper, profile_papers):
    q = normalize(vec.transform([tfidf_text(paper)]))
    if not profile_papers:
        return 0.0
    m = vec.transform([tfidf_text(p) for p in profile_papers]).mean(axis=0)
    from scipy.sparse import csr_matrix

    prof = normalize(csr_matrix(m))
    return float(q.multiply(prof).sum())


def load_openalex_by_text():
    path = DATA_DIR / "gold_openalex.jsonl"
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec.get("text_key")
            if key:
                out[key] = rec
    return out


def oa_for(p, oa_by_text):
    return oa_by_text.get(gold_text(p)) or {}


def sf_name(raw):
    if isinstance(raw, dict):
        return raw.get("display_name") or raw.get("name")
    return raw


def year_of(p, oa_by_text):
    y = paper_year_from_gold(p)
    if y is not None:
        return y
    oy = oa_for(p, oa_by_text).get("year")
    try:
        return int(oy) if oy is not None else None
    except (TypeError, ValueError):
        return None


def recency_feats(paper, profile_papers, oa_by_text):
    year_p = year_of(paper, oa_by_text)
    years = [year_of(p, oa_by_text) for p in profile_papers]
    years_ok = [y for y in years if y is not None]
    if year_p is None or not years_ok:
        return 1.0, 0.0
    past = [y for y in years_ok if y <= year_p]
    last = max(past) if past else max(years_ok)
    year_gap = (year_p - last) / 11.0
    pool = past if past else years_ok
    frac3 = float(np.mean([1.0 if y >= year_p - 3 else 0.0 for y in pool]))
    return float(year_gap), frac3


def openalex_pair_feats(paper, profile_papers, oa_by_text):
    pmeta = oa_for(paper, oa_by_text)
    p_sf = sf_name(pmeta.get("subfield"))
    p_tp = pmeta.get("topic_id")
    p_id = pmeta.get("openalex_id")
    p_refs = set(pmeta.get("referenced_works") or [])
    r_sfs = []
    r_topics = set()
    r_ids = set()
    r_refs = set()
    cites = 0.0
    as_of = year_of(paper, oa_by_text)
    for rp in profile_papers:
        rm = oa_for(rp, oa_by_text)
        y = year_of(rp, oa_by_text)
        if as_of is not None and y is not None and y > as_of:
            continue
        if rm.get("subfield"):
            r_sfs.append(sf_name(rm.get("subfield")))
        if rm.get("topic_id"):
            r_topics.add(rm["topic_id"])
        if rm.get("openalex_id"):
            r_ids.add(rm["openalex_id"])
        r_refs.update(rm.get("referenced_works") or [])
        cites += float(rm.get("cited_by_count") or 0)
    same_sf = 1.0 if p_sf and p_sf in r_sfs else 0.0
    share = float(r_sfs.count(p_sf) / len(r_sfs)) if (p_sf and r_sfs) else 0.0
    same_tp = 1.0 if p_tp and p_tp in r_topics else 0.0
    cites_r = 1.0 if (p_refs and r_ids and (p_refs & r_ids)) else 0.0
    r_cites_p = 1.0 if (p_id and p_id in r_refs) else 0.0
    if p_refs or r_refs:
        inter = len(p_refs & r_refs)
        union = len(p_refs | r_refs)
        jac = inter / union if union else 0.0
    else:
        jac = 0.0
    n_sf = (len(set(r_sfs)) / 11.0) if r_sfs else 0.0
    log_c = float(np.log1p(cites) / np.log1p(10_000.0))
    return same_sf, share, same_tp, cites_r, r_cites_p, jac, n_sf, log_c


def pair_row(
    profile_papers,
    paper,
    stage1_score,
    stage1_rank,
    vec,
    title_to_sets,
    gold_by_text,
    oa_by_text,
):
    year_gap, frac3 = recency_feats(paper, profile_papers, oa_by_text)
    tfidf = tfidf_cosine(vec, paper, profile_papers)
    p_tags = tags_for(paper, title_to_sets, gold_by_text)
    as_of = year_of(paper, oa_by_text)
    r_tags = {k: set() for k in EMPTY_TAGS}
    for rp in profile_papers:
        y = year_of(rp, oa_by_text)
        if as_of is not None and y is not None and y > as_of:
            continue
        ts = tags_for(rp, title_to_sets, gold_by_text)
        for k in r_tags:
            r_tags[k].update(ts[k])
    r_tags = {k: frozenset(v) for k, v in r_tags.items()}
    n_pap = len(profile_papers) / 25.0
    same_sf, share, same_tp, cites_r, r_cites_p, jac, n_sf, log_c = openalex_pair_feats(
        paper, profile_papers, oa_by_text
    )
    return np.array(
        [
            stage1_score,
            1.0 / (stage1_rank + 1.0),
            same_sf,
            share,
            same_tp,
            cites_r,
            r_cites_p,
            jac,
            year_gap,
            frac3,
            tfidf,
            tag_jaccard(p_tags["topics"], r_tags["topics"]),
            tag_coverage(p_tags["topics"], r_tags["topics"]),
            tag_jaccard(p_tags["methodologies"], r_tags["methodologies"]),
            tag_coverage(p_tags["methodologies"], r_tags["methodologies"]),
            tag_jaccard(p_tags["applications"], r_tags["applications"]),
            tag_coverage(p_tags["applications"], r_tags["applications"]),
            n_pap,
            n_sf,
            log_c,
        ],
        dtype=np.float32,
    )


def score_pairs(pairs, model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text):
    """pairs = list of (profile_papers, paper). Returns dict of score lists."""
    s1 = [tower_sim_gold(model, device, prof, paper, emb_by_text) for prof, paper in pairs]
    spec = [specter2_sim_gold(prof, paper, emb_by_text, weighted=False) for prof, paper in pairs]
    order = np.argsort(-np.asarray(s1, dtype=np.float64))
    rank = np.empty(len(pairs), dtype=np.int32)
    rank[order] = np.arange(len(pairs), dtype=np.int32)
    X = np.stack(
        [
            pair_row(prof, paper, s1[i], int(rank[i]), vec, title_to_sets, gold_by_text, oa_by_text)
            for i, (prof, paper) in enumerate(pairs)
        ]
    )
    out = {"stage1": s1, "specter2": spec}
    for name, booster in boosters.items():
        out[name] = booster.predict(X).tolist()
    return out


def eval_cmu(model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text):
    cases = cmu.build_cases({})
    buckets = {k: [] for k in ["stage1", "specter2", *boosters]}
    n_qwen = 0
    n_papers = 0
    for c in cases:
        expertise = [e for _, e in c["candidates"]]
        if len(set(expertise)) < 2:
            continue
        pairs = [(c["profile_papers"], p) for p, _ in c["candidates"]]
        scored = score_pairs(pairs, model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text)
        for p, _ in c["candidates"]:
            n_papers += 1
            if tags_for(p, title_to_sets, gold_by_text) is not EMPTY_TAGS:
                n_qwen += 1
        for name, sims in scored.items():
            rho, _ = spearmanr(sims, expertise)
            if not np.isnan(rho):
                buckets[name].append(float(rho))
    return {
        name: {"mean_spearman": float(np.mean(v)) if v else float("nan"), "n": len(v)}
        for name, v in buckets.items()
    } | {"qwen_tagged_candidate_papers": n_qwen, "candidate_papers": n_papers}


def eval_lrb(model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text):
    cases = lrb.load_cases()
    names = ["stage1", "specter2", *boosters]
    correct = {k: 0 for k in names}
    total = {k: 0 for k in names}

    for c in cases:
        if c["kind"] == "pc":
            pairs = [(c["profile_a"], c["fixed_paper"]), (c["profile_b"], c["fixed_paper"])]
        else:
            pairs = [(c["fixed_profile"], c["paper_a"]), (c["fixed_profile"], c["paper_b"])]
        scored = score_pairs(pairs, model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text)
        for name in names:
            a, b = scored[name]
            if a == b:
                continue
            total[name] += 1
            if (a > b) == c["a_is_positive"]:
                correct[name] += 1
    return {
        name: {
            "pairwise_accuracy": (correct[name] / total[name]) if total[name] else float("nan"),
            "n": total[name],
        }
        for name in names
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--reranker_nocite", default=None)
    ap.add_argument("--eval_out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    import torch

    device = torch.device("cpu")
    from tower_features import TowerFeatureStore

    store = TowerFeatureStore()
    model = load_model(store, args.tower)
    model.to(device)

    boosters = {
        "reranker_all_clues": lgb.Booster(
            model_file=str(Path(args.reranker) if args.reranker else (TOWER_DIR / "reranker.txt"))
        ),
        "reranker_no_cite": lgb.Booster(
            model_file=str(
                Path(args.reranker_nocite) if args.reranker_nocite else (TOWER_DIR / "reranker_no_cite.txt")
            )
        ),
    }
    assert list(boosters["reranker_no_cite"].feature_name()) == FEATURE_NAMES

    texts = collect_gold_texts()
    print(f"Unique gold texts: {len(texts)}", flush=True)
    emb_by_text = embed_texts_specter2(texts, TOWER_DIR / "gold_specter2_cache.npz")
    vec = fit_tfidf()
    print("Loading Qwen tags (title match + optional gold file) ...", flush=True)
    title_to_sets, gold_by_text = load_qwen_by_title()
    oa_by_text = load_openalex_by_text()
    n_hit = sum(1 for t in texts if title_to_sets.get(norm_title(t.split("\n")[0])) or t in gold_by_text)
    n_oa = sum(1 for t in texts if (oa_by_text.get(t) or {}).get("openalex_id"))
    print(f"  Qwen tags: catalog+file hits on {n_hit}/{len(texts)} texts (file={len(gold_by_text)})", flush=True)
    print(f"  OpenAlex resolved: {n_oa}/{len(texts)}", flush=True)

    print("CMU ...", flush=True)
    cmu_out = eval_cmu(model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text)
    print(cmu_out, flush=True)
    print("LR-Bench ...", flush=True)
    lrb_out = eval_lrb(model, device, emb_by_text, vec, boosters, title_to_sets, gold_by_text, oa_by_text)
    print(lrb_out, flush=True)

    out = {
        "cmu": cmu_out,
        "lr_bench": lrb_out,
        "openalex_resolved": n_oa,
        "notes": (
            "OpenAlex fields/citations/years used where a title lookup succeeded. "
            "Qwen tags from catalog title match plus aspect_profiles_gold_qwen3.jsonl. "
            "reranker_no_cite never saw citation clues; reranker_all_clues is the product tree."
        ),
        "elapsed_s": time.time() - t0,
    }
    path = Path(args.eval_out) if args.eval_out else (DATA_DIR / "reranker_gold_eval.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
