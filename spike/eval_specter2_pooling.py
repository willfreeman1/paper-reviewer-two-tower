"""
SPECTER2 pooling check on CMU + LR-Bench. No training.

vector_mean = average the paper fingerprints, then cosine (headline).
score_* = cosine(query, each profile paper), then collapse those scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from tower_features import DATA_DIR, TOWER_DIR  # noqa: E402
from train_two_tower import embed_texts_specter2, gold_text  # noqa: E402

VARIANTS = ("vector_mean", "score_mean", "score_max", "score_p75", "score_top3")


def collapse(sims, how):
    s = np.asarray(sims, dtype=np.float64)
    if s.size == 0:
        return 0.0
    if how == "score_mean":
        return float(s.mean())
    if how == "score_max":
        return float(s.max())
    if how == "score_p75":
        return float(np.percentile(s, 75))
    if how == "score_top3":
        k = min(3, s.size)
        return float(np.sort(s)[-k:].mean())
    raise ValueError(how)


def sim_vector_mean(q, profs):
    if not profs:
        return 0.0
    m = np.mean(np.stack(profs), axis=0)
    n = float(np.linalg.norm(m))
    if n < 1e-8:
        return 0.0
    return float((m / n) @ q)


def sim_score(q, profs, how):
    if not profs:
        return 0.0
    sims = [float(q @ v) for v in profs]
    return collapse(sims, how)


def vecs_for(papers, emb):
    out = []
    for p in papers:
        v = emb.get(gold_text(p))
        if v is not None:
            out.append(v)
    return out


def eval_cmu(cases, emb, how):
    rhos = []
    for c in cases:
        profs = vecs_for(c["profile_papers"], emb)
        scores, exps = [], []
        for p, e in c["candidates"]:
            q = emb.get(gold_text(p))
            if q is None or not profs:
                scores.append(0.0)
            elif how == "vector_mean":
                scores.append(sim_vector_mean(q, profs))
            else:
                scores.append(sim_score(q, profs, how))
            exps.append(e)
        if len(set(exps)) < 2:
            continue
        rho, _ = spearmanr(scores, exps)
        if not np.isnan(rho):
            rhos.append(float(rho))
    return {"mean_spearman": float(np.mean(rhos)) if rhos else None, "n": len(rhos)}


def eval_lrb(cases, emb, how):
    def fn(profile, paper):
        profs = vecs_for(profile, emb)
        q = emb.get(gold_text(paper))
        if q is None or not profs:
            return 0.0
        if how == "vector_mean":
            return sim_vector_mean(q, profs)
        return sim_score(q, profs, how)

    pc = [c for c in cases if c["kind"] == "pc"]
    rc = [c for c in cases if c["kind"] == "rc"]
    acc_all, n_all = lrb.pairwise_accuracy(cases, fn)
    acc_pc, n_pc = lrb.pairwise_accuracy(pc, fn)
    acc_rc, n_rc = lrb.pairwise_accuracy(rc, fn)
    return {
        "pairwise_accuracy": float(acc_all),
        "n": int(n_all),
        "ties_skipped": True,
        "paper_centric": {"pairwise_accuracy": float(acc_pc), "n": int(n_pc)},
        "reviewer_centric": {"pairwise_accuracy": float(acc_rc), "n": int(n_rc)},
    }


def main():
    print("Loading gold cases ...", flush=True)
    cmu_cases = cmu.build_cases({})
    lrb_cases = lrb.load_cases()
    texts = []
    for c in cmu_cases:
        texts.extend(gold_text(p) for p in c["profile_papers"])
        texts.extend(gold_text(p) for p, _ in c["candidates"])
    for c in lrb_cases:
        if c["kind"] == "pc":
            texts.append(gold_text(c["fixed_paper"]))
            texts.extend(gold_text(p) for p in c["profile_a"])
            texts.extend(gold_text(p) for p in c["profile_b"])
        else:
            texts.extend(gold_text(p) for p in c["fixed_profile"])
            texts.append(gold_text(c["paper_a"]))
            texts.append(gold_text(c["paper_b"]))
    texts = list(dict.fromkeys(t for t in texts if t.strip()))
    print(f"  unique texts {len(texts)}  CMU cases {len(cmu_cases)}  LRB {len(lrb_cases)}", flush=True)
    emb = embed_texts_specter2(texts, TOWER_DIR / "gold_specter2_cache.npz")

    out = {
        "trained_on": None,
        "notes": (
            "Scored on CMU Spearman (n=participants) and LR-Bench pairwise "
            "accuracy (ties skipped). vector_mean is the headline recipe. "
            "score_* collapse per-paper cosines, not averaged vectors."
        ),
        "variants": {},
    }
    for how in VARIANTS:
        print(f"Scoring {how} ...", flush=True)
        cmu_r = eval_cmu(cmu_cases, emb, how)
        lrb_r = eval_lrb(lrb_cases, emb, how)
        out["variants"][how] = {"cmu": cmu_r, "lr_bench": lrb_r}
        print(
            f"  CMU {cmu_r['mean_spearman']:.4f} n={cmu_r['n']}  "
            f"LRB {lrb_r['pairwise_accuracy']:.4f} n={lrb_r['n']} "
            f"(pc {lrb_r['paper_centric']['pairwise_accuracy']:.4f} "
            f"rc {lrb_r['reviewer_centric']['pairwise_accuracy']:.4f})",
            flush=True,
        )

    vm = out["variants"]["vector_mean"]
    out["sanity_vector_mean_matches_headline"] = {
        "cmu_ok": abs((vm["cmu"]["mean_spearman"] or 0) - 0.434) < 0.015,
        "lrb_ok": abs((vm["lr_bench"]["pairwise_accuracy"] or 0) - 0.732) < 0.015,
        "expected_cmu": 0.434,
        "expected_lrb": 0.732,
    }
    path = DATA_DIR / "specter2_pooling_gold.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print("sanity", out["sanity_vector_mean_matches_headline"], flush=True)


if __name__ == "__main__":
    main()
