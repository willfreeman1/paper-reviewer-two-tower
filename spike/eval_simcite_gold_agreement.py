"""
Does SimCite's homemade 'right reviewer' list agree with the human
gold tests (CMU + LR-Bench)?

SimCite = take a paper's bibliography, keep the 10 cited papers whose
SPECTER2 fingerprint is closest to the query, treat those authors as
a match. Cite-all = skip the top-10 filter and keep every cited author.

This script does not retrain anything. It only asks: when we can line
up the gold person to an OpenAlex author ID, do they appear on that
list — and does that happen more often when humans said they were a
good fit?
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from fetch_gold_cited_works import canon_id  # noqa: E402
from tower_features import DATA_DIR, TOWER_DIR  # noqa: E402
from train_two_tower import (  # noqa: E402
    embed_texts_specter2,
    gold_text,
    paper_year_from_gold,
)

TOP_K = 10
WORKS_PATH = DATA_DIR / "gold_cited_works.jsonl"
OA_PATH = DATA_DIR / "gold_openalex.jsonl"
OUT_PATH = DATA_DIR / "simcite_gold_agreement.json"
EMB_CACHE = TOWER_DIR / "gold_cited_specter2_cache.npz"


def norm_name(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def names_match(a, b):
    ta, tb = set(norm_name(a).split()), set(norm_name(b).split())
    if not ta or not tb:
        return False
    if ta == tb or ta <= tb or tb <= ta:
        return True
    # last-token match + shared first initial
    la, lb = norm_name(a).split(), norm_name(b).split()
    if la[-1] == lb[-1] and la[0][:1] == lb[0][:1]:
        return True
    return False


def load_jsonl_by_id(path, key="id"):
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            kid = rec.get(key)
            if kid:
                out[canon_id(kid) if key == "id" else kid] = rec
    return out


def load_works():
    raw = {}
    if not WORKS_PATH.exists():
        return raw
    with open(WORKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and not rec.get("missing"):
                raw[canon_id(rec["id"])] = rec
    return raw


def load_gold_oa():
    out = {}
    with open(OA_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["text_key"]] = rec
    return out


def profile_author_counts(profile_papers, gold_oa, works):
    counts = Counter()
    names = {}
    n_resolved = 0
    for p in profile_papers:
        rec = gold_oa.get(gold_text(p)) or {}
        oid = rec.get("openalex_id")
        if not oid:
            continue
        w = works.get(canon_id(oid))
        if not w:
            continue
        n_resolved += 1
        for a in w.get("authors") or []:
            aid = a.get("id")
            if not aid:
                continue
            counts[aid] += 1
            if a.get("name"):
                names[aid] = a["name"]
        # catalog-seeded rows may only have author_ids
        if not w.get("authors"):
            for aid in w.get("author_ids") or []:
                counts[aid] += 1
    return counts, names, n_resolved


def join_majority(counts, min_papers=2):
    if not counts:
        return None, 0
    aid, n = counts.most_common(1)[0]
    runner = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    if n < min_papers or n <= runner:
        return None, n
    return aid, n


def join_cmu_reviewer(name, profile_papers, gold_oa, works):
    counts, names, n_resolved = profile_author_counts(profile_papers, gold_oa, works)
    maj, maj_n = join_majority(counts)
    named = [aid for aid, an in names.items() if names_match(name, an)]
    if maj and maj in named:
        return {"author_id": maj, "how": "majority+name", "n_profile_hits": maj_n, "n_resolved": n_resolved}
    if len(named) == 1 and counts[named[0]] >= 2:
        return {
            "author_id": named[0],
            "how": "name_only",
            "n_profile_hits": counts[named[0]],
            "n_resolved": n_resolved,
        }
    if maj:
        return {"author_id": maj, "how": "majority", "n_profile_hits": maj_n, "n_resolved": n_resolved}
    return None


def join_lrb_reviewer(profile_papers, gold_oa, works):
    counts, _names, n_resolved = profile_author_counts(profile_papers, gold_oa, works)
    maj, maj_n = join_majority(counts)
    if not maj:
        return None
    return {"author_id": maj, "how": "majority", "n_profile_hits": maj_n, "n_resolved": n_resolved}


def cmu_participant_name(pid):
    fp = Path(__file__).parent / "data" / "gold_cmu" / "data" / "participants" / f"{pid}.json"
    if not fp.exists():
        return ""
    return json.loads(fp.read_text(encoding="utf-8")).get("name") or ""


def recency_mult(query_year, cited_year, half_life):
    if query_year is None or cited_year is None or half_life <= 0:
        return 1.0
    age = max(0, int(query_year) - int(cited_year))
    return 0.5 ** (age / float(half_life))


def is_hub(work, cite_floor=1000, min_subfields=2):
    cites = work.get("cited_by_count") or 0
    n_sf = len(work.get("subfields") or [])
    return cites >= cite_floor and n_sf >= min_subfields


def work_text(w):
    return (w.get("title") or "") + "\n" + (w.get("abstract") or "")


def build_labels(query_oa, works, qvec, emb_by_id, variant):
    """Return {author_id: best_score} for one query paper."""
    refs = [canon_id(r) for r in (query_oa.get("referenced_works") or [])]
    qyear = query_oa.get("year")
    cands = []
    for rid in refs:
        w = works.get(rid)
        if not w:
            continue
        cands.append(w)
    if not cands:
        return {}

    if variant == "cite_all":
        scores = {}
        for w in cands:
            for aid in w.get("author_ids") or []:
                scores[aid] = max(scores.get(aid, 0.0), 1.0)
        return scores

    drop_hub = "hub" in variant
    half = None
    if "h2" in variant:
        half = 2
    elif "h3" in variant:
        half = 3
    elif "h5" in variant:
        half = 5

    ranked = []
    for w in cands:
        if drop_hub and is_hub(w):
            continue
        wid = canon_id(w["id"])
        vec = emb_by_id.get(wid)
        if vec is None:
            continue
        sim = float(np.dot(qvec, vec))
        if half is not None:
            sim = sim * recency_mult(qyear, w.get("year"), half)
        ranked.append((sim, w))
    ranked.sort(key=lambda x: -x[0])
    top = ranked[:TOP_K]
    scores = {}
    for sim, w in top:
        for aid in w.get("author_ids") or []:
            scores[aid] = max(scores.get(aid, 0.0), float(sim))
    return scores


def mean_or_none(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def pairwise_pref(pos_score, neg_score):
    if pos_score > neg_score:
        return "correct"
    if pos_score < neg_score:
        return "wrong"
    return "tie"


def summarize_pairwise(decisions):
    n = len(decisions)
    n_c = sum(1 for d in decisions if d == "correct")
    n_w = sum(1 for d in decisions if d == "wrong")
    n_t = sum(1 for d in decisions if d == "tie")
    decided = n_c + n_w
    return {
        "n": n,
        "correct": n_c,
        "wrong": n_w,
        "tie": n_t,
        "accuracy_incl_ties_as_wrong": (n_c / n) if n else None,
        "accuracy_among_decided": (n_c / decided) if decided else None,
    }


def hit_by_bucket(rows):
    buckets = {"1-2": [], "3": [], "4-5": []}
    for hit, exp in rows:
        if exp <= 2:
            buckets["1-2"].append(hit)
        elif exp == 3:
            buckets["3"].append(hit)
        else:
            buckets["4-5"].append(hit)
    return {k: {"n": len(v), "hit_rate": float(np.mean(v)) if v else None} for k, v in buckets.items()}


def collect_query_embed_texts(gold_oa, works, query_keys):
    texts = []
    id_for_text = []
    for key in query_keys:
        rec = gold_oa.get(key) or {}
        oid = rec.get("openalex_id")
        if oid:
            w = works.get(canon_id(oid))
            if w:
                texts.append(work_text(w) if (w.get("title") or w.get("abstract")) else key)
                id_for_text.append(("query", key, canon_id(oid)))
            else:
                texts.append(key)
                id_for_text.append(("query", key, None))
        else:
            texts.append(key)
            id_for_text.append(("query", key, None))
        for ref in rec.get("referenced_works") or []:
            w = works.get(canon_id(ref))
            if not w:
                continue
            texts.append(work_text(w))
            id_for_text.append(("ref", key, canon_id(w["id"])))
    return texts, id_for_text


VARIANTS = ("cite_all", "simcite", "recency_h2", "recency_h3", "recency_h5", "hub_drop", "recency_h3_hub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true", help="Cite-all only; no SPECTER2")
    args = ap.parse_args()

    print("Loading gold OpenAlex + cited works ...", flush=True)
    gold_oa = load_gold_oa()
    works = load_works()
    print(f"  gold papers {len(gold_oa)}  cited/gold works fetched {len(works)}", flush=True)

    cmu_cases = cmu.build_cases({})
    lrb_cases = lrb.load_cases()

    print("Joining reviewers to OpenAlex author IDs ...", flush=True)
    cmu_join = {}
    for c in cmu_cases:
        pid = c["participant_id"]
        name = cmu_participant_name(pid)
        j = join_cmu_reviewer(name, c["profile_papers"], gold_oa, works)
        if j and j["how"] in ("majority+name", "name_only"):
            cmu_join[pid] = j
        elif j:
            # majority-without-name kept only for the coverage note
            cmu_join[f"_unchecked_{pid}"] = j
    n_cmu_trusted = sum(1 for k in cmu_join if not str(k).startswith("_unchecked_"))
    print(
        f"  CMU name-confirmed {n_cmu_trusted}/{len(cmu_cases)}  "
        f"majority-only leftover {len(cmu_join) - n_cmu_trusted}",
        flush=True,
    )
    cmu_join = {k: v for k, v in cmu_join.items() if not str(k).startswith("_unchecked_")}

    lrb_join = {}  # cache key -> join dict

    def lrb_key(papers):
        return tuple(sorted(gold_text(p) for p in papers))

    n_lrb_profiles = 0
    n_lrb_joined = 0
    for c in lrb_cases:
        profiles = [c["profile_a"], c["profile_b"]] if c["kind"] == "pc" else [c["fixed_profile"]]
        for prof in profiles:
            n_lrb_profiles += 1
            k = lrb_key(prof)
            if k not in lrb_join:
                j = join_lrb_reviewer(prof, gold_oa, works)
                lrb_join[k] = j
            if lrb_join[k]:
                n_lrb_joined += 1
    n_unique = len(lrb_join)
    n_unique_ok = sum(1 for v in lrb_join.values() if v)
    print(
        f"  LR-Bench unique profiles joined {n_unique_ok}/{n_unique} "
        f"(appearances {n_lrb_joined}/{n_lrb_profiles})",
        flush=True,
    )

    query_keys = set()
    for c in cmu_cases:
        for p, _ in c["candidates"]:
            query_keys.add(gold_text(p))
    for c in lrb_cases:
        if c["kind"] == "pc":
            query_keys.add(gold_text(c["fixed_paper"]))
        else:
            query_keys.add(gold_text(c["paper_a"]))
            query_keys.add(gold_text(c["paper_b"]))

    emb_by_id = {}
    qvec_by_key = {}
    variants = ("cite_all",) if args.skip_embed else VARIANTS

    if not args.skip_embed:
        print("Embedding query + cited texts with SPECTER2 (cached) ...", flush=True)
        texts, id_for_text = collect_query_embed_texts(gold_oa, works, query_keys)
        unique_texts = list(dict.fromkeys(texts))
        emb_by_text = embed_texts_specter2(unique_texts, EMB_CACHE, batch_size=8)
        for text, spec in zip(texts, id_for_text):
            vec = emb_by_text.get(text)
            if vec is None:
                continue
            kind, key, wid = spec
            if kind == "query":
                qvec_by_key[key] = vec
            if wid:
                emb_by_id[wid] = vec
        # Query papers that were never fetched: embed the gold title/abstract.
        missing_q = [k for k in query_keys if k not in qvec_by_key]
        if missing_q:
            extra = embed_texts_specter2(missing_q, TOWER_DIR / "gold_specter2_cache.npz", batch_size=8)
            for k in missing_q:
                if k in extra:
                    qvec_by_key[k] = extra[k]
        print(f"  query vecs {len(qvec_by_key)}  cited vecs {len(emb_by_id)}", flush=True)

    def labels_for(key, variant):
        rec = gold_oa.get(key) or {}
        if not rec.get("referenced_works"):
            return None
        if variant == "cite_all":
            return build_labels(rec, works, None, {}, variant)
        qvec = qvec_by_key.get(key)
        if qvec is None:
            return None
        return build_labels(rec, works, qvec, emb_by_id, variant)

    report = {
        "coverage": {
            "cmu_joined": len(cmu_join),
            "cmu_cases": len(cmu_cases),
            "lrb_unique_profiles_joined": n_unique_ok,
            "lrb_unique_profiles": n_unique,
            "works_fetched": len(works),
            "query_papers": len(query_keys),
        },
        "variants": {},
        "cmu_join_how": dict(Counter(j["how"] for j in cmu_join.values())),
        "notes": (
            "Trained-on vs scored-on: this is not a model score. It asks "
            "whether the SimCite / cite-all author list for a gold paper "
            "contains the human-labeled person. Only rows where we could "
            "join that person to an OpenAlex author ID and the paper has "
            "a fetched bibliography are scored."
        ),
    }

    for variant in variants:
        print(f"Scoring {variant} ...", flush=True)
        cmu_rows = []  # (hit, expertise, best_score, participant)
        cmu_rhos_hit = []
        cmu_rhos_score = []
        for c in cmu_cases:
            j = cmu_join.get(c["participant_id"])
            if not j:
                continue
            hits, exps, scores = [], [], []
            for p, exp in c["candidates"]:
                labs = labels_for(gold_text(p), variant)
                if labs is None:
                    continue
                sc = float(labs.get(j["author_id"], 0.0))
                hit = 1.0 if j["author_id"] in labs else 0.0
                cmu_rows.append((hit, exp, sc, c["participant_id"]))
                hits.append(hit)
                exps.append(exp)
                scores.append(sc)
            if len(hits) >= 3 and len(set(exps)) >= 2:
                rho_h, _ = spearmanr(hits, exps)
                rho_s, _ = spearmanr(scores, exps)
                if not np.isnan(rho_h):
                    cmu_rhos_hit.append(float(rho_h))
                if not np.isnan(rho_s):
                    cmu_rhos_score.append(float(rho_s))
        hit_when = [h for h, e, _, _ in cmu_rows if h]
        miss = [e for h, e, _, _ in cmu_rows if not h]
        hit_exp = [e for h, e, _, _ in cmu_rows if h]

        pc_dec, rc_dec = [], []
        for c in lrb_cases:
            if c["kind"] == "pc":
                ja = lrb_join.get(lrb_key(c["profile_a"]))
                jb = lrb_join.get(lrb_key(c["profile_b"]))
                labs = labels_for(gold_text(c["fixed_paper"]), variant)
                if not ja or not jb or labs is None:
                    continue
                pos = float(labs.get(ja["author_id"], 0.0))
                neg = float(labs.get(jb["author_id"], 0.0))
                # a is the human-better reviewer
                pc_dec.append(pairwise_pref(pos, neg))
            else:
                j = lrb_join.get(lrb_key(c["fixed_profile"]))
                la = labels_for(gold_text(c["paper_a"]), variant)
                lb = labels_for(gold_text(c["paper_b"]), variant)
                if not j or la is None or lb is None:
                    continue
                pos = float(la.get(j["author_id"], 0.0))
                neg = float(lb.get(j["author_id"], 0.0))
                rc_dec.append(pairwise_pref(pos, neg))

        n_authors = []
        for key in query_keys:
            labs = labels_for(key, variant)
            if labs is not None:
                n_authors.append(len(labs))

        report["variants"][variant] = {
            "cmu": {
                "n_pairs": len(cmu_rows),
                "n_participants_scored": len({p for *_, p in cmu_rows}),
                "hit_rate": float(np.mean([h for h, *_ in cmu_rows])) if cmu_rows else None,
                "mean_expertise_when_hit": mean_or_none(hit_exp),
                "mean_expertise_when_miss": mean_or_none(miss),
                "hit_rate_by_expertise": hit_by_bucket([(h, e) for h, e, _, _ in cmu_rows]),
                "spearman_hit": mean_or_none(cmu_rhos_hit),
                "spearman_score": mean_or_none(cmu_rhos_score),
                "n_spearman": len(cmu_rhos_hit),
            },
            "lr_bench_pc": summarize_pairwise(pc_dec),
            "lr_bench_rc": summarize_pairwise(rc_dec),
            "mean_author_set_size": mean_or_none(n_authors),
            "n_query_papers_with_labels": len(n_authors),
        }
        v = report["variants"][variant]
        print(
            f"  CMU hit {v['cmu']['hit_rate']}  n={v['cmu']['n_pairs']}  "
            f"Spearman(hit) {v['cmu']['spearman_hit']}  "
            f"LRB-pc acc {v['lr_bench_pc']['accuracy_incl_ties_as_wrong']} "
            f"(ties {v['lr_bench_pc']['tie']}/{v['lr_bench_pc']['n']})",
            flush=True,
        )

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
