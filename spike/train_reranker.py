"""
Stage-2 re-ranker: take the two-tower's top-K people per paper, build
pairwise clues (shared field, citations, word overlap, Qwen tag overlap),
train a tree model to reorder that shortlist.

The tree only sees people the receptionist already proposed, so it cannot
rescue a match that Stage 1 ranked below K.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from tower_features import DATA_DIR, TOWER_DIR, TowerFeatureStore  # noqa: E402
from train_two_tower import (  # noqa: E402
    SEED,
    as_of_index,
    build_pairs,
    encode_all_reviewers,
    encode_papers,
    set_seed,
    split_pairs,
)
from two_tower_model import TwoTower  # noqa: E402

K = 100
FEATURE_NAMES = [
    "stage1_score",
    "stage1_rank_inv",
    "same_subfield",
    "subfield_share",
    "same_topic",
    "paper_cites_reviewer",
    "reviewer_cites_paper",
    "shared_ref_jaccard",
    "year_gap",
    "frac_papers_last_3y",
    "tfidf_cosine",
    "qwen_topics_jaccard",
    "qwen_topics_coverage",
    "qwen_methods_jaccard",
    "qwen_methods_coverage",
    "qwen_apps_jaccard",
    "qwen_apps_coverage",
    "n_papers_norm",
    "n_subfields_norm",
    "log_citations",
]


def load_model(store, ckpt=None):
    ckpt = Path(ckpt) if ckpt else (TOWER_DIR / "two_tower.pt")
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt, map_location="cpu")
    model = TwoTower(
        n_subfields=store.meta["n_subfields"],
        n_topics=store.meta["n_topics"],
        text_dim=store.meta["embedding_dim"],
        hidden=blob["meta"].get("hidden", 256),
    )
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_paper_side_data(store):
    """Texts, in-corpus citation sets, aligned to store paper rows."""
    n = len(store.paper_ids)
    texts = [""] * n
    refs = [set() for _ in range(n)]
    print("Loading paper texts + citation lists ...", flush=True)
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            row = store.paper_id_to_row.get(p["id"])
            if row is None:
                continue
            texts[row] = ((p.get("title") or "") + " " + (p.get("abstract") or "")).strip()
            cited = set()
            for rid in p.get("referenced_works") or []:
                rr = store.paper_id_to_row.get(rid)
                if rr is not None:
                    cited.add(int(rr))
            refs[row] = cited
    return texts, refs


def load_qwen_sets(store):
    """Per-paper lowercased tag sets; empty if Qwen didn't tag the paper."""
    n = len(store.paper_ids)
    out = {
        "topics": [frozenset() for _ in range(n)],
        "methodologies": [frozenset() for _ in range(n)],
        "applications": [frozenset() for _ in range(n)],
    }
    path = DATA_DIR / "aspect_profiles_qwen3.jsonl"
    n_hit = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            row = store.paper_id_to_row.get(rec.get("paper_id"))
            if row is None:
                continue
            prof = rec.get("profile") or {}
            n_hit += 1
            for key in out:
                out[key][row] = frozenset(
                    str(x).strip().lower() for x in (prof.get(key) or []) if str(x).strip()
                )
    print(f"  Qwen tags for {n_hit}/{n} papers", flush=True)
    return out


def tag_jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def tag_coverage(paper_set, rev_set):
    if not paper_set:
        return 0.0
    return len(paper_set & rev_set) / len(paper_set)


def build_reviewer_side(store, refs, qwen):
    n_r = len(store.reviewer_ids)
    n_y = len(store.as_of_years)
    n_sf = store.meta["n_subfields"] + 1
    subfield_hist = np.zeros((n_r, n_sf), dtype=np.float32)
    topic_sets = [set() for _ in range(n_r)]
    wrote = [set() for _ in range(n_r)]
    cited = [set() for _ in range(n_r)]
    last_year = np.full((n_y, n_r), -1, dtype=np.int16)
    frac_3y = np.zeros((n_y, n_r), dtype=np.float32)
    q_union = {
        k: [[frozenset() for _ in range(n_r)] for _ in range(n_y)]
        for k in ("topics", "methodologies", "applications")
    }

    print("Building per-reviewer citation / tag / recency snapshots ...", flush=True)
    for r in range(n_r):
        rows = store.reviewer_paper_rows(r)
        if len(rows) == 0:
            continue
        years = store.paper_year[rows]
        sfs = store.paper_subfield_id[rows]
        tps = store.paper_topic_id[rows]
        wrote[r] = set(int(x) for x in rows.tolist())
        topic_sets[r] = set(int(x) for x in tps.tolist())
        for rr in rows:
            cited[r].update(refs[int(rr)])
        counts = np.bincount(sfs, minlength=n_sf).astype(np.float32)
        subfield_hist[r] = counts / max(float(counts.sum()), 1.0)

        # year snapshots: papers with year <= as_of
        for yi, as_of in enumerate(store.as_of_years):
            mask = years <= as_of
            if not np.any(mask):
                continue
            last_year[yi, r] = int(years[mask].max())
            frac_3y[yi, r] = float(np.mean((years[mask] >= as_of - 3)))
            kept = rows[mask]
            for key in q_union:
                acc = set()
                for pr in kept:
                    acc.update(qwen[key][int(pr)])
                q_union[key][yi][r] = frozenset(acc)
        if (r + 1) % 3000 == 0:
            print(f"  ...{r + 1}/{n_r} reviewers", flush=True)
    return {
        "subfield_hist": subfield_hist,
        "topic_sets": topic_sets,
        "wrote": wrote,
        "cited": cited,
        "last_year": last_year,
        "frac_3y": frac_3y,
        "q_union": q_union,
    }


def fit_tfidf(texts, store):
    print("Fitting TF-IDF on title+abstract ...", flush=True)
    vec = TfidfVectorizer(max_features=20_000, min_df=3, stop_words="english")
    paper_x = vec.fit_transform(texts)
    paper_x = normalize(paper_x)
    print(f"  paper matrix {paper_x.shape}", flush=True)
    from scipy.sparse import csr_matrix

    dim = paper_x.shape[1]
    n_r = len(store.reviewer_ids)
    rev_x = []
    for r in range(n_r):
        rows = store.reviewer_paper_rows(r)
        if len(rows) == 0:
            rev_x.append(csr_matrix((1, dim), dtype=np.float32))
        else:
            m = paper_x[rows].mean(axis=0)
            rev_x.append(csr_matrix(m, dtype=np.float32))
    rev_x = normalize(vstack(rev_x, format="csr"))
    print(f"  reviewer matrix {rev_x.shape}", flush=True)
    return paper_x, rev_x


def pair_tfidf_cosine(paper_x, rev_x, p_rows, r_rows):
    a = paper_x[p_rows]
    b = rev_x[r_rows]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def stage1_topk(store, model, paper_rows, k=K):
    device = torch.device("cpu")
    print(f"Encoding {len(paper_rows)} papers + all reviewers (Stage 1) ...", flush=True)
    rev_z = encode_all_reviewers(model, store, device)
    p_rows = np.asarray(paper_rows, dtype=np.int64)
    p_z = np.zeros((len(p_rows), model.text_dim), dtype=np.float32)
    bs = 4096
    for start in range(0, len(p_rows), bs):
        sl = slice(start, min(start + bs, len(p_rows)))
        p_z[sl] = encode_papers(model, store, p_rows[sl], device)
        if start % (bs * 4) == 0:
            print(f"  papers {min(start + bs, len(p_rows))}/{len(p_rows)}", flush=True)

    years = store.paper_year[p_rows]
    yi = as_of_index(years, store.as_of_years)
    top_idx = np.zeros((len(p_rows), k), dtype=np.int32)
    top_scores = np.zeros((len(p_rows), k), dtype=np.float32)
    for year_i in range(len(store.as_of_years)):
        mask = yi == year_i
        if not np.any(mask):
            continue
        db = rev_z[year_i]  # [R, D]
        scores = p_z[mask] @ db.T  # [n, R]
        # top-k (largest)
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-row_scores, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        row_scores = np.take_along_axis(row_scores, order, axis=1)
        top_idx[mask] = part.astype(np.int32)
        top_scores[mask] = row_scores.astype(np.float32)
        print(f"  year {store.as_of_years[year_i]}: {int(mask.sum())} papers", flush=True)
    return top_idx, top_scores, yi


def build_feature_matrix(store, p_rows, top_idx, top_scores, yi, refs, rev_side, paper_x, rev_x, qwen):
    n_q, k = top_idx.shape
    p_rep = np.repeat(p_rows.astype(np.int32), k)
    r_rep = top_idx.reshape(-1)
    yi_rep = np.repeat(yi.astype(np.int32), k)
    rank = np.tile(np.arange(k, dtype=np.float32), n_q)
    scores = top_scores.reshape(-1)

    p_sf = store.paper_subfield_id[p_rep]
    p_tp = store.paper_topic_id[p_rep]
    p_year = store.paper_year[p_rep].astype(np.float32)
    r_sf = store.reviewer_top_subfield_id[r_rep]
    same_sf = (p_sf == r_sf).astype(np.float32)
    share = rev_side["subfield_hist"][r_rep, p_sf]

    same_tp = np.zeros(len(r_rep), dtype=np.float32)
    cites_r = np.zeros(len(r_rep), dtype=np.float32)
    r_cites_p = np.zeros(len(r_rep), dtype=np.float32)
    jac = np.zeros(len(r_rep), dtype=np.float32)
    q_feats = {name: np.zeros(len(r_rep), dtype=np.float32) for name in FEATURE_NAMES if name.startswith("qwen_")}

    print(f"  pairwise clues for {len(r_rep):,} rows ...", flush=True)
    topic_sets = rev_side["topic_sets"]
    wrote = rev_side["wrote"]
    cited = rev_side["cited"]
    q_union = rev_side["q_union"]
    t0 = time.time()
    for i in range(len(r_rep)):
        pr = int(p_rep[i])
        rr = int(r_rep[i])
        yii = int(yi_rep[i])
        same_tp[i] = 1.0 if int(p_tp[i]) in topic_sets[rr] else 0.0
        pref = refs[pr]
        rwrote = wrote[rr]
        rcited = cited[rr]
        cites_r[i] = 1.0 if (pref & rwrote) else 0.0
        r_cites_p[i] = 1.0 if pr in rcited else 0.0
        if pref or rcited:
            inter = len(pref & rcited)
            union = len(pref | rcited)
            jac[i] = inter / union if union else 0.0
        for dim, jname, cname in (
            ("topics", "qwen_topics_jaccard", "qwen_topics_coverage"),
            ("methodologies", "qwen_methods_jaccard", "qwen_methods_coverage"),
            ("applications", "qwen_apps_jaccard", "qwen_apps_coverage"),
        ):
            ps, rs = qwen[dim][pr], q_union[dim][yii][rr]
            q_feats[jname][i] = tag_jaccard(ps, rs)
            q_feats[cname][i] = tag_coverage(ps, rs)
        if (i + 1) % 500_000 == 0:
            print(f"    ...{i + 1:,}/{len(r_rep):,}  {time.time() - t0:.0f}s", flush=True)

    last = rev_side["last_year"][yi_rep, r_rep].astype(np.float32)
    year_gap = np.where(last >= 0, (p_year - last) / 11.0, 1.0).astype(np.float32)
    frac3 = rev_side["frac_3y"][yi_rep, r_rep]
    tfidf = pair_tfidf_cosine(paper_x, rev_x, p_rep, r_rep)
    n_pap = store.reviewer_n_papers[r_rep].astype(np.float32) / 25.0
    n_sf = store.reviewer_n_subfields[r_rep].astype(np.float32) / 11.0
    log_c = np.log1p(store.reviewer_total_citations[r_rep].astype(np.float32)) / np.log1p(10_000.0)

    cols = [
        scores,
        1.0 / (rank + 1.0),
        same_sf,
        share,
        same_tp,
        cites_r,
        r_cites_p,
        jac,
        year_gap,
        frac3,
        tfidf,
        q_feats["qwen_topics_jaccard"],
        q_feats["qwen_topics_coverage"],
        q_feats["qwen_methods_jaccard"],
        q_feats["qwen_methods_coverage"],
        q_feats["qwen_apps_jaccard"],
        q_feats["qwen_apps_coverage"],
        n_pap,
        n_sf,
        log_c,
    ]
    X = np.stack(cols, axis=1).astype(np.float32)
    return X, p_rep, r_rep


def labels_for(p_rep, r_rep, pair_index):
    y = np.zeros(len(p_rep), dtype=np.float32)
    for i in range(len(p_rep)):
        if int(r_rep[i]) in pair_index.get(int(p_rep[i]), ()):
            y[i] = 1.0
    return y


def groups_of_k(n_rows, k=K):
    assert n_rows % k == 0
    return np.full(n_rows // k, k, dtype=np.int32)


def train_lgb(X_tr, y_tr, X_va, y_va, k=K):
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=groups_of_k(len(y_tr), k), feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_va, label=y_va, group=groups_of_k(len(y_va), k), feature_name=FEATURE_NAMES, reference=dtrain)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": SEED,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=250,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )
    return booster


def recall_at(order, positives, ks):
    order = np.asarray(order)
    kmax = len(order)
    ranks = []
    for r in positives:
        hits = np.where(order == r)[0]
        ranks.append(int(hits[0]) + 1 if len(hits) else kmax + 1)
    best = min(ranks)
    out = {"mrr": 1.0 / best}
    for k in ks:
        out[f"recall@{k}"] = float(any(rk <= k for rk in ranks))
    return out


def eval_shortlists(p_rows, top_idx, top_scores, gbt_scores, pair_index, ks=(10, 50), max_papers=1500, seed=SEED):
    rng = random.Random(seed)
    idxs = list(range(len(p_rows)))
    rng.shuffle(idxs)
    idxs = idxs[:max_papers]
    s1 = {f"recall@{k}": [] for k in ks}
    s1["mrr"] = []
    gb = {f"recall@{k}": [] for k in ks}
    gb["mrr"] = []
    in_k = []
    n_used = 0
    k = top_idx.shape[1]
    for j in idxs:
        pos = pair_index.get(int(p_rows[j])) or set()
        if not pos:
            continue
        n_used += 1
        s1_order = top_idx[j]  # already sorted by stage1 score
        m1 = recall_at(s1_order, pos, ks)
        for key, v in m1.items():
            s1[key].append(v)
        g_order = s1_order[np.argsort(-gbt_scores[j * k : (j + 1) * k])]
        mg = recall_at(g_order, pos, ks)
        for key, v in mg.items():
            gb[key].append(v)
        in_k.append(float(any(r in set(s1_order.tolist()) for r in pos)))
    def mean(d):
        return {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()} | {"n_papers": n_used}
    return mean(s1), mean(gb), float(np.mean(in_k) if in_k else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--train_papers", type=int, default=20000)
    ap.add_argument("--lgb_val_papers", type=int, default=2000)
    ap.add_argument("--eval_papers", type=int, default=1500)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--tower", default=None, help="Stage-1 mixer checkpoint")
    ap.add_argument("--model_full", default="reranker.txt")
    ap.add_argument("--model_nocite", default="reranker_no_cite.txt")
    ap.add_argument("--eval_out", default=None)
    args = ap.parse_args()

    set_seed()
    t_all = time.time()
    store = TowerFeatureStore()
    model = load_model(store, args.tower)
    pairs = build_pairs(store, args.pairs)
    train_pairs, val_pairs, val_paper_set = split_pairs(pairs)
    pair_index = defaultdict(set)
    for p, r, _ in train_pairs + val_pairs:
        pair_index[p].add(r)

    train_papers = sorted({p for p, _, _ in train_pairs})
    random.shuffle(train_papers)
    lgb_val_ps = train_papers[: args.lgb_val_papers]
    lgb_tr_ps = train_papers[args.lgb_val_papers : args.lgb_val_papers + args.train_papers]
    val_ps = sorted(val_paper_set)
    print(
        f"GBT train papers={len(lgb_tr_ps)}  lgb-val={len(lgb_val_ps)}  "
        f"retrieval-val={len(val_ps)} (held out of GBT)",
        flush=True,
    )

    texts, refs = load_paper_side_data(store)
    qwen = load_qwen_sets(store)
    rev_side = build_reviewer_side(store, refs, qwen)
    paper_x, rev_x = fit_tfidf(texts, store)

    need = lgb_tr_ps + lgb_val_ps + val_ps
    # unique preserve order
    seen, paper_rows = set(), []
    for p in need:
        if p not in seen:
            seen.add(p)
            paper_rows.append(p)
    paper_rows = np.array(paper_rows, dtype=np.int64)
    loc = {int(p): i for i, p in enumerate(paper_rows)}

    top_idx, top_scores, yi = stage1_topk(store, model, paper_rows, k=args.k)
    X, p_rep, r_rep = build_feature_matrix(
        store, paper_rows, top_idx, top_scores, yi, refs, rev_side, paper_x, rev_x, qwen
    )
    y = labels_for(p_rep, r_rep, pair_index)

    def slice_papers(ps):
        rows = []
        for p in ps:
            i = loc[int(p)]
            sl = slice(i * args.k, (i + 1) * args.k)
            rows.append(np.arange(sl.start, sl.stop))
        idx = np.concatenate(rows)
        return X[idx], y[idx]

    X_tr, y_tr = slice_papers(lgb_tr_ps)
    X_va, y_va = slice_papers(lgb_val_ps)

    def drop_all_negative_queries(X, y, k):
        n_q = len(y) // k
        keep = []
        for i in range(n_q):
            sl = slice(i * k, (i + 1) * k)
            if y[sl].sum() > 0:
                keep.append(i)
        if not keep:
            raise SystemExit("No training queries with a positive in the shortlist")
        idx = np.concatenate([np.arange(i * k, (i + 1) * k) for i in keep])
        print(f"  kept {len(keep)}/{n_q} queries that have a SimCite person in top-{k}", flush=True)
        return X[idx], y[idx]

    X_tr, y_tr = drop_all_negative_queries(X_tr, y_tr, args.k)
    X_va, y_va = drop_all_negative_queries(X_va, y_va, args.k)
    print(f"Train rows {len(y_tr):,}  pos_rate={y_tr.mean():.4f}  lgb-val rows {len(y_va):,}", flush=True)

    X_eval, _ = slice_papers(val_ps)
    val_arr = np.array(val_ps, dtype=np.int64)
    val_top = np.stack([top_idx[loc[int(p)]] for p in val_ps])
    val_sc = np.stack([top_scores[loc[int(p)]] for p in val_ps])

    def fit_and_eval(Xtr, Xva, Xev, tag, model_name):
        print(f"\n=== {tag} ===", flush=True)
        booster = train_lgb(Xtr, y_tr, Xva, y_va, k=args.k)
        path = TOWER_DIR / model_name
        booster.save_model(str(path))
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        importance = sorted(
            [
                {"feature": n, "gain": float(g), "split": int(s)}
                for n, g, s in zip(FEATURE_NAMES, gain, split)
            ],
            key=lambda d: -d["gain"],
        )
        print("Feature importance (gain):")
        for row in importance:
            print(f"  {row['gain']:10.1f}  {row['feature']}")
        pred = booster.predict(Xev)
        s1, gb, cov = eval_shortlists(
            val_arr, val_top, val_sc, pred, pair_index, max_papers=args.eval_papers
        )
        print(f"  Stage-1 positives in top-{args.k}: {100*cov:.1f}%  (ceiling)")
        print(f"  Stage-1  {s1}")
        print(f"  Re-ranker {gb}")
        return {
            "stage1": s1,
            "reranker": gb,
            "positive_in_shortlist": cov,
            "feature_importance": importance,
            "model": str(path),
        }

    full = fit_and_eval(X_tr, X_va, X_eval, "all clues (product model)", args.model_full)

    # SimCite labels come from the citation graph, so "does this paper cite
    # that person?" almost *is* the answer key. Train a second tree with those
    # three clues zeroed so we can see whether word overlap / Qwen / fields
    # still move the ranking without that shortcut.
    cite_idx = [
        FEATURE_NAMES.index(n)
        for n in ("paper_cites_reviewer", "reviewer_cites_paper", "shared_ref_jaccard")
    ]
    X_tr_nc, X_va_nc, X_eval_nc = X_tr.copy(), X_va.copy(), X_eval.copy()
    X_tr_nc[:, cite_idx] = 0
    X_va_nc[:, cite_idx] = 0
    X_eval_nc[:, cite_idx] = 0
    no_cite = fit_and_eval(
        X_tr_nc, X_va_nc, X_eval_nc,
        "citation clues zeroed (fairer SimCite readout)",
        args.model_nocite,
    )

    out = {
        "k": args.k,
        "n_train_papers": len(lgb_tr_ps),
        "elapsed_s": time.time() - t_all,
        "full": full,
        "no_cite_clues": no_cite,
        # top-level copies of the product model so older readers still work
        "stage1": full["stage1"],
        "reranker": full["reranker"],
        "positive_in_shortlist": full["positive_in_shortlist"],
        "feature_importance": full["feature_importance"],
        "model": full["model"],
    }
    out_path = Path(args.eval_out) if args.eval_out else (DATA_DIR / "reranker_eval.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
