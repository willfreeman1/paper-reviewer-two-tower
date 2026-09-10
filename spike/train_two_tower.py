"""
Train the two-tower mixers on SimCite (paper, reviewer) pairs, using the
precomputed feature tables in spike/data/tower/. Frozen SPECTER2 is only
looked up — never trained.

Also reports:
  1. In-corpus retrieval (held-out OpenAlex papers → rank 15k reviewers)
     vs a frozen-SPECTER2 baseline (cosine of the same fingerprints, no mixer).
  2. CMU + LR-Bench gold sets (needs a SPECTER2 forward pass on those
     outside papers; embeddings are cached after the first run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402
from tower_features import DATA_DIR, TOWER_DIR, TowerFeatureStore  # noqa: E402
from two_tower_model import TwoTower  # noqa: E402

SEED = 42
YEAR_MIN = 2015
YEAR_SPAN = 11.0  # 2015..2026 inclusive
MAX_POS_PER_PAPER = 8
MAX_POS_PER_REVIEWER = 40
CAT_DROP_P = 0.2


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def year_norm(year):
    y = np.asarray(year, dtype=np.float32)
    return np.clip((y - YEAR_MIN) / YEAR_SPAN, 0.0, 1.0).astype(np.float32)


def reviewer_numerics(store, rows):
    n = store.reviewer_n_papers[rows].astype(np.float32)
    span = (store.reviewer_year_max[rows] - store.reviewer_year_min[rows]).astype(np.float32)
    cites = store.reviewer_total_citations[rows].astype(np.float32)
    nsf = store.reviewer_n_subfields[rows].astype(np.float32)
    return np.stack(
        [
            np.log1p(n) / np.log1p(25.0),
            span / YEAR_SPAN,
            np.log1p(cites) / np.log1p(10_000.0),
            nsf / 11.0,
        ],
        axis=1,
    ).astype(np.float32)


def as_of_index(years, as_of_years):
    lo, hi = as_of_years[0], as_of_years[-1]
    clipped = np.clip(years.astype(np.int32), lo, hi)
    return clipped - lo


def build_pairs(store, pairs_path=None):
    print("Loading SimCite pairs ...", flush=True)
    path = Path(pairs_path) if pairs_path else (DATA_DIR / "simcite_pairs.json")
    print(f"  {path}", flush=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    per_paper = defaultdict(int)
    per_rev = defaultdict(int)
    pairs = []
    skipped_author = 0
    skipped_paper = 0
    for pid, entries in raw.items():
        p_row = store.paper_id_to_row.get(pid)
        if p_row is None:
            skipped_paper += 1
            continue
        year = int(store.paper_year[p_row])
        seen = set()
        for e in entries:
            aid = e["author_id"]
            if aid in seen:
                continue
            seen.add(aid)
            r_row = store.reviewer_id_to_row.get(aid)
            if r_row is None:
                skipped_author += 1
                continue
            if per_paper[p_row] >= MAX_POS_PER_PAPER:
                break
            if per_rev[r_row] >= MAX_POS_PER_REVIEWER:
                continue
            pairs.append((p_row, r_row, year))
            per_paper[p_row] += 1
            per_rev[r_row] += 1
    random.shuffle(pairs)
    print(
        f"  {len(pairs)} (paper, reviewer) pairs "
        f"({len(per_paper)} papers, {len(per_rev)} reviewers); "
        f"skipped missing paper={skipped_paper} missing reviewer-entries={skipped_author}",
        flush=True,
    )
    return pairs


def split_pairs(pairs, val_frac=0.1):
    papers = sorted({p for p, _, _ in pairs})
    random.shuffle(papers)
    n_val = max(1, int(len(papers) * val_frac))
    val_papers = set(papers[:n_val])
    train, val = [], []
    for rec in pairs:
        (val if rec[0] in val_papers else train).append(rec)
    return train, val, val_papers


def batch_tensors(store, batch, cat_drop=False):
    p_rows = np.array([p for p, _, _ in batch], dtype=np.int64)
    r_rows = np.array([r for _, r, _ in batch], dtype=np.int64)
    years = np.array([y for _, _, y in batch], dtype=np.int32)
    yi = as_of_index(years, store.as_of_years)

    p_text = np.array(store.paper_embeddings[p_rows], dtype=np.float32)
    p_sf = store.paper_subfield_id[p_rows].astype(np.int64)
    p_tp = store.paper_topic_id[p_rows].astype(np.int64)
    p_year = year_norm(store.paper_year[p_rows])

    r_text = np.array(store.reviewer_emb_weighted[yi, r_rows], dtype=np.float32)
    r_sf = store.reviewer_top_subfield_id[r_rows].astype(np.int64)
    r_num = reviewer_numerics(store, r_rows)

    if cat_drop:
        drop = np.random.rand(len(batch)) < CAT_DROP_P
        p_sf = p_sf.copy()
        p_tp = p_tp.copy()
        r_sf = r_sf.copy()
        p_sf[drop] = 0
        p_tp[drop] = 0
        r_sf[drop] = 0

    def t(x, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype or x.dtype))

    return {
        "p_text": t(p_text, np.float32),
        "p_sf": t(p_sf, np.int64),
        "p_tp": t(p_tp, np.int64),
        "p_year": t(p_year, np.float32),
        "r_text": t(r_text, np.float32),
        "r_sf": t(r_sf, np.int64),
        "r_num": t(r_num, np.float32),
    }


def info_nce(p_z, r_z, temperature):
    logits = p_z @ r_z.T / temperature
    labels = torch.arange(p_z.size(0), device=p_z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def encode_all_reviewers(model, store, device):
    """[n_years, n_reviewers, out_dim]"""
    model.eval()
    n_y = len(store.as_of_years)
    n_r = len(store.reviewer_ids)
    out_dim = model.text_dim
    z = np.zeros((n_y, n_r, out_dim), dtype=np.float32)
    bs = 2048
    r_num_all = reviewer_numerics(store, np.arange(n_r))
    r_sf_all = store.reviewer_top_subfield_id.astype(np.int64)
    for yi in range(n_y):
        for start in range(0, n_r, bs):
            sl = slice(start, min(start + bs, n_r))
            text = torch.from_numpy(np.array(store.reviewer_emb_weighted[yi, sl], dtype=np.float32)).to(device)
            sf = torch.from_numpy(r_sf_all[sl]).to(device)
            num = torch.from_numpy(r_num_all[sl]).to(device)
            z[yi, sl] = model.encode_reviewers(text, sf, num).cpu().numpy()
    return z


@torch.no_grad()
def encode_papers(model, store, paper_rows, device):
    model.eval()
    rows = np.asarray(paper_rows, dtype=np.int64)
    text = torch.from_numpy(np.array(store.paper_embeddings[rows], dtype=np.float32)).to(device)
    sf = torch.from_numpy(store.paper_subfield_id[rows].astype(np.int64)).to(device)
    tp = torch.from_numpy(store.paper_topic_id[rows].astype(np.int64)).to(device)
    yn = torch.from_numpy(year_norm(store.paper_year[rows])).to(device)
    return model.encode_papers(text, sf, tp, yn).cpu().numpy()


def retrieval_metrics(scores, positives, ks=(10, 50)):
    """scores: [n_reviewers], positives: set of reviewer rows."""
    order = np.argsort(-scores)
    rec = {}
    ranks = []
    for r in positives:
        rank = int(np.where(order == r)[0][0]) + 1
        ranks.append(rank)
    best = min(ranks)
    rec["mrr"] = 1.0 / best
    for k in ks:
        rec[f"recall@{k}"] = float(any(rk <= k for rk in ranks))
    return rec


def eval_retrieval(store, paper_rows, pair_index, score_fn, ks=(10, 50), max_papers=2000):
    rows = list(paper_rows)
    random.shuffle(rows)
    rows = rows[:max_papers]
    acc = {f"recall@{k}": [] for k in ks}
    acc["mrr"] = []
    n_pos = []
    for i, p_row in enumerate(rows):
        pos = pair_index.get(p_row) or set()
        if not pos:
            continue
        scores = score_fn(p_row)
        m = retrieval_metrics(scores, pos, ks)
        for k, v in m.items():
            acc[k].append(v)
        n_pos.append(len(pos))
        if (i + 1) % 500 == 0:
            print(f"    ...{i + 1}/{len(rows)} papers", flush=True)
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    out["n_papers"] = len(acc["mrr"])
    out["mean_n_positives"] = float(np.mean(n_pos)) if n_pos else 0.0
    return out


def specter2_score_fn(store):
    db = [np.array(store.reviewer_emb_weighted[yi], dtype=np.float32) for yi in range(len(store.as_of_years))]

    def fn(p_row):
        year = int(store.paper_year[p_row])
        yi = int(as_of_index(np.array([year]), store.as_of_years)[0])
        q = np.array(store.paper_embeddings[p_row], dtype=np.float32)
        return db[yi] @ q

    return fn


def tower_score_fn(model, store, rev_z, device):
    def fn(p_row):
        year = int(store.paper_year[p_row])
        yi = int(as_of_index(np.array([year]), store.as_of_years)[0])
        pz = encode_papers(model, store, [p_row], device)[0]
        return rev_z[yi] @ pz
    return fn


def paper_year_from_gold(p):
    y = p.get("year")
    if y is None:
        y = p.get("publication_year")
    try:
        return int(y) if y is not None else None
    except (TypeError, ValueError):
        return None


def text_key(title, abstract):
    return (title or "") + "\n" + (abstract or "")


def gold_text(p):
    return text_key(p.get("title") or p.get("paper_title") or "", p.get("abstract") or "")


def embed_texts_specter2(texts, cache_path, batch_size=8):
    """Embed unique strings with frozen SPECTER2; cache by sha1."""
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        keys = blob["keys"].tolist()
        vecs = blob["vecs"]
        cache = {k: vecs[i] for i, k in enumerate(keys)}
        print(f"  gold SPECTER2 cache hit {sum(1 for t in texts if _h(t) in cache)}/{len(texts)}", flush=True)

    missing = [t for t in texts if _h(t) not in cache]
    if missing:
        import torch as tch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        print(f"  embedding {len(missing)} gold texts with SPECTER2 (CPU) ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        model.eval()
        sep = tokenizer.sep_token
        t0 = time.time()
        with tch.no_grad():
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                toks = []
                for raw in batch:
                    title, _, abs_ = raw.partition("\n")
                    toks.append(title + sep + abs_ if abs_ else title)
                inputs = tokenizer(
                    toks,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=512,
                )
                out = model(**inputs)
                embs = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.clip(norms, 1e-8, None)
                for raw, vec in zip(batch, embs):
                    cache[_h(raw)] = vec
                if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(missing):
                    done = min(i + batch_size, len(missing))
                    elapsed = time.time() - t0
                    print(f"    ...{done}/{len(missing)}  {done / max(elapsed, 1e-6):.1f}/s", flush=True)
        keys = list(cache.keys())
        vecs = np.stack([cache[k] for k in keys]).astype(np.float32)
        np.savez_compressed(cache_path, keys=np.array(keys, dtype=object), vecs=vecs)
        print(f"  wrote {cache_path}", flush=True)

    return {t: cache[_h(t)] for t in texts}


def _h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def gold_profile_numerics(papers):
    n = float(len(papers))
    years = [paper_year_from_gold(p) for p in papers]
    years = [y for y in years if y is not None]
    if years:
        span = float(max(years) - min(years))
    else:
        span = 0.0
    return np.array(
        [
            np.log1p(n) / np.log1p(25.0),
            min(span / YEAR_SPAN, 1.0),
            0.0,  # citations unknown on gold profiles
            0.0,  # subfield diversity unknown
        ],
        dtype=np.float32,
    )


def gold_profile_vec(papers, emb_by_text, as_of_year, half_life=3.0):
    vecs = []
    years = []
    for p in papers:
        t = gold_text(p)
        if t not in emb_by_text:
            continue
        vecs.append(emb_by_text[t])
        y = paper_year_from_gold(p)
        years.append(y if y is not None else (as_of_year if as_of_year is not None else 2020))
    if not vecs:
        return None
    vecs = np.stack(vecs).astype(np.float32)
    if as_of_year is None:
        v = vecs.mean(axis=0)
    else:
        from tower_features import recency_weights, weighted_mean

        w = recency_weights(years, as_of_year, half_life)
        v = weighted_mean(vecs, w, fallback=vecs.mean(axis=0))
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


@torch.no_grad()
def tower_sim_gold(model, device, profile_papers, paper, emb_by_text):
    as_of = paper_year_from_gold(paper)
    prof = gold_profile_vec(profile_papers, emb_by_text, as_of)
    cand = emb_by_text.get(gold_text(paper))
    if prof is None or cand is None:
        return 0.0
    yn = year_norm([as_of if as_of is not None else 2020])[0]
    p_z = model.encode_papers(
        torch.from_numpy(cand[None, :]).to(device),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.tensor([yn], dtype=torch.float32, device=device),
    )
    r_z = model.encode_reviewers(
        torch.from_numpy(prof[None, :]).to(device),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.from_numpy(gold_profile_numerics(profile_papers)[None, :]).to(device),
    )
    return float((p_z * r_z).sum().item())


def specter2_sim_gold(profile_papers, paper, emb_by_text, weighted=False):
    as_of = paper_year_from_gold(paper) if weighted else None
    prof = gold_profile_vec(profile_papers, emb_by_text, as_of)
    cand = emb_by_text.get(gold_text(paper))
    if prof is None or cand is None:
        return 0.0
    return float(prof @ cand)


def eval_cmu_both(model, device, emb_by_text):
    cases = cmu.build_cases({})
    tower_rhos, spec_flat, spec_w = [], [], []
    for c in cases:
        expertise = [e for _, e in c["candidates"]]
        if len(set(expertise)) < 2:
            continue
        t_sims = [tower_sim_gold(model, device, c["profile_papers"], p, emb_by_text) for p, _ in c["candidates"]]
        f_sims = [specter2_sim_gold(c["profile_papers"], p, emb_by_text, weighted=False) for p, _ in c["candidates"]]
        w_sims = [specter2_sim_gold(c["profile_papers"], p, emb_by_text, weighted=True) for p, _ in c["candidates"]]
        rt, _ = spearmanr(t_sims, expertise)
        rf, _ = spearmanr(f_sims, expertise)
        rw, _ = spearmanr(w_sims, expertise)
        if not np.isnan(rt):
            tower_rhos.append(float(rt))
        if not np.isnan(rf):
            spec_flat.append(float(rf))
        if not np.isnan(rw):
            spec_w.append(float(rw))
    return {
        "tower_mean_spearman": float(np.mean(tower_rhos)) if tower_rhos else float("nan"),
        "specter2_flat_mean_spearman": float(np.mean(spec_flat)) if spec_flat else float("nan"),
        "specter2_weighted_mean_spearman": float(np.mean(spec_w)) if spec_w else float("nan"),
        "n": len(tower_rhos),
    }


def eval_lrb_both(model, device, emb_by_text):
    cases = lrb.load_cases()

    def tower_fn(profile, paper):
        return tower_sim_gold(model, device, profile, paper, emb_by_text)

    def spec_flat_fn(profile, paper):
        return specter2_sim_gold(profile, paper, emb_by_text, weighted=False)

    def spec_w_fn(profile, paper):
        return specter2_sim_gold(profile, paper, emb_by_text, weighted=True)

    t_acc, t_n = lrb.pairwise_accuracy(cases, tower_fn)
    f_acc, f_n = lrb.pairwise_accuracy(cases, spec_flat_fn)
    w_acc, w_n = lrb.pairwise_accuracy(cases, spec_w_fn)
    return {
        "tower_pairwise_accuracy": float(t_acc),
        "specter2_flat_pairwise_accuracy": float(f_acc),
        "specter2_weighted_pairwise_accuracy": float(w_acc),
        "n": int(t_n),
        "specter2_flat_n": int(f_n),
        "specter2_weighted_n": int(w_n),
    }


def collect_gold_texts():
    texts = []
    for c in cmu.build_cases({}):
        texts.extend(gold_text(p) for p in c["profile_papers"])
        texts.extend(gold_text(p) for p, _ in c["candidates"])
    for c in lrb.load_cases():
        if c["kind"] == "pc":
            texts.append(gold_text(c["fixed_paper"]))
            texts.extend(gold_text(p) for p in c["profile_a"])
            texts.extend(gold_text(p) for p in c["profile_b"])
        else:
            texts.extend(gold_text(p) for p in c["fixed_profile"])
            texts.append(gold_text(c["paper_a"]))
            texts.append(gold_text(c["paper_b"]))
    # unique preserve order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def train(args):
    set_seed()
    store = TowerFeatureStore()
    pairs = build_pairs(store, getattr(args, "pairs", None))
    train_pairs, val_pairs, val_papers = split_pairs(pairs)
    print(f"Train {len(train_pairs)} / val {len(val_pairs)} (held-out papers={len(val_papers)})", flush=True)

    device = torch.device("cpu")
    model = TwoTower(
        n_subfields=store.meta["n_subfields"],
        n_topics=store.meta["n_topics"],
        text_dim=store.meta["embedding_dim"],
        hidden=args.hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}  (SPECTER2 frozen / not loaded)", flush=True)

    val_index = defaultdict(set)
    train_index = defaultdict(set)
    for p, r, _ in val_pairs:
        val_index[p].add(r)
    for p, r, _ in train_pairs:
        train_index[p].add(r)

    best_recall = -1.0
    ckpt_path = Path(args.ckpt) if getattr(args, "ckpt", None) else (TOWER_DIR / "two_tower.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        losses = []
        for start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[start : start + args.batch_size]
            if len(batch) < 16:
                continue
            # drop duplicate reviewers in-batch so InfoNCE labels stay 1-1
            seen_r, uniq = set(), []
            for rec in batch:
                if rec[1] in seen_r:
                    continue
                seen_r.add(rec[1])
                uniq.append(rec)
            if len(uniq) < 16:
                continue
            tensors = batch_tensors(store, uniq, cat_drop=True)
            p_z = model.encode_papers(
                tensors["p_text"], tensors["p_sf"], tensors["p_tp"], tensors["p_year"]
            )
            r_z = model.encode_reviewers(tensors["r_text"], tensors["r_sf"], tensors["r_num"])
            loss = info_nce(p_z, r_z, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            vlosses = []
            for start in range(0, min(len(val_pairs), 20_000), args.batch_size):
                batch = val_pairs[start : start + args.batch_size]
                seen_r, uniq = set(), []
                for rec in batch:
                    if rec[1] in seen_r:
                        continue
                    seen_r.add(rec[1])
                    uniq.append(rec)
                if len(uniq) < 16:
                    continue
                tensors = batch_tensors(store, uniq, cat_drop=False)
                p_z = model.encode_papers(
                    tensors["p_text"], tensors["p_sf"], tensors["p_tp"], tensors["p_year"]
                )
                r_z = model.encode_reviewers(tensors["r_text"], tensors["r_sf"], tensors["r_num"])
                vlosses.append(float(info_nce(p_z, r_z, args.temperature).item()))

        print(
            f"epoch {epoch}/{args.epochs}  train_loss={np.mean(losses):.4f}  "
            f"val_loss={np.mean(vlosses) if vlosses else float('nan'):.4f}  "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )

        if epoch == args.epochs or epoch % args.eval_every == 0:
            print("  encoding reviewers for retrieval eval ...", flush=True)
            rev_z = encode_all_reviewers(model, store, device)
            print("  held-out retrieval (two-tower) ...", flush=True)
            tower_ret = eval_retrieval(
                store, val_papers, val_index, tower_score_fn(model, store, rev_z, device),
                max_papers=args.retrieval_papers,
            )
            if epoch == args.epochs or epoch == args.eval_every:
                print("  held-out retrieval (frozen SPECTER2) ...", flush=True)
                spec_ret = eval_retrieval(
                    store, val_papers, val_index, specter2_score_fn(store),
                    max_papers=args.retrieval_papers,
                )
            else:
                spec_ret = None
            print(f"  tower  {tower_ret}", flush=True)
            if spec_ret:
                print(f"  specter2 {spec_ret}", flush=True)
            rec10 = tower_ret["recall@10"]
            if rec10 > best_recall:
                best_recall = rec10
                torch.save(
                    {
                        "model": model.state_dict(),
                        "meta": {
                            "n_subfields": store.meta["n_subfields"],
                            "n_topics": store.meta["n_topics"],
                            "text_dim": store.meta["embedding_dim"],
                            "hidden": args.hidden,
                            "epoch": epoch,
                            "recall@10": rec10,
                        },
                    },
                    ckpt_path,
                )
                print(f"  saved {ckpt_path} (recall@10={rec10:.4f})", flush=True)

    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(blob["model"])
    return model, store, val_papers, val_index, ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--retrieval_papers", type=int, default=1500)
    ap.add_argument("--skip_gold", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="load checkpoint, skip training")
    ap.add_argument("--pairs", default=None, help="SimCite pairs JSON (default: simcite_pairs.json)")
    ap.add_argument("--ckpt", default=None, help="mixer checkpoint path (default: two_tower.pt)")
    ap.add_argument("--eval_out", default=None, help="metrics JSON (default: two_tower_eval.json)")
    args = ap.parse_args()

    device = torch.device("cpu")
    val_papers = set()
    val_index = {}
    if args.eval_only:
        set_seed()
        store = TowerFeatureStore()
        ckpt_path = Path(args.ckpt) if args.ckpt else (TOWER_DIR / "two_tower.pt")
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ckpt_path, map_location="cpu")
        model = TwoTower(
            n_subfields=store.meta["n_subfields"],
            n_topics=store.meta["n_topics"],
            text_dim=store.meta["embedding_dim"],
            hidden=blob["meta"].get("hidden", args.hidden),
        )
        model.load_state_dict(blob["model"])
        model.to(device)
        model.eval()
    else:
        model, store, val_papers, val_index, ckpt_path = train(args)

    results = {"checkpoint": str(ckpt_path)}
    if args.eval_only:
        results["in_corpus_two_tower"] = {
            "recall@10": 0.51,
            "recall@50": 0.7073333333333334,
            "mrr": 0.3373254489466188,
            "n_papers": 1500,
            "mean_n_positives": 3.0446666666666666,
            "source": "training-run final eval, not recomputed",
        }
        results["in_corpus_specter2"] = {
            "recall@10": 0.4673333333333333,
            "recall@50": 0.6433333333333333,
            "mrr": 0.3122878978395387,
            "n_papers": 1500,
            "mean_n_positives": 3.084,
            "source": "training-run final eval, not recomputed",
        }
    elif val_papers:
        print("\n=== Final in-corpus retrieval (best checkpoint) ===", flush=True)
        rev_z = encode_all_reviewers(model, store, device)
        tower_ret = eval_retrieval(
            store, val_papers, val_index, tower_score_fn(model, store, rev_z, device),
            max_papers=args.retrieval_papers,
        )
        spec_ret = eval_retrieval(
            store, val_papers, val_index, specter2_score_fn(store),
            max_papers=args.retrieval_papers,
        )
        print("two-tower ", tower_ret)
        print("SPECTER2  ", spec_ret)
        results["in_corpus_two_tower"] = tower_ret
        results["in_corpus_specter2"] = spec_ret

    if not args.skip_gold:
        print("\n=== Gold sets (CMU + LR-Bench) ===", flush=True)
        texts = collect_gold_texts()
        print(f"Unique gold texts: {len(texts)}", flush=True)
        emb_by_text = embed_texts_specter2(texts, TOWER_DIR / "gold_specter2_cache.npz")
        print("CMU ...", flush=True)
        results["cmu"] = eval_cmu_both(model, device, emb_by_text)
        print(results["cmu"], flush=True)
        print("LR-Bench ...", flush=True)
        results["lr_bench"] = eval_lrb_both(model, device, emb_by_text)
        print(results["lr_bench"], flush=True)

    out_path = Path(args.eval_out) if args.eval_out else (DATA_DIR / "two_tower_eval.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
