"""
Rebuild one-hop SimCite keys with optional down-weights.

Still: bibliography ∩ catalog → rank cited papers → those authors are
"a right person." Vanilla SimCite ranks by SPECTER2 only. Here we
multiply that closeness by some of:

  recency  — newer cites count more (half-life 3 years)
  topic    — Qwen topic-phrase overlap (exact or MiniLM very-close)
  method   — same for Qwen method phrases

MiniLM is a small "do these two short phrases mean the same thing?"
model. High bar (0.85) so GNN ≈ graph neural network, but tree ≉ GNN.

Writes four pair files; does not overwrite simcite_pairs.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
TOWER_DIR = DATA_DIR / "tower"
TOP_K = 10
HALF_LIFE = 3.0
TAG_FLOOR = 0.20
SOFT_THRESH = 0.85
PHRASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PHRASE_CACHE = TOWER_DIR / "qwen_phrase_minilm.npz"

VARIANTS = {
    "recency": {"recency": True, "topic": False, "method": False},
    "topic": {"recency": False, "topic": True, "method": False},
    "method": {"recency": False, "topic": False, "method": True},
    "all": {"recency": True, "topic": True, "method": True},
}


def norm_phrase(s):
    return " ".join(str(s).strip().lower().split())


def load_qwen():
    topics, methods = {}, {}
    path = DATA_DIR / "aspect_profiles_qwen3.jsonl"
    phrases = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("paper_id")
            prof = rec.get("profile") or {}
            t = {norm_phrase(x) for x in (prof.get("topics") or []) if norm_phrase(x)}
            m = {norm_phrase(x) for x in (prof.get("methodologies") or []) if norm_phrase(x)}
            topics[pid] = t
            methods[pid] = m
            phrases.update(t)
            phrases.update(m)
    return topics, methods, sorted(phrases)


def encode_phrases(phrases):
    TOWER_DIR.mkdir(parents=True, exist_ok=True)
    if PHRASE_CACHE.exists():
        blob = np.load(PHRASE_CACHE, allow_pickle=True)
        cached = {k: blob["vecs"][i] for i, k in enumerate(blob["keys"].tolist())}
        if all(p in cached for p in phrases):
            print(f"  MiniLM phrase cache hit {len(phrases)}", flush=True)
            return cached
        print("  MiniLM cache stale; re-encoding ...", flush=True)

    print(f"  embedding {len(phrases)} Qwen phrases with MiniLM ...", flush=True)
    tok = AutoTokenizer.from_pretrained(PHRASE_MODEL)
    model = AutoModel.from_pretrained(PHRASE_MODEL)
    model.eval()
    vecs = []
    t0 = time.time()
    bs = 64
    with torch.no_grad():
        for i in range(0, len(phrases), bs):
            batch = phrases[i : i + bs]
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=64)
            out = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy().astype(np.float32))
            if (i // bs + 1) % 50 == 0 or i + bs >= len(phrases):
                print(f"    ...{min(i + bs, len(phrases))}/{len(phrases)}  {time.time() - t0:.0f}s", flush=True)
    mat = np.vstack(vecs)
    np.savez_compressed(PHRASE_CACHE, keys=np.array(phrases, dtype=object), vecs=mat)
    return {p: mat[i] for i, p in enumerate(phrases)}


def soft_coverage(query_tags, cite_tags, emb):
    """How much of the new paper's tag list is covered by the cite."""
    if not query_tags or not cite_tags:
        return None
    cite_list = list(cite_tags)
    cite_mat = np.stack([emb[t] for t in cite_list])
    hits = 0
    for q in query_tags:
        if q in cite_tags:
            hits += 1
            continue
        qv = emb.get(q)
        if qv is None:
            continue
        if float(np.max(cite_mat @ qv)) >= SOFT_THRESH:
            hits += 1
    return hits / len(query_tags)


def tag_mult(overlap):
    if overlap is None:
        return 1.0
    return TAG_FLOOR + (1.0 - TAG_FLOOR) * float(overlap)


def recency_mult(qyear, cyear):
    if qyear is None or cyear is None:
        return 1.0
    age = max(0, int(qyear) - int(cyear))
    return 0.5 ** (age / HALF_LIFE)


def main():
    print("Loading papers, Qwen tags, SPECTER2 ...", flush=True)
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

    topics, methods, phrases = load_qwen()
    print(f"  Qwen papers {len(topics)}  unique phrases {len(phrases)}", flush=True)
    emb = encode_phrases(phrases)

    emb_ids = json.loads((DATA_DIR / "paper_embedding_ids_full.json").read_text(encoding="utf-8"))
    embs = np.load(DATA_DIR / "paper_embeddings_full.npy")
    id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs_n = (embs / norms).astype(np.float32)

    out = {name: {} for name in VARIANTS}
    n_soft_topic = n_soft_method = n_exact_topic = n_exact_method = 0
    t0 = time.time()
    done = 0
    for pid, p in papers.items():
        done += 1
        if done % 20000 == 0:
            print(f"  ...{done}/{len(papers)} ({time.time() - t0:.0f}s)", flush=True)
        refs = p.get("referenced_works") or []
        if not refs or pid not in id_to_idx:
            continue
        cands = [r for r in refs if r in papers and r in paper_to_authors and r in id_to_idx]
        if not cands:
            continue
        qvec = embs_n[id_to_idx[pid]]
        sims = embs_n[[id_to_idx[r] for r in cands]] @ qvec
        qyear = p.get("year")
        qt, qm = topics.get(pid) or set(), methods.get(pid) or set()

        scored = {name: [] for name in VARIANTS}
        for i, rid in enumerate(cands):
            spec = float(sims[i])
            rec = recency_mult(qyear, papers[rid].get("year"))
            t_ov = soft_coverage(qt, topics.get(rid) or set(), emb)
            m_ov = soft_coverage(qm, methods.get(rid) or set(), emb)
            if t_ov is not None:
                if t_ov > 0 and t_ov > (len(qt & (topics.get(rid) or set())) / max(len(qt), 1) + 1e-9):
                    n_soft_topic += 1
                if qt & (topics.get(rid) or set()):
                    n_exact_topic += 1
            if m_ov is not None:
                if m_ov > 0 and m_ov > (len(qm & (methods.get(rid) or set())) / max(len(qm), 1) + 1e-9):
                    n_soft_method += 1
                if qm & (methods.get(rid) or set()):
                    n_exact_method += 1
            for name, flags in VARIANTS.items():
                s = spec
                if flags["recency"]:
                    s *= rec
                if flags["topic"]:
                    s *= tag_mult(t_ov)
                if flags["method"]:
                    s *= tag_mult(m_ov)
                scored[name].append((s, spec, rid))

        for name, rows in scored.items():
            rows.sort(key=lambda x: -x[0])
            top = rows[:TOP_K]
            entries = []
            for s, spec, cited_pid in top:
                for aid in paper_to_authors[cited_pid]:
                    entries.append({
                        "author_id": aid,
                        "cited_paper_id": cited_pid,
                        "similarity": float(s),
                        "specter2": float(spec),
                    })
            out[name][pid] = entries

    for name, pairs in out.items():
        path = DATA_DIR / f"simcite_pairs_{name}.json"
        path.write_text(json.dumps(pairs), encoding="utf-8")
        n_auth = sum(len(v) for v in pairs.values())
        print(f"  wrote {path}  papers={len(pairs)}  pairs={n_auth}")

    print(
        f"Soft extra matches (beyond exact): topic {n_soft_topic}  method {n_soft_method}  "
        f"(exact-hit cites: topic {n_exact_topic} method {n_exact_method})"
    )
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
