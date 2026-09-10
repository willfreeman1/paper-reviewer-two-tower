"""
Score a fine-tuned (Authors or SimCite) SPECTER2 adapter on CMU and
LR-Bench, using the same case-building code as cmu_gold_eval.py and
lr_bench_eval.py. Does not score on the OpenAlex homemade key — that would
be circular for a model trained on that key.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from adapters import AutoAdapterModel
from scipy.stats import spearmanr
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import cmu_gold_eval as cmu  # noqa: E402
import lr_bench_eval as lrb  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def load_model(variant):
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    adapter_name = f"{variant}_contrastive"
    model.load_adapter(str(DATA_DIR / f"adapter_{variant}"), load_as=adapter_name, set_active=True)
    model.eval()
    return tokenizer, model


def embed_batch(tokenizer, model, texts, device):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                            return_token_type_ids=False, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy()


def eval_cmu(variant, device):
    tokenizer, model = load_model(variant)
    model.to(device)
    cache = {}
    cases = cmu.build_cases(cache)
    print(f"[CMU] {len(cases)} usable participant cases")

    correlations = []
    for i, c in enumerate(cases):
        profile_texts = [cmu.paper_text(p) for p in c["profile_papers"]]
        profile_emb = embed_batch(tokenizer, model, profile_texts, device)
        profile_vec = profile_emb.mean(axis=0)
        profile_vec /= (np.linalg.norm(profile_vec) or 1)

        cand_texts = [cmu.paper_text(p) for p, _ in c["candidates"]]
        cand_emb = embed_batch(tokenizer, model, cand_texts, device)
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
    return float(np.mean(correlations)), len(correlations)


def eval_lr_bench(variant, device):
    tokenizer, model = load_model(variant)
    model.to(device)
    cases = lrb.load_cases()
    print(f"[LR-Bench] {len(cases)} pairwise comparisons")

    unique = lrb.all_unique_papers(cases)
    keys = list(unique.keys())
    print(f"Embedding {len(keys)} unique papers...")
    embs = []
    batch_size = 64
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        texts = [
            (unique[k].get("title") or unique[k].get("paper_title") or "")
            + tokenizer.sep_token + (unique[k].get("abstract") or "")
            for k in batch_keys
        ]
        embs.append(embed_batch(tokenizer, model, texts, device))
        if (i // batch_size + 1) % 20 == 0:
            print(f"  ...{i + len(batch_keys)}/{len(keys)} papers embedded")
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    key_to_row = {k: i for i, k in enumerate(keys)}

    def sim_fn(profile_papers, single_paper):
        prof_idx = [key_to_row[lrb.paper_text(p)] for p in profile_papers]
        prof_vec = embs[prof_idx].mean(axis=0)
        prof_vec /= (np.linalg.norm(prof_vec) or 1)
        cand_vec = embs[key_to_row[lrb.paper_text(single_paper)]]
        return float(cand_vec @ prof_vec)

    return lrb.pairwise_accuracy(cases, sim_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["authors", "simcite"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    cmu_corr, cmu_n = eval_cmu(args.variant, device)
    print(f"\n[CMU] {args.variant}: mean Spearman correlation = {cmu_corr:.3f} (n={cmu_n})")

    lrb_acc, lrb_n = eval_lr_bench(args.variant, device)
    print(f"[LR-Bench] {args.variant}: pairwise accuracy = {lrb_acc:.3f} (n={lrb_n})")

    out = {
        "variant": args.variant,
        "cmu_mean_spearman": cmu_corr,
        "cmu_n": cmu_n,
        "lr_bench_pairwise_accuracy": lrb_acc,
        "lr_bench_n": lrb_n,
    }
    out_path = DATA_DIR / f"finetuned_eval_{args.variant}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
