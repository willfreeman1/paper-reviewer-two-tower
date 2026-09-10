"""
Contrastive fine-tuning from frozen SPECTER2, one run per label variant
(authors / simcite). Same base model, LoRA config, hyperparameters, and
step count — only the training_pairs_*.jsonl file changes.

SPECTER2 encoder weights stay frozen; a small LoRA adapter is trained.
Loss is in-batch-negative contrastive (symmetric InfoNCE): pull matching
(anchor, positive) pairs together, push other pairs in the batch apart.

Needs a GPU.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel, LoRAConfig
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
MAX_LENGTH = 512


def load_paper_text():
    paper_text = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""
            paper_text[p["id"]] = (title, abstract)
    return paper_text


def load_pairs(variant, paper_text):
    pairs = []
    with open(DATA_DIR / f"training_pairs_{variant}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["anchor_id"] in paper_text and d["positive_id"] in paper_text:
                pairs.append((d["anchor_id"], d["positive_id"]))
    return pairs


def to_text(tokenizer, title, abstract):
    return title + tokenizer.sep_token + abstract if abstract else title


def encode(tokenizer, model, ids, paper_text, device, max_length=MAX_LENGTH):
    texts = [to_text(tokenizer, *paper_text[pid]) for pid in ids]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.last_hidden_state[:, 0, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["authors", "simcite"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--max_pairs", type=int, default=None, help="for quick smoke tests")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, variant={args.variant}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, lr={args.lr}, temperature={args.temperature}, "
          f"lora_r={args.lora_r}")

    paper_text = load_paper_text()
    pairs = load_pairs(args.variant, paper_text)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Loaded {len(pairs)} training pairs for variant={args.variant}")

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    adapter_name = f"{args.variant}_contrastive"
    model.add_adapter(adapter_name, config=LoRAConfig(r=args.lora_r, alpha=args.lora_r * 2))
    model.train_adapter(adapter_name)  # freezes base weights, only LoRA params require grad
    model.set_active_adapters(adapter_name)
    model.to(device)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    n_steps_per_epoch = len(pairs) // args.batch_size
    total_steps = n_steps_per_epoch * args.epochs
    print(f"{n_steps_per_epoch} steps/epoch, {total_steps} total steps")

    log_path = DATA_DIR / f"train_log_{args.variant}.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs) - args.batch_size + 1, args.batch_size):
            batch = pairs[i:i + args.batch_size]
            anchor_ids = [a for a, _ in batch]
            positive_ids = [b for _, b in batch]

            anchor_emb = encode(tokenizer, model, anchor_ids, paper_text, device)
            pos_emb = encode(tokenizer, model, positive_ids, paper_text, device)
            anchor_emb = F.normalize(anchor_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)

            logits = anchor_emb @ pos_emb.T / args.temperature
            labels = torch.arange(len(batch), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 20 == 0 or step == total_steps:
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (total_steps - step) / rate if rate > 0 else 0
                msg = (f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                       f"({elapsed:.0f}s elapsed, {rate:.2f} steps/s, ETA {eta:.0f}s)")
                print(msg, flush=True)
                log_f.write(msg + "\n")
                log_f.flush()

    log_f.close()

    out_dir = DATA_DIR / f"adapter_{args.variant}"
    model.save_adapter(str(out_dir), adapter_name)
    print(f"\nSaved adapter to {out_dir}")


if __name__ == "__main__":
    main()
