"""
Embed papers_final.jsonl with off-the-shelf SPECTER2 (GPU recommended).

Writes paper_embeddings_full.npy and paper_embedding_ids_full.json for
SimCite pair construction.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 128
MAX_LENGTH = 512


def load_papers():
    papers = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def load_specter2(device):
    print("Loading SPECTER2 (base + proximity adapter)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()
    model.to(device)
    print(f"  loaded in {time.time() - t0:.1f}s, device={device}")
    return tokenizer, model


def embed_papers(tokenizer, model, papers_list, device, batch_size=BATCH_SIZE):
    ids = []
    texts = []
    for p in papers_list:
        title = p.get("title") or ""
        abstract = p.get("abstract") or ""
        text = title + tokenizer.sep_token + abstract if abstract else title
        ids.append(p["id"])
        texts.append(text)

    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False,
                                max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(emb)
            batch_num = i // batch_size + 1
            if batch_num % 20 == 0 or batch_num == n_batches:
                elapsed = time.time() - t0
                rate = batch_num / elapsed
                eta = (n_batches - batch_num) / rate if rate > 0 else 0
                papers_done = min(batch_num * batch_size, len(texts))
                print(f"  batch {batch_num}/{n_batches}  ({papers_done}/{len(texts)} papers, "
                      f"{elapsed:.0f}s elapsed, {papers_done/elapsed:.1f} papers/s, ETA {eta:.0f}s)",
                      flush=True)
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    return ids, embs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    papers = load_papers()
    print(f"Loaded {len(papers)} papers to embed.")

    tokenizer, model = load_specter2(device)

    t0 = time.time()
    ids, embs = embed_papers(tokenizer, model, papers, device)
    print(f"\nDone embedding in {time.time() - t0:.0f}s. Shape: {embs.shape}")

    np.save(DATA_DIR / "paper_embeddings_full.npy", embs)
    (DATA_DIR / "paper_embedding_ids_full.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved paper_embeddings_full.npy ({embs.nbytes / 1e6:.1f} MB) and "
          f"paper_embedding_ids_full.json.")


if __name__ == "__main__":
    main()
