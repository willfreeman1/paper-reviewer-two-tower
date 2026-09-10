"""
Topic / method / application tags with self-hosted Qwen3 (title + abstract).
Supports a small smoke test and a resumable full-catalog run.

Thinking mode is off — we want a short JSON answer, not a hidden
reasoning dump.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """You are analyzing a computer science research paper to extract a \
structured expertise profile. Extract exactly three fields into a JSON object:

1. "topics": 3-5 high-level research topics (e.g. "graph neural networks", \
"distributed systems")
2. "methodologies": 3-5 specific algorithmic or technical approaches used \
(e.g. "contrastive learning", "Byzantine fault tolerance")
3. "applications": 3-5 specific application domains or use cases \
(e.g. "medical image segmentation", "recommender systems")

Each item should be a short phrase (2-5 words). If a field cannot be \
supported by clear evidence in the title/abstract, return an empty list for \
that field rather than guessing or inventing content.

Return ONLY a JSON object with exactly these three keys: "topics", \
"methodologies", "applications". No other text."""


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Prefer the object that actually contains "topics". First "{" is often
    # leftover LaTeX in the abstract (e.g. $\mathbf{94\%}$) if decoding
    # accidentally includes prompt tokens.
    key_idx = text.find('"topics"')
    if key_idx != -1:
        start = text.rfind("{", 0, key_idx)
    else:
        start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    for key in ("topics", "methodologies", "applications"):
        val = obj.get(key, [])
        if not isinstance(val, list):
            val = []
        obj[key] = [str(x).strip() for x in val if str(x).strip()]
    return {k: obj[k] for k in ("topics", "methodologies", "applications")}


def load_papers(limit=None, skip_ids=None, path=None):
    skip_ids = skip_ids or set()
    path = Path(path) if path else DATA_DIR / "papers_final.jsonl"
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if not p.get("title") or not p.get("abstract"):
                continue
            pid = p.get("id") or p.get("text_key")
            if pid in skip_ids:
                continue
            p["id"] = pid
            papers.append(p)
            if limit is not None and len(papers) >= limit:
                break
    return papers


def load_done_ids(path):
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("paper_id") and "profile" in rec:
                    done.add(rec["paper_id"])
            except json.JSONDecodeError:
                continue
    return done


def load_model():
    print(f"Loading {MODEL_NAME} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model ready in {time.time() - t0:.0f}s. cuda={torch.cuda.is_available()}", flush=True)
    return tokenizer, model


def build_prompt(tokenizer, title, abstract):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_one(tokenizer, model, title, abstract):
    text = build_prompt(tokenizer, title, abstract)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_json_object(raw), raw


def extract_batch(tokenizer, model, papers):
    texts = [build_prompt(tokenizer, p["title"], p["abstract"]) for p in papers]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    results = []
    # Use the padded prompt length, not attention_mask.sum(). With left
    # padding, mask-sum is the *unpadded* token count, so decoding from there
    # includes leftover prompt text (title/abstract) and any "{" in LaTeX.
    padded_prompt_len = inputs["input_ids"].shape[1]
    for i, p in enumerate(papers):
        raw = tokenizer.decode(out[i][padded_prompt_len:], skip_special_tokens=True)
        try:
            results.append((parse_json_object(raw), raw, None))
        except Exception as e:
            results.append((None, raw, e))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only process this many papers and print them")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--papers", default=str(DATA_DIR / "papers_final.jsonl"),
                    help="jsonl with id/title/abstract (and optional text_key)")
    ap.add_argument("--out", default=str(DATA_DIR / "aspect_profiles_qwen3.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out)
    done = set() if args.smoke else load_done_ids(out_path)
    limit = args.smoke if args.smoke else None
    papers = load_papers(limit=limit, skip_ids=done, path=Path(args.papers))
    print(f"Papers to process: {len(papers)} (already done: {len(done)})", flush=True)
    if not papers:
        print("Nothing to do.")
        return

    tokenizer, model = load_model()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    n_ok = 0
    n_err = 0
    t0 = time.time()
    done_n = 0

    out_f = None if args.smoke else open(out_path, "a", encoding="utf-8")
    try:
        bs = max(1, args.batch_size)
        for start in range(0, len(papers), bs):
            batch = papers[start : start + bs]
            if bs == 1:
                p = batch[0]
                try:
                    profile, raw = extract_one(tokenizer, model, p["title"], p["abstract"])
                    packed = [(profile, raw, None)]
                except Exception as e:
                    packed = [(None, None, e)]
            else:
                packed = extract_batch(tokenizer, model, batch)
            for p, (profile, raw, err) in zip(batch, packed):
                done_n += 1
                if err or profile is None:
                    n_err += 1
                    print(f"  ERROR {p.get('id')}: {err}", flush=True)
                    continue
                n_ok += 1
                if args.smoke:
                    print(f"\n[{done_n}/{len(papers)}] {p['title'][:90]}")
                    print(f"  Topics:        {profile['topics']}")
                    print(f"  Methodologies: {profile['methodologies']}")
                    print(f"  Applications:  {profile['applications']}")
                else:
                    row = {"paper_id": p["id"], "profile": profile}
                    if p.get("text_key"):
                        row["text_key"] = p["text_key"]
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
            elapsed = time.time() - t0
            rate = done_n / elapsed if elapsed else 0
            eta_s = (len(papers) - done_n) / rate if rate else 0
            print(
                f"  ...{done_n}/{len(papers)}  {rate:.2f}/s  "
                f"ETA {eta_s/60:.1f} min  ok={n_ok} err={n_err}",
                flush=True,
            )
    finally:
        if out_f:
            out_f.close()

    elapsed = time.time() - t0
    print(f"\nDone. {n_ok} ok, {n_err} errors, {elapsed:.1f}s ({(n_ok+n_err)/max(elapsed,1e-6):.2f}/s)")
    if not args.smoke:
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
