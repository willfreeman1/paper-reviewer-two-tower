"""Batched Qwen3 extraction via vLLM (same prompt/schema as qwen_aspect_extract.py)."""
import argparse
import json
import re
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(text[:200])
    obj = json.loads(text[start : end + 1])
    out = {}
    for key in ("topics", "methodologies", "applications"):
        val = obj.get(key, [])
        if not isinstance(val, list):
            val = []
        out[key] = [str(x).strip() for x in val if str(x).strip()]
    return out


def load_papers(limit=None):
    papers = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("title") and p.get("abstract"):
                papers.append(p)
            if limit and len(papers) >= limit:
                break
    return papers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    papers = load_papers(limit=args.smoke or None)
    print(f"Loaded {len(papers)} papers", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = []
    for p in papers:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {p['title']}\n\nAbstract: {p['abstract']}"},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    print("Starting vLLM ...", flush=True)
    llm = LLM(model=MODEL_NAME, dtype="bfloat16", max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=256)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    n_ok = 0
    n_err = 0
    for p, out in zip(papers, outputs):
        raw = out.outputs[0].text
        try:
            profile = parse_json_object(raw)
            n_ok += 1
            if args.smoke:
                print(f"{p['title'][:80]}")
                print(f"  {profile}")
        except Exception as e:
            n_err += 1
            print(f"ERR {p['id']}: {e} raw={raw[:180]!r}")
    print(
        f"\nDone. {n_ok} ok, {n_err} err, {elapsed:.1f}s, "
        f"{len(papers)/max(elapsed,1e-6):.2f}/s"
    )


if __name__ == "__main__":
    main()
