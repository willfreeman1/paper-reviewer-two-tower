"""
Dry run of P2R-style structured aspect extraction (Topics / Methodologies /
Applications) on a small sample of papers, to spot-check quality before
committing to a full-corpus batch job.

Prompt adapted from P2R (arXiv:2604.05866), verified against the paper
directly: strict JSON schema, empty-list-if-unsupported to reduce
hallucination, low temperature for determinism.
"""
import io
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_SIZE = 40
MODEL = "gpt-5-mini"
SEED = 42

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


def load_sample():
    papers = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("title") and p.get("abstract"):
                papers.append(p)
    random.seed(SEED)
    return random.sample(papers, SAMPLE_SIZE)


def extract_aspects(client, title, abstract):
    user_content = f"Title: {title}\n\nAbstract: {abstract}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    papers = load_sample()
    print(f"Loaded {len(papers)} sample papers. Extracting aspects with {MODEL}...\n")

    results = []
    for i, p in enumerate(papers):
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        try:
            profile = extract_aspects(client, title, abstract)
        except Exception as e:
            print(f"  [{i}] ERROR on paper {p.get('id')}: {e}")
            continue

        rec = {
            "paper_id": p.get("id"),
            "title": title,
            "profile": profile,
        }
        results.append(rec)

        print(f"[{i+1}/{len(papers)}] {title[:80]}")
        print(f"   Topics:        {profile.get('topics')}")
        print(f"   Methodologies: {profile.get('methodologies')}")
        print(f"   Applications:  {profile.get('applications')}")
        print()

    out_path = DATA_DIR / "aspect_dryrun_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")

    # Quick coverage stats
    empty_counts = {"topics": 0, "methodologies": 0, "applications": 0}
    for r in results:
        for k in empty_counts:
            if not r["profile"].get(k):
                empty_counts[k] += 1
    print(f"\nEmpty-field counts (out of {len(results)}): {empty_counts}")


if __name__ == "__main__":
    main()
