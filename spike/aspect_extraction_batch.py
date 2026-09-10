"""
Full-corpus P2R-style structured aspect extraction (Topics / Methodologies /
Applications), concurrent + resumable, same pattern as pull_openalex.py:
thread pool for concurrency, a rate limiter to stay under the account's
token/request limits, and append-only checkpointing so an interrupted run
can resume without redoing already-processed papers.
"""
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_FILE = DATA_DIR / "aspect_profiles.jsonl"
MODEL = "gpt-5-mini"
REASONING_EFFORT = "minimal"
MAX_WORKERS = 25
MAX_RETRIES = 5

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


class RateLimiter:
    """Thread-safe token-bucket-ish limiter on requests/sec."""

    def __init__(self, rate_per_sec):
        self.rate = rate_per_sec
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            min_interval = 1.0 / self.rate
            wait_time = self.last + min_interval - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.monotonic()
            self.last = now


write_lock = threading.Lock()
limiter = RateLimiter(rate_per_sec=20)  # well under the 5000/min = 83/sec cap


def load_papers():
    papers = []
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("title") and p.get("abstract"):
                papers.append(p)
    return papers


def load_done_ids():
    if not CHECKPOINT_FILE.exists():
        return set()
    done = set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["paper_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def extract_one(client, paper):
    user_content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=REASONING_EFFORT,
            )
            profile = json.loads(resp.choices[0].message.content)
            return {"paper_id": paper["id"], "profile": profile}
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    return {"paper_id": paper["id"], "error": str(last_err)}


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    papers = load_papers()
    done_ids = load_done_ids()
    todo = [p for p in papers if p["id"] not in done_ids]
    print(f"Total papers: {len(papers)}. Already done: {len(done_ids)}. Remaining: {len(todo)}.")

    if not todo:
        print("Nothing to do.")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_done = 0
    n_errors = 0
    start = time.time()

    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_one, client, p): p for p in todo}
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                n_done += 1
                if "error" in rec:
                    n_errors += 1
                if n_done % 200 == 0:
                    elapsed = time.time() - start
                    rate = n_done / elapsed
                    remaining = len(todo) - n_done
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"  ...{n_done}/{len(todo)} done ({rate:.1f}/s, "
                        f"ETA {eta/60:.1f} min, {n_errors} errors so far)"
                    )

    elapsed = time.time() - start
    print(f"\nDone. {n_done} processed in {elapsed/60:.1f} min. {n_errors} errors.")
    print(f"Output: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
