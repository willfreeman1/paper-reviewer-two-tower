"""
Small test: does extra text change Topics / Methodology / Application tags?

For ~100 papers that already have a free arXiv PDF:
  A) title + abstract
  B) title + abstract + introduction
  C) title + abstract + full extracted PDF text (capped)

Uses gpt-5-mini with minimal reasoning so we can run it quickly on OpenAI
without standing up a GPU. Cost should stay well under $2.
"""
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "arxiv_pdf_sample"
PDF_DIR.mkdir(exist_ok=True)

MODEL = "gpt-5-mini"
TARGET_N = 100
MAX_INTRO_CHARS = 6000
MAX_FULL_CHARS = 24000
ARXIV_SLEEP = 3.0

SYSTEM_PROMPT = """You are analyzing a computer science research paper to extract a \
structured expertise profile. Extract exactly three fields into a JSON object:

1. "topics": 3-5 high-level research topics (e.g. "graph neural networks", \
"distributed systems")
2. "methodologies": 3-5 specific algorithmic or technical approaches used \
(e.g. "contrastive learning", "Byzantine fault tolerance")
3. "applications": 3-5 specific application domains or use cases \
(e.g. "medical image segmentation", "recommender systems")

Each item should be a short phrase (2-5 words). If a field cannot be \
supported by clear evidence in the provided text, return an empty list for \
that field rather than guessing or inventing content.

Return ONLY a JSON object with exactly these three keys: "topics", \
"methodologies", "applications". No other text."""


def arxiv_id_from_work(w):
    for loc in w.get("locations") or []:
        url = (loc.get("pdf_url") or loc.get("landing_page_url") or "") + " "
        url += ((w.get("open_access") or {}).get("oa_url") or "")
        url += ((w.get("best_oa_location") or {}).get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:pdf|abs|html)/(\d{4}\.\d{4,5})(?:v\d+)?", url, re.I)
        if m:
            return m.group(1)
    doi = (w.get("doi") or "") + " " + ((w.get("ids") or {}).get("doi") or "")
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
    if m:
        return m.group(1)
    return None


def load_candidates():
    works = json.load(open(DATA_DIR / "fulltext_coverage_sample.json", encoding="utf-8"))
    papers = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p

    cands = []
    for w in works:
        aid = arxiv_id_from_work(w)
        p = papers.get(w.get("id"))
        if not aid or not p or not p.get("abstract"):
            continue
        cands.append(
            {
                "paper_id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "arxiv_id": aid,
            }
        )
    # unique by arxiv id
    seen = set()
    uniq = []
    for c in cands:
        if c["arxiv_id"] in seen:
            continue
        seen.add(c["arxiv_id"])
        uniq.append(c)
    return uniq


def download_pdf(arxiv_id):
    dest = PDF_DIR / f"{arxiv_id}.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    contact = (os.environ.get("OPENALEX_MAILTO") or "").strip()
    ua = "paper-reviewer-matching/0.1"
    if contact:
        ua += f" ({contact})"
    headers = {"User-Agent": ua}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def pdf_text(path):
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:20]:  # first 20 pages is enough for intro + most CS papers
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_intro(full_text):
    if not full_text:
        return ""
    # Start at Introduction if we can find it; else start after Abstract.
    start = None
    for pat in [
        r"\n\s*1[\.\s]+Introduction\b",
        r"\n\s*I[\.\s]+Introduction\b",
        r"\n\s*Introduction\b",
    ]:
        m = re.search(pat, full_text, re.I)
        if m:
            start = m.start()
            break
    if start is None:
        m = re.search(r"\n\s*Abstract\b", full_text, re.I)
        start = m.end() if m else 0
    chunk = full_text[start:]
    end = None
    for pat in [
        r"\n\s*2[\.\s]+Related Work\b",
        r"\n\s*2[\.\s]+Background\b",
        r"\n\s*2[\.\s]+Preliminar",
        r"\n\s*II[\.\s]+Related Work\b",
        r"\n\s*Related Work\b",
        r"\n\s*2[\.\s]+\w+",
    ]:
        m = re.search(pat, chunk[80:], re.I)  # skip the intro heading itself
        if m:
            end = 80 + m.start()
            break
    if end:
        chunk = chunk[:end]
    return chunk[:MAX_INTRO_CHARS].strip()


def extract_profile(client, title, body):
    user = f"Title: {title}\n\n{body}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        reasoning_effort="minimal",
    )
    return json.loads(resp.choices[0].message.content)


def norm_tags(tags):
    return {re.sub(r"\s+", " ", (t or "").strip().lower()) for t in (tags or []) if t}


def jaccard(a, b):
    a, b = norm_tags(a), norm_tags(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    cands = load_candidates()
    print(f"Found {len(cands)} arXiv papers in the coverage sample.")

    usable = []
    print("Downloading PDFs and extracting introductions (polite arXiv pacing)...")
    for c in cands:
        if len(usable) >= TARGET_N:
            break
        try:
            time.sleep(ARXIV_SLEEP)
            path = download_pdf(c["arxiv_id"])
            full = pdf_text(path)
            intro = extract_intro(full)
            if len(intro) < 400 or len(full) < 800:
                print(f"  skip {c['arxiv_id']}: too little extracted text")
                continue
            c["intro"] = intro
            c["full"] = full[:MAX_FULL_CHARS]
            usable.append(c)
            if len(usable) % 10 == 0:
                print(f"  ...{len(usable)} usable papers")
        except Exception as e:
            print(f"  skip {c['arxiv_id']}: {e}")

    print(f"\nUsable papers: {len(usable)}")
    if len(usable) < 30:
        print("Not enough usable PDFs to trust the comparison. Stopping.")
        return

    variants = ["abstract", "intro", "full"]
    print(f"Extracting 3 variants x {len(usable)} papers with {MODEL}...")

    def run_one(item):
        title = item["title"]
        out = {"paper_id": item["paper_id"], "arxiv_id": item["arxiv_id"], "title": title}
        try:
            out["abstract"] = extract_profile(
                client, title, f"Abstract: {item['abstract']}"
            )
            out["intro"] = extract_profile(
                client,
                title,
                f"Abstract: {item['abstract']}\n\nIntroduction:\n{item['intro']}",
            )
            out["full"] = extract_profile(
                client,
                title,
                f"Abstract: {item['abstract']}\n\nPaper text:\n{item['full']}",
            )
        except Exception as e:
            out["error"] = str(e)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_one, c) for c in usable]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 10 == 0:
                print(f"  ...{i}/{len(usable)} extracted")

    ok = [r for r in results if "error" not in r]
    print(f"\nSuccessful extractions: {len(ok)} / {len(results)}")

    fields = ["topics", "methodologies", "applications"]
    pairs = [("abstract", "intro"), ("abstract", "full"), ("intro", "full")]
    print("\nMean tag-overlap (1.0 = identical lists, 0.0 = no shared tags):")
    print(f"{'field':<16} {'abs vs +intro':>14} {'abs vs full':>12} {'intro vs full':>13}")
    stats = {}
    for field in fields:
        stats[field] = {}
        row = [field]
        for a, b in pairs:
            scores = [jaccard(r[a].get(field), r[b].get(field)) for r in ok]
            mean = sum(scores) / len(scores)
            stats[field][f"{a}_vs_{b}"] = mean
            row.append(f"{mean:.2f}")
        print(f"{row[0]:<16} {row[1]:>14} {row[2]:>12} {row[3]:>13}")

    print("\nHow often Methodology tags changed at all (overlap < 1.0):")
    for a, b, label in [
        ("abstract", "intro", "abstract -> +intro"),
        ("abstract", "full", "abstract -> full"),
    ]:
        changed = sum(
            1 for r in ok if jaccard(r[a].get("methodologies"), r[b].get("methodologies")) < 1.0
        )
        print(f"  {label}: {changed}/{len(ok)} ({100*changed/len(ok):.0f}%)")

    # Show a few biggest methodology shifts
    scored = []
    for r in ok:
        j = jaccard(r["abstract"].get("methodologies"), r["intro"].get("methodologies"))
        scored.append((j, r))
    scored.sort(key=lambda x: x[0])
    print("\n--- Largest methodology changes after adding the introduction ---")
    for j, r in scored[:8]:
        print(f"\n[{j:.2f}] {r['title'][:90]}")
        print(f"  abstract only: {r['abstract'].get('methodologies')}")
        print(f"  + intro:       {r['intro'].get('methodologies')}")

    print("\n--- Cases where methodology barely changed ---")
    for j, r in scored[-5:]:
        print(f"\n[{j:.2f}] {r['title'][:90]}")
        print(f"  abstract only: {r['abstract'].get('methodologies')}")
        print(f"  + intro:       {r['intro'].get('methodologies')}")

    out_path = DATA_DIR / "aspect_text_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "n": len(ok), "results": results}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
