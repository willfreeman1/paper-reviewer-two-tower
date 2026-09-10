"""
Stage 3: four-persona committee on one paper + one reviewer profile.

One API call returns four 1-5 scores (area chair, topics, method,
application). Mean is the committee score. Resume-safe cache.

This is the last cut only — in the full product it would run on Stage 2's
top 15. On the human tests the "shortlist" is already the gold candidates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).parent / "data"
TOWER_DIR = DATA_DIR / "tower"
CACHE_PATH = TOWER_DIR / "committee_cache.jsonl"
MODEL = os.environ.get("STAGE3_MODEL", "gpt-4o-mini")
MAX_PROFILE_PAPERS = 5
ABS_CHARS = 700

SYSTEM = """You are scoring how well a candidate reviewer fits a submitted \
computer-science paper. You will play FOUR personas and return one JSON object.

Personas:
1. area_chair — holistic: could this person write a useful review? Problem \
fit, method fit, and whether their recent work is in the same conversation.
2. topics — only the research question / subject. Ignore methods.
3. methodology — only techniques and models. Ignore the application domain.
4. application — only the use case / domain (medical images, systems, etc.).

Score each persona from 1.0 to 5.0 (decimals allowed):
1 = wrong area, 3 = related but a stretch, 5 = clearly among the right people.

Use the paper tags and reviewer tags when present. Near-synonyms count as a \
match (GNN = graph neural network). Cousins do not (decision tree ≠ GNN).

Return ONLY JSON:
{"area_chair":{"score":0,"why":"one short sentence"},
 "topics":{"score":0,"why":"..."},
 "methodology":{"score":0,"why":"..."},
 "application":{"score":0,"why":"..."}}
"""


def _clip(s, n):
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def paper_title(p):
    return (p.get("title") or p.get("paper_title") or "").strip()


def paper_abs(p):
    return (p.get("abstract") or "").strip()


def format_paper(p, tags=None):
    lines = [f"Title: {paper_title(p)}", f"Abstract: {_clip(paper_abs(p), ABS_CHARS)}"]
    if tags:
        for k, label in (("topics", "Topics"), ("methodologies", "Methods"), ("applications", "Applications")):
            vals = tags.get(k) or []
            if vals:
                lines.append(f"{label}: {', '.join(vals)}")
    return "\n".join(lines)


def format_profile(papers, tag_union=None):
    chunks = []
    for i, p in enumerate(papers[:MAX_PROFILE_PAPERS], 1):
        chunks.append(f"{i}. {paper_title(p)}\n   {_clip(paper_abs(p), 400)}")
    text = "\n".join(chunks) if chunks else "(no papers)"
    if tag_union:
        extra = []
        for k, label in (("topics", "Topics"), ("methodologies", "Methods"), ("applications", "Applications")):
            vals = sorted(tag_union.get(k) or [])[:12]
            if vals:
                extra.append(f"{label}: {', '.join(vals)}")
        if extra:
            text += "\nReviewer tags: " + " | ".join(extra)
    return text


def pair_key(paper, profile_papers, model=MODEL):
    blob = json.dumps(
        {
            "m": model,
            "q": paper_title(paper) + "\n" + paper_abs(paper),
            "r": [(paper_title(p), paper_abs(p)[:200]) for p in profile_papers[:MAX_PROFILE_PAPERS]],
        },
        ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def load_cache(path=CACHE_PATH):
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("key"):
                out[rec["key"]] = rec
    return out


def append_cache(rec, path=CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_scores(raw):
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        raw = json.loads(raw)
    out = {}
    for name in ("area_chair", "topics", "methodology", "application"):
        block = raw.get(name) or {}
        try:
            sc = float(block.get("score"))
        except (TypeError, ValueError):
            sc = None
        if sc is not None:
            sc = max(1.0, min(5.0, sc))
        out[name] = {"score": sc, "why": (block.get("why") or "")[:240]}
    scores = [out[n]["score"] for n in out if out[n]["score"] is not None]
    out["mean"] = float(sum(scores) / len(scores)) if scores else None
    return out


def score_pair(client, paper, profile_papers, paper_tags=None, rev_tags=None, cache=None, model=MODEL):
    key = pair_key(paper, profile_papers, model)
    if cache is not None and key in cache and cache[key].get("scores", {}).get("mean") is not None:
        return cache[key]
    user = (
        "SUBMITTED PAPER\n"
        + format_paper(paper, paper_tags)
        + "\n\nCANDIDATE REVIEWER (their papers)\n"
        + format_profile(profile_papers, rev_tags)
    )
    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content
            scores = parse_scores(text)
            rec = {
                "key": key,
                "model": model,
                "scores": scores,
                "paper_title": paper_title(paper),
            }
            if cache is not None:
                cache[key] = rec
            append_cache(rec)
            return rec
        except Exception as e:
            last_err = e
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"committee call failed: {last_err}")
