"""
Two-tower feature tables: what each paper / reviewer looks like as numbers.

Paper side (one row per paper):
  - SPECTER2 fingerprint of title+abstract (already computed; we only look it up)
  - OpenAlex primary subfield + primary topic (categorical ids)
  - publication year
  - NOT citation count (a new submission has zero citations; using it would cheat)

Reviewer side (one row per person, plus their paper list):
  - Flat mean of their papers' SPECTER2 fingerprints
  - Time-weighted mean as of a given year T: recent papers count more,
    papers published after T are ignored (so we don't peek into the future)
  - Counts: n papers, year span, total citations of *their* papers (seniority),
    how many distinct subfields (diversity), most-common subfield

Qwen topic/method/application tags are intentionally NOT here — those are
pairwise (paper vs reviewer) and belong in the Stage-2 re-ranker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
TOWER_DIR = DATA_DIR / "tower"

DEFAULT_HALF_LIFE = 3.0
AS_OF_YEARS = list(range(2015, 2027))  # corpus years; index 0 = 2015


def recency_weights(paper_years, as_of_year, half_life_years=DEFAULT_HALF_LIFE):
    """Weight 1.0 at year T, halved every `half_life_years`. Future papers: 0."""
    years = np.asarray(paper_years, dtype=np.float64)
    age = np.clip(float(as_of_year) - years, 0.0, None)
    weights = np.power(0.5, age / float(half_life_years))
    weights[years > as_of_year] = 0.0
    return weights.astype(np.float32)


def weighted_mean(vectors, weights, fallback=None):
    """Weighted average of rows. If all weights are 0, use fallback or unweighted mean."""
    vectors = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    total = float(weights.sum())
    if total <= 1e-8:
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32)
        return vectors.mean(axis=0).astype(np.float32)
    return (vectors * weights[:, None]).sum(axis=0) / total


def l2_normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return vec
    return vec / n


class TowerFeatureStore:
    """Load the tables written by build_tower_features.py."""

    def __init__(self, tower_dir=None):
        self.dir = Path(tower_dir) if tower_dir else TOWER_DIR
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No tower features at {self.dir}. Run spike/build_tower_features.py first."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.paper_ids = json.loads((self.dir / "paper_ids.json").read_text(encoding="utf-8"))
        self.reviewer_ids = json.loads((self.dir / "reviewer_ids.json").read_text(encoding="utf-8"))
        self.paper_id_to_row = {pid: i for i, pid in enumerate(self.paper_ids)}
        self.reviewer_id_to_row = {rid: i for i, rid in enumerate(self.reviewer_ids)}
        self.subfield_vocab = json.loads((self.dir / "subfield_vocab.json").read_text(encoding="utf-8"))
        self.topic_vocab = json.loads((self.dir / "topic_vocab.json").read_text(encoding="utf-8"))

        self.paper_embeddings = np.load(self.dir / "paper_embeddings.npy", mmap_mode="r")
        self.paper_year = np.load(self.dir / "paper_year.npy")
        self.paper_subfield_id = np.load(self.dir / "paper_subfield_id.npy")
        self.paper_topic_id = np.load(self.dir / "paper_topic_id.npy")

        self.reviewer_offsets = np.load(self.dir / "reviewer_offsets.npy")
        self.reviewer_paper_idx = np.load(self.dir / "reviewer_paper_idx.npy")
        self.reviewer_n_papers = np.load(self.dir / "reviewer_n_papers.npy")
        self.reviewer_year_min = np.load(self.dir / "reviewer_year_min.npy")
        self.reviewer_year_max = np.load(self.dir / "reviewer_year_max.npy")
        self.reviewer_total_citations = np.load(self.dir / "reviewer_total_citations.npy")
        self.reviewer_n_subfields = np.load(self.dir / "reviewer_n_subfields.npy")
        self.reviewer_top_subfield_id = np.load(self.dir / "reviewer_top_subfield_id.npy")
        self.reviewer_emb_flat = np.load(self.dir / "reviewer_emb_flat.npy", mmap_mode="r")
        self.reviewer_emb_weighted = np.load(
            self.dir / "reviewer_emb_weighted.npy", mmap_mode="r"
        )
        self.as_of_years = list(self.meta["as_of_years"])
        self.half_life = float(self.meta["half_life_years"])
        self._as_of_to_index = {int(y): i for i, y in enumerate(self.as_of_years)}

    def reviewer_paper_rows(self, reviewer_row):
        start = int(self.reviewer_offsets[reviewer_row])
        end = int(self.reviewer_offsets[reviewer_row + 1])
        return self.reviewer_paper_idx[start:end]

    def paper_numeric(self, paper_row):
        """Year only. Caller should scale (e.g. (year-2015)/11) in the model."""
        return {
            "year": int(self.paper_year[paper_row]),
            "subfield_id": int(self.paper_subfield_id[paper_row]),
            "topic_id": int(self.paper_topic_id[paper_row]),
        }

    def reviewer_numeric(self, reviewer_row):
        y0 = int(self.reviewer_year_min[reviewer_row])
        y1 = int(self.reviewer_year_max[reviewer_row])
        return {
            "n_papers": int(self.reviewer_n_papers[reviewer_row]),
            "year_min": y0,
            "year_max": y1,
            "year_span": y1 - y0,
            "total_citations": int(self.reviewer_total_citations[reviewer_row]),
            "n_subfields": int(self.reviewer_n_subfields[reviewer_row]),
            "top_subfield_id": int(self.reviewer_top_subfield_id[reviewer_row]),
        }

    def reviewer_embedding(self, reviewer_row, as_of_year=None, half_life=None, normalize=True):
        """
        as_of_year=None -> flat mean (every paper equal).
        Otherwise time-weighted as of that year. Uses the precomputed table
        when half-life matches the build; otherwise recomputes from the paper list.
        """
        if as_of_year is None:
            vec = np.array(self.reviewer_emb_flat[reviewer_row], dtype=np.float32, copy=True)
        else:
            hl = self.half_life if half_life is None else float(half_life)
            pre_i = self._as_of_to_index.get(int(as_of_year))
            if pre_i is not None and abs(hl - self.half_life) < 1e-6:
                vec = np.array(
                    self.reviewer_emb_weighted[pre_i, reviewer_row],
                    dtype=np.float32,
                    copy=True,
                )
            else:
                rows = self.reviewer_paper_rows(reviewer_row)
                embs = np.array(self.paper_embeddings[rows], dtype=np.float32)
                years = self.paper_year[rows]
                w = recency_weights(years, as_of_year, hl)
                vec = weighted_mean(embs, w, fallback=self.reviewer_emb_flat[reviewer_row])
        if normalize:
            vec = l2_normalize(vec)
        return vec
