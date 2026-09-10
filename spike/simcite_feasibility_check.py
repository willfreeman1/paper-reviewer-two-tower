"""
Phase 1 pre-check: is "SimCite" ground truth even viable on our corpus?

SimCite defines positives as: authors of the top-K most similar CITED papers
for a given paper P. That only works for citations that land on a paper we
already have full data for (embedding + author list) -- i.e., a citation
target that's also in our own papers_final.jsonl pool. This script checks,
cheaply (no API calls, no embeddings yet), how much of our corpus actually
has that kind of "self-contained" citation coverage before we spend any
time building embeddings/training around it.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    papers = {}
    with open(DATA_DIR / "papers_final.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            papers[p["id"]] = p
    print(f"Loaded {len(papers)} papers in final corpus.")

    author_paper_map = json.loads((DATA_DIR / "author_paper_map_final.json").read_text(encoding="utf-8"))
    paper_to_authors = {}
    for aid, pids in author_paper_map.items():
        for pid in pids:
            paper_to_authors.setdefault(pid, []).append(aid)
    print(f"Loaded {len(author_paper_map)} final authors, "
          f"{len(paper_to_authors)} papers with a known kept-author.")

    n_with_refs = 0
    n_with_any_incorpus_ref = 0
    n_with_authored_incorpus_ref = 0  # in-corpus ref whose author is one of our kept authors
    incorpus_ref_counts = []
    authored_incorpus_ref_counts = []

    for pid, p in papers.items():
        refs = p.get("referenced_works") or []
        if not refs:
            continue
        n_with_refs += 1
        incorpus_refs = [r for r in refs if r in papers]
        if incorpus_refs:
            n_with_any_incorpus_ref += 1
        incorpus_ref_counts.append(len(incorpus_refs))

        authored_refs = [r for r in incorpus_refs if r in paper_to_authors]
        if authored_refs:
            n_with_authored_incorpus_ref += 1
        authored_incorpus_ref_counts.append(len(authored_refs))

    n_total = len(papers)
    print(f"\nPapers with any referenced_works listed: {n_with_refs}/{n_total} "
          f"({100 * n_with_refs / n_total:.1f}%)")
    print(f"Papers with >=1 reference that's ALSO in our own corpus: "
          f"{n_with_any_incorpus_ref}/{n_total} ({100 * n_with_any_incorpus_ref / n_total:.1f}%)")
    print(f"Papers with >=1 in-corpus reference AUTHORED by one of our kept "
          f"authors (i.e., a usable SimCite positive source): "
          f"{n_with_authored_incorpus_ref}/{n_total} "
          f"({100 * n_with_authored_incorpus_ref / n_total:.1f}%)")

    def pct_at_least(counts, k):
        return sum(1 for c in counts if c >= k) / len(counts) if counts else 0

    print("\nDistribution of in-corpus reference count (papers with >=1 ref listed, n="
          f"{len(incorpus_ref_counts)}):")
    for k in [1, 2, 3, 5, 10]:
        print(f"  >= {k} in-corpus refs: {100 * pct_at_least(incorpus_ref_counts, k):.1f}%")

    print("\nDistribution of AUTHORED in-corpus reference count (the ones that "
          "actually matter for SimCite positives):")
    for k in [1, 2, 3, 5, 10]:
        print(f"  >= {k} authored in-corpus refs: {100 * pct_at_least(authored_incorpus_ref_counts, k):.1f}%")

    avg_authored = sum(authored_incorpus_ref_counts) / max(1, len(authored_incorpus_ref_counts))
    print(f"\nAvg authored in-corpus refs per paper (among papers with any refs): {avg_authored:.2f}")

    print("\n===== VERDICT =====")
    usable_frac = n_with_authored_incorpus_ref / n_total
    print(f"{100 * usable_frac:.1f}% of papers in the corpus could get >=1 SimCite "
          f"positive from data we already have (no new API calls needed).")
    if usable_frac < 0.15:
        print("-> LOW coverage: SimCite as designed (top-10 similar, in-corpus only) "
              "is likely not viable at this scale without pulling more citation-target "
              "papers. Consider: (a) relaxing 'top-10' to 'top-K available' per paper, "
              "(b) fetching missing cited-paper metadata for a sample, or (c) dropping "
              "SimCite in favor of a different second ground truth.")
    elif usable_frac < 0.4:
        print("-> MODERATE coverage: viable for a subset of papers. Training would need "
              "to restrict to the subset with coverage, or mix with Authors-derived pairs "
              "as a fallback for papers with 0 SimCite positives.")
    else:
        print("-> GOOD coverage: SimCite is viable as designed on this corpus as-is.")


if __name__ == "__main__":
    main()
