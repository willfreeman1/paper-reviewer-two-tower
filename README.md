# Paper–reviewer matching

This is a **portfolio project.** I built it to show two skills:

- **Two-tower modelling:** give each side (here, the new paper and the candidate reviewer) its own stamp, then rank people by how close those stamps are. Fast enough to look at thousands of names.
- **Learning to rank:** a second model that looks at this paper and this
person *together* and reorders a shortlist. Same family as a lot of
web search ranking.

I set out to beat a strong off-the-shelf scientific-paper reader
(**SPECTER2**: title + abstract in, fingerprint out). To train anything,
you need an **outcome**: a label for “this person is a good reviewer for
this paper.” The outcome I actually care about is **human expertise** —
someone saying they know the work well enough to review it, or picking
A over B. Those ratings exist, but only in two small quizzes (58 people
on one; about 1,400 pairs on the other). That is enough to *check* a
model. It is not enough to *train* one on 15,000 names.

So I built a large homemade answer key in the style other researchers
call **SimCite**: for a paper, look at the bibliography, find similar
cited papers in the catalog, and treat those authors as “a right
person.” Cheap, automatic, tens of thousands of examples. It is **not**
the same thing as human expertise. It is a citation-shaped stand-in.
Training on it made the add-on and the reorder tree better at that
stand-in. It did **not** make them better than untouched SPECTER2 at
matching real human choices. Counting distinctive words, with no
training, **did** beat SPECTER2 on the larger human quiz.

That is the binding constraint: I never had a training outcome that
transferred. I tried rewriting SimCite (newer cites, matching topics,
matching methods) and lightly retraining SPECTER2 itself. None of the
models I trained beat SPECTER2 on the humans. Asking a language model
to score pairs was a wash (different pair count; the other quiz went
the other way). I stopped rather than keep optimizing a circular label.

The interesting result is not “my model wins.” It is that I built
retrieve-and-reorder, found that the labels were teaching “who got
cited,” measured that against humans, and stopped.

Longer write-up: [docs/how-it-works.md](docs/how-it-works.md). Official
numbers: [results/scoreboard.json](results/scoreboard.json).

## The problem

Conferences get a pile of new computer-science papers and have to pick
people who know enough to review them. This project scores **expertise
fit** from **title + abstract** (I do not read the PDF). Each candidate
reviewer is represented by **their own papers**. A real chair would
still handle conflicts of interest, workload, and seniority.

It is a **retrieve-then-reorder** matcher:

1. A fast first pass stamps the paper and each person separately, then
   ranks about 15,000 people by how close those stamps are.
2. A slower second pass looks at this paper and this person **together**
   and reorders the first 100 names. It cannot fetch someone the first
   pass never proposed.

The first-pass stamp starts from SPECTER2. I do not retrain SPECTER2.
I trained a small add-on on top of it, then a tree that reorders the
shortlist.

## What I found

Two human tests. **LR-Bench** is an A-vs-B quiz (chance is 50%). **CMU**
is 58 people rating papers 1–5; the number is how well my ranking
matches theirs (−1 to +1; 0 means no relationship).


| Method                                                                   | Trained on            | LR-Bench                | CMU   |
| ------------------------------------------------------------------------ | --------------------- | ----------------------- | ----- |
| Word overlap (rare words count more)                                     | nothing               | **75.8%** (1,392 pairs) | 0.427 |
| Untouched SPECTER2 (average a person’s paper fingerprints, then compare) | nothing               | 73.2% (1,392 pairs)     | 0.434 |
| First pass (SPECTER2 plus a small add-on)                                | homemade citation key | 72.1%                   | 0.381 |
| Second pass tree, “did they cite them?” hidden                           | homemade citation key | 70.8%                   | 0.438 |
| Same tree, homemade key rebuilt with recency / topic / method            | that new key          | 71.3%                   | 0.435 |
| Language-model votes alone                                               | nothing               | 73.8% (1,244 pairs)     | 0.412 |


Nothing **I trained** beat untouched SPECTER2 on these human tests.
**Word overlap did** on LR-Bench (and nearly tied on CMU). That gap is
larger than any trained-stage gap vs SPECTER2. Word overlap **inside
the tree** is not the same thing: the tree was trained on the homemade
citation key, then lost on LR-Bench (70.8–71.3%).

The language-model row is votes **alone** (four made-up personas, then
the average). It uses fewer pairs than SPECTER2, because tied votes
were skipped. It is a wash, not a win. I did **not** test the original
plan: votes only on the last 15 names, then into the tree as one more
clue. I did not buy thousands of those votes for the whole catalog.

Those human quizzes score a **tiny list the humans already gave** (about
10 papers on CMU; two options on LR-Bench). They are not “search 15,000
people, then reorder 100.”

I also ran a **separate quiz on my catalog** using that same
SimCite key (about 193,000 computer-science papers and 15,000 people).
There are no human ratings on that pile, which is why I used the
key in the first place. I then asked, for 1,500 papers the model had
not trained on: did **at least one** of those homemade right people
appear in the top 10 names I recommended?

Untouched SPECTER2 got that right **47%** of the time. My trained first
pass got **51%**. The second pass, **without** being allowed to see “does
this paper cite this person?”, got **60.9%** — a real jump on this exam,
and not cheating with the key. When I **did** let the tree see that
citation clue, it got **77.3%**. That is also the ceiling: the second
pass only reorders the first 100 names, and a homemade right person was
already in that stack 77.3% of the time. If they were, the tree that
could see “did they get cited?” almost always promoted them into the
top 10. That clue is nearly the answer key, because the key *is*
“authors of similar papers this one cited.” Useful in a real chair’s
tools; inflated as a score on **this** quiz. That is why I always
report the version with citation clues hidden, and why this catalog
quiz is **not** the headline. When it disagrees with the humans,
believe the humans.

## What’s in this repo

- `spike/` — Python scripts (pull, train, score)
- `results/` — frozen scores and copies of the small result files
- `docs/how-it-works.md` — methods, caveats, and how to read the numbers
- `requirements.txt` and `.env.example`

**Not** in this repo: the 193,000-paper catalog, fingerprint files,
trained mixer/tree weights, API keys, or copies of other people’s
survey data.

## Setup

Python 3.10+ with a virtual environment. From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys only if you call OpenAlex / OpenAI
```

SPECTER2 downloads from Hugging Face the first time you run it
(`allenai/specter2_base` plus the `allenai/specter2` proximity adapter).

## Re-running the human quizzes

Word overlap on the CMU survey (58 people) is the part a clone can
re-run today:

1. Download [CMU gold](https://github.com/niharshah/goldstandard-reviewer-paper-match)
   and unpack it to `spike/data/gold_cmu/`.
2. `python spike/cmu_gold_eval.py --method tfidf`

SPECTER2 on the same files: `--method specter2`.

**LR-Bench** ([Hugging Face](https://huggingface.co/datasets/Gnociew/LR-bench))
is scored by `spike/lr_bench_eval.py`, but this repo does not yet include
a converter from the Hugging Face download to the two local JSON files
that script expects. I am not uploading my copies (other people’s
surveys). Both human sets are for **research**, not operational reviewer
assignment.

Rebuilding the OpenAlex catalog and retraining is a separate, slow
recipe (`spike/pull_openalex.py` and the train scripts). One command
does not reproduce the 193k-paper numbers.

## Data I used (linked, not re-hosted)

- [OpenAlex](https://openalex.org/) — catalog of papers and authors
- [SPECTER2](https://huggingface.co/allenai/specter2) — paper fingerprints
- [CMU gold](https://github.com/niharshah/goldstandard-reviewer-paper-match) — 1–5 self-ratings
- [LR-Bench](https://huggingface.co/datasets/Gnociew/LR-bench) — pairwise A-vs-B ratings

