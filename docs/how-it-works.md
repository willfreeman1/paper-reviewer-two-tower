# How the matcher works

This is the in-depth public write-up. Short version and setup:
[README.md](../README.md). Official numbers:
[results/scoreboard.json](../results/scoreboard.json).

## Project overview

This is a portfolio project. I built it to show two skills:

- **Two-tower modelling:** give each side (here, the new paper and the
  candidate reviewer) its own stamp, then rank people by how close those
  stamps are. Fast enough to look at thousands of names.
- **Learning to rank:** a second model that looks at this paper and this
  person together and reorders a shortlist. Same family as a lot of web
  search ranking.

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

## The job

A conference must assign reviewers to new papers. I only score
**expertise fit**: how well this person’s past work matches this paper.
A real area chair would still handle conflicts of interest, workload,
and seniority.

I use **title + abstract**, not the full text. Each reviewer is a
**profile**: the papers they have written. There is never a single
correct reviewer, so this write-up talks about **a** good match, not
**the** match.

## The two tests in more detail

**CMU** ([Stelmakh et al.](https://github.com/niharshah/goldstandard-reviewer-paper-match)):
58 researchers rated papers they had already read, 1–5 (“how expert am
I on this?”). I ask whether my ranking of those papers agrees with
theirs. That agreement is a **Spearman correlation**, a number from −1
(backwards) to +1 (perfect). 0 means no relationship. About **0.43** is
moderate.

**LR-Bench** ([RATE paper](https://huggingface.co/datasets/Gnociew/LR-bench)):
pairwise “is A a better reviewer than B for this paper?” (sometimes the
other way around: same person, two papers). Chance is 50%. For scale:
SPECTER2 sits around **73%** on this quiz with no extra training.

Both quizzes are for **research**, not for assigning real reviewers.

Because those sets are tiny, I also pulled a large catalog from
[OpenAlex](https://openalex.org/), a free scholarly database:
**193,160** computer-science papers and **15,196** reviewers. There are
no human ratings on that pile, which is why SimCite exists.

When SimCite and the humans disagree, **believe the humans.**

## How I score those two tests

**On the catalog (SimCite).** For 1,500 papers the model did not train
on, did **at least one** homemade right person appear in the **top 10**
names I recommended? Average that yes/no. “Top 10” just means I only
look at the first ten names.

**On CMU.** Rank the papers each person rated; compare that ranking to
their 1–5 scores (the Spearman number above).

**On LR-Bench.** For each A-vs-B pair, did I put the better option
first? Skip ties.

**What the human tests are not.** They score a **tiny list the humans
already gave** (about 10 papers per CMU person; two options on
LR-Bench). They do **not** run “search 15,196 people, then reorder
100.” Only the catalog quiz does that full search. When I later score
the second-pass tree on CMU or LR-Bench, I am asking whether its
*scoring rule* lines up with humans on those short lists — not whether
the full retrieve-then-reorder pipeline found the right people out of
15,000.

Most of the catalog clues (who-cites-whom inside the pile, field codes)
are missing or “no” on those outside papers. A real chair would often
have more of those facts.

## Two passes

Think of assigning a newspaper story to a reporter.

**First pass.** Looks at the new paper and at each person’s past papers
separately. Each side gets a stamp (a list of numbers). Rank by how
close the two stamps are. From about **15,000** people in the catalog,
it hands the second pass a stack of **100** names. Fast. Cannot look at
the pair side by side.

The stamp starts from SPECTER2. Similar papers get similar stamps. I do
**not** retrain it.

Headline recipe: fingerprint each of a person’s papers, **average those
fingerprints with equal weight**, then compare to the new paper. I also
tried “closest paper only” and “average of the three closest” (common
in the papers that published these quizzes). Those agreed more with the
CMU 1–5 ratings, but did not lift LR-Bench, so averaging stays the
headline.

I trained a **small add-on** on top of SPECTER2: start from that
fingerprint, nudge it with extra facts (field, year, how senior they
are), and keep the result on the same scale. Day one of training *is*
untouched SPECTER2. I trained that add-on on SimCite.

**Second pass.** Looks at **this paper and this person together**. Uses
hand-built clues: shared words, shared field, overlapping citations,
topic/method tags, first-pass rank, and so on. It is a tree model (a
stack of yes/no questions over those clues; the same family as a lot of
web search ranking). It **only reorders** the first 100 names. If a
homemade right person never made that stack, this pass cannot help. On
the catalog quiz that happens often enough that the second pass cannot
score above **77.3%**, no matter how clever it is.

I trained two copies of the tree, both on SimCite. One was allowed to
see “does this paper cite this person?” The other was not. The first
copy is useful in a real chair’s tools (cited authors are often
wanted). It is **not** a fair score on SimCite, because that quiz *is*
“authors of similar papers this one cited.” I always report the copy
with those clues hidden.

**Optional last cut.** Four made-up personas (area chair, topics,
methods, applications) each give a 1–5 vote from a language model. I
tested those votes as a **standalone scorer** on CMU and LR-Bench. I did
**not** test the original plan (votes only on the last 15 names, then
into the tree as one more clue). I did not buy thousands of votes for
the whole catalog.

## Scoreboard (human tests)

Copied from `results/scoreboard.json`. **TF-IDF** here means word
overlap that weights rare words more than “the” or “model.”

| Method | Trained on | LR-Bench | CMU |
|---|---|---|---|
| Word overlap (TF-IDF) | nothing | **75.8%** (1,392 pairs) | 0.427 (58 people) |
| Untouched SPECTER2 | nothing | 73.2% (1,392) | 0.434 |
| First pass (SPECTER2 plus add-on) | SimCite | 72.1% | 0.381 |
| Second pass, citation clues hidden | SimCite | 70.8% | 0.438 |
| Same tree, SimCite rebuilt with recency / topic / method | that new key | 71.3% | 0.435 |
| Language-model votes alone (`gpt-4o-mini`) | nothing | 73.8% (1,244 decided / 1,396) | 0.412 |

Nothing **I trained** beat untouched SPECTER2 on the human tests;
**word overlap did** on LR-Bench (and nearly tied on CMU). That
LR-Bench gap is larger than any trained-stage gap vs SPECTER2.

The language-model row uses a **different pair count** than SPECTER2
(ties skipped when the four votes were a draw). It is a wash, not a
win. CMU went the other way.

Word overlap **inside the tree** is not the same as word overlap
**alone**. The tree was trained on SimCite, then lost on LR-Bench
(70.8–71.3%). By itself, counting distinctive words is the strongest
frozen baseline on that quiz.

## Catalog quiz (not the headline)

These were trained as if SimCite were the truth, then scored on that
same key: did a homemade right person land in the top 10?

| Method | Homemade right person in the top 10 |
|---|---|
| Untouched SPECTER2 | 47% |
| Trained first pass | 51% |
| Second pass, citation clues hidden | 60.9% |
| Second pass, citation clues on | 77.3% (hits the cap) |

The 77.3% tree’s most used clue, by a huge margin, was “does the paper
cite this person?” Useful as a product feature; inflated as a score on
**this** exam.

I also listed every cited author I could find and compared that list to
the CMU 1–5 ratings. Only **15%** of people who had rated themselves 4
or 5 appeared. Nobody who had rated themselves 1 or 2 appeared. The
bibliography points the right way and still misses most people who said
they were a good fit. That was a name-list check, not a model score.

## What I tried and stopped

- Lightly retraining SPECTER2 itself on “same author” or SimCite pairs
  did not beat untouched SPECTER2 on CMU or LR-Bench.
- Rebuilding SimCite so newer cites, matching topics, or matching
  methods counted more, then retraining, still did not beat **73.2%**
  SPECTER2 on LR-Bench — and did not beat **75.8%** word overlap.
- A fancier keyword score (BM25, common in search engines) lost to both
  word overlap and SPECTER2. I dropped it.
- I did not grow a bigger citation graph. I did not buy catalog-scale
  language-model votes.

In short, I built retrieve-and-reorder, found that the training labels
were teaching “who got cited,” measured that against humans, and
stopped.

## Scripts (what lives in `spike/`)

Every file in `spike/` is part of building the catalog, scoring the
human quizzes, or training/scoring the two passes.

**Catalog (OpenAlex → cleaned CS pool)**

| Script | Role |
|---|---|
| `pull_openalex.py` | Pull papers and authors from OpenAlex |
| `dedup_papers.py` | Merge reprint/published copies of the same work |
| `filter_cs_only.py` | Keep papers whose primary field is computer science |
| `disambiguation_check.py` | Flag likely merged author identities |
| `apply_disambiguation_filter.py` | Drop those authors; write the final catalog files |
| `embed_and_eval.py` | SPECTER2 fingerprints + a small catalog self-test |
| `embed_full_corpus.py` | SPECTER2 fingerprints for the whole catalog |
| `tfidf_selfrecall.py` | Same catalog self-test with word overlap |

**Human quizzes and extra facts on those papers**

| Script | Role |
|---|---|
| `extract_cmu.py` | Unpack the CMU zip into `spike/data/gold_cmu/` |
| `cmu_gold_eval.py` | Word overlap / SPECTER2 / BM25 on the CMU 1–5 survey |
| `lr_bench_eval.py` | Same methods on the LR-Bench A-vs-B quiz |
| `eval_specter2_pooling.py` | How to combine a person’s many paper fingerprints |
| `collect_gold_papers.py` | Unique gold papers for OpenAlex lookup |
| `enrich_gold_openalex.py` | Field / year / citations for those papers |
| `fetch_gold_cited_works.py` | Bibliography records for the cite-all name list |
| `eval_simcite_gold_agreement.py` | Do cited authors show up among the human 4–5s? |

**Homemade citation key (SimCite) and topic/method tags**

| Script | Role |
|---|---|
| `build_simcite_pairs.py` | Authors of similar cited papers → training pairs |
| `build_simcite_weighted.py` | Same key with recency / topic / method weights |
| `qwen_aspect_extract.py` | Topic / method / application tags (title + abstract) |
| `qwen_setup.sh` | GPU environment for that tagger |
| `fulltext_coverage_check.py` | How often a catalog paper has a free PDF |
| `aspect_text_ablation.py` | Does extra PDF text change those tags? |

**First pass, second pass, votes, and the SPECTER2 retraining try**

| Script | Role |
|---|---|
| `tower_features.py` | What each paper / person looks like as numbers |
| `build_tower_features.py` | Build those tables from the catalog |
| `two_tower_model.py` | The small add-on on top of SPECTER2 |
| `train_two_tower.py` | Train that add-on; catalog + human scores |
| `train_reranker.py` | Train the second-pass tree on the catalog shortlist |
| `eval_reranker_gold.py` | Score that tree on CMU and LR-Bench |
| `run_simcite_ablation.py` | Retrain on the weighted keys and score humans |
| `stage3_committee.py` | Four-persona language-model votes |
| `eval_stage3_gold.py` | Those votes on CMU and LR-Bench |
| `build_training_pairs.py` | Same-author vs SimCite pairs to retrain SPECTER2 |
| `train_contrastive.py` | Light SPECTER2 adapter training (needs a GPU) |
| `eval_finetuned.py` | Score that adapter on CMU and LR-Bench |
| `lambda_setup.sh` | GPU environment for embedding / contrastive training |

A clone can download the CMU survey and re-run `cmu_gold_eval.py`.
LR-Bench scoring expects two local JSON files this repo does not
re-host, and there is not yet a converter from the Hugging Face
download. Catalog numbers need the local 193k-paper files and trained
weights, which are not in this repo.

SPECTER2 is loaded as `allenai/specter2_base` plus the
`allenai/specter2` adapter named **proximity** (are these two papers
about similar work?).

## How I tried to keep the numbers honest

- Say what I **trained on** and what I **scored on**.
- If a clue is almost the answer key, report a version **without** it.
- Human quizzes beat SimCite when they disagree.
- Put every frozen method on the **same** board (word overlap,
  SPECTER2, standalone language-model votes).
- If a test is missing facts a real chair would have (field codes, the
  citation web), say so. The first-pass add-on uses extras that are
  often missing on the human papers.
- I did not slice results by seniority or by people who work across
  fields.
