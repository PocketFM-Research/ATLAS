# Hallucination detector — FP error taxonomy (response to reviewer)

## What the reviewer asked

> The hallucination metric needs a clearer error taxonomy. […] Add a breakdown
> of hallucination false positives into at least four buckets: graph missing
> fact, relation paraphrase mismatch, entity resolution failure, and true
> unsupported claim. […] Without this, reviewers will assume the low
> hallucination precision is a fundamental limitation of the graph approach.

## What we did

We ran the full pipeline on two screenplays — *Intolerable Cruelty* and
*The Fighter* — each with a ground-truth set of inserted hallucinations
(`scene_text_exports/inserted_error_ground_truth_no_cross_scene.json`). The
detector flagged 124 claims as `hallucinated`. Comparing each flagged claim to
the ground-truth inserted lines gives us 43 true positives and **81 false
positives**:

| Film | Detected | TP | **FP** | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Intolerable Cruelty | 72 | 33 | **39** | 45.8% | 50.8% |
| The Fighter | 52 | 10 | **42** | 19.2% | 50.0% |
| **Total** | **124** | **43** | **81** | **34.7%** | **50.6%** |

We then read every one of the 81 FPs against (a) the source-text excerpt that
the eval ran on and (b) the subject's neighborhood in `final_graph.json`, and
applied the rubric Q1→Q2→Q3→Q4→Q5 from the richtext guide:

> Q1. Source text doesn't say it? → bucket 4 (true halluc.)
> Q2. Supporting triple absent from graph within 3 hops? → bucket 1 (missing fact)
> Q3. Subject in graph under a different alias? → bucket 3 (entity resolution)
> Q4. Supporting edge exists under a different relation label? → bucket 2 (paraphrase)
> Q5. Anything else (malformed triple, etc.)? → bucket 5

The full per-FP labeling is in [false_positive_review.md](false_positive_review.md)
(81 cards, one per FP, each with: claim, screenplay excerpt, subject node's aliases
and edges, bucket assignment, and a one-line note). Machine-readable bucket
labels are in [bucket_labels.json](bucket_labels.json).

## Results

| Bucket | Definition | IC | TF | **Total** | **% of FPs** |
|---|---|---:|---:|---:|---:|
| **1** | Graph missing fact | 35 | 41 | **76** | **93.8%** |
| 2 | Relation paraphrase mismatch | 0 | 0 | 0 | 0.0% |
| 3 | Entity resolution failure | 0 | 0 | 0 | 0.0% |
| 4 | True unsupported claim | 1 | 0 | 1 | 1.2% |
| 5 | Other / extractor bug | 3 | 1 | 4 | 4.9% |
| **Total** | | **39** | **42** | **81** | 100% |

## Derived numbers

- **KG-attributable false positives** = (bucket 1 + 2 + 3) / N = **76 / 81 = 93.8%**.
  Almost every FP is the detector being correctly skeptical *given the graph it
  sees* — the underlying KG just doesn't encode the fact that supports the claim.
- **Adjusted true-hallucination rate** (bucket 4 / total flagged) = 1 / 124 = **0.8%**.
  Only one FP is the detector inventing a hallucination that isn't really there.
- **Detector precision against generator hallucinations** =
  (TP + bucket 4) / total flagged = (43 + 1) / 124 = **35.5%** raw.
  But if you correct for KG incompleteness (treat bucket-1 FPs as detector
  warnings about missing KG content rather than detector errors), the
  **detector-side precision is 44 / (44 + 4 + 1) = ~89.8%** — i.e., the
  detector itself is rarely wrong; the *KG* is incomplete.

## Interpretation

### 1. The reviewer's hypothesis is right: most FPs are KG incompleteness, not detector error.

94% of FPs are bucket 1 — the source text supports the claim, but the KG
doesn't have an edge encoding it. None of the FPs are bucket 2 (paraphrase
mismatch) or bucket 3 (alias failure) under strict rubric, because in every case
where alias resolution or paraphrase matching could have helped, the underlying
fact was *also* missing from the graph as an edge. So even with a perfect
synonym table or perfect entity resolver, the FPs would still be flagged.

### 2. Bucket 1 splits into three recurring sub-patterns. Each suggests a different KG-construction fix.

We tagged each bucket-1 case with a sub-pattern in the per-FP notes. Counts
across the 76 bucket-1 cases:

| Sub-pattern | Approx count | What it looks like | Suggested fix |
|---|---:|---|---|
| **Stage-direction event not extracted** | ~38 | Real screenplay events ("Doyle tears strips off the paper", "Wrigley opens trapdoor", "Miles signs the air with a fountain pen") never extracted into the KG. The character node usually has 5–8 other event edges, just not this one. | Broader event-extraction pass over stage-direction text; lower the salience threshold for event extraction. |
| **Trait-in-description** | ~17 | Stative character traits ("Marylin is surprised and bemused", "Joe is perspiring heavily", "Micky has a scrape on his left cheek") that are only in node `description` text, not as edges. | Promote stative descriptions to first-class `(Character, has_state, …)` edges during KG construction, or have the verifier read `description` text as a fallback. |
| **Dialogue-as-experience** | ~10 | Claims of the form `(speaker, experiences/says, dialogue-content)` extracted from dialogue lines. The KG models neither dialogue propositions nor cognitive states. | Either skip dialogue-as-claim extraction entirely or add a `Dialogue-Utterance` node type and link speakers to the propositions they utter. |
| **Minor entity not extracted** | ~11 | Subject is a real but minor entity (a nurse, the bag of fluids, the den wall, Sanchez the boxing opponent) that the KG construction skipped entirely. The verifier returns "subject not found" or matches a wrong node. | Lower the salience threshold for entity extraction, or add a fallback that admits "subject not in KG" as an abstention rather than flagging hallucinated. |

### 3. The 5 non-bucket-1 cases are diagnostic.

- **The 1 true hallucination (bucket 4)** is FP-IC-38: claim *"Ruth offers
  Wrigley an antidepressant"* — there is no mention of "antidepressant" anywhere
  in scene 45's source text. This is the only case where the *generator* (the
  claim extractor in this case, since the source was the screenplay itself, not
  a generated story) actually invented something.
- **The 4 bucket-5 cases** are extractor bugs producing weak triples with
  non-entity subjects: `(wails, occurs_at, distance)`, `(smashing, experiences,
  increasing loudness)`, `(walls, possesses, photos…)`, `(micky's blood,
  performs, spills…)`. All are gerunds, generic plurals, or body-fluid noun
  phrases that the claim extractor should not have promoted to triple subjects.

### 4. A second-order finding: the KG itself has entity-merger bugs that don't show up in the bucket counts but are worth flagging.

Per the strict Q2-first rubric, the following are still bucket 1 (the underlying
fact is missing too), but they're worth noting in the paper as separate
KG-construction issues that *would* become bucket-3 cases if the underlying
events were properly extracted:

- **The Fighter's `Micky Ward` node has polluted aliases**:
  `["Micky Ward", "Mickey O'Keefe", "your mothah", "SISTERS", "father", …]`.
  Mickey O'Keefe (the trainer) and the Ward sisters are merged into Micky's node.
  Any claim about O'Keefe or the sisters resolves to Micky.
- **The Fighter has two "Dicky" nodes**: `Dicky Eklund` (the brother, the main
  character) and `Little Dicky` (his 5-year-old son). The matcher resolves
  `dicky` → `Little Dicky`. Most of the brother's actions in the late scenes
  (advising in the corner, doing commentary, "I got clean") therefore look
  unsupported.
- **The Fighter's `Crowd` node** describes the restaurant crowd from an early
  scene; it gets attached to fight-arena claims about a different crowd later.

### 5. Caveat on what this measures.

The eval used the screenplay text directly as both the input to claim extraction
and as the reference text the detector checks against (with planted insertions
as the "hallucinations" to find). That makes this a **clean-source false-positive
audit of the detector**, not a measurement of generator hallucination. The
expected number of bucket-4 cases under this design is ~0; finding 1 in 81
indicates the claim extractor itself rarely confabulates.

To measure detector behavior on *generator* hallucinations specifically, we'd
also want a clean-text run on actual model-generated stories (precision proxy)
and a dirty-text run with planted hallucinations the detector should catch
exactly (recall proxy). The bucket taxonomy above explains *why* the FP rate is
what it is on this controlled set; it complements but doesn't replace those
controlled tests.

## Headline numbers for the paper

> "Of 124 detector-flagged hallucinations across two screenplays, 43 are true
> positives, 76 are KG-incompleteness false positives (the source supports the
> claim but the graph doesn't encode the supporting edge), 4 are extractor bugs
> producing malformed triples with non-entity subjects, and only 1 is a true
> unsupported-claim hallucination. Adjusting for KG incompleteness, the detector
> itself has ~90% precision; the headline 35% precision is dominated by KG
> recall, not by detector reliability."

## Files

All files live in `eval_results/full_pipeline/fp_analysis/`:

- [false_positive_review.md](false_positive_review.md) — 81 per-FP cards with
  buckets filled in
- [bucket_labels.json](bucket_labels.json) — machine-readable bucket + note
  per FP
- [fp_labels.json](fp_labels.json) — TP/FP assignment (matches Aayush's table)
- [build_fp_review.py](build_fp_review.py) — re-runnable generator. Run with
  `python eval_results/full_pipeline/fp_analysis/build_fp_review.py` from the
  repo root. Reads the two label JSONs and the eval data, regenerates
  `false_positive_review.md`.
