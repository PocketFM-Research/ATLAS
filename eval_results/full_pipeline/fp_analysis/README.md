# Hallucination detector — false positive analysis

Manual error taxonomy of the 81 false positives the graph-based hallucination
detector produces on *Intolerable Cruelty* and *The Fighter*. Built in response
to a reviewer asking us to break down FPs into **graph missing fact**,
**relation paraphrase mismatch**, **entity resolution failure**, and **true
unsupported claim**.

## Read this first

[**hallucination_fp_analysis.md**](hallucination_fp_analysis.md) — the writeup.
Bucket counts, derived metrics (KG-attributable %, adjusted hallucination rate,
detector precision), interpretation, and headline numbers for the paper.

## Supporting artifacts

- [false_positive_review.md](false_positive_review.md) — one card per FP with
  the claim, the screenplay excerpt, the subject's graph node (aliases +
  edges), and the bucket assignment. Use this to audit individual labels.
- [bucket_labels.json](bucket_labels.json) — machine-readable bucket + note
  per FP. Edit here to override any label, then re-run the generator.
- [fp_labels.json](fp_labels.json) — TP/FP assignment per detector-flagged
  claim, matching Aayush's eval table (IC: TP=33 FP=39 / TF: TP=10 FP=42).
- [build_fp_review.py](build_fp_review.py) — regenerator. Reads the two label
  JSONs and the eval data, rewrites `false_positive_review.md`. Run from the
  repo root: `python eval_results/full_pipeline/fp_analysis/build_fp_review.py`.

## Headline result

| Bucket | Definition | Count | % of FPs |
|---|---|---:|---:|
| 1 | Graph missing fact | 76 | 93.8% |
| 2 | Relation paraphrase mismatch | 0 | 0.0% |
| 3 | Entity resolution failure | 0 | 0.0% |
| 4 | True unsupported claim | 1 | 1.2% |
| 5 | Other / extractor bug | 4 | 4.9% |
| **Total** | | **81** | 100% |

After correcting for KG incompleteness, **detector-side precision is ~90%**;
the headline ~35% precision is dominated by KG recall, not detector reliability.
See [hallucination_fp_analysis.md](hallucination_fp_analysis.md) for the full
interpretation.
