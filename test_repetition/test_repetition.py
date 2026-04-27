"""Smoke test and precision/recall evaluation for the repetition metric."""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.repetition_eval import RepetitionEvaluator


INPUT_PATH = Path(__file__).with_name("fake_repetition_text.txt")
OBVIOUS_PATH = Path(__file__).with_name("obvious_repetition_text.txt")
CLEAN_PATH = Path(__file__).with_name("clean_text.txt")
SUMMARY_PATH = Path(__file__).with_name("repetition_test_results.json")
DETAILS_PATH = Path(__file__).with_name("repetition_test_instances.csv")

# ── Ground-truth labels ──────────────────────────────────────────────────────
# For obvious_repetition_text.txt:
#   Scene 1 (blue notebook) ≈ Scene 3 (blue notebook again) — near-verbatim
#   Scene 2 (river conversation) ≈ Scene 5 (by the river again) — paraphrase
OBVIOUS_POSITIVE_PAIRS = {
    ("scene_1_the_blue_notebook", "scene_3_the_blue_notebook_again"),
    ("scene_2_the_river_conversation", "scene_5_by_the_river_again"),
}

# For clean_text.txt all pairs should be non-repetitive.
CLEAN_ALL_NEGATIVE = True


def _precision_recall(instances, positive_pairs):
    """Compute precision and recall against a set of labeled positive pairs."""
    flagged = {
        (inst.scene_i_id, inst.scene_j_id)
        for inst in instances
        if inst.is_repetitive
    }
    tp = len(flagged & positive_pairs)
    fp = len(flagged - positive_pairs)
    fn = len(positive_pairs - flagged)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall, tp, fp, fn


def _eval_story(evaluator, path, label):
    story_text = path.read_text(encoding="utf-8")
    metrics, instances = evaluator.evaluate_repetition(story_text)
    print(f"\n-- {label} --")
    print(f"  scenes:           {len(set([i.scene_i_id for i in instances] + [i.scene_j_id for i in instances]))}")
    print(f"  total pairs:      {metrics.total_scene_pairs}")
    print(f"  repetitive pairs: {metrics.repetitive_pairs}")
    print(f"  repetition rate:  {metrics.repetition_rate:.3f}")
    flagged = [(i.scene_i_id, i.scene_j_id) for i in instances if i.is_repetitive]
    if flagged:
        for pair in flagged:
            print(f"    flagged: {pair[0]}  <->  {pair[1]}")
    return metrics, instances


def main() -> None:
    config = EvaluationConfig()
    evaluator = RepetitionEvaluator(config)

    # ── 1. Original subtle story (no strict thresholds, just report) ─────────
    orig_metrics, orig_instances = _eval_story(evaluator, INPUT_PATH, "fake_repetition_text.txt")
    evaluator.save_instances(orig_instances, str(DETAILS_PATH))

    # ── 2. Obvious repetition story (recall test) ────────────────────────────
    obv_metrics, obv_instances = _eval_story(evaluator, OBVIOUS_PATH, "obvious_repetition_text.txt")
    obv_prec, obv_rec, obv_tp, obv_fp, obv_fn = _precision_recall(
        obv_instances, OBVIOUS_POSITIVE_PAIRS
    )
    print(f"\n  RECALL  TEST => precision={obv_prec:.3f}  recall={obv_rec:.3f}  "
          f"(TP={obv_tp} FP={obv_fp} FN={obv_fn})")

    # ── 3. Clean story (precision test — all negatives) ──────────────────────
    clean_metrics, clean_instances = _eval_story(evaluator, CLEAN_PATH, "clean_text.txt")
    clean_flagged = [i for i in clean_instances if i.is_repetitive]
    n_total_clean = clean_metrics.total_scene_pairs
    clean_prec = 1.0 - (len(clean_flagged) / n_total_clean) if n_total_clean > 0 else 1.0
    print(f"\n  PRECISION TEST => false-positive rate={1-clean_prec:.3f}  "
          f"(FP={len(clean_flagged)} / {n_total_clean} pairs)")
    if clean_flagged:
        print("  False positives flagged in clean story:")
        for inst in clean_flagged:
            print(f"    {inst.scene_i_id}  <->  {inst.scene_j_id}  "
                  f"(sem={inst.semantic_similarity:.3f})")

    # ── 4. API smoke checks ───────────────────────────────────────────────────
    list_metrics, _ = evaluator.evaluate_repetition([
        "Alex enters the lab and hides the blue notebook in the drawer.",
        "Alex goes into the laboratory and hides the blue notebook inside the desk drawer.",
        "Alex calls his sister beside the river and apologizes.",
    ])
    dict_metrics, _ = evaluator.evaluate_repetition({
        "scene_a": "Brooke unlocks the back door and leaves the silver key beneath the flower pot.",
        "scene_b": "Brooke opens the rear door and places the silver key under the flower pot.",
    })
    nested_rejected = False
    try:
        evaluator.evaluate_repetition({"scene_a": {"Alex": "Nested character text is invalid."}})
    except TypeError:
        nested_rejected = True

    # ── 5. Internal repetition (original story) ───────────────────────────────
    internal_instances = evaluator.evaluate_internal_repetition(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    # ── 6. Overall pass/fail ──────────────────────────────────────────────────
    recall_ok = obv_rec >= 0.95
    precision_ok = clean_prec >= 0.95
    print("\n== SUMMARY ==")
    print(f"  Recall    >= 0.95 : {'PASS' if recall_ok else 'FAIL'}  ({obv_rec:.3f})")
    print(f"  Precision >= 0.95 : {'PASS' if precision_ok else 'FAIL'}  ({clean_prec:.3f})")
    if not recall_ok:
        print(f"  !! Missed repetitive pairs: {OBVIOUS_POSITIVE_PAIRS - {(i.scene_i_id, i.scene_j_id) for i in obv_instances if i.is_repetitive}}")

    repetitive_orig = [i for i in orig_instances if i.is_repetitive]
    repetition_type_counts = Counter(
        t for i in repetitive_orig for t in i.repetition_types
    )

    summary = {
        "thresholds": {
            "semantic_similarity": config.similarity_threshold,
            "ngram_overlap": config.ngram_threshold,
        },
        "original_story_metrics": orig_metrics.to_dict(),
        "obvious_repetition_metrics": obv_metrics.to_dict(),
        "clean_story_metrics": clean_metrics.to_dict(),
        "precision_recall": {
            "obvious_story_recall": round(obv_rec, 4),
            "obvious_story_precision": round(obv_prec, 4),
            "obvious_story_tp": obv_tp,
            "obvious_story_fp": obv_fp,
            "obvious_story_fn": obv_fn,
            "clean_story_false_positive_rate": round(1 - clean_prec, 4),
            "clean_story_false_positives": len(clean_flagged),
            "clean_story_total_pairs": n_total_clean,
            "recall_pass": recall_ok,
            "precision_pass": precision_ok,
        },
        "repetition_type_counts": dict(repetition_type_counts),
        "smoke_checks": {
            "list_input_pairs": list_metrics.total_scene_pairs,
            "dict_input_pairs": dict_metrics.total_scene_pairs,
            "nested_input_rejected": nested_rejected,
        },
        "internal_repetition_summary": {
            "scene_count": len(internal_instances),
            "repeated_sentence_count": sum(
                len(i.repeated_sentences) for i in internal_instances
            ),
            "repeated_phrase_count": sum(
                len(i.repeated_phrases) for i in internal_instances
            ),
        },
        "internal_scene_repetitions": [
            {
                "scene_id": inst.scene_id,
                "internal_repetition_score": round(inst.internal_repetition_score, 4),
                "internal_sentence_repetition": round(inst.internal_sentence_repetition, 4),
                "internal_ngram_repetition": round(inst.internal_ngram_repetition, 4),
                "repeated_sentences": inst.repeated_sentences,
                "repeated_phrases": inst.repeated_phrases[:10],
                "text_sample": inst.sample_text,
            }
            for inst in internal_instances
        ],
        "detected_repetitions": [
            {
                "scene_i_id": inst.scene_i_id,
                "scene_j_id": inst.scene_j_id,
                "semantic_similarity": round(inst.semantic_similarity, 4),
                "ngram_overlap": round(inst.ngram_overlap, 4),
                "sentence_overlap": round(inst.sentence_overlap, 4),
                "event_overlap": round(inst.event_overlap, 4),
                "state_overlap": round(inst.state_overlap, 4),
                "structure_similarity": round(inst.structure_similarity, 4),
                "novelty_score": round(inst.novelty_score, 4),
                "motif_overlap": round(inst.motif_overlap, 4),
                "beat_overlap": round(inst.beat_overlap, 4),
                "internal_repetition_score": round(inst.internal_repetition_score, 4),
                "repetition_types": inst.repetition_types,
                "text_i_sample": inst.sample_text_i,
                "text_j_sample": inst.sample_text_j,
            }
            for inst in repetitive_orig
        ],
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILS_PATH}")

    if not (recall_ok and precision_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
