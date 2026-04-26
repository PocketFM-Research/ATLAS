"""Smoke test for the repetition metric using intentionally repeated text."""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.repetition_eval import RepetitionEvaluator


INPUT_PATH = Path(__file__).with_name("fake_repetition_text.txt")
SUMMARY_PATH = Path(__file__).with_name("repetition_test_results.json")
DETAILS_PATH = Path(__file__).with_name("repetition_test_instances.csv")


def main() -> None:
    config = EvaluationConfig()
    evaluator = RepetitionEvaluator(config)
    story_text = INPUT_PATH.read_text(encoding="utf-8")

    metrics, instances = evaluator.evaluate_repetition(story_text)
    internal_repetition_instances = evaluator.evaluate_internal_repetition(story_text)
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

    evaluator.save_instances(instances, str(DETAILS_PATH))

    repetitive_instances = [instance for instance in instances if instance.is_repetitive]
    repetition_type_counts = Counter(
        repetition_type
        for instance in repetitive_instances
        for repetition_type in instance.repetition_types
    )
    summary = {
        "thresholds": {
            "semantic_similarity": config.similarity_threshold,
            "ngram_overlap": config.ngram_threshold,
        },
        "metrics": metrics.to_dict(),
        "repetition_type_counts": dict(repetition_type_counts),
        "smoke_checks": {
            "list_input_pairs": list_metrics.total_scene_pairs,
            "dict_input_pairs": dict_metrics.total_scene_pairs,
            "nested_input_rejected": nested_rejected,
        },
        "internal_repetition_summary": {
            "scene_count": len(internal_repetition_instances),
            "repeated_sentence_count": sum(
                len(instance.repeated_sentences)
                for instance in internal_repetition_instances
            ),
            "repeated_phrase_count": sum(
                len(instance.repeated_phrases)
                for instance in internal_repetition_instances
            ),
        },
        "internal_scene_repetitions": [
            {
                "scene_id": instance.scene_id,
                "internal_repetition_score": round(instance.internal_repetition_score, 4),
                "internal_sentence_repetition": round(
                    instance.internal_sentence_repetition,
                    4,
                ),
                "internal_ngram_repetition": round(instance.internal_ngram_repetition, 4),
                "repeated_sentences": instance.repeated_sentences,
                "repeated_phrases": instance.repeated_phrases[:10],
                "text_sample": instance.sample_text,
            }
            for instance in internal_repetition_instances
        ],
        "detected_repetitions": [
            {
                "scene_i_id": instance.scene_i_id,
                "scene_j_id": instance.scene_j_id,
                "semantic_similarity": round(instance.semantic_similarity, 4),
                "ngram_overlap": round(instance.ngram_overlap, 4),
                "sentence_overlap": round(instance.sentence_overlap, 4),
                "event_overlap": round(instance.event_overlap, 4),
                "state_overlap": round(instance.state_overlap, 4),
                "structure_similarity": round(instance.structure_similarity, 4),
                "novelty_score": round(instance.novelty_score, 4),
                "motif_overlap": round(instance.motif_overlap, 4),
                "beat_overlap": round(instance.beat_overlap, 4),
                "internal_repetition_score": round(instance.internal_repetition_score, 4),
                "repetition_types": instance.repetition_types,
                "text_i_sample": instance.sample_text_i,
                "text_j_sample": instance.sample_text_j,
            }
            for instance in repetitive_instances
        ],
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILS_PATH}")


if __name__ == "__main__":
    main()
