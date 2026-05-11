#!/usr/bin/env python3
"""Temporary cross-scene repetition-only runner.

This mirrors the repetition portion of eval_experiments without running
hallucination or consistency. It uses RepetitionEvaluator for cross-scene
pair scoring and writes outputs under eval_results/full_pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.experiment_runner import load_scene_text, normalize_scene_text
from stage_kg.evaluation.repetition_eval import RepetitionEvaluator


class LexicalRepetitionEvaluator(RepetitionEvaluator):
    """Offline fallback for exact-copy cross-scene benchmarks."""

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        tokens1 = self._tokens(text1)
        tokens2 = self._tokens(text2)
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9']+", text.lower())
            if len(token) > 2
        }


def scene_id_from_heading(heading: str) -> str:
    normalized = re.sub(r"\s+", "_", heading.strip().lower())
    normalized = re.sub(r"[^a-z0-9_.-]+", "", normalized).strip("._-")
    return normalized


def compare_to_ground_truth(instances, gt_items: Iterable[dict]) -> dict:
    gt_items = list(gt_items)
    detected = {(item.scene_i_id, item.scene_j_id) for item in instances if item.is_repetitive}
    detected |= {(right, left) for left, right in detected}
    detected_count = sum(1 for item in instances if item.is_repetitive)

    matched_ids = []
    for item in gt_items:
        source_id = scene_id_from_heading(item["source_heading"])
        repeated_id = scene_id_from_heading(item["repeated_heading"])
        if (source_id, repeated_id) in detected:
            matched_ids.append(item["id"])

    gt_count = len(gt_items)
    return {
        "ground_truth_pairs": gt_count,
        "matched_ground_truth_pairs": len(matched_ids),
        "detected_repetitive_pairs": detected_count,
        "precision_against_gt": len(matched_ids) / detected_count if detected_count else 0.0,
        "recall_against_gt": len(matched_ids) / gt_count if gt_count else 0.0,
        "matched_ids": matched_ids,
    }


def run_case(args) -> dict:
    config = EvaluationConfig()
    config.similarity_threshold = args.similarity_threshold
    scenes = normalize_scene_text(load_scene_text(args.text_path))
    evaluator_cls = LexicalRepetitionEvaluator if args.lexical_similarity else RepetitionEvaluator
    evaluator = evaluator_cls(config)
    metrics, instances = evaluator.evaluate_repetition(scenes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_instances(instances, str(output_dir / "cross_scene_repetition_instances.csv"))

    gt_summary = {}
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_items = gt_data.get(args.movie_id, gt_data if isinstance(gt_data, list) else [])
        gt_summary = compare_to_ground_truth(instances, gt_items)
        gt_summary["ground_truth_path"] = str(gt_path)

    summary = {
        "metadata": {
            "movie_id": args.movie_id,
            "text_path": str(Path(args.text_path)),
            "scene_count": len(scenes),
            "lexical_similarity": args.lexical_similarity,
        },
        "metrics": {
            "cross_scene": metrics.to_dict(),
            **gt_summary,
        },
    }
    (output_dir / "cross_scene_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-scene repetition only.")
    parser.add_argument("--movie_id", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ground_truth")
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument(
        "--lexical_similarity",
        action="store_true",
        help="Avoid sentence-transformer downloads for exact-copy cross-scene tests.",
    )
    args = parser.parse_args()
    print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
