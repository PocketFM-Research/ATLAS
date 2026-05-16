#!/usr/bin/env python3
"""Cross-scene repetition runner with stratified gold evaluation.

Patch over the original runner. Behaviour matches the old version when the
ground-truth JSON contains the legacy flat schema (every entry implicitly a
positive). When entries carry the new fields (`tier`, `is_repetition`) the
runner produces:

  - per-tier precision / recall
  - bootstrap 95% CI for each metric (resampled over gold cases)
  - hard-negative handling: `is_repetition: false` entries are NOT counted in
    the recall denominator and ARE counted as false positives when flagged
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


# --------------------------- stratified eval ---------------------------


def _detected_pairs(instances) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for item in instances:
        if item.is_repetitive:
            pairs.add((item.scene_i_id, item.scene_j_id))
            pairs.add((item.scene_j_id, item.scene_i_id))
    return pairs


def _gold_pair_id(item: dict) -> Tuple[str, str]:
    return (
        scene_id_from_heading(item["source_heading"]),
        scene_id_from_heading(item["repeated_heading"]),
    )


def _precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def _bootstrap_ci(
    rows: List[Tuple[bool, bool]],   # (is_positive_gold, was_detected)
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    extra_fp: int = 0,               # FP from items outside the resampled rows (none by default)
) -> Dict[str, Tuple[float, float]]:
    """Resample over gold rows. Each row is (gold_is_positive, detected).
    Returns 95% CIs for precision and recall."""
    if not rows:
        return {"precision": (0.0, 0.0), "recall": (0.0, 0.0)}
    rng = random.Random(seed)
    n = len(rows)
    precisions: List[float] = []
    recalls: List[float] = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        tp = sum(1 for is_pos, det in sample if is_pos and det)
        fn = sum(1 for is_pos, det in sample if is_pos and not det)
        fp = sum(1 for is_pos, det in sample if not is_pos and det) + extra_fp
        p, r = _precision_recall(tp, fp, fn)
        precisions.append(p)
        recalls.append(r)
    lo_idx = int((1 - confidence) / 2 * n_boot)
    hi_idx = int((1 + confidence) / 2 * n_boot) - 1
    precisions.sort()
    recalls.sort()
    return {
        "precision": (precisions[lo_idx], precisions[hi_idx]),
        "recall": (recalls[lo_idx], recalls[hi_idx]),
    }


def _eval_stratum(
    gt_items: List[dict],
    detected: Set[Tuple[str, str]],
    extra_fp: int = 0,
) -> Dict[str, Any]:
    """Evaluate one tier (or pooled). Returns counts, P/R, and bootstrap CI."""
    rows: List[Tuple[bool, bool]] = []
    matched_ids: List[str] = []
    false_negatives: List[str] = []
    false_positives_in_negatives: List[str] = []
    tp = fp = fn = tn = 0
    for item in gt_items:
        pair = _gold_pair_id(item)
        det = pair in detected
        is_pos = bool(item.get("is_repetition", True))
        rows.append((is_pos, det))
        if is_pos and det:
            tp += 1
            matched_ids.append(item.get("id", ""))
        elif is_pos and not det:
            fn += 1
            false_negatives.append(item.get("id", ""))
        elif (not is_pos) and det:
            fp += 1
            false_positives_in_negatives.append(item.get("id", ""))
        else:
            tn += 1
    p, r = _precision_recall(tp, fp + extra_fp, fn)
    ci = _bootstrap_ci(rows, extra_fp=extra_fp)
    return {
        "n_gold_positive": tp + fn,
        "n_gold_negative": tn + fp,
        "true_positives": tp,
        "false_positives": fp + extra_fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": p,
        "recall": r,
        "precision_ci95": list(ci["precision"]),
        "recall_ci95": list(ci["recall"]),
        "matched_ids": matched_ids,
        "missed_ids": false_negatives,
        "flagged_negative_ids": false_positives_in_negatives,
    }


def compare_to_ground_truth(instances, gt_items: Iterable[dict]) -> Dict[str, Any]:
    gt_items = list(gt_items)
    detected = _detected_pairs(instances)
    detected_count = sum(1 for item in instances if item.is_repetitive) // 2 \
        if False else sum(1 for item in instances if item.is_repetitive)

    has_tier = any("tier" in item for item in gt_items)
    has_is_rep = any("is_repetition" in item for item in gt_items)

    # ----- legacy schema: every entry is a positive, pooled -----
    if not has_tier and not has_is_rep:
        for item in gt_items:
            item.setdefault("is_repetition", True)
            item.setdefault("tier", "legacy_whole_scene_rewrite")

    pooled_positive_items = [g for g in gt_items if g.get("is_repetition", True)]
    pooled_negative_items = [g for g in gt_items if not g.get("is_repetition", True)]

    # FP that come from detections NOT in any gold entry
    gold_pairs_all = {_gold_pair_id(g) for g in gt_items}
    # Detection pairs are stored both directions; dedupe to canonical ordering
    canonical_detected: Set[Tuple[str, str]] = set()
    for a, b in detected:
        canonical_detected.add(tuple(sorted([a, b])))
    canonical_gold = {tuple(sorted(p)) for p in gold_pairs_all}
    extra_fp_pairs = canonical_detected - canonical_gold
    extra_fp = len(extra_fp_pairs)

    pooled = _eval_stratum(
        gt_items=pooled_positive_items + pooled_negative_items,
        detected=detected,
        extra_fp=extra_fp,
    )

    # ----- per-tier breakdown -----
    by_tier: Dict[str, List[dict]] = defaultdict(list)
    for item in gt_items:
        by_tier[item.get("tier", "unspecified")].append(item)

    per_tier: Dict[str, Any] = {}
    for tier_name, items in sorted(by_tier.items()):
        # Per-tier metric: do NOT add extra_fp (those are off-gold detections,
        # not attributable to any single tier). Pooled metric carries them.
        per_tier[tier_name] = _eval_stratum(items, detected)

    return {
        "ground_truth_pairs": len(gt_items),
        "ground_truth_positives": len(pooled_positive_items),
        "ground_truth_negatives": len(pooled_negative_items),
        "detected_repetitive_pairs_total": detected_count,
        "detections_not_in_gold": extra_fp,
        "pooled": pooled,
        "per_tier": per_tier,
        # legacy keys for back-compat with older eval consumers
        "matched_ground_truth_pairs": pooled["true_positives"],
        "detected_repetitive_pairs": detected_count,
        "precision_against_gt": pooled["precision"],
        "recall_against_gt": pooled["recall"],
        "matched_ids": pooled["matched_ids"],
    }


# --------------------------- runner ---------------------------


def run_case(args) -> dict:
    config = EvaluationConfig()
    config.similarity_threshold = args.similarity_threshold
    scenes = normalize_scene_text(load_scene_text(args.text_path))
    evaluator_cls = (
        LexicalRepetitionEvaluator if args.lexical_similarity else RepetitionEvaluator
    )
    evaluator = evaluator_cls(config)
    metrics, instances = evaluator.evaluate_repetition(scenes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_instances(
        instances, str(output_dir / "cross_scene_repetition_instances.csv")
    )

    gt_summary: Dict[str, Any] = {}
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_items = gt_data.get(
            args.movie_id, gt_data if isinstance(gt_data, list) else []
        )
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
    _print_human_readable(gt_summary)
    return summary


def _print_human_readable(gt_summary: Dict[str, Any]) -> None:
    if not gt_summary or "pooled" not in gt_summary:
        return
    pooled = gt_summary["pooled"]
    print(
        f"\nPOOLED: P={pooled['precision']:.3f} "
        f"[{pooled['precision_ci95'][0]:.3f}, {pooled['precision_ci95'][1]:.3f}]  "
        f"R={pooled['recall']:.3f} "
        f"[{pooled['recall_ci95'][0]:.3f}, {pooled['recall_ci95'][1]:.3f}]  "
        f"(positives={pooled['n_gold_positive']}, "
        f"negatives={pooled['n_gold_negative']}, "
        f"off-gold FP={gt_summary.get('detections_not_in_gold', 0)})"
    )
    per_tier = gt_summary.get("per_tier", {})
    if per_tier:
        print("\nPER TIER:")
        for tier, m in per_tier.items():
            n_pos = m["n_gold_positive"]
            n_neg = m["n_gold_negative"]
            print(
                f"  {tier:24s} n_pos={n_pos:2d} n_neg={n_neg:2d}  "
                f"P={m['precision']:.3f} "
                f"[{m['precision_ci95'][0]:.3f},{m['precision_ci95'][1]:.3f}]  "
                f"R={m['recall']:.3f} "
                f"[{m['recall_ci95'][0]:.3f},{m['recall_ci95'][1]:.3f}]"
            )
            if m["missed_ids"]:
                print(f"      missed: {', '.join(m['missed_ids'])}")
            if m["flagged_negative_ids"]:
                print(
                    f"      false-flagged negatives: "
                    f"{', '.join(m['flagged_negative_ids'])}"
                )


def main(argv: Optional[Sequence[str]] = None) -> None:
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
    args = parser.parse_args(argv)
    print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
