"""Smoke test for the consistency metric using text with planted contradictions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.consistency_eval import ConsistencyEvaluator


INPUT_PATH = Path(__file__).with_name("fake_consistency_text.txt")
SUMMARY_PATH = Path(__file__).with_name("consistency_test_results.json")
DETAILS_PATH = Path(__file__).with_name("consistency_test_violations.csv")

# Planted contradictions we expect the evaluator to detect:
#   Marcus is described as alive in Scene 1, dead in Scene 2, then alive again in Scene 4+5
#   Elena is described as alone in Scene 1, together with Marcus in Scene 4+5
EXPECTED_VIOLATIONS = [
    # (scene_t, scene_tk, state_word_pair)
    ("scene_1_the_market", "scene_2_the_alley", "alive/dead"),
    ("scene_2_the_alley", "scene_4_the_station", "alive/dead"),
    ("scene_2_the_alley", "scene_5_the_journey", "alive/dead"),
]


def main() -> None:
    config = EvaluationConfig()
    evaluator = ConsistencyEvaluator(config)

    # ── 1. Full-text input ────────────────────────────────────────────────────
    story_text = INPUT_PATH.read_text(encoding="utf-8")
    metrics, violations = evaluator.evaluate_consistency(story_text)

    print("-- Full text input --")
    print(f"  scenes:              {metrics.total_scene_pairs + 1}")
    print(f"  scene pairs:         {metrics.total_scene_pairs}")
    print(f"  contradictions:      {metrics.contradictions}")
    print(f"  contradiction rate:  {metrics.contradiction_rate:.3f}")

    if violations:
        print("  Detected violations:")
        for v in violations:
            print(f"    [{v.scene_t}] vs [{v.scene_tk}]")
            print(f"      claim_t:  {v.claim_t!r}")
            print(f"      claim_tk: {v.claim_tk!r}")
            print(f"      reason:   {v.explanation}")

    # ── 2. List input ─────────────────────────────────────────────────────────
    list_metrics, _ = evaluator.evaluate_consistency([
        "Marcus was alive and walking through the market.",
        "Marcus was dead in the alley.",
        "Marcus arrived at the station and greeted Elena.",
    ])
    print(f"\n-- List input --")
    print(f"  scene pairs:    {list_metrics.total_scene_pairs}")
    print(f"  contradictions: {list_metrics.contradictions}")

    # ── 3. Dict input ─────────────────────────────────────────────────────────
    dict_metrics, _ = evaluator.evaluate_consistency({
        "opening": "Marcus was alive and walking through the market.",
        "midpoint": "Marcus was dead in the alley behind the bakery.",
    })
    print(f"\n-- Dict input --")
    print(f"  scene pairs:    {dict_metrics.total_scene_pairs}")
    print(f"  contradictions: {dict_metrics.contradictions}")

    # ── 4. Nested dict must raise TypeError ───────────────────────────────────
    nested_rejected = False
    try:
        evaluator.evaluate_consistency({"scene_1": {"Marcus": "Marcus was alive."}})
    except TypeError:
        nested_rejected = True
    print(f"\n-- Nested dict rejected: {nested_rejected} --")

    # ── 5. Check expected violations are detected ─────────────────────────────
    detected_pairs = {(v.scene_t, v.scene_tk) for v in violations}
    missed = []
    for scene_t, scene_tk, label in EXPECTED_VIOLATIONS:
        if (scene_t, scene_tk) not in detected_pairs:
            missed.append(f"{scene_t} vs {scene_tk} ({label})")

    print("\n== SUMMARY ==")
    print(f"  Violations detected:     {metrics.contradictions}")
    print(f"  Expected violations:     {len(EXPECTED_VIOLATIONS)}")
    print(f"  Missed expected:         {len(missed)}")
    print(f"  Nested input rejected:   {nested_rejected}")
    if missed:
        print("  Missed violations:")
        for m in missed:
            print(f"    {m}")

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    evaluator.save_violations(violations, str(DETAILS_PATH))

    violation_list = [
        {
            "scene_t": v.scene_t,
            "scene_tk": v.scene_tk,
            "claim_t": v.claim_t,
            "claim_tk": v.claim_tk,
            "violation_type": v.violation_type,
            "severity": v.severity,
            "explanation": v.explanation,
        }
        for v in violations
    ]
    summary = {
        "metrics": metrics.to_dict(),
        "smoke_checks": {
            "list_input_contradictions": list_metrics.contradictions,
            "dict_input_contradictions": dict_metrics.contradictions,
            "nested_input_rejected": nested_rejected,
        },
        "expected_violations_detected": len(EXPECTED_VIOLATIONS) - len(missed),
        "expected_violations_total": len(EXPECTED_VIOLATIONS),
        "missed_expected": missed,
        "violations": violation_list,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILS_PATH}")

    all_ok = (
        nested_rejected
        and list_metrics.contradictions >= 1
        and dict_metrics.contradictions >= 1
        and len(missed) == 0
    )
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
