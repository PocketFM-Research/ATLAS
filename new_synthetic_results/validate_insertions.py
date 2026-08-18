#!/usr/bin/env python3
"""Validate continuity-error insertion answer keys.

This script audits every story folder under new_synthetic_results. It tags
each insertion as valid/invalid without rewriting the source story, answer key,
or hallucinated story.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
CAPS = {"5k": 8, "10k": 15, "15k": 20}
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
ARTICLES = {"a", "an", "the"}
OPENING_QUOTES = {'"', "'", "“", "‘"}
CLOSING_QUOTES = {'"', "'", "”", "’"}
KNOWLEDGE_OR_MOTIVATION_TERMS = {
    "act",
    "acts",
    "aware",
    "belief",
    "believe",
    "character",
    "compliance",
    "deception",
    "emotion",
    "emotional",
    "feeling",
    "forgot",
    "forgotten",
    "information",
    "intent",
    "intention",
    "interiority",
    "know",
    "knowledge",
    "learn",
    "learned",
    "motivation",
    "motive",
    "recognition",
    "remember",
    "suspicion",
    "told",
    "watching",
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def words_changed(original: str, edited: str) -> int:
    before = WORD_RE.findall(original)
    after = WORD_RE.findall(edited)
    matcher = difflib.SequenceMatcher(a=before, b=after)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def target_for(length_bucket: str, total_scenes: int) -> int:
    raw = round(total_scenes * 0.6)
    return min(raw, CAPS[length_bucket], total_scenes - 1)


def story_dirs() -> list[Path]:
    return sorted(path.parent for path in ROOT.glob("*/*/*/insertions.json"))


def normalize_entity(entity: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", entity.lower())
    return " ".join(token for token in tokens if token not in ARTICLES)


def quote_wrapped_in_quotes(quote: str) -> bool:
    stripped = quote.strip()
    if not stripped:
        return False
    return stripped[0] in OPENING_QUOTES or stripped[-1] in CLOSING_QUOTES


def scene_text(scenes_by_number: dict[int, dict[str, Any]], scene_num: Any) -> str | None:
    try:
        return scenes_by_number[int(scene_num)]["text"]
    except (KeyError, TypeError, ValueError):
        return None


def hard_is_distance_only(item: dict[str, Any], total_scenes: int) -> bool:
    if item.get("difficulty") != "hard":
        return False
    scene_distance = int(item.get("scene_distance") or 0)
    if scene_distance <= total_scenes / 2:
        return False
    evidence = " ".join(
        str(item.get(field, ""))
        for field in ("difficulty_reason", "fact", "severity_reason")
    ).lower()
    return not any(term in evidence for term in KNOWLEDGE_OR_MOTIVATION_TERMS)


def apply_valid_insertions(text: str, valid_items: list[dict[str, Any]]) -> tuple[str, list[str]]:
    patched = text
    failures = []
    for item in sorted(valid_items, key=lambda row: int(row.get("char_offset") or 0), reverse=True):
        original = item["violation_quote_original"]
        edited = item["violation_quote_edited"]
        offset = int(item.get("char_offset") or -1)
        if offset >= 0 and patched[offset : offset + len(original)] == original:
            patched = patched[:offset] + edited + patched[offset + len(original) :]
            continue

        occurrences = [match.start() for match in re.finditer(re.escape(original), patched)]
        if len(occurrences) != 1:
            failures.append(
                f"valid insertion {item.get('id')} could not be applied uniquely "
                f"while building expected hallucinated text"
            )
            continue
        start = occurrences[0]
        patched = patched[:start] + edited + patched[start + len(original) :]
    return patched, failures


def validate_hallucinated(
    story_dir: Path, renamed: str, hallucinated: str, valid_items: list[dict[str, Any]]
) -> dict[str, Any]:
    failures = []
    for item in valid_items:
        edited = item["violation_quote_edited"]
        original = item["violation_quote_original"]
        anchor = item["anchor_quote"]
        edited_count = hallucinated.count(edited)
        if edited_count != 1:
            failures.append(
                f"insertion {item.get('id')}: edited quote appears {edited_count} times"
            )
        if original in hallucinated:
            failures.append(f"insertion {item.get('id')}: original violation quote survives")
        if anchor not in hallucinated:
            failures.append(f"insertion {item.get('id')}: anchor quote changed or missing")

    expected, apply_failures = apply_valid_insertions(renamed, valid_items)
    failures.extend(apply_failures)
    if expected != hallucinated:
        failures.append("hallucinated.txt differs from renamed.txt outside valid edited sentences")

    return {
        "passed": not failures,
        "failures": failures,
        "renamed_word_count": word_count(renamed),
        "hallucinated_word_count": word_count(hallucinated),
        "word_count_delta": word_count(hallucinated) - word_count(renamed),
        "path": str(story_dir / "hallucinated.txt"),
    }


def validate_story(story_dir: Path) -> dict[str, Any]:
    insertions_payload = json.loads((story_dir / "insertions.json").read_text(encoding="utf-8"))
    scenes_payload = json.loads((story_dir / "scenes.json").read_text(encoding="utf-8"))
    metadata = json.loads((story_dir / "metadata.json").read_text(encoding="utf-8"))
    renamed = (story_dir / "renamed.txt").read_text(encoding="utf-8")
    hallucinated_path = story_dir / "hallucinated.txt"
    hallucinated = hallucinated_path.read_text(encoding="utf-8") if hallucinated_path.exists() else ""

    scenes = scenes_payload["scenes"]
    scenes_by_number = {int(scene["scene"]): scene for scene in scenes}
    total_scenes = int(scenes_payload["total_scenes"])
    length_bucket = metadata["length_bucket"]
    computed_target = target_for(length_bucket, total_scenes)

    seen_violation_scenes: set[int] = set()
    seen_entities: set[str] = set()
    tagged_insertions = []
    invalid_by_reason: Counter[str] = Counter()

    for item in insertions_payload.get("insertions", []):
        tagged = dict(item)
        failing_checks = []

        if any(mention.get("still_conflicts") is True for mention in item.get("other_mentions", [])):
            failing_checks.append("other_mentions_still_conflict")

        actual_changed = words_changed(
            item.get("violation_quote_original", ""),
            item.get("violation_quote_edited", ""),
        )
        tagged["actual_words_changed"] = actual_changed
        if int(item.get("words_changed") or 0) > 5:
            failing_checks.append("words_changed_reported_gt_5")
        if actual_changed > 5:
            failing_checks.append("words_changed_actual_gt_5")

        anchor_quote = item.get("anchor_quote", "")
        violation_quote = item.get("violation_quote_original", "")
        if anchor_quote not in renamed:
            failing_checks.append("anchor_quote_missing_in_renamed")
        if violation_quote not in renamed:
            failing_checks.append("violation_quote_original_missing_in_renamed")

        try:
            violation_scene = int(item.get("violation_scene"))
        except (TypeError, ValueError):
            violation_scene = -1
            failing_checks.append("violation_scene_not_integer")
        if violation_scene in seen_violation_scenes:
            failing_checks.append("duplicate_violation_scene")
        else:
            seen_violation_scenes.add(violation_scene)

        if quote_wrapped_in_quotes(anchor_quote):
            failing_checks.append("anchor_quote_wrapped_in_quotation_marks")
        if quote_wrapped_in_quotes(violation_quote):
            failing_checks.append("violation_quote_original_wrapped_in_quotation_marks")

        try:
            anchor_scene = int(item.get("anchor_scene"))
        except (TypeError, ValueError):
            anchor_scene = -1
            failing_checks.append("anchor_scene_not_integer")

        anchor_scene_text = scene_text(scenes_by_number, anchor_scene)
        violation_scene_text = scene_text(scenes_by_number, violation_scene)
        if anchor_scene_text is None or anchor_quote not in anchor_scene_text:
            failing_checks.append("anchor_quote_not_found_in_claimed_scene")
        if violation_scene_text is None or violation_quote not in violation_scene_text:
            failing_checks.append("violation_quote_original_not_found_in_claimed_scene")

        normalized_entity = normalize_entity(item.get("anchor_entity", ""))
        tagged["normalized_anchor_entity"] = normalized_entity
        if normalized_entity in seen_entities:
            failing_checks.append("duplicate_anchor_entity")
        else:
            seen_entities.add(normalized_entity)

        if violation_scene <= anchor_scene:
            failing_checks.append("violation_scene_not_after_anchor_scene")
        if violation_scene == 1:
            failing_checks.append("violation_scene_is_1")

        tagged["valid"] = not failing_checks
        tagged["failing_checks"] = failing_checks
        tagged_insertions.append(tagged)
        for reason in failing_checks:
            invalid_by_reason[reason] += 1

    valid_items = [item for item in tagged_insertions if item["valid"]]
    difficulty_counts = Counter(item.get("difficulty", "unknown") for item in valid_items)
    severity_counts = Counter(item.get("severity", "unknown") for item in valid_items)
    hard_items = [item for item in valid_items if item.get("difficulty") == "hard"]
    distance_only_hard = [item for item in hard_items if hard_is_distance_only(item, total_scenes)]
    hard_distance_only_fraction = (
        len(distance_only_hard) / len(hard_items) if hard_items else 0.0
    )
    has_hard_major = any(
        item.get("difficulty") == "hard" and item.get("severity") == "major"
        for item in valid_items
    )
    has_easy_minor = any(
        item.get("difficulty") == "easy" and item.get("severity") == "minor"
        for item in valid_items
    )

    hallucinated_verification = validate_hallucinated(
        story_dir, renamed, hallucinated, valid_items
    )

    story_result = {
        "story_id": insertions_payload.get("story_id") or story_dir.name,
        "path": str(story_dir),
        "length_bucket": length_bucket,
        "genre": metadata["genre"],
        "wiki_title": metadata["wiki_title"],
        "total_scenes": total_scenes,
        "computed_target_insertions": computed_target,
        "insertions_produced": len(tagged_insertions),
        "valid_insertions": len(valid_items),
        "invalid_insertions": len(tagged_insertions) - len(valid_items),
        "invalid_by_reason": dict(sorted(invalid_by_reason.items())),
        "surviving_difficulty_counts": {
            key: difficulty_counts.get(key, 0) for key in ("easy", "medium", "hard")
        },
        "surviving_severity_counts": {
            key: severity_counts.get(key, 0) for key in ("minor", "moderate", "major")
        },
        "has_hard_major": has_hard_major,
        "has_easy_minor": has_easy_minor,
        "hard_distance_only_fraction": round(hard_distance_only_fraction, 3),
        "hard_distance_only_flag": hard_distance_only_fraction > 0.5,
        "hard_distance_only_insertion_ids": [item.get("id") for item in distance_only_hard],
        "mean_scene_distance_valid": round(
            mean(int(item.get("scene_distance") or 0) for item in valid_items), 2
        )
        if valid_items
        else 0,
        "hallucinated_verification": hallucinated_verification,
        "insertions": tagged_insertions,
    }

    (story_dir / "validated.json").write_text(
        json.dumps(story_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return story_result


def counts_to_str(counts: dict[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in counts.items())


def main() -> None:
    results = [validate_story(story_dir) for story_dir in story_dirs()]

    summary_fields = [
        "length_bucket",
        "genre",
        "wiki_title",
        "story_id",
        "total_scenes",
        "computed_target_insertions",
        "insertions_produced",
        "valid_insertions",
        "invalid_insertions",
        "surviving_difficulty_counts",
        "surviving_severity_counts",
        "has_hard_major",
        "has_easy_minor",
        "hard_distance_only_fraction",
        "hard_distance_only_flag",
        "hallucinated_verification_passed",
        "mean_scene_distance_valid",
        "path",
    ]
    with (ROOT / "validation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "length_bucket": result["length_bucket"],
                    "genre": result["genre"],
                    "wiki_title": result["wiki_title"],
                    "story_id": result["story_id"],
                    "total_scenes": result["total_scenes"],
                    "computed_target_insertions": result["computed_target_insertions"],
                    "insertions_produced": result["insertions_produced"],
                    "valid_insertions": result["valid_insertions"],
                    "invalid_insertions": result["invalid_insertions"],
                    "surviving_difficulty_counts": counts_to_str(
                        result["surviving_difficulty_counts"]
                    ),
                    "surviving_severity_counts": counts_to_str(
                        result["surviving_severity_counts"]
                    ),
                    "has_hard_major": result["has_hard_major"],
                    "has_easy_minor": result["has_easy_minor"],
                    "hard_distance_only_fraction": result["hard_distance_only_fraction"],
                    "hard_distance_only_flag": result["hard_distance_only_flag"],
                    "hallucinated_verification_passed": result[
                        "hallucinated_verification"
                    ]["passed"],
                    "mean_scene_distance_valid": result["mean_scene_distance_valid"],
                    "path": result["path"],
                }
            )

    total_insertions = sum(result["insertions_produced"] for result in results)
    total_valid = sum(result["valid_insertions"] for result in results)
    invalid_reasons = Counter()
    for result in results:
        invalid_reasons.update(result["invalid_by_reason"])

    print(f"Total insertions: {total_insertions}")
    print(f"Valid insertions: {total_valid}")
    print(f"Invalid insertions: {total_insertions - total_valid}")
    print("Invalid by reason:")
    if invalid_reasons:
        for reason, count in sorted(invalid_reasons.items()):
            print(f"  {reason}: {count}")
    else:
        print("  none")
    print()
    print(
        f"{'story':<46} {'target':>6} {'made':>5} {'valid':>5} {'invalid':>7} "
        f"{'difficulty':<23} {'severity':<26} {'hm':>3} {'em':>3} {'dist-only':>9} {'hall':>5}"
    )
    print("-" * 148)
    for result in results:
        title = f"{result['length_bucket']} {result['genre']} {result['wiki_title']}"
        print(
            f"{title[:46]:<46} {result['computed_target_insertions']:>6} "
            f"{result['insertions_produced']:>5} {result['valid_insertions']:>5} "
            f"{result['invalid_insertions']:>7} "
            f"{counts_to_str(result['surviving_difficulty_counts']):<23} "
            f"{counts_to_str(result['surviving_severity_counts']):<26} "
            f"{str(result['has_hard_major']):>3} {str(result['has_easy_minor']):>3} "
            f"{result['hard_distance_only_fraction']:>9} "
            f"{str(result['hallucinated_verification']['passed']):>5}"
        )

    flagged_hard = [result for result in results if result["hard_distance_only_flag"]]
    print()
    print("Hard distance-only flags:")
    if flagged_hard:
        for result in flagged_hard:
            print(
                f"  {result['length_bucket']}/{result['genre']}/{Path(result['path']).name}: "
                f"fraction={result['hard_distance_only_fraction']} "
                f"ids={result['hard_distance_only_insertion_ids']}"
            )
    else:
        print("  none")

    hallucinated_failures = [
        result for result in results if not result["hallucinated_verification"]["passed"]
    ]
    print()
    print("Hallucinated verification failures:")
    if hallucinated_failures:
        for result in hallucinated_failures:
            print(f"  {result['length_bucket']}/{result['genre']}/{Path(result['path']).name}:")
            for failure in result["hallucinated_verification"]["failures"]:
                print(f"    - {failure}")
    else:
        print("  none")
    print()
    print(f"Corpus-wide total of valid items: {total_valid}")


if __name__ == "__main__":
    main()
