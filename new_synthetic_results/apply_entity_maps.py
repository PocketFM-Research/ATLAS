#!/usr/bin/env python3
"""Apply entity_map.json files and regenerate scenes over renamed.txt."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import segment_stories


ROOT = Path(__file__).resolve().parent


def boundary_pattern(term: str) -> re.Pattern:
    return re.compile(
        rf"(?<![\w])(?P<term>{re.escape(term)})(?P<possessive>'s|’s)?(?![\w])",
        re.IGNORECASE,
    )


def adapt_case(matched: str, replacement: str) -> str:
    if matched.isupper():
        return replacement.upper()
    if matched[:1].isupper() and matched[1:].islower() and " " not in matched:
        return replacement[:1].upper() + replacement[1:]
    return replacement


def replacement_pairs(entity_map: dict) -> list[tuple[str, str]]:
    pairs: dict[str, str] = {}
    for item in entity_map["mappings"]:
        variants = list(zip(item.get("variants", []), item.get("variant_replacements", [])))
        variant_originals = {original for original, _replacement in variants}
        candidates = []
        if item["original"] not in variant_originals:
            candidates.append((item["original"], item["replacement"]))
        candidates.extend(variants)
        for original, replacement in candidates:
            if not original or not replacement:
                continue
            prior = pairs.get(original)
            if prior is not None and prior != replacement:
                raise ValueError(
                    f"Conflicting replacements for {original!r}: {prior!r} vs {replacement!r}"
                )
            pairs[original] = replacement
    return sorted(pairs.items(), key=lambda item: len(item[0]), reverse=True)


def apply_pairs(text: str, pairs: list[tuple[str, str]]) -> str:
    renamed = text
    for original, replacement in pairs:
        pattern = boundary_pattern(original)

        def repl(match: re.Match) -> str:
            base = adapt_case(match.group("term"), replacement)
            return base + (match.group("possessive") or "")

        renamed = pattern.sub(repl, renamed)
    return renamed


def survivor_hits(text: str, pairs: list[tuple[str, str]]) -> list[dict]:
    hits = []
    seen_terms = set()
    for original, _replacement in pairs:
        if original in seen_terms:
            continue
        seen_terms.add(original)
        pattern = boundary_pattern(original)
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        contexts = []
        for match in matches[:5]:
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 45)
            contexts.append(re.sub(r"\s+", " ", text[start:end]).strip())
        hits.append(
            {
                "term": original,
                "count": len(matches),
                "contexts": contexts,
            }
        )
    return hits


def segment_renamed(story_dir: Path, metadata: dict, renamed: str) -> dict:
    chapter_boundaries = segment_stories.collect_chapter_boundaries(renamed)
    if chapter_boundaries:
        split_method = "chapter_heading"
        starts = [start for start, _heading in chapter_boundaries]
        headings = {start: heading for start, heading in chapter_boundaries}
    else:
        section_starts = segment_stories.collect_section_break_boundaries(renamed)
        if section_starts:
            split_method = "section_break"
            starts = section_starts
            headings = {start: None for start in starts}
        else:
            split_method = "heuristic"
            starts = segment_stories.heuristic_boundaries(renamed)
            headings = {start: None for start in starts}

    scenes = segment_stories.build_scenes(renamed, starts, headings, split_method)
    payload = {
        "story_id": metadata.get("document_id") or story_dir.name,
        "split_method": split_method,
        "total_scenes": len(scenes),
        "scenes": scenes,
    }
    (story_dir / "scenes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def process_story(story_dir: Path) -> dict:
    metadata = json.loads((story_dir / "metadata.json").read_text(encoding="utf-8"))
    entity_map = json.loads((story_dir / "entity_map.json").read_text(encoding="utf-8"))
    cleaned = (story_dir / "cleaned.txt").read_text(encoding="utf-8")
    pairs = replacement_pairs(entity_map)
    renamed = apply_pairs(cleaned, pairs)
    (story_dir / "renamed.txt").write_text(renamed, encoding="utf-8")
    scenes = segment_renamed(story_dir, metadata, renamed)
    hits = survivor_hits(renamed, pairs)
    before_words = segment_stories.word_count(cleaned)
    after_words = segment_stories.word_count(renamed)
    return {
        **metadata,
        "story_id": metadata.get("document_id") or story_dir.name,
        "cleaned_word_count": before_words,
        "renamed_word_count": after_words,
        "word_count_delta": after_words - before_words,
        "survivor_hit_count": sum(hit["count"] for hit in hits),
        "surviving_terms": "; ".join(f"{hit['term']} ({hit['count']})" for hit in hits),
        "survivor_hits": hits,
        "split_method": scenes["split_method"],
        "total_scenes": scenes["total_scenes"],
        "path": str(story_dir),
    }


def validate_offsets() -> None:
    for scene_path in ROOT.glob("*/*/*/scenes.json"):
        story_dir = scene_path.parent
        renamed = (story_dir / "renamed.txt").read_text(encoding="utf-8")
        data = json.loads(scene_path.read_text(encoding="utf-8"))
        reconstructed = "".join(scene["text"] for scene in data["scenes"])
        if reconstructed != renamed:
            raise ValueError(f"Scene text does not reconstruct renamed.txt for {story_dir}")
        for scene in data["scenes"]:
            if renamed[scene["char_start"] : scene["char_end"]] != scene["text"]:
                raise ValueError(f"Offset mismatch in {story_dir}, scene {scene['scene']}")


def main() -> None:
    story_dirs = sorted(path.parent for path in ROOT.glob("*/*/*/cleaned.txt"))
    reports = [process_story(story_dir) for story_dir in story_dirs]
    validate_offsets()

    fields = [
        "length_bucket",
        "genre",
        "wiki_title",
        "story_id",
        "cleaned_word_count",
        "renamed_word_count",
        "word_count_delta",
        "survivor_hit_count",
        "surviving_terms",
        "split_method",
        "total_scenes",
        "path",
    ]
    with (ROOT / "rename_verification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)

    with (ROOT / "rename_survivor_hits.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "story_id": report["story_id"],
                    "wiki_title": report["wiki_title"],
                    "survivor_hits": report["survivor_hits"],
                }
                for report in reports
                if report["survivor_hits"]
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(
        f"{'bucket':<6} {'genre':<14} {'title':<38} {'clean':>7} {'renamed':>7} "
        f"{'delta':>6} {'hits':>5} scenes"
    )
    print("-" * 104)
    for report in reports:
        print(
            f"{report['length_bucket']:<6} {report['genre']:<14} "
            f"{report['wiki_title'][:38]:<38} "
            f"{report['cleaned_word_count']:>7} {report['renamed_word_count']:>7} "
            f"{report['word_count_delta']:>6} {report['survivor_hit_count']:>5} "
            f"{report['split_method']}:{report['total_scenes']}"
        )

    total_hits = sum(report["survivor_hit_count"] for report in reports)
    print(f"\nValidated offsets for {len(reports)} renamed scene files")
    print(f"Surviving mapped original/variant hits: {total_hits}")


if __name__ == "__main__":
    main()
