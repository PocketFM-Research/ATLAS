#!/usr/bin/env python3
"""Segment cleaned sampled stories into scenes.

The splitter preserves text byte-for-byte from cleaned.txt inside each scene.
It only chooses offsets and writes JSON metadata around those slices.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent

WORD_RE = re.compile(r"\b\w+\b")
ROMAN = r"(?:m{0,4}(?:cm|cd|d?c{0,3})?(?:xc|xl|l?x{0,3})?(?:ix|iv|v?i{0,3})|i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)"
NUMBER_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth"
)

CHAPTER_HEADING_RE = re.compile(
    rf"""
    ^\s*
    (?:
        (?P<label>chapter|book|part|act|scene|canto)\s+
        (?P<num>{ROMAN}|\d+|{NUMBER_WORD})
        (?P<title>(?:[\s.:-]+.+)?)?
      |
        (?P<fit>fit\s+the\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))
        (?P<fit_title>(?:\s+.+)?)?
      |
        (?P<numbered>\d+\s+[A-Z][A-Za-z0-9' -]{{2,80}})
      |
        (?P<roman_only>{ROMAN})
    )
    \.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

SECTION_BREAK_RE = re.compile(r"^\s*(?:\*{3,}|\*(?:\s+\*){2,}|[-=]{3,})\s*$")
TITLE_LINE_RE = re.compile(r"^\s*[A-Z][A-Za-z0-9 ,;:'\"()!?._-]{2,90}\s*$")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        spans.append((pos, end, line.rstrip("\n")))
        pos = end
    if text and not text.endswith("\n"):
        spans.append((pos, pos, ""))
    return spans


def next_nonblank_line(spans: list[tuple[int, int, str]], idx: int) -> tuple[int, str] | None:
    for next_idx in range(idx + 1, min(idx + 4, len(spans))):
        line = spans[next_idx][2].strip()
        if line:
            return next_idx, line
    return None


def normalize_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def collect_chapter_boundaries(text: str) -> list[tuple[int, str]]:
    spans = line_spans(text)
    candidates: list[tuple[int, str, int]] = []
    for idx, (start, _end, line) in enumerate(spans):
        stripped = line.strip()
        if not stripped:
            continue
        match = CHAPTER_HEADING_RE.match(stripped)
        if not match:
            continue

        heading = normalize_heading(stripped)
        next_line = next_nonblank_line(spans, idx)
        if next_line:
            _next_idx, title = next_line
            if (
                match.group("label")
                and not match.group("title")
                and TITLE_LINE_RE.match(title)
                and not CHAPTER_HEADING_RE.match(title)
                and len(title.split()) <= 12
            ):
                heading = f"{heading} {normalize_heading(title)}"
        candidates.append((start, heading, idx))

    if len(candidates) < 2:
        return []

    # Avoid treating dense numbered poem stanzas as chapter structure.
    dense_numeric = [
        c
        for c in candidates
        if re.match(r"^\d+\.?$", c[1]) or re.match(r"^\d+\s", c[1])
    ]
    if dense_numeric and len(dense_numeric) > 20:
        candidates = [c for c in candidates if c not in dense_numeric]

    # Roman-only headings are useful for works like The Secret Sharer, but a
    # single isolated Roman line is usually noise.
    roman_only = [c for c in candidates if re.fullmatch(ROMAN, c[1], re.IGNORECASE)]
    if roman_only and len(roman_only) < 2:
        candidates = [c for c in candidates if c not in roman_only]

    return [(start, heading) for start, heading, _idx in candidates]


def collect_section_break_boundaries(text: str) -> list[int]:
    spans = line_spans(text)
    starts = [0]
    for _start, end, line in spans:
        if SECTION_BREAK_RE.match(line):
            while end < len(text) and text[end] in "\n\r":
                end += 1
            if end < len(text):
                starts.append(end)
    return sorted(set(starts)) if len(starts) > 1 else []


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n{2,}|\Z)", text, re.DOTALL):
        para = match.group(0)
        if para.strip():
            spans.append((match.start(), match.end()))
    return spans


def heuristic_boundaries(text: str) -> list[int]:
    paras = paragraph_spans(text)
    total_words = word_count(text)
    target_scenes = max(8, min(15, round(total_words / 1000) or 8))
    target_words = max(1, round(total_words / target_scenes))

    starts = [0]
    acc_words = 0
    for idx, (_start, end) in enumerate(paras):
        acc_words += word_count(text[paras[idx][0] : end])
        remaining_paras = len(paras) - idx - 1
        remaining_scenes = target_scenes - len(starts)
        if (
            acc_words >= target_words
            and remaining_scenes > 0
            and remaining_paras >= remaining_scenes
        ):
            next_start = paras[idx + 1][0] if idx + 1 < len(paras) else len(text)
            starts.append(next_start)
            acc_words = 0
    return sorted(set(starts))


def build_scenes(
    text: str, starts: list[int], headings: dict[int, str | None], split_method: str
) -> list[dict]:
    starts = sorted(set(start for start in starts if 0 <= start < len(text)))
    if not starts or starts[0] != 0:
        starts = [0] + starts

    scenes = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        scene_text = text[start:end]
        scenes.append(
            {
                "scene": idx + 1,
                "heading": headings.get(start),
                "char_start": start,
                "char_end": end,
                "word_count": word_count(scene_text),
                "text": scene_text,
            }
        )
    return scenes


def segment_story(cleaned_path: Path) -> dict:
    story_dir = cleaned_path.parent
    metadata_path = story_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    text = cleaned_path.read_text(encoding="utf-8")

    chapter_boundaries = collect_chapter_boundaries(text)
    if chapter_boundaries:
        split_method = "chapter_heading"
        starts = [start for start, _heading in chapter_boundaries]
        headings = {start: heading for start, heading in chapter_boundaries}
    else:
        section_starts = collect_section_break_boundaries(text)
        if section_starts:
            split_method = "section_break"
            starts = section_starts
            headings = {start: None for start in starts}
        else:
            split_method = "heuristic"
            starts = heuristic_boundaries(text)
            headings = {start: None for start in starts}

    scenes = build_scenes(text, starts, headings, split_method)
    report = {
        "story_id": metadata.get("document_id") or story_dir.name,
        "split_method": split_method,
        "total_scenes": len(scenes),
        "scenes": scenes,
    }
    (story_dir / "scenes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**metadata, **report, "path": str(story_dir)}


def review_flags(word_counts: list[int]) -> list[str]:
    flags: list[str] = []
    if len(word_counts) < 5:
        flags.append("fewer than 5 scenes")
    if len(word_counts) > 40:
        flags.append("more than 40 scenes")
    if word_counts:
        med = median(word_counts)
        if med and max(word_counts) > 3 * med:
            flags.append("scene longer than 3x median")
    return flags


def main() -> None:
    cleaned_paths = sorted(ROOT.glob("*/*/*/cleaned.txt"))
    reports = [segment_story(path) for path in cleaned_paths]

    summary_fields = [
        "length_bucket",
        "genre",
        "wiki_title",
        "story_id",
        "split_method",
        "total_scenes",
        "mean_scene_word_count",
        "min_scene_word_count",
        "max_scene_word_count",
        "manual_review_flags",
        "path",
    ]
    summary_rows = []

    print(
        f"{'story_id':<42} {'method':<15} {'scenes':>6} {'mean':>7} {'min':>6} {'max':>6} flags"
    )
    print("-" * 104)
    for report in reports:
        counts = [scene["word_count"] for scene in report["scenes"]]
        flags = review_flags(counts)
        mean_count = round(sum(counts) / len(counts), 1) if counts else 0
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        print(
            f"{report['story_id']:<42} {report['split_method']:<15} "
            f"{report['total_scenes']:>6} {mean_count:>7.1f} "
            f"{min_count:>6} {max_count:>6} {', '.join(flags)}"
        )
        summary_rows.append(
            {
                **report,
                "mean_scene_word_count": mean_count,
                "min_scene_word_count": min_count,
                "max_scene_word_count": max_count,
                "manual_review_flags": "; ".join(flags),
            }
        )

    with (ROOT / "scene_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
