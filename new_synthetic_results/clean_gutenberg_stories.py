#!/usr/bin/env python3
"""Clean sampled Project Gutenberg stories without regenerating prose.

The cleaner is intentionally deletion-oriented: it removes Project Gutenberg
wrappers, detected front/back matter, and standalone artifact lines, then writes
the remaining text with a consistent newline policy.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


START_RE = re.compile(
    r"^\s*\*{2,}\s*START\s+OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK\b.*?\*{2,}\s*$",
    re.IGNORECASE,
)
END_RE = re.compile(
    r"^\s*\*{2,}\s*END\s+OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK\b.*?\*{2,}\s*$",
    re.IGNORECASE,
)
OLD_END_RE = re.compile(
    r"^\s*\*+\s*END\s*\*+\s*THE\s+SMALL\s+PRINT\b.*$|^\s*End\s+of\s+(?:Project\s+Gutenberg|the\s+Project\s+Gutenberg)\b.*$",
    re.IGNORECASE,
)

FRONT_SECTION_RE = re.compile(
    r"^\s*(?:preface|foreword|introduction|translator'?s note|transcriber'?s note|"
    r"dedication|contents|table of contents|list of illustrations|dramatis personae|"
    r"characters|persons of the play|notes?|editor'?s note|advertisement)\s*[:.]?\s*$",
    re.IGNORECASE,
)
BACK_SECTION_RE = re.compile(
    r"^\s*(?:transcriber'?s notes?|footnotes?|appendix|appendices|"
    r"advertisements?|other books by|bibliography|glossary)\b.*$",
    re.IGNORECASE,
)
NOTES_HEADING_RE = re.compile(r"^\s*notes?\s*[:.]?\s*$", re.IGNORECASE)
ROMAN_HEADING = r"(?:xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)"
START_HEADING_RE = re.compile(
    r"^\s*(?:(?:chapter|book|part|act|scene|canto)\s+(?:" + ROMAN_HEADING + r"|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|(?:" + ROMAN_HEADING + r")\.?\s*$"
    r"|\d+\s+[A-Z][A-Za-z0-9' -]{2,80}\s*$"
    r"|the\s+play\s*$"
    r"|the\s+story\s*$"
    r"|prologue\s*$"
    r"|epilogue\s*$"
    r"|fit\s+the\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s*$"
    r"|section\s+(?:[ivxlcdm]+|\d+)\b)",
    re.IGNORECASE,
)
SPEAKER_CUE_RE = re.compile(r"^\s*_?[A-Z][A-Z .'-]{2,40}_?\s*$")
MANUAL_START_RE = {
    "apocolocyntosis": re.compile(r"^\s*I wish to place on record\b"),
    "a_martian_odyssey": re.compile(r"^\s*Jarvis stretched himself\b"),
    "cavalleria_rusticana": re.compile(r"^\s*SCENE I\.?\s*$"),
    "chitra": re.compile(r"^\s*SCENE I\s*$"),
    "second_variety": re.compile(r"^\s*The Russian soldier made his way\b"),
    "the_black_cat": re.compile(r"^\s*CHAPTER I\.?\s*$"),
    "the_canterville_ghost": re.compile(r"^\s*When Mr\. Hiram B\. Otis\b"),
    "the_devil_s_foot": re.compile(r"^\s*In recording from time to time\b"),
    "the_frogs": re.compile(r"^\s*_XANTHIAS_\s*$"),
    "the_hunting_of_the_snark": re.compile(r"^\s*Fit the First\s*$"),
    "the_inca_of_perusalem": re.compile(r"^\s*PROLOGUE\s*$"),
    "the_new_paul_and_virginia": re.compile(r"^\s*The magnificent ocean-steamer\b"),
    "the_witch_of_atlas": re.compile(r"^\s*1\.\s*$"),
    "how_he_lied_to_her_husband": re.compile(r"^\s*It is eight o'clock in the evening\b"),
}
STORY_END_RE = re.compile(
    r"^\s*(?:THE\s+END\.?|THE\s+CURTAIN\s+FALLS\s+RAPIDLY\.?|CURTAIN\.?|FINIS\.?)\s*$",
    re.IGNORECASE,
)
INLINE_ARTIFACT_RE = re.compile(
    r"^\s*(?:\[illustration\b.*?\]|\[transcriber'?s note\b.*?\]|\[footnote\b.*?\]|"
    r"\[\d+\]|\[\*[^\]]*\])\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)
FOOTNOTE_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|\[\w+\]|\d+\.)\s+.{0,140}$")

COMMON_MOJIBAKE = {
    "â\x80\x99": "'",
    "â\x80\x98": "'",
    "â\x80\x9c": '"',
    "â\x80\x9d": '"',
    "â\x80\x94": "--",
    "â\x80\x93": "-",
    "Â ": " ",
    "Â": "",
}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def normalize_encoding(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    for bad, good in COMMON_MOJIBAKE.items():
        text = text.replace(bad, good)
    return text


def split_at_markers(text: str) -> tuple[str, bool, bool]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None

    for idx, line in enumerate(lines):
        if START_RE.match(line):
            start_idx = idx
            break
    if start_idx is None:
        start_idx = -1

    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if END_RE.match(line) or OLD_END_RE.match(line):
            end_idx = idx
            break

    body = lines[start_idx + 1 : end_idx] if end_idx is not None else lines[start_idx + 1 :]
    return "\n".join(body), start_idx != -1, end_idx is not None


def is_short_non_sentence(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) > 80:
        return False
    if re.search(r"[.!?;:]\s*$", stripped):
        return False
    if re.search(r"\.{2,}\s*\d+\s*$", stripped):
        return True
    if re.match(r"^(?:chapter|book|part|act|scene|canto)\b", stripped, re.IGNORECASE):
        return True
    return len(stripped.split()) <= 8


def find_after_section(lines: list[str], start: int) -> int:
    """Skip a front-matter section until a plausible narrative/major heading."""
    idx = start + 1
    blank_run = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            blank_run += 1
            idx += 1
            continue
        if START_HEADING_RE.match(stripped) and not FRONT_SECTION_RE.match(stripped):
            return idx
        if blank_run >= 2 and looks_like_narrative_line(stripped):
            return idx
        blank_run = 0
        idx += 1
    return start + 1


def skip_short_front_run(lines: list[str], start: int) -> int:
    idx = start
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or is_short_non_sentence(stripped):
            idx += 1
            continue
        if re.fullmatch(r"[A-Z0-9 .,'!-]{1,80}", stripped) and len(stripped.split()) <= 10:
            idx += 1
            continue
        return idx
    return start


def looks_like_narrative_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if FRONT_SECTION_RE.match(stripped) or BACK_SECTION_RE.match(stripped):
        return False
    if re.match(r"^(?:title|author|release date|language|credits|produced by)\b", stripped, re.I):
        return False
    if re.fullmatch(r"[A-Z0-9 .,'!-]{1,70}", stripped) and len(stripped.split()) <= 8:
        return False
    return bool(re.search(r"[.!?\"']\s*$", stripped)) and len(stripped.split()) >= 6


def paragraph_at(lines: list[str], start: int) -> tuple[str, int]:
    idx = start
    paragraph: list[str] = []
    while idx < len(lines) and lines[idx].strip():
        paragraph.append(lines[idx].strip())
        idx += 1
    return re.sub(r"\s+", " ", " ".join(paragraph)).strip(), idx


def looks_like_narrative_paragraph(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if any(
        phrase in lower
        for phrase in (
            "produced by",
            "distributed proofreading",
            "copyright on this publication",
            "this ebook was produced",
            "this etext was produced",
            "release date:",
            "language:",
            "credits:",
        )
    ):
        return False
    if len(text.split()) < 6:
        return False
    if re.fullmatch(r"[A-Z0-9 .,'!?:;_-]{1,120}", text) and len(text.split()) <= 12:
        return False
    return bool(re.search(r"[.!?\"']", text))


def detect_front_start(lines: list[str]) -> tuple[int, list[str]]:
    removed: list[str] = []
    scan_limit = min(len(lines), 500)

    for story_slug, pattern in MANUAL_START_RE.items():
        for idx in range(scan_limit):
            if pattern.match(lines[idx]):
                return idx, ["title/author/publisher front matter", "preface/introduction"]

    for idx in range(scan_limit):
        stripped = lines[idx].strip()
        if FRONT_SECTION_RE.match(stripped):
            label = FRONT_SECTION_RE.match(stripped).group(0).lower().strip(" .:")
            if label not in removed:
                removed.append(label)

    idx = 0
    while idx < scan_limit:
        stripped = lines[idx].strip()
        lower = stripped.lower()
        if not stripped:
            idx += 1
            continue
        front_match = FRONT_SECTION_RE.match(stripped)
        if front_match:
            label = front_match.group(0).lower().strip(" .:")
            if "contents" in label or "illustrations" in label or "characters" in label:
                idx = skip_short_front_run(lines, idx + 1)
            else:
                idx = find_after_section(lines, idx)
            continue
        if START_HEADING_RE.match(stripped):
            if idx > 0:
                removed.append("title/author/publisher front matter")
            return idx, sorted(set(removed))
        if (
            lower.startswith(("produced by", "title:", "author:", "release date:", "language:", "credits:"))
            or re.fullmatch(r"(?:by|translated by|illustrated by)", lower)
            or (re.fullmatch(r"[A-Z0-9 .,'!-]{1,80}", stripped) and len(stripped.split()) <= 10)
        ):
            idx += 1
            continue
        paragraph, para_end = paragraph_at(lines, idx)
        if looks_like_narrative_paragraph(paragraph):
            if idx > 0:
                removed.append("title/author/publisher front matter")
            return idx, sorted(set(removed))
        if looks_like_narrative_line(stripped):
            if idx > 0:
                removed.append("title/author/publisher front matter")
            return idx, sorted(set(removed))
        idx += 1

    return 0, sorted(set(removed))


def line_signature(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def repeated_headers(lines: Iterable[str]) -> set[str]:
    counts = Counter()
    for line in lines:
        sig = line_signature(line)
        if not sig:
            continue
        if 3 <= len(sig) <= 60 and not re.search(r"[.!?;:]$", sig):
            counts[sig] += 1
    return {sig for sig, count in counts.items() if count >= 4}


def remove_artifact_lines(lines: list[str]) -> list[str]:
    headers = repeated_headers(lines)
    cleaned: list[str] = []
    skip_bracket_block = False
    for line in lines:
        stripped = line.strip()
        sig = line_signature(line)

        if skip_bracket_block:
            if "]" in stripped:
                skip_bracket_block = False
            continue

        if stripped.startswith("[") and not stripped.endswith("]") and re.match(
            r"^\[(?:illustration|transcriber'?s note|footnote)\b", stripped, re.I
        ):
            skip_bracket_block = True
            continue

        if INLINE_ARTIFACT_RE.match(stripped):
            continue
        if PAGE_NUMBER_RE.match(stripped):
            continue
        if sig in headers:
            continue
        if FOOTNOTE_LINE_RE.match(stripped) and re.search(r"\b(?:ibid|see|page|chapter|volume|note)\b", stripped, re.I):
            continue
        cleaned.append(line.rstrip())
    return cleaned


def truncate_back_matter(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if idx < 20:
            continue
        stripped = line.strip()
        if STORY_END_RE.match(stripped):
            return lines[: idx + 1]
        if BACK_SECTION_RE.match(stripped) or NOTES_HEADING_RE.match(stripped):
            return lines[:idx]
    return lines


def collapse_blank_lines(lines: list[str]) -> str:
    out: list[str] = []
    blanks = 0
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            blanks += 1
            if blanks <= 2:
                out.append("")
            continue
        blanks = 0
        out.append(stripped)
    return "\n".join(out).strip() + "\n"


def clean_story(source_path: Path) -> dict:
    story_dir = source_path.parent
    metadata_path = story_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = normalize_encoding(source_path.read_text(encoding="utf-8", errors="replace"))
    raw_words = word_count(raw)

    body, start_found, end_found = split_at_markers(raw)
    body = normalize_encoding(body)
    lines = body.splitlines()

    start_idx, front_removed = detect_front_start(lines)
    lines = lines[start_idx:]
    lines = truncate_back_matter(lines)
    lines = remove_artifact_lines(lines)
    cleaned = collapse_blank_lines(lines)

    cleaned_path = story_dir / "cleaned.txt"
    cleaned_path.write_text(cleaned, encoding="utf-8")

    cleaned_words = word_count(cleaned)
    pct_removed = round((1 - (cleaned_words / raw_words)) * 100, 2) if raw_words else 0.0
    report = {
        "story_id": metadata.get("document_id") or story_dir.name,
        "raw_word_count": raw_words,
        "cleaned_word_count": cleaned_words,
        "pct_removed": pct_removed,
        "start_marker_found": start_found,
        "end_marker_found": end_found,
        "front_matter_removed": front_removed,
        "first_200_chars": cleaned[:200],
        "last_200_chars": cleaned[-200:],
    }
    report["manual_review_flags"] = manual_flags(report)
    (story_dir / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**metadata, **report, "path": str(story_dir)}


def first_text_looks_ok(text: str) -> bool:
    stripped = text.lstrip()
    first_line = stripped.splitlines()[0].strip() if stripped.splitlines() else ""
    if START_HEADING_RE.match(first_line) or SPEAKER_CUE_RE.match(first_line):
        return True
    if re.fullmatch(r"\d+\.?", first_line):
        return True
    first_para = re.sub(r"\s+", " ", stripped.split("\n\n", 1)[0]).strip()
    if len(first_para.split()) >= 6 and re.search(r"[.!?\"'“”‘’]", first_para):
        return True
    return looks_like_narrative_line(first_line)


def manual_flags(report: dict) -> list[str]:
    flags: list[str] = []
    pct = report["pct_removed"]
    if pct > 25:
        flags.append("pct_removed > 25%")
    if pct < 2:
        flags.append("pct_removed < 2%")
    if not report["start_marker_found"]:
        flags.append("start marker not found")
    if not report["end_marker_found"]:
        flags.append("end marker not found")
    if not first_text_looks_ok(report["first_200_chars"]):
        flags.append("first_200_chars may not be narrative/chapter heading")
    return flags


def main() -> None:
    source_paths = sorted(ROOT.glob("*/*/*/source.txt"))
    reports = [clean_story(path) for path in source_paths]

    fields = [
        "length_bucket",
        "genre",
        "wiki_title",
        "raw_word_count",
        "cleaned_word_count",
        "pct_removed",
        "start_marker_found",
        "end_marker_found",
        "manual_review_flags",
        "path",
    ]
    with (ROOT / "clean_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            row = dict(report)
            row["manual_review_flags"] = "; ".join(row.get("manual_review_flags", []))
            writer.writerow(row)

    print(
        f"{'bucket':<6} {'genre':<14} {'title':<42} {'raw':>7} {'clean':>7} {'rem%':>6} flags"
    )
    print("-" * 104)
    for report in reports:
        flags = ", ".join(report.get("manual_review_flags", []))
        print(
            f"{report['length_bucket']:<6} {report['genre']:<14} "
            f"{report['wiki_title'][:42]:<42} "
            f"{report['raw_word_count']:>7} {report['cleaned_word_count']:>7} "
            f"{report['pct_removed']:>6.2f} {flags}"
        )


if __name__ == "__main__":
    main()
