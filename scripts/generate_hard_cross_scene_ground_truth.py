#!/usr/bin/env python3
"""Generate harder cross-scene repetition ground-truth cases.

The original `inserted_error_ground_truth_cross_scene.json` cases were near-
verbatim copies with mechanical synonym substitution (door->entry, walks->moves,
car->vehicle). Detectors score ~100/100 on them which tells you almost nothing.

This script produces four tiers per movie:

  paraphrase_full   -- whole scene rewritten; same characters/events/outcomes,
                       max ~0.25 content-word Jaccard with the source.
  beat_only         -- same dramatic function (e.g. "antagonist coerces
                       protagonist into commitment via emotional leverage"),
                       different characters, different location, different
                       surface content. Tests beat-level abstraction.
  partial_overlap   -- scene with new unrelated opening + a paraphrased replay
                       of the source's climactic beat in the back half. Tests
                       localization.
  motif_distractor  -- HARD NEGATIVE. A recurring callback (shared line, image,
                       or symbol) that is narratively justified, not a replay.
                       is_repetition=false.

Outputs:
  scene_text_exports/<movie>_cross_scene_hard.txt
      = clean scene text + appended hard scenes, numbered continuing after the
        last clean scene.
  scene_text_exports/inserted_error_ground_truth_cross_scene_hard.json
      = ground truth keyed by movie with per-instance tier, narrative function,
        target/actual Jaccard, is_repetition flag.

API: Google Gemini via google-genai. Set GEMINI_API_KEY or pass --api_key_file.
The endpoint is fixed by the SDK (generativelanguage.googleapis.com); pass
--model to swap (default gemini-2.5-flash).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SCENE_HEADING_RE = re.compile(
    r"^\s*(\d+)\s*[\).:：、-]\s*(.+?)\s*$"
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
STOPWORDS = {
    "a", "about", "again", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "been", "before", "but", "by", "could", "did", "didn", "do",
    "does", "don", "even", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "like", "maybe", "no", "not", "of", "on", "or", "s", "she", "so", "still",
    "t", "that", "the", "there", "they", "this", "though", "to", "too", "ve",
    "was", "wasn", "what", "when", "where", "while", "why", "with", "would",
    "you", "your", "we", "our", "us", "out", "up", "down", "off", "over",
    "under", "back", "him", "her",
}


# ----------------------------- scene parsing ------------------------------


def parse_scenes(text: str) -> Dict[int, Tuple[str, str]]:
    """Return {scene_number: (heading, body_text)}."""
    scenes: Dict[int, Tuple[str, str]] = {}
    current_num: Optional[int] = None
    current_heading: str = ""
    current_lines: List[str] = []

    def flush():
        if current_num is not None and current_lines:
            body = "\n".join(current_lines).strip()
            scenes[current_num] = (current_heading, body)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = SCENE_HEADING_RE.match(line)
        if match:
            flush()
            current_num = int(match.group(1))
            current_heading = line.strip()
            current_lines = []
            continue
        if current_num is not None:
            current_lines.append(raw_line)

    flush()
    return scenes


def content_tokens(text: str) -> set:
    return {
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ----------------------------- generation ---------------------------------


@dataclass
class TierSpec:
    name: str
    max_jaccard: float
    is_repetition: bool
    system_instructions: str
    expected_difficulty: str
    subtype: str


TIER_SPECS: Dict[str, TierSpec] = {
    "paraphrase_full": TierSpec(
        name="paraphrase_full",
        max_jaccard=0.30,
        is_repetition=True,
        expected_difficulty="medium",
        subtype="paraphrase_whole_scene",
        system_instructions=(
            "You are rewriting a screenplay scene so that an automated repetition "
            "detector that looks at surface lexical overlap will MISS it, but a "
            "detector that understands semantic content SHOULD catch it.\n\n"
            "Requirements:\n"
            "- Preserve the SAME characters (use their canonical names exactly as "
            "in the source), the SAME sequence of events, the SAME outcomes, the "
            "SAME emotional turn.\n"
            "- Rewrite every line. No sentence from the source may appear with "
            "more than three consecutive shared content words.\n"
            "- Change the location/setting heading to a different but plausible "
            "place (e.g. kitchen -> back patio, gym -> rooftop).\n"
            "- Vary sentence structure, dialogue phrasing, and stage direction "
            "vocabulary. Avoid one-to-one synonym swaps; restructure sentences.\n"
            "- Match the source scene's length within +/- 30%.\n"
            "- Do NOT include the source scene number or heading in the output."
        ),
    ),
    "beat_only": TierSpec(
        name="beat_only",
        max_jaccard=0.18,
        is_repetition=True,
        expected_difficulty="hard",
        subtype="dramatic_beat_replay",
        system_instructions=(
            "You are writing a new screenplay scene that replays the SAME "
            "dramatic beat as a source scene -- the same setup, conflict, "
            "tactic, and resolution function -- but with DIFFERENT characters, "
            "DIFFERENT location, DIFFERENT surface content, and DIFFERENT props.\n\n"
            "Requirements:\n"
            "- Identify the dramatic function of the source scene (e.g. "
            "'authority figure pressures subordinate into commitment via "
            "guilt; subordinate folds').\n"
            "- Write a scene that performs the same function with completely "
            "different people, place, and topic. The characters here must NOT "
            "be the same as the source characters. The location must be a "
            "context that has not appeared yet.\n"
            "- No character name from the source may appear in this scene.\n"
            "- No proper noun (location, object, organization) from the source "
            "may appear.\n"
            "- The audience-felt beat (who pushes, who folds, why) must match. "
            "The lexical surface must not.\n"
            "- Output: scene heading + full scene body, normal screenplay "
            "format. Do not annotate or label the beat structure."
        ),
    ),
    "partial_overlap": TierSpec(
        name="partial_overlap",
        max_jaccard=0.22,
        is_repetition=True,
        expected_difficulty="hard",
        subtype="partial_paraphrase_back_half",
        system_instructions=(
            "You are writing a screenplay scene that begins with new unrelated "
            "material and then, in its back half, replays the climactic beat of "
            "a source scene in paraphrased form.\n\n"
            "Requirements:\n"
            "- First ~60% of the new scene: original action with characters and "
            "events that do not appear in the source. This part must be "
            "lexically and semantically independent of the source.\n"
            "- Final ~40%: a paraphrased replay of the source scene's climactic "
            "beat. Same characters (use canonical names) and same outcome as "
            "the source's climax, but rewritten (no sentence shares more than "
            "three consecutive content words with the source).\n"
            "- Insert a clear transition (a new arrival, a phone ring, a cut) "
            "between the two halves. The scene heading should reflect the "
            "opening setting.\n"
            "- Length: 1.1x to 1.4x the source scene length."
        ),
    ),
    "motif_distractor": TierSpec(
        name="motif_distractor",
        max_jaccard=0.40,  # may share more surface than other tiers
        is_repetition=False,
        expected_difficulty="hard_negative",
        subtype="justified_callback",
        system_instructions=(
            "You are writing a screenplay scene that is NOT a repetition of the "
            "source scene, but that DOES contain a recurring motif from it "
            "(e.g. a recurring line of dialogue, a recurring object, a recurring "
            "gesture). The motif appears because the story calls back to it on "
            "purpose, not because the scene replays a prior beat.\n\n"
            "Requirements:\n"
            "- The plot of the new scene must be different from the source: "
            "different conflict, different stakes, different outcome.\n"
            "- Reuse exactly ONE motif from the source (one quoted line OR one "
            "named object OR one specific gesture). Keep the motif close to "
            "verbatim so a naive lexical detector will be tempted.\n"
            "- The motif must function as a callback (irony, contrast, or "
            "thematic echo), not as a replay of the source's beat.\n"
            "- Characters may overlap with the source; events and outcomes "
            "must not.\n"
            "- Length: 60-120% of the source scene."
        ),
    ),
}


def build_prompt(tier: TierSpec, source_heading: str, source_body: str,
                 next_scene_number: int) -> Tuple[str, str]:
    system = tier.system_instructions
    user = (
        f"SOURCE SCENE (number {source_heading.split(chr(12289))[0].strip() if chr(12289) in source_heading else '?'}):\n"
        f"{source_heading}\n{source_body}\n\n"
        f"Write the new scene now. Output ONLY the new scene -- a scene "
        f"heading on its own line followed by the scene body. The heading "
        f"must start with '{next_scene_number}、' (the scene number "
        f"followed by the ideographic comma U+3001), then 'INT.' or 'EXT.', "
        f"then the new location, optionally a time-of-day. Example heading: "
        f"'{next_scene_number}、INT. SOMEWHERE NEW - NIGHT'. Do not "
        f"include any commentary, labels, or markdown -- just the scene.\n"
    )
    return system, user


# ----------------------------- LLM client ---------------------------------


def get_gemini_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "google-genai not installed. Run: pip install google-genai"
        ) from exc
    return genai.Client(api_key=api_key)


def call_gemini(client, model: str, system: str, user: str,
                temperature: float, max_tokens: int,
                max_retries: int = 4) -> str:
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=config,
            )
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2 * attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# ----------------------------- pipeline -----------------------------------


@dataclass
class GoldEntry:
    id: str
    category: str
    subtype: str
    tier: str
    is_repetition: bool
    source_scene: int
    repeated_scene: int
    source_heading: str
    repeated_heading: str
    target_max_jaccard: float
    actual_jaccard: float
    expected_difficulty: str
    inserted_change: str
    repeated_body_preview: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MovieConfig:
    movie_key: str           # e.g. "the_fighter"
    clean_text_path: Path
    out_text_path: Path
    tier_counts: Dict[str, int] = field(default_factory=dict)
    seed: int = 42
    pairs: List[Tuple[int, str]] = field(default_factory=list)
    # pairs = [(source_scene_number, tier_name), ...]


def default_pair_plan(
    scene_numbers: List[int],
    seed: int,
    counts: Dict[str, int],
) -> List[Tuple[int, str]]:
    """Pick source scenes spread across the script, one per tier slot.

    `counts` is `{tier_name: n_cases}`. The total N source scenes are sampled
    evenly across the script (with replacement only if N > number of scenes).
    """
    total = sum(counts.values())
    rng = random.Random(seed)
    if not scene_numbers or total == 0:
        return []
    if total <= len(scene_numbers):
        step = len(scene_numbers) / total
        chosen = [scene_numbers[int(i * step)] for i in range(total)]
    else:
        # not enough unique scenes -- allow repeats so we still get N cases
        chosen = [scene_numbers[i % len(scene_numbers)] for i in range(total)]
    rng.shuffle(chosen)
    tiers: List[str] = []
    for tier_name, n in counts.items():
        tiers.extend([tier_name] * n)
    rng.shuffle(tiers)
    return list(zip(chosen, tiers))


def render_heading(scene_num: int, generated_first_line: str) -> str:
    """Force the heading to start with `<num>、` for consistency with the
    rest of the file."""
    first = generated_first_line.strip()
    body = re.sub(r"^\s*\d+\s*[、\).:：、-]\s*", "", first).strip()
    if not body:
        body = "INT. UNKNOWN LOCATION"
    return f"{scene_num}、{body}"


def split_first_line(text: str) -> Tuple[str, str]:
    parts = text.split("\n", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def run_movie(cfg: MovieConfig, client, model: str, temperature: float,
              max_tokens: int, dry_run: bool) -> Tuple[str, List[GoldEntry]]:
    clean_text = cfg.clean_text_path.read_text(encoding="utf-8")
    scenes = parse_scenes(clean_text)
    if not scenes:
        raise SystemExit(f"No scenes parsed from {cfg.clean_text_path}")

    last_clean_num = max(scenes)
    next_num = last_clean_num + 1
    appended: List[str] = []
    gold: List[GoldEntry] = []

    pairs = cfg.pairs or default_pair_plan(
        sorted(scenes), seed=cfg.seed, counts=cfg.tier_counts,
    )

    for idx, (source_num, tier_name) in enumerate(pairs, start=1):
        if source_num not in scenes:
            print(f"  skip: source scene {source_num} not in {cfg.movie_key}",
                  file=sys.stderr)
            continue
        tier = TIER_SPECS[tier_name]
        source_heading, source_body = scenes[source_num]

        system, user = build_prompt(tier, source_heading, source_body, next_num)
        if dry_run:
            generated = (
                f"{next_num}、INT. DRY-RUN PLACEHOLDER - DAY\n"
                f"[Would call Gemini for tier={tier.name} on source scene "
                f"{source_num}]"
            )
        else:
            print(
                f"  {cfg.movie_key}: tier={tier.name} src={source_num} "
                f"-> scene {next_num}",
                file=sys.stderr,
            )
            generated = call_gemini(
                client, model, system, user, temperature, max_tokens,
            )
        first_line, body = split_first_line(generated)
        heading = render_heading(next_num, first_line)
        full_scene_text = f"{heading}\n{body.strip()}".strip()
        appended.append(full_scene_text)

        actual_j = jaccard(source_body, body)
        gold.append(GoldEntry(
            id=f"HXR{idx:03d}",
            category="cross_scene_repetition",
            subtype=tier.subtype,
            tier=tier.name,
            is_repetition=tier.is_repetition,
            source_scene=source_num,
            repeated_scene=next_num,
            source_heading=source_heading,
            repeated_heading=heading,
            target_max_jaccard=tier.max_jaccard,
            actual_jaccard=round(actual_j, 4),
            expected_difficulty=tier.expected_difficulty,
            inserted_change=_describe(tier, source_heading),
            repeated_body_preview=body.strip()[:300],
        ))
        next_num += 1

    out_text = clean_text.rstrip() + "\n\n\n" + "\n\n\n".join(appended) + "\n"
    return out_text, gold


def _describe(tier: TierSpec, source_heading: str) -> str:
    if tier.name == "paraphrase_full":
        return (
            "Whole-scene paraphrase: same characters, sequence of events, and "
            "outcome as the source; rewritten with no shared multi-word spans; "
            "setting changed."
        )
    if tier.name == "beat_only":
        return (
            "Beat-level replay: same dramatic function (setup/conflict/tactic/"
            "resolution) as the source; different characters, location, and "
            "surface content. Tests beat-structure detection independent of "
            "lexical or entity overlap."
        )
    if tier.name == "partial_overlap":
        return (
            "Partial replay: scene opens on unrelated new material then "
            "paraphrases the source's climactic beat in its back half. Tests "
            "whether the detector localizes a repeated beat inside a longer "
            "scene."
        )
    if tier.name == "motif_distractor":
        return (
            "HARD NEGATIVE: scene reuses one motif (line / object / gesture) "
            "from the source as a narratively justified callback, but the "
            "plot, conflict, and outcome differ. is_repetition=false. Tests "
            "precision against recurring motifs."
        )
    return ""


# ----------------------------- CLI ---------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate harder cross-scene repetition ground-truth cases "
            "(paraphrase / beat / partial / hard-negative)."
        )
    )
    parser.add_argument(
        "--scene_text_dir", type=Path,
        default=Path("scene_text_exports"),
        help="Directory containing <movie>_clean_scene_text.txt files.",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path("scene_text_exports"),
        help="Where to write <movie>_cross_scene_hard.txt and the gold JSON.",
    )
    parser.add_argument(
        "--gold_out_name", type=str,
        default="inserted_error_ground_truth_cross_scene_hard.json",
    )
    parser.add_argument(
        "--movies", nargs="+",
        default=["the_fighter", "intolerable_cruelty"],
        help="Movie keys (matching <key>_clean_scene_text.txt).",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model id.",
    )
    parser.add_argument(
        "--api_key", default=None,
        help="Gemini API key (overrides env + file).",
    )
    parser.add_argument(
        "--api_key_file", type=Path, default=None,
        help="File containing the Gemini API key (e.g. gemini.txt).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Higher temperature -> more diverse paraphrases. Default 0.7.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Skip LLM calls; emit placeholder scenes. Useful for plumbing.",
    )
    parser.add_argument(
        "--n_paraphrase_full", type=int, default=12,
        help="Number of paraphrase_full cases per movie. Default 12.",
    )
    parser.add_argument(
        "--n_beat_only", type=int, default=12,
        help="Number of beat_only cases per movie. Default 12.",
    )
    parser.add_argument(
        "--n_partial_overlap", type=int, default=6,
        help="Number of partial_overlap cases per movie. Default 6.",
    )
    parser.add_argument(
        "--n_motif_distractor", type=int, default=6,
        help="Number of motif_distractor hard-negative cases per movie. Default 6.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for source-scene selection and tier shuffling.",
    )
    return parser.parse_args(argv)


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file and args.api_key_file.exists():
        return args.api_key_file.read_text(encoding="utf-8").strip()
    env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env:
        return env.strip()
    raise SystemExit(
        "No API key. Pass --api_key, --api_key_file gemini.txt, or set "
        "GEMINI_API_KEY in the environment."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        client = None
    else:
        client = get_gemini_client(resolve_api_key(args))

    tier_counts = {
        "paraphrase_full": args.n_paraphrase_full,
        "beat_only": args.n_beat_only,
        "partial_overlap": args.n_partial_overlap,
        "motif_distractor": args.n_motif_distractor,
    }
    total_per_movie = sum(tier_counts.values())
    print(
        f"Plan: {total_per_movie} cases per movie  "
        + "  ".join(f"{k}={v}" for k, v in tier_counts.items()),
        file=sys.stderr,
    )

    gold_all: Dict[str, List[dict]] = {}
    for movie_key in args.movies:
        clean_path = args.scene_text_dir / f"{movie_key}_clean_scene_text.txt"
        if not clean_path.exists():
            print(f"missing: {clean_path}", file=sys.stderr)
            continue
        out_text_path = args.out_dir / f"{movie_key}_cross_scene_hard.txt"
        cfg = MovieConfig(
            movie_key=movie_key,
            clean_text_path=clean_path,
            out_text_path=out_text_path,
            tier_counts=tier_counts,
            seed=args.seed,
        )
        out_text, gold = run_movie(
            cfg, client, args.model,
            args.temperature, args.max_tokens,
            args.dry_run,
        )
        out_text_path.write_text(out_text, encoding="utf-8")
        gold_all[movie_key] = [g.to_dict() for g in gold]

        # quality report
        by_tier: Dict[str, List[float]] = {}
        violations = []
        for g in gold:
            by_tier.setdefault(g.tier, []).append(g.actual_jaccard)
            if g.is_repetition and g.actual_jaccard > g.target_max_jaccard:
                violations.append(g)
        print(f"\n{movie_key}: wrote {out_text_path}")
        for tier, vals in by_tier.items():
            print(
                f"  {tier:18s}  n={len(vals)}  "
                f"jaccard mean={sum(vals)/len(vals):.3f}  "
                f"max={max(vals):.3f}"
            )
        if violations:
            print(
                f"  WARNING: {len(violations)} case(s) above target jaccard "
                f"-- regenerate or increase temperature:"
            )
            for v in violations:
                print(
                    f"    {v.id} tier={v.tier} actual={v.actual_jaccard:.3f} "
                    f"target<={v.target_max_jaccard:.3f}"
                )

    gold_path = args.out_dir / args.gold_out_name
    gold_path.write_text(
        json.dumps(gold_all, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote ground truth: {gold_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
