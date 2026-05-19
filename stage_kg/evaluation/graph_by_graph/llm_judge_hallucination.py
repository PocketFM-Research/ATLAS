"""
LLM-as-a-judge baseline for temporal hallucination detection.

Reads the whole screenplay scene-by-scene and asks the LLM to output
hallucinations in the canonical format.

This is the head-to-head baseline against graph_verifier.py and must use
the SAME LLM configuration (Azure GPT-5.4 mini) as graph construction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...llm.base import BaseLLM
from ...utils.json_repair import parse_llm_json
from .graph_verifier import HALLUCINATION_FORMAT, parse_hallucination_string

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are an exhaustive temporal continuity checker for screenplays. "
    "You read the whole screenplay scene by scene and report ONLY temporal "
    "hallucinations / continuity errors: places where a later scene "
    "contradicts an earlier-established fact, OR a later scene assumes a "
    "prior state that no earlier scene established and the current scene "
    "does not show being established. "
    "Do NOT flag normal new information that the current scene clearly "
    "introduces (first arrivals, first reveals, first uses of a known place). "
    "Do NOT flag normal movement between locations or rooms. "
    "DO flag durable contradictions on attribute categories such as color, "
    "number/quantity, date/time, body side, cause/origin, object identity, "
    "material, owner, lock/key state, kinship/marital/life status, named "
    "role or title, and named entities. "
    "Aim for high recall — better to risk a false positive than to silently "
    "skip a real contradiction. You output only JSON."
)


def _build_prompt(scenes_block: str) -> str:
    fmt_example = HALLUCINATION_FORMAT.format(
        x="X", fact="[FACT]", y="Y", event="[EVENT]"
    )
    return f"""SCREENPLAY (scene-by-scene):
{scenes_block}

TASK:
Find EVERY temporal hallucination / continuity error in the screenplay.
A temporal hallucination is one of:
  (a) a later scene that contradicts a fact explicitly established in an
      earlier scene; or
  (b) a later scene that presupposes a prior state that no earlier scene
      established, and the current scene does not show that state being
      established.

Procedure — apply this systematically:
  1. For every named entity introduced (characters, locations, objects,
     organizations), track its durable attributes as you read forward:
     name/title, identity/type, color, material, side, owner, count, age,
     date facts, role, relationships, life/death status, locked/intact
     status, cause/origin of past events, etc.
  2. Whenever a later scene asserts a value that contradicts a previously
     tracked value for the same entity/attribute, emit a hallucination.
  3. Whenever a later scene presupposes a prior state that no earlier
     scene established, and the current scene does not show that state
     being established, emit a hallucination.

Do NOT flag:
  - Normal new information that the current scene clearly introduces (first
    arrivals, first reveals, first uses of a known place).
  - Ordinary movement between rooms, buildings, or locations.
  - Re-statements of the same fact in different surface wording.

OUTPUT — strictly valid JSON list, no prose, no markdown:
[
  {{"id": 1, "hallucination": "{fmt_example}"}}
]

Every "hallucination" string MUST follow exactly this template:
"{fmt_example}"

Where:
  - X is the earlier scene number that established (or should have established)
    the relevant fact.
  - Y is the later scene number where the contradicting/unsupported event happens.
  - X < Y, both are integers.
  - [FACT] is a short noun phrase or clause (≤15 words) describing the
    earlier-established fact. Do NOT include the words "was established"
    inside [FACT].
  - [EVENT] is a short noun phrase or clause (≤15 words) describing what
    happens in the later scene. Do NOT include the word "happens" inside
    [EVENT].

If there are no hallucinations, output [].
"""


def _format_scenes(scenes: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for scene in scenes:
        s = scene.get("scene_number") or scene.get("scene_id")
        title = scene.get("title", "")
        content = scene.get("content", "")
        header = f"=== SCENE {s} ==="
        if title:
            header += f" {title}"
        blocks.append(f"{header}\n{content}".strip())
    return "\n\n".join(blocks)


def run_llm_judge(
    scenes: List[Dict[str, Any]],
    llm: BaseLLM,
    max_tokens: int = 12288,
    raw_output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run the LLM judge baseline over an ordered list of scene dicts.

    Each scene dict should have:
      - "scene_number" (int) OR "scene_id" (str/int)
      - "title" (optional)
      - "content" (the scene text)

    Returns a list of {"id": int, "hallucination": str} dicts in the exact
    template. Malformed entries are dropped.
    """
    if not scenes:
        return []

    scenes_block = _format_scenes(scenes)
    prompt = _build_prompt(scenes_block)

    try:
        raw = llm.complete(
            prompt=prompt,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM judge call failed: %s", exc)
        return []

    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(raw or "", encoding="utf-8")

    parsed = parse_llm_json(raw, schema_hint="list")
    if parsed is None:
        parsed = _fallback_extract_hallucination_items(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    if not parsed:
        parsed = _fallback_extract_hallucination_items(raw)

    cleaned: List[Dict[str, Any]] = []
    seen: set = set()
    next_id = 1
    for item in parsed:
        if not isinstance(item, dict):
            continue
        h = item.get("hallucination")
        if not isinstance(h, str):
            continue
        parsed_tuple = parse_hallucination_string(h)
        if parsed_tuple is None:
            # Skip strings that don't follow the template.
            continue
        x, _fact, y, _event = parsed_tuple
        if x >= y:
            # Temporal hallucinations require the contradicting/unsupported
            # event to be in a strictly LATER scene than the established fact.
            continue
        if h in seen:
            continue
        seen.add(h)
        cleaned.append({"id": next_id, "hallucination": h})
        next_id += 1

    return cleaned


def _fallback_extract_hallucination_items(raw: str) -> List[Dict[str, str]]:
    """Recover template-formatted hallucinations from non-JSON model output."""
    if not raw:
        return []
    pattern = re.compile(
        r"In\s+scene\s+\d+\s*,\s*.+?\s+was\s+established\s*,\s*yet\s+in\s+scene\s+\d+\s*,\s*.+?\s+happens\.?",
        re.IGNORECASE | re.DOTALL,
    )
    items: List[Dict[str, str]] = []
    for match in pattern.finditer(raw):
        text = " ".join(match.group(0).split())
        items.append({"hallucination": text})
    return items
