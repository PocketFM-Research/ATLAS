"""
Per-scene durable-fact extractor.

Phase-0 add-on for the graph_verifier pipeline. For each scene, the LLM
emits an entity-anchored profile: every durable predicate the scene asserts
is bound to a named narrative entity (Character / Object / Place / Event /
Organization / Vehicle), never left as a free-floating noun.

This is the key constraint vs a naive attribute extractor: "BLUE CERAMIC MUG"
on its own is useless for continuity — what matters is that *Marcus* drinks
from it. The prompt forces that binding and explicitly covers quantities,
named dates, possessions, and absence-of-state predicates ("no children",
"no prior arrest"), which the previous version dropped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ...llm.base import BaseLLM
from ...utils.json_repair import parse_llm_json

logger = logging.getLogger(__name__)


_SYSTEM = (
    "You extract DURABLE facts about NAMED narrative entities from a "
    "screenplay scene. Every fact you emit must be anchored to a specific "
    "named entity — a person, a named object, a named place, a named "
    "organization, a named vehicle, or a named event. Do NOT emit "
    "free-floating descriptions of unnamed nouns ('BLUE MUG' on its own is "
    "wrong — what matters is the entity who uses, owns, or interacts with "
    "it). Capture only facts whose value would create a continuity error if "
    "a later scene asserted a different value. Skip transient state (current "
    "location, current mood, current action) unless the action establishes a "
    "durable predicate (e.g. an injury, an arrest, a marriage). "
    "Output strict JSON only."
)


def _build_prompt(scene_number: int, scene_text: str) -> str:
    return f"""SCENE {scene_number}:
{scene_text}

TASK — produce an entity-anchored fact profile of this scene.

Step 1 (silent): identify every NAMED narrative entity in the scene.
That means people with names, named objects (a specific named item, a brand,
a uniquely described physical object), named places, named organizations,
named vehicles, named events.

Step 2: for each named entity, list every durable predicate the scene
asserts about it. Predicates fall into these categories — use them as keys:

  Identity / classification:
    species, type, occupation, role, title, name_form, identity_type
  Physical:
    color, material, side, age, condition, distinguishing_feature
  Quantitative (CAPTURE EVERY NUMBER):
    count, quantity, money_amount, age_years, year, date, day, duration,
    measurement, price
  Possession / use:
    owns, holds, wears, uses, drinks, eats, drives, carries
  Social / kinship:
    kinship, marital_status, life_status, employer, affiliated_with
  State (INCLUDING ABSENCES — "no children" is a fact):
    has_children, has_siblings, is_married, is_alive, is_arrested,
    is_injured, is_imprisoned, locked_state, intact_state, hidden_location
  Causal / origin:
    cause_of_injury, cause_of_death, origin_of_object, source_of_warning

You may use other category keys if needed, but every key must describe a
DURABLE property (one whose value is stable across scenes).

Step 3: emit JSON of this exact shape:

{{
  "<EntityName as it appears in the scene>": {{
    "<predicate_key>": "<value in 1-5 words>",
    ...
  }},
  ...
}}

CRITICAL RULES:
  - Every entity key must be a named entity. Do NOT include a free-floating
    noun like "BLUE CERAMIC MUG" — instead, attach that fact to the
    character who owns/uses/holds the mug, e.g.
      "Marcus": {{"drinks": "tea", "holds": "blue ceramic mug"}}.
  - Capture EVERY number mentioned (counts, years, durations, money,
    quantities). Numbers are the highest-signal continuity facts.
  - Capture absence predicates explicitly. If a character says "I have no
    children," emit {{"has_children": "no"}}. Continuity errors against
    absence are common.
  - Capture name spellings, titles, roles ("Deputy Briggs", "Dr Lee",
    "editor Claire", "first production at X") under name_form / title /
    role / occupation. These are common continuity errors.
  - Do NOT include scene-transient facts (current location, current action,
    current mood) unless the action establishes a durable state.
  - Omit any entity with no durable predicates worth listing.

If the scene asserts nothing durable about any named entity, output {{}}.
"""


def extract_scene_attributes(
    scene_number: int,
    scene_text: str,
    llm: BaseLLM,
    max_tokens: int = 3072,
) -> Dict[str, Dict[str, Any]]:
    """Run one LLM call to extract the durable-fact profile for one scene."""
    if not scene_text or not scene_text.strip():
        return {}

    prompt = _build_prompt(scene_number, scene_text)
    try:
        raw = llm.complete(
            prompt=prompt,
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("attribute extraction failed for scene %d: %s", scene_number, exc)
        return {}

    parsed = parse_llm_json(raw, schema_hint="object")
    if not isinstance(parsed, dict):
        return {}

    cleaned: Dict[str, Dict[str, Any]] = {}
    for entity, attrs in parsed.items():
        if not isinstance(entity, str) or not isinstance(attrs, dict):
            continue
        kept: Dict[str, Any] = {}
        for attr_key, attr_val in attrs.items():
            if not isinstance(attr_key, str):
                continue
            if attr_val is None:
                continue
            kept[attr_key.strip()] = attr_val
        if kept:
            cleaned[entity.strip()] = kept
    return cleaned


def extract_all_scene_attributes(
    scene_records: Iterable[Any],
    llm: BaseLLM,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Extract entity-anchored fact profiles for every scene.

    Args:
        scene_records: iterable of SceneRecord-like objects (must have
            .scene_id, .order, .content).
        llm: LLM used for extraction.
        cache_dir: if provided, profiles are read/written under
            <cache_dir>/scene_<NNN>_attributes.json. Reused on subsequent runs.
        force: if True, ignore cached files and re-extract.

    Returns:
        {scene_number: {entity_name: {predicate: value}}}
    """
    results: Dict[int, Dict[str, Dict[str, Any]]] = {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for scene in scene_records:
        try:
            scene_number = int(scene.scene_id)
        except (TypeError, ValueError, AttributeError):
            scene_number = getattr(scene, "order", 0) + 1

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_path = cache_dir / f"scene_{scene_number:03d}_attributes.json"
            if cache_path.exists() and not force:
                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if isinstance(cached, dict):
                        results[scene_number] = cached
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("could not read attribute cache %s: %s", cache_path, exc)

        profile = extract_scene_attributes(
            scene_number=scene_number,
            scene_text=scene.content,
            llm=llm,
        )
        results[scene_number] = profile

        if cache_path is not None:
            try:
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(profile, f, ensure_ascii=False, indent=2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not write attribute cache %s: %s", cache_path, exc)

    return results
