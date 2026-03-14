"""
Entity extraction (Pass 2).

Extracts entities anchored to the event inventory from Pass 1.
Node types: Character, Location, TimePoint, Object, Concept.
"""

import logging
import uuid
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import entity_extraction as eep
from ..utils.json_repair import parse_llm_json, validate_entity_list
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000
VALID_TYPES = {"Character", "Location", "TimePoint", "Object", "Concept"}


def extract_entities_for_movie(
    scenes: List[SceneRecord],
    scene_events: Dict[str, List[Dict]],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict[str, List[Dict]]:
    """
    Run entity extraction over all scenes.

    Args:
        scene_events: Output from event extraction {scene_id -> [events]}.

    Returns:
        Dict mapping scene_id -> list of raw entity dicts.
    """
    all_entities: Dict[str, List[Dict]] = {}

    for scene in scenes:
        events = scene_events.get(scene.scene_id, [])
        scene_ents = extract_entities_for_scene(
            scene=scene,
            events=events,
            llm=llm,
            movie_id=movie_id,
            movie_title=movie_title,
            cache=cache,
            prompt_logger=prompt_logger,
        )
        all_entities[scene.scene_id] = scene_ents

    total = sum(len(v) for v in all_entities.values())
    logger.info(
        "[%s] Entity extraction complete: %d entities across %d scenes",
        movie_id, total, len(scenes),
    )
    return all_entities


def extract_entities_for_scene(
    scene: SceneRecord,
    events: List[Dict],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> List[Dict]:
    """Extract entities from a single scene."""
    scene_ents: List[Dict] = []

    for chunk in scene.chunks:
        chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        cache_key = Cache.make_key(movie_id, "entities", scene.scene_id, chunk_id)

        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: entities for %s/%s", scene.scene_id, chunk_id)
            scene_ents.extend(cached)
            continue

        chunk_text = chunk.get("content", "")
        if not chunk_text.strip():
            continue
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]

        # Filter events to those from this chunk
        chunk_events = [e for e in events if e.get("chunk_id") == chunk_id] or events

        prompt = eep.build_prompt(
            scene_id=scene.scene_id,
            scene_title=scene.title,
            scene_text=chunk_text,
            events=_summarize_events(chunk_events),
            chunk_id=chunk_id,
            movie_title=movie_title,
        )

        raw = llm.complete(prompt, system=eep.SYSTEM_PROMPT, temperature=0.0)

        if prompt_logger:
            prompt_logger.log("entity_extraction", scene.scene_id, prompt, raw, llm.model_id)

        entities = parse_llm_json(raw, schema_hint="entity_list")
        if entities is None:
            logger.warning(
                "Entity extraction returned unparseable JSON for %s/%s", scene.scene_id, chunk_id
            )
            cache.set(cache_key, [])
            continue

        if not isinstance(entities, list):
            entities = [entities]

        # Validate and enrich
        valid_ents = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") not in VALID_TYPES:
                logger.debug("Skipping entity with invalid type: %s", ent.get("type"))
                continue
            _enrich_entity(ent, scene, chunk_id, movie_id)
            valid_ents.append(ent)

        cache.set(cache_key, valid_ents)
        scene_ents.extend(valid_ents)
        logger.debug(
            "Extracted %d entities from scene=%s chunk=%s",
            len(valid_ents), scene.scene_id, chunk_id,
        )

    return scene_ents


def _summarize_events(events: List[Dict]) -> List[Dict]:
    """Return slim event dicts safe to embed in entity extraction prompt."""
    return [
        {
            "temp_id": e.get("id", e.get("temp_id", "")),
            "name": e.get("name", ""),
            "description": e.get("description", ""),
            "participants": e.get("participants", []),
        }
        for e in events[:20]  # cap to avoid prompt overload
    ]


def _enrich_entity(ent: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    """Add stable IDs and provenance fields in place."""
    temp = ent.get("temp_id", "")
    ent["id"] = f"ent_{movie_id[:8]}_{scene.scene_id}_{temp or uuid.uuid4().hex[:6]}"
    ent.setdefault("scene_id", scene.scene_id)
    ent.setdefault("chunk_id", chunk_id)
    ent.setdefault("surface_forms", [ent.get("canonical_name", ent.get("name", ""))])
    ent.setdefault("canonical_name", ent.get("name", ""))
    ent.setdefault("description", "")
    ent.setdefault("evidence", [])
    ent.setdefault("linked_event_ids", [])
    ent["movie_id"] = movie_id
