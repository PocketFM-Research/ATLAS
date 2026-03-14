"""
Event extraction (Pass 1).

Processes scenes/chunks and extracts narratively salient events via LLM.
Results are cached by (movie_id, scene_id, chunk_id).
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import event_extraction as ep
from ..utils.json_repair import parse_llm_json, validate_event_list
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

# Maximum characters per chunk sent to LLM
MAX_CHUNK_CHARS = 3000


def extract_events_for_movie(
    scenes: List[SceneRecord],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict[str, List[Dict]]:
    """
    Run event extraction over all scenes of a movie.

    Returns:
        Dict mapping scene_id -> list of raw event dicts.
    """
    all_events: Dict[str, List[Dict]] = {}

    for scene in scenes:
        scene_events = extract_events_for_scene(
            scene=scene,
            llm=llm,
            movie_id=movie_id,
            movie_title=movie_title,
            cache=cache,
            prompt_logger=prompt_logger,
        )
        all_events[scene.scene_id] = scene_events

    total = sum(len(v) for v in all_events.values())
    logger.info(
        "[%s] Event extraction complete: %d events across %d scenes",
        movie_id, total, len(scenes),
    )
    return all_events


def extract_events_for_scene(
    scene: SceneRecord,
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> List[Dict]:
    """Extract events from a single scene (may span multiple chunks)."""
    scene_events: List[Dict] = []

    for chunk in scene.chunks:
        chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        cache_key = Cache.make_key(movie_id, "events", scene.scene_id, chunk_id)

        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: events for %s/%s", scene.scene_id, chunk_id)
            scene_events.extend(cached)
            continue

        chunk_text = chunk.get("content", "")
        if not chunk_text.strip():
            continue

        # Truncate very long chunks
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]
            logger.debug("Truncated chunk %s to %d chars", chunk_id, MAX_CHUNK_CHARS)

        prompt = ep.build_prompt(
            scene_id=scene.scene_id,
            scene_title=scene.title,
            scene_text=chunk_text,
            scene_summary=scene.summary,
            chunk_id=chunk_id,
            movie_title=movie_title,
        )

        raw = llm.complete(prompt, system=ep.SYSTEM_PROMPT, temperature=0.0)

        if prompt_logger:
            prompt_logger.log("event_extraction", scene.scene_id, prompt, raw, llm.model_id)

        events = parse_llm_json(raw, schema_hint="event_list")
        if events is None:
            logger.warning(
                "Event extraction returned unparseable JSON for %s/%s", scene.scene_id, chunk_id
            )
            cache.set(cache_key, [])
            continue

        if not isinstance(events, list):
            events = [events]

        # Enrich with stable IDs and provenance
        for ev in events:
            _enrich_event(ev, scene, chunk_id, movie_id)

        if not validate_event_list(events):
            logger.warning(
                "Event list failed validation for %s/%s — keeping anyway", scene.scene_id, chunk_id
            )

        cache.set(cache_key, events)
        scene_events.extend(events)
        logger.debug(
            "Extracted %d events from scene=%s chunk=%s", len(events), scene.scene_id, chunk_id
        )

    return scene_events


def _enrich_event(ev: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    """Add stable IDs and provenance fields to an event dict in place."""
    # Assign stable ID if temp_id is missing or needs globalization
    temp = ev.get("temp_id", "")
    ev["id"] = f"ev_{movie_id[:8]}_{scene.scene_id}_{temp or uuid.uuid4().hex[:6]}"
    ev.setdefault("scene_id", scene.scene_id)
    ev.setdefault("chunk_id", chunk_id)
    ev.setdefault("evidence", [])
    ev.setdefault("participants", [])
    ev.setdefault("location_hint", None)
    ev.setdefault("time_hint", None)
    ev.setdefault("scope", "local")
    ev["type"] = "Event"
    ev["movie_id"] = movie_id
