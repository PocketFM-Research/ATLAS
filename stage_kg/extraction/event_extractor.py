"""
Event extraction (Pass 1).

Processes scenes/chunks and extracts narratively salient events via LLM.
Each extraction is cached by (movie_id, scene_id, chunk_id).
"""

import logging
import re
import uuid
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import event_extraction as ep
from ..utils.json_repair import parse_llm_json, validate_event_list
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger
from .chunk_utils import MAX_CHUNK_CHARS, iter_extraction_chunks

logger = logging.getLogger(__name__)

def extract_events_for_movie(
    scenes: List[SceneRecord],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict[str, List[Dict]]:
    all_events: Dict[str, List[Dict]] = {}
    for scene in scenes:
        all_events[scene.scene_id] = extract_events_for_scene(
            scene, llm, movie_id, movie_title, cache, prompt_logger
        )
    total = sum(len(v) for v in all_events.values())
    logger.info("[%s] Event extraction complete: %d events across %d scenes",
                movie_id, total, len(scenes))
    return all_events


def extract_events_for_scene(
    scene: SceneRecord,
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> List[Dict]:
    scene_events: List[Dict] = []

    for chunk in scene.chunks:
        source_chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        for extraction_chunk in iter_extraction_chunks({**chunk, "id": source_chunk_id}):
            chunk_id = extraction_chunk["id"]
            cache_key = Cache.make_key(movie_id, "events", scene.scene_id, chunk_id)

            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: events for %s/%s", scene.scene_id, chunk_id)
                scene_events.extend(cached)
                continue

            chunk_text = extraction_chunk.get("content", "")
            if not chunk_text.strip():
                continue

            def extract_fn(feedback: Optional[str]):
                feedback_block = (
                    f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues): {feedback}\n"
                    if feedback else ""
                )
                p = ep.build_prompt(
                    scene_id=scene.scene_id,
                    scene_title=scene.title,
                    scene_text=chunk_text,
                    scene_summary=scene.summary,
                    chunk_id=chunk_id,
                    movie_title=movie_title,
                ) + feedback_block
                raw = llm.complete(p, system=ep.SYSTEM_PROMPT, temperature=0.0, max_tokens=12288)
                if prompt_logger:
                    prompt_logger.log("event_extraction", scene.scene_id, p, raw, llm.model_id, chunk_id=chunk_id)
                result = parse_llm_json(raw, schema_hint="event_list")
                if result is not None and not isinstance(result, list):
                    result = [result]
                return result, p

            events, _ = extract_fn(None)

            if not events:
                logger.warning("Event extraction yielded nothing for %s/%s", scene.scene_id, chunk_id)
                cache.set(cache_key, [])
                continue

            for ev in events:
                _enrich_event(ev, scene, chunk_id, movie_id)

            if not validate_event_list(events):
                logger.warning("Event list failed validation for %s/%s — keeping anyway",
                               scene.scene_id, chunk_id)

            cache.set(cache_key, events)
            scene_events.extend(events)
            logger.debug("Extracted %d events from scene=%s chunk=%s",
                         len(events), scene.scene_id, chunk_id)

    return scene_events


def _enrich_event(ev: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    temp = ev.get("temp_id", "")
    chunk_slug = _chunk_slug(chunk_id)
    ev["id"] = f"ev_{movie_id[:8]}_{scene.scene_id}_{chunk_slug}_{temp or uuid.uuid4().hex[:6]}"
    ev.setdefault("scene_id", scene.scene_id)
    ev.setdefault("chunk_id", chunk_id)
    ev.setdefault("evidence", [])
    ev.setdefault("participants", [])
    ev.setdefault("location_hint", None)
    ev.setdefault("time_hint", None)
    ev.setdefault("scope", "local")
    ev["type"] = "Event"
    ev["movie_id"] = movie_id


def _chunk_slug(chunk_id: str) -> str:
    """Create a stable, ID-safe chunk slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(chunk_id or "chunk")).strip("_")
    return slug or "chunk"
