import logging
import re
import uuid
import unicodedata
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import entity_extraction as eep
from ..utils.json_repair import parse_llm_json, validate_entity_list
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger
from .chunk_utils import MAX_CHUNK_CHARS, iter_extraction_chunks

logger = logging.getLogger(__name__)

VALID_TYPES = {"Character", "Location", "TimePoint", "Object", "Vehicle", "Concept"}


def extract_entities_for_movie(
    scenes: List[SceneRecord],
    scene_events: Dict[str, List[Dict]],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict[str, List[Dict]]:
    all_entities: Dict[str, List[Dict]] = {}
    for scene in scenes:
        all_entities[scene.scene_id] = extract_entities_for_scene(
            scene, scene_events.get(scene.scene_id, []),
            llm, movie_id, movie_title, cache, prompt_logger,
        )
    total = sum(len(v) for v in all_entities.values())
    logger.info("[%s] Entity extraction complete: %d entities across %d scenes",
                movie_id, total, len(scenes))
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
    scene_ents: List[Dict] = []

    for chunk in scene.chunks:
        source_chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        for extraction_chunk in iter_extraction_chunks({**chunk, "id": source_chunk_id}):
            chunk_id = extraction_chunk["id"]
            cache_key = Cache.make_key(movie_id, "entities", scene.scene_id, chunk_id)

            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: entities for %s/%s", scene.scene_id, chunk_id)
                scene_ents.extend(cached)
                continue

            chunk_text = extraction_chunk.get("content", "")
            if not chunk_text.strip():
                continue

            chunk_events = [e for e in events if e.get("chunk_id") == chunk_id] or events

            def extract_fn(feedback: Optional[str]):
                feedback_block = (
                    f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues): {feedback}\n"
                    if feedback else ""
                )
                p = eep.build_prompt(
                    scene_id=scene.scene_id,
                    scene_title=scene.title,
                    scene_text=chunk_text,
                    events=_summarize_events(chunk_events),
                    chunk_id=chunk_id,
                    movie_title=movie_title,
                ) + feedback_block
                raw = llm.complete(p, system=eep.SYSTEM_PROMPT, temperature=0.0, max_tokens=12288)
                if prompt_logger:
                    prompt_logger.log("entity_extraction", scene.scene_id, p, raw, llm.model_id, chunk_id=chunk_id)
                result = parse_llm_json(raw, schema_hint="entity_list")
                if result is not None and not isinstance(result, list):
                    result = [result]
                if result:
                    result = [e for e in result
                              if isinstance(e, dict) and e.get("type") in VALID_TYPES]
                return result, p

            entities, _ = extract_fn(None)

            if not entities:
                logger.warning("Entity extraction yielded nothing for %s/%s",
                               scene.scene_id, chunk_id)
                cache.set(cache_key, [])
                continue

            for ent in entities:
                _enrich_entity(ent, scene, chunk_id, movie_id)

            cache.set(cache_key, entities)
            scene_ents.extend(entities)
            logger.debug("Extracted %d entities from scene=%s chunk=%s",
                         len(entities), scene.scene_id, chunk_id)

    return scene_ents


def _summarize_events(events: List[Dict]) -> List[Dict]:
    return [
        {
            "temp_id": e.get("id", e.get("temp_id", "")),
            "name": e.get("name", ""),
            "description": e.get("description", ""),
            "participants": e.get("participants", []),
        }
        for e in events[:20]
    ]


def _enrich_entity(ent: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    temp = ent.get("temp_id", "")
    chunk_slug = _chunk_slug(chunk_id)
    ent["id"] = f"ent_{movie_id[:8]}_{scene.scene_id}_{chunk_slug}_{temp or uuid.uuid4().hex[:6]}"
    ent.setdefault("scene_id", scene.scene_id)
    ent.setdefault("chunk_id", chunk_id)
    ent.setdefault("surface_forms", [ent.get("canonical_name", ent.get("name", ""))])
    ent.setdefault("canonical_name", ent.get("name", ""))
    ent.setdefault("description", "")
    ent.setdefault("evidence", [])
    ent.setdefault("linked_event_ids", [])
    ent["surface_forms"] = _sanitize_surface_forms(ent)
    ent["movie_id"] = movie_id


def _chunk_slug(chunk_id: str) -> str:
    """Create a stable, ID-safe chunk slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(chunk_id or "chunk")).strip("_")
    return slug or "chunk"


def _sanitize_surface_forms(ent: Dict) -> List[str]:
    canonical = str(ent.get("canonical_name", ent.get("name", "")) or "").strip()
    forms = _coerce_str_list(ent.get("surface_forms", [])) or ([canonical] if canonical else [])
    forms = list(dict.fromkeys([canonical, *forms] if canonical else forms))

    if ent.get("type") not in {"Location", "Object", "Concept", "Vehicle"}:
        return forms

    evidence = _coerce_str_list(ent.get("evidence", []))
    filtered = []
    canonical_norm = _normalize_text(canonical)
    for form in forms:
        form_norm = _normalize_text(form)
        if not form_norm:
            continue
        if canonical_norm and (form_norm == canonical_norm or form_norm in canonical_norm or canonical_norm in form_norm):
            filtered.append(form)
            continue
        if any(_supports_surface_form(form_norm, snippet) for snippet in evidence):
            filtered.append(form)

    return list(dict.fromkeys(filtered or ([canonical] if canonical else forms)))


def _supports_surface_form(form_norm: str, snippet: str) -> bool:
    snippet_norm = _normalize_text(snippet)
    if not snippet_norm:
        return False
    if form_norm in snippet_norm:
        return True

    form_tokens = set(form_norm.split())
    snippet_tokens = set(snippet_norm.split())
    if not form_tokens or not snippet_tokens:
        return False
    return form_tokens <= snippet_tokens


def _normalize_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))


def _coerce_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    cleaned = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned
