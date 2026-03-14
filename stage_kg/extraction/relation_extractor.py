"""
Relation extraction (Pass 3).

Extracts typed edges between entities and events using schema constraints.
"""

import logging
import uuid
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import relation_extraction as rp
from ..schema import is_valid_triple, NodeType, RelationType
from ..utils.json_repair import parse_llm_json, validate_relation_list
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000


def extract_relations_for_movie(
    scenes: List[SceneRecord],
    scene_events: Dict[str, List[Dict]],
    scene_entities: Dict[str, List[Dict]],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict[str, List[Dict]]:
    """
    Run relation extraction over all scenes.

    Returns:
        Dict mapping scene_id -> list of raw relation dicts.
    """
    all_relations: Dict[str, List[Dict]] = {}

    for scene in scenes:
        events = scene_events.get(scene.scene_id, [])
        entities = scene_entities.get(scene.scene_id, [])

        if not events and not entities:
            all_relations[scene.scene_id] = []
            continue

        scene_rels = extract_relations_for_scene(
            scene=scene,
            events=events,
            entities=entities,
            llm=llm,
            movie_id=movie_id,
            movie_title=movie_title,
            cache=cache,
            prompt_logger=prompt_logger,
        )
        all_relations[scene.scene_id] = scene_rels

    total = sum(len(v) for v in all_relations.values())
    logger.info(
        "[%s] Relation extraction complete: %d relations across %d scenes",
        movie_id, total, len(scenes),
    )
    return all_relations


def extract_relations_for_scene(
    scene: SceneRecord,
    events: List[Dict],
    entities: List[Dict],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> List[Dict]:
    """Extract relations from a single scene."""
    scene_rels: List[Dict] = []

    for chunk in scene.chunks:
        chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        cache_key = Cache.make_key(movie_id, "relations", scene.scene_id, chunk_id)

        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: relations for %s/%s", scene.scene_id, chunk_id)
            scene_rels.extend(cached)
            continue

        chunk_text = chunk.get("content", "")
        if not chunk_text.strip():
            continue
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]

        # Subset to chunk-relevant events and entities
        chunk_events = [e for e in events if e.get("chunk_id") == chunk_id] or events
        chunk_entities = [e for e in entities if e.get("chunk_id") == chunk_id] or entities

        prompt = rp.build_prompt(
            scene_id=scene.scene_id,
            scene_title=scene.title,
            scene_text=chunk_text,
            events=_slim_events(chunk_events),
            entities=_slim_entities(chunk_entities),
            chunk_id=chunk_id,
            movie_title=movie_title,
        )

        raw = llm.complete(prompt, system=rp.SYSTEM_PROMPT, temperature=0.0)

        if prompt_logger:
            prompt_logger.log("relation_extraction", scene.scene_id, prompt, raw, llm.model_id)

        relations = parse_llm_json(raw, schema_hint="relation_list")
        if relations is None:
            logger.warning(
                "Relation extraction returned unparseable JSON for %s/%s", scene.scene_id, chunk_id
            )
            cache.set(cache_key, [])
            continue

        if not isinstance(relations, list):
            relations = [relations]

        # Validate schema and enrich
        valid_rels = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            if not _validate_schema(rel):
                logger.debug(
                    "Schema violation: (%s, %s, %s) — skipping",
                    rel.get("source_type"), rel.get("relation"), rel.get("target_type"),
                )
                continue
            _enrich_relation(rel, scene, chunk_id, movie_id)
            valid_rels.append(rel)

        cache.set(cache_key, valid_rels)
        scene_rels.extend(valid_rels)
        logger.debug(
            "Extracted %d valid relations from scene=%s chunk=%s",
            len(valid_rels), scene.scene_id, chunk_id,
        )

    return scene_rels


def _validate_schema(rel: Dict) -> bool:
    """Check that (source_type, relation, target_type) is schema-valid."""
    try:
        src_type = NodeType(rel.get("source_type", ""))
        relation = RelationType(rel.get("relation", ""))
        tgt_type = NodeType(rel.get("target_type", ""))
        return is_valid_triple(src_type, relation, tgt_type)
    except ValueError:
        return False


def _slim_events(events: List[Dict]) -> List[Dict]:
    return [
        {"temp_id": e.get("id", e.get("temp_id", "")), "name": e.get("name", "")}
        for e in events[:20]
    ]


def _slim_entities(entities: List[Dict]) -> List[Dict]:
    return [
        {
            "temp_id": e.get("id", e.get("temp_id", "")),
            "canonical_name": e.get("canonical_name", e.get("name", "")),
            "type": e.get("type", ""),
        }
        for e in entities[:30]
    ]


def _enrich_relation(rel: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    """Add stable IDs and provenance fields in place."""
    temp = rel.get("temp_id", "")
    rel["id"] = f"rel_{movie_id[:8]}_{scene.scene_id}_{temp or uuid.uuid4().hex[:6]}"
    rel.setdefault("scene_id", scene.scene_id)
    rel.setdefault("chunk_id", chunk_id)
    rel.setdefault("evidence", [])
    rel.setdefault("confidence", 0.8)
    rel["movie_id"] = movie_id
