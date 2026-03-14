"""
Relation extraction (Pass 3) with reflection-based QC (Appendix C.3).
"""

import logging
import uuid
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import relation_extraction as rp
from ..prompts.reflection import build_relation_reflection_prompt
from ..schema import is_valid_triple, NodeType, RelationType
from ..utils.json_repair import parse_llm_json
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger
from .reflection import reflection_loop

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
    all_relations: Dict[str, List[Dict]] = {}
    for scene in scenes:
        events = scene_events.get(scene.scene_id, [])
        entities = scene_entities.get(scene.scene_id, [])
        if not events and not entities:
            all_relations[scene.scene_id] = []
            continue
        all_relations[scene.scene_id] = extract_relations_for_scene(
            scene, events, entities, llm, movie_id, movie_title, cache, prompt_logger
        )
    total = sum(len(v) for v in all_relations.values())
    logger.info("[%s] Relation extraction complete: %d relations across %d scenes",
                movie_id, total, len(scenes))
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

        chunk_events = [e for e in events if e.get("chunk_id") == chunk_id] or events
        chunk_entities = [e for e in entities if e.get("chunk_id") == chunk_id] or entities

        def extract_fn(feedback: Optional[str]):
            feedback_block = (
                f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues): {feedback}\n"
                if feedback else ""
            )
            p = rp.build_prompt(
                scene_id=scene.scene_id,
                scene_title=scene.title,
                scene_text=chunk_text,
                events=_slim_events(chunk_events),
                entities=_slim_entities(chunk_entities),
                chunk_id=chunk_id,
                movie_title=movie_title,
            ) + feedback_block
            raw = llm.complete(p, system=rp.SYSTEM_PROMPT, temperature=0.0, max_tokens=8192)
            if prompt_logger:
                prompt_logger.log("relation_extraction", scene.scene_id, p, raw, llm.model_id)
            result = parse_llm_json(raw, schema_hint="relation_list")
            if result is not None and not isinstance(result, list):
                result = [result]
            # Schema filter — reject invalid triples immediately
            if result:
                result = [r for r in result
                          if isinstance(r, dict) and _validate_schema(r)]
                if len(result) < len(result or []):
                    logger.debug("Schema filter removed %d invalid relations", 0)
            return result, p

        def reflect_fn(relations):
            return build_relation_reflection_prompt(
                chunk_text, relations or [], chunk_events, chunk_entities
            )

        relations = reflection_loop(
            extract_fn=extract_fn,
            reflect_fn=reflect_fn,
            llm=llm,
            scene_id=f"{scene.scene_id}/{chunk_id}",
            prompt_logger=prompt_logger,
        )

        if not relations:
            logger.warning("Relation extraction yielded nothing for %s/%s — check prompt logs",
                           scene.scene_id, chunk_id)
            cache.set(cache_key, [])
            continue

        # Schema repair: if endpoint types uniquely determine a valid relation, rewrite it
        relations = [_repair_relation(r) for r in relations]
        relations = [r for r in relations if r is not None]

        for rel in relations:
            _enrich_relation(rel, scene, chunk_id, movie_id)

        cache.set(cache_key, relations)
        scene_rels.extend(relations)
        logger.debug("Extracted %d valid relations from scene=%s chunk=%s",
                     len(relations), scene.scene_id, chunk_id)

    return scene_rels


# ------------------------------------------------------------------ schema helpers

def _validate_schema(rel: Dict) -> bool:
    try:
        st = NodeType(rel.get("source_type", ""))
        rt = RelationType(rel.get("relation", ""))
        tt = NodeType(rel.get("target_type", ""))
        return is_valid_triple(st, rt, tt)
    except ValueError:
        return False


def _repair_relation(rel: Dict) -> Optional[Dict]:
    """
    Schema-constrained type repair (Appendix C.3, Table 12).

    If a relation violates the schema but the endpoint types uniquely
    determine a valid relation type, rewrite it.  If multiple candidates
    exist or none is valid, discard the relation.
    """
    if _validate_schema(rel):
        return rel  # already valid

    try:
        src_type = NodeType(rel.get("source_type", ""))
        tgt_type = NodeType(rel.get("target_type", ""))
    except ValueError:
        return None  # unknown node type — discard

    from ..schema import get_valid_relations_for
    candidates = get_valid_relations_for(src_type, tgt_type)

    if len(candidates) == 1:
        repaired = candidates.pop()
        logger.debug(
            "Schema repair: (%s, %s, %s) -> %s",
            src_type, rel.get("relation"), tgt_type, repaired,
        )
        rel = dict(rel)
        rel["relation"] = repaired.value
        return rel

    # Multiple candidates or none — discard
    logger.debug(
        "Schema violation discarded: (%s, %s, %s) — %d candidates",
        src_type, rel.get("relation"), tgt_type, len(candidates),
    )
    return None


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
    temp = rel.get("temp_id", "")
    rel["id"] = f"rel_{movie_id[:8]}_{scene.scene_id}_{temp or uuid.uuid4().hex[:6]}"
    rel.setdefault("scene_id", scene.scene_id)
    rel.setdefault("chunk_id", chunk_id)
    rel.setdefault("evidence", [])
    rel.setdefault("confidence", 0.8)
    rel["movie_id"] = movie_id
