"""
Relation extraction (Pass 3).
Each extraction is cached by (movie_id, scene_id, chunk_id).
"""

import logging
import re
import uuid
from typing import List, Dict, Optional

from ..llm.base import BaseLLM
from ..ingest.loader import SceneRecord
from ..prompts import relation_extraction as rp
from ..schema import is_valid_triple, NodeType, RelationType
from ..utils.json_repair import parse_llm_json
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger
from .chunk_utils import MAX_CHUNK_CHARS, iter_extraction_chunks

logger = logging.getLogger(__name__)

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
        source_chunk_id = chunk.get("id", f"{scene.scene_id}_chunk_0")
        for extraction_chunk in iter_extraction_chunks({**chunk, "id": source_chunk_id}):
            chunk_id = extraction_chunk["id"]
            cache_key = Cache.make_key(movie_id, "relations", scene.scene_id, chunk_id)

            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: relations for %s/%s", scene.scene_id, chunk_id)
                scene_rels.extend(cached)
                continue

            chunk_text = extraction_chunk.get("content", "")
            if not chunk_text.strip():
                continue

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
                raw = llm.complete(p, system=rp.SYSTEM_PROMPT, temperature=0.0, max_tokens=16384)
                if prompt_logger:
                    prompt_logger.log("relation_extraction", scene.scene_id, p, raw, llm.model_id)
                result = parse_llm_json(raw, schema_hint="relation_list")
                if result is not None and not isinstance(result, list):
                    result = [result]
                if result:
                    result = [r for r in result if isinstance(r, dict)]
                return result, p

            relations, _ = extract_fn(None)

            if not relations:
                logger.warning("Relation extraction yielded nothing for %s/%s — check prompt logs",
                               scene.scene_id, chunk_id)
                cache.set(cache_key, [])
                continue

            relations = _postprocess_relations(relations, chunk_events, chunk_entities)
            relations = _ensure_event_coverage(relations, chunk_events, chunk_entities)
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
    rel = dict(rel)
    raw_relation = str(rel.get("relation", "")).strip().lower()
    if raw_relation == "before":
        rel["relation"] = RelationType.PRECEDES.value
    elif raw_relation == "after":
        rel["relation"] = RelationType.PRECEDES.value
        rel["source_id"], rel["target_id"] = rel.get("target_id"), rel.get("source_id")
        rel["source_type"], rel["target_type"] = rel.get("target_type"), rel.get("source_type")
    elif raw_relation == "caused_by":
        rel["relation"] = RelationType.CAUSES.value
        rel["source_id"], rel["target_id"] = rel.get("target_id"), rel.get("source_id")
        rel["source_type"], rel["target_type"] = rel.get("target_type"), rel.get("source_type")

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


def _postprocess_relations(
    relations: List[Dict],
    events: List[Dict],
    entities: List[Dict],
) -> List[Dict]:
    """Repair, validate, and lightly sanity-check extracted relations."""
    event_index = _build_node_index(events)
    entity_index = _build_node_index(entities)

    cleaned: List[Dict] = []
    seen = set()
    for relation in relations:
        repaired = _repair_relation(relation)
        if repaired is None or not _validate_schema(repaired):
            continue

        repaired["evidence"] = _normalize_evidence_list(repaired.get("evidence", []))
        if not repaired["evidence"]:
            continue

        if not _passes_basic_relation_checks(repaired, event_index, entity_index):
            continue

        key = (repaired.get("source_id"), repaired.get("relation"), repaired.get("target_id"))
        if key in seen:
            continue
        seen.add(key)

        # Stamp source/target names for fallback ID resolution in graph builder
        combined_index = {**event_index, **entity_index}
        src_node = combined_index.get(repaired.get("source_id"), {})
        tgt_node = combined_index.get(repaired.get("target_id"), {})
        repaired["source_name"] = src_node.get("canonical_name") or src_node.get("name", "")
        repaired["target_name"] = tgt_node.get("canonical_name") or tgt_node.get("name", "")

        cleaned.append(repaired)

    return cleaned


def _ensure_event_coverage(
    relations: List[Dict],
    events: List[Dict],
    entities: List[Dict],
) -> List[Dict]:
    """Synthesize performs edges for events that have no event-role relation."""
    event_role_types = {
        RelationType.PERFORMS.value,
        RelationType.UNDERGOES.value,
        RelationType.EXPERIENCES.value,
    }
    covered_event_ids = set()
    for rel in relations:
        if rel.get("relation") in event_role_types:
            covered_event_ids.add(rel.get("target_id"))

    entity_index = _build_node_index(entities)
    new_relations = list(relations)

    for ev in events:
        ev_id = ev.get("id") or ev.get("temp_id")
        if not ev_id or ev_id in covered_event_ids:
            continue

        # Try to match a participant to an entity
        participants = ev.get("participants", [])
        matched_entity = None
        for participant in participants:
            norm_p = _normalize_name(participant)
            for ent in entities:
                if ent.get("type") not in ("Character", "Object", "Vehicle"):
                    continue
                ent_name = _normalize_name(
                    ent.get("canonical_name", "") or ent.get("name", "")
                )
                if not ent_name:
                    continue
                # Check token overlap
                p_tokens = set(norm_p.split())
                e_tokens = set(ent_name.split())
                if p_tokens and e_tokens and (p_tokens <= e_tokens or e_tokens <= p_tokens):
                    matched_entity = ent
                    break
            if matched_entity:
                break

        if not matched_entity:
            continue

        ent_id = matched_entity.get("id") or matched_entity.get("temp_id")
        evidence = ev.get("evidence", []) or [ev.get("description", "")]
        new_relations.append({
            "temp_id": f"synth_{ev_id}_{ent_id}",
            "source_id": ent_id,
            "source_type": matched_entity.get("type", "Character"),
            "relation": RelationType.PERFORMS.value,
            "target_id": ev_id,
            "target_type": "Event",
            "evidence": evidence[:1] if evidence else [],
            "confidence": 0.6,
            "source_name": matched_entity.get("canonical_name") or matched_entity.get("name", ""),
            "target_name": ev.get("name", ""),
        })
        logger.debug("Synthesized performs edge: %s -> %s", ent_id, ev_id)

    return new_relations


def _build_node_index(nodes: List[Dict]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for node in nodes:
        for key in (node.get("id"), node.get("temp_id")):
            if key:
                index[key] = node
    return index


def _normalize_evidence_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    cleaned = []
    seen = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _passes_basic_relation_checks(
    relation: Dict,
    event_index: Dict[str, Dict],
    entity_index: Dict[str, Dict],
) -> bool:
    rel_type = relation.get("relation")
    if rel_type not in {
        RelationType.PERFORMS.value,
        RelationType.UNDERGOES.value,
        RelationType.EXPERIENCES.value,
    }:
        return True

    event = event_index.get(relation.get("target_id"))
    entity = entity_index.get(relation.get("source_id"))
    if not event or not entity:
        return True

    linked_ids = {str(value) for value in entity.get("linked_event_ids", []) if value}
    direct_link = event.get("id") in linked_ids or event.get("temp_id") in linked_ids

    participants = {
        _normalize_name(participant)
        for participant in event.get("participants", [])
        if _normalize_name(participant)
    }
    entity_forms = {
        _normalize_name(entity.get("canonical_name", "") or entity.get("name", ""))
    }
    entity_forms.update(
        _normalize_name(form) for form in entity.get("surface_forms", []) if _normalize_name(form)
    )
    participant_match = any(_participant_matches_entity(participant, entity_forms) for participant in participants)

    if linked_ids:
        return direct_link or participant_match
    # If only participants exist (no linked_event_ids), allow relation through —
    # participant lists from event extraction are often incomplete.
    return True


def _participant_matches_entity(participant: str, entity_forms: set) -> bool:
    participant_tokens = set(participant.split())
    if not participant_tokens:
        return False
    for entity_form in entity_forms:
        if participant == entity_form:
            return True
        entity_tokens = set(entity_form.split())
        if participant_tokens <= entity_tokens or entity_tokens <= participant_tokens:
            return True
    return False


def _normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _slim_events(events: List[Dict]) -> List[Dict]:
    return [
        {"temp_id": e.get("id", e.get("temp_id", "")), "name": e.get("name", "")}
        for e in events[:40]
    ]


def _slim_entities(entities: List[Dict]) -> List[Dict]:
    return [
        {
            "temp_id": e.get("id", e.get("temp_id", "")),
            "canonical_name": e.get("canonical_name", e.get("name", "")),
            "type": e.get("type", ""),
        }
        for e in entities[:60]
    ]


def _enrich_relation(rel: Dict, scene: SceneRecord, chunk_id: str, movie_id: str) -> None:
    temp = rel.get("temp_id", "")
    chunk_slug = _chunk_slug(chunk_id)
    rel["id"] = f"rel_{movie_id[:8]}_{scene.scene_id}_{chunk_slug}_{temp or uuid.uuid4().hex[:6]}"
    rel.setdefault("scene_id", scene.scene_id)
    rel.setdefault("chunk_id", chunk_id)
    rel.setdefault("evidence", [])
    rel.setdefault("confidence", 0.8)
    rel["movie_id"] = movie_id


def _chunk_slug(chunk_id: str) -> str:
    """Create a stable, ID-safe chunk slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(chunk_id or "chunk")).strip("_")
    return slug or "chunk"
