"""
Canonicalization and normalization of extracted nodes.

Strategy (paper-faithful):
1. String normalization (lowercase, strip, collapse whitespace).
2. Apply dataset rename_map if available.
3. String-similarity clustering (using difflib) to find candidate duplicates.
4. LLM adjudication for ambiguous candidate pairs.
5. Merge confirmed duplicates, preserving alias lists and provenance.
"""

import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from ..llm.base import BaseLLM
from ..prompts import merge_adjudication as mp
from ..utils.json_repair import parse_llm_json
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

# Similarity threshold for triggering LLM adjudication
SIMILARITY_THRESHOLD = 0.80
# Auto-merge threshold (no LLM needed above this)
AUTO_MERGE_THRESHOLD = 0.95


def normalize_nodes(
    nodes: List[Dict],
    node_type: str,
    rename_map: Dict[str, str],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Normalize and deduplicate a list of nodes of the same type.

    Args:
        nodes: Raw extracted nodes (events or entities of one type).
        node_type: 'Event', 'Character', 'Location', etc.
        rename_map: Dataset-provided alias -> canonical name map.
        llm: LLM backend for adjudication.
        movie_id: Movie identifier.
        movie_title: Movie title for prompts.
        cache: Disk cache.
        prompt_logger: Optional prompt logger.

    Returns:
        (merged_nodes, merge_log)
        - merged_nodes: deduplicated node list
        - merge_log: list of merge decision records for audit
    """
    if not nodes:
        return [], []

    # Step 1: apply rename_map to canonical_name / name fields
    for node in nodes:
        _apply_rename_map(node, rename_map)

    # Step 2: build name -> node index
    # Group nodes by normalized canonical name first
    groups: Dict[str, List[Dict]] = {}
    for node in nodes:
        key = _normalize_string(node.get("canonical_name") or node.get("name", ""))
        groups.setdefault(key, []).append(node)

    # Step 3: merge identical-name groups automatically
    proto_nodes: List[Dict] = []
    for key, group in groups.items():
        merged = _merge_group(group)
        proto_nodes.append(merged)

    # Step 4: similarity-based clustering → LLM adjudication
    merge_log: List[Dict] = []
    final_nodes, log = _similarity_clustering(
        proto_nodes,
        node_type=node_type,
        llm=llm,
        movie_id=movie_id,
        movie_title=movie_title,
        cache=cache,
        prompt_logger=prompt_logger,
    )
    merge_log.extend(log)

    logger.info(
        "[%s] %s normalization: %d raw -> %d merged nodes",
        movie_id, node_type, len(nodes), len(final_nodes),
    )
    return final_nodes, merge_log


def _apply_rename_map(node: Dict, rename_map: Dict[str, str]) -> None:
    """Apply dataset rename_map to all surface forms and canonical name."""
    canonical = node.get("canonical_name") or node.get("name", "")
    if canonical in rename_map:
        node["canonical_name"] = rename_map[canonical]

    # Also check each surface form
    forms = node.get("surface_forms", [])
    new_forms = []
    for form in forms:
        new_forms.append(rename_map.get(form, form))
    node["surface_forms"] = list(dict.fromkeys(new_forms))  # deduplicate, preserve order


def _normalize_string(s: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _merge_group(nodes: List[Dict]) -> Dict:
    """Merge a list of nodes with the same canonical name into one."""
    if len(nodes) == 1:
        return nodes[0]

    base = dict(nodes[0])
    # Track all raw IDs that were merged into this node for edge resolution
    merged_raw_ids = set(base.get("_merged_raw_ids", []))
    if base.get("id"):
        merged_raw_ids.add(base["id"])

    for node in nodes[1:]:
        # Record this node's raw ID as merged
        if node.get("id"):
            merged_raw_ids.add(node["id"])
        merged_raw_ids.update(node.get("_merged_raw_ids", []))

        # Merge aliases
        existing_forms = set(base.get("surface_forms", []))
        for form in node.get("surface_forms", []):
            existing_forms.add(form)
        base["surface_forms"] = list(existing_forms)

        # Merge scene_refs
        existing_scenes = set(base.get("scene_refs", [base.get("scene_id", "")]))
        existing_scenes.add(node.get("scene_id", ""))
        existing_scenes.discard("")
        base["scene_refs"] = list(existing_scenes)

        # Merge evidence
        existing_ev = set(base.get("evidence", []))
        for ev in node.get("evidence", []):
            existing_ev.add(ev)
        base["evidence"] = list(existing_ev)

        # Prefer longer description
        if len(node.get("description", "")) > len(base.get("description", "")):
            base["description"] = node["description"]

    base["_merged_raw_ids"] = list(merged_raw_ids)

    # Assign a new stable canonical ID
    if not base.get("id"):
        base["id"] = f"node_{uuid.uuid4().hex[:12]}"

    return base


def _similarity_clustering(
    nodes: List[Dict],
    node_type: str,
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Cluster nodes by name similarity and adjudicate merges via LLM.

    Uses a simple O(n²) approach — acceptable for scene-level node counts.
    """
    merge_log: List[Dict] = []
    merged_into: Dict[int, int] = {}  # idx -> representative idx

    names = [_normalize_string(n.get("canonical_name") or n.get("name", "")) for n in nodes]

    for i in range(len(nodes)):
        if i in merged_into:
            continue
        for j in range(i + 1, len(nodes)):
            if j in merged_into:
                continue

            sim = _similarity(names[i], names[j])
            if sim < SIMILARITY_THRESHOLD:
                continue

            if sim >= AUTO_MERGE_THRESHOLD:
                # Auto-merge without LLM
                decision = {
                    "merge": True,
                    "canonical_name": _pick_canonical(nodes[i], nodes[j]),
                    "reason": f"Auto-merged: string similarity {sim:.2f} >= {AUTO_MERGE_THRESHOLD}",
                    "confidence": sim,
                }
            else:
                # LLM adjudication
                decision = _llm_adjudicate(
                    nodes[i], nodes[j], node_type=node_type,
                    llm=llm, movie_id=movie_id, movie_title=movie_title,
                    cache=cache, prompt_logger=prompt_logger,
                )

            merge_log.append({
                "node_a_id": nodes[i].get("id", i),
                "node_b_id": nodes[j].get("id", j),
                "similarity": sim,
                "decision": decision,
            })

            if decision.get("merge"):
                # Merge j into i
                merged = _merge_group([nodes[i], nodes[j]])
                merged["canonical_name"] = decision.get("canonical_name", merged["canonical_name"])
                nodes[i] = merged
                names[i] = _normalize_string(merged.get("canonical_name", ""))
                merged_into[j] = i

    # Collect non-merged nodes
    final = [nodes[i] for i in range(len(nodes)) if i not in merged_into]
    return final, merge_log


def _pick_canonical(a: Dict, b: Dict) -> str:
    """Pick the longer / more specific canonical name."""
    na = a.get("canonical_name") or a.get("name", "")
    nb = b.get("canonical_name") or b.get("name", "")
    return na if len(na) >= len(nb) else nb


def _llm_adjudicate(
    node_a: Dict,
    node_b: Dict,
    node_type: str,
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
) -> Dict:
    """Ask LLM to decide whether two nodes should be merged."""
    import json

    key_a = node_a.get("id", node_a.get("canonical_name", ""))
    key_b = node_b.get("id", node_b.get("canonical_name", ""))
    cache_key = Cache.make_key(movie_id, "merge", node_type, key_a, key_b)

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if node_type == "Event":
        prompt = mp.build_event_merge_prompt(node_a, node_b, movie_title)
    else:
        prompt = mp.build_entity_merge_prompt(node_a, node_b, movie_title)

    raw = llm.complete(prompt, system=mp.SYSTEM_PROMPT, temperature=0.0, max_tokens=512)

    if prompt_logger:
        prompt_logger.log("merge_adjudication", f"{key_a}|{key_b}", prompt, raw, llm.model_id)

    decision = parse_llm_json(raw, schema_hint="merge_decision")
    if not isinstance(decision, dict):
        decision = {"merge": False, "reason": "parse failure", "confidence": 0.0}

    cache.set(cache_key, decision)
    return decision
