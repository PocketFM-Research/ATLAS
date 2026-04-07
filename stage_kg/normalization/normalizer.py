"""
Entity normalization and disambiguation (Appendix C.2).

Full pipeline:
1. LLM-based scope/type normalization (stabilize names and types before merging).
2. Dual embeddings: name embedding + description embedding per entity.
3. Combined similarity: α·sim_name + (1-α)·sim_desc.
4. k-NN graph → Laplacian → eigengap heuristic for cluster count.
5. k-means clustering on joint [β·name; (1-β)·desc] embeddings.
6. LLM-assisted disambiguation per cluster (Figure 9 prompt format).
7. Apply merge decisions: canonical name, alias set, merged provenance.
"""

import json
import logging
import re
import unicodedata
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..llm.base import BaseLLM
from ..utils.json_repair import parse_llm_json
from ..utils.cache import Cache
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)

# Combined similarity threshold below which two nodes are never merged
# (even if they land in the same cluster, the LLM adjudicates)
SIMILARITY_FLOOR = 0.65

# Name/description weight for combined similarity
ALPHA = 0.6   # weight on name similarity
BETA  = 0.6   # weight on name embedding in joint vector

CHARACTER_TITLE_TOKENS = {
    "dr",
    "doctor",
    "mr",
    "mister",
    "mrs",
    "ms",
    "miss",
    "lt",
    "lieutenant",
    "capt",
    "captain",
    "cmdr",
    "commander",
    "adm",
    "admiral",
    "ens",
    "ensign",
}


# ------------------------------------------------------------------ entry point

def normalize_nodes(
    nodes: List[Dict],
    node_type: str,
    rename_map: Dict[str, str],
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Normalize and deduplicate a list of nodes of the same type.

    Returns:
        (merged_nodes, merge_log)
    """
    if not nodes:
        return [], []

    # Step 1: apply rename_map and string-normalize canonical names
    for node in nodes:
        _apply_rename_map(node, rename_map)

    # Step 2: merge nodes that are exactly identical after string normalization
    proto_nodes, exact_log = _exact_merge(nodes, node_type)
    merge_log = list(exact_log)

    if len(proto_nodes) <= 1:
        logger.info("[%s] %s: %d raw -> %d after exact merge",
                    movie_id, node_type, len(nodes), len(proto_nodes))
        return proto_nodes, merge_log

    # Step 3: embed names and descriptions
    embedder = _get_embedder(api_key)
    names = [_get_canonical(n) for n in proto_nodes]
    descs = [n.get("description", "") or _get_canonical(n) for n in proto_nodes]

    name_vecs = embedder(names)       # (N, D)
    desc_vecs = embedder(descs)       # (N, D)

    # Step 4: combined similarity
    from .embeddings import dual_similarity
    S = dual_similarity(name_vecs, desc_vecs, alpha=ALPHA)

    # Step 5: joint embeddings for clustering
    # β·name + (1-β)·desc, both already unit-normalized → renormalize joint
    joint = np.concatenate([BETA * name_vecs, (1 - BETA) * desc_vecs], axis=1)
    norms = np.linalg.norm(joint, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    joint = joint / norms

    # Step 6: cluster
    from .clustering import cluster_nodes
    clusters = cluster_nodes(joint, S, k_nn=min(5, len(proto_nodes) - 1))

    # Step 7: LLM adjudication per cluster
    final_nodes = list(proto_nodes)
    merged_away: set = set()

    for cluster_idxs in clusters:
        cluster_nodes_list = [proto_nodes[i] for i in cluster_idxs]

        # Skip if pairwise similarities are all below the floor
        cluster_S = S[np.ix_(cluster_idxs, cluster_idxs)]
        max_sim = cluster_S[np.triu_indices(len(cluster_idxs), k=1)].max() if len(cluster_idxs) > 1 else 0.0
        if max_sim < SIMILARITY_FLOOR:
            continue

        decision = _llm_adjudicate_cluster(
            cluster_nodes_list,
            node_type=node_type,
            llm=llm,
            movie_id=movie_id,
            movie_title=movie_title,
            cache=cache,
            prompt_logger=prompt_logger,
        )

        merge_log.append({
            "cluster": [n.get("id", _get_canonical(n)) for n in cluster_nodes_list],
            "max_similarity": float(max_sim),
            "decision": decision,
        })

        # Apply merges
        merges = decision.get("merges", [])
        for merge_group in merges:
            canonical_name = merge_group.get("canonical_name", "")
            aliases = merge_group.get("aliases", [])
            # Find which proto_nodes are mentioned in aliases or canonical
            to_merge = [
                i for i, n in enumerate(proto_nodes)
                if i in cluster_idxs
                and (
                    _get_canonical(n) == canonical_name
                    or _get_canonical(n) in aliases
                    or any(f in aliases for f in n.get("surface_forms", []))
                )
            ]
            if len(to_merge) < 2:
                continue

            # Merge them all into the first
            representative = proto_nodes[to_merge[0]]
            for j in to_merge[1:]:
                _merge_into_node(representative, proto_nodes[j])
                merged_away.add(j)
                merged_away.add(to_merge[0])  # keep, but track merged_raw_ids

            representative["canonical_name"] = canonical_name
            representative["aliases"] = list(set(
                representative.get("aliases", []) + aliases + [canonical_name]
            ))
            if not representative.get("id"):
                representative["id"] = f"node_{uuid.uuid4().hex[:12]}"

    # Collect surviving nodes by removing only merged-away secondary nodes.
    # The previous implementation attempted `merged_away - {...}` with a
    # placeholder dict expression, which crashes with `set - dict`.
    surviving_idxs = set(range(len(proto_nodes)))
    for cluster_idxs in clusters:
        cluster_nodes_list = [proto_nodes[i] for i in cluster_idxs]
        decision = next(
            (log["decision"] for log in merge_log
             if isinstance(log.get("cluster"), list)
             and set(log["cluster"]) == {n.get("id", _get_canonical(n)) for n in cluster_nodes_list}),
            {}
        )
        for merge_group in decision.get("merges", []):
            canonical_name = merge_group.get("canonical_name", "")
            aliases = merge_group.get("aliases", [])
            to_merge_idxs = [
                i for i in cluster_idxs
                if _get_canonical(proto_nodes[i]) == canonical_name
                or _get_canonical(proto_nodes[i]) in aliases
                or any(f in aliases for f in proto_nodes[i].get("surface_forms", []))
            ]
            # Remove all but the first
            for j in to_merge_idxs[1:]:
                surviving_idxs.discard(j)

    final_nodes = [proto_nodes[i] for i in sorted(surviving_idxs)]

    logger.info(
        "[%s] %s normalization: %d raw -> %d proto -> %d merged",
        movie_id, node_type, len(nodes), len(proto_nodes), len(final_nodes),
    )
    return final_nodes, merge_log


# ------------------------------------------------------------------ helpers

def _get_canonical(node: Dict) -> str:
    return node.get("canonical_name") or node.get("name", "")


def _normalize_string(s: str, node_type: str = "") -> str:
    text = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    if not text:
        return ""

    tokens = re.findall(r"[a-z0-9]+", text)
    if node_type == "Character":
        stripped = [token for token in tokens if token not in CHARACTER_TITLE_TOKENS]
        if stripped:
            tokens = stripped

    return " ".join(tokens)


def _apply_rename_map(node: Dict, rename_map: Dict[str, str]) -> None:
    canonical = _get_canonical(node)
    if canonical in rename_map:
        node["canonical_name"] = rename_map[canonical]
    else:
        normalized_rename_map = {
            _normalize_string(src, node.get("type", "")): dst
            for src, dst in rename_map.items()
        }
        normalized_canonical = _normalize_string(canonical, node.get("type", ""))
        if normalized_canonical in normalized_rename_map:
            node["canonical_name"] = normalized_rename_map[normalized_canonical]
    forms = _coerce_str_list(node.get("surface_forms", []))
    node["surface_forms"] = list(
        dict.fromkeys(rename_map.get(form, form) for form in forms)
    )


def _exact_merge(nodes: List[Dict], node_type: str) -> Tuple[List[Dict], List[Dict]]:
    """Merge nodes with identical normalized canonical names."""
    groups: Dict[str, List[Dict]] = {}
    for node in nodes:
        key = _normalize_string(_get_canonical(node), node_type)
        groups.setdefault(key, []).append(node)

    merged = []
    log = []
    for key, group in groups.items():
        base = dict(group[0])
        merged_ids = {base.get("id", "")}
        for n in group[1:]:
            _merge_into_node(base, n)
            merged_ids.add(n.get("id", ""))
            log.append({"type": "exact_merge", "key": key,
                        "merged_ids": list(merged_ids)})
        base["_merged_raw_ids"] = list(merged_ids - {""})
        if not base.get("id"):
            base["id"] = f"node_{uuid.uuid4().hex[:12]}"
        merged.append(base)

    return merged, log


def _merge_into_node(base: Dict, other: Dict) -> None:
    """Merge `other` into `base` in place."""
    # Surface forms / aliases
    forms = set(_coerce_str_list(base.get("surface_forms", []))) | set(
        _coerce_str_list(other.get("surface_forms", []))
    )
    base["surface_forms"] = list(forms)

    # Scene refs
    refs = set(_coerce_str_list(base.get("scene_refs", [base.get("scene_id", "")]))) | set(
        _coerce_str_list(other.get("scene_refs", [other.get("scene_id", "")]))
    )
    refs.discard("")
    base["scene_refs"] = list(refs)

    # Evidence
    ev = set(_coerce_str_list(base.get("evidence", []))) | set(
        _coerce_str_list(other.get("evidence", []))
    )
    base["evidence"] = list(ev)

    # Merged raw IDs (for edge resolution)
    merged_ids = set(_coerce_str_list(base.get("_merged_raw_ids", [])))
    merged_ids.update(_coerce_str_list(other.get("_merged_raw_ids", [])))
    if other.get("id"):
        merged_ids.add(other["id"])
    base["_merged_raw_ids"] = list(merged_ids)

    # Prefer longer description
    if len(other.get("description", "")) > len(base.get("description", "")):
        base["description"] = other["description"]


def _coerce_str_list(value) -> List[str]:
    """Convert mixed scalars/lists/dicts into a flat list of strings."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    results: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            extracted = None
            for key in ("text", "name", "canonical_name", "id", "scene_id"):
                candidate = item.get(key)
                if candidate:
                    extracted = candidate
                    break
            item = extracted if extracted is not None else json.dumps(item, sort_keys=True, ensure_ascii=False)
        text = str(item).strip()
        if text:
            results.append(text)

    return results


def _get_embedder(api_key: Optional[str]):
    """Return an embedder function, preferring Gemini if api_key is available."""
    from .embeddings import get_gemini_embedder, _tfidf_embedder
    if api_key:
        return get_gemini_embedder(api_key)
    return _tfidf_embedder


# ------------------------------------------------------------------ LLM adjudication

# Paper Figure 9 prompt format
_CLUSTER_ADJUDICATION_SYSTEM = (
    "You are an expert annotator for narrative knowledge graphs. "
    "Your output must be valid JSON only — no prose, no markdown fences."
)

_CLUSTER_ADJUDICATION_TEMPLATE = """\
Your task is to determine whether the following entity mentions refer to the same \
narrative entity or should remain separate.

Decision guidelines:
- Similar surface forms alone are insufficient for merging.
- Merge only if identity, narrative role, and story function are consistent.
- Do not merge disguises, substitutions, parallel versions, or different life stages.
- Mentions with explicit version or instance identifiers (e.g. numbered variants \
like "Ceti Alpha V" vs "Ceti Alpha VI", or "Model T-1" vs "Model T-2") MUST remain distinct.
- Do not merge a specific individual into a generic category; if both appear, \
prefer the individual as canonical.
- When merging, select a well-formed and narratively appropriate canonical name.

Movie: {movie_title}
Entity type: {node_type}

Input entity information:
{entity_descriptions}

Return ONLY the following JSON format:
{{
  "merges": [
    {{
      "canonical_name": "...",
      "aliases": ["..."],
      "justification": "..."
    }}
  ],
  "unmerged": [
    {{
      "name": "...",
      "justification": "..."
    }}
  ]
}}

If all entities should remain separate, return an empty merges list.
If all entities should merge, return a single entry in merges and an empty unmerged list.
"""


def _llm_adjudicate_cluster(
    cluster_nodes: List[Dict],
    node_type: str,
    llm: BaseLLM,
    movie_id: str,
    movie_title: str,
    cache: Cache,
    prompt_logger: Optional[PromptLogger],
) -> Dict:
    import json

    # Cache key based on sorted canonical names
    names_key = "|".join(sorted(_get_canonical(n) for n in cluster_nodes))
    cache_key = Cache.make_key(movie_id, "cluster_merge", node_type, names_key)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Build entity description block
    ent_descs = []
    for node in cluster_nodes:
        ent_descs.append({
            "name": _get_canonical(node),
            "aliases": node.get("surface_forms", []),
            "description": node.get("description", ""),
            "scene_refs": node.get("scene_refs", [node.get("scene_id", "")]),
            "evidence_sample": node.get("evidence", [])[:2],
        })

    prompt = _CLUSTER_ADJUDICATION_TEMPLATE.format(
        movie_title=movie_title,
        node_type=node_type,
        entity_descriptions=json.dumps(ent_descs, ensure_ascii=False, indent=2),
    )

    raw = llm.complete(
        prompt,
        system=_CLUSTER_ADJUDICATION_SYSTEM,
        temperature=0.0,
        max_tokens=1024,
    )

    if prompt_logger:
        prompt_logger.log("cluster_adjudication", names_key, prompt, raw, llm.model_id)

    decision = parse_llm_json(raw, schema_hint="cluster_adjudication")
    if not isinstance(decision, dict):
        decision = {"merges": [], "unmerged": [{"name": _get_canonical(n)} for n in cluster_nodes]}

    cache.set(cache_key, decision)
    return decision
