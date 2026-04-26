"""
Movie-level knowledge graph builder.

Consolidates scene-level extraction outputs into a single coherent graph:
- Deduplicates nodes across scenes.
- Links cross-scene node references via canonical IDs.
- Resolves edge source/target IDs to final canonical node IDs.
- Assigns a globally unique node ID to each merged canonical node.
"""

import json
import logging
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..schema import NodeType, RelationType, is_valid_triple

logger = logging.getLogger(__name__)

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
GENERIC_CHARACTER_ALIASES = {
    "voice",
    "captain",
    "mister",
    "sir",
    "crew",
    "guy",
    "he",
    "she",
    "they",
    "person"
}


class KnowledgeGraph:
    """In-memory movie-level knowledge graph."""

    def __init__(self, movie_id: str, title: str = ""):
        self.movie_id = movie_id
        self.title = title
        self.nodes: Dict[str, Dict] = {}   # canonical_id -> node dict
        self.edges: List[Dict] = []

        # Lookup indexes
        self._name_to_id: Dict[str, str] = {}  # normalized_name -> canonical_id
        self._temp_to_canonical: Dict[str, str] = {}  # unambiguous raw extraction id -> canonical_id
        self._chunked_temp_to_canonical: Dict[str, str] = {}  # chunk-scoped raw id -> canonical_id
        self._ambiguous_temp_ids = set()

    # ------------------------------------------------------------------ nodes

    def add_node(self, node: Dict) -> str:
        """
        Add or merge a node into the graph.

        Returns the canonical node ID.
        """
        node_type = node.get("type", "")
        canonical_name = node.get("canonical_name") or node.get("name", "")
        norm_name = _norm(canonical_name, node_type)

        # Lookup key is (type, normalized_name)
        lookup_key = f"{node_type}::{norm_name}"

        if lookup_key in self._name_to_id:
            canonical_id = self._name_to_id[lookup_key]
            existing = self.nodes[canonical_id]
            _merge_into(existing, node)
        else:
            proposed_id = node.get("id") or ""
            if (
                proposed_id
                and proposed_id in self.nodes
                and self._can_reuse_existing_id(self.nodes[proposed_id], node)
            ):
                # ID already exists under a different name (e.g. name changed during
                # normalization).  Merge into the existing node instead of creating
                # a duplicate with an auto-generated n_ ID.
                canonical_id = proposed_id
                _merge_into(self.nodes[canonical_id], node)
                # Index the new name as well so future lookups by this name work.
                self._name_to_id[lookup_key] = canonical_id
            else:
                canonical_id = proposed_id or f"n_{uuid.uuid4().hex[:12]}"
                # If even the generated ID collides (extremely unlikely), re-generate.
                while canonical_id in self.nodes:
                    canonical_id = f"n_{uuid.uuid4().hex[:12]}"

                new_node = {
                    "id": canonical_id,
                    "type": node_type,
                    "name": canonical_name,
                    "aliases": _clean_aliases(
                        node_type,
                        canonical_name,
                        node.get("surface_forms", [canonical_name]),
                    ),
                    "description": node.get("description", ""),
                    "scene_refs": _as_list(node.get("scene_refs") or node.get("scene_id")),
                    "evidence": _coerce_str_list(node.get("evidence", [])),
                }
                self.nodes[canonical_id] = new_node
                self._name_to_id[lookup_key] = canonical_id

        # Map raw temp/extraction id to canonical_id
        self._register_raw_ids(node, canonical_id)

        return canonical_id

    def resolve_id(self, raw_id: str, chunk_id: Optional[str] = None) -> Optional[str]:
        """Map a raw extraction temp_id or scene-level id to a canonical node id."""
        if chunk_id:
            chunk_key = f"{chunk_id}::{raw_id}"
            if chunk_key in self._chunked_temp_to_canonical:
                return self._chunked_temp_to_canonical[chunk_key]
        return self._temp_to_canonical.get(raw_id)

    def register_or_merge_raw_node(self, node: Dict) -> str:
        """Attach a raw node to an existing canonical node when possible."""
        for raw_id in (node.get("id"), node.get("temp_id")):
            canonical_id = self.resolve_id(raw_id, node.get("chunk_id")) if raw_id else None
            if canonical_id and canonical_id in self.nodes:
                _merge_into(self.nodes[canonical_id], node)
                self._register_raw_ids(node, canonical_id)
                return canonical_id

        canonical_name = node.get("canonical_name") or node.get("name", "")
        lookup_key = f"{node.get('type', '')}::{_norm(canonical_name, node.get('type', ''))}"
        if lookup_key in self._name_to_id:
            canonical_id = self._name_to_id[lookup_key]
            _merge_into(self.nodes[canonical_id], node)
            self._register_raw_ids(node, canonical_id)
            return canonical_id

        return self.add_node(node)

    def _can_reuse_existing_id(self, existing: Dict, incoming: Dict) -> bool:
        """Reuse an existing node id only when it clearly refers to the same node."""
        if existing.get("type") != incoming.get("type", ""):
            return False

        existing_norm = _norm(existing.get("name", ""), existing.get("type", ""))
        incoming_name = incoming.get("canonical_name") or incoming.get("name", "")
        incoming_norm = _norm(incoming_name, incoming.get("type", ""))
        return bool(existing_norm and incoming_norm and existing_norm == incoming_norm)

    def _register_raw_ids(self, node: Dict, canonical_id: str) -> None:
        chunk_id = node.get("chunk_id")
        for raw_id in [node.get("id"), node.get("temp_id")]:
            self._register_raw_reference(raw_id, canonical_id, chunk_id)

    def _register_raw_reference(
        self,
        raw_id: Optional[str],
        canonical_id: str,
        chunk_id: Optional[str] = None,
    ) -> None:
        if not raw_id:
            return

        raw_id = str(raw_id)
        if chunk_id:
            self._chunked_temp_to_canonical[f"{chunk_id}::{raw_id}"] = canonical_id

        existing = self._temp_to_canonical.get(raw_id)
        if existing and existing != canonical_id:
            self._ambiguous_temp_ids.add(raw_id)
            self._temp_to_canonical.pop(raw_id, None)
            return

        if raw_id not in self._ambiguous_temp_ids:
            self._temp_to_canonical[raw_id] = canonical_id

    # ------------------------------------------------------------------ edges

    def add_edge(self, rel: Dict) -> None:
        """
        Add an edge, resolving source/target IDs to canonical node IDs.

        Skips edges with unresolvable endpoints or schema violations.
        """
        src_raw = rel.get("source_id")
        tgt_raw = rel.get("target_id")

        src_id = self.resolve_id(src_raw, rel.get("chunk_id")) if src_raw else None
        tgt_id = self.resolve_id(tgt_raw, rel.get("chunk_id")) if tgt_raw else None

        # Name-based fallback when ID resolution fails
        if (not src_id or src_id not in self.nodes) and rel.get("source_name") and rel.get("source_type"):
            lookup = f"{rel['source_type']}::{_norm(rel['source_name'], rel['source_type'])}"
            src_id = self._name_to_id.get(lookup)
        if (not tgt_id or tgt_id not in self.nodes) and rel.get("target_name") and rel.get("target_type"):
            lookup = f"{rel['target_type']}::{_norm(rel['target_name'], rel['target_type'])}"
            tgt_id = self._name_to_id.get(lookup)

        if not src_id or src_id not in self.nodes:
            logger.warning("Edge skipped: unresolved source %s (name=%s)", src_raw, rel.get("source_name", ""))
            return
        if not tgt_id or tgt_id not in self.nodes:
            logger.warning("Edge skipped: unresolved target %s (name=%s)", tgt_raw, rel.get("target_name", ""))
            return
        if src_id == tgt_id:
            logger.debug(
                "Edge skipped: self-reference (%s, %s, %s)",
                src_id,
                rel.get("relation", ""),
                tgt_id,
            )
            return

        # Schema check
        try:
            src_type = NodeType(self.nodes[src_id]["type"])
            tgt_type = NodeType(self.nodes[tgt_id]["type"])
            rel_type = RelationType(rel["relation"])
            if not is_valid_triple(src_type, rel_type, tgt_type):
                logger.debug(
                    "Edge skipped: schema violation (%s, %s, %s)",
                    src_type, rel_type, tgt_type,
                )
                return
        except (ValueError, KeyError):
            logger.debug("Edge skipped: unknown type/relation: %s", rel)
            return

        # Deduplication: same (src, relation, tgt)
        key = (src_id, rel["relation"], tgt_id)
        for existing in self.edges:
            if (existing["source"], existing["relation"], existing["target"]) == key:
                # Merge evidence
                existing["evidence"] = list(
                    set(_coerce_str_list(existing.get("evidence", [])))
                    | set(_coerce_str_list(rel.get("evidence", [])))
                )
                existing["scene_refs"] = list(
                    set(existing.get("scene_refs", [])) | _as_set(rel.get("scene_id"))
                )
                existing["confidence"] = max(
                    existing.get("confidence", 0.0), rel.get("confidence", 0.0)
                )
                return

        self.edges.append({
            "id": f"e_{uuid.uuid4().hex[:12]}",
            "source": src_id,
            "relation": rel["relation"],
            "target": tgt_id,
            "scene_refs": _as_list(rel.get("scene_id")),
            "evidence": rel.get("evidence", []),
            "confidence": rel.get("confidence", 0.8),
        })

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> Dict:
        """Serialize graph to the STAGE paper output format."""
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved graph to %s (%d nodes, %d edges)", path, len(self.nodes), len(self.edges))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = cls(movie_id=data["movie_id"], title=data.get("title", ""))
        for node in data.get("nodes", []):
            g.nodes[node["id"]] = node
            norm_key = f"{node['type']}::{_norm(node['name'], node['type'])}"
            g._name_to_id[norm_key] = node["id"]
        g.edges = data.get("edges", [])
        return g


# -------------------------------------------------------------------- helpers

def build_graph_from_extractions(
    movie_id: str,
    title: str,
    all_events: Dict[str, List[Dict]],
    all_entities: Dict[str, List[Dict]],
    all_relations: Dict[str, List[Dict]],
    normalized_events: List[Dict],
    normalized_entities: Dict[str, List[Dict]],
) -> KnowledgeGraph:
    """
    Build a movie-level KG from extraction outputs.

    Normalized nodes take precedence over raw extractions.
    All raw extraction IDs (including those merged during normalization)
    are registered in _temp_to_canonical so edges resolve correctly.
    """
    g = KnowledgeGraph(movie_id=movie_id, title=title)

    # Step 1: Add normalized events — these are the canonical nodes
    for ev in normalized_events:
        ev_node = {**ev, "type": "Event"}
        canonical_id = g.add_node(ev_node)
        # Register any raw IDs that were merged into this node
        for raw_id in ev_node.get("_merged_raw_ids", []):
            g._register_raw_reference(raw_id, canonical_id)

    # Step 2: Add normalized entities by type
    for node_type, nodes in normalized_entities.items():
        for ent in nodes:
            canonical_id = g.add_node(ent)
            for raw_id in ent.get("_merged_raw_ids", []):
                g._register_raw_reference(raw_id, canonical_id)

    # Step 3: Register raw extraction ids and only add genuinely unmapped nodes.
    for scene_id, events in all_events.items():
        for ev in events:
            ev_node = {**ev, "type": "Event"}
            g.register_or_merge_raw_node(ev_node)

    for scene_id, entities in all_entities.items():
        for ent in entities:
            g.register_or_merge_raw_node(ent)

    # Step 4: Add edges — resolve source/target IDs via _temp_to_canonical
    for scene_id, relations in all_relations.items():
        for rel in relations:
            g.add_edge(rel)

    _repair_graph(g)

    logger.info(
        "Built graph for '%s': %d nodes, %d edges", title or movie_id, len(g.nodes), len(g.edges)
    )
    return g


def _norm(s: str, node_type: str = "") -> str:
    text = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    if not text:
        return ""
    text = re.sub(r"'s\b", "", text)  
    text = re.sub(r"'", "", text)     

    tokens = re.findall(r"[a-z0-9]+", text)
    if node_type == "Character":
        stripped = [token for token in tokens if token not in CHARACTER_TITLE_TOKENS]
        if stripped:
            tokens = stripped
    return " ".join(tokens)


def _as_list(val) -> List:
    return _coerce_str_list(val)


def _as_set(val) -> set:
    return set(_as_list(val))


def _merge_into(existing: Dict, incoming: Dict) -> None:
    """Merge incoming node data into existing node dict in place."""
    # Merge aliases
    aliases = set(_coerce_str_list(existing.get("aliases", [])))
    incoming_aliases = incoming.get("surface_forms") or incoming.get("aliases") or [
        incoming.get("canonical_name", incoming.get("name", ""))
    ]
    for form in _coerce_str_list(incoming_aliases):
        aliases.add(form)
    aliases.add(incoming.get("canonical_name", incoming.get("name", "")))
    existing["aliases"] = _clean_aliases(existing.get("type", ""), existing.get("name", ""), aliases)

    # Merge scene_refs
    refs = set(_coerce_str_list(existing.get("scene_refs", [])))
    refs.update(_as_list(incoming.get("scene_refs") or incoming.get("scene_id")))
    refs.discard("")
    existing["scene_refs"] = list(refs)

    # Merge evidence
    ev = set(_coerce_str_list(existing.get("evidence", [])))
    ev.update(_coerce_str_list(incoming.get("evidence", [])))
    existing["evidence"] = list(ev)

    # Prefer longer description
    if len(incoming.get("description", "")) > len(existing.get("description", "")):
        existing["description"] = incoming["description"]


def _coerce_str_list(val) -> List[str]:
    """Convert mixed scalar/list inputs into a clean list of strings."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        items = list(val)
    else:
        items = [val]

    cleaned = []
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
            cleaned.append(text)
    return cleaned


def _clean_aliases(node_type: str, canonical_name: str, aliases) -> List[str]:
    """Dedupe aliases and drop obvious generic garbage for characters."""
    ordered = []
    seen = set()
    for alias in [canonical_name, *_coerce_str_list(aliases)]:
        alias = str(alias).strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)

        if node_type == "Character":
            normalized_alias = _norm(alias, node_type)
            normalized_canonical = _norm(canonical_name, node_type)
            if (
                normalized_alias != normalized_canonical
                and (
                    _looks_like_generic_character_alias(alias)
                    or not normalized_alias
                )
            ):
                continue

        ordered.append(alias)

    return ordered


def _looks_like_generic_character_alias(alias: str) -> bool:
    ascii_alias = unicodedata.normalize("NFKD", str(alias or "")).encode("ascii", "ignore").decode("ascii")
    raw_tokens = re.findall(r"[a-z0-9]+", ascii_alias.lower())
    if not raw_tokens:
        return True

    generic_tokens = {
        "the",
        "your",
        "voice",
        "captain",
        "mister",
        "sir",
        "crew",
        "subject",
        "vessel",
        "protege",
    }
    if " ".join(raw_tokens) in GENERIC_CHARACTER_ALIASES:
        return True
    if all(token in CHARACTER_TITLE_TOKENS or token in generic_tokens for token in raw_tokens):
        return True
    return False


def _repair_graph(graph: KnowledgeGraph) -> None:
    """Apply post-build graph repairs required by downstream consumers."""
    _coerce_performs_sources_to_characters(graph)
    _dedupe_and_repair_edges(graph)


def _coerce_performs_sources_to_characters(graph: KnowledgeGraph) -> None:
    for edge in graph.edges:
        if edge.get("relation") != RelationType.PERFORMS.value:
            continue
        source = graph.nodes.get(edge.get("source"))
        if source and source.get("type") not in {NodeType.CHARACTER.value, NodeType.VEHICLE.value}:
            source["type"] = NodeType.CHARACTER.value


def _dedupe_and_repair_edges(graph: KnowledgeGraph) -> None:
    existing = {
        (edge.get("source"), edge.get("relation"), edge.get("target"))
        for edge in graph.edges
    }
    reverse_edges: List[Dict] = []
    for edge in graph.edges:
        relation = edge.get("relation")
        if relation in {
            RelationType.KINSHIP_WITH.value,
            RelationType.AFFINITY_WITH.value,
            RelationType.HOSTILITY_WITH.value,
        }:
            reverse_relation = relation
        elif relation == RelationType.PRECEDES.value:
            reverse_relation = RelationType.OCCURS_AFTER.value
        else:
            continue

        reverse_key = (edge.get("target"), reverse_relation, edge.get("source"))
        if reverse_key in existing:
            continue

        existing.add(reverse_key)
        reverse_edges.append({
            "id": f"e_{uuid.uuid4().hex[:12]}",
            "source": edge.get("target"),
            "relation": reverse_relation,
            "target": edge.get("source"),
            "scene_refs": list(edge.get("scene_refs", [])),
            "evidence": list(edge.get("evidence", [])),
            "confidence": edge.get("confidence", 0.8),
        })

    graph.edges.extend(reverse_edges)
