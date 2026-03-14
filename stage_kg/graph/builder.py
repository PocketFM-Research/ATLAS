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
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..schema import NodeType, RelationType, is_valid_triple

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory movie-level knowledge graph."""

    def __init__(self, movie_id: str, title: str = ""):
        self.movie_id = movie_id
        self.title = title
        self.nodes: Dict[str, Dict] = {}   # canonical_id -> node dict
        self.edges: List[Dict] = []

        # Lookup indexes
        self._name_to_id: Dict[str, str] = {}  # normalized_name -> canonical_id
        self._temp_to_canonical: Dict[str, str] = {}  # raw extraction id -> canonical_id

    # ------------------------------------------------------------------ nodes

    def add_node(self, node: Dict) -> str:
        """
        Add or merge a node into the graph.

        Returns the canonical node ID.
        """
        canonical_name = node.get("canonical_name") or node.get("name", "")
        norm_name = _norm(canonical_name)
        node_type = node.get("type", "")

        # Lookup key is (type, normalized_name)
        lookup_key = f"{node_type}::{norm_name}"

        if lookup_key in self._name_to_id:
            canonical_id = self._name_to_id[lookup_key]
            existing = self.nodes[canonical_id]
            _merge_into(existing, node)
        else:
            proposed_id = node.get("id") or ""
            if proposed_id and proposed_id in self.nodes:
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
                    "aliases": list(set(node.get("surface_forms", [canonical_name]))),
                    "description": node.get("description", ""),
                    "scene_refs": _as_list(node.get("scene_refs") or node.get("scene_id")),
                    "evidence": node.get("evidence", []),
                }
                self.nodes[canonical_id] = new_node
                self._name_to_id[lookup_key] = canonical_id

        # Map raw temp/extraction id to canonical_id
        for raw_id in [node.get("id"), node.get("temp_id")]:
            if raw_id:
                self._temp_to_canonical[raw_id] = canonical_id

        return canonical_id

    def resolve_id(self, raw_id: str) -> Optional[str]:
        """Map a raw extraction temp_id or scene-level id to a canonical node id."""
        return self._temp_to_canonical.get(raw_id)

    # ------------------------------------------------------------------ edges

    def add_edge(self, rel: Dict) -> None:
        """
        Add an edge, resolving source/target IDs to canonical node IDs.

        Skips edges with unresolvable endpoints or schema violations.
        """
        src_raw = rel.get("source_id")
        tgt_raw = rel.get("target_id")

        src_id = self.resolve_id(src_raw) if src_raw else None
        tgt_id = self.resolve_id(tgt_raw) if tgt_raw else None

        if not src_id or src_id not in self.nodes:
            logger.debug("Edge skipped: unresolved source %s", src_raw)
            return
        if not tgt_id or tgt_id not in self.nodes:
            logger.debug("Edge skipped: unresolved target %s", tgt_raw)
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
                    set(existing.get("evidence", [])) | set(rel.get("evidence", []))
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
            norm_key = f"{node['type']}::{_norm(node['name'])}"
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
            if raw_id and raw_id not in g._temp_to_canonical:
                g._temp_to_canonical[raw_id] = canonical_id

    # Step 2: Add normalized entities by type
    for node_type, nodes in normalized_entities.items():
        for ent in nodes:
            canonical_id = g.add_node(ent)
            for raw_id in ent.get("_merged_raw_ids", []):
                if raw_id and raw_id not in g._temp_to_canonical:
                    g._temp_to_canonical[raw_id] = canonical_id

    # Step 3: Register all raw extraction IDs that weren't covered by normalization
    # (merge-by-name will handle duplicates; new nodes fill any gaps)
    for scene_id, events in all_events.items():
        for ev in events:
            ev_node = {**ev, "type": "Event"}
            g.add_node(ev_node)

    for scene_id, entities in all_entities.items():
        for ent in entities:
            g.add_node(ent)

    # Step 4: Add edges — resolve source/target IDs via _temp_to_canonical
    for scene_id, relations in all_relations.items():
        for rel in relations:
            g.add_edge(rel)

    logger.info(
        "Built graph for '%s': %d nodes, %d edges", title or movie_id, len(g.nodes), len(g.edges)
    )
    return g


def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s.strip().lower())


def _as_list(val) -> List:
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if v]
    return [val]


def _as_set(val) -> set:
    return set(_as_list(val))


def _merge_into(existing: Dict, incoming: Dict) -> None:
    """Merge incoming node data into existing node dict in place."""
    # Merge aliases
    aliases = set(existing.get("aliases", []))
    for form in incoming.get("surface_forms", []):
        aliases.add(form)
    aliases.add(incoming.get("canonical_name", incoming.get("name", "")))
    existing["aliases"] = list(aliases)

    # Merge scene_refs
    refs = set(existing.get("scene_refs", []))
    refs.update(_as_list(incoming.get("scene_refs") or incoming.get("scene_id")))
    refs.discard("")
    existing["scene_refs"] = list(refs)

    # Merge evidence
    ev = set(existing.get("evidence", []))
    ev.update(incoming.get("evidence", []))
    existing["evidence"] = list(ev)

    # Prefer longer description
    if len(incoming.get("description", "")) > len(existing.get("description", "")):
        existing["description"] = incoming["description"]
