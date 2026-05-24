import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any

from ..schema import is_valid_triple, NodeType, RelationType

logger = logging.getLogger(__name__)


def compute_stats(graph: Dict) -> Dict[str, Any]:
    """
    Compute intrinsic statistics for a final graph dict.

    Args:
        graph: Dict with 'nodes' and 'edges' lists.

    Returns:
        Statistics dictionary.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    stats: Dict[str, Any] = {
        "movie_id": graph.get("movie_id", ""),
        "title": graph.get("title", ""),
    }

    # Node counts by type
    type_counts = Counter(n.get("type", "UNKNOWN") for n in nodes)
    stats["node_count_total"] = len(nodes)
    stats["node_count_by_type"] = dict(type_counts)

    # Edge counts by relation
    rel_counts = Counter(e.get("relation", "UNKNOWN") for e in edges)
    stats["edge_count_total"] = len(edges)
    stats["edge_count_by_relation"] = dict(rel_counts)

    # Evidence coverage
    nodes_with_evidence = sum(1 for n in nodes if n.get("evidence"))
    edges_with_evidence = sum(1 for e in edges if e.get("evidence"))
    stats["nodes_with_evidence_pct"] = (
        round(nodes_with_evidence / len(nodes) * 100, 1) if nodes else 0.0
    )
    stats["edges_with_evidence_pct"] = (
        round(edges_with_evidence / len(edges) * 100, 1) if edges else 0.0
    )

    # Schema violations
    violations = 0
    for edge in edges:
        try:
            src_id = edge.get("source")
            tgt_id = edge.get("target")
            src_node = next((n for n in nodes if n["id"] == src_id), None)
            tgt_node = next((n for n in nodes if n["id"] == tgt_id), None)
            if src_node and tgt_node:
                st = NodeType(src_node["type"])
                tt = NodeType(tgt_node["type"])
                rt = RelationType(edge["relation"])
                if not is_valid_triple(st, rt, tt):
                    violations += 1
        except (ValueError, KeyError):
            violations += 1
    stats["schema_violation_count"] = violations

    # Average degree
    degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge.get("source", "")] += 1
        degree[edge.get("target", "")] += 1
    stats["avg_degree"] = (
        round(sum(degree.values()) / len(nodes), 2) if nodes else 0.0
    )

    # Disconnected components (simple union-find)
    stats["disconnected_component_count"] = _count_components(nodes, edges)

    # Alias / duplicate stats
    stats["avg_aliases_per_node"] = (
        round(sum(len(n.get("aliases", [])) for n in nodes) / len(nodes), 2) if nodes else 0.0
    )

    # Confidence stats for edges
    confidences = [e.get("confidence", 0.0) for e in edges if isinstance(e.get("confidence"), (int, float))]
    stats["avg_edge_confidence"] = (
        round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    )

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print statistics to the logger."""
    logger.info("=" * 60)
    logger.info("Graph stats for: %s (%s)", stats.get("title", ""), stats.get("movie_id", ""))
    logger.info("Nodes total: %d", stats["node_count_total"])
    for t, c in sorted(stats["node_count_by_type"].items()):
        logger.info("  %-15s : %d", t, c)
    logger.info("Edges total: %d", stats["edge_count_total"])
    for r, c in sorted(stats["edge_count_by_relation"].items(), key=lambda x: -x[1]):
        logger.info("  %-20s : %d", r, c)
    logger.info("Evidence coverage: nodes=%.1f%%, edges=%.1f%%",
                stats["nodes_with_evidence_pct"], stats["edges_with_evidence_pct"])
    logger.info("Schema violations: %d", stats["schema_violation_count"])
    logger.info("Avg degree: %.2f", stats["avg_degree"])
    logger.info("Disconnected components: %d", stats["disconnected_component_count"])
    logger.info("Avg edge confidence: %.3f", stats["avg_edge_confidence"])
    logger.info("=" * 60)


def compute_pre_post_merge_stats(
    raw_nodes: List[Dict], merged_nodes: List[Dict]
) -> Dict[str, Any]:
    """Compare node counts before and after merge."""
    return {
        "raw_count": len(raw_nodes),
        "merged_count": len(merged_nodes),
        "duplicate_rate": (
            round((len(raw_nodes) - len(merged_nodes)) / len(raw_nodes), 3)
            if raw_nodes else 0.0
        ),
    }


def _count_components(nodes: List[Dict], edges: List[Dict]) -> int:
    """Count disconnected components using union-find."""
    node_ids = [n["id"] for n in nodes]
    parent = {nid: nid for nid in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in parent and tgt in parent:
            union(src, tgt)

    roots = {find(nid) for nid in node_ids}
    return len(roots)
