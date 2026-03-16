"""Multi-hop graph verification for claims."""

import json
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

from stage_kg.schema import VALID_TRIPLES, NodeType, RelationType
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.claim_extractor import Claim


@dataclass
class VerificationResult:
    """Result of verifying a claim against the graph."""
    claim: Claim
    status: str  # "grounded", "grounded_multihop", "partial", "hallucinated", "contradiction"
    depth: int  # Number of hops to ground the claim
    path: List[str] = field(default_factory=list)  # Node IDs in the path
    intermediate_types: List[str] = field(default_factory=list)  # Types of intermediate nodes
    relation_subset: List[str] = field(default_factory=list)  # Relations used in path
    supported_node_types: List[str] = field(default_factory=list)  # Which node types support the claim
    confidence: float = 0.0  # Confidence in the grounding
    
    def to_dict(self):
        return {
            "claim": self.claim.to_dict(),
            "status": self.status,
            "depth": self.depth,
            "path": self.path,
            "intermediate_types": self.intermediate_types,
            "relation_subset": self.relation_subset,
            "supported_node_types": self.supported_node_types,
            "confidence": self.confidence,
        }


class KnowledgeGraphVerifier:
    """Verify claims against a knowledge graph using multi-hop reasoning."""
    
    def __init__(self, graph_path: str, config: EvaluationConfig = None):
        """
        Initialize verifier with a graph.
        
        Args:
            graph_path: Path to final_graph.json
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.graph_path = Path(graph_path)
        
        # Load graph
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        # Build efficient lookup structures
        self.nodes_by_id = {n["id"]: n for n in self.graph_data["nodes"]}
        self.nodes_by_name = {}  # name -> [node_ids] (many-to-one due to aliases)
        self.edges = self.graph_data.get("edges", [])
        
        # Build adjacency: source -> [(target, relation), ...]
        self.adjacency = {}
        for node in self.graph_data["nodes"]:
            self.adjacency[node["id"]] = []
        
        for edge in self.edges:
            src, tgt = edge["source"], edge["target"]
            self.adjacency.setdefault(src, []).append({
                "target": tgt,
                "relation": edge["relation"],
                "confidence": edge.get("confidence", 1.0),
                "edge_id": f"{src}-{edge['relation']}-{tgt}",
            })
        
        # Build name index
        for node in self.graph_data["nodes"]:
            # Index canonical name
            self.nodes_by_name.setdefault(node["name"].lower(), []).append(node["id"])
            # Index aliases
            for alias in node.get("aliases", []):
                self.nodes_by_name.setdefault(alias.lower(), []).append(node["id"])
    
    def verify_claim(self, claim: Claim) -> VerificationResult:
        """
        Verify a single claim against the graph.
        
        Returns: VerificationResult with status, path, and metadata
        """
        # Canonicalize subject and object to node IDs
        subject_ids = self._resolve_entity(claim.subject)
        object_ids = self._resolve_entity(claim.object)
        
        if not subject_ids:
            return VerificationResult(
                claim=claim,
                status="hallucinated",
                depth=0,
                confidence=0.0
            )
        
        if claim.object and not object_ids:
            return VerificationResult(
                claim=claim,
                status="hallucinated",
                depth=0,
                confidence=0.0
            )
        
        # Try to find grounding via BFS
        if claim.object:
            # SVO claim: try to find path from subject to object
            best_result = None
            for subj_id in subject_ids:
                for obj_id in object_ids:
                    result = self._bfs_verify(
                        subj_id, obj_id, claim.predicate, claim
                    )
                    if best_result is None or result.depth < best_result.depth:
                        best_result = result
            return best_result or VerificationResult(
                claim=claim, status="hallucinated", depth=0
            )
        else:
            # SV claim: verify subject has the property/type
            result = VerificationResult(
                claim=claim,
                status="grounded" if subject_ids else "hallucinated",
                depth=1 if subject_ids else 0,
                path=subject_ids,
                confidence=0.9 if subject_ids else 0.0
            )
            return result
    
    def _resolve_entity(self, entity_name: str) -> List[str]:
        """
        Resolve an entity name to node IDs.
        
        Returns list of node IDs (may be multiple due to aliases)
        """
        if not entity_name:
            return []
        
        name_key = entity_name.lower().strip()
        return self.nodes_by_name.get(name_key, [])
    
    def _bfs_verify(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        claim: Claim,
        max_depth: int = None
    ) -> VerificationResult:
        """
        BFS to find a path from source to target that supports the predicate.
        
        Returns VerificationResult with the shortest path found.
        """
        max_depth = max_depth or self.config.max_hop_depth
        
        # Check direct edge first
        direct = self._check_direct_edge(source_id, target_id, predicate)
        if direct:
            return VerificationResult(
                claim=claim,
                status="grounded",
                depth=1,
                path=[source_id, target_id],
                relation_subset=[predicate],
                confidence=direct.get("confidence", 0.8)
            )
        
        # BFS for indirect paths
        queue = deque([(source_id, [source_id], [], 0)])
        visited = {source_id}
        
        while queue:
            node_id, path, relations, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            # Check neighbors
            for edge_info in self.adjacency.get(node_id, []):
                neighbor = edge_info["target"]
                
                if neighbor == target_id:
                    # Found path to target
                    return VerificationResult(
                        claim=claim,
                        status="grounded_multihop" if depth > 0 else "grounded",
                        depth=depth + 1,
                        path=path + [neighbor],
                        relation_subset=relations + [edge_info["relation"]],
                        confidence=edge_info.get("confidence", 0.8) * (1.0 / (depth + 1))
                    )
                
                if neighbor not in visited and depth < max_depth:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path + [neighbor],
                        relations + [edge_info["relation"]],
                        depth + 1
                    ))
        
        return VerificationResult(
            claim=claim,
            status="hallucinated",
            depth=0,
            confidence=0.0
        )
    
    def _check_direct_edge(
        self,
        source_id: str,
        target_id: str,
        predicate: str
    ) -> Optional[Dict]:
        """Check if a direct edge exists matching the predicate."""
        for edge_info in self.adjacency.get(source_id, []):
            if edge_info["target"] == target_id:
                # Exact match or semantic similarity
                if edge_info["relation"] == predicate or self._predicate_match(
                    predicate, edge_info["relation"]
                ):
                    return edge_info
        return None
    
    def _predicate_match(self, claim_pred: str, graph_pred: str) -> bool:
        """Check if a claim predicate matches a graph predicate semantically."""
        # Simple heuristics: "is with" -> "located_at" or "affinity_with"
        synonyms = {
            "is": "is_a",
            "has": "possesses",
            "meets": "occurs_at",
            "knows": "affiliated_with",
            "likes": "affinity_with",
            "dislikes": "hostility_with",
            "related": "kinship_with",
            "happens": "occurs_on",
        }
        
        return claim_pred in synonyms and synonyms[claim_pred] == graph_pred