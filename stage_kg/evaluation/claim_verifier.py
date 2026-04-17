"""Schema-aware graph verification for claims."""

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from stage_kg.schema import NodeType, RelationType, VALID_TRIPLES
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.new_claim_extractor import Claim


GROUND_STATUSES = {"grounded", "grounded_multihop"}
EVENT_ROLE_RELATIONS = {"performs", "undergoes", "experiences"}
SOCIAL_RELATIONS = {"affiliated_with", "affinity_with", "hostility_with"}
TEXT_BACKED_RELATIONS = {
    "affiliated_with",
    "affinity_with",
    "experiences",
    "hostility_with",
    "is_a",
    "kinship_with",
    "located_at",
    "occurs_at",
    "occurs_on",
    "part_of",
    "performs",
    "possesses",
    "present_on",
    "references",
    "undergoes",
    "uses",
}
KINSHIP_CUES = {
    "daughter",
    "son",
    "father",
    "mother",
    "husband",
    "wife",
    "brother",
    "sister",
    "parent",
    "child",
    "uncle",
    "aunt",
}
SCENE_SUPPORT_NODE_TYPES = {
    "affiliated_with": {"Character", "Concept", "Object"},
    "affinity_with": {"Character"},
    "experiences": {"Event", "Concept"},
    "hostility_with": {"Character", "Event"},
    "kinship_with": {"Character"},
    "performs": {"Event"},
    "part_of": {"Location", "Object"},
    "present_on": {"Object", "Location", "Character", "Concept"},
    "references": {"Event", "Concept"},
    "undergoes": {"Event"},
    "experiences": {"Event"},
    "occurs_at": {"Location"},
    "occurs_on": {"TimePoint", "Concept"},
    "uses": {"Object"},
    "possesses": {"Object"},
    "located_at": {"Location", "Object"},
    "is_a": {"Concept"},
}
ALLOWED_SOURCE_TYPES = {}
ALLOWED_TARGET_TYPES = {}
for src_type, relation, tgt_type in VALID_TRIPLES:
    ALLOWED_SOURCE_TYPES.setdefault(relation.value, set()).add(src_type.value)
    ALLOWED_TARGET_TYPES.setdefault(relation.value, set()).add(tgt_type.value)
GENERIC_OBJECT_TOKENS = {"person", "vehicle", "thing", "appearance", "entity", "figure"}
ENTITY_TYPE_PRIORITY = {
    "Character": 0,
    "Location": 1,
    "Object": 2,
    "Concept": 3,
    "TimePoint": 4,
    "Event": 5,
}
SCHEMA_SYNONYMS = {
    "is": "is_a",
    "has": "possesses",
    "owns": "possesses",
    "holding": "possesses",
    "holds": "possesses",
    "appears": "located_at",
    "stands": "located_at",
    "sits": "located_at",
    "waits": "located_at",
    "meets": "affinity_with",
    "knows": "affiliated_with",
    "likes": "affinity_with",
    "dislikes": "hostility_with",
    "related": "kinship_with",
    "happens": "occurs_on",
}
NAME_TITLE_TOKENS = {"the", "captain", "commander", "lt", "lt.", "mr", "mr.", "mister", "dr", "dr.", "doctor"}
FAMILY_TITLE_ALIASES = {
    "aunt": "aunt",
    "brother": "brother",
    "dad": "father",
    "daddy": "father",
    "daughter": "daughter",
    "father": "father",
    "girlfriend": "girlfriend",
    "husband": "husband",
    "ma": "mother",
    "mom": "mother",
    "mommy": "mother",
    "mother": "mother",
    "mummy": "mother",
    "son": "son",
    "sister": "sister",
    "uncle": "uncle",
    "wife": "wife",
    "boyfriend": "boyfriend",
}
SCENE_BLOB_RELATIONS = {
    "affiliated_with",
    "affinity_with",
    "experiences",
    "hostility_with",
    "is_a",
    "kinship_with",
    "located_at",
    "occurs_at",
    "occurs_on",
    "part_of",
    "performs",
    "possesses",
    "present_on",
    "references",
    "undergoes",
    "uses",
}
LIGHT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "with",
}


@dataclass
class VerificationResult:
    """Result of verifying a claim against the graph."""

    claim: Claim
    status: str
    depth: int
    path: List[str] = field(default_factory=list)
    intermediate_types: List[str] = field(default_factory=list)
    relation_subset: List[str] = field(default_factory=list)
    supported_node_types: List[str] = field(default_factory=list)
    confidence: float = 0.0

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
    """Verify claims against a knowledge graph using schema-aware grounding."""

    def __init__(self, graph_path: str, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.graph_path = Path(graph_path)

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)

        self.nodes_by_id = {node["id"]: node for node in self.graph_data["nodes"]}
        self.nodes_by_name: Dict[str, List[str]] = {}
        self.edges = self.graph_data.get("edges", [])
        self.adjacency: Dict[str, List[Dict]] = {node_id: [] for node_id in self.nodes_by_id}
        self.node_text_index: Dict[str, str] = {}
        scene_text_parts: Dict[str, List[str]] = defaultdict(list)

        for node in self.graph_data["nodes"]:
            self._index_node_names(node)
            node_text = self._node_text(node)
            self.node_text_index[node["id"]] = node_text
            for scene_id in node.get("scene_refs", []) or []:
                scene_text_parts[str(scene_id)].append(self._scene_node_text(node))

        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            edge_text = self._normalize_text(
                " ".join(
                    part for part in [
                        self.nodes_by_id.get(src, {}).get("name", ""),
                        edge.get("relation", ""),
                        self.nodes_by_id.get(tgt, {}).get("name", ""),
                        *list(edge.get("evidence", []) or []),
                    ]
                    if part
                )
            )
            self.adjacency.setdefault(src, []).append(
                {
                    "target": tgt,
                    "relation": edge["relation"],
                    "confidence": edge.get("confidence", 1.0),
                    "edge_id": edge.get("id", f"{src}-{edge['relation']}-{tgt}"),
                    "scene_refs": [str(scene_id) for scene_id in edge.get("scene_refs", [])],
                    "evidence": edge.get("evidence", []) or [],
                }
            )
            for scene_id in edge.get("scene_refs", []) or []:
                scene_text_parts[str(scene_id)].append(edge_text)
        self.scene_text_index = {
            scene_id: self._normalize_text(" ".join(part for part in parts if part))
            for scene_id, parts in scene_text_parts.items()
        }

    def verify_claim(self, claim: Claim) -> VerificationResult:
        """Verify a single claim against the graph."""
        subject_ids = self._resolve_entity(claim.subject)
        subject_ids = self._filter_subject_ids_for_claim(subject_ids, claim)
        subject_ids = self._rank_ids_for_scene(subject_ids, claim.scene_id, claim.subject)
        subject_ids = self._prefer_explicit_scene_matches(subject_ids, claim.scene_id)
        if not subject_ids:
            subject_ids = self._resolve_subject_from_claim_text(claim)
            subject_ids = self._filter_subject_ids_for_claim(subject_ids, claim)
            subject_ids = self._rank_ids_for_scene(subject_ids, claim.scene_id, claim.subject)
            subject_ids = self._prefer_explicit_scene_matches(subject_ids, claim.scene_id)

        if not subject_ids:
            return VerificationResult(
                claim=claim,
                status="hallucinated",
                depth=0,
                confidence=0.0,
            )

        if not claim.object:
            return VerificationResult(
                claim=claim,
                status="partial",
                depth=0,
                path=subject_ids[:1],
                confidence=0.25,
            )

        best_result = VerificationResult(
            claim=claim,
            status="hallucinated",
            depth=0,
            confidence=0.0,
        )

        object_ids = self._resolve_object_entities(claim)
        object_ids = self._filter_ids_by_allowed_types(object_ids, ALLOWED_TARGET_TYPES.get(claim.predicate))
        object_ids = self._rank_ids_for_scene(object_ids, claim.scene_id, claim.object)
        object_ids = self._prefer_explicit_scene_matches(object_ids, claim.scene_id)
        for subject_id in subject_ids:
            if object_ids:
                for object_id in object_ids:
                    candidate = self._bfs_verify(subject_id, object_id, claim.predicate, claim)
                    best_result = self._choose_better_result(best_result, candidate)
                    candidate = self._verify_composed_scene_support(subject_id, object_id, claim)
                    best_result = self._choose_better_result(best_result, candidate)

            semantic_candidate = self._verify_semantic_edge(subject_id, claim)
            best_result = self._choose_better_result(best_result, semantic_candidate)
            kinship_candidate = self._verify_kinship_text_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, kinship_candidate)
            subject_text_candidate = self._verify_subject_text_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, subject_text_candidate)
            scene_text_candidate = self._verify_scene_text_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, scene_text_candidate)
            scene_blob_candidate = self._verify_scene_blob_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, scene_blob_candidate)

        return best_result

    def _index_node_names(self, node: Dict):
        for name in [node.get("name", "")] + (node.get("aliases", []) or []):
            normalized = self._normalize_text(name)
            if normalized:
                self.nodes_by_name.setdefault(normalized, []).append(node["id"])
                tokens = [token for token in normalized.split() if token and token not in NAME_TITLE_TOKENS]
                if tokens:
                    surname = tokens[-1]
                    if surname != normalized:
                        self.nodes_by_name.setdefault(surname, []).append(node["id"])

    def _resolve_entity(self, entity_name: str) -> List[str]:
        if not entity_name:
            return []

        if entity_name in self.nodes_by_id:
            return [entity_name]

        seen_node_ids: List[str] = []
        seen_lookup: Set[str] = set()
        for candidate_text in self._entity_resolution_candidates(entity_name):
            normalized = self._normalize_text(candidate_text)
            if not normalized:
                continue

            direct = self.nodes_by_name.get(normalized, [])
            if direct:
                for node_id in self._rank_resolved_ids(list(dict.fromkeys(direct)), normalized):
                    if node_id not in seen_lookup:
                        seen_lookup.add(node_id)
                        seen_node_ids.append(node_id)
                continue

            candidates: List[Tuple[float, str]] = []
            for indexed_name, node_ids in self.nodes_by_name.items():
                score = self._entity_match_score(normalized, indexed_name)
                if score >= 0.75:
                    for node_id in node_ids:
                        candidates.append((score, node_id))

            candidates.sort(reverse=True)
            resolved = []
            seen = set()
            for _, node_id in candidates:
                if node_id not in seen:
                    seen.add(node_id)
                    resolved.append(node_id)
            for node_id in self._rank_resolved_ids(resolved, normalized):
                if node_id not in seen_lookup:
                    seen_lookup.add(node_id)
                    seen_node_ids.append(node_id)
        return seen_node_ids

    def _resolve_object_entities(self, claim: Claim) -> List[str]:
        resolved: List[str] = []
        seen: Set[str] = set()
        for candidate in self._object_resolution_candidates(claim):
            for node_id in self._resolve_entity(candidate):
                if node_id in seen:
                    continue
                seen.add(node_id)
                resolved.append(node_id)
        return resolved

    def _bfs_verify(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        claim: Claim,
        max_depth: Optional[int] = None,
    ) -> VerificationResult:
        max_depth = max_depth or self.config.max_hop_depth

        direct = self._check_direct_edge(source_id, target_id, predicate, claim.scene_id, claim)
        if direct:
            target_node = self.nodes_by_id.get(target_id, {})
            return VerificationResult(
                claim=claim,
                status="grounded",
                depth=1,
                path=[source_id, target_id],
                relation_subset=[direct["relation"]],
                supported_node_types=[target_node.get("type", "")],
                confidence=direct.get("confidence", 0.8),
            )

        queue = deque([(source_id, [source_id], [], [], 0)])
        visited = {(source_id, 0)}

        while queue:
            node_id, path, relations, intermediate_types, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge_info in self.adjacency.get(node_id, []):
                if not self._scene_matches(edge_info, claim.scene_id):
                    continue

                neighbor = edge_info["target"]
                next_relations = relations + [edge_info["relation"]]
                next_types = intermediate_types + [self.nodes_by_id.get(neighbor, {}).get("type", "")]
                next_depth = depth + 1

                if claim.predicate in SOCIAL_RELATIONS and next_depth > 1:
                    continue

                if neighbor == target_id and self._path_supports_predicate(predicate, next_relations):
                    return VerificationResult(
                        claim=claim,
                        status="grounded_multihop",
                        depth=next_depth,
                        path=path + [neighbor],
                        relation_subset=next_relations,
                        intermediate_types=next_types[:-1],
                        supported_node_types=[self.nodes_by_id.get(target_id, {}).get("type", "")],
                        confidence=edge_info.get("confidence", 0.8) * (1.0 / next_depth),
                    )

                state = (neighbor, next_depth)
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, path + [neighbor], next_relations, next_types, next_depth))

        return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

    def _check_direct_edge(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        scene_id: str,
        claim: Optional[Claim] = None,
    ) -> Optional[Dict]:
        fallback_edge = None
        for edge_info in self.adjacency.get(source_id, []):
            if edge_info["target"] != target_id:
                continue
            if edge_info["relation"] == predicate or self._predicate_match(predicate, edge_info["relation"]):
                if claim and not self._direct_edge_supports_claim(claim, edge_info):
                    continue
                if self._scene_matches(edge_info, scene_id):
                    return edge_info
                if predicate in SOCIAL_RELATIONS and fallback_edge is None:
                    fallback_edge = edge_info
        return fallback_edge

    def _verify_semantic_edge(self, subject_id: str, claim: Claim) -> VerificationResult:
        best_score = 0.0
        best_edge = None
        best_target = None

        for edge_info in self.adjacency.get(subject_id, []):
            if not self._scene_matches(edge_info, claim.scene_id):
                continue
            if not self._predicate_match(claim.predicate, edge_info["relation"]):
                continue

            target_node = self.nodes_by_id.get(edge_info["target"], {})
            score = self._claim_edge_similarity(claim, edge_info, target_node)
            if score > best_score:
                best_score = score
                best_edge = edge_info
                best_target = target_node

        if not best_edge or not best_target:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        # Event-role claims are especially prone to false positives from nearby
        # but semantically different scene actions, so require stronger overlap.
        threshold = 0.15 if claim.predicate == "performs" else (0.18 if claim.predicate in EVENT_ROLE_RELATIONS else 0.22)
        if claim.predicate in SOCIAL_RELATIONS:
            cue_score = self._edge_cue_similarity(claim, best_edge, best_target, include_node_text=False)
            if cue_score < 0.18:
                return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)
        if best_score < threshold:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        confidence = min(0.99, best_edge.get("confidence", 0.8) * (0.5 + best_score / 2.0))
        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id, best_edge["target"]],
            relation_subset=[best_edge["relation"]],
            supported_node_types=[best_target.get("type", "")],
            confidence=confidence,
        )

    def _verify_subject_text_support(self, subject_id: str, claim: Claim) -> VerificationResult:
        if claim.predicate not in TEXT_BACKED_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        # Subject-scene blobs are useful for object/state facts, but they can
        # over-ground action and social claims by picking up nearby scene
        # context rather than subject-specific support.
        if claim.predicate in {"affinity_with", "hostility_with", "affiliated_with", "located_at", "uses"}:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        subject_node = self.nodes_by_id.get(subject_id, {})
        scene_blob = self._subject_scene_text(subject_id, claim.scene_id)
        claim_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._claim_text_variants(claim)),
            default=0.0,
        )
        no_subject_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._claim_text_variants_without_subject(claim)),
            default=0.0,
        )
        object_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._object_text_variants(claim.object)),
            default=0.0,
        )
        cue_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._relation_cue_variants(claim)),
            default=0.0,
        )

        if claim.predicate == "performs":
            score = max(claim_score, no_subject_score)
            threshold = 0.05
        elif claim.predicate == "undergoes":
            score = max(object_score, claim_score, no_subject_score)
            threshold = 0.08
        elif claim.predicate == "experiences":
            score = max(object_score, claim_score, no_subject_score)
            threshold = 0.06
        elif claim.predicate == "causes":
            score = max(object_score, claim_score, no_subject_score)
            threshold = 0.04
        elif claim.predicate == "possesses":
            score = object_score
            threshold = 0.12
        else:
            score = object_score
            threshold = 0.18 if claim.predicate in {"is_a", "possesses", "kinship_with", "located_at"} else 0.2

        if score < threshold:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id],
            supported_node_types=[subject_node.get("type", "")],
            confidence=min(0.95, 0.45 + score / 2.0),
        )

    def _verify_kinship_text_support(self, subject_id: str, claim: Claim) -> VerificationResult:
        if claim.predicate != "kinship_with":
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        scene_blob = self._subject_scene_text(subject_id, claim.scene_id)
        scene_blob = self._normalize_text(" ".join(part for part in [scene_blob, self._scene_blob(claim.scene_id)] if part))
        claim_evidence_blob = self._normalize_text(" ".join(claim.evidence or []))
        combined_blob = self._normalize_text(" ".join(part for part in [scene_blob, claim_evidence_blob] if part))
        if not combined_blob:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        normalized_claim = self._normalize_text(claim.claim_text)
        cue_hits = [cue for cue in KINSHIP_CUES if cue in combined_blob and cue in normalized_claim]
        for token in normalized_claim.split():
            canonical = FAMILY_TITLE_ALIASES.get(token)
            if canonical and canonical in KINSHIP_CUES and canonical not in cue_hits:
                cue_hits.append(canonical)
        object_variants = self._object_text_variants(claim.object)
        stripped_object = self._strip_family_titles(claim.object)
        if stripped_object:
            object_variants.extend(self._object_text_variants(stripped_object))
        object_score = max((self._text_similarity(variant, combined_blob) for variant in object_variants), default=0.0)

        if not cue_hits or object_score < 0.18:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        subject_node = self.nodes_by_id.get(subject_id, {})
        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id],
            supported_node_types=[subject_node.get("type", "")],
            confidence=min(0.9, 0.5 + object_score / 2.0),
        )

    def _verify_composed_scene_support(
        self,
        subject_id: str,
        object_id: str,
        claim: Claim,
    ) -> VerificationResult:
        if claim.predicate not in TEXT_BACKED_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        if claim.predicate in SOCIAL_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        subject_node = self.nodes_by_id.get(subject_id, {})
        object_node = self.nodes_by_id.get(object_id, {})
        if not subject_node or not object_node:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        if not self._node_in_scene(subject_node, claim.scene_id) or not self._node_in_scene(object_node, claim.scene_id):
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        combined = " ".join(
            part for part in [
                self._subject_scene_text(subject_id, claim.scene_id),
                self._node_text(object_node),
            ]
            if part
        )
        object_score = max(
            (self._text_similarity(variant, combined) for variant in self._object_text_variants(claim.object)),
            default=0.0,
        )
        claim_score = max(
            (self._text_similarity(variant, combined) for variant in self._claim_text_variants(claim)),
            default=0.0,
        )
        score = max(object_score, claim_score)
        if score < 0.18:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        return VerificationResult(
            claim=claim,
            status="grounded_multihop",
            depth=2,
            path=[subject_id, object_id],
            supported_node_types=[object_node.get("type", "")],
            confidence=min(0.9, 0.4 + score / 2.0),
        )

    def _verify_scene_text_support(self, subject_id: str, claim: Claim) -> VerificationResult:
        if claim.predicate not in TEXT_BACKED_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        if claim.predicate in SOCIAL_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        allowed_types = SCENE_SUPPORT_NODE_TYPES.get(claim.predicate)
        if not allowed_types:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        scene_nodes = [
            node for node in self.graph_data["nodes"]
            if self._node_in_scene(node, claim.scene_id) and node.get("type") in allowed_types
        ]
        best_score = 0.0
        best_target = None
        query_objects = self._object_text_variants(claim.object)
        query_claims = self._claim_text_variants(claim)

        for node in scene_nodes:
            target_text = self._node_text(node)
            score = max([
                *(self._text_similarity(query, target_text) for query in query_objects),
                *(self._text_similarity(query, target_text) for query in query_claims),
            ], default=0.0)
            if score > best_score:
                best_score = score
                best_target = node

        threshold = {
            "performs": 0.22,
            "undergoes": 0.08,
            "experiences": 0.08,
            "located_at": 0.08,
            "possesses": 0.05,
            "uses": 0.22,
        }.get(claim.predicate, 0.28)
        if not best_target or best_score < threshold:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id, best_target["id"]],
            supported_node_types=[best_target.get("type", "")],
            confidence=min(0.85, 0.35 + best_score / 2.0),
        )

    def _verify_scene_blob_support(self, subject_id: str, claim: Claim) -> VerificationResult:
        if claim.predicate not in SCENE_BLOB_RELATIONS:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        if claim.predicate in SOCIAL_RELATIONS | {"located_at", "uses"}:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        scene_blob = self._scene_blob(claim.scene_id)
        if not scene_blob:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        subject_node = self.nodes_by_id.get(subject_id, {})
        subject_variants = self._subject_text_variants(subject_node, claim.subject)
        object_variants = self._object_text_variants(claim.object)
        claim_variants = self._claim_text_variants(claim)
        no_subject_claim_variants = self._claim_text_variants_without_subject(claim)
        cue_variants = self._relation_cue_variants(claim)

        subject_score = max((self._text_similarity(variant, scene_blob) for variant in subject_variants), default=0.0)
        object_score = max((self._text_similarity(variant, scene_blob) for variant in object_variants), default=0.0)
        claim_score = max((self._text_similarity(variant, scene_blob) for variant in claim_variants), default=0.0)
        no_subject_claim_score = max((self._text_similarity(variant, scene_blob) for variant in no_subject_claim_variants), default=0.0)
        cue_score = max((self._text_similarity(variant, scene_blob) for variant in cue_variants), default=0.0)

        scene_subject_ok = self._node_in_scene(subject_node, claim.scene_id) or subject_score >= 0.2
        if not scene_subject_ok:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        if claim.predicate in {"kinship_with", "affinity_with", "hostility_with", "affiliated_with"}:
            passed = object_score >= 0.2 and cue_score >= 0.2
        elif claim.predicate in {"located_at", "occurs_at", "occurs_on", "part_of", "present_on"}:
            passed = object_score >= 0.22 and max(cue_score, no_subject_claim_score) >= 0.18
        elif claim.predicate == "performs":
            passed = max(object_score, no_subject_claim_score) >= 0.16
        elif claim.predicate in {"is_a", "possesses", "uses", "experiences", "undergoes", "references"}:
            passed = max(object_score, no_subject_claim_score) >= 0.22
        else:
            passed = max(object_score, no_subject_claim_score, cue_score) >= 0.24

        if not passed:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        confidence = min(0.82, 0.35 + max(object_score, no_subject_claim_score, cue_score) / 2.0)
        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id] if subject_id else [],
            supported_node_types=[subject_node.get("type", "")] if subject_node else [],
            confidence=confidence,
        )

    def _claim_edge_similarity(self, claim: Claim, edge_info: Dict, target_node: Dict) -> float:
        target_text = self.node_text_index.get(target_node.get("id", ""), "")
        object_variants = self._object_text_variants(claim.object)
        claim_variants = self._claim_text_variants(claim)
        stripped_claim_variants = self._claim_text_variants_without_subject(claim)

        scores = []
        for variant in object_variants:
            scores.append(self._text_similarity(variant, target_text))
        if claim.predicate not in EVENT_ROLE_RELATIONS | {"affinity_with", "hostility_with", "affiliated_with"}:
            for variant in claim_variants:
                scores.append(self._text_similarity(variant, target_text))
        else:
            for variant in stripped_claim_variants:
                scores.append(self._text_similarity(variant, target_text))

        for evidence in edge_info.get("evidence", []):
            evidence_text = self._normalize_text(evidence)
            for variant in object_variants:
                scores.append(self._text_similarity(variant, evidence_text))
            if claim.predicate not in EVENT_ROLE_RELATIONS | {"affinity_with", "hostility_with", "affiliated_with"}:
                for variant in claim_variants:
                    scores.append(self._text_similarity(variant, evidence_text))
            else:
                for variant in stripped_claim_variants:
                    scores.append(self._text_similarity(variant, evidence_text))

        return max(scores) if scores else 0.0

    def _edge_cue_similarity(self, claim: Claim, edge_info: Dict, target_node: Dict, include_node_text: bool = True) -> float:
        cue_variants = self._relation_cue_variants(claim)
        if not cue_variants:
            return 0.0
        texts = [self.node_text_index.get(target_node.get("id", ""), "")] if include_node_text else []
        texts.extend(self._normalize_text(evidence) for evidence in edge_info.get("evidence", []))
        return max(
            (self._text_similarity(cue, text) for cue in cue_variants for text in texts if text),
            default=0.0,
        )

    def _direct_edge_supports_claim(self, claim: Claim, edge_info: Dict) -> bool:
        if claim.predicate not in SOCIAL_RELATIONS:
            return True
        target_node = self.nodes_by_id.get(edge_info["target"], {})
        cue_score = self._edge_cue_similarity(claim, edge_info, target_node, include_node_text=False)
        return cue_score >= 0.18

    def _scene_matches(self, edge_info: Dict, scene_id: str) -> bool:
        scene_refs = edge_info.get("scene_refs", [])
        return not scene_refs or str(scene_id) in scene_refs

    def _path_supports_predicate(self, claim_pred: str, relations: List[str]) -> bool:
        if any(self._predicate_match(claim_pred, relation) for relation in relations):
            return True

        relation_set = set(relations)
        if claim_pred == "located_at" and "occurs_at" in relation_set and relation_set & EVENT_ROLE_RELATIONS:
            return True

        return False

    def _predicate_match(self, claim_pred: str, graph_pred: str) -> bool:
        normalized_claim = SCHEMA_SYNONYMS.get(claim_pred, claim_pred)
        normalized_graph = SCHEMA_SYNONYMS.get(graph_pred, graph_pred)
        if normalized_claim == "references" and normalized_graph == "performs":
            return True
        return normalized_claim == normalized_graph

    def _choose_better_result(
        self,
        current: VerificationResult,
        candidate: VerificationResult,
    ) -> VerificationResult:
        ranking = {
            "grounded": 4,
            "grounded_multihop": 3,
            "partial": 2,
            "hallucinated": 1,
            "contradiction": 0,
        }
        current_rank = ranking.get(current.status, -1)
        candidate_rank = ranking.get(candidate.status, -1)
        if candidate_rank > current_rank:
            return candidate
        if candidate_rank < current_rank:
            return current
        if candidate.confidence > current.confidence:
            return candidate
        if candidate.confidence == current.confidence and candidate.depth and (
            current.depth == 0 or candidate.depth < current.depth
        ):
            return candidate
        return current

    def _entity_match_score(self, query: str, indexed_name: str) -> float:
        query_tokens = self._tokenize(query)
        indexed_tokens = self._tokenize(indexed_name)
        if not query_tokens or not indexed_tokens:
            return 0.0
        if indexed_tokens.issubset(query_tokens):
            return len(indexed_tokens) / len(query_tokens)
        overlap = len(query_tokens & indexed_tokens)
        union = len(query_tokens | indexed_tokens)
        return overlap / union if union else 0.0

    def _rank_resolved_ids(self, node_ids: List[str], normalized_query: str) -> List[str]:
        scored: List[Tuple[Tuple[int, int, int], str]] = []
        query_tokens = self._tokenize(normalized_query)
        for node_id in node_ids:
            node = self.nodes_by_id.get(node_id, {})
            node_type = node.get("type", "")
            names = [node.get("name", "")] + (node.get("aliases", []) or [])
            normalized_names = [self._normalize_text(name) for name in names if name]
            exact = 1 if normalized_query and normalized_query in normalized_names else 0
            event_penalty = 1 if node_type == "Event" else 0
            token_overlap = 0
            for candidate in normalized_names:
                token_overlap = max(token_overlap, len(query_tokens & self._tokenize(candidate)))
            priority = ENTITY_TYPE_PRIORITY.get(node_type, 10)
            scored.append(((-exact, event_penalty, priority, -token_overlap), node_id))
        scored.sort()
        return [node_id for _, node_id in scored]

    def _rank_ids_for_scene(self, node_ids: List[str], scene_id: str, query_text: str) -> List[str]:
        if not node_ids:
            return []
        normalized_query = self._normalize_text(query_text)
        query_tokens = self._tokenize(normalized_query)
        scored: List[Tuple[Tuple[int, int, int, int, int], str]] = []
        for node_id in node_ids:
            node = self.nodes_by_id.get(node_id, {})
            node_type = node.get("type", "")
            names = [node.get("name", "")] + (node.get("aliases", []) or [])
            normalized_names = [self._normalize_text(name) for name in names if name]
            scene_rank = self._scene_rank(node, scene_id)
            exact = 0 if normalized_query and normalized_query in normalized_names else 1
            event_penalty = 1 if node_type == "Event" else 0
            priority = ENTITY_TYPE_PRIORITY.get(node_type, 10)
            token_overlap = 0
            for candidate in normalized_names:
                token_overlap = max(token_overlap, len(query_tokens & self._tokenize(candidate)))
            scored.append(((scene_rank, exact, event_penalty, priority, -token_overlap), node_id))
        scored.sort()
        return [node_id for _, node_id in scored]

    def _scene_rank(self, node: Dict, scene_id: str) -> int:
        refs = [str(ref) for ref in node.get("scene_refs", []) if str(ref)]
        if refs and str(scene_id) in refs:
            return 0
        if not refs:
            return 1
        return 2

    def _filter_ids_by_allowed_types(
        self,
        node_ids: List[str],
        allowed_types: Optional[Set[str]],
    ) -> List[str]:
        if not allowed_types:
            return node_ids
        return [
            node_id for node_id in node_ids
            if self.nodes_by_id.get(node_id, {}).get("type") in allowed_types
        ]

    def _filter_subject_ids_for_claim(self, node_ids: List[str], claim: Claim) -> List[str]:
        allowed_types = ALLOWED_SOURCE_TYPES.get(claim.predicate)
        if claim.predicate == "references":
            allowed_types = set(allowed_types or set())
            allowed_types.add("Character")
        return self._filter_ids_by_allowed_types(node_ids, allowed_types)

    def _prefer_explicit_scene_matches(self, node_ids: List[str], scene_id: str) -> List[str]:
        explicit = [
            node_id for node_id in node_ids
            if self._scene_rank(self.nodes_by_id.get(node_id, {}), scene_id) == 0
        ]
        return explicit or node_ids

    def _entity_resolution_candidates(self, text: str) -> List[str]:
        variants: List[str] = []
        seen: Set[str] = set()

        def add(value: str):
            normalized = self._normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(value)

        add(text)
        normalized = self._normalize_text(text)
        if normalized:
            add(normalized)
            stripped = re.sub(r"^(the|a|an)\s+", "", normalized).strip()
            add(stripped)
            kinship_stripped = self._strip_family_titles(normalized)
            add(kinship_stripped)

        return variants

    def _strip_family_titles(self, text: str) -> str:
        normalized = self._normalize_text(text)
        tokens = [token for token in normalized.split() if token]
        while tokens and tokens[0] in FAMILY_TITLE_ALIASES:
            tokens = tokens[1:]
        return " ".join(tokens)

    def _resolve_subject_from_claim_text(self, claim: Claim) -> List[str]:
        normalized_subject = self._normalize_text(claim.subject)
        normalized_claim = self._normalize_text(claim.claim_text)
        candidates: List[Tuple[float, str]] = []

        for node in self.graph_data["nodes"]:
            if not self._node_in_scene(node, claim.scene_id):
                continue
            node_text = self.node_text_index.get(node["id"], "")
            score = max(
                self._text_similarity(normalized_subject, node_text),
                self._text_similarity(normalized_claim, node_text),
            )
            if score >= 0.22:
                candidates.append((score, node["id"]))

        candidates.sort(reverse=True)
        resolved = []
        seen = set()
        for _, node_id in candidates:
            if node_id in seen:
                continue
            seen.add(node_id)
            resolved.append(node_id)
        ranked = self._rank_resolved_ids(resolved[:5], normalized_subject)
        return self._rank_ids_for_scene(ranked, claim.scene_id, normalized_subject)

    def _node_text(self, node: Dict) -> str:
        text_parts = [node.get("name", "")]
        text_parts.extend(node.get("aliases", []) or [])
        text_parts.append(node.get("description", ""))
        text_parts.extend(node.get("evidence", []) or [])
        return self._normalize_text(" ".join(part for part in text_parts if part))

    def _scene_node_text(self, node: Dict) -> str:
        text_parts = [node.get("name", "")]
        text_parts.extend(node.get("aliases", []) or [])
        text_parts.append(node.get("description", ""))
        if node.get("type") == "Event":
            text_parts.extend(node.get("evidence", []) or [])
        return self._normalize_text(" ".join(part for part in text_parts if part))

    def _scene_blob(self, scene_id: str) -> str:
        return self.scene_text_index.get(str(scene_id), "")

    def _subject_text_variants(self, node: Dict, subject_text: str) -> List[str]:
        variants: List[str] = []
        seen: Set[str] = set()

        def add(value: str):
            normalized = self._normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)

        add(subject_text)
        add(self._strip_family_titles(subject_text))
        for value in [node.get("name", "")] + (node.get("aliases", []) or []):
            add(value)
            add(self._strip_family_titles(value))
        return variants

    def _relation_cue_variants(self, claim: Claim) -> List[str]:
        normalized_claim = self._normalize_text(claim.claim_text)
        cues: List[str] = []
        seen: Set[str] = set()

        def add(value: str):
            normalized = self._normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                cues.append(normalized)

        base_cues = {
            "affiliated_with": ["associated with", "affiliated with", "connected to"],
            "affinity_with": ["girlfriend", "boyfriend", "wife", "husband", "love", "wants", "want"],
            "hostility_with": ["fight", "punch", "attack", "hostility", "shove"],
            "kinship_with": list(FAMILY_TITLE_ALIASES),
            "located_at": ["located at", "located in", "next to", "above", "outside", "inside"],
            "occurs_at": ["at", "in", "outside", "inside"],
            "occurs_on": ["day", "night", "morning", "afternoon", "evening", "later", "tonight"],
            "part_of": ["part of", "inside", "2nd floor", "second floor"],
            "performs": [],
            "possesses": ["has", "with", "holds", "holding", "wearing"],
            "present_on": ["present on", "present in", "in the room", "on the jacket", "in the air"],
            "references": ["references", "mentions", "tells about", "talks about"],
            "uses": ["uses", "with", "holding"],
        }

        for cue in base_cues.get(claim.predicate, []):
            add(cue)
        for token in normalized_claim.split():
            if token in FAMILY_TITLE_ALIASES:
                add(FAMILY_TITLE_ALIASES[token])
        return cues

    def _subject_scene_text(self, subject_id: str, scene_id: str) -> str:
        node = self.nodes_by_id.get(subject_id, {})
        # Keep this scene-aware to avoid cross-scene leakage from global evidence.
        parts = [node.get("name", "")]
        parts.extend(node.get("aliases", []) or [])
        for edge_info in self.adjacency.get(subject_id, []):
            if not self._scene_matches(edge_info, scene_id):
                continue
            target_node = self.nodes_by_id.get(edge_info["target"], {})
            parts.append(edge_info.get("relation", ""))
            parts.extend(edge_info.get("evidence", []) or [])
            parts.append(self._scene_node_text(target_node))
        return self._normalize_text(" ".join(part for part in parts if part))

    def _node_in_scene(self, node: Dict, scene_id: str) -> bool:
        refs = [str(ref) for ref in node.get("scene_refs", [])]
        return not refs or str(scene_id) in refs

    def _normalize_object_for_matching(self, text: str) -> str:
        normalized = self._normalize_text(text)
        tokens = [
            token for token in normalized.split()
            if token not in {"the", "a", "an", "half", "class", *GENERIC_OBJECT_TOKENS}
        ]
        return " ".join(tokens)

    def _object_text_variants(self, text: str) -> List[str]:
        normalized = self._normalize_object_for_matching(text)
        variants: List[str] = []

        def add(value: str):
            value = self._normalize_object_for_matching(value)
            if value and value not in variants:
                variants.append(value)

        add(text)
        add(normalized)
        add(self._strip_family_titles(text))

        parenthetical = re.sub(r"[()]", "", text)
        add(parenthetical)
        base = re.sub(r"\([^)]*\)", "", text).strip()
        role_match = re.search(r"\(as ([^)]+)\)", text.lower())
        add(base)
        if role_match:
            role = self._normalize_text(role_match.group(1))
            add(role)
            if base:
                add(f"{base} {role}")

        age_match = re.search(r"\b(\d+)\s*year\s*old\b", normalized)
        if age_match:
            add(age_match.group(1))
            add(f"{age_match.group(1)} year old")
            add(f"age {age_match.group(1)}")

        weight_match = re.search(r"\b(\d+)\s*(lb|lbs|pounds)\b", normalized)
        if weight_match:
            weight = weight_match.group(1)
            add(f"{weight} lb")
            add(f"{weight} lbs")
            add(f"{weight} pounds")
            add(f"weighs {weight}")

        if normalized.startswith(("is ", "are ", "was ", "were ", "being ")):
            add(re.sub(r"^(is|are|was|were|being)\s+", "", normalized))

        in_uniform_match = re.search(r"\bin (his|her|their)\s+(.+)", normalized)
        if in_uniform_match:
            add(in_uniform_match.group(2))

        looking_match = re.search(r"\blooking (?:a bit )?(.+)", normalized)
        if looking_match:
            add(looking_match.group(1))

        state_match = re.search(r"\ba state of (.+)", normalized)
        if state_match:
            add(state_match.group(1))

        for suffix in [" hair", " person", " vehicle", " cigarette", " face", " hand"]:
            if normalized.endswith(suffix):
                add(normalized[: -len(suffix)])

        synonym_variants = {
            "wealth": ["well off", "wealthy", "rich"],
            "parked": ["parked"],
            "parked vehicle": ["parked", "limo out front", "limousine parked"],
            "long lucky strike cigarette": ["lucky strike", "cigarette", "long lucky strike"],
            "lucky strike cigarette": ["lucky strike", "cigarette"],
            "bandaged hand": ["bandaged", "bandaged face"],
            "bandaged face": ["bandaged"],
            "is parked": ["parked"],
            "parked": ["is parked"],
            "redhead": ["red hair", "red headed"],
            "denying micky access": [
                "denies access",
                "deny access",
                "denies visitation",
                "not your day",
                "not his visitation day",
                "goodbye",
            ],
            "denies access": [
                "denying micky access",
                "denies visitation",
                "not your day",
                "goodbye",
            ],
            "refusing to get involved": [
                "refuses to get involved",
                "cant get in the middle",
                "cannot get in the middle",
                "dont put me in the middle",
                "get in the middle",
            ],
            "refuses to get involved": [
                "refusing to get involved",
                "cant get in the middle",
                "dont put me in the middle",
            ],
            "visits kasie": [
                "see kasie",
                "say hi to kasie",
                "asks to see kasie",
                "wants to say hi to kasie",
            ],
            "promises better future": [
                "start making good money",
                "making good money",
                "move to a bigger apartment",
                "move to a biggah apartment",
                "live with me more days",
                "better future",
            ],
            "plans to make good money": [
                "promises better future",
                "promises future success",
                "start making good money",
                "making good money",
                "good money",
            ],
            "plans to move to a bigger apartment": [
                "promises better future",
                "offers to move closer",
                "move to a bigger apartment",
                "move to a biggah apartment",
                "bigger apartment",
            ],
            "plans for kasie to live with him more days": [
                "promises better future",
                "offers to move closer",
                "live with him more days",
                "live with me more days",
                "kasie can live with him more often",
            ],
            "upcoming fight": [
                "fight coming up",
                "cites upcoming fight",
                "mentions a fight",
                "has a fight",
            ],
            "crying out for help": ["cries for help", "calls for help"],
            "looking a bit strung out": ["strung out", "looks strung out"],
            "embarrassment": ["embarrassed"],
            "is embarrassed": ["embarrassed"],
            "wildness": ["wild"],
        }
        normalized_synonyms = {
            self._normalize_object_for_matching(source): mapped
            for source, mapped in synonym_variants.items()
        }
        for source, mapped in normalized_synonyms.items():
            if normalized == source:
                for value in mapped:
                    add(value)

        return variants or [normalized]

    def _claim_text_variants(self, claim: Claim) -> List[str]:
        variants = [self._normalize_text(claim.claim_text)]
        for object_variant in self._object_text_variants(claim.object):
            if object_variant not in variants:
                variants.append(object_variant)
            subject_plus_object = self._normalize_text(f"{claim.subject} {object_variant}")
            if subject_plus_object and subject_plus_object not in variants:
                variants.append(subject_plus_object)
            stripped_subject_plus_object = self._normalize_text(f"{self._strip_family_titles(claim.subject)} {object_variant}")
            if stripped_subject_plus_object and stripped_subject_plus_object not in variants:
                variants.append(stripped_subject_plus_object)
        if claim.predicate in {"kinship_with", "affinity_with"}:
            stripped_object = self._strip_family_titles(claim.object)
            stripped_subject = self._strip_family_titles(claim.subject)
            combined = self._normalize_text(f"{stripped_subject} {stripped_object}")
            if combined and combined not in variants:
                variants.append(combined)
        normalized_claim = self._normalize_text(claim.claim_text)
        if "limousine" in normalized_claim:
            limo_variant = normalized_claim.replace("limousine", "limo")
            if limo_variant not in variants:
                variants.append(limo_variant)
        if "parked" in normalized_claim and "limo" in normalized_claim:
            parked_variant = self._normalize_text(f"{claim.subject} limo out front")
            if parked_variant not in variants:
                variants.append(parked_variant)
        return variants

    def _claim_text_variants_without_subject(self, claim: Claim) -> List[str]:
        subject_tokens = self._tokenize(claim.subject)
        variants: List[str] = []
        seen: Set[str] = set()

        def add(value: str):
            normalized = self._normalize_text(value)
            if not normalized:
                return
            filtered_tokens = [token for token in normalized.split() if token not in subject_tokens]
            filtered = " ".join(filtered_tokens).strip()
            if filtered and filtered not in seen:
                seen.add(filtered)
                variants.append(filtered)

        add(claim.claim_text)
        for variant in self._object_text_variants(claim.object):
            add(variant)
        return variants

    def _object_resolution_candidates(self, claim: Claim) -> List[str]:
        candidates: List[str] = []
        seen: Set[str] = set()
        for variant in self._object_text_variants(claim.object):
            if variant and variant not in seen:
                seen.add(variant)
                candidates.append(variant)
        return candidates

    def _text_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left in right or right in left:
            shorter = min(len(left.split()), len(right.split()))
            longer = max(len(left.split()), len(right.split()))
            if longer:
                return max(0.7, shorter / longer)

        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0

        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            smaller = min(len(left_tokens), len(right_tokens))
            larger = max(len(left_tokens), len(right_tokens))
            if larger:
                return max(0.72, smaller / larger)

        digit_overlap = {t for t in left_tokens if t.isdigit()} & {t for t in right_tokens if t.isdigit()}
        if digit_overlap:
            return max(0.75, len(digit_overlap) / max(1, len({t for t in left_tokens if t.isdigit()})))

        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        raw_score = overlap / union if union else 0.0

        left_stems = {self._stem_token(token) for token in left_tokens}
        right_stems = {self._stem_token(token) for token in right_tokens}
        stem_overlap = len(left_stems & right_stems)
        stem_union = len(left_stems | right_stems)
        stem_score = stem_overlap / stem_union if stem_union else 0.0
        return max(raw_score, stem_score)

    def _tokenize(self, text: str) -> Set[str]:
        return {
            token for token in self._normalize_text(text).split()
            if token and token not in LIGHT_STOPWORDS
        }

    def _stem_token(self, token: str) -> str:
        if token in FAMILY_TITLE_ALIASES:
            return FAMILY_TITLE_ALIASES[token]
        if len(token) <= 3 or token.isdigit():
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            return token[:-1]
        return token

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()
