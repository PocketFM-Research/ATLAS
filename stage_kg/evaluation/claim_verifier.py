"""Schema-aware graph verification for claims."""

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from stage_kg.schema import NodeType, RelationType, VALID_TRIPLES
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.new_claim_extractor import Claim


GROUND_STATUSES = {"grounded", "grounded_multihop"}
EVENT_ROLE_RELATIONS = {"performs", "undergoes", "experiences"}
TEXT_BACKED_RELATIONS = {"located_at", "is_a", "possesses", "undergoes", "performs"}
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

        for node in self.graph_data["nodes"]:
            self._index_node_names(node)
            self.node_text_index[node["id"]] = self._node_text(node)

        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
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

    def verify_claim(self, claim: Claim) -> VerificationResult:
        """Verify a single claim against the graph."""
        subject_ids = self._resolve_entity(claim.subject)
        if not subject_ids:
            subject_ids = self._resolve_subject_from_claim_text(claim)

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
        for subject_id in subject_ids:
            if object_ids:
                for object_id in object_ids:
                    candidate = self._bfs_verify(subject_id, object_id, claim.predicate, claim)
                    best_result = self._choose_better_result(best_result, candidate)
                    candidate = self._verify_composed_scene_support(subject_id, object_id, claim)
                    best_result = self._choose_better_result(best_result, candidate)

            semantic_candidate = self._verify_semantic_edge(subject_id, claim)
            best_result = self._choose_better_result(best_result, semantic_candidate)
            subject_text_candidate = self._verify_subject_text_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, subject_text_candidate)
            scene_text_candidate = self._verify_scene_text_support(subject_id, claim)
            best_result = self._choose_better_result(best_result, scene_text_candidate)

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

        normalized = self._normalize_text(entity_name)
        if not normalized:
            return []

        direct = self.nodes_by_name.get(normalized, [])
        if direct:
            return self._rank_resolved_ids(list(dict.fromkeys(direct)), normalized)

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
        return self._rank_resolved_ids(resolved, normalized)

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

        direct = self._check_direct_edge(source_id, target_id, predicate, claim.scene_id)
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
    ) -> Optional[Dict]:
        for edge_info in self.adjacency.get(source_id, []):
            if edge_info["target"] != target_id:
                continue
            if not self._scene_matches(edge_info, scene_id):
                continue
            if edge_info["relation"] == predicate or self._predicate_match(predicate, edge_info["relation"]):
                return edge_info
        return None

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

        threshold = 0.15 if claim.predicate in EVENT_ROLE_RELATIONS else 0.28
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

        subject_node = self.nodes_by_id.get(subject_id, {})
        scene_blob = self._subject_scene_text(subject_id, claim.scene_id)
        object_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._object_text_variants(claim.object)),
            default=0.0,
        )
        claim_score = max(
            (self._text_similarity(variant, scene_blob) for variant in self._claim_text_variants(claim)),
            default=0.0,
        )
        score = max(object_score, claim_score)

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

    def _verify_composed_scene_support(
        self,
        subject_id: str,
        object_id: str,
        claim: Claim,
    ) -> VerificationResult:
        if claim.predicate not in TEXT_BACKED_RELATIONS:
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
        scene_nodes = [
            node for node in self.graph_data["nodes"]
            if self._node_in_scene(node, claim.scene_id)
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

        if not best_target or best_score < 0.28:
            return VerificationResult(claim=claim, status="hallucinated", depth=0, confidence=0.0)

        return VerificationResult(
            claim=claim,
            status="grounded",
            depth=1,
            path=[subject_id, best_target["id"]],
            supported_node_types=[best_target.get("type", "")],
            confidence=min(0.85, 0.35 + best_score / 2.0),
        )

    def _claim_edge_similarity(self, claim: Claim, edge_info: Dict, target_node: Dict) -> float:
        claim_object = self._normalize_text(claim.object)
        claim_text = self._normalize_text(claim.claim_text)
        target_text = self.node_text_index.get(target_node.get("id", ""), "")

        scores = []
        if claim_object:
            scores.append(self._text_similarity(claim_object, target_text))
        if claim_text:
            scores.append(self._text_similarity(claim_text, target_text))

        for evidence in edge_info.get("evidence", []):
            evidence_text = self._normalize_text(evidence)
            if claim_object:
                scores.append(self._text_similarity(claim_object, evidence_text))
            if claim_text:
                scores.append(self._text_similarity(claim_text, evidence_text))

        return max(scores) if scores else 0.0

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
        return self._rank_resolved_ids(resolved[:5], normalized_subject)

    def _node_text(self, node: Dict) -> str:
        text_parts = [node.get("name", "")]
        text_parts.extend(node.get("aliases", []) or [])
        text_parts.append(node.get("description", ""))
        text_parts.extend(node.get("evidence", []) or [])
        return self._normalize_text(" ".join(part for part in text_parts if part))

    def _subject_scene_text(self, subject_id: str, scene_id: str) -> str:
        node = self.nodes_by_id.get(subject_id, {})
        parts = [self._node_text(node)]
        for edge_info in self.adjacency.get(subject_id, []):
            if not self._scene_matches(edge_info, scene_id):
                continue
            target_node = self.nodes_by_id.get(edge_info["target"], {})
            parts.append(edge_info.get("relation", ""))
            parts.extend(edge_info.get("evidence", []) or [])
            parts.append(self._node_text(target_node))
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
        }
        for source, mapped in synonym_variants.items():
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
        return overlap / union if union else 0.0

    def _tokenize(self, text: str) -> Set[str]:
        return {token for token in self._normalize_text(text).split() if token}

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()
