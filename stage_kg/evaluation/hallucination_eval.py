"""Hallucination evaluation (Metric 1)."""

import csv
from typing import List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

from stage_kg.evaluation.claim_extractor import (
    Claim,
    ClaimExtractor,
    ExtractionAbstention,
    LowConfidenceClaim,
)
from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier, VerificationResult
from stage_kg.evaluation.config import EvaluationConfig


@dataclass
class HallucinationMetrics:
    """Aggregated hallucination metrics."""
    total_claims: int
    unscorable_claims: int
    low_confidence_claims: int
    grounded_claims: int
    grounded_multihop_claims: int
    partial_claims: int
    hallucinated_claims: int
    contradictions: int
    
    hallucination_rate: float  # (hallucinated + contradiction) / total
    grounding_rate: float  # (grounded + grounded_multihop) / total
    
    by_relation_type: Dict[str, Dict] = None  # relation_type -> {rate, count, weight}
    by_character: Dict[str, Dict] = None  # character_id -> {rate, count, claims}
    
    def to_dict(self):
        return {
            "total_claims": self.total_claims,
            "unscorable_claims": self.unscorable_claims,
            "low_confidence_claims": self.low_confidence_claims,
            "grounded_claims": self.grounded_claims,
            "grounded_multihop_claims": self.grounded_multihop_claims,
            "partial_claims": self.partial_claims,
            "hallucinated_claims": self.hallucinated_claims,
            "contradictions": self.contradictions,
            "hallucination_rate": self.hallucination_rate,
            "grounding_rate": self.grounding_rate,
            "by_relation_type": self.by_relation_type or {},
            "by_character": self.by_character or {},
        }


class HallucinationEvaluator:
    """Evaluate hallucination rate per scene/character."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.claim_extractor = ClaimExtractor()
    
    def evaluate_scene(
        self,
        generated_text: str,
        scene_id: str,
        character_id: str,
        verifier: KnowledgeGraphVerifier
    ) -> 'Tuple[HallucinationMetrics, List[VerificationResult], List[ExtractionAbstention], List[LowConfidenceClaim]]':
        """
        Evaluate hallucination for generated text of a scene.
        
        Returns: (HallucinationMetrics, list of verification results)
        """
        # Extract claims
        claims, abstentions, low_confidence_claims = self.claim_extractor.extract_claims_with_quality(
            generated_text, scene_id, character_id
        )
        claims = self._filter_claims_for_character(claims, character_id, verifier)
        low_confidence_claims = self._filter_low_confidence_for_character(
            low_confidence_claims, character_id, verifier
        )
        
        # Verify each claim
        results = []
        for claim in claims:
            result = verifier.verify_claim(claim)
            results.append(result)
        
        # Compute metrics
        metrics = self._compute_metrics(results, abstentions, low_confidence_claims)
        
        return metrics, results, abstentions, low_confidence_claims
    
    def _compute_metrics(
        self,
        results: List[VerificationResult],
        abstentions: List[ExtractionAbstention] = None,
        low_confidence_claims: List[LowConfidenceClaim] = None,
    ) -> HallucinationMetrics:
        """Compute aggregated hallucination metrics from verification results."""
        abstentions = abstentions or []
        low_confidence_claims = low_confidence_claims or []
        if not results:
            return HallucinationMetrics(
                total_claims=0,
                unscorable_claims=len(abstentions),
                low_confidence_claims=len(low_confidence_claims),
                grounded_claims=0,
                grounded_multihop_claims=0,
                partial_claims=0,
                hallucinated_claims=0,
                contradictions=0,
                hallucination_rate=0.0,
                grounding_rate=0.0,
                by_relation_type={},
                by_character={},
            )
        
        total = len(results)
        grounded = sum(1 for r in results if r.status == "grounded")
        grounded_multihop = sum(1 for r in results if r.status == "grounded_multihop")
        partial = sum(1 for r in results if r.status == "partial")
        hallucinated = sum(1 for r in results if r.status == "hallucinated")
        contradictions = sum(1 for r in results if r.status == "contradiction")
        
        hallucination_rate = (hallucinated + contradictions) / total if total > 0 else 0.0
        grounding_rate = (grounded + grounded_multihop) / total if total > 0 else 0.0
        
        # Aggregate by relation type
        by_relation_type = {}
        for result in results:
            rel_type = self.config.get_relation_group(result.claim.predicate)
            if rel_type not in by_relation_type:
                by_relation_type[rel_type] = {
                    "grounded": 0,
                    "hallucinated": 0,
                    "count": 0,
                    "weight": self.config.get_weight(result.claim.predicate)
                }
            by_relation_type[rel_type]["count"] += 1
            if result.status in ["grounded", "grounded_multihop"]:
                by_relation_type[rel_type]["grounded"] += 1
            elif result.status in ["hallucinated", "contradiction"]:
                by_relation_type[rel_type]["hallucinated"] += 1
        
        # Compute rate per relation type
        for rel in by_relation_type.values():
            rel["rate"] = rel["hallucinated"] / rel["count"] if rel["count"] > 0 else 0.0
        
        metrics = HallucinationMetrics(
            total_claims=total,
            unscorable_claims=len(abstentions),
            low_confidence_claims=len(low_confidence_claims),
            grounded_claims=grounded,
            grounded_multihop_claims=grounded_multihop,
            partial_claims=partial,
            hallucinated_claims=hallucinated,
            contradictions=contradictions,
            hallucination_rate=hallucination_rate,
            grounding_rate=grounding_rate,
            by_relation_type=by_relation_type,
            by_character={},
        )
        
        return metrics

    def _filter_claims_for_character(
        self,
        claims: List[Claim],
        character_id: str,
        verifier: KnowledgeGraphVerifier,
    ) -> List[Claim]:
        """Keep only claims that are plausibly centered on the requested character."""
        if character_id not in verifier.nodes_by_id:
            return claims

        character_node = verifier.nodes_by_id[character_id]
        character_names = [character_node.get("name", "")]
        character_names.extend(character_node.get("aliases", []) or [])
        character_names = [
            verifier._normalize_text(name) for name in character_names if verifier._normalize_text(name)
        ]

        filtered = []
        for claim in claims:
            if claim.subject == character_id:
                filtered.append(claim)
                continue

            subject_ids = verifier._resolve_entity(claim.subject)
            if character_id in subject_ids:
                filtered.append(claim)
                continue

            if subject_ids:
                if any(
                    verifier.nodes_by_id.get(subject_id, {}).get("type") == "Character"
                    for subject_id in subject_ids
                ):
                    continue
                filtered.append(claim)
                continue

            normalized_subject = verifier._normalize_text(claim.subject)
            if any(verifier._entity_match_score(normalized_subject, name) >= 0.75 for name in character_names):
                filtered.append(claim)
                continue

            if normalized_subject:
                filtered.append(claim)

        return filtered

    def _filter_low_confidence_for_character(
        self,
        claims: List[LowConfidenceClaim],
        character_id: str,
        verifier: KnowledgeGraphVerifier,
    ) -> List[LowConfidenceClaim]:
        """Apply the same character-centric filtering to low-confidence claims."""
        pseudo_claims = [
            Claim(
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.object,
                claim_text=claim.claim_text,
                scene_id=claim.scene_id,
                character_id=claim.character_id,
                sentence_idx=claim.sentence_idx,
            )
            for claim in claims
        ]
        kept_claims = self._filter_claims_for_character(pseudo_claims, character_id, verifier)
        kept_keys = {
            (claim.subject, claim.predicate, claim.object, claim.sentence_idx)
            for claim in kept_claims
        }
        filtered = [
            claim for claim in claims
            if (claim.subject, claim.predicate, claim.object, claim.sentence_idx) in kept_keys
        ]
        deduped = []
        seen = set()
        for claim in filtered:
            key = (claim.scene_id, claim.character_id, claim.subject, claim.predicate, claim.object)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)
        return deduped
    
    def save_results(
        self,
        results: List[VerificationResult],
        output_path: str,
        abstentions: List[ExtractionAbstention] = None,
        abstentions_output_path: str = None,
        low_confidence_claims: List[LowConfidenceClaim] = None,
        low_confidence_output_path: str = None,
    ):
        """Save detailed verification results to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene_id", "character_id", "claim_text", "subject", "predicate", "object",
                "status", "depth", "confidence", "relation_type", "path", "supported_types"
            ])
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    "scene_id": result.claim.scene_id,
                    "character_id": result.claim.character_id,
                    "claim_text": result.claim.claim_text,
                    "subject": result.claim.subject,
                    "predicate": result.claim.predicate,
                    "object": result.claim.object,
                    "status": result.status,
                    "depth": result.depth,
                    "confidence": result.confidence,
                    "relation_type": self.config.get_relation_group(result.claim.predicate),
                    "path": " -> ".join(result.path),
                    "supported_types": ",".join(result.supported_node_types),
                })

        if abstentions is not None and abstentions_output_path:
            with open(abstentions_output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scene_id",
                        "character_id",
                        "sentence_idx",
                        "text",
                        "reason",
                        "speaker",
                        "context_subject",
                    ],
                )
                writer.writeheader()
                for abstention in abstentions:
                    writer.writerow(abstention.to_dict())

        if low_confidence_claims is not None and low_confidence_output_path:
            with open(low_confidence_output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scene_id",
                        "character_id",
                        "sentence_idx",
                        "claim_text",
                        "subject",
                        "predicate",
                        "object",
                        "score",
                        "reasons",
                    ],
                )
                writer.writeheader()
                for claim in low_confidence_claims:
                    writer.writerow(claim.to_dict())
