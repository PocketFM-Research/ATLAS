"""Hallucination evaluation (Metric 1)."""

import csv
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from stage_kg.evaluation.llm_claim_extractor import (
    Claim,
    ClaimExtractor,
    ExtractionAbstention,
    LowConfidenceClaim,
    assess_claim_quality,
    should_abstain_low_confidence_claim,
)
from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier, VerificationResult
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.utils.cache import Cache


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
    """Evaluate hallucination rate for generated scene text."""
    
    def __init__(
        self,
        claim_extractor: Optional[ClaimExtractor] = None,
        config: EvaluationConfig = None,
        cache: Optional[Cache] = None,
        cache_namespace: str = "hallucination_eval",
    ):
        self.config = config or EvaluationConfig()
        self.claim_extractor = claim_extractor
        self.cache = cache
        self.cache_namespace = cache_namespace
        self._claim_cache: Dict[Tuple[str, str], List[Claim]] = {}
    
    def evaluate_scene(
        self,
        generated_text: str,
        scene_id: str,
        verifier: KnowledgeGraphVerifier
    ) -> 'Tuple[HallucinationMetrics, List[VerificationResult], List[Any], List[Any]]':
        """
        Evaluate hallucination for generated text of a scene.
        
        Returns: (HallucinationMetrics, list of verification results)
        """
        if self.claim_extractor is None:
            raise ValueError("HallucinationEvaluator requires a claim-centric ClaimExtractor instance.")

        cache_key = (scene_id, generated_text.strip())
        claims = self._claim_cache.get(cache_key)
        if claims is None:
            claims = self.claim_extractor.extract_claims(
                generated_text,
                scene_id,
                movie_id=self.cache_namespace,
                cache=self.cache,
            )
            self._claim_cache[cache_key] = claims
        claims, abstentions, low_confidence_claims = self._partition_claims(claims, use_low_confidence=True)
        
        # Verify each claim
        results = []
        for claim in claims:
            result = verifier.verify_claim(claim)
            results.append(result)
        results, demoted_low_confidence = self._demote_low_confidence_hallucinations(results)
        low_confidence_claims.extend(demoted_low_confidence)
        
        # Compute metrics
        metrics = self._compute_metrics(results, abstentions, low_confidence_claims)
        
        return metrics, results, abstentions, low_confidence_claims

    def evaluate_claims(
        self,
        claims: List[Claim],
        verifier: KnowledgeGraphVerifier,
    ) -> 'Tuple[HallucinationMetrics, List[VerificationResult], List[Any], List[Any]]':
        """Evaluate a precomputed list of claims without re-running extraction."""
        claims, abstentions, low_confidence_claims = self._partition_claims(
            claims,
            use_low_confidence=False,
            treat_low_confidence_as_abstain=True,
        )

        results = [verifier.verify_claim(claim) for claim in claims]
        results, demoted_low_confidence = self._demote_low_confidence_hallucinations(results)
        low_confidence_claims.extend(demoted_low_confidence)
        metrics = self._compute_metrics(results, abstentions, low_confidence_claims)
        return metrics, results, abstentions, low_confidence_claims

    def _demote_low_confidence_hallucinations(
        self,
        results: List[VerificationResult],
    ) -> Tuple[List[VerificationResult], List[LowConfidenceClaim]]:
        threshold = float(getattr(self.config, "hallucination_min_scored_confidence", 0.955))
        kept_results: List[VerificationResult] = []
        low_confidence: List[LowConfidenceClaim] = []
        for result in results:
            if result.status == "hallucinated" and result.claim.confidence < threshold:
                low_confidence.append(
                    LowConfidenceClaim(
                        subject=result.claim.subject,
                        predicate=result.claim.predicate,
                        object=result.claim.object,
                        claim_text=result.claim.claim_text,
                        scene_id=result.claim.scene_id,
                        sentence_idx=result.claim.sentence_idx,
                        score=result.claim.confidence,
                        reasons=["low_extractor_confidence_hallucination"],
                    )
                )
                continue
            kept_results.append(result)
        return kept_results, low_confidence

    def _partition_claims(
        self,
        claims: List[Claim],
        use_low_confidence: bool,
        treat_low_confidence_as_abstain: bool = False,
    ) -> Tuple[List[Claim], List[ExtractionAbstention], List[LowConfidenceClaim]]:
        """Split claims into scorable, abstained, and low-confidence buckets."""
        scorable: List[Claim] = []
        abstentions: List[ExtractionAbstention] = []
        low_confidence_claims: List[LowConfidenceClaim] = []

        for claim in claims:
            decision, score, reasons = assess_claim_quality(claim)
            if decision == "keep":
                scorable.append(claim)
                continue
            if decision == "low_confidence":
                if treat_low_confidence_as_abstain and should_abstain_low_confidence_claim(claim, reasons):
                    abstentions.append(
                        ExtractionAbstention(
                            text=claim.claim_text,
                            scene_id=claim.scene_id,
                            sentence_idx=claim.sentence_idx,
                            reason=";".join(reasons) if reasons else "low_confidence_claim",
                            context_subject=claim.subject,
                        )
                    )
                    continue
                if not use_low_confidence:
                    scorable.append(claim)
                    continue
                low_confidence_claims.append(
                    LowConfidenceClaim(
                        subject=claim.subject,
                        predicate=claim.predicate,
                        object=claim.object,
                        claim_text=claim.claim_text,
                        scene_id=claim.scene_id,
                        sentence_idx=claim.sentence_idx,
                        score=score,
                        reasons=reasons,
                    )
                )
                continue
            abstentions.append(
                ExtractionAbstention(
                    text=claim.claim_text,
                    scene_id=claim.scene_id,
                    sentence_idx=claim.sentence_idx,
                    reason=";".join(reasons) if reasons else "low_quality_claim",
                    context_subject=claim.subject,
                )
            )

        return scorable, abstentions, low_confidence_claims
    
    def _compute_metrics(
        self,
        results: List[VerificationResult],
        abstentions: List[Any] = None,
        low_confidence_claims: List[Any] = None,
    ) -> HallucinationMetrics:
        """Compute aggregated hallucination metrics from verification results."""
        abstentions = abstentions or []
        low_confidence_claims = low_confidence_claims or []
        result_unscorable = [r for r in results if r.status == "unscorable"]
        scored_results = [r for r in results if r.status != "unscorable"]
        if not scored_results:
            return HallucinationMetrics(
                total_claims=0,
                unscorable_claims=len(abstentions) + len(result_unscorable),
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
        
        total = len(scored_results)
        grounded = sum(1 for r in scored_results if r.status == "grounded")
        grounded_multihop = sum(1 for r in scored_results if r.status == "grounded_multihop")
        partial = sum(1 for r in scored_results if r.status == "partial")
        hallucinated = sum(1 for r in scored_results if r.status == "hallucinated")
        contradictions = sum(1 for r in scored_results if r.status == "contradiction")
        
        hallucination_rate = (hallucinated + contradictions) / total if total > 0 else 0.0
        grounding_rate = (grounded + grounded_multihop) / total if total > 0 else 0.0
        
        # Aggregate by relation type
        by_relation_type = {}
        for result in scored_results:
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
            unscorable_claims=len(abstentions) + len(result_unscorable),
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

    def save_results(
        self,
        results: List[VerificationResult],
        output_path: str,
        abstentions: List[Any] = None,
        abstentions_output_path: str = None,
        low_confidence_claims: List[Any] = None,
        low_confidence_output_path: str = None,
    ):
        """Save detailed verification results to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene_id", "claim_text", "subject", "predicate", "object",
                "status", "depth", "confidence", "relation_type", "path", "supported_types"
            ])
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    "scene_id": result.claim.scene_id,
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
                        "sentence_idx",
                        "text",
                        "reason",
                        "speaker",
                        "context_subject",
                    ],
                )
                writer.writeheader()
                for abstention in abstentions:
                    writer.writerow(
                        {
                            "scene_id": abstention.scene_id,
                            "sentence_idx": abstention.sentence_idx,
                            "text": abstention.text,
                            "reason": abstention.reason,
                            "speaker": abstention.speaker,
                            "context_subject": abstention.context_subject,
                        }
                    )

        if low_confidence_claims is not None and low_confidence_output_path:
            with open(low_confidence_output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scene_id",
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
                    writer.writerow(
                        {
                            "scene_id": claim.scene_id,
                            "sentence_idx": claim.sentence_idx,
                            "claim_text": claim.claim_text,
                            "subject": claim.subject,
                            "predicate": claim.predicate,
                            "object": claim.object,
                            "score": claim.score,
                            "reasons": ";".join(claim.reasons),
                        }
                    )
