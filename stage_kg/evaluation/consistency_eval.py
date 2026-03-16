"""Self-consistency evaluation (Metric 2)."""

import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from stage_kg.evaluation.claim_extractor import ClaimExtractor, Claim
from stage_kg.evaluation.config import EvaluationConfig


@dataclass
class ConsistencyMetrics:
    """Self-consistency metrics."""
    total_character_pairs: int
    contradictions: int
    state_changes_without_events: int
    contradiction_rate: float
    
    by_character: Dict[str, Dict] = None  # character_id -> {contradictions, drift_instances}
    
    def to_dict(self):
        return {
            "total_character_pairs": self.total_character_pairs,
            "contradictions": self.contradictions,
            "state_changes_without_events": self.state_changes_without_events,
            "contradiction_rate": self.contradiction_rate,
            "by_character": self.by_character or {},
        }


@dataclass
class ConsistencyViolation:
    """A detected consistency violation."""
    character_id: str
    scene_t: str
    scene_tk: str
    claim_t: str
    claim_tk: str
    violation_type: str  # "contradiction", "state_change_without_event", "attribute_mismatch"
    severity: float  # 0-1
    explanation: str


class ConsistencyEvaluator:
    """Evaluate self-consistency across scenes."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.claim_extractor = ClaimExtractor()
    
    def evaluate_consistency(
        self,
        scene_descriptions: Dict[str, Dict[str, str]]
    ) -> 'Tuple[ConsistencyMetrics, List[ConsistencyViolation]]':
        """
        Evaluate consistency across scene descriptions.
        
        Args:
            scene_descriptions: {scene_id: {character_id: description_text}}
            
        Returns: (ConsistencyMetrics, list of violations)
        """
        violations = []
        
        # Extract all claims by character and scene
        claims_by_char_scene = {}
        
        for scene_id, characters in scene_descriptions.items():
            for char_id, description in characters.items():
                claims = self.claim_extractor.extract_claims(
                    description, scene_id, char_id
                )
                
                key = (char_id, scene_id)
                claims_by_char_scene[key] = claims
        
        # Check consistency across scenes for each character
        by_character = {}
        scene_ids = sorted(set(s for _, s in claims_by_char_scene.keys()))
        
        for char_id in set(c for c, _ in claims_by_char_scene.keys()):
            char_violations = []
            
            # Compare each pair of scenes (t, t+k)
            for i, scene_t in enumerate(scene_ids):
                for scene_tk in scene_ids[i+1:]:
                    key_t = (char_id, scene_t)
                    key_tk = (char_id, scene_tk)
                    
                    claims_t = claims_by_char_scene.get(key_t, [])
                    claims_tk = claims_by_char_scene.get(key_tk, [])
                    
                    # Check for contradictions
                    pair_violations = self._check_contradiction(
                        char_id, scene_t, scene_tk, claims_t, claims_tk
                    )
                    char_violations.extend(pair_violations)
                    violations.extend(pair_violations)
            
            by_character[char_id] = {
                "contradictions": len(char_violations),
                "drift_instances": sum(1 for v in char_violations 
                                       if v.violation_type == "state_change_without_event"),
            }
        
        # Compute metrics
        total_pairs = len(scene_ids) * (len(scene_ids) - 1) // 2 if scene_ids else 0
        total_possible = total_pairs * len(by_character) if total_pairs > 0 else 0
        metrics = ConsistencyMetrics(
            total_character_pairs=total_pairs,
            contradictions=len(violations),
            state_changes_without_events=sum(
                1 for v in violations if v.violation_type == "state_change_without_event"
            ),
            contradiction_rate=len(violations) / max(1, total_possible),
            by_character=by_character,
        )
        
        return metrics, violations
    
    def _check_contradiction(
        self,
        char_id: str,
        scene_t: str,
        scene_tk: str,
        claims_t: List[Claim],
        claims_tk: List[Claim]
    ) -> 'List[ConsistencyViolation]':
        """Check pairwise claims for contradictions."""
        violations = []
        
        for claim_t in claims_t:
            for claim_tk in claims_tk:
                # Simple heuristic: check if predicates are opposing
                if self._are_contradicting(claim_t, claim_tk):
                    violations.append(ConsistencyViolation(
                        character_id=char_id,
                        scene_t=scene_t,
                        scene_tk=scene_tk,
                        claim_t=claim_t.claim_text,
                        claim_tk=claim_tk.claim_text,
                        violation_type="contradiction",
                        severity=0.8,
                        explanation=f"Contradicting predicates: '{claim_t.predicate}' vs '{claim_tk.predicate}'"
                    ))
        
        return violations
    
    def _are_contradicting(self, claim1: Claim, claim2: Claim) -> bool:
        """Check if two claims directly contradict."""
        # Opposite predicates
        opposites = {
            "alive": "dead",
            "present": "absent",
            "together": "separated",
            "hostile": "friendly",
        }
        
        for key, val in opposites.items():
            if (key in claim1.predicate and val in claim2.predicate) or \
               (val in claim1.predicate and key in claim2.predicate):
                # Same subject and object
                if claim1.subject == claim2.subject and claim1.object == claim2.object:
                    return True
        
        return False
    
    def save_violations(
        self,
        violations: List[ConsistencyViolation],
        output_path: str
    ):
        """Save violations to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "character_id", "scene_t", "scene_tk", "claim_t", "claim_tk",
                "violation_type", "severity", "explanation"
            ])
            writer.writeheader()
            
            for v in violations:
                writer.writerow({
                    "character_id": v.character_id,
                    "scene_t": v.scene_t,
                    "scene_tk": v.scene_tk,
                    "claim_t": v.claim_t,
                    "claim_tk": v.claim_tk,
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "explanation": v.explanation,
                })