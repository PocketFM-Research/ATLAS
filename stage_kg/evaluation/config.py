"""Configuration for evaluation experiments."""

from typing import Dict, List

# Relation type groupings for heterogeneous weighting
RELATION_GROUPS = {
    "social": [
        "kinship_with",
        "affinity_with",
        "hostility_with",
        "affiliated_with",
    ],
    "event_role": [
        "performs",
        "undergoes",
        "experiences",
    ],
    "inter_event": [
        "precedes",
        "causes",
        "contrasts_with",
        "references",
    ],
    "spatiotemporal": [
        "occurs_at",
        "occurs_on",
        "located_at",
        "present_on",
    ],
    "object": [
        "possesses",
        "uses",
        "part_of",
        "is_a",
    ],
}

# Error weights by relation type (higher = stricter)
RELATION_WEIGHTS = {
    "social": 2.0,
    "event_role": 1.5,
    "inter_event": 1.5,
    "spatiotemporal": 1.0,
    "object": 1.0,
}

# Multi-hop verification parameters
MAX_HOP_DEPTH = 3
SIMILARITY_THRESHOLD_REPETITION = 0.85
NGRAM_OVERLAP_THRESHOLD = 0.3

# NLI model for consistency checking
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

# Sentence embedding model for similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class EvaluationConfig:
    """Configuration object for evaluation runs."""
    
    max_hop_depth: int = MAX_HOP_DEPTH
    similarity_threshold: float = SIMILARITY_THRESHOLD_REPETITION
    ngram_threshold: float = NGRAM_OVERLAP_THRESHOLD
    relation_groups: Dict[str, List[str]] = RELATION_GROUPS
    relation_weights: Dict[str, float] = RELATION_WEIGHTS
    
    def get_relation_group(self, relation: str) -> str:
        """Get the group (key) for a relation type."""
        for group, relations in self.relation_groups.items():
            if relation in relations:
                return group
        return "unknown"
    
    def get_weight(self, relation: str) -> float:
        """Get the weight for a relation type."""
        group = self.get_relation_group(relation)
        return self.relation_weights.get(group, 1.0)