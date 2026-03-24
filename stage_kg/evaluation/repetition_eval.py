"""Repetition evaluation (Metric 3)."""

import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from stage_kg.evaluation.config import EvaluationConfig


@dataclass
class RepetitionMetrics:
    """Repetition metrics."""
    total_scene_pairs: int
    repetitive_pairs: int
    avg_semantic_similarity: float
    avg_ngram_overlap: float
    repetition_rate: float
    
    by_character: Dict[str, Dict] = None  # character_id -> {repetitions, avg_similarity}
    
    def to_dict(self):
        return {
            "total_scene_pairs": self.total_scene_pairs,
            "repetitive_pairs": self.repetitive_pairs,
            "avg_semantic_similarity": self.avg_semantic_similarity,
            "avg_ngram_overlap": self.avg_ngram_overlap,
            "repetition_rate": self.repetition_rate,
            "by_character": self.by_character or {},
        }


@dataclass
class RepetitionInstance:
    """A detected repetition."""
    character_id: str
    scene_i: str
    scene_j: str
    semantic_similarity: float
    ngram_overlap: float
    is_repetitive: bool
    sample_text_i: str
    sample_text_j: str


class RepetitionEvaluator:
    """Evaluate semantic repetition across scenes."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.embedding_model = None
        
        if HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer(
                    config.EMBEDDING_MODEL if hasattr(config, 'EMBEDDING_MODEL') else "all-MiniLM-L6-v2"
                )
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
    
    def evaluate_repetition(
        self,
        scene_descriptions: Dict[str, Dict[str, str]]
    ) -> 'Tuple[RepetitionMetrics, List[RepetitionInstance]]':
        """
        Evaluate repetition across scene descriptions.
        
        Args:
            scene_descriptions: {scene_id: {character_id: description_text}}
            
        Returns: (RepetitionMetrics, list of instances)
        """
        instances = []
        by_character = {}
        
        # For each character, compare descriptions across scenes
        characters = set()
        for scenes in scene_descriptions.values():
            characters.update(scenes.keys())
        
        for char_id in characters:
            char_instances = []
            scenes_with_char = [
                (scene_id, scene_descriptions[scene_id][char_id])
                for scene_id in sorted(scene_descriptions.keys())
                if char_id in scene_descriptions[scene_id]
            ]
            
            # Compare each pair of scenes
            for i, (scene_i, text_i) in enumerate(scenes_with_char):
                for scene_j, text_j in scenes_with_char[i+1:]:
                    # Compute similarity
                    sem_sim = self._semantic_similarity(text_i, text_j)
                    ngram_sim = self._ngram_overlap(text_i, text_j)
                    
                    is_rep = (
                        sem_sim >= self.config.similarity_threshold and
                        ngram_sim >= self.config.ngram_threshold
                    )
                    
                    instance = RepetitionInstance(
                        character_id=char_id,
                        scene_i=scene_i,
                        scene_j=scene_j,
                        semantic_similarity=sem_sim,
                        ngram_overlap=ngram_sim,
                        is_repetitive=is_rep,
                        sample_text_i=text_i[:100],
                        sample_text_j=text_j[:100],
                    )
                    
                    instances.append(instance)
                    if is_rep:
                        char_instances.append(instance)
            
            total_pairs_for_character = max(0, len(scenes_with_char) * (len(scenes_with_char) - 1) // 2)
            by_character[char_id] = {
                "repetitions": len(char_instances),
                "avg_similarity": (
                    sum(inst.semantic_similarity for inst in char_instances) / len(char_instances)
                    if char_instances else 0.0
                ),
                "pairs_compared": total_pairs_for_character,
            }
        
        repetitive_pairs = sum(1 for inst in instances if inst.is_repetitive)
        
        metrics = RepetitionMetrics(
            total_scene_pairs=len(instances),
            repetitive_pairs=repetitive_pairs,
            avg_semantic_similarity=sum(inst.semantic_similarity for inst in instances) / len(instances) if instances else 0.0,
            avg_ngram_overlap=sum(inst.ngram_overlap for inst in instances) / len(instances) if instances else 0.0,
            repetition_rate=repetitive_pairs / len(instances) if instances else 0.0,
            by_character=by_character,
        )
        
        return metrics, instances
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.embedding_model:
            return self._simple_similarity(text1, text2)
        
        try:
            emb1 = self.embedding_model.encode(text1, convert_to_tensor=False)
            emb2 = self.embedding_model.encode(text2, convert_to_tensor=False)
            
            # Cosine similarity
            import numpy as np
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        except Exception as e:
            print(f"Embedding error: {e}")
            return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Fallback: simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
    
    def _ngram_overlap(self, text1: str, text2: str, n: int = 2) -> float:
        """Compute n-gram overlap between texts."""
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        overlap = len(ngrams1 & ngrams2)
        total = len(ngrams1 | ngrams2)
        
        return overlap / total if total > 0 else 0.0
    
    def save_instances(
        self,
        instances: List[RepetitionInstance],
        output_path: str
    ):
        """Save repetition instances to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "character_id", "scene_i", "scene_j", "semantic_similarity",
                "ngram_overlap", "is_repetitive", "text_i_sample", "text_j_sample"
            ])
            writer.writeheader()
            
            for inst in instances:
                writer.writerow({
                    "character_id": inst.character_id,
                    "scene_i": inst.scene_i,
                    "scene_j": inst.scene_j,
                    "semantic_similarity": f"{inst.semantic_similarity:.3f}",
                    "ngram_overlap": f"{inst.ngram_overlap:.3f}",
                    "is_repetitive": inst.is_repetitive,
                    "text_i_sample": inst.sample_text_i,
                    "text_j_sample": inst.sample_text_j,
                })
