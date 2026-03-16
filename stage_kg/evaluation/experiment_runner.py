"""Orchestration and reporting for evaluation experiments."""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import asdict
import logging

from stage_kg.ingest.loader import load_movie
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.claim_extractor import ClaimExtractor
from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier
from stage_kg.evaluation.hallucination_eval import HallucinationEvaluator, HallucinationMetrics
from stage_kg.evaluation.consistency_eval import ConsistencyEvaluator, ConsistencyMetrics
from stage_kg.evaluation.repetition_eval import RepetitionEvaluator, RepetitionMetrics

logger = logging.getLogger(__name__)


class EvaluationExperiment:
    """Run complete evaluation experiment on graph-to-text generation."""
    
    def __init__(
        self,
        movie_ids: List[str],
        graph_dir: str,
        output_dir: str,
        config: EvaluationConfig = None,
        text_generator = None
    ):
        """
        Initialize experiment.
        
        Args:
            movie_ids: List of movie IDs to evaluate
            graph_dir: Directory containing final_graph.json files
            output_dir: Directory to save results
            config: Evaluation configuration
            text_generator: Optional text generation function(graph, scene_id, character_id) -> str
        """
        self.movie_ids = movie_ids
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.config = config or EvaluationConfig()
        self.text_generator = text_generator
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluators
        self.hallucination_eval = HallucinationEvaluator(config)
        self.consistency_eval = ConsistencyEvaluator(config)
        self.repetition_eval = RepetitionEvaluator(config)
    
    def run(self) -> Dict:
        """Run complete evaluation pipeline."""
        results = {
            "movies": {},
            "global_metrics": {
                "hallucination": None,
                "consistency": None,
                "repetition": None,
            }
        }
        
        all_hallucination_results = []
        all_consistency_violations = []
        all_repetition_instances = []
        
        for movie_id in self.movie_ids:
            logger.info(f"Evaluating movie: {movie_id}")
            
            # Find graph file
            graph_path = self.graph_dir / movie_id / "final_graph.json"
            if not graph_path.exists():
                logger.warning(f"Graph not found: {graph_path}")
                continue
            
            # Load graph
            verifier = KnowledgeGraphVerifier(str(graph_path), self.config)
            
            # Generate text or load from cache
            scene_descriptions = self._get_scene_descriptions(movie_id, graph_path)
            
            # Metric 1: Hallucination
            hal_metrics, hal_results = self._eval_hallucination(
                scene_descriptions, verifier
            )
            all_hallucination_results.extend(hal_results)
            
            # Metric 2: Consistency
            cons_metrics, cons_violations = self._eval_consistency(scene_descriptions)
            all_consistency_violations.extend(cons_violations)
            
            # Metric 3: Repetition
            rep_metrics, rep_instances = self._eval_repetition(scene_descriptions)
            all_repetition_instances.extend(rep_instances)
            
            results["movies"][movie_id] = {
                "hallucination": asdict(hal_metrics),
                "consistency": asdict(cons_metrics),
                "repetition": asdict(rep_metrics),
            }
        
        # Compute global metrics
        results["global_metrics"]["hallucination"] = self._aggregate_hallucination(
            all_hallucination_results
        )
        results["global_metrics"]["consistency"] = self._aggregate_consistency(
            all_consistency_violations
        )
        results["global_metrics"]["repetition"] = self._aggregate_repetition(
            all_repetition_instances
        )
        
        # Save results
        self._save_results(results, all_hallucination_results, all_consistency_violations, all_repetition_instances)
        
        return results
    
    def _get_scene_descriptions(
        self,
        movie_id: str,
        graph_path: Path
    ) -> Dict[str, Dict[str, str]]:
        """
        Get scene descriptions. Generate if needed, else load from cache.
        
        Structure: {scene_id: {character_id: description_text}}
        """
        cache_path = self.output_dir / f"{movie_id}_descriptions.json"
        
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        
        if not self.text_generator:
            logger.warning(f"No text generator provided; skipping {movie_id}")
            return {}
        
        # Generate descriptions
        import json as json_module
        with open(graph_path) as f:
            graph_data = json_module.load(f)
        
        scene_descriptions = {}
        
        for node in graph_data["nodes"]:
            if node["type"] != "Character":
                continue
            
            char_id = node["id"]
            for scene_id in node.get("scene_refs", []):
                if scene_id not in scene_descriptions:
                    scene_descriptions[scene_id] = {}
                
                # Generate description
                text = self.text_generator(graph_data, scene_id, char_id)
                scene_descriptions[scene_id][char_id] = text
        
        # Cache
        with open(cache_path, 'w') as f:
            json_module.dump(scene_descriptions, f, indent=2)
        
        return scene_descriptions
    
    def _eval_hallucination(
        self,
        scene_descriptions: Dict[str, Dict[str, str]],
        verifier: KnowledgeGraphVerifier
    ) -> 'Tuple[HallucinationMetrics, List]':
        """Evaluate hallucination across all scenes."""
        all_results = []
        
        for scene_id, characters in scene_descriptions.items():
            for char_id, text in characters.items():
                _, results = self.hallucination_eval.evaluate_scene(
                    text, scene_id, char_id, verifier
                )
                all_results.extend(results)
        
        metrics = self.hallucination_eval._compute_metrics(all_results)
        return metrics, all_results
    
    def _eval_consistency(
        self,
        scene_descriptions: Dict[str, Dict[str, str]]
    ) -> 'Tuple[ConsistencyMetrics, List]':
        """Evaluate consistency across scenes."""
        metrics, violations = self.consistency_eval.evaluate_consistency(scene_descriptions)
        return metrics, violations
    
    def _eval_repetition(
        self,
        scene_descriptions: Dict[str, Dict[str, str]]
    ) -> 'Tuple[RepetitionMetrics, List]':
        """Evaluate repetition across scenes."""
        metrics, instances = self.repetition_eval.evaluate_repetition(scene_descriptions)
        return metrics, instances
    
    def _aggregate_hallucination(self, all_results) -> Dict:
        """Aggregate hallucination metrics across all movies."""
        if not all_results:
            return {"total_claims": 0, "hallucination_rate": 0.0}
        
        metrics = self.hallucination_eval._compute_metrics(all_results)
        return asdict(metrics)
    
    def _aggregate_consistency(self, violations) -> Dict:
        """Aggregate consistency metrics across all movies."""
        return {
            "total_violations": len(violations),
            "contradiction_rate": len(violations) / max(1, len(violations)) if violations else 0.0,
        }
    
    def _aggregate_repetition(self, instances) -> Dict:
        """Aggregate repetition metrics across all movies."""
        if not instances:
            return {"total_pairs": 0, "repetition_rate": 0.0}
        
        repetitive = sum(1 for inst in instances if inst.is_repetitive)
        return {
            "total_pairs": len(instances),
            "repetitive_pairs": repetitive,
            "repetition_rate": repetitive / len(instances) if instances else 0.0,
            "avg_semantic_similarity": sum(inst.semantic_similarity for inst in instances) / len(instances),
        }
    
    def _save_results(
        self,
        results: Dict,
        hal_results,
        cons_violations,
        rep_instances
    ):
        """Save all results to output directory."""
        # Summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
        # Detailed CSVs
        self.hallucination_eval.save_results(
            hal_results,
            str(self.output_dir / "hallucination_details.csv")
        )
        self.consistency_eval.save_violations(
            cons_violations,
            str(self.output_dir / "consistency_violations.csv")
        )
        self.repetition_eval.save_instances(
            rep_instances,
            str(self.output_dir / "repetition_instances.csv")
        )
        
        logger.info(f"Saved detailed results to {self.output_dir}")