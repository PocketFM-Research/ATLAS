"""Orchestration and reporting for evaluation experiments."""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import asdict
import logging

from stage_kg.ingest.loader import load_movie
from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.claim_verifier import KnowledgeGraphVerifier
from stage_kg.evaluation.hallucination_eval import HallucinationEvaluator, HallucinationMetrics
from stage_kg.evaluation.new_claim_extractor import ClaimExtractor
from stage_kg.evaluation.consistency_eval import ConsistencyEvaluator, ConsistencyMetrics
from stage_kg.evaluation.repetition_eval import RepetitionEvaluator, RepetitionMetrics
from stage_kg.evaluation.scene_text import (
    LowEvidenceCharacterScene,
    SceneTextBuildStats,
    ScreenplaySceneTextBuilder,
)

logger = logging.getLogger(__name__)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return path.name


class EvaluationExperiment:
    """Run complete evaluation experiment on graph-to-text generation."""
    
    def __init__(
        self,
        movie_ids: List[str],
        graph_dir: str,
        output_dir: str,
        config: EvaluationConfig = None,
        text_generator=None,
        dataset_dir: Optional[str] = None,
        language: Optional[str] = None,
        text_source: str = "scene_script",
        force_rebuild_descriptions: bool = False,
        scene_descriptions_path: Optional[str] = None,
        hallucination_claim_extractor: Optional[ClaimExtractor] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            movie_ids: List of movie IDs to evaluate
            graph_dir: Directory containing final_graph.json files
            output_dir: Directory to save results
            config: Evaluation configuration
            text_generator: Optional text generation function(graph, scene_id, character_id) -> str
            dataset_dir: Dataset root containing English/ and Chinese/ folders
            language: Optional dataset language override
            text_source: One of "scene_script" or "generator"
            force_rebuild_descriptions: Ignore cached descriptions and rebuild them
            scene_descriptions_path: Optional JSON file with precomputed scene text
        """
        self.movie_ids = movie_ids
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.config = config or EvaluationConfig()
        self.text_generator = text_generator
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path.cwd()
        self.language = language
        self.text_source = text_source
        self.force_rebuild_descriptions = force_rebuild_descriptions
        self.scene_descriptions_path = (
            Path(scene_descriptions_path) if scene_descriptions_path else None
        )
        self.scene_text_builder = ScreenplaySceneTextBuilder()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if hallucination_claim_extractor is not None:
            self.hallucination_claim_extractor = hallucination_claim_extractor
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                default_key_file = Path("gemini.txt")
                if default_key_file.exists():
                    api_key = default_key_file.read_text(encoding="utf-8").strip() or None
            self.hallucination_claim_extractor = (
                ClaimExtractor.for_gemini(api_key=api_key) if api_key else None
            )
        
        # Initialize evaluators
        self.hallucination_eval = HallucinationEvaluator(
            claim_extractor=self.hallucination_claim_extractor,
            config=config,
        )
        self.consistency_eval = ConsistencyEvaluator(config)
        self.repetition_eval = RepetitionEvaluator(config)
    
    def run(self) -> Dict:
        """Run complete evaluation pipeline."""
        results = {
            "metadata": {
                "text_source": self.text_source,
                "dataset_dir": _relpath(self.dataset_dir, Path.cwd()),
                "language": self.language,
                "scene_descriptions_path": (
                    _relpath(self.scene_descriptions_path, Path.cwd()) if self.scene_descriptions_path else None
                ),
            },
            "movies": {},
            "global_metrics": {
                "hallucination": None,
                "consistency": None,
                "repetition": None,
            }
        }
        
        all_hallucination_results = []
        all_hallucination_abstentions = []
        all_low_confidence_claims = []
        all_consistency_violations = []
        all_repetition_instances = []
        all_consistency_metrics = []
        all_low_evidence_entries = []
        
        for movie_id in self.movie_ids:
            logger.info(f"Evaluating movie: {movie_id}")
            
            # Find graph file
            graph_path = self.graph_dir / movie_id / "final_graph.json"
            if not graph_path.exists():
                logger.warning(f"Graph not found: {graph_path}")
                continue
            
            # Load graph
            verifier = KnowledgeGraphVerifier(str(graph_path), self.config)
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            
            # Generate text or load from cache
            scene_descriptions, scene_text_stats, low_evidence_entries = self._get_scene_descriptions(
                movie_id, graph_path, graph_data
            )
            all_low_evidence_entries.extend(low_evidence_entries)
            
            # Metric 1: Hallucination
            hal_metrics, hal_results, hal_abstentions, hal_low_confidence = self._eval_hallucination(
                scene_descriptions, verifier
            )
            all_hallucination_results.extend(hal_results)
            all_hallucination_abstentions.extend(hal_abstentions)
            all_low_confidence_claims.extend(hal_low_confidence)
            
            # Metric 2: Consistency
            cons_metrics, cons_violations = self._eval_consistency(scene_descriptions)
            all_consistency_metrics.append(cons_metrics)
            all_consistency_violations.extend(cons_violations)
            
            # Metric 3: Repetition
            rep_metrics, rep_instances = self._eval_repetition(scene_descriptions)
            all_repetition_instances.extend(rep_instances)
            
            results["movies"][movie_id] = {
                "graph_path": _relpath(graph_path, self.graph_dir),
                "scene_text_stats": scene_text_stats.to_dict(),
                "low_evidence_count": len(low_evidence_entries),
                "low_evidence_entries": [entry.to_dict() for entry in low_evidence_entries],
                "hallucination": asdict(hal_metrics),
                "hallucination_abstentions": len(hal_abstentions),
                "hallucination_low_confidence": len(hal_low_confidence),
                "consistency": asdict(cons_metrics),
                "repetition": asdict(rep_metrics),
            }
        
        # Compute global metrics
        results["global_metrics"]["hallucination"] = self._aggregate_hallucination(
            all_hallucination_results,
            all_hallucination_abstentions,
            all_low_confidence_claims,
        )
        results["global_metrics"]["consistency"] = self._aggregate_consistency(
            all_consistency_violations, all_consistency_metrics
        )
        results["global_metrics"]["repetition"] = self._aggregate_repetition(
            all_repetition_instances
        )
        results["global_metrics"]["low_evidence"] = {
            "total_entries": len(all_low_evidence_entries),
        }
        
        # Save results
        self._save_results(
            results,
            all_hallucination_results,
            all_hallucination_abstentions,
            all_low_confidence_claims,
            all_consistency_violations,
            all_repetition_instances,
            all_low_evidence_entries,
        )
        
        return results
    
    def _get_scene_descriptions(
        self,
        movie_id: str,
        graph_path: Path,
        graph_data: Dict,
    ) -> 'Tuple[Dict[str, Dict[str, str]], SceneTextBuildStats, List[LowEvidenceCharacterScene]]':
        """
        Get scene descriptions. Generate if needed, else load from cache.
        
        Structure: {scene_id: {character_id: description_text}}
        """
        if self.scene_descriptions_path:
            descriptions = self._load_external_scene_descriptions(movie_id)
            return descriptions, SceneTextBuildStats(), []

        cache_stem = f"{movie_id}_{self.text_source}_descriptions"
        cache_path = self.output_dir / f"{cache_stem}.json"
        stats_path = self.output_dir / f"{cache_stem}_stats.json"
        
        if cache_path.exists() and not self.force_rebuild_descriptions:
            with open(cache_path, encoding="utf-8") as f:
                cached_descriptions = json.load(f)

            if stats_path.exists():
                with open(stats_path, encoding="utf-8") as f:
                    cached_stats = SceneTextBuildStats(**json.load(f))
            else:
                cached_stats = SceneTextBuildStats()

            low_evidence_path = self.output_dir / f"{cache_stem}_low_evidence.json"
            if low_evidence_path.exists():
                with open(low_evidence_path, encoding="utf-8") as f:
                    raw_entries = json.load(f)
                low_evidence_entries = [
                    LowEvidenceCharacterScene(**entry) for entry in raw_entries
                ]
            else:
                low_evidence_entries = []

            return cached_descriptions, cached_stats, low_evidence_entries

        if self.text_source == "scene_script":
            descriptions, stats, low_evidence_entries = self._build_scene_script_descriptions(
                movie_id, graph_data
            )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats.to_dict(), f, indent=2)
            low_evidence_path = self.output_dir / f"{cache_stem}_low_evidence.json"
            with open(low_evidence_path, "w", encoding="utf-8") as f:
                json.dump(
                    [entry.to_dict() for entry in low_evidence_entries],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return descriptions, stats, low_evidence_entries
        
        if not self.text_generator:
            logger.warning(f"No text generator provided; skipping {movie_id}")
            return {}, SceneTextBuildStats(), []
        
        # Generate descriptions
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
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(scene_descriptions, f, indent=2, ensure_ascii=False)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(SceneTextBuildStats().to_dict(), f, indent=2)
        
        return scene_descriptions, SceneTextBuildStats(), []

    def _build_scene_script_descriptions(
        self,
        movie_id: str,
        graph_data: Dict,
    ) -> 'Tuple[Dict[str, Dict[str, str]], SceneTextBuildStats, List[LowEvidenceCharacterScene]]':
        """Build evaluation text from screenplay scenes instead of dummy summaries."""
        language = self.language or self._infer_language(movie_id)
        language_dir = "English" if language == "en" else "Chinese"
        movie_dir = self.dataset_dir / language_dir / movie_id

        if not movie_dir.exists():
            logger.warning("Movie directory not found for screenplay text: %s", movie_dir)
            return {}, SceneTextBuildStats(), []

        movie = load_movie(movie_dir=movie_dir, movie_id=movie_id, language=language)
        return self.scene_text_builder.build(movie, graph_data)

    def _load_external_scene_descriptions(self, movie_id: str) -> Dict[str, Dict[str, str]]:
        """Load precomputed scene descriptions from JSON."""
        if not self.scene_descriptions_path or not self.scene_descriptions_path.exists():
            logger.warning("Scene descriptions JSON not found: %s", self.scene_descriptions_path)
            return {}

        with open(self.scene_descriptions_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            logger.warning("Scene descriptions JSON must be a dictionary at the top level.")
            return {}

        if movie_id in payload and isinstance(payload[movie_id], dict):
            return payload[movie_id]
        if all(isinstance(value, dict) for value in payload.values()):
            return payload

        logger.warning(
            "Scene descriptions JSON has an unexpected format; expected `{scene_id: {character_id: text}}` or `{movie_id: ...}`"
        )
        return {}

    def _infer_language(self, movie_id: str) -> str:
        """Infer dataset language from the movie id prefix."""
        return "zh" if movie_id.lower().startswith("ch") else "en"
    
    def _eval_hallucination(
        self,
        scene_descriptions: Dict[str, Dict[str, str]],
        verifier: KnowledgeGraphVerifier
    ) -> 'Tuple[HallucinationMetrics, List, List, List]':
        """Evaluate hallucination across all scenes."""
        all_results = []
        all_abstentions = []
        all_low_confidence_claims = []
        seen_scene_texts = set()
        
        for scene_id, characters in scene_descriptions.items():
            for char_id, text in characters.items():
                normalized_text = text.strip()
                if not normalized_text:
                    continue

                dedupe_key = (scene_id, normalized_text)
                if dedupe_key in seen_scene_texts:
                    logger.debug(
                        "Skipping duplicate hallucination text for scene=%s character=%s",
                        scene_id,
                        char_id,
                    )
                    continue
                seen_scene_texts.add(dedupe_key)

                _, results, abstentions, low_confidence_claims = self.hallucination_eval.evaluate_scene(
                    normalized_text, scene_id, verifier
                )
                all_results.extend(results)
                all_abstentions.extend(abstentions)
                all_low_confidence_claims.extend(low_confidence_claims)
        
        metrics = self.hallucination_eval._compute_metrics(
            all_results,
            all_abstentions,
            all_low_confidence_claims,
        )
        return metrics, all_results, all_abstentions, all_low_confidence_claims
    
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
    
    def _aggregate_hallucination(self, all_results, all_abstentions, all_low_confidence_claims) -> Dict:
        """Aggregate hallucination metrics across all movies."""
        if not all_results and not all_abstentions and not all_low_confidence_claims:
            return {
                "total_claims": 0,
                "unscorable_claims": 0,
                "low_confidence_claims": 0,
                "hallucination_rate": 0.0,
            }
        
        metrics = self.hallucination_eval._compute_metrics(
            all_results,
            all_abstentions,
            all_low_confidence_claims,
        )
        return asdict(metrics)
    
    def _aggregate_consistency(self, violations, metrics_list) -> Dict:
        """Aggregate consistency metrics across all movies."""
        total_possible = sum(
            metric.total_possible_comparisons for metric in metrics_list
        )
        return {
            "total_violations": len(violations),
            "total_possible_comparisons": total_possible,
            "contradiction_rate": (
                len(violations) / max(1, total_possible) if total_possible else 0.0
            ),
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
        hal_abstentions,
        hal_low_confidence_claims,
        cons_violations,
        rep_instances,
        low_evidence_entries,
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
            str(self.output_dir / "hallucination_details.csv"),
            abstentions=hal_abstentions,
            abstentions_output_path=str(self.output_dir / "hallucination_abstentions.csv"),
            low_confidence_claims=hal_low_confidence_claims,
            low_confidence_output_path=str(self.output_dir / "hallucination_low_confidence.csv"),
        )
        self.consistency_eval.save_violations(
            cons_violations,
            str(self.output_dir / "consistency_violations.csv")
        )
        self.repetition_eval.save_instances(
            rep_instances,
            str(self.output_dir / "repetition_instances.csv")
        )
        self._save_low_evidence_entries(
            low_evidence_entries,
            str(self.output_dir / "low_evidence_character_scenes.csv")
        )
        
        logger.info(f"Saved detailed results to {self.output_dir}")

    def _save_low_evidence_entries(
        self,
        entries: List[LowEvidenceCharacterScene],
        output_path: str,
    ):
        """Save low-evidence character/scene pairs to CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scene_id",
                    "character_id",
                    "character_name",
                    "message",
                    "reason",
                    "script_snippet_hits",
                    "graph_evidence_hits",
                ],
            )
            writer.writeheader()

            for entry in entries:
                writer.writerow(entry.to_dict())
