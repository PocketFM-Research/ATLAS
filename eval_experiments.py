#!/usr/bin/env python3
"""CLI for running evaluation experiments."""

import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

from stage_kg.evaluation.experiment_runner import EvaluationExperiment
from stage_kg.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)


def setup_eval_logging(output_dir: Path) -> Path:
    """Log eval runs both to console and to an output-local log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers when rerun in the same interpreter.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(file_handler)
    return log_path


def dummy_text_generator(graph_data: dict, scene_id: str, character_id: str) -> str:
    """
    Dummy text generator for demonstration.
    
    In practice, replace with LLM-based or templating generation.
    """
    # Find character node
    char_name = None
    for node in graph_data.get("nodes", []):
        if node["id"] == character_id:
            char_name = node.get("name", "The character")
            break
    
    if not char_name:
        return "No description available."
    
    # Find events this character participated in
    events = []
    for edge in graph_data.get("edges", []):
        if edge["source"] == character_id and edge["relation"] in ["performs", "experiences", "undergoes"]:
            # Find event node
            for node in graph_data["nodes"]:
                if node["id"] == edge["target"] and node["type"] == "Event":
                    if scene_id in node.get("scene_refs", []):
                        events.append(node.get("name", "an event"))
    
    if events:
        return f"{char_name} {', '.join(events)} in scene {scene_id}."
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Run graph-to-text evaluation experiments"
    )
    parser.add_argument(
        "--movie_ids",
        nargs="+",
        required=True,
        help="Movie IDs to evaluate"
    )
    parser.add_argument(
        "--graph_dir",
        default="output",
        help="Directory containing final_graph.json files (default: output)"
    )
    parser.add_argument(
        "--dataset_dir",
        default=".",
        help="Dataset root containing English/ and Chinese/ directories (default: current directory)"
    )
    parser.add_argument(
        "--output_dir",
        default="eval_results",
        help="Output directory for results (default: eval_results)"
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh"],
        default=None,
        help="Optional dataset language override. If omitted, inferred from movie id."
    )
    parser.add_argument(
        "--text_source",
        choices=["scene_script", "generator"],
        default="scene_script",
        help="Where evaluation text should come from (default: scene_script)"
    )
    parser.add_argument(
        "--scene_descriptions_json",
        default=None,
        help="Optional JSON file containing precomputed `{scene_id: {character_id: text}}` descriptions"
    )
    parser.add_argument(
        "--refresh_descriptions",
        action="store_true",
        help="Rebuild scene description caches instead of reusing previous outputs"
    )
    parser.add_argument(
        "--max_hop_depth",
        type=int,
        default=3,
        help="Maximum BFS depth for claim verification (default: 3)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for repetition detection (default: 0.85)"
    )
    
    args = parser.parse_args()
    log_path = setup_eval_logging(Path(args.output_dir))
    logger.info("Eval log: %s", log_path)
    
    # Create config
    config = EvaluationConfig()
    config.max_hop_depth = args.max_hop_depth
    config.similarity_threshold = args.similarity_threshold
    
    # Run experiment
    text_generator = dummy_text_generator if args.text_source == "generator" else None
    experiment = EvaluationExperiment(
        movie_ids=args.movie_ids,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        config=config,
        text_generator=text_generator,
        dataset_dir=args.dataset_dir,
        language=args.language,
        text_source=args.text_source,
        force_rebuild_descriptions=args.refresh_descriptions,
        scene_descriptions_path=args.scene_descriptions_json,
    )
    
    logger.info("Starting evaluation experiment...")
    results = experiment.run()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    global_metrics = results["global_metrics"]
    
    if global_metrics["hallucination"]:
        logger.info(f"\nHallucination Rate: {global_metrics['hallucination'].get('hallucination_rate', 'N/A'):.2%}")
        logger.info(f"Grounding Rate: {global_metrics['hallucination'].get('grounding_rate', 'N/A'):.2%}")
    
    if global_metrics["consistency"]:
        logger.info(f"\nTotal Contradictions: {global_metrics['consistency'].get('total_violations', 0)}")
    
    if global_metrics["repetition"]:
        logger.info(f"\nRepetition Rate: {global_metrics['repetition'].get('repetition_rate', 'N/A'):.2%}")
        logger.info(f"Avg Semantic Similarity: {global_metrics['repetition'].get('avg_semantic_similarity', 'N/A'):.3f}")
    
    logger.info(f"\n✓ Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
