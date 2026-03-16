#!/usr/bin/env python3
"""CLI for running evaluation experiments."""

import argparse
import logging
import sys
from pathlib import Path
import json

# Configure logging BEFORE imports
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from stage_kg.evaluation.experiment_runner import EvaluationExperiment
from stage_kg.evaluation.config import EvaluationConfig


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
            break
    
    if events:
        return f"{char_name} {', '.join(events)} in scene {scene_id}."
    else:
        return f"{char_name} appears in scene {scene_id}."


def main():
    print("\n" + "="*60)
    print("GRAPH-TO-TEXT EVALUATION EXPERIMENT")
    print("="*60 + "\n", flush=True)
    
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
        "--output_dir",
        default="eval_results",
        help="Output directory for results (default: eval_results)"
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
    
    print(f"Configuration:", flush=True)
    print(f"  Movies: {', '.join(args.movie_ids)}", flush=True)
    print(f"  Graph dir: {args.graph_dir}", flush=True)
    print(f"  Output dir: {args.output_dir}", flush=True)
    print(f"  Max hop depth: {args.max_hop_depth}", flush=True)
    print(f"  Similarity threshold: {args.similarity_threshold}\n", flush=True)
    
    # Create config
    config = EvaluationConfig()
    config.max_hop_depth = args.max_hop_depth
    config.similarity_threshold = args.similarity_threshold
    
    try:
        # Run experiment
        print("Initializing experiment...", flush=True)
        experiment = EvaluationExperiment(
            movie_ids=args.movie_ids,
            graph_dir=args.graph_dir,
            output_dir=args.output_dir,
            config=config,
            text_generator=dummy_text_generator
        )
        
        print("Starting evaluation experiment...\n", flush=True)
        results = experiment.run()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60 + "\n", flush=True)
        
        global_metrics = results["global_metrics"]
        
        if global_metrics.get("hallucination"):
            hal = global_metrics["hallucination"]
            print(f"HALLUCINATION METRICS:", flush=True)
            print(f"  Total claims: {hal.get('total_claims', 'N/A')}", flush=True)
            print(f"  Hallucination rate: {hal.get('hallucination_rate', 0):.2%}", flush=True)
            print(f"  Grounding rate: {hal.get('grounding_rate', 0):.2%}", flush=True)
        
        if global_metrics.get("consistency"):
            cons = global_metrics["consistency"]
            print(f"\nCONSISTENCY METRICS:", flush=True)
            print(f"  Total contradictions: {cons.get('total_violations', 0)}", flush=True)
            print(f"  Contradiction rate: {cons.get('contradiction_rate', 0):.2%}", flush=True)
        
        if global_metrics.get("repetition"):
            rep = global_metrics["repetition"]
            print(f"\nREPETITION METRICS:", flush=True)
            print(f"  Total scene pairs: {rep.get('total_pairs', 0)}", flush=True)
            print(f"  Repetition rate: {rep.get('repetition_rate', 0):.2%}", flush=True)
            if rep.get('avg_semantic_similarity'):
                print(f"  Avg semantic similarity: {rep.get('avg_semantic_similarity', 0):.3f}", flush=True)
        
        print(f"\n{'='*60}")
        print(f"✓ Results saved to: {args.output_dir}", flush=True)
        print(f"  - summary.json (aggregated metrics)", flush=True)
        print(f"  - hallucination_details.csv (per-claim verification)", flush=True)
        print(f"  - consistency_violations.csv (contradictions)", flush=True)
        print(f"  - repetition_instances.csv (semantic duplicates)", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        return 0
    
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}\n", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
