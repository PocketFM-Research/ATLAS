#!/usr/bin/env python3
"""
Example end-to-end run on the Star Trek II movie (first 3 scenes).

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.

Usage:
    cd /home/ubuntu/STAGE
    ANTHROPIC_API_KEY=sk-... python scripts/run_example.py
    # or
    OPENAI_API_KEY=sk-... python scripts/run_example.py --provider openai
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage_kg.ingest.loader import load_movie
from stage_kg.llm import get_llm
from stage_kg.pipeline import run_pipeline
from stage_kg.utils.logging_utils import setup_logging


EXAMPLE_MOVIE_ID = "en04052c0f20834cf1bac19927d8f758e0"  # Star Trek II: The Wrath of Khan
EXAMPLE_TITLE = "Star Trek II: The Wrath of Khan"
DATASET_DIR = Path(__file__).parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="anthropic", choices=["openai", "anthropic", "vllm"])
    p.add_argument("--model_name", default=None)
    p.add_argument("--max_scenes", type=int, default=3)
    p.add_argument("--output_dir", type=Path, default=DATASET_DIR / "output")
    p.add_argument("--skip_normalization", action="store_true")
    args = p.parse_args()

    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)

    movie_dir = DATASET_DIR / "English" / EXAMPLE_MOVIE_ID
    if not movie_dir.exists():
        logger.error("Movie directory not found: %s", movie_dir)
        sys.exit(1)

    logger.info("Loading movie: %s", EXAMPLE_TITLE)
    movie = load_movie(movie_dir, EXAMPLE_MOVIE_ID, EXAMPLE_TITLE, language="en")
    movie.scenes = movie.scenes[: args.max_scenes]
    logger.info("Loaded %d scenes (limited to %d)", len(movie.scenes), args.max_scenes)

    logger.info("Initializing LLM: %s", args.provider)
    llm = get_llm(provider=args.provider, model=args.model_name)

    graph = run_pipeline(
        movie=movie,
        llm=llm,
        output_dir=args.output_dir,
        skip_normalization=args.skip_normalization,
    )

    out_path = args.output_dir / EXAMPLE_MOVIE_ID / "final_graph.json"
    logger.info("Example complete. Final graph: %s", out_path)
    logger.info("Nodes: %d | Edges: %d", len(graph.nodes), len(graph.edges))

    # Print a sample
    sample = {
        "movie_id": graph.movie_id,
        "node_sample": list(graph.nodes.values())[:3],
        "edge_sample": graph.edges[:3],
    }
    print("\n--- Sample output ---")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
