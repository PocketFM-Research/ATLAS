#!/usr/bin/env python3
"""
STAGE Knowledge Graph Builder — main CLI entry point.

Usage:
    python build_graph.py --input_dir /path/to/STAGE --output_dir ./output --model anthropic

    # Process a single movie by ID:
    python build_graph.py --input_dir . --output_dir ./output --model openai \
        --movie_ids en04052c0f20834cf1bac19927d8f758e0

    # Use a local vLLM endpoint:
    python build_graph.py --input_dir . --output_dir ./output \
        --model vllm --model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --base_url http://localhost:8000/v1

    # Use Azure OpenAI (GPT-5.4 mini):
    python build_graph.py --input_dir . --output_dir ./output \
        --model azure_openai --model_name gpt-5.4-mini \
        --api_key_file openai.txt --base_url_file base_url.txt

    # Skip LLM merge adjudication (faster):
    python build_graph.py --input_dir . --output_dir ./output --model anthropic \
        --skip_normalization

    # Process only first N scenes (for quick testing):
    python build_graph.py --input_dir . --output_dir ./output --model anthropic \
        --movie_ids en04052c0f20834cf1bac19927d8f758e0 --max_scenes 5

    # Build one graph per scene (per-scene mode):
    python build_graph.py --input_dir . --output_dir ./output --model azure_openai \
        --movie_ids my_story --per_scene
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from stage_kg.ingest.loader import load_movie, load_all_movies
from stage_kg.llm import get_llm
from stage_kg.pipeline import run_pipeline, run_pipeline_per_scene
from stage_kg.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build knowledge graphs from STAGE movie screenplays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("."),
        help="Root STAGE dataset directory (contains English/ and Chinese/ subdirs).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output"),
        help="Directory to write output graphs and intermediates.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["openai", "azure_openai", "anthropic", "gemini", "vllm"],
        help="LLM provider.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specific model name/ID (provider default if omitted).",
    )
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key string (overrides --api_key_file and env vars).",
    )
    p.add_argument(
        "--api_key_file",
        type=Path,
        default=None,
        help="Path to a file containing the API key (e.g. gemini.txt).",
    )
    p.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Override LLM endpoint URL (for vLLM / Azure).",
    )
    p.add_argument(
        "--base_url_file",
        type=Path,
        default=None,
        help="Path to a file containing the endpoint URL (e.g. base_url.txt).",
    )
    p.add_argument(
        "--api_version",
        type=str,
        default=None,
        help="Azure OpenAI API version (e.g. 2024-12-01-preview). "
             "Only required for plain Azure endpoints without /openai/deployments/ in URL.",
    )
    p.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Movie language to process.",
    )
    p.add_argument(
        "--movie_ids",
        nargs="+",
        default=None,
        help="Process only these movie IDs (space-separated). Processes all if omitted.",
    )
    p.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Limit scenes per movie (useful for quick tests).",
    )
    p.add_argument(
        "--skip_normalization",
        action="store_true",
        help="Skip LLM-based merge adjudication (faster, lower quality).",
    )
    p.add_argument(
        "--per_scene",
        "--scene_graphs",
        dest="per_scene",
        action="store_true",
        help="Build one independent graph per scene instead of one merged movie graph. "
             "Outputs to <output_dir>/<movie_id>/scene_graphs/scene_<N>/final_graph.json.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    args = parse_args()

    setup_logging(
        log_dir=args.output_dir / "logs",
        verbose=args.verbose,
    )
    logger = logging.getLogger(__name__)

    # Validate input directory
    if not args.input_dir.exists():
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    # Resolve API key: explicit string > key file > env var (handled inside each client)
    api_key = args.api_key
    if not api_key and args.api_key_file:
        api_key = _read_text_file(Path(args.api_key_file))

    # Resolve base URL similarly
    base_url = args.base_url
    if not base_url and args.base_url_file:
        base_url = _read_text_file(Path(args.base_url_file))

    # Build LLM backend
    logger.info("Initializing LLM backend: provider=%s model=%s", args.model, args.model_name)
    llm = get_llm(
        provider=args.model,
        model=args.model_name,
        api_key=api_key,
        base_url=base_url,
        api_version=args.api_version,
    )
    logger.info("LLM ready: %s", llm.model_id)

    # Load movies
    if args.movie_ids:
        # Load specified movies — auto-detect their directories
        movies = []
        for mid in args.movie_ids:
            # Try English then Chinese
            for lang, lang_dir in [("en", "English"), ("zh", "Chinese")]:
                movie_dir = args.input_dir / lang_dir / mid
                if movie_dir.exists():
                    from stage_kg.ingest.loader import load_movie
                    import csv
                    csv_map = {"en": "english_movie_info.csv", "zh": "chinese_movie_info.csv"}
                    csv_path = args.input_dir / csv_map[lang]
                    title = ""
                    if csv_path.exists():
                        with csv_path.open() as f:
                            for row in csv.DictReader(f):
                                if row["movie_id"] == mid:
                                    title = row.get("title", "")
                                    break
                    movie = load_movie(movie_dir, mid, title, lang)
                    movies.append(movie)
                    break
            else:
                logger.warning("Movie %s not found in English/ or Chinese/", mid)
    else:
        movies = load_all_movies(
            dataset_dir=args.input_dir,
            language=args.language,
            movie_ids=args.movie_ids,
        )

    if not movies:
        logger.error("No movies loaded. Check --input_dir and --language.")
        sys.exit(1)

    logger.info("Processing %d movie(s)", len(movies))

    # Process each movie
    failed = []
    for movie in movies:
        try:
            if args.max_scenes:
                movie.scenes = movie.scenes[: args.max_scenes]
                logger.info(
                    "Limiting to first %d scenes for %s", args.max_scenes, movie.movie_id
                )
            if args.per_scene:
                run_pipeline_per_scene(
                    movie=movie,
                    llm=llm,
                    output_dir=args.output_dir,
                    skip_normalization=args.skip_normalization,
                    api_key=api_key,
                )
            else:
                run_pipeline(
                    movie=movie,
                    llm=llm,
                    output_dir=args.output_dir,
                    skip_normalization=args.skip_normalization,
                    api_key=api_key,
                )
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", movie.movie_id, e, exc_info=True)
            failed.append(movie.movie_id)

    if failed:
        logger.warning("Failed movies: %s", failed)
    else:
        logger.info("All movies processed successfully.")


if __name__ == "__main__":
    main()
