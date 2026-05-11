#!/usr/bin/env python3
"""CLI for unified KG/text evaluation."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from stage_kg.evaluation.config import EvaluationConfig
from stage_kg.evaluation.experiment_runner import EvaluationRunner, load_scene_text

logger = logging.getLogger(__name__)


def setup_eval_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    ):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hallucination, consistency, and repetition evaluation for one KG/text pair."
    )
    parser.add_argument("--graph_path", required=True, help="Path to final_graph.json")
    parser.add_argument(
        "--text_path",
        required=True,
        help="Path to .txt or .json scene text. JSON can be {scene_id: text} or nested {scene_id: {character_id: text}}.",
    )
    parser.add_argument("--output_dir", default="eval_results", help="Directory for summary/detail outputs")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "anthropic"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api_key", default=None, help="Optional API key; otherwise uses provider env var")
    parser.add_argument("--base_url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--cache_dir", default=None, help="Optional claim extraction cache directory")
    parser.add_argument("--max_hop_depth", type=int, default=3)
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_path = setup_eval_logging(output_dir)
    logger.info("Eval log: %s", log_path)

    config = EvaluationConfig()
    config.max_hop_depth = args.max_hop_depth
    config.similarity_threshold = args.similarity_threshold

    scene_text = load_scene_text(args.text_path)
    runner = EvaluationRunner(
        graph_path=args.graph_path,
        output_dir=output_dir,
        config=config,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
    )
    results = runner.run(scene_text)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(json.dumps(results["metrics"], indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
