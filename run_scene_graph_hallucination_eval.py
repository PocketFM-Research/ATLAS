#!/usr/bin/env python3
"""
End-to-end scene-graph temporal hallucination evaluation.

For each case:
  1. Load the movie/story.
  2. Build one knowledge graph per scene.
  3. Run the graph_verifier over ordered scene graphs.
  4. Run the LLM judge baseline (same Azure GPT-5.4 mini config).
  5. Compare both methods to gold annotations.
  6. Save per-case + aggregate metrics.

Usage — single case:
    python3 run_scene_graph_hallucination_eval.py \\
        --movie_dir English/synthetic_hallucination_001 \\
        --annotations English/synthetic_hallucination_001/annotations.json \\
        --output_dir movie_results/synthetic_hallucination_001 \\
        --model azure_openai --model_name gpt-5.4-mini \\
        --api_key_file openai.txt --base_url_file base_url.txt

Usage — batch via manifest:
    python3 run_scene_graph_hallucination_eval.py \\
        --manifest synthetic_hallucination_cases.json \\
        --output_dir movie_results/synthetic_hallucination_batch \\
        --model azure_openai --model_name gpt-5.4-mini \\
        --api_key_file openai.txt --base_url_file base_url.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure package is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_kg.evaluation.graph_by_graph.attribute_extractor import extract_all_scene_attributes
from stage_kg.evaluation.graph_by_graph.graph_verifier import (
    _verify_against_full_text,
    load_scene_graphs,
    verify_scene_graphs,
)
from stage_kg.evaluation.graph_by_graph.hallucination_metrics import (
    DEFAULT_THRESHOLD,
    aggregate_metrics,
    compute_metrics,
)
from stage_kg.evaluation.graph_by_graph.llm_judge_hallucination import run_llm_judge
from stage_kg.ingest.loader import load_movie
from stage_kg.llm import get_llm
from stage_kg.pipeline import run_pipeline_per_scene
from stage_kg.utils.logging_utils import setup_logging


GRAPH_METHOD_NAME = "graph_verifier"
LLM_JUDGE_METHOD_NAME = "llm_judge"


@dataclass
class Case:
    story_id: str
    movie_dir: Path
    annotations: Optional[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-scene graphs and run temporal hallucination evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input selection
    p.add_argument("--manifest", type=Path, default=None,
                   help="JSON list of cases: [{story_id, movie_dir, annotations}, ...].")
    p.add_argument("--movie_dir", type=Path, default=None,
                   help="Single case: path containing script.json.")
    p.add_argument("--annotations", type=Path, default=None,
                   help="Single case: path to gold annotations JSON.")
    p.add_argument("--story_id", type=str, default=None,
                   help="Single case: story id (defaults to movie_dir name).")

    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for all per-case and aggregate outputs.")

    # LLM provider
    p.add_argument("--model", type=str, default="azure_openai",
                   choices=["openai", "azure_openai", "anthropic", "gemini", "vllm"],
                   help="LLM provider for graph construction AND LLM judge.")
    p.add_argument("--model_name", type=str, default="gpt-5.4-mini",
                   help="Model/deployment name.")
    p.add_argument("--api_key", type=str, default=None,
                   help="API key string (overrides --api_key_file).")
    p.add_argument("--api_key_file", type=Path, default=None,
                   help="Path to file containing the API key (one line).")
    p.add_argument("--base_url", type=str, default=None,
                   help="LLM endpoint URL (overrides --base_url_file).")
    p.add_argument("--base_url_file", type=Path, default=None,
                   help="Path to file containing the endpoint URL.")
    p.add_argument("--api_version", type=str, default=None,
                   help="Azure API version (e.g. 2024-12-01-preview).")

    # Pipeline knobs
    p.add_argument("--max_scenes", type=int, default=None,
                   help="Cap scenes per case (debug only).")
    p.add_argument("--skip_normalization", action="store_true",
                   help="Skip merge adjudication (faster, lower quality).")
    p.add_argument("--reuse_existing_graphs", action="store_true",
                   help="If scene_graphs/ already exist, skip rebuilding them.")
    p.add_argument("--verify_existing_proposals", action="store_true",
                   help="Reuse graph_method_proposals.json and rerun only the final text verifier.")
    p.add_argument("--skip_graph_method", action="store_true",
                   help="Skip the graph_verifier method.")
    p.add_argument("--skip_llm_judge", action="store_true",
                   help="Skip the LLM judge baseline.")
    p.add_argument("--match_threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Token-overlap threshold for matching gold vs prediction.")

    p.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _resolve_secret(arg_value: Optional[str], file_arg: Optional[Path]) -> Optional[str]:
    if arg_value:
        return arg_value
    if file_arg:
        return _read_text_file(file_arg)
    return None


def load_manifest(path: Path) -> List[Case]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Manifest {path} must be a JSON list.")
    base = path.resolve().parent
    cases: List[Case] = []
    for entry in raw:
        movie_dir = Path(entry["movie_dir"])
        if not movie_dir.is_absolute():
            movie_dir = (base / movie_dir).resolve()
        ann = entry.get("annotations")
        ann_path: Optional[Path] = None
        if ann:
            ann_path = Path(ann)
            if not ann_path.is_absolute():
                ann_path = (base / ann_path).resolve()
        cases.append(
            Case(
                story_id=entry["story_id"],
                movie_dir=movie_dir,
                annotations=ann_path,
            )
        )
    return cases


def load_gold(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scene_dict_for_judge(movie) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    for scene in movie.scenes:
        try:
            scene_number = int(scene.scene_id)
        except (TypeError, ValueError):
            scene_number = scene.order + 1
        scenes.append(
            {
                "scene_number": scene_number,
                "scene_id": scene.scene_id,
                "title": scene.title,
                "content": scene.content,
            }
        )
    return scenes


def _scene_texts(movie) -> Dict[int, str]:
    texts: Dict[int, str] = {}
    for scene in movie.scenes:
        try:
            n = int(scene.scene_id)
        except (TypeError, ValueError):
            n = scene.order + 1
        texts[n] = scene.content
    return texts


def _has_complete_scene_graphs(scene_graphs_dir: Path, expected_count: int) -> bool:
    if expected_count <= 0 or not scene_graphs_dir.exists():
        return False
    return len(list(scene_graphs_dir.glob("scene_*/final_graph.json"))) >= expected_count


def process_case(
    case: Case,
    output_dir: Path,
    llm,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    logger.info("=" * 70)
    logger.info("CASE %s — %s", case.story_id, case.movie_dir)
    logger.info("=" * 70)

    case_out = output_dir / case.story_id
    case_out.mkdir(parents=True, exist_ok=True)

    # 1. Load movie
    movie = load_movie(
        movie_dir=case.movie_dir,
        movie_id=case.story_id,
        title=case.story_id,
        language=args.language,
    )
    if args.max_scenes:
        movie.scenes = movie.scenes[: args.max_scenes]
    if not movie.scenes:
        logger.error("No scenes found in %s — skipping.", case.movie_dir)
        return []

    # 2. Build per-scene graphs (or reuse if present)
    scene_graphs_dir = case_out / "scene_graphs"
    must_build = not (
        args.reuse_existing_graphs
        and _has_complete_scene_graphs(scene_graphs_dir, len(movie.scenes))
    )
    if args.verify_existing_proposals:
        logger.info("Skipping graph construction; verifying existing proposals only.")
    elif must_build and not args.skip_graph_method:
        logger.info("Building per-scene graphs for %s ...", case.story_id)
        run_pipeline_per_scene(
            movie=movie,
            llm=llm,
            output_dir=output_dir,
            skip_normalization=args.skip_normalization,
            api_key=args.api_key
            or (_read_text_file(args.api_key_file) if args.api_key_file else None),
        )
    elif args.skip_graph_method:
        logger.info("Skipping graph construction (--skip_graph_method).")
    else:
        logger.info("Reusing existing scene graphs in %s", scene_graphs_dir)

    # 3. Graph-method hallucinations
    graph_predictions: List[Dict[str, Any]] = []
    if not args.skip_graph_method:
        if args.verify_existing_proposals:
            proposals_path = case_out / "graph_method_proposals.json"
            if not proposals_path.exists():
                logger.error(
                    "Cannot verify existing proposals; missing %s", proposals_path
                )
                return []
            with proposals_path.open("r", encoding="utf-8") as f:
                proposals = json.load(f)
            if not isinstance(proposals, list):
                logger.error("Existing proposals file is not a JSON list: %s", proposals_path)
                return []

            confirmed, verification_log = _verify_against_full_text(
                llm=llm,
                proposals=proposals,
                scene_texts=_scene_texts(movie),
                debug_path=case_out / "graph_method_verification_raw.json",
                with_trace=True,
            )
            graph_predictions = [
                {"id": idx, "hallucination": prop["text"]}
                for idx, prop in enumerate(confirmed, start=1)
            ]
            _save_json(
                {
                    "raw_proposals": len(proposals),
                    "after_dedupe": len(proposals),
                    "confirmed": len(confirmed),
                    "final_detections": len(graph_predictions),
                    "verification_only": True,
                    "verification_log": verification_log,
                },
                case_out / "graph_method_audit.json",
            )
        else:
            scene_graphs = load_scene_graphs(scene_graphs_dir)

            # Per-scene durable-attribute extraction (cached on disk).
            attributes_dir = case_out / "attributes"
            scene_attributes = extract_all_scene_attributes(
                scene_records=movie.scenes,
                llm=llm,
                cache_dir=attributes_dir,
            )
            logger.info(
                "Attribute extractor produced profiles for %d scenes",
                sum(1 for v in scene_attributes.values() if v),
            )

            graph_predictions = verify_scene_graphs(
                scene_graphs=scene_graphs,
                llm=llm,
                scene_texts=_scene_texts(movie),
                proposals_debug_path=case_out / "graph_method_proposals.json",
                scene_attributes=scene_attributes,
                audit_log_path=case_out / "graph_method_audit.json",
            )
        _save_json(graph_predictions, case_out / "graph_method_hallucinations.json")
        logger.info("Graph method: %d detections", len(graph_predictions))

    # 5. LLM judge baseline
    judge_predictions: List[Dict[str, Any]] = []
    if not args.skip_llm_judge:
        judge_predictions = run_llm_judge(
            scenes=_scene_dict_for_judge(movie),
            llm=llm,
            raw_output_path=case_out / "llm_judge_raw_response.txt",
        )
        _save_json(judge_predictions, case_out / "llm_judge_hallucinations.json")
        logger.info("LLM judge: %d detections", len(judge_predictions))

    # 7. Compare to gold
    gold = load_gold(case.annotations)
    logger.info("Gold annotations: %d", len(gold))

    per_case_rows: List[Dict[str, Any]] = []
    if not args.skip_graph_method:
        per_case_rows.append(
            compute_metrics(
                story_id=case.story_id,
                method=GRAPH_METHOD_NAME,
                gold=gold,
                predicted=graph_predictions,
                threshold=args.match_threshold,
            )
        )
    if not args.skip_llm_judge:
        per_case_rows.append(
            compute_metrics(
                story_id=case.story_id,
                method=LLM_JUDGE_METHOD_NAME,
                gold=gold,
                predicted=judge_predictions,
                threshold=args.match_threshold,
            )
        )

    _save_json(per_case_rows, case_out / "comparison_metrics.json")
    return per_case_rows


def main() -> None:
    args = parse_args()

    setup_logging(log_dir=args.output_dir / "logs", verbose=args.verbose)
    logger = logging.getLogger(__name__)

    if not args.manifest and not args.movie_dir:
        logger.error("Provide either --manifest or --movie_dir.")
        sys.exit(1)

    # Resolve credentials once. The same LLM instance is used for both
    # graph construction and the LLM judge — required by the spec.
    api_key = _resolve_secret(args.api_key, args.api_key_file)
    base_url = _resolve_secret(args.base_url, args.base_url_file)

    if args.model in ("azure_openai", "azure-openai", "azure"):
        if not api_key:
            logger.error("Azure OpenAI requires --api_key or --api_key_file.")
            sys.exit(1)
        if not base_url:
            logger.error("Azure OpenAI requires --base_url or --base_url_file.")
            sys.exit(1)

    logger.info(
        "LLM config: provider=%s model_name=%s api_version=%s",
        args.model,
        args.model_name,
        args.api_version,
    )
    llm = get_llm(
        provider=args.model,
        model=args.model_name,
        api_key=api_key,
        base_url=base_url,
        api_version=args.api_version,
    )
    logger.info("LLM ready: %s", llm.model_id)

    # Collect cases
    if args.manifest:
        cases = load_manifest(args.manifest)
    else:
        story_id = args.story_id or args.movie_dir.name
        cases = [
            Case(story_id=story_id, movie_dir=args.movie_dir, annotations=args.annotations)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for case in cases:
        rows = process_case(case, args.output_dir, llm, args, logger)
        all_rows.extend(rows)

    # Aggregate
    agg = aggregate_metrics(all_rows)
    combined = all_rows + agg

    _save_json(combined, args.output_dir / "comparison_metrics.json")
    _save_csv(combined, args.output_dir / "comparison_metrics.csv")

    logger.info("Saved aggregate metrics to %s", args.output_dir / "comparison_metrics.csv")
    for row in combined:
        logger.info(
            "%-30s %-15s gold=%d pred=%d tp=%d fp=%d fn=%d P=%.3f R=%.3f F1=%.3f",
            row["story_id"],
            row["method"],
            row["Gold"],
            row["Pred."],
            row["TP"],
            row["FP"],
            row["FN"],
            row["precision"],
            row["recall"],
            row["F1"],
        )


if __name__ == "__main__":
    main()
