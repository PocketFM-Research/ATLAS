#!/usr/bin/env python3
"""LLM judge for hallucination and in-scene repetition only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from llm_as_judge import (
    JudgeConfig,
    build_detail_rows,
    build_graph_context,
    chunk_scene_text,
    ensure_openai_client,
    judge_hallucination_chunk,
    judge_in_scene_repetition_chunk,
    load_json,
    load_scene_text,
    merge_chunk_judgments,
    merge_scene_chunk_results,
    write_csv,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone LLM judge for hallucinations and in-scene repetition."
    )
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", ""))
    parser.add_argument("--api_key", default=os.environ.get("AZURE_AI_FOUNDRY_API_KEY", ""))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_AI_FOUNDRY_DEPLOYMENT", ""))
    parser.add_argument("--api_style", choices=["chat", "responses"], default="chat")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_graph_items_per_scene", type=int, default=80)
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=2500,
        help="Chunk each scene before judging so one content-filtered chunk does not skip the whole scene.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args(argv)


def require_judge_config(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in {
            "--endpoint": args.endpoint,
            "--api_key or AZURE_AI_FOUNDRY_API_KEY": args.api_key,
            "--deployment": args.deployment,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required judge configuration: {', '.join(missing)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    graph_path = Path(args.graph_path)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_json(graph_path)
    if not isinstance(graph, dict):
        raise TypeError("Graph input must be a JSON object.")
    scenes = load_scene_text(text_path)
    if not scenes:
        raise ValueError("No scene text found after normalization.")
    if args.max_scenes is not None:
        scenes = dict(list(scenes.items())[: args.max_scenes])

    raw_path = output_dir / "raw_judge_responses.jsonl"
    raw_path.write_text("", encoding="utf-8")

    config = JudgeConfig(
        endpoint=args.endpoint,
        api_key=args.api_key,
        deployment=args.deployment,
        api_style=args.api_style,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    client = None
    if not args.validate_only:
        require_judge_config(args)
        client = ensure_openai_client(config)

    scene_results = []
    for scene_id, scene_text in scenes.items():
        graph_context = build_graph_context(graph, scene_id, args.max_graph_items_per_scene)
        chunks = chunk_scene_text(scene_text, args.max_chunk_chars)
        chunk_results = []
        for chunk_index, chunk_text in chunks:
            hallucination_result = judge_hallucination_chunk(
                client,
                config,
                scene_id,
                chunk_text,
                graph_context,
                raw_path,
                args.validate_only,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
            )
            repetition_result = judge_in_scene_repetition_chunk(
                client,
                config,
                scene_id,
                chunk_text,
                raw_path,
                args.validate_only,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
            )
            chunk_results.append(
                merge_chunk_judgments(
                    scene_id,
                    chunk_text,
                    hallucination_result,
                    repetition_result,
                )
            )
        scene_results.append(merge_scene_chunk_results(scene_id, scene_text, chunk_results))

    hallucination_rows, in_scene_rows = build_detail_rows(scene_results)
    write_csv(
        output_dir / "hallucinations.csv",
        ["scene_id", "chunk_index", "snippet", "reason", "severity", "confidence", "graph_evidence"],
        hallucination_rows,
    )
    write_csv(
        output_dir / "in_scene_repetition.csv",
        ["scene_id", "chunk_index", "snippet", "positions", "reason", "confidence"],
        in_scene_rows,
    )

    hallucinations_by_scene = {}
    for row in hallucination_rows:
        scene_id = str(row.get("scene_id", ""))
        hallucinations_by_scene[scene_id] = hallucinations_by_scene.get(scene_id, 0) + 1

    summary = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph_path": str(graph_path),
            "text_path": str(text_path),
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_style": args.api_style,
            "validate_only": bool(args.validate_only),
            "max_graph_items_per_scene": args.max_graph_items_per_scene,
            "max_chunk_chars": args.max_chunk_chars,
        },
        "metrics": {
            "total_scenes_judged": len(scene_results),
            "total_hallucination_snippets": len(hallucination_rows),
            "hallucination_snippets_per_scene": hallucinations_by_scene,
            "hallucination_rate_per_scene": len(hallucination_rows) / max(1, len(scene_results)),
            "total_in_scene_repetition_snippets": len(in_scene_rows),
            "in_scene_repetition_rate": len(in_scene_rows) / max(1, len(scene_results)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
