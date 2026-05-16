#!/usr/bin/env python3
"""LLM judge for cross-scene repetition only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_as_judge import (
    JudgeConfig,
    append_raw,
    call_judge,
    compact_text,
    cross_scene_prompt,
    ensure_openai_client,
    load_scene_text,
    normalize_list,
    parse_json_object,
    repair_json_response,
    retry_empty_response,
    write_csv,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone LLM judge for cross-scene repetition only."
    )
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", ""))
    parser.add_argument("--api_key", default=os.environ.get("AZURE_AI_FOUNDRY_API_KEY", ""))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_AI_FOUNDRY_DEPLOYMENT", ""))
    parser.add_argument("--api_style", choices=["chat", "responses"], default="chat")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_scene_chars", type=int, default=2500)
    parser.add_argument("--summary_chars", type=int, default=420)
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


def scene_summary_prompt(scene_id: str, scene_text: str, summary_chars: int) -> tuple[str, str]:
    system = (
        "You create compact scene summaries for a cross-scene repetition benchmark. "
        "Use only the supplied generated scene text. Preserve concrete story beats, character actions, "
        "relationships, locations, conflicts, outcomes, and repeated wording that would help detect whether "
        "another scene replays the same beat. Return only valid JSON."
    )
    schema = {
        "scene_id": scene_id,
        "summary": "one compact scene summary",
        "key_beats": ["short beat 1", "short beat 2", "short beat 3"],
        "snippets": ["short representative snippet 1", "short representative snippet 2"],
    }
    prompt = (
        f"SCENE ID: {scene_id}\n\n"
        f"GENERATED SCENE TEXT:\n{scene_text}\n\n"
        f"Keep the summary under about {summary_chars} characters. "
        "Do not evaluate repetition yet; only summarize the scene for later comparison.\n\n"
        "Return JSON with this exact top-level shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def summarize_scene(
    client: Any,
    config: JudgeConfig,
    scene_id: str,
    scene_text: str,
    raw_path: Path,
    validate_only: bool,
    summary_chars: int,
) -> Dict[str, Any]:
    fallback = {
        "scene_id": scene_id,
        "summary": compact_text(scene_text, summary_chars),
        "key_beats": [],
        "snippets": [compact_text(scene_text, min(220, summary_chars))],
    }
    if validate_only:
        append_raw(raw_path, {
            "kind": "scene_summary",
            "scene_id": scene_id,
            "validate_only": True,
            "parsed": fallback,
        })
        return fallback

    system, prompt = scene_summary_prompt(scene_id, scene_text, summary_chars)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape:", 1)[-1]
    raw_record = {
        "kind": "scene_summary",
        "scene_id": scene_id,
        "repaired": False,
    }
    try:
        response_text = call_judge(client, config, system, prompt)
        raw_record["raw_response"] = response_text
    except Exception as call_error:
        raw_record["error"] = str(call_error)
        raw_record["error_stage"] = "judge_call"
        append_raw(raw_path, raw_record)
        return fallback
    if not response_text.strip():
        raw_record["empty_response_retry"] = True
        try:
            response_text = retry_empty_response(client, config, prompt, expected_schema)
            raw_record["retry_response"] = response_text
        except Exception as retry_error:
            raw_record["error"] = f"empty response; retry failed: {retry_error}"
            raw_record["error_stage"] = "empty_response_retry"
            append_raw(raw_path, raw_record)
            return fallback
        if not response_text.strip():
            raw_record["error"] = "Judge returned an empty response body twice."
            raw_record["error_stage"] = "empty_response"
            append_raw(raw_path, raw_record)
            return fallback

    try:
        parsed = parse_json_object(response_text)
    except Exception as first_error:
        try:
            repaired = repair_json_response(client, config, response_text, "Scene summary object.")
            parsed = parse_json_object(repaired)
            raw_record["repaired"] = True
            raw_record["repair_response"] = repaired
        except Exception as second_error:
            raw_record["error"] = f"{first_error}; repair failed: {second_error}"
            append_raw(raw_path, raw_record)
            return fallback

    result = {
        "scene_id": str(parsed.get("scene_id") or scene_id),
        "summary": compact_text(parsed.get("summary") or fallback["summary"], summary_chars),
        "key_beats": [
            compact_text(item, 180)
            for item in (parsed.get("key_beats") if isinstance(parsed.get("key_beats"), list) else [])
        ][:8],
        "snippets": [
            compact_text(item, 220)
            for item in (parsed.get("snippets") if isinstance(parsed.get("snippets"), list) else [])
        ][:4],
    }
    if not result["snippets"]:
        result["snippets"] = fallback["snippets"]
    raw_record["parsed"] = result
    append_raw(raw_path, raw_record)
    return result


def cross_scene_direct_prompt(scene_inputs: Sequence[Dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are a strict benchmark judge for cross-scene repetition. Use only the supplied generated scene text. "
        "Identify pairs of scenes that repeat the same wording, narrative beat, event pattern, "
        "character interaction, or semantic content without meaningful progression. Do not flag intentional continuity "
        "unless the later scene merely replays the same beat. Return only valid JSON."
    )
    schema = {
        "cross_scene_repetition": [
            {
                "scene_i_id": "scene id",
                "scene_j_id": "scene id",
                "repeated_beat": "shared repeated beat or wording",
                "snippets": ["snippet from scene i", "snippet from scene j"],
                "reason": "why this is cross-scene repetition",
                "confidence": 0.0,
            }
        ]
    }
    prompt = (
        "GENERATED SCENES:\n"
        f"{json.dumps(scene_inputs, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact top-level shape. Use an empty array when nothing is found:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def judge_cross_scene_direct(
    client: Any,
    config: JudgeConfig,
    scene_inputs: Sequence[Dict[str, Any]],
    raw_path: Path,
    validate_only: bool,
) -> List[Dict[str, Any]]:
    if validate_only or len(scene_inputs) < 2:
        append_raw(raw_path, {
            "kind": "cross_scene",
            "validate_only": validate_only,
            "parsed": {"cross_scene_repetition": []},
        })
        return []

    system, prompt = cross_scene_direct_prompt(scene_inputs)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape.", 1)[-1]
    raw_record = {
        "kind": "cross_scene",
        "repaired": False,
    }
    try:
        response_text = call_judge(client, config, system, prompt)
        raw_record["raw_response"] = response_text
    except Exception as call_error:
        raw_record["error"] = str(call_error)
        raw_record["error_stage"] = "judge_call"
        append_raw(raw_path, raw_record)
        return []
    if not response_text.strip():
        raw_record["empty_response_retry"] = True
        try:
            response_text = retry_empty_response(client, config, prompt, expected_schema)
            raw_record["retry_response"] = response_text
        except Exception as retry_error:
            raw_record["error"] = f"empty response; retry failed: {retry_error}"
            raw_record["error_stage"] = "empty_response_retry"
            append_raw(raw_path, raw_record)
            return []
        if not response_text.strip():
            raw_record["error"] = "Judge returned an empty response body twice."
            raw_record["error_stage"] = "empty_response"
            append_raw(raw_path, raw_record)
            return []

    try:
        parsed = parse_json_object(response_text)
    except Exception as first_error:
        try:
            repaired = repair_json_response(client, config, response_text, "Object with cross_scene_repetition array.")
            parsed = parse_json_object(repaired)
            raw_record["repaired"] = True
            raw_record["repair_response"] = repaired
        except Exception as second_error:
            raw_record["error"] = f"{first_error}; repair failed: {second_error}"
            append_raw(raw_path, raw_record)
            return []

    rows = normalize_list(parsed.get("cross_scene_repetition"))
    raw_record["parsed"] = {"cross_scene_repetition": rows}
    append_raw(raw_path, raw_record)
    return rows


def judge_cross_scene_from_summaries(
    client: Any,
    config: JudgeConfig,
    scene_summaries: Sequence[Dict[str, Any]],
    raw_path: Path,
    validate_only: bool,
) -> List[Dict[str, Any]]:
    if validate_only or len(scene_summaries) < 2:
        append_raw(raw_path, {
            "kind": "cross_scene",
            "validate_only": validate_only,
            "parsed": {"cross_scene_repetition": []},
        })
        return []

    system, prompt = cross_scene_prompt(scene_summaries)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape.", 1)[-1]
    raw_record = {
        "kind": "cross_scene",
        "input_kind": "scene_summaries",
        "repaired": False,
    }
    try:
        response_text = call_judge(client, config, system, prompt)
        raw_record["raw_response"] = response_text
    except Exception as call_error:
        raw_record["error"] = str(call_error)
        raw_record["error_stage"] = "judge_call"
        append_raw(raw_path, raw_record)
        return []
    if not response_text.strip():
        raw_record["empty_response_retry"] = True
        try:
            response_text = retry_empty_response(client, config, prompt, expected_schema)
            raw_record["retry_response"] = response_text
        except Exception as retry_error:
            raw_record["error"] = f"empty response; retry failed: {retry_error}"
            raw_record["error_stage"] = "empty_response_retry"
            append_raw(raw_path, raw_record)
            return []
        if not response_text.strip():
            raw_record["error"] = "Judge returned an empty response body twice."
            raw_record["error_stage"] = "empty_response"
            append_raw(raw_path, raw_record)
            return []

    try:
        parsed = parse_json_object(response_text)
    except Exception as first_error:
        try:
            repaired = repair_json_response(client, config, response_text, "Object with cross_scene_repetition array.")
            parsed = parse_json_object(repaired)
            raw_record["repaired"] = True
            raw_record["repair_response"] = repaired
        except Exception as second_error:
            raw_record["error"] = f"{first_error}; repair failed: {second_error}"
            append_raw(raw_path, raw_record)
            return []

    rows = normalize_list(parsed.get("cross_scene_repetition"))
    raw_record["parsed"] = {"cross_scene_repetition": rows}
    append_raw(raw_path, raw_record)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    text_path = Path(args.text_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    scene_summaries = []
    for scene_id, scene_text in scenes.items():
        compact_scene = compact_text(scene_text, args.max_scene_chars)
        scene_summaries.append(
            summarize_scene(
                client,
                config,
                scene_id,
                compact_scene,
                raw_path,
                args.validate_only,
                args.summary_chars,
            )
        )

    cross_scene_rows = judge_cross_scene_from_summaries(
        client,
        config,
        scene_summaries,
        raw_path,
        args.validate_only,
    )
    write_csv(
        output_dir / "cross_scene_repetition.csv",
        ["scene_i_id", "scene_j_id", "repeated_beat", "snippets", "reason", "confidence"],
        cross_scene_rows,
    )

    scene_count = len(scene_summaries)
    possible_pairs = scene_count * (scene_count - 1) // 2
    summary = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text_path": str(text_path),
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_style": args.api_style,
            "validate_only": bool(args.validate_only),
            "max_scene_chars": args.max_scene_chars,
            "summary_chars": args.summary_chars,
            "cross_scene_input": "scene_summaries",
        },
        "metrics": {
            "total_scenes_judged": scene_count,
            "total_cross_scene_repeated_pairs": len(cross_scene_rows),
            "possible_cross_scene_pairs": possible_pairs,
            "cross_scene_repetition_rate": len(cross_scene_rows) / max(1, possible_pairs),
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
