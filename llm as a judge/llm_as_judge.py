#!/usr/bin/env python3
"""Standalone LLM-as-judge benchmark for scene text against a graph.

This script intentionally does not import the existing STAGE evaluation stack.
It uses only local parsing, compact graph evidence, and an Azure AI Foundry /
OpenAI v1-compatible judge deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCENE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:scene|sc)\s*[\w.-]+(?:\s*[-:]\s*.+)?"
    r"|(?:int|ext|int/ext|i/e)\.\s+.+"
    r"|\d+\s*[\).:：、-]\s*.+"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class JudgeConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_style: str
    temperature: float
    max_tokens: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_scene_text(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = load_json(path)
        if not isinstance(raw, dict):
            raise TypeError("JSON text input must be {scene_id: text} or {scene_id: {character_id: text}}.")
        scenes: Dict[str, str] = {}
        for scene_id, value in raw.items():
            if isinstance(value, dict):
                parts = [str(text).strip() for text in value.values() if str(text).strip()]
                if parts:
                    scenes[str(scene_id)] = "\n\n".join(parts)
            elif isinstance(value, str) and value.strip():
                scenes[str(scene_id)] = value.strip()
        return scenes

    return split_story_into_scenes(path.read_text(encoding="utf-8"))


def split_story_into_scenes(story_text: str) -> Dict[str, str]:
    scenes: Dict[str, str] = {}
    current_id = ""
    current_lines: List[str] = []
    heading_count = 0

    def flush() -> None:
        if current_id and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                scenes[current_id] = text

    for raw_line in story_text.splitlines():
        line = raw_line.strip()
        if line and SCENE_HEADING_RE.match(line):
            flush()
            heading_count += 1
            current_id = scene_id_from_heading(line, heading_count)
            current_lines = [line]
            continue
        if current_id:
            current_lines.append(raw_line)

    flush()
    stripped = story_text.strip()
    return scenes or ({"scene_001": stripped} if stripped else {})


def scene_id_from_heading(heading: str, index: int) -> str:
    match = re.match(r"^\s*(\d+)[\).:：、-]", heading)
    if match:
        return match.group(1)
    normalized = re.sub(r"\s+", "_", heading.strip().lower())
    normalized = re.sub(r"[^a-z0-9_.-]+", "", normalized).strip("._-")
    return normalized or f"scene_{index:03d}"


def scene_ref_matches(scene_refs: Iterable[Any], scene_id: str) -> bool:
    wanted = str(scene_id)
    aliases = {wanted, f"scene_{wanted}", wanted.removeprefix("scene_")}
    return any(str(ref) in aliases for ref in scene_refs or [])


def compact_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def chunk_scene_text(scene_text: str, max_chunk_chars: int) -> List[Tuple[int, str]]:
    """Split one scene into smaller chunks so one filtered call does not drop the whole scene."""
    text = scene_text.strip()
    if not text:
        return []
    if max_chunk_chars <= 0 or len(text) <= max_chunk_chars:
        return [(1, text)]

    chunks: List[str] = []
    current = ""
    for part in [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]:
        if len(part) > max_chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_text(part, max_chunk_chars))
            continue
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())
    return [(index, chunk) for index, chunk in enumerate(chunks, start=1)]


def split_long_text(text: str, max_chunk_chars: int) -> List[str]:
    pieces: List[str] = []
    current = ""
    sentences = re.findall(r".+?(?:[.!?](?:\s+|$)|\n+|$)", text, flags=re.DOTALL)
    for sentence in sentences or [text]:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chunk_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chunk_chars):
                pieces.append(sentence[start:start + max_chunk_chars].strip())
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            if current:
                pieces.append(current.strip())
            current = sentence
    if current:
        pieces.append(current.strip())
    return pieces or [text[:max_chunk_chars]]


def build_graph_context(graph: Dict[str, Any], scene_id: str, max_items: int) -> Dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    node_by_id = {str(node.get("id")): node for node in nodes if isinstance(node, dict)}
    scene_nodes = [
        compact_node(node)
        for node in nodes
        if isinstance(node, dict) and scene_ref_matches(node.get("scene_refs", []), scene_id)
    ][:max_items]
    scene_edges = [
        compact_edge(edge, node_by_id)
        for edge in edges
        if isinstance(edge, dict) and scene_ref_matches(edge.get("scene_refs", []), scene_id)
    ][:max_items]

    return {
        "movie_id": graph.get("movie_id"),
        "title": graph.get("title"),
        "scene_id": scene_id,
        "nodes": scene_nodes,
        "edges": scene_edges,
        "truncated": {
            "max_items_per_type": max_items,
            "nodes_available": sum(
                1 for node in nodes if isinstance(node, dict) and scene_ref_matches(node.get("scene_refs", []), scene_id)
            ),
            "edges_available": sum(
                1 for edge in edges if isinstance(edge, dict) and scene_ref_matches(edge.get("scene_refs", []), scene_id)
            ),
        },
    }


def compact_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node.get("id"),
        "type": node.get("type"),
        "name": compact_text(node.get("name"), 160),
        "description": compact_text(node.get("description"), 260),
        "evidence": [compact_text(item, 220) for item in (node.get("evidence") or [])[:3]],
        "scene_refs": node.get("scene_refs", []),
    }


def compact_edge(edge: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source_id = str(edge.get("source"))
    target_id = str(edge.get("target"))
    source = node_by_id.get(source_id, {})
    target = node_by_id.get(target_id, {})
    return {
        "id": edge.get("id"),
        "relation": edge.get("relation"),
        "source": compact_text(source.get("name") or source_id, 140),
        "target": compact_text(target.get("name") or target_id, 140),
        "triple": compact_text(
            f"{source.get('name') or source_id} --{edge.get('relation')}--> {target.get('name') or target_id}",
            320,
        ),
        "evidence": [compact_text(item, 220) for item in (edge.get("evidence") or [])[:3]],
        "confidence": edge.get("confidence"),
        "scene_refs": edge.get("scene_refs", []),
    }


def ensure_openai_client(config: JudgeConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: pip install openai") from exc
    return OpenAI(base_url=config.endpoint, api_key=config.api_key)


def call_judge(client: Any, config: JudgeConfig, system: str, prompt: str) -> str:
    if config.api_style == "responses":
        response = client.responses.create(
            model=config.deployment,
            instructions=system,
            input=prompt,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        return getattr(response, "output_text", "") or response.model_dump_json()

    response = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(stripped[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("Judge response did not contain a JSON object.")


def repair_json_response(client: Any, config: JudgeConfig, bad_response: str, expected_schema: str) -> str:
    system = "You repair malformed JSON. Return only one valid JSON object, no markdown."
    prompt = (
        "Repair this model output into valid JSON matching the expected schema. "
        "Do not add facts that are not already present.\n\n"
        f"EXPECTED SCHEMA:\n{expected_schema}\n\n"
        f"MODEL OUTPUT:\n{bad_response}"
    )
    return call_judge(client, config, system, prompt)


def retry_empty_response(client: Any, config: JudgeConfig, original_prompt: str, expected_schema: str) -> str:
    system = "You are a JSON-only evaluation engine. Return one valid JSON object and no markdown."
    prompt = (
        "The previous attempt returned an empty response. Retry the same evaluation more concisely. "
        "Return only valid JSON. Use empty arrays only if there are truly no findings.\n\n"
        f"EXPECTED JSON SHAPE:\n{expected_schema}\n\n"
        "ORIGINAL EVALUATION INPUT:\n"
        f"{compact_text(original_prompt, 12000)}"
    )
    return call_judge(client, config, system, prompt)


def scene_prompt(
    scene_id: str,
    scene_text: str,
    graph_context: Dict[str, Any],
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Tuple[str, str]:
    system = (
        "You are a strict benchmark judge. Evaluate generated scene text only against the supplied graph evidence. "
        "Do not use outside knowledge. Mark hallucinations only when the text asserts a concrete fact/event/relation "
        "that is unsupported by, absent from, or contradicted by the graph evidence. For in-scene repetition, ignore "
        "the graph and judge only the generated scene text for repeated wording or repeated semantic beats that add "
        "no new information. Return only valid JSON."
    )
    schema = {
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "hallucinations": [
            {
                "snippet": "exact or near-exact unsupported text snippet",
                "reason": "why it is unsupported or contradicted",
                "severity": "low|medium|high",
                "confidence": 0.0,
                "graph_evidence": ["relevant graph evidence or empty if absent"],
            }
        ],
        "in_scene_repetition": [
            {
                "snippet": "repeated wording or semantic beat",
                "positions": ["brief location descriptions if inferable"],
                "reason": "why it is repetitive",
                "confidence": 0.0,
            }
        ],
        "scene_summary": {
            "summary": "one compact sentence describing the scene text",
            "key_beats": ["short beat 1", "short beat 2"],
            "confidence": 0.0,
        },
    }
    prompt = (
        f"SCENE ID: {scene_id}\n\n"
        f"CHUNK: {chunk_index} of {chunk_count}\n\n"
        f"GENERATED SCENE TEXT CHUNK:\n{scene_text}\n\n"
        "SCENE GRAPH EVIDENCE JSON:\n"
        f"{json.dumps(graph_context, ensure_ascii=False, indent=2)}\n\n"
        "Important: use graph evidence only for hallucination judgment. Judge in-scene repetition from the generated "
        "scene text alone.\n\n"
        "Return JSON with this exact top-level shape. Use empty arrays when nothing is found:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def hallucination_prompt(
    scene_id: str,
    scene_text: str,
    graph_context: Dict[str, Any],
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Tuple[str, str]:
    system = (
        "You are a strict hallucination benchmark judge. Use only the supplied graph evidence. "
        "Do not use outside knowledge. Mark hallucinations only when the generated text asserts a concrete "
        "fact, event, relationship, location, object use, character state, or causal link that is unsupported by, "
        "absent from, or contradicted by the graph evidence. Return only valid JSON."
    )
    schema = {
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "hallucinations": [
            {
                "snippet": "exact or near-exact unsupported text snippet",
                "reason": "why it is unsupported or contradicted",
                "severity": "low|medium|high",
                "confidence": 0.0,
                "graph_evidence": ["relevant graph evidence or empty if absent"],
            }
        ],
    }
    prompt = (
        f"SCENE ID: {scene_id}\n"
        f"CHUNK: {chunk_index} of {chunk_count}\n\n"
        f"GENERATED SCENE TEXT CHUNK:\n{scene_text}\n\n"
        "SCENE GRAPH EVIDENCE JSON:\n"
        f"{json.dumps(graph_context, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact top-level shape. Use an empty hallucinations array when nothing is found:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def in_scene_repetition_prompt(
    scene_id: str,
    scene_text: str,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Tuple[str, str]:
    system = (
        "You are a strict in-scene repetition benchmark judge. Use only the generated scene text. "
        "Identify repeated wording or repeated semantic beats inside this chunk that add no meaningful new "
        "information. Do not use graph evidence or outside knowledge. Return only valid JSON."
    )
    schema = {
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "in_scene_repetition": [
            {
                "snippet": "repeated wording or semantic beat",
                "positions": ["brief location descriptions if inferable"],
                "reason": "why it is repetitive",
                "confidence": 0.0,
            }
        ],
    }
    prompt = (
        f"SCENE ID: {scene_id}\n"
        f"CHUNK: {chunk_index} of {chunk_count}\n\n"
        f"GENERATED SCENE TEXT CHUNK:\n{scene_text}\n\n"
        "Return JSON with this exact top-level shape. Use an empty in_scene_repetition array when nothing is found:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def cross_scene_prompt(scene_summaries: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "You are a strict benchmark judge for cross-scene repetition. Use only the supplied scene summaries and "
        "snippets. Identify pairs of scenes that reuse the same wording, narrative beat, or event pattern without "
        "meaningful progression. Return only valid JSON."
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
        "SCENE SUMMARIES AND JUDGE SNIPPETS:\n"
        f"{json.dumps(scene_summaries, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact top-level shape. Use an empty array when nothing is found:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system, prompt


def normalize_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def append_raw(raw_path: Path, record: Dict[str, Any]) -> None:
    with raw_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def empty_scene_result(scene_id: str, scene_text: str) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "hallucinations": [],
        "in_scene_repetition": [],
        "scene_summary": {
            "summary": compact_text(scene_text, 220),
            "key_beats": [],
            "confidence": 0.0,
        },
    }


def empty_chunk_result(scene_id: str, scene_text: str, chunk_index: int, chunk_count: int) -> Dict[str, Any]:
    result = empty_scene_result(scene_id, scene_text)
    result["chunk_index"] = chunk_index
    result["chunk_count"] = chunk_count
    return result


def judge_scene(
    client: Any,
    config: JudgeConfig,
    scene_id: str,
    scene_text: str,
    graph_context: Dict[str, Any],
    raw_path: Path,
    validate_only: bool,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Dict[str, Any]:
    if validate_only:
        result = empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count)
        append_raw(raw_path, {
            "kind": "scene",
            "scene_id": scene_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "validate_only": True,
            "parsed": result,
        })
        return result

    system, prompt = scene_prompt(scene_id, scene_text, graph_context, chunk_index, chunk_count)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape.", 1)[-1]
    raw_record = {
        "kind": "scene",
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "repaired": False,
    }
    try:
        response_text = call_judge(client, config, system, prompt)
        raw_record["raw_response"] = response_text
    except Exception as call_error:
        raw_record["error"] = str(call_error)
        raw_record["error_stage"] = "judge_call"
        append_raw(raw_path, raw_record)
        return {
            **empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count),
            "judge_error": str(call_error),
        }
    if not response_text.strip():
        raw_record["empty_response_retry"] = True
        try:
            response_text = retry_empty_response(client, config, prompt, expected_schema)
            raw_record["retry_response"] = response_text
        except Exception as retry_error:
            raw_record["error"] = f"empty response; retry failed: {retry_error}"
            raw_record["error_stage"] = "empty_response_retry"
            append_raw(raw_path, raw_record)
            return {
                **empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count),
                "judge_error": raw_record["error"],
            }
        if not response_text.strip():
            raw_record["error"] = "Judge returned an empty response body twice."
            raw_record["error_stage"] = "empty_response"
            append_raw(raw_path, raw_record)
            return {
                **empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count),
                "judge_error": raw_record["error"],
            }

    try:
        parsed = parse_json_object(response_text)
    except Exception as first_error:
        try:
            repaired = repair_json_response(client, config, response_text, expected_schema)
            parsed = parse_json_object(repaired)
            raw_record["repaired"] = True
            raw_record["repair_response"] = repaired
        except Exception as second_error:
            raw_record["error"] = f"{first_error}; repair failed: {second_error}"
            append_raw(raw_path, raw_record)
            return {
                **empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count),
                "judge_error": raw_record["error"],
            }

    parsed.setdefault("scene_id", scene_id)
    parsed["chunk_index"] = chunk_index
    parsed["chunk_count"] = chunk_count
    parsed["hallucinations"] = normalize_list(parsed.get("hallucinations"))
    parsed["in_scene_repetition"] = normalize_list(parsed.get("in_scene_repetition"))
    if not isinstance(parsed.get("scene_summary"), dict):
        parsed["scene_summary"] = empty_scene_result(scene_id, scene_text)["scene_summary"]
    raw_record["parsed"] = parsed
    append_raw(raw_path, raw_record)
    return parsed


def judge_json_prompt(
    client: Any,
    config: JudgeConfig,
    system: str,
    prompt: str,
    raw_path: Path,
    raw_record: Dict[str, Any],
    expected_schema: str,
) -> Optional[Dict[str, Any]]:
    """Call a judge prompt with empty-response retry and JSON repair."""
    try:
        response_text = call_judge(client, config, system, prompt)
        raw_record["raw_response"] = response_text
    except Exception as call_error:
        raw_record["error"] = str(call_error)
        raw_record["error_stage"] = "judge_call"
        append_raw(raw_path, raw_record)
        return None

    if not response_text.strip():
        raw_record["empty_response_retry"] = True
        try:
            response_text = retry_empty_response(client, config, prompt, expected_schema)
            raw_record["retry_response"] = response_text
        except Exception as retry_error:
            raw_record["error"] = f"empty response; retry failed: {retry_error}"
            raw_record["error_stage"] = "empty_response_retry"
            append_raw(raw_path, raw_record)
            return None
        if not response_text.strip():
            raw_record["error"] = "Judge returned an empty response body twice."
            raw_record["error_stage"] = "empty_response"
            append_raw(raw_path, raw_record)
            return None

    try:
        parsed = parse_json_object(response_text)
    except Exception as first_error:
        try:
            repaired = repair_json_response(client, config, response_text, expected_schema)
            parsed = parse_json_object(repaired)
            raw_record["repaired"] = True
            raw_record["repair_response"] = repaired
        except Exception as second_error:
            raw_record["error"] = f"{first_error}; repair failed: {second_error}"
            raw_record["error_stage"] = "json_parse"
            append_raw(raw_path, raw_record)
            return None

    raw_record["parsed"] = parsed
    append_raw(raw_path, raw_record)
    return parsed


def judge_hallucination_chunk(
    client: Any,
    config: JudgeConfig,
    scene_id: str,
    scene_text: str,
    graph_context: Dict[str, Any],
    raw_path: Path,
    validate_only: bool,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Dict[str, Any]:
    base = empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count)
    base["hallucinations"] = []
    if validate_only:
        append_raw(raw_path, {
            "kind": "hallucination",
            "scene_id": scene_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "validate_only": True,
            "parsed": {
                "scene_id": scene_id,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "hallucinations": [],
            },
        })
        return base

    system, prompt = hallucination_prompt(scene_id, scene_text, graph_context, chunk_index, chunk_count)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape.", 1)[-1]
    raw_record = {
        "kind": "hallucination",
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "repaired": False,
    }
    parsed = judge_json_prompt(client, config, system, prompt, raw_path, raw_record, expected_schema)
    if parsed is None:
        base["judge_error"] = raw_record.get("error", "judge failed")
        return base
    parsed.setdefault("scene_id", scene_id)
    parsed["chunk_index"] = chunk_index
    parsed["chunk_count"] = chunk_count
    parsed["hallucinations"] = normalize_list(parsed.get("hallucinations"))
    parsed.setdefault("in_scene_repetition", [])
    parsed.setdefault("scene_summary", base["scene_summary"])
    return parsed


def judge_in_scene_repetition_chunk(
    client: Any,
    config: JudgeConfig,
    scene_id: str,
    scene_text: str,
    raw_path: Path,
    validate_only: bool,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> Dict[str, Any]:
    base = empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count)
    base["in_scene_repetition"] = []
    if validate_only:
        append_raw(raw_path, {
            "kind": "in_scene_repetition",
            "scene_id": scene_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "validate_only": True,
            "parsed": {
                "scene_id": scene_id,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "in_scene_repetition": [],
            },
        })
        return base

    system, prompt = in_scene_repetition_prompt(scene_id, scene_text, chunk_index, chunk_count)
    expected_schema = prompt.rsplit("Return JSON with this exact top-level shape.", 1)[-1]
    raw_record = {
        "kind": "in_scene_repetition",
        "scene_id": scene_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "repaired": False,
    }
    parsed = judge_json_prompt(client, config, system, prompt, raw_path, raw_record, expected_schema)
    if parsed is None:
        base["judge_error"] = raw_record.get("error", "judge failed")
        return base
    parsed.setdefault("scene_id", scene_id)
    parsed["chunk_index"] = chunk_index
    parsed["chunk_count"] = chunk_count
    parsed["in_scene_repetition"] = normalize_list(parsed.get("in_scene_repetition"))
    parsed.setdefault("hallucinations", [])
    parsed.setdefault("scene_summary", base["scene_summary"])
    return parsed


def merge_chunk_judgments(
    scene_id: str,
    scene_text: str,
    hallucination_result: Dict[str, Any],
    repetition_result: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_index = hallucination_result.get("chunk_index", repetition_result.get("chunk_index", 1))
    chunk_count = hallucination_result.get("chunk_count", repetition_result.get("chunk_count", 1))
    result = empty_chunk_result(scene_id, scene_text, chunk_index, chunk_count)
    result["hallucinations"] = normalize_list(hallucination_result.get("hallucinations"))
    result["in_scene_repetition"] = normalize_list(repetition_result.get("in_scene_repetition"))
    errors = []
    if hallucination_result.get("judge_error"):
        errors.append({"kind": "hallucination", "error": hallucination_result.get("judge_error")})
    if repetition_result.get("judge_error"):
        errors.append({"kind": "in_scene_repetition", "error": repetition_result.get("judge_error")})
    if errors:
        result["judge_error"] = json.dumps(errors, ensure_ascii=False)
    return result


def merge_scene_chunk_results(scene_id: str, scene_text: str, chunk_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hallucinations: List[Dict[str, Any]] = []
    in_scene_repetition: List[Dict[str, Any]] = []
    judge_errors: List[Dict[str, Any]] = []
    summaries: List[str] = []
    key_beats: List[str] = []

    for result in chunk_results:
        chunk_index = result.get("chunk_index", 1)
        for item in normalize_list(result.get("hallucinations")):
            row = dict(item)
            row.setdefault("chunk_index", chunk_index)
            hallucinations.append(row)
        for item in normalize_list(result.get("in_scene_repetition")):
            row = dict(item)
            row.setdefault("chunk_index", chunk_index)
            in_scene_repetition.append(row)
        if result.get("judge_error"):
            judge_errors.append({"chunk_index": chunk_index, "error": result.get("judge_error")})
        summary = result.get("scene_summary")
        if isinstance(summary, dict):
            if summary.get("summary"):
                summaries.append(str(summary.get("summary")))
            for beat in summary.get("key_beats", []) or []:
                beat = str(beat)
                if beat and beat not in key_beats:
                    key_beats.append(beat)

    merged: Dict[str, Any] = {
        "scene_id": scene_id,
        "chunk_count": len(chunk_results),
        "hallucinations": hallucinations,
        "in_scene_repetition": in_scene_repetition,
        "scene_summary": {
            "summary": compact_text(" ".join(summaries) or scene_text, 400),
            "key_beats": key_beats[:20],
            "confidence": 0.0 if judge_errors else 1.0,
        },
    }
    if judge_errors:
        merged["judge_errors"] = judge_errors
    return merged


def judge_cross_scene(
    client: Any,
    config: JudgeConfig,
    scene_results: Sequence[Dict[str, Any]],
    raw_path: Path,
    validate_only: bool,
) -> List[Dict[str, Any]]:
    summaries = []
    for result in scene_results:
        summaries.append({
            "scene_id": result.get("scene_id"),
            "scene_summary": result.get("scene_summary", {}),
            "hallucination_snippets": [
                item.get("snippet", "") for item in normalize_list(result.get("hallucinations"))
            ][:8],
            "in_scene_repetition_snippets": [
                item.get("snippet", "") for item in normalize_list(result.get("in_scene_repetition"))
            ][:8],
        })

    if validate_only or len(summaries) < 2:
        append_raw(raw_path, {
            "kind": "cross_scene",
            "validate_only": validate_only,
            "parsed": {"cross_scene_repetition": []},
        })
        return []

    system, prompt = cross_scene_prompt(summaries)
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


def build_detail_rows(scene_results: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hallucination_rows: List[Dict[str, Any]] = []
    repetition_rows: List[Dict[str, Any]] = []
    for result in scene_results:
        scene_id = str(result.get("scene_id", ""))
        for item in normalize_list(result.get("hallucinations")):
            hallucination_rows.append({
                "scene_id": scene_id,
                "chunk_index": item.get("chunk_index", ""),
                "snippet": item.get("snippet", ""),
                "reason": item.get("reason", ""),
                "severity": item.get("severity", ""),
                "confidence": item.get("confidence", ""),
                "graph_evidence": item.get("graph_evidence", []),
            })
        for item in normalize_list(result.get("in_scene_repetition")):
            repetition_rows.append({
                "scene_id": scene_id,
                "chunk_index": item.get("chunk_index", ""),
                "snippet": item.get("snippet", ""),
                "positions": item.get("positions", []),
                "reason": item.get("reason", ""),
                "confidence": item.get("confidence", ""),
            })
    return hallucination_rows, repetition_rows


def compute_summary(
    scene_count: int,
    hallucination_rows: Sequence[Dict[str, Any]],
    in_scene_rows: Sequence[Dict[str, Any]],
    cross_scene_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    graph_path: Path,
    text_path: Path,
) -> Dict[str, Any]:
    hallucinations_by_scene: Dict[str, int] = {}
    for row in hallucination_rows:
        scene_id = str(row.get("scene_id", ""))
        hallucinations_by_scene[scene_id] = hallucinations_by_scene.get(scene_id, 0) + 1

    possible_pairs = scene_count * (scene_count - 1) // 2
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph_path": str(graph_path),
            "text_path": str(text_path),
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_style": args.api_style,
            "validate_only": bool(args.validate_only),
            "max_graph_items_per_scene": args.max_graph_items_per_scene,
            "max_chunk_chars": getattr(args, "max_chunk_chars", None),
        },
        "metrics": {
            "total_scenes_judged": scene_count,
            "total_hallucination_snippets": len(hallucination_rows),
            "hallucination_snippets_per_scene": hallucinations_by_scene,
            "hallucination_rate_per_scene": len(hallucination_rows) / max(1, scene_count),
            "total_in_scene_repetition_snippets": len(in_scene_rows),
            "in_scene_repetition_rate": len(in_scene_rows) / max(1, scene_count),
            "total_cross_scene_repeated_pairs": len(cross_scene_rows),
            "possible_cross_scene_pairs": possible_pairs,
            "cross_scene_repetition_rate": len(cross_scene_rows) / max(1, possible_pairs),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Azure AI Foundry/OpenAI-v1-compatible LLM-as-judge benchmark."
    )
    parser.add_argument("--graph_path", required=True, help="Path to final_graph.json or similar graph JSON.")
    parser.add_argument("--text_path", required=True, help="Path to generated scene text .txt or .json.")
    parser.add_argument("--output_dir", required=True, help="Directory for benchmark outputs.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", ""),
        help="Foundry/OpenAI-v1-compatible endpoint, often ending in /openai/v1/.",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("AZURE_AI_FOUNDRY_API_KEY", ""),
        help="Foundry API key. Defaults to AZURE_AI_FOUNDRY_API_KEY.",
    )
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
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Parse inputs and write empty output files without calling a judge deployment.",
    )
    return parser.parse_args(argv)


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

    client = None
    config = JudgeConfig(
        endpoint=args.endpoint,
        api_key=args.api_key,
        deployment=args.deployment,
        api_style=args.api_style,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if not args.validate_only:
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
        client = ensure_openai_client(config)

    scene_results: List[Dict[str, Any]] = []
    for scene_id, scene_text in scenes.items():
        graph_context = build_graph_context(graph, scene_id, args.max_graph_items_per_scene)
        chunks = chunk_scene_text(scene_text, args.max_chunk_chars)
        chunk_results = []
        for chunk_index, chunk_text in chunks:
            chunk_results.append(
                judge_scene(
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
            )
        scene_results.append(merge_scene_chunk_results(scene_id, scene_text, chunk_results))

    cross_scene_rows = judge_cross_scene(client, config, scene_results, raw_path, args.validate_only)
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
    write_csv(
        output_dir / "cross_scene_repetition.csv",
        ["scene_i_id", "scene_j_id", "repeated_beat", "snippets", "reason", "confidence"],
        cross_scene_rows,
    )

    summary = compute_summary(
        scene_count=len(scene_results),
        hallucination_rows=hallucination_rows,
        in_scene_rows=in_scene_rows,
        cross_scene_rows=cross_scene_rows,
        args=args,
        graph_path=graph_path,
        text_path=text_path,
    )
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
