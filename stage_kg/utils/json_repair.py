"""
JSON validation and repair utilities for LLM outputs.

LLMs sometimes produce near-valid JSON: trailing commas, missing brackets,
markdown code fences, truncated output.  This module attempts repairs in
escalating order before falling back to a full rejection.
"""

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ detection

def is_truncated(text: str) -> bool:
    """
    Heuristic: return True if the response looks like it was cut off mid-JSON.

    Symptoms: unmatched brackets/braces, string ends inside a value without
    closing ] or }, response ends with a comma or a partial key.
    """
    text = text.strip()
    if not text:
        return False

    # Count unmatched open brackets/braces
    depth = 0
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                depth += 1
            elif ch in ("}", "]"):
                depth -= 1

    # If we're still inside open structures, it's truncated
    if depth > 0:
        return True

    # Ends with a comma (incomplete array/object member)
    if re.search(r",\s*$", text):
        return True

    return False


# ------------------------------------------------------------------ extraction

def extract_json_block(text: str) -> str:
    """Strip markdown code fences and extract the first JSON object or array."""
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Find first [ or { and the matching closing bracket using a depth counter
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        # Depth never reached 0 — truncated; return everything from start
        return text[start:]

    return text


# ------------------------------------------------------------------ repair

def repair_json(text: str) -> Optional[str]:
    """
    Attempt lightweight repairs on malformed JSON.

    Handles: trailing commas, JS comments, control characters, truncation
    (by closing open brackets/braces).
    """
    text = extract_json_block(text)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Remove JS-style comments
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Strip control characters (except \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # If still truncated, close any open brackets/braces
    text = _close_open_structures(text)

    return text


def _close_open_structures(text: str) -> str:
    """Append closing brackets/braces to fix truncated JSON."""
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]") and stack:
                stack.pop()

    # Remove trailing comma before we close
    text = re.sub(r",\s*$", "", text.rstrip())

    # Close open structures in reverse
    return text + "".join(reversed(stack))


# ------------------------------------------------------------------ parse

def parse_llm_json(raw: str, schema_hint: str = "") -> Optional[Any]:
    """
    Parse JSON from an LLM response with escalating repair attempts.

    Returns parsed Python object, or None if all attempts fail.
    Logs a distinct WARNING when truncation is detected.
    """
    if not raw or not raw.strip():
        logger.warning("Empty LLM response (schema=%s)", schema_hint)
        return None

    if is_truncated(raw):
        logger.warning(
            "LLM response appears TRUNCATED (schema=%s) — attempting repair. "
            "Consider increasing max_tokens. Tail: ...%s",
            schema_hint,
            raw[-120:],
        )

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract JSON block then parse
    extracted = extract_json_block(raw)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair (including bracket closing) then parse
    repaired = repair_json(raw)
    if repaired:
        try:
            result = json.loads(repaired)
            logger.debug("JSON repaired successfully (schema=%s)", schema_hint)
            return result
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON repair failed (schema=%s): %s | tail: ...%s",
                schema_hint, e, raw[-200:],
            )

    return None


# ------------------------------------------------------------------ validators

def validate_event_list(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    required = {"name", "description", "scene_id"}
    return all(isinstance(i, dict) and required.issubset(i) for i in data)


def validate_entity_list(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    required = {"name", "type", "scene_id"}
    return all(isinstance(i, dict) and required.issubset(i) for i in data)


def validate_relation_list(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    required = {"source_id", "relation", "target_id"}
    return all(isinstance(i, dict) and required.issubset(i) for i in data)
