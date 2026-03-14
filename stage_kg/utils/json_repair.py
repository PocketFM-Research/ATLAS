"""
JSON validation and repair utilities for LLM outputs.

LLMs sometimes produce near-valid JSON: trailing commas, missing brackets,
markdown code fences, etc.  This module attempts lightweight repairs before
falling back to a full rejection.
"""

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_block(text: str) -> str:
    """Strip markdown code fences and extract the first JSON block."""
    # Remove ```json ... ``` or ``` ... ```
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find first { or [ to start of JSON
    for start_char, end_char in (("{", "}"), ("[", "]")):
        idx = text.find(start_char)
        if idx != -1:
            # Find matching end — scan for last occurrence
            last_idx = text.rfind(end_char)
            if last_idx > idx:
                return text[idx : last_idx + 1]

    return text


def repair_json(text: str) -> Optional[str]:
    """
    Attempt to repair malformed JSON strings.

    Returns repaired JSON string or None if repair fails.
    """
    text = extract_json_block(text)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single quotes used as string delimiters (simple heuristic)
    # Only replace when clearly used as JSON string quotes, not inside strings
    # This is a rough pass — be conservative
    # text = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', text)

    # Remove comments (// and /* */)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


def parse_llm_json(raw: str, schema_hint: str = "") -> Optional[Any]:
    """
    Parse JSON from an LLM response with repair attempts.

    Args:
        raw: Raw LLM output string.
        schema_hint: Human-readable description of expected schema for logging.

    Returns:
        Parsed Python object or None if all attempts fail.
    """
    if not raw or not raw.strip():
        logger.warning("Empty LLM response; cannot parse JSON")
        return None

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

    # Attempt 3: repair then parse
    repaired = repair_json(raw)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON repair failed (schema=%s): %s | raw snippet: %s",
                schema_hint,
                e,
                raw[:200],
            )

    return None


def validate_event_list(data: Any) -> bool:
    """Validate that data is a list of event dicts with required fields."""
    if not isinstance(data, list):
        return False
    required = {"name", "description", "scene_id"}
    for item in data:
        if not isinstance(item, dict):
            return False
        if not required.issubset(item.keys()):
            return False
    return True


def validate_entity_list(data: Any) -> bool:
    """Validate that data is a list of entity dicts with required fields."""
    if not isinstance(data, list):
        return False
    required = {"name", "type", "scene_id"}
    for item in data:
        if not isinstance(item, dict):
            return False
        if not required.issubset(item.keys()):
            return False
    return True


def validate_relation_list(data: Any) -> bool:
    """Validate that data is a list of relation dicts with required fields."""
    if not isinstance(data, list):
        return False
    required = {"source", "relation", "target"}
    for item in data:
        if not isinstance(item, dict):
            return False
        if not required.issubset(item.keys()):
            return False
    return True
