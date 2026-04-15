"""
Reflection-based acceptance check (Appendix C.3).

After each LLM extraction, score the result 0-10 once and keep the result.
"""

import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from ..llm.base import BaseLLM
from ..prompts.reflection import SYSTEM_PROMPT, build_event_reflection_prompt, build_entity_reflection_prompt, build_relation_reflection_prompt
from ..utils.json_repair import parse_llm_json
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)


def _salvage_partial_score(raw: str) -> Optional[dict]:
    """Best-effort recovery for truncated reflection JSON."""
    values = {}
    for key in ("accuracy", "consistency", "redundancy", "overall"):
        match = re.search(rf'"{key}"\s*:\s*(\d+)', raw)
        if match:
            try:
                values[key] = int(match.group(1))
            except ValueError:
                continue

    if not values:
        return None

    if "overall" not in values:
        component_scores = [
            values[key]
            for key in ("accuracy", "consistency", "redundancy")
            if key in values
        ]
        if component_scores:
            values["overall"] = round(sum(component_scores) / len(component_scores))

    feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*)', raw, re.S)
    values["feedback"] = feedback_match.group(1).strip() if feedback_match else ""
    return values


def score_extraction(
    llm: BaseLLM,
    reflection_prompt: str,
    scene_id: str,
    prompt_logger: Optional[PromptLogger],
) -> Tuple[int, str]:
    """
    Ask the LLM to score an extraction.

    Returns (overall_score, feedback_text).
    """
    raw = llm.complete(
        reflection_prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=1024,
    )
    if prompt_logger:
        prompt_logger.log("reflection", scene_id, reflection_prompt, raw, llm.model_id)

    result = parse_llm_json(raw, schema_hint="reflection_score")
    if not isinstance(result, dict):
        result = _salvage_partial_score(raw)

    if not isinstance(result, dict):
        logger.warning("Reflection scoring returned unparseable JSON for scene=%s", scene_id)
        return 5, "Could not parse reflection score; proceeding."  # assume mediocre

    if "overall" not in result:
        recovered = _salvage_partial_score(raw)
        if recovered and "overall" in recovered:
            result = {**recovered, **result}

    score = int(result.get("overall", 5))
    feedback = result.get("feedback", "")
    logger.debug(
        "Reflection score=%d (acc=%s, con=%s, red=%s) for scene=%s: %s",
        score,
        result.get("accuracy"),
        result.get("consistency"),
        result.get("redundancy"),
        scene_id,
        feedback,
    )
    return score, feedback


def reflection_loop(
    extract_fn: Callable[[Optional[str]], Tuple[Any, str]],
    reflect_fn: Callable[[Any], str],
    llm: BaseLLM,
    scene_id: str,
    prompt_logger: Optional[PromptLogger],
) -> Any:
    """
    Run extraction with a single reflection score.

    Args:
        extract_fn: Callable(feedback) -> (result, raw_prompt).
                    Feedback is always None because retries are disabled.
        reflect_fn: Callable(result) -> reflection_prompt_string.
        llm: LLM backend.
        scene_id: For logging.
        prompt_logger: Optional prompt logger.

    Returns:
        The extracted result from the single attempt.
    """
    result, _ = extract_fn(None)

    if result is None:
        logger.warning("Extraction returned None for scene=%s", scene_id)
        return None

    reflection_prompt = reflect_fn(result)
    score, feedback = score_extraction(llm, reflection_prompt, scene_id, prompt_logger)

    if score < 7:
        logger.warning(
            "Reflection score below threshold for scene=%s (score=%d); keeping single-pass result. Feedback: %s",
            scene_id,
            score,
            feedback,
        )

    return result
