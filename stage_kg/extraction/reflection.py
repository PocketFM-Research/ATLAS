"""
Reflection-based acceptance loop (Appendix C.3).

After each LLM extraction, score the result 0-10.
If score < ACCEPTANCE_THRESHOLD, re-extract using feedback as guidance,
up to MAX_RETRIES times. Keep the highest-scoring result across all attempts.
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

from ..llm.base import BaseLLM
from ..prompts.reflection import (
    ACCEPTANCE_THRESHOLD,
    MAX_RETRIES,
    SYSTEM_PROMPT,
    build_event_reflection_prompt,
    build_entity_reflection_prompt,
    build_relation_reflection_prompt,
)
from ..utils.json_repair import parse_llm_json
from ..utils.logging_utils import PromptLogger

logger = logging.getLogger(__name__)


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
        max_tokens=512,
    )
    if prompt_logger:
        prompt_logger.log("reflection", scene_id, reflection_prompt, raw, llm.model_id)

    result = parse_llm_json(raw, schema_hint="reflection_score")
    if not isinstance(result, dict):
        logger.warning("Reflection scoring returned unparseable JSON for scene=%s", scene_id)
        return 5, "Could not parse reflection score; proceeding."  # assume mediocre

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
    Run extraction with bounded reflection retries.

    Args:
        extract_fn: Callable(feedback) -> (result, raw_prompt).
                    On first call, feedback is None.
                    On retries, feedback is the previous reflection's feedback string.
        reflect_fn: Callable(result) -> reflection_prompt_string.
        llm: LLM backend.
        scene_id: For logging.
        prompt_logger: Optional prompt logger.

    Returns:
        The highest-scoring result across all attempts (or the last one if all fail).
    """
    best_result = None
    best_score = -1

    feedback: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        result, _ = extract_fn(feedback)

        if result is None:
            logger.warning(
                "Extraction returned None on attempt %d/%d for scene=%s",
                attempt, MAX_RETRIES, scene_id,
            )
            continue

        reflection_prompt = reflect_fn(result)
        score, feedback = score_extraction(llm, reflection_prompt, scene_id, prompt_logger)

        if score > best_score:
            best_score = score
            best_result = result
            logger.debug(
                "New best result (score=%d) on attempt %d for scene=%s",
                score, attempt, scene_id,
            )

        if score >= ACCEPTANCE_THRESHOLD:
            logger.debug(
                "Accepted on attempt %d (score=%d >= %d) for scene=%s",
                attempt, score, ACCEPTANCE_THRESHOLD, scene_id,
            )
            break
        else:
            logger.info(
                "Score %d < %d on attempt %d for scene=%s — retrying with feedback: %s",
                score, ACCEPTANCE_THRESHOLD, attempt, scene_id, feedback,
            )

    if best_score < ACCEPTANCE_THRESHOLD:
        logger.warning(
            "All %d attempts scored below %d for scene=%s; using best (score=%d)",
            MAX_RETRIES, ACCEPTANCE_THRESHOLD, scene_id, best_score,
        )

    return best_result
