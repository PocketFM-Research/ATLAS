"""OpenAI LLM backend with retry logic."""

import logging
import time
from typing import Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI chat-completions backend."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    @property
    def model_id(self) -> str:
        return f"openai/{self._model}"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if _is_content_filter_error(e):
                    logger.warning(
                        "OpenAI request blocked by content filter; returning empty response so pipeline can continue"
                    )
                    return ""
                logger.warning(
                    "OpenAI attempt %d/%d failed: %s", attempt, self._max_retries, e
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
                else:
                    raise
        return ""


def _is_content_filter_error(error: Exception) -> bool:
    """Return True for Azure/OpenAI content-filter BadRequest errors."""
    code = getattr(error, "code", None)
    if code == "content_filter":
        return True

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") == "content_filter":
            return True

    return "content_filter" in str(error)
