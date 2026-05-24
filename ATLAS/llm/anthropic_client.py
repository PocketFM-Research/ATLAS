import logging
import time
from typing import Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic Messages API backend."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)

    @property
    def model_id(self) -> str:
        return f"anthropic/{self._model}"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs = {
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system

                resp = self._client.messages.create(**kwargs)
                return resp.content[0].text
            except Exception as e:
                logger.warning(
                    "Anthropic attempt %d/%d failed: %s", attempt, self._max_retries, e
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
                else:
                    raise
        return ""
