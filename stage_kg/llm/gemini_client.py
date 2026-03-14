"""Gemini LLM backend using google-genai SDK."""

import logging
import time
from typing import Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini backend via google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        from google import genai as _genai
        self._client = _genai.Client(api_key=api_key)
        self._types = types

    @property
    def model_id(self) -> str:
        return f"gemini/{self._model}"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        # Prepend system prompt to user prompt since Gemini's system_instruction
        # is set at client level; easiest to merge them inline
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=full_prompt,
                    config=config,
                )
                return resp.text or ""
            except Exception as e:
                logger.warning(
                    "Gemini attempt %d/%d failed: %s", attempt, self._max_retries, e
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
                else:
                    raise
        return ""
