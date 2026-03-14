"""Base LLM interface."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a prompt and return the completion text.

        Args:
            prompt: User-facing prompt text.
            system: Optional system message.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw completion string.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return a human-readable model identifier."""
        ...
