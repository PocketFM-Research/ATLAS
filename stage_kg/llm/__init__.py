"""LLM backend factory."""

from typing import Optional
from .base import BaseLLM


def get_llm(
    provider: str,
    model=None,
    api_key=None,
    base_url=None,
) -> BaseLLM:
    """
    Factory function to instantiate the correct LLM backend.

    Args:
        provider: One of 'openai', 'anthropic', 'vllm'.
        model: Model name/ID (provider-specific default if None).
        api_key: API key (reads from env if None).
        base_url: Override endpoint URL (mainly for vllm).
    """
    provider = provider.lower()

    if provider == "openai":
        from .openai_client import OpenAILLM
        return OpenAILLM(model=model or "gpt-4o", api_key=api_key)

    elif provider == "anthropic":
        from .anthropic_client import AnthropicLLM
        return AnthropicLLM(model=model or "claude-sonnet-4-6", api_key=api_key)

    elif provider == "vllm":
        from .vllm_client import VLLMLLm
        return VLLMLLm(
            model=model or "mistralai/Mistral-7B-Instruct-v0.2",
            base_url=base_url or "http://localhost:8000/v1",
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: openai, anthropic, vllm"
        )
