import logging
from typing import Optional

from .openai_client import OpenAILLM

logger = logging.getLogger(__name__)


class VLLMLLm(OpenAILLM):
    """
    vLLM backend using the OpenAI-compatible endpoint.

    Typically served at http://localhost:8000/v1 with a locally loaded model.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        super().__init__(
            model=model,
            api_key="EMPTY",  # vLLM doesn't require a real key
            base_url=base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    @property
    def model_id(self) -> str:
        return f"vllm/{self._model}"
