import logging
import time
from typing import Optional
from urllib.parse import parse_qs, urlsplit, urlunsplit

from .base import BaseLLM

logger = logging.getLogger(__name__)


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI chat-completions backend."""

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        if not api_key:
            raise ValueError("AzureOpenAILLM requires an api_key.")
        if not base_url:
            raise ValueError("AzureOpenAILLM requires base_url (Azure endpoint).")

        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._base_url, inferred_api_version = _normalize_azure_url(base_url)
        self._api_version = api_version or inferred_api_version

        openai_compatible_url = _is_openai_compatible_url(self._base_url)

        if openai_compatible_url:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install openai: pip install openai") from exc

            # Strip a trailing /chat/completions if present so the SDK can append it.
            url = self._base_url
            for suffix in ("/chat/completions", "/completions"):
                if url.endswith(suffix):
                    url = url[: -len(suffix)]
            url = url.rstrip("/")

            default_query = (
                {"api-version": self._api_version}
                if self._api_version and _should_send_api_version(url)
                else None
            )

            self._mode = "openai-with-base-url"
            self._client = OpenAI(
                api_key=api_key,
                base_url=url,
                default_query=default_query,
                default_headers={"api-key": api_key},
            )
        else:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise ImportError("Install openai: pip install openai") from exc

            if not self._api_version:
                raise ValueError(
                    "Azure endpoint without /openai/deployments/ in URL requires "
                    "--api_version (e.g. 2024-12-01-preview)."
                )

            self._mode = "azure-openai"
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self._base_url,
                api_version=self._api_version,
            )

    @property
    def model_id(self) -> str:
        return f"azure_openai/{self._model}"

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
                        "Azure OpenAI request blocked by content filter; returning empty"
                    )
                    return ""
                logger.warning(
                    "Azure OpenAI attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
                else:
                    raise
        return ""


def _is_content_filter_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    if code == "content_filter":
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") == "content_filter":
            return True
    return "content_filter" in str(error)


def _normalize_azure_url(raw_url: str) -> tuple[str, Optional[str]]:
    """
    Accept either a plain Azure endpoint or a copied Azure AI Foundry Target URI.

    Target URIs are often pasted as:
      .../openai/deployments/<deployment>/chat/completions?api-version=...

    The OpenAI SDK wants base_url without the final /chat/completions and with
    api-version supplied separately as a default query, so normalize that here.
    """
    cleaned = raw_url.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    query = parse_qs(parsed.query)
    inferred_api_version = None
    if "api-version" in query and query["api-version"]:
        inferred_api_version = query["api-version"][0]

    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    normalized = urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))
    return normalized.rstrip("/"), inferred_api_version


def _is_openai_compatible_url(url: str) -> bool:
    """
    True when the URL should be used with the standard OpenAI client.

    Azure AI Foundry / Azure AI Services "Target URI" endpoints often live on
    *.services.ai.azure.com instead of *.openai.azure.com. They are still
    OpenAI-compatible REST URLs, not classic AzureOpenAI resource endpoints.
    """
    parsed = urlsplit(url)
    return (
        "/openai/deployments/" in parsed.path
        or "/models" in parsed.path
        or "/openai/v1" in parsed.path
        or parsed.netloc.endswith(".services.ai.azure.com")
    )


def _should_send_api_version(url: str) -> bool:
    """The newer /openai/v1 endpoints reject api-version query parameters."""
    parsed = urlsplit(url)
    return "/openai/deployments/" in parsed.path and "/openai/v1" not in parsed.path
