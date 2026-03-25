"""Core module — LLMProxy class and provider implementations."""

from __future__ import annotations

import abc
import logging
import random
import time
from typing import Any

import httpx

from llmproxy.config import ProviderConfig, RetryConfig
from llmproxy.utils import (
    CompletionResponse,
    Timer,
    build_anthropic_payload,
    build_openai_payload,
    estimate_cost as _estimate_cost,
    normalise_anthropic_response,
    normalise_openai_response,
)

logger = logging.getLogger("llmproxy")

# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------


class BaseProvider(abc.ABC):
    """Abstract base for all LLM provider implementations."""

    name: str

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abc.abstractmethod
    async def complete(self, prompt: str, model: str, **kwargs: Any) -> CompletionResponse:
        """Send a completion request and return a normalised response."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return a dict with at least a ``healthy`` boolean key."""

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """Return the list of models this provider exposes."""


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]


class OpenAIProvider(BaseProvider):
    """Provider for the OpenAI chat completions API."""

    name = "openai"

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> CompletionResponse:
        model = model or self.config.default_model or "gpt-4o"
        payload = build_openai_payload(prompt, model, **kwargs)
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            with Timer() as t:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        return normalise_openai_response(data, self.name, t.elapsed_ms)

    async def health_check(self) -> dict[str, Any]:
        url = f"{self.config.base_url}/models"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
            return {"healthy": resp.status_code == 200, "status_code": resp.status_code}
        except httpx.HTTPError as exc:
            return {"healthy": False, "error": str(exc)}

    def list_models(self) -> list[str]:
        return list(_OPENAI_MODELS)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

_ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]


class AnthropicProvider(BaseProvider):
    """Provider for the Anthropic messages API."""

    name = "anthropic"

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> CompletionResponse:
        model = model or self.config.default_model or "claude-sonnet-4-20250514"
        payload = build_anthropic_payload(prompt, model, **kwargs)
        url = f"{self.config.base_url}/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            with Timer() as t:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        return normalise_anthropic_response(data, self.name, t.elapsed_ms)

    async def health_check(self) -> dict[str, Any]:
        """Anthropic has no lightweight health endpoint; we attempt a tiny request."""
        try:
            response = await self.complete("hi", model="claude-3-haiku-20240307", max_tokens=1)
            return {"healthy": True, "model": response.model}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def list_models(self) -> list[str]:
        return list(_ANTHROPIC_MODELS)


# ---------------------------------------------------------------------------
# Mock provider (for testing / offline work)
# ---------------------------------------------------------------------------


class MockProvider(BaseProvider):
    """A provider that returns deterministic responses without network calls."""

    name = "mock"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config or ProviderConfig())
        self._call_count = 0

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> CompletionResponse:
        self._call_count += 1
        content = f"Mock response to: {prompt[:80]}"
        return CompletionResponse(
            content=content,
            model=model or "mock-v1",
            provider=self.name,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(content) // 4,
            total_tokens=(len(prompt) + len(content)) // 4,
            latency_ms=0.1,
        )

    async def health_check(self) -> dict[str, Any]:
        return {"healthy": True, "calls": self._call_count}

    def list_models(self) -> list[str]:
        return ["mock-v1"]


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}

# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class _UsageRecord:
    """Mutable usage accumulator for a single provider."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_latency_ms: float = 0.0

    def record(self, resp: CompletionResponse) -> None:
        self.request_count += 1
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        self.total_tokens += resp.total_tokens
        self.total_latency_ms += resp.latency_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 8),
            "avg_latency_ms": (
                round(self.total_latency_ms / self.request_count, 2)
                if self.request_count
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# LLMProxy — main entry point
# ---------------------------------------------------------------------------


class LLMProxy:
    """Unified interface for multiple LLM providers.

    Features
    --------
    * ``complete()``          — send a prompt and get a normalised response.
    * ``add_provider()``      — register a provider with its configuration.
    * ``set_fallback_chain()``— define automatic fallback order.
    * ``get_available_models()`` — list models across all registered providers.
    * ``estimate_cost()``     — estimate the cost of a request before sending.
    * ``get_usage_stats()``   — retrieve accumulated usage statistics.
    * ``health_check()``      — check provider availability.
    * ``configure_retry()``   — tune retry / backoff behaviour.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._fallback_chain: list[str] = []
        self._retry_config = RetryConfig()
        self._usage: dict[str, _UsageRecord] = {}

    # -- Provider management ------------------------------------------------

    def add_provider(self, name: str, config: dict[str, Any] | ProviderConfig) -> None:
        """Register a provider.

        *name* should match a key in ``PROVIDER_REGISTRY`` (``openai``,
        ``anthropic``, ``mock``) or you can pass any ``BaseProvider`` subclass
        instance via ``add_provider_instance()``.
        """
        if isinstance(config, dict):
            config = ProviderConfig(**config)
        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Available: {', '.join(PROVIDER_REGISTRY.keys())}"
            )
        self._providers[name] = cls(config)
        self._usage.setdefault(name, _UsageRecord())
        logger.info("Registered provider: %s", name)

    def add_provider_instance(self, name: str, provider: BaseProvider) -> None:
        """Register an already-instantiated provider."""
        self._providers[name] = provider
        self._usage.setdefault(name, _UsageRecord())

    def set_fallback_chain(self, providers: list[str]) -> None:
        """Define the ordered fallback chain of provider names."""
        for p in providers:
            if p not in self._providers:
                raise ValueError(f"Provider '{p}' is not registered.")
        self._fallback_chain = list(providers)
        logger.info("Fallback chain set: %s", " -> ".join(self._fallback_chain))

    # -- Completion ---------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        model: str = "",
        provider: str = "",
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a completion request, honouring retries and fallback.

        Parameters
        ----------
        prompt : str
            The user prompt / message.
        model : str, optional
            Model identifier (e.g. ``gpt-4o``).  Falls back to the provider
            default when omitted.
        provider : str, optional
            Preferred provider name.  When omitted the first entry in the
            fallback chain is used.
        """
        chain = self._resolve_chain(provider)
        last_error: Exception | None = None

        for prov_name in chain:
            prov = self._providers[prov_name]
            for attempt in range(1, self._retry_config.max_retries + 1):
                try:
                    resp = await prov.complete(prompt, model, **kwargs)
                    self._usage[prov_name].record(resp)
                    return resp
                except Exception as exc:
                    last_error = exc
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Provider %s attempt %d failed: %s — retrying in %.1fs",
                        prov_name,
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            logger.error("Provider %s exhausted retries.", prov_name)

        raise RuntimeError(
            f"All providers in chain {chain} failed. Last error: {last_error}"
        )

    # -- Informational ------------------------------------------------------

    def get_available_models(self) -> dict[str, list[str]]:
        """Return a mapping of provider name to its available models."""
        return {name: prov.list_models() for name, prov in self._providers.items()}

    def estimate_cost(
        self,
        prompt: str,
        model: str = "gpt-4o",
        expected_completion_tokens: int = 256,
    ) -> dict[str, Any]:
        """Estimate the cost of a completion without making a request."""
        return _estimate_cost(prompt, model, expected_completion_tokens)

    def get_usage_stats(self) -> dict[str, dict[str, Any]]:
        """Return accumulated usage statistics per provider."""
        return {name: rec.as_dict() for name, rec in self._usage.items()}

    async def health_check(self, provider: str) -> dict[str, Any]:
        """Run a health check against the given *provider*."""
        if provider not in self._providers:
            raise ValueError(f"Provider '{provider}' is not registered.")
        return await self._providers[provider].health_check()

    # -- Configuration ------------------------------------------------------

    def configure_retry(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        jitter: bool = True,
    ) -> None:
        """Update retry and backoff settings."""
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            jitter=jitter,
        )
        logger.info("Retry config updated: %s", self._retry_config.model_dump())

    # -- Internals ----------------------------------------------------------

    def _resolve_chain(self, preferred: str) -> list[str]:
        """Build the ordered list of providers to try."""
        if preferred:
            if preferred not in self._providers:
                raise ValueError(f"Provider '{preferred}' is not registered.")
            chain = [preferred] + [p for p in self._fallback_chain if p != preferred]
        elif self._fallback_chain:
            chain = list(self._fallback_chain)
        elif self._providers:
            chain = list(self._providers.keys())
        else:
            raise RuntimeError("No providers registered.")
        return chain

    def _backoff(self, attempt: int) -> float:
        """Calculate backoff duration for the given *attempt* number."""
        cfg = self._retry_config
        wait = min(cfg.backoff_base * (2 ** (attempt - 1)), cfg.backoff_max)
        if cfg.jitter:
            wait *= random.uniform(0.5, 1.0)  # noqa: S311
        return wait
