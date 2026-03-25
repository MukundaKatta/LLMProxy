"""Tests for LLMProxy core functionality using the MockProvider."""

from __future__ import annotations

import pytest

from llmproxy import CompletionResponse, LLMProxy, MockProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def proxy() -> LLMProxy:
    """Return an LLMProxy instance with a mock provider pre-registered."""
    p = LLMProxy()
    p.add_provider_instance("mock", MockProvider())
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for the complete() method."""

    async def test_basic_completion(self, proxy: LLMProxy) -> None:
        """A simple prompt should return a CompletionResponse with content."""
        resp = await proxy.complete(prompt="Hello, world!", model="mock-v1", provider="mock")
        assert isinstance(resp, CompletionResponse)
        assert "Hello, world!" in resp.content
        assert resp.provider == "mock"
        assert resp.model == "mock-v1"
        assert resp.total_tokens > 0

    async def test_fallback_chain(self) -> None:
        """When the first provider fails, the proxy should fall through."""

        class FailingProvider(MockProvider):
            name = "failing"

            async def complete(self, prompt: str, model: str, **kwargs: object) -> CompletionResponse:
                raise RuntimeError("Simulated failure")

        proxy = LLMProxy()
        proxy.add_provider_instance("failing", FailingProvider())
        proxy.add_provider_instance("mock", MockProvider())
        proxy.set_fallback_chain(["failing", "mock"])
        proxy.configure_retry(max_retries=1, backoff_base=0.01)

        resp = await proxy.complete(prompt="test fallback")
        assert resp.provider == "mock"

    async def test_all_providers_fail_raises(self) -> None:
        """RuntimeError should be raised when every provider exhausts retries."""

        class AlwaysFail(MockProvider):
            name = "fail"

            async def complete(self, prompt: str, model: str, **kwargs: object) -> CompletionResponse:
                raise RuntimeError("boom")

        proxy = LLMProxy()
        proxy.add_provider_instance("fail", AlwaysFail())
        proxy.configure_retry(max_retries=1, backoff_base=0.01)

        with pytest.raises(RuntimeError, match="All providers"):
            await proxy.complete(prompt="should fail")


class TestProviderManagement:
    """Tests for provider registration and model listing."""

    def test_add_provider_unknown_raises(self) -> None:
        proxy = LLMProxy()
        with pytest.raises(ValueError, match="Unknown provider"):
            proxy.add_provider("not_real", {"api_key": "x"})

    def test_get_available_models(self, proxy: LLMProxy) -> None:
        models = proxy.get_available_models()
        assert "mock" in models
        assert "mock-v1" in models["mock"]


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_returns_dict(self, proxy: LLMProxy) -> None:
        result = proxy.estimate_cost("Hello!", model="gpt-4o")
        assert "estimated_cost" in result
        assert result["estimated_input_tokens"] > 0
        assert result["estimated_cost"] > 0


class TestUsageStats:
    """Tests for usage tracking."""

    async def test_usage_increments(self, proxy: LLMProxy) -> None:
        await proxy.complete(prompt="one", provider="mock")
        await proxy.complete(prompt="two", provider="mock")
        stats = proxy.get_usage_stats()
        assert stats["mock"]["request_count"] == 2


class TestHealthCheck:
    """Tests for health_check."""

    async def test_mock_health_check(self, proxy: LLMProxy) -> None:
        result = await proxy.health_check("mock")
        assert result["healthy"] is True

    async def test_health_check_unknown_provider(self, proxy: LLMProxy) -> None:
        with pytest.raises(ValueError, match="not registered"):
            await proxy.health_check("unknown")


class TestRetryConfig:
    """Tests for configure_retry."""

    def test_configure_retry(self, proxy: LLMProxy) -> None:
        proxy.configure_retry(max_retries=5, backoff_base=2.0, backoff_max=60.0, jitter=False)
        assert proxy._retry_config.max_retries == 5
        assert proxy._retry_config.backoff_base == 2.0
        assert proxy._retry_config.jitter is False
