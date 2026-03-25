"""Utility helpers: request building, response normalisation, cost estimation."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CompletionResponse(BaseModel):
    """Normalised response returned by every provider."""

    content: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Rough average characters-per-token for common models.
_CHARS_PER_TOKEN: dict[str, float] = {
    "gpt-4o": 3.8,
    "gpt-4o-mini": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    "gpt-3.5-turbo": 4.0,
    "claude-sonnet-4-20250514": 3.5,
    "claude-3-5-sonnet-20241022": 3.5,
    "claude-3-opus-20240229": 3.5,
    "claude-3-haiku-20240307": 3.5,
}

# Cost per 1 000 tokens (input / output) in USD.
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


def estimate_token_count(text: str, model: str = "gpt-4o") -> int:
    """Return a rough token estimate for *text* given a *model*."""
    cpt = _CHARS_PER_TOKEN.get(model, 4.0)
    return max(1, int(len(text) / cpt))


def estimate_cost(
    prompt: str,
    model: str = "gpt-4o",
    expected_completion_tokens: int = 256,
) -> dict[str, Any]:
    """Estimate the cost of a completion request.

    Returns a dict with token counts and estimated USD cost.
    """
    input_tokens = estimate_token_count(prompt, model)
    costs = _COST_PER_1K.get(model, (0.01, 0.03))
    input_cost = (input_tokens / 1000) * costs[0]
    output_cost = (expected_completion_tokens / 1000) * costs[1]
    return {
        "model": model,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": expected_completion_tokens,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_cost": round(input_cost + output_cost, 8),
    }


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def build_openai_payload(
    prompt: str,
    model: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Build a JSON payload for the OpenAI chat completions endpoint."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def build_anthropic_payload(
    prompt: str,
    model: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Build a JSON payload for the Anthropic messages endpoint."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# Response normalisers
# ---------------------------------------------------------------------------


def normalise_openai_response(
    data: dict[str, Any],
    provider: str,
    latency_ms: float,
) -> CompletionResponse:
    """Normalise an OpenAI-style JSON response into a CompletionResponse."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    return CompletionResponse(
        content=message.get("content", ""),
        model=data.get("model", ""),
        provider=provider,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        latency_ms=latency_ms,
        raw=data,
    )


def normalise_anthropic_response(
    data: dict[str, Any],
    provider: str,
    latency_ms: float,
) -> CompletionResponse:
    """Normalise an Anthropic-style JSON response into a CompletionResponse."""
    content_blocks = data.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    return CompletionResponse(
        content=text,
        model=data.get("model", ""),
        provider=provider,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        latency_ms=latency_ms,
        raw=data,
    )


class Timer:
    """Simple context-manager stopwatch returning elapsed milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
