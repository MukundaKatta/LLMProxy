# Architecture

This document describes the internal architecture of LLMProxy.

## Overview

LLMProxy is structured as a thin orchestration layer that sits between your application and one or more LLM provider APIs. It normalises requests and responses into a common schema, adds retry/fallback logic, and tracks usage.

## Module Layout

```
src/llmproxy/
  __init__.py      Public API surface
  core.py          LLMProxy class, BaseProvider, provider implementations
  config.py        Pydantic settings and configuration models
  utils.py         Request builders, response normalisers, cost estimation
```

## Key Components

### LLMProxy (core.py)

The central class. Responsibilities:

- **Provider registry** — stores instantiated `BaseProvider` objects keyed by name.
- **Fallback chain** — an ordered list of provider names. When one fails after retries, the next is tried.
- **Retry engine** — exponential backoff with optional jitter, configurable via `configure_retry()`.
- **Usage tracking** — per-provider counters for requests, tokens, cost, and latency.

### BaseProvider (core.py)

Abstract base class that all providers implement:

| Method          | Purpose                                |
|-----------------|----------------------------------------|
| `complete()`    | Send prompt, return `CompletionResponse` |
| `health_check()`| Verify the provider is reachable       |
| `list_models()` | Return supported model identifiers     |

Concrete implementations: `OpenAIProvider`, `AnthropicProvider`, `MockProvider`.

### Configuration (config.py)

- `ProviderConfig` — per-provider settings (API key, base URL, timeout).
- `RetryConfig` — retry count, backoff base/max, jitter toggle.
- `LLMProxySettings` — environment-variable-driven settings via `pydantic-settings`.

### Utilities (utils.py)

- **Request builders** — `build_openai_payload()`, `build_anthropic_payload()`.
- **Response normalisers** — convert provider-specific JSON into `CompletionResponse`.
- **Cost estimation** — character-based token estimation and per-model pricing lookup.
- **Timer** — context manager for measuring request latency.

## Request Flow

```
User calls proxy.complete(prompt, model, provider)
  |
  v
Resolve fallback chain (preferred provider first)
  |
  v
For each provider in chain:
  |
  +---> For each retry attempt:
  |       |
  |       +---> provider.complete(prompt, model)
  |       |       |
  |       |       +---> Build request payload
  |       |       +---> HTTP POST via httpx
  |       |       +---> Normalise response -> CompletionResponse
  |       |       +---> Record usage stats
  |       |       +---> Return
  |       |
  |       +---> On failure: backoff, retry
  |
  +---> On exhaustion: try next provider
  |
  v
All providers failed -> raise RuntimeError
```

## Extending

To add a new provider:

1. Subclass `BaseProvider`.
2. Implement `complete()`, `health_check()`, and `list_models()`.
3. Add the class to `PROVIDER_REGISTRY` in `core.py`.

The system is designed so that adding a provider requires zero changes to `LLMProxy` itself.
