"""Configuration models for LLMProxy."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RetryConfig(BaseModel):
    """Retry behaviour configuration."""

    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_base: float = Field(default=1.0, gt=0)
    backoff_max: float = Field(default=30.0, gt=0)
    jitter: bool = True


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    timeout: float = Field(default=60.0, gt=0)
    extra: dict[str, object] = Field(default_factory=dict)


class LLMProxySettings(BaseSettings):
    """Top-level settings loaded from environment variables."""

    model_config = {"env_prefix": "LLMPROXY_", "env_file": ".env", "extra": "ignore"}

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    default_provider: str = "openai"
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    log_level: str = "INFO"
