"""Centralized, provider-agnostic chat-completions client for ATIS."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List
from urllib import error, request

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

logger = logging.getLogger("ATIS_LLM")


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM configuration is missing or invalid."""


class LLMRequestError(RuntimeError):
    """Raised when the provider returns a non-retryable or exhausted error."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    fallback_model: str
    temperature: float
    max_tokens: int
    timeout: float
    max_retries: int


def _load_environment() -> None:
    if load_dotenv is None:
        _load_simple_dotenv(".env")
        _load_simple_dotenv(".env.txt")
        return
    load_dotenv()
    load_dotenv(".env.txt")


def _load_simple_dotenv(path: str) -> None:
    """Load simple KEY=VALUE entries when python-dotenv is unavailable."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LLMConfigurationError(f"Missing required LLM configuration: {name}")
    return value


def load_config() -> LLMConfig:
    _load_environment()
    config = LLMConfig(
        provider=_required("LLM_PROVIDER"),
        api_key=_required("LLM_API_KEY"),
        base_url=_required("LLM_BASE_URL").rstrip("/"),
        model=_required("LLM_MODEL"),
        fallback_model=os.getenv("LLM_FALLBACK_MODEL", "").strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )
    if config.temperature < 0 or config.max_tokens <= 0 or config.timeout <= 0 or config.max_retries <= 0:
        raise LLMConfigurationError("LLM numeric configuration values are invalid")
    logger.info("LLM provider=%s model=%s fallback=%s", config.provider, config.model, config.fallback_model or "none")
    return config


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_config()

    def chat(self, messages: List[Dict[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> str:
        models = [self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            models.append(self.config.fallback_model)
        last_error: Exception | None = None
        for model in models:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    return self._request(model, messages, temperature, max_tokens)
                except LLMConfigurationError:
                    raise
                except LLMRequestError as exc:
                    last_error = exc
                    if not str(exc).startswith("retryable:"):
                        raise
                    if attempt < self.config.max_retries:
                        delay = 2 ** (attempt - 1)
                        logger.warning("Retryable LLM error for %s; retrying in %ss", model, delay)
                        time.sleep(delay)
                    elif model != models[-1]:
                        logger.warning("Primary LLM model exhausted; trying fallback model")
        raise LLMRequestError(f"LLM request failed after retries: {last_error}")

    def _request(self, model: str, messages: List[Dict[str, str]], temperature: float | None, max_tokens: int | None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 422):
                raise LLMRequestError(f"non-retryable provider error ({exc.code})") from exc
            raise LLMRequestError(f"retryable: provider error ({exc.code})") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LLMRequestError(f"retryable: connection failure ({exc})") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMRequestError(f"non-retryable invalid provider response ({exc})") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("non-retryable invalid chat-completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("non-retryable empty chat-completion response")
        return content


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def chat(messages: List[Dict[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> str:
    return get_client().chat(messages, temperature=temperature, max_tokens=max_tokens)