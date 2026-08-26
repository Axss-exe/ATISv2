"""
ATISv2 — Robust Provider-Neutral LLM Client
============================================

Architecture:
    Pipelines → LLMClient → ProviderAdapter → Provider SDK → Model

Design goals:
  • Pipelines never know provider-specific parameter names.
  • Adapters translate generic requests into provider-valid payloads.
  • Capabilities prevent invalid requests before they leave the client.
  • Errors preserve full provider diagnostics.
  • Retries are automatic for transient failures only.
  • Token budgets are validated locally.

Author: Forensic rebuild — 2026-08-26
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("atis.llm")


def _redact_messages(messages: List[Dict[str, Any]], max_chars: int = 120) -> List[Dict[str, Any]]:
    """Return a truncated/redacted copy of messages for safe logging."""
    redacted = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_chars:
            content = content[:max_chars] + "... [truncated]"
        redacted.append({
            "role": msg.get("role"),
            "content": content,
        })
    return redacted


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM client errors."""
    pass


class LLMProviderError(LLMError):
    """
    Normalized provider error with full diagnostics.

    Attributes:
        provider: Provider name (e.g., "mistral")
        model: Model identifier (e.g., "labs-leanstral-1-5")
        status_code: HTTP status code (if applicable)
        error_type: Provider error type / code
        message: Human-readable error message
        provider_payload: Raw provider error response (redacted)
        retryable: Whether this error should be retried
        request_params: Parameters that were actually sent (redacted)
    """
    def __init__(
        self,
        provider: str,
        model: str,
        status_code: Optional[int],
        error_type: str,
        message: str,
        provider_payload: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        request_params: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.provider_payload = provider_payload or {}
        self.retryable = retryable
        self.request_params = request_params or {}
        super().__init__(
            f"[{provider}/{model}] {error_type} (HTTP {status_code}): {message}"
        )


class LLMConfigError(LLMError):
    """Local configuration or capability mismatch."""
    pass


class LLMTokenLimitError(LLMConfigError):
    """Request exceeds provider token limits."""
    pass


class LLMRequestError(LLMError):
    """Generic request failure (legacy compat)."""
    pass


# ---------------------------------------------------------------------------
# Capability model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCapabilities:
    """Capability declaration for a specific provider/model pair."""
    provider: str
    model: str
    supports_json: bool = True
    supports_json_schema: bool = False
    supports_tools: bool = False
    supports_temperature: bool = True
    supports_seed: bool = True
    seed_param_name: str = "seed"          # e.g., "random_seed" for Mistral
    supports_streaming: bool = True
    supports_system_messages: bool = True
    supports_max_tokens: bool = True
    supports_stop_sequences: bool = True
    supports_top_p: bool = True
    supports_presence_penalty: bool = False
    supports_frequency_penalty: bool = False
    max_context_tokens: int = 128_000
    max_output_tokens: int = 4096
    recommended_output_tokens: int = 4096


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class LLMProvider(Enum):
    MISTRAL = "mistral"
    OPENAI = "openai"
    CEREBRAS = "cerebras"


# Capability database — keyed by (provider, model)
# Models not listed here fall back to safe defaults.
_CAPABILITY_REGISTRY: Dict[Tuple[str, str], ModelCapabilities] = {
    ("mistral", "labs-leanstral-1-5"): ModelCapabilities(
        provider="mistral",
        model="labs-leanstral-1-5",
        supports_json=True,
        supports_json_schema=True,
        supports_tools=True,
        supports_temperature=True,
        supports_seed=True,
        seed_param_name="random_seed",
        supports_streaming=True,
        supports_system_messages=True,
        supports_max_tokens=True,
        supports_stop_sequences=True,
        supports_top_p=True,
        supports_presence_penalty=True,
        supports_frequency_penalty=True,
        max_context_tokens=262_144,
        max_output_tokens=131_072,
        recommended_output_tokens=4096,
    ),
    ("mistral", "mistral-small-latest"): ModelCapabilities(
        provider="mistral",
        model="mistral-small-latest",
        supports_json=True,
        supports_json_schema=True,
        supports_tools=True,
        supports_temperature=True,
        supports_seed=True,
        seed_param_name="random_seed",
        supports_streaming=True,
        supports_system_messages=True,
        supports_max_tokens=True,
        supports_stop_sequences=True,
        supports_top_p=True,
        supports_presence_penalty=True,
        supports_frequency_penalty=True,
        max_context_tokens=128_000,
        max_output_tokens=4096,
        recommended_output_tokens=4096,
    ),
    ("openai", "gpt-4o"): ModelCapabilities(
        provider="openai",
        model="gpt-4o",
        supports_json=True,
        supports_json_schema=True,
        supports_tools=True,
        supports_temperature=True,
        supports_seed=True,
        seed_param_name="seed",
        supports_streaming=True,
        supports_system_messages=True,
        supports_max_tokens=True,
        supports_stop_sequences=True,
        supports_top_p=True,
        supports_presence_penalty=True,
        supports_frequency_penalty=True,
        max_context_tokens=128_000,
        max_output_tokens=4096,
        recommended_output_tokens=4096,
    ),
    ("cerebras", "llama3.1-70b"): ModelCapabilities(
        provider="cerebras",
        model="llama3.1-70b",
        supports_json=True,
        supports_json_schema=False,
        supports_tools=False,
        supports_temperature=True,
        supports_seed=False,
        seed_param_name="seed",
        supports_streaming=True,
        supports_system_messages=True,
        supports_max_tokens=True,
        supports_stop_sequences=True,
        supports_top_p=True,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        max_context_tokens=128_000,
        max_output_tokens=4096,
        recommended_output_tokens=4096,
    ),
}


def _get_capabilities(provider: str, model: str) -> ModelCapabilities:
    """Look up capabilities; fall back to safe defaults if unknown."""
    key = (provider.lower(), model.lower())
    if key in _CAPABILITY_REGISTRY:
        return _CAPABILITY_REGISTRY[key]
    # Safe fallback — conservative assumptions
    logger.warning(
        "No capability registry entry for %s/%s; using conservative defaults.",
        provider, model,
    )
    return ModelCapabilities(
        provider=provider,
        model=model,
        supports_json=True,
        supports_json_schema=False,
        supports_tools=False,
        supports_temperature=True,
        supports_seed=False,
        seed_param_name="seed",
        supports_streaming=False,
        supports_system_messages=True,
        supports_max_tokens=True,
        supports_stop_sequences=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        max_context_tokens=128_000,
        max_output_tokens=4096,
        recommended_output_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Token estimation (very rough, but sufficient for safety checks)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total input tokens from messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += _estimate_tokens(part["text"])
    # Add overhead for message formatting
    total += len(messages) * 4
    return total


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model: str
    fallback_model: Optional[str] = None
    provider: LLMProvider = LLMProvider.MISTRAL
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    @classmethod
    def from_env(cls, _dotenv_loaded: bool = False) -> "LLMConfig":
        """Build config from environment variables, with .env fallback."""
        # Attempt to load .env file if python-dotenv is available
        # This helps in containerized or development environments
        if not _dotenv_loaded:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=False)
            except ImportError:
                pass  # python-dotenv not installed — proceed with os.environ

        provider_str = os.getenv("LLM_PROVIDER", "mistral").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise LLMConfigError(f"Unknown LLM_PROVIDER: {provider_str}")

        model = os.getenv("LLM_MODEL", "labs-leanstral-1-5")
        fallback = os.getenv("LLM_FALLBACK_MODEL") or None
        api_key = os.getenv("LLM_API_KEY") or os.getenv("MISTRAL_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None

        if not api_key:
            # Build a detailed diagnostic message
            checked = []
            if "LLM_API_KEY" in os.environ:
                checked.append("LLM_API_KEY=<<empty>>")
            else:
                checked.append("LLM_API_KEY=<not set>")
            if "MISTRAL_API_KEY" in os.environ:
                checked.append("MISTRAL_API_KEY=<<empty>>")
            else:
                checked.append("MISTRAL_API_KEY=<not set>")
            raise LLMConfigError(
                f"No API key found. Checked: {', '.join(checked)}. "
                f"Set LLM_API_KEY or MISTRAL_API_KEY environment variable."
            )

        return cls(
            model=model,
            fallback_model=fallback,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

class RetryPolicy:
    """Determines whether an error is retryable and computes backoff."""

    RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
    NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

    def __init__(self, max_retries: int, base_delay: float, max_delay: float):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def is_retryable(self, error: Exception) -> bool:
        """Determine if an exception warrants a retry."""
        if isinstance(error, LLMProviderError):
            if error.status_code in self.NON_RETRYABLE_STATUS_CODES:
                return False
            if error.status_code in self.RETRYABLE_STATUS_CODES:
                return True
            # Network-level errors without status code may be retryable
            if error.status_code is None:
                return True
            return False
        # Connection errors, timeouts, etc.
        return True

    def compute_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        import random
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 1)
        return min(delay + jitter, self.max_delay)


# ---------------------------------------------------------------------------
# Provider Adapters (abstract + concrete)
# ---------------------------------------------------------------------------

class ProviderAdapter(ABC):
    """Abstract base for provider-specific LLM adapters."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.capabilities = _get_capabilities(config.provider.value, config.model)

    @abstractmethod
    def _init_client(self) -> Any:
        """Initialize and return the provider SDK client."""
        pass

    @abstractmethod
    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        output_format: Optional[str],
        stop: Optional[List[str]],
        top_p: Optional[float],
    ) -> Dict[str, Any]:
        """Build the provider-specific request payload."""
        pass

    @abstractmethod
    def _send_request(self, payload: Dict[str, Any]) -> Any:
        """Send the request and return the raw response."""
        pass

    @abstractmethod
    def _extract_content(self, response: Any) -> str:
        """Extract the text content from the provider response."""
        pass

    @abstractmethod
    def _normalize_error(self, exception: Exception, payload: Dict[str, Any]) -> LLMProviderError:
        """Convert a provider exception into a normalized LLMProviderError."""
        pass

    def validate_request(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        seed: Optional[int],
        output_format: Optional[str],
    ) -> None:
        """Validate request against declared capabilities."""
        caps = self.capabilities

        # Token budget check
        input_tokens = _estimate_message_tokens(messages)
        total = input_tokens + max_tokens
        if total > caps.max_context_tokens:
            raise LLMTokenLimitError(
                f"Estimated total tokens ({total}) exceeds provider context limit "
                f"({caps.max_context_tokens}). Input: ~{input_tokens}, Requested output: {max_tokens}."
            )
        if max_tokens > caps.max_output_tokens:
            raise LLMTokenLimitError(
                f"Requested max_tokens ({max_tokens}) exceeds provider output limit "
                f"({caps.max_output_tokens})."
            )

        # Seed check
        if seed is not None and not caps.supports_seed:
            logger.warning(
                "Provider %s does not support seed/deterministic generation. "
                "Seed will be omitted.",
                caps.provider,
            )

        # Output format check
        if output_format == "json" and not caps.supports_json:
            raise LLMConfigError(
                f"Provider {caps.provider}/{caps.model} does not support JSON output."
            )
        if output_format == "json_schema" and not caps.supports_json_schema:
            raise LLMConfigError(
                f"Provider {caps.provider}/{caps.model} does not support JSON schema output."
            )


class MistralAdapter(ProviderAdapter):
    """Adapter for Mistral AI (labs-leanstral-1-5, mistral-small-latest, etc.)."""

    def _init_client(self) -> Any:
        try:
            from mistralai import Mistral
        except ImportError as e:
            raise LLMConfigError(
                "The 'mistralai' Python SDK is not installed. "
                "Install it with: pip install mistralai"
            ) from e
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["server_url"] = self.config.base_url
        return Mistral(**kwargs)

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        output_format: Optional[str],
        stop: Optional[List[str]],
        top_p: Optional[float],
    ) -> Dict[str, Any]:
        caps = self.capabilities
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        if caps.supports_temperature:
            payload["temperature"] = temperature
        if caps.supports_max_tokens:
            payload["max_tokens"] = max_tokens
        if caps.supports_top_p and top_p is not None:
            payload["top_p"] = top_p
        if caps.supports_stop_sequences and stop:
            payload["stop"] = stop

        # CRITICAL FIX: Map generic "seed" to Mistral-specific "random_seed"
        if seed is not None and caps.supports_seed:
            payload[caps.seed_param_name] = seed  # "random_seed"

        # JSON mode
        if output_format == "json" and caps.supports_json:
            payload["response_format"] = {"type": "json_object"}
        elif output_format == "json_schema" and caps.supports_json_schema:
            # Caller must provide schema separately; we just enable the mode here
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _send_request(self, payload: Dict[str, Any]) -> Any:
        client = self._init_client()
        return client.chat.complete(**payload)

    def _extract_content(self, response: Any) -> str:
        return response.choices[0].message.content

    def _normalize_error(self, exception: Exception, payload: Dict[str, Any]) -> LLMProviderError:
        status_code = None
        error_type = "unknown"
        message = str(exception)
        provider_payload = {}

        # Try to extract structured error info from Mistral SDK exceptions
        if hasattr(exception, "status_code"):
            status_code = exception.status_code
        if hasattr(exception, "body"):
            provider_payload = exception.body if isinstance(exception.body, dict) else {}
        if hasattr(exception, "message"):
            message = exception.message

        # Heuristic classification
        if status_code == 400:
            error_type = "bad_request"
            retryable = False
        elif status_code == 401:
            error_type = "authentication_error"
            retryable = False
        elif status_code == 403:
            error_type = "permission_error"
            retryable = False
        elif status_code == 404:
            error_type = "not_found"
            retryable = False
        elif status_code == 422:
            error_type = "validation_error"
            retryable = False
        elif status_code == 429:
            error_type = "rate_limit"
            retryable = True
        elif status_code and status_code >= 500:
            error_type = "server_error"
            retryable = True
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            error_type = "sdk_not_installed"
            retryable = False
            message = f"SDK not installed: {message}"
        else:
            error_type = "network_or_unknown"
            retryable = True

        # Redact sensitive fields from logged payload
        safe_payload = {k: v for k, v in payload.items() if k != "messages"}
        safe_payload["messages"] = _redact_messages(payload.get("messages", []))

        return LLMProviderError(
            provider="mistral",
            model=self.config.model,
            status_code=status_code,
            error_type=error_type,
            message=message,
            provider_payload=provider_payload,
            retryable=retryable,
            request_params=safe_payload,
        )


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible APIs."""

    def _init_client(self) -> Any:
        try:
            import openai
        except ImportError as e:
            raise LLMConfigError(
                "The 'openai' Python SDK is not installed. "
                "Install it with: pip install openai"
            ) from e
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return openai.OpenAI(**kwargs)

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        output_format: Optional[str],
        stop: Optional[List[str]],
        top_p: Optional[float],
    ) -> Dict[str, Any]:
        caps = self.capabilities
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if caps.supports_temperature:
            payload["temperature"] = temperature
        if caps.supports_max_tokens:
            payload["max_tokens"] = max_tokens
        if caps.supports_top_p and top_p is not None:
            payload["top_p"] = top_p
        if caps.supports_stop_sequences and stop:
            payload["stop"] = stop
        if seed is not None and caps.supports_seed:
            payload[caps.seed_param_name] = seed  # "seed" for OpenAI
        if output_format == "json" and caps.supports_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _send_request(self, payload: Dict[str, Any]) -> Any:
        client = self._init_client()
        return client.chat.completions.create(**payload)

    def _extract_content(self, response: Any) -> str:
        return response.choices[0].message.content

    def _normalize_error(self, exception: Exception, payload: Dict[str, Any]) -> LLMProviderError:
        status_code = None
        error_type = "unknown"
        message = str(exception)
        provider_payload = {}

        if hasattr(exception, "status_code"):
            status_code = exception.status_code
        if hasattr(exception, "response"):
            try:
                provider_payload = exception.response.json()
            except Exception:
                pass

        retryable = status_code in {429, 502, 503, 504} if status_code else True

        safe_payload = {k: v for k, v in payload.items() if k != "messages"}
        safe_payload["messages"] = _redact_messages(payload.get("messages", []))

        return LLMProviderError(
            provider="openai",
            model=self.config.model,
            status_code=status_code,
            error_type=error_type,
            message=message,
            provider_payload=provider_payload,
            retryable=retryable,
            request_params=safe_payload,
        )


class CerebrasAdapter(ProviderAdapter):
    """Adapter for Cerebras Cloud API."""

    def _init_client(self) -> Any:
        # Cerebras uses the OpenAI SDK with a different base URL
        try:
            import openai
        except ImportError as e:
            raise LLMConfigError(
                "The 'openai' Python SDK is required for Cerebras. "
                "Install it with: pip install openai"
            ) from e
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        else:
            kwargs["base_url"] = "https://api.cerebras.ai/v1"
        return openai.OpenAI(**kwargs)

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        output_format: Optional[str],
        stop: Optional[List[str]],
        top_p: Optional[float],
    ) -> Dict[str, Any]:
        caps = self.capabilities
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if caps.supports_temperature:
            payload["temperature"] = temperature
        if caps.supports_max_tokens:
            payload["max_tokens"] = max_tokens
        if caps.supports_top_p and top_p is not None:
            payload["top_p"] = top_p
        if caps.supports_stop_sequences and stop:
            payload["stop"] = stop
        # Cerebras does not support seed — omit silently
        if output_format == "json" and caps.supports_json:
            # Cerebras may require explicit JSON instruction in prompt
            pass
        return payload

    def _send_request(self, payload: Dict[str, Any]) -> Any:
        client = self._init_client()
        return client.chat.completions.create(**payload)

    def _extract_content(self, response: Any) -> str:
        return response.choices[0].message.content

    def _normalize_error(self, exception: Exception, payload: Dict[str, Any]) -> LLMProviderError:
        status_code = None
        if hasattr(exception, "status_code"):
            status_code = exception.status_code
        retryable = status_code in {429, 502, 503, 504} if status_code else True

        safe_payload = {k: v for k, v in payload.items() if k != "messages"}
        safe_payload["messages"] = _redact_messages(payload.get("messages", []))

        return LLMProviderError(
            provider="cerebras",
            model=self.config.model,
            status_code=status_code,
            error_type="cerebras_error",
            message=str(exception),
            provider_payload={},
            retryable=retryable,
            request_params=safe_payload,
        )


def _create_adapter(config: LLMConfig) -> ProviderAdapter:
    """Factory: create the correct adapter for the configured provider."""
    if config.provider == LLMProvider.MISTRAL:
        return MistralAdapter(config)
    elif config.provider == LLMProvider.OPENAI:
        return OpenAIAdapter(config)
    elif config.provider == LLMProvider.CEREBRAS:
        return CerebrasAdapter(config)
    else:
        raise LLMConfigError(f"No adapter for provider: {config.provider}")


# ---------------------------------------------------------------------------
# Core LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-neutral LLM client with capability-aware request building,
    retry logic, and normalized error handling.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.adapter = _create_adapter(self.config)
        self.retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay,
            max_delay=self.config.retry_max_delay,
        )
        self._request_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Public API — provider-neutral
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        output_format: Optional[str] = None,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with "role" and "content".
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens to generate.
            seed: Seed for deterministic generation. Automatically translated
                  to provider-specific parameter names (e.g., random_seed for Mistral).
            output_format: "json" or "json_schema" to request structured output.
            stop: Optional list of stop sequences.
            top_p: Optional nucleus sampling parameter.

        Returns:
            The generated text content.

        Raises:
            LLMProviderError: On provider failures (includes retryability info).
            LLMConfigError: On local validation failures.
        """
        request_id = str(uuid.uuid4())[:8]
        self._request_count += 1

        # Local validation
        self.adapter.validate_request(
            messages=messages,
            max_tokens=max_tokens,
            seed=seed,
            output_format=output_format,
        )

        # Build provider-specific payload
        payload = self.adapter._build_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            output_format=output_format,
            stop=stop,
            top_p=top_p,
        )

        # Log request (redacted)
        safe_payload = {k: v for k, v in payload.items() if k != "messages"}
        safe_payload["messages"] = _redact_messages(messages)
        logger.info(
            "[%s] LLM request → %s/%s | tokens=%s | params=%s",
            request_id,
            self.config.provider.value,
            self.config.model,
            _estimate_message_tokens(messages) + max_tokens,
            safe_payload,
        )

        # Execute with retries
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_policy.max_retries + 1):
            start_time = time.time()
            try:
                raw_response = self.adapter._send_request(payload)
                content = self.adapter._extract_content(raw_response)
                elapsed = time.time() - start_time

                logger.info(
                    "[%s] LLM success in %.2fs | content_len=%d",
                    request_id, elapsed, len(content),
                )
                return content

            except Exception as exc:
                elapsed = time.time() - start_time
                normalized = self.adapter._normalize_error(exc, payload)
                last_error = normalized
                self._error_count += 1

                logger.warning(
                    "[%s] LLM error (attempt %d/%d) in %.2fs | %s | status=%s | retryable=%s",
                    request_id,
                    attempt + 1,
                    self.retry_policy.max_retries + 1,
                    elapsed,
                    normalized.error_type,
                    normalized.status_code,
                    normalized.retryable,
                )

                if not normalized.retryable or attempt >= self.retry_policy.max_retries:
                    raise normalized

                delay = self.retry_policy.compute_delay(attempt)
                logger.info("[%s] Retrying in %.2fs...", request_id, delay)
                time.sleep(delay)

        # Should never reach here
        raise last_error if last_error else LLMRequestError("Unknown failure")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        """Return client diagnostics."""
        # Check SDK availability
        sdk_status = "unknown"
        try:
            self.adapter._init_client()
            sdk_status = "available"
        except LLMConfigError as e:
            sdk_status = f"missing: {str(e)}"
        except Exception as e:
            sdk_status = f"error: {str(e)}"

        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "fallback_model": self.config.fallback_model,
            "sdk_status": sdk_status,
            "capabilities": {
                "supports_seed": self.adapter.capabilities.supports_seed,
                "seed_param_name": self.adapter.capabilities.seed_param_name,
                "max_context_tokens": self.adapter.capabilities.max_context_tokens,
                "max_output_tokens": self.adapter.capabilities.max_output_tokens,
                "supports_json": self.adapter.capabilities.supports_json,
                "supports_json_schema": self.adapter.capabilities.supports_json_schema,
            },
            "request_count": self._request_count,
            "error_count": self._error_count,
        }

    def self_test(self, test_prompt: str = "Respond with exactly: OK") -> Dict[str, Any]:
        """
        Run a safe diagnostic sequence against the configured provider.

        Tests:
        1. SDK availability
        2. Authentication + minimal generation
        3. Requested token generation (40 tokens)
        4. JSON generation (if supported)
        5. Error normalization

        Returns a dict with test results.
        """
        results = {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "tests": {},
            "overall": "unknown",
        }

        # Test 0: SDK availability
        try:
            self.adapter._init_client()
            results["tests"]["sdk_availability"] = {"status": "PASS"}
        except LLMConfigError as e:
            results["tests"]["sdk_availability"] = {"status": "FAIL", "reason": "SDK_NOT_INSTALLED", "detail": str(e)}
            results["overall"] = "SDK_NOT_INSTALLED"
            return results
        except Exception as e:
            results["tests"]["sdk_availability"] = {"status": "FAIL", "reason": "SDK_ERROR", "detail": str(e)}
            results["overall"] = "SDK_ERROR"
            return results

        # Test 1: Minimal generation (auth + connectivity)
        try:
            response = self.chat(
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,
                max_tokens=40,
            )
            results["tests"]["minimal_generation"] = {
                "status": "PASS",
                "response_preview": response[:100],
            }
        except LLMProviderError as e:
            if e.status_code == 401:
                results["tests"]["minimal_generation"] = {"status": "FAIL", "reason": "AUTH_FAILURE", "detail": e.message}
                results["overall"] = "AUTH_FAILURE"
                return results
            elif e.status_code == 404:
                results["tests"]["minimal_generation"] = {"status": "FAIL", "reason": "MODEL_FAILURE", "detail": e.message}
                results["overall"] = "MODEL_FAILURE"
                return results
            else:
                results["tests"]["minimal_generation"] = {"status": "FAIL", "reason": "REQUEST_FAILURE", "detail": e.message}
                results["overall"] = "REQUEST_FAILURE"
                return results
        except Exception as e:
            results["tests"]["minimal_generation"] = {"status": "FAIL", "reason": "NETWORK_FAILURE", "detail": str(e)}
            results["overall"] = "NETWORK_FAILURE"
            return results

        # Test 2: Larger token generation
        try:
            response = self.chat(
                messages=[{"role": "user", "content": "Write a short paragraph about artificial intelligence."}],
                temperature=0.7,
                max_tokens=4096,
            )
            results["tests"]["large_token_generation"] = {
                "status": "PASS",
                "response_length": len(response),
            }
        except LLMProviderError as e:
            if e.status_code == 429:
                results["tests"]["large_token_generation"] = {"status": "FAIL", "reason": "RATE_LIMIT", "detail": e.message}
            elif "context" in e.message.lower() or "limit" in e.message.lower():
                results["tests"]["large_token_generation"] = {"status": "FAIL", "reason": "CONTEXT_LIMIT", "detail": e.message}
            else:
                results["tests"]["large_token_generation"] = {"status": "FAIL", "reason": "OUTPUT_LIMIT", "detail": e.message}
        except Exception as e:
            results["tests"]["large_token_generation"] = {"status": "FAIL", "reason": "UNKNOWN", "detail": str(e)}

        # Test 3: JSON generation (if supported)
        if self.adapter.capabilities.supports_json:
            try:
                response = self.chat(
                    messages=[
                        {"role": "system", "content": "Respond with valid JSON only."},
                        {"role": "user", "content": 'Return {"status": "ok"}'},
                    ],
                    temperature=0.0,
                    max_tokens=100,
                    output_format="json",
                )
                import json
                parsed = json.loads(response)
                results["tests"]["json_generation"] = {
                    "status": "PASS",
                    "parsed": parsed,
                }
            except json.JSONDecodeError as e:
                results["tests"]["json_generation"] = {"status": "FAIL", "reason": "PARSING_FAILURE", "detail": str(e)}
            except LLMProviderError as e:
                results["tests"]["json_generation"] = {"status": "FAIL", "reason": "PROVIDER_ERROR", "detail": e.message}
            except Exception as e:
                results["tests"]["json_generation"] = {"status": "FAIL", "reason": "UNKNOWN", "detail": str(e)}
        else:
            results["tests"]["json_generation"] = {"status": "SKIP", "reason": "JSON_NOT_SUPPORTED"}

        # Overall
        all_pass = all(
            t.get("status") == "PASS"
            for t in results["tests"].values()
        )
        results["overall"] = "PASS" if all_pass else "PARTIAL"
        return results


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

# Maintain the same import interface as the old client
# so pipelines do not need to change.

def get_client(api_key: Optional[str] = None) -> LLMClient:
    """
    Factory function — same signature as legacy client, with optional api_key override.

    Args:
        api_key: Optional explicit API key. If provided, bypasses environment lookup.
    """
    if api_key:
        config = LLMConfig(
            model=os.getenv("LLM_MODEL", "labs-leanstral-1-5"),
            fallback_model=os.getenv("LLM_FALLBACK_MODEL") or None,
            provider=LLMProvider(os.getenv("LLM_PROVIDER", "mistral").lower()),
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
        )
        return LLMClient(config)
    return LLMClient()


# Allow direct instantiation for tests
__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "LLMError",
    "LLMProviderError",
    "LLMConfigError",
    "LLMTokenLimitError",
    "LLMRequestError",
    "get_client",
    "ModelCapabilities",
    "RetryPolicy",
    "MistralAdapter",
    "OpenAIAdapter",
    "CerebrasAdapter",
]
