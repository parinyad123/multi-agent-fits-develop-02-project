"""
app/core/groq_client.py

Groq API client — drop-in replacement for AstroSage/OpenAI
Uses the official `groq` Python SDK (AsyncGroq).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from groq import AsyncGroq, APIConnectionError, APITimeoutError, RateLimitError

from app.core.config import settings

logger = logging.getLogger(__name__)


# ==================================
# Custom Exceptions
# ==================================

class GroqConnectionError(Exception):
    """Cannot reach Groq API"""

class GroqTimeoutError(Exception):
    """Groq API request timed out"""

class GroqRateLimitError(Exception):
    """Groq API rate limit exceeded"""

class GroqInvalidResponseError(Exception):
    """Groq API returned unexpected response structure"""


# ==================================
# Response dataclass
# ==================================

class GroqResponse:
    """Normalised response from Groq API"""

    def __init__(
        self,
        content: str,
        model_used: str,
        tokens_used: int = 0,
        response_time: float = 0.0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.content = content
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.response_time = response_time
        self.success = success
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GroqResponse(model={self.model_used!r}, "
            f"tokens={self.tokens_used}, success={self.success})"
        )


# ==================================
# Main Client
# ==================================

class GroqClient:
    """
    Async client for Groq chat-completion API.

    Usage::

        client = GroqClient()
        response: GroqResponse = await client.chat(
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.content)
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        self.temperature = settings.groq_default_temperature
        self.max_tokens = settings.groq_default_max_tokens
        self.timeout = settings.groq_timeout
        self.max_retries = settings.groq_max_retries
        self.retry_delay = settings.groq_retry_delay

        logger.info(
            "GroqClient initialised: model=%s, timeout=%ss, max_retries=%s",
            self.model,
            self.timeout,
            self.max_retries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GroqResponse:
        """
        Send a chat-completion request to Groq.

        Args:
            messages:    OpenAI-style message list
                         [{"role": "system"|"user"|"assistant", "content": "..."}]
            model:       Override the default model for this call.
            temperature: Override default temperature.
            max_tokens:  Override default max_tokens.

        Returns:
            GroqResponse

        Raises:
            GroqConnectionError, GroqTimeoutError, GroqRateLimitError,
            GroqInvalidResponseError
        """
        start_time = time.time()

        effective_model = model or self.model
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            raw = await self._call_with_retry(
                messages=messages,
                model=effective_model,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
        except (GroqConnectionError, GroqTimeoutError, GroqRateLimitError, GroqInvalidResponseError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in GroqClient.chat: %s", exc, exc_info=True)
            raise

        content = self._extract_content(raw)
        tokens = self._extract_tokens(raw)
        response_time = time.time() - start_time

        logger.info(
            "Groq response: model=%s, tokens=%s, time=%.2fs",
            effective_model,
            tokens,
            response_time,
        )

        return GroqResponse(
            content=content,
            model_used=effective_model,
            tokens_used=tokens,
            response_time=response_time,
            success=True,
        )

    # ------------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "Groq API attempt %d/%d", attempt + 1, self.max_retries + 1
                )
                return await self._call_api(messages, model, temperature, max_tokens)

            except APITimeoutError as exc:
                last_error = exc
                logger.warning("Groq timeout on attempt %d: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise GroqTimeoutError(
                        f"Groq timed out after {self.max_retries + 1} attempts"
                    ) from exc

            except APIConnectionError as exc:
                last_error = exc
                logger.error("Groq connection error on attempt %d: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise GroqConnectionError(
                        "Cannot connect to Groq API"
                    ) from exc

            except RateLimitError as exc:
                last_error = exc
                logger.warning("Groq rate limit on attempt %d: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    # Longer wait for rate-limit
                    await asyncio.sleep(self.retry_delay * (2 ** attempt) * 2)
                else:
                    raise GroqRateLimitError(
                        "Groq rate limit exceeded"
                    ) from exc

            except GroqInvalidResponseError:
                raise

            except Exception as exc:
                last_error = exc
                logger.error("Unexpected Groq error on attempt %d: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

        raise Exception(
            f"Groq failed after {self.max_retries + 1} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Raw API call
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

        if not self._validate(response):
            raise GroqInvalidResponseError("Groq response failed validation")

        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(response: Any) -> bool:
        try:
            return bool(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError):
            return False

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    @staticmethod
    def _extract_tokens(response: Any) -> int:
        try:
            return response.usage.total_tokens or 0
        except AttributeError:
            return 0


# ==================================
# Module-level singleton (optional)
# ==================================

_groq_client: GroqClient | None = None


def get_groq_client() -> GroqClient:
    """Return the module-level GroqClient singleton."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
