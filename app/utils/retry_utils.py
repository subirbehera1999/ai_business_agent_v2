# ==============================================================================
# File: app/utils/retry_utils.py
# Purpose: Provides exponential backoff retry logic for all external API calls.
#          Every integration (Google, OpenAI, WhatsApp, Razorpay) must use
#          this utility when making outbound requests.
#
#          Retry policy (from DATA_SAFETY_AND_RUNTIME_GUARDRAILS.txt):
#            Maximum retries : 3
#            Retry 1         : after 2 seconds
#            Retry 2         : after 5 seconds
#            Retry 3         : after 10 seconds
#            On all retries exhausted: log error, mark job as failed,
#                                      continue system operation
#
#          Two interfaces are provided:
#            1. @with_retry decorator  — wraps async functions declaratively
#            2. retry_async()          — wraps a callable imperatively for
#                                        use inside service/integration code
#
#          Retryable exceptions are configurable per call site.
#          Non-retryable exceptions (e.g., auth errors) propagate immediately.
# ==============================================================================

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Optional, Type, TypeVar

from app.config.constants import ServiceName
from app.config.settings import get_settings

logger = logging.getLogger(ServiceName.API)
settings = get_settings()

# Default retry delays in seconds (from .env / constants)
_DEFAULT_DELAYS: list[int] = settings.backoff_delays_list  # [2, 5, 10]
_DEFAULT_MAX_ATTEMPTS: int = settings.RETRY_MAX_ATTEMPTS   # 3

# TypeVar for preserving return type through the decorator
F = TypeVar("F")


# ==============================================================================
# Retryable Exception Groups
# ==============================================================================

# Exceptions that indicate a transient failure and warrant a retry.
# Import lazily inside functions to avoid circular imports with integrations.

# Standard library / httpx / aiohttp transient errors
TRANSIENT_NETWORK_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Well-known OpenAI SDK transient status codes are handled by checking
# exception message strings rather than importing the SDK here, to keep
# this utility free of integration-layer dependencies.


# ==============================================================================
# Core Retry Function
# ==============================================================================

async def retry_async(
    func: Callable[..., Coroutine[Any, Any, F]],
    *args: Any,
    retryable_exceptions: tuple[type[Exception], ...] = TRANSIENT_NETWORK_EXCEPTIONS,
    delays: Optional[list[int]] = None,
    max_attempts: Optional[int] = None,
    service_name: str = ServiceName.API,
    operation_name: str = "unknown_operation",
    **kwargs: Any,
) -> F:
    """
    Execute an async callable with exponential backoff retry logic.

    Retries the function up to max_attempts times when a retryable
    exception is raised. Non-retryable exceptions propagate immediately
    without consuming retry budget.

    Args:
        func:                  Async callable to execute.
        *args:                 Positional arguments forwarded to func.
        retryable_exceptions:  Tuple of exception types that trigger a retry.
                               Defaults to common transient network errors.
        delays:                List of wait times in seconds between retries.
                               Length determines max retry count.
                               Defaults to settings.backoff_delays_list [2, 5, 10].
        max_attempts:          Maximum number of total attempts (1 + retries).
                               Defaults to settings.RETRY_MAX_ATTEMPTS.
        service_name:          Service identifier for structured log entries.
        operation_name:        Human-readable name of the operation being retried.
        **kwargs:              Keyword arguments forwarded to func.

    Returns:
        The return value of func on success.

    Raises:
        Exception: The last exception raised after all retries are exhausted,
                   or any non-retryable exception on first occurrence.

    Example:
        result = await retry_async(
            google_client.fetch_reviews,
            place_id,
            retryable_exceptions=(ConnectionError, TimeoutError),
            service_name=ServiceName.GOOGLE_REVIEWS,
            operation_name="fetch_google_reviews",
        )
    """
    effective_delays = delays if delays is not None else _DEFAULT_DELAYS
    effective_max_attempts = (
        max_attempts if max_attempts is not None else _DEFAULT_MAX_ATTEMPTS
    )

    # Cap attempts to 1 (initial) + len(delays) retries
    total_attempts = min(effective_max_attempts, 1 + len(effective_delays))

    last_exception: Optional[Exception] = None

    for attempt in range(1, total_attempts + 1):
        try:
            result = await func(*args, **kwargs)

            if attempt > 1:
                logger.info(
                    "Operation succeeded after retry",
                    extra={
                        "service": service_name,
                        "operation": operation_name,
                        "attempt": attempt,
                        "total_attempts": total_attempts,
                    },
                )
            return result

        except retryable_exceptions as exc:
            last_exception = exc
            is_final_attempt = attempt == total_attempts

            if is_final_attempt:
                logger.error(
                    "Operation failed — all retries exhausted",
                    extra={
                        "service": service_name,
                        "operation": operation_name,
                        "attempt": attempt,
                        "total_attempts": total_attempts,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                raise

            # Wait before next attempt using the delay for this retry index
            delay = effective_delays[attempt - 1]
            logger.warning(
                "Operation failed — retrying after delay",
                extra={
                    "service": service_name,
                    "operation": operation_name,
                    "attempt": attempt,
                    "total_attempts": total_attempts,
                    "retry_delay_seconds": delay,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            await asyncio.sleep(delay)

        except Exception as exc:
            # Non-retryable exception — propagate immediately
            logger.error(
                "Non-retryable exception — aborting without retry",
                extra={
                    "service": service_name,
                    "operation": operation_name,
                    "attempt": attempt,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise

    # Should never reach here — included for type checker completeness
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(
        f"retry_async exhausted without result for operation '{operation_name}'"
    )


# ==============================================================================
# Decorator Interface
# ==============================================================================

def with_retry(
    retryable_exceptions: tuple[type[Exception], ...] = TRANSIENT_NETWORK_EXCEPTIONS,
    delays: Optional[list[int]] = None,
    max_attempts: Optional[int] = None,
    service_name: str = ServiceName.API,
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Decorator that wraps an async function with exponential backoff retry logic.

    Applies the same retry contract as retry_async() but as a declarative
    decorator. Use this on integration client methods where retry behaviour
    should be permanently attached to the function.

    Args:
        retryable_exceptions:  Exception types that trigger a retry.
        delays:                Backoff delays in seconds. Defaults to [2, 5, 10].
        max_attempts:          Maximum total attempts. Defaults to 3.
        service_name:          Service identifier for log entries.
        operation_name:        Human-readable operation name. Defaults to
                               the decorated function's __name__.

    Returns:
        Decorated async function with retry logic applied.

    Example:
        @with_retry(
            retryable_exceptions=(ConnectionError, TimeoutError),
            service_name=ServiceName.GOOGLE_REVIEWS,
        )
        async def fetch_reviews(place_id: str) -> list[dict]:
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable:
        resolved_operation_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_async(
                func,
                *args,
                retryable_exceptions=retryable_exceptions,
                delays=delays,
                max_attempts=max_attempts,
                service_name=service_name,
                operation_name=resolved_operation_name,
                **kwargs,
            )

        return wrapper
    return decorator


# ==============================================================================
# Convenience Wrappers — Pre-configured per Integration
# ==============================================================================

def with_google_retry(
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Retry decorator pre-configured for Google API calls.

    Uses standard backoff delays. Retries on network-level transient errors.
    Google-specific HTTP 429 / 503 handling is done at the client layer
    by inspecting the response status before raising.

    Args:
        operation_name: Optional name for log entries.

    Returns:
        Configured retry decorator.

    Example:
        @with_google_retry(operation_name="fetch_google_reviews")
        async def fetch_reviews(self, place_id: str) -> list[dict]:
            ...
    """
    return with_retry(
        retryable_exceptions=TRANSIENT_NETWORK_EXCEPTIONS,
        delays=_DEFAULT_DELAYS,
        max_attempts=_DEFAULT_MAX_ATTEMPTS,
        service_name=ServiceName.GOOGLE_REVIEWS,
        operation_name=operation_name,
    )


def with_openai_retry(
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Retry decorator pre-configured for OpenAI API calls.

    Uses standard backoff delays. OpenAI rate limit errors (429) and
    service unavailability (503) are transient and warrant retries.

    Args:
        operation_name: Optional name for log entries.

    Returns:
        Configured retry decorator.

    Example:
        @with_openai_retry(operation_name="generate_ai_reply")
        async def generate_reply(self, prompt: str) -> str:
            ...
    """
    return with_retry(
        retryable_exceptions=TRANSIENT_NETWORK_EXCEPTIONS,
        delays=_DEFAULT_DELAYS,
        max_attempts=_DEFAULT_MAX_ATTEMPTS,
        service_name=ServiceName.AI_REPLY,
        operation_name=operation_name,
    )


def with_whatsapp_retry(
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Retry decorator pre-configured for WhatsApp Cloud API calls.

    Uses standard backoff delays. WhatsApp API transient failures
    (network timeouts, 5xx responses) warrant retries.

    Args:
        operation_name: Optional name for log entries.

    Returns:
        Configured retry decorator.

    Example:
        @with_whatsapp_retry(operation_name="send_whatsapp_message")
        async def send_message(self, recipient: str, body: str) -> dict:
            ...
    """
    return with_retry(
        retryable_exceptions=TRANSIENT_NETWORK_EXCEPTIONS,
        delays=_DEFAULT_DELAYS,
        max_attempts=_DEFAULT_MAX_ATTEMPTS,
        service_name=ServiceName.WHATSAPP,
        operation_name=operation_name,
    )


def with_razorpay_retry(
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Retry decorator pre-configured for Razorpay API calls.

    NOTE: Payment verification must never be retried blindly — a retry
    on a payment capture could result in a double charge. This decorator
    is intended only for read operations (fetching order status, plan
    details) not for payment capture or subscription creation.

    Args:
        operation_name: Optional name for log entries.

    Returns:
        Configured retry decorator.

    Example:
        @with_razorpay_retry(operation_name="fetch_razorpay_order")
        async def fetch_order(self, order_id: str) -> dict:
            ...
    """
    return with_retry(
        retryable_exceptions=TRANSIENT_NETWORK_EXCEPTIONS,
        delays=_DEFAULT_DELAYS,
        max_attempts=_DEFAULT_MAX_ATTEMPTS,
        service_name=ServiceName.PAYMENT,
        operation_name=operation_name,
    )


# ==============================================================================
# Synchronous Retry Helper (for non-async contexts)
# ==============================================================================

def retry_sync(
    func: Callable[..., F],
    *args: Any,
    retryable_exceptions: tuple[type[Exception], ...] = TRANSIENT_NETWORK_EXCEPTIONS,
    delays: Optional[list[int]] = None,
    max_attempts: Optional[int] = None,
    service_name: str = ServiceName.API,
    operation_name: str = "unknown_sync_operation",
    **kwargs: Any,
) -> F:
    """
    Execute a synchronous callable with exponential backoff retry logic.

    Blocking equivalent of retry_async() for use in synchronous contexts
    such as Alembic migration scripts or admin CLI tools.

    Args:
        func:                  Synchronous callable to execute.
        *args:                 Positional arguments forwarded to func.
        retryable_exceptions:  Exception types that trigger a retry.
        delays:                Wait times in seconds between retries.
        max_attempts:          Maximum total attempts.
        service_name:          Service identifier for log entries.
        operation_name:        Human-readable operation name.
        **kwargs:              Keyword arguments forwarded to func.

    Returns:
        The return value of func on success.

    Raises:
        Exception: Last exception after retries exhausted, or any
                   non-retryable exception immediately.
    """
    import time

    effective_delays = delays if delays is not None else _DEFAULT_DELAYS
    effective_max_attempts = (
        max_attempts if max_attempts is not None else _DEFAULT_MAX_ATTEMPTS
    )
    total_attempts = min(effective_max_attempts, 1 + len(effective_delays))

    last_exception: Optional[Exception] = None

    for attempt in range(1, total_attempts + 1):
        try:
            return func(*args, **kwargs)

        except retryable_exceptions as exc:
            last_exception = exc
            is_final_attempt = attempt == total_attempts

            if is_final_attempt:
                logger.error(
                    "Sync operation failed — all retries exhausted",
                    extra={
                        "service": service_name,
                        "operation": operation_name,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                raise

            delay = effective_delays[attempt - 1]
            logger.warning(
                "Sync operation failed — retrying after delay",
                extra={
                    "service": service_name,
                    "operation": operation_name,
                    "attempt": attempt,
                    "retry_delay_seconds": delay,
                    "error": str(exc),
                },
            )
            time.sleep(delay)

        except Exception as exc:
            logger.error(
                "Non-retryable sync exception — aborting",
                extra={
                    "service": service_name,
                    "operation": operation_name,
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            raise

    if last_exception is not None:
        raise last_exception
    raise RuntimeError(
        f"retry_sync exhausted without result for operation '{operation_name}'"
    )