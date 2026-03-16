# ==============================================================================
# File: app/logging/logger.py
# Purpose: Central logging system for the AI Business Agent platform.
#
#          Every module in the system calls:
#            logger = logging.getLogger(ServiceName.X)
#
#          This file configures what happens when those loggers fire.
#          It must be imported and initialised once at application
#          startup (in main.py) before any other module loads.
#
#          Design decisions:
#
#          1. STRUCTURED JSON LOGGING IN PRODUCTION
#             In production (LOG_FORMAT=json), every log line is a
#             single JSON object containing:
#               timestamp, level, service, message, business_id (if present),
#               request_id (if present), error (if present), environment
#
#             Structured logs are machine-readable — they can be ingested
#             by log aggregators (Datadog, CloudWatch, Loki) for alerting
#             and querying without regex parsing.
#
#          2. HUMAN-READABLE LOGS IN DEVELOPMENT
#             In development (LOG_FORMAT=text), logs are printed as
#             readable lines:
#               2025-03-10 09:00:01 | INFO | review_monitor | business=abc123 | msg
#
#          3. SENSITIVE DATA NEVER LOGGED
#             The SensitiveDataFilter strips known sensitive field names
#             from log records before they are emitted. This prevents
#             accidental leakage of API keys, phone numbers, or tokens
#             into log files.
#
#          4. REQUEST ID PROPAGATION
#             A per-request UUID can be bound to the logging context
#             using the context manager bind_request_id(). This traces
#             all log lines for a single API request across services.
#
#          5. LOG LEVELS BY ENVIRONMENT
#             PRODUCTION:  WARNING and above (reduces log volume)
#             DEVELOPMENT: DEBUG and above (full visibility)
#             Overridable via LOG_LEVEL environment variable.
#
#          6. THIRD-PARTY LIBRARY NOISE SUPPRESSION
#             httpx, sqlalchemy.engine, apscheduler, and other libraries
#             are set to WARNING to prevent them from flooding the log
#             stream with routine connection and query events.
#
#          Usage:
#            # In main.py at startup:
#            from app.logging.logger import configure_logging
#            configure_logging()
#
#            # In any module:
#            import logging
#            from app.config.constants import ServiceName
#            logger = logging.getLogger(ServiceName.REVIEW_MONITOR)
#            logger.info("Processing review", extra={"business_id": "abc"})
# ==============================================================================

import json
import logging
import logging.config
import os
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Context variable — stores request ID for the current async context
# Each FastAPI request sets this; log records automatically include it.
# ---------------------------------------------------------------------------
_request_id_var: ContextVar[Optional[str]] = ContextVar(
    "_request_id_var", default=None
)

# ---------------------------------------------------------------------------
# Sensitive field names — values for these keys are masked in log output
# ---------------------------------------------------------------------------
_SENSITIVE_FIELDS = frozenset({
    "password",
    "api_key",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "razorpay_key_secret",
    "whatsapp_api_token",
    "openai_api_key",
    "phone",
    "phone_number",
    "whatsapp_number",
    "email",
    "card_number",
    "cvv",
    "authorization",
})

_MASKED_VALUE = "***REDACTED***"

# ---------------------------------------------------------------------------
# Third-party loggers to suppress (set to WARNING)
# ---------------------------------------------------------------------------
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "apscheduler",
    "apscheduler.scheduler",
    "apscheduler.executors",
    "uvicorn.access",
    "urllib3",
    "google.auth",
    "google.api_core",
)


# ==============================================================================
# Sensitive Data Filter
# ==============================================================================

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that masks sensitive field values in log records.

    Scans the `extra` dict attached to each log record and replaces
    values for known sensitive keys with _MASKED_VALUE.

    This filter is attached to the root logger so it runs on every
    log record regardless of which handler emits it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Mask sensitive fields. Always returns True (never suppresses records).
        """
        for field in _SENSITIVE_FIELDS:
            if hasattr(record, field):
                setattr(record, field, _MASKED_VALUE)

        # Also scan the message string for obvious leakage patterns
        # (e.g. if someone accidentally logs a dict containing secrets)
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for field in _SENSITIVE_FIELDS:
                # Pattern: "field_name": "value" or field_name=value
                if field in record.msg.lower():
                    # Don't try to parse — just mark the record
                    record.msg = (
                        f"[SENSITIVE_DATA_WARNING] {record.msg[:80]}... "
                        f"(message may contain sensitive field: {field})"
                    )
                    break

        return True


# ==============================================================================
# JSON Formatter
# ==============================================================================

class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Each line contains:
      - timestamp:   ISO 8601 UTC
      - level:       DEBUG / INFO / WARNING / ERROR / CRITICAL
      - service:     Logger name (ServiceName constant)
      - message:     Log message string
      - environment: From APP_ENV env var
      - request_id:  Current request UUID (if set)
      - business_id: From extra dict (if present)
      - error:       Exception info (if present)
      - ...plus any other fields in the extra dict

    All fields are strings or primitives — no nested objects —
    so log aggregators can index every field without schema changes.
    """

    def __init__(self, environment: str = "production") -> None:
        super().__init__()
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
            "environment": self._environment,
        }

        # Attach request ID from context var
        request_id = _request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Attach any extra fields from the log call
        # e.g. logger.info("msg", extra={"business_id": "abc"})
        _STANDARD_FIELDS = frozenset(logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()) | {"message", "asctime"}

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                if key in _SENSITIVE_FIELDS:
                    log_data[key] = _MASKED_VALUE
                else:
                    log_data[key] = value

        # Attach exception info if present
        if record.exc_info:
            log_data["error"] = self.formatException(record.exc_info)
            log_data["error_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else "UnknownError"
            )

        # Attach stack info if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info

        try:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except Exception:
            # Never let the logger itself crash — fall back to plain text
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "service": "logger",
                "message": f"Log serialisation failed for record: {record.name}",
            })


# ==============================================================================
# Text Formatter (development)
# ==============================================================================

class TextFormatter(logging.Formatter):
    """
    Human-readable formatter for development environments.

    Format:
      2025-03-10 09:00:01.234 | INFO  | review_monitor | [req=abc123] msg
                                                          [biz=xyz456]
    """

    _LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname.ljust(8)
        service = record.name[:25].ljust(25)
        message = record.getMessage()

        # Context annotations
        annotations = []
        request_id = _request_id_var.get()
        if request_id:
            annotations.append(f"req={request_id[:8]}")
        if hasattr(record, "business_id"):
            annotations.append(f"biz={record.business_id}")
        annotation_str = f"[{', '.join(annotations)}] " if annotations else ""

        if self._use_colors:
            color = self._LEVEL_COLORS.get(record.levelname, "")
            level_colored = f"{color}{level}{self._RESET}"
        else:
            level_colored = level

        line = (
            f"{timestamp} | {level_colored} | {service} | "
            f"{annotation_str}{message}"
        )

        # Append exception traceback if present
        if record.exc_info:
            tb = self.formatException(record.exc_info)
            line = f"{line}\n{tb}"

        return line


# ==============================================================================
# Request ID context manager
# ==============================================================================

class RequestContext:
    """
    Context manager that binds a request ID to the current async context.

    Usage (in FastAPI middleware):
        async with RequestContext(request_id="abc-123"):
            await call_next(request)

    All log lines emitted during the request will include request_id="abc-123".
    """

    def __init__(self, request_id: Optional[str] = None) -> None:
        self._request_id = request_id or str(uuid.uuid4())
        self._token = None

    def __enter__(self) -> "RequestContext":
        self._token = _request_id_var.set(self._request_id)
        return self

    def __exit__(self, *args) -> None:
        if self._token is not None:
            _request_id_var.reset(self._token)

    async def __aenter__(self) -> "RequestContext":
        return self.__enter__()

    async def __aexit__(self, *args) -> None:
        self.__exit__()

    @property
    def request_id(self) -> str:
        return self._request_id


def get_request_id() -> Optional[str]:
    """Return the current request ID from context, or None."""
    return _request_id_var.get()


def bind_request_id(request_id: Optional[str] = None) -> str:
    """
    Manually bind a request ID to the current context.
    Returns the bound request ID.
    """
    rid = request_id or str(uuid.uuid4())
    _request_id_var.set(rid)
    return rid


# ==============================================================================
# Configure logging — called once at application startup
# ==============================================================================

def configure_logging() -> None:
    """
    Configure the root logging system for the entire application.

    Must be called once at application startup in main.py before
    any other module initialises its logger.

    Reads from environment variables:
      APP_ENV:    "production" | "development" | "testing"  (default: production)
      LOG_LEVEL:  "DEBUG" | "INFO" | "WARNING" | "ERROR"    (default: by env)
      LOG_FORMAT: "json" | "text"                           (default: by env)

    Production defaults:
      - LOG_LEVEL=WARNING
      - LOG_FORMAT=json
      - Output: stdout (captured by container/systemd)

    Development defaults:
      - LOG_LEVEL=DEBUG
      - LOG_FORMAT=text
      - Output: stdout with colours
    """
    environment = os.getenv("APP_ENV", "production").lower()
    is_production = environment == "production"
    is_testing = environment == "testing"

    # Determine log level
    default_level = "WARNING" if is_production else "DEBUG"
    if is_testing:
        default_level = "ERROR"   # keep test output clean
    log_level_str = os.getenv("LOG_LEVEL", default_level).upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    # Determine format
    default_format = "json" if is_production else "text"
    log_format = os.getenv("LOG_FORMAT", default_format).lower()

    # Build formatter
    if log_format == "json":
        formatter = JsonFormatter(environment=environment)
    else:
        use_colors = not is_testing
        formatter = TextFormatter(use_colors=use_colors)

    # Console handler — all logs go to stdout
    # In production this is captured by the container runtime / systemd
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Sensitive data filter — attached to handler so it runs before emit
    sensitive_filter = SensitiveDataFilter()
    console_handler.addFilter(sensitive_filter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers (prevents duplicate output if called twice)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy_logger_name in _NOISY_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    # SQLAlchemy query logging — only enable in development at DEBUG level
    if not is_production and log_level <= logging.DEBUG:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # Confirm logging is initialised
    startup_logger = logging.getLogger("app.startup")
    startup_logger.info(
        "Logging configured",
        extra={
            "environment": environment,
            "log_level": log_level_str,
            "log_format": log_format,
        },
    )


# ==============================================================================
# get_logger — convenience wrapper used throughout the codebase
# ==============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Convenience wrapper so modules can import from one place:

        from app.logging.logger import get_logger
        logger = get_logger(ServiceName.REVIEW_MONITOR)

    Equivalent to logging.getLogger(name) but signals intent clearly.

    Args:
        name: Logger name — should be a ServiceName constant.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


# ==============================================================================
# log_error — structured error logging helper
# ==============================================================================

def log_error(
    logger: logging.Logger,
    message: str,
    exc: Optional[Exception] = None,
    business_id: Optional[str] = None,
    service: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Emit a structured ERROR log with consistent fields.

    Centralises error log formatting so every error in the system
    has the same shape — easier to query in log aggregators.

    Args:
        logger:       Logger instance (from get_logger or getLogger).
        message:      Human-readable error description.
        exc:          Exception instance (optional).
        business_id:  Affected business UUID (optional).
        service:      Service name override (optional).
        extra:        Additional fields to include (optional).
    """
    log_extra: dict[str, Any] = {}

    if business_id:
        log_extra["business_id"] = business_id
    if service:
        log_extra["service"] = service
    if extra:
        log_extra.update(extra)
    if exc:
        log_extra["error"] = str(exc)
        log_extra["error_type"] = type(exc).__name__
        log_extra["traceback"] = traceback.format_exc()

    logger.error(message, extra=log_extra, exc_info=exc is not None)