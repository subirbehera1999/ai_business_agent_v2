# ==============================================================================
# File: app/utils/system_health.py
# Purpose: Infrastructure health checks for the AI Business Agent platform.
#
#          This file is a pure utility module — it only checks whether each
#          external dependency is reachable and operational. It never sends
#          notifications, never writes to the database, and never schedules
#          anything. Those concerns belong to admin_health_report.py and
#          admin_notification_service.py.
#
#          Functions exposed:
#
#            check_database_health()      — verifies PostgreSQL is reachable
#            check_scheduler_health()     — verifies APScheduler is running
#            check_google_api_health()    — verifies Google API reachability
#            check_whatsapp_api_health()  — verifies WhatsApp Cloud API
#            check_openai_api_health()    — verifies OpenAI API reachability
#            run_all_checks()             — runs all 5 checks, returns SystemHealthResult
#
#          SystemHealthResult dataclass:
#            Bundles the boolean result of every check plus a timestamp.
#            Used by admin_health_report.py to populate send_health_summary().
#
#          Check strategy:
#            Each external API check performs a lightweight HTTP HEAD or GET
#            request with a short timeout (HEALTH_CHECK_TIMEOUT_SECONDS).
#            A 2xx or 4xx response is considered "reachable" — the API is
#            responding even if it rejected our unauthenticated request.
#            A connection error, timeout, or 5xx response is "unreachable".
#
#            Why accept 4xx as healthy?
#            Pinging https://api.openai.com without credentials returns 401.
#            That 401 proves the API is up and routing correctly. A 500 or
#            connection refused means the API itself is having problems.
#
#          Database check:
#            Delegates to app.database.db.check_database_health() which
#            already executes "SELECT 1" on the live connection pool.
#            No duplicate logic here.
#
#          Scheduler check:
#            Accepts an optional SchedulerManager instance. If None is passed
#            (e.g. in a context where the scheduler is not available), the
#            check returns False rather than crashing.
#
#          Timeout:
#            All HTTP checks use HEALTH_CHECK_TIMEOUT_SECONDS = 8 seconds.
#            This is shorter than the external API retry timeout to ensure
#            health checks complete well within the scheduler job window.
#
#          Never raises:
#            All check functions catch all exceptions and return False on
#            any failure. Health checks must be safe to call from any context.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config.constants import HEALTH_CHECK_URLS, ServiceName
from app.database.db import check_database_health as _db_health_check

logger = logging.getLogger(ServiceName.SYSTEM_HEALTH)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Timeout in seconds for each external API HTTP check.
# Short enough that all checks complete within 40 seconds total.
HEALTH_CHECK_TIMEOUT_SECONDS: float = 8.0

# HTTP status codes that indicate the API is UP (even if auth failed).
# 401, 403, 404 all mean the server responded — it's reachable.
_REACHABLE_STATUS_CODES = frozenset(range(100, 500))  # 1xx–4xx = reachable

# Individual API health check URLs.
# These are the base URLs — we do not need a valid endpoint, just connectivity.
_OPENAI_HEALTH_URL:    str = "https://api.openai.com"
_WHATSAPP_HEALTH_URL:  str = "https://graph.facebook.com"
_GOOGLE_HEALTH_URL:    str = "https://maps.googleapis.com"


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class SystemHealthResult:
    """
    Consolidated result of all system infrastructure health checks.

    Produced by run_all_checks() and consumed by admin_health_report.py
    to populate admin_notification_service.send_health_summary().

    Attributes:
        db_ok:              PostgreSQL database is reachable.
        scheduler_ok:       APScheduler is running and active.
        google_api_ok:      Google API endpoint is reachable.
        whatsapp_api_ok:    WhatsApp Cloud API endpoint is reachable.
        openai_api_ok:      OpenAI API endpoint is reachable.
        all_ok:             True only if every check passed.
        checked_at:         UTC timestamp when checks were run.
        failures:           List of failed check names for log context.
    """
    db_ok:           bool
    scheduler_ok:    bool
    google_api_ok:   bool
    whatsapp_api_ok: bool
    openai_api_ok:   bool
    checked_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failures:        list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """True only if every infrastructure check passed."""
        return all([
            self.db_ok,
            self.scheduler_ok,
            self.google_api_ok,
            self.whatsapp_api_ok,
            self.openai_api_ok,
        ])

    def to_log_dict(self) -> dict:
        """Return a structured dict for logging."""
        return {
            "db_ok":           self.db_ok,
            "scheduler_ok":    self.scheduler_ok,
            "google_api_ok":   self.google_api_ok,
            "whatsapp_api_ok": self.whatsapp_api_ok,
            "openai_api_ok":   self.openai_api_ok,
            "all_ok":          self.all_ok,
            "failures":        self.failures,
            "checked_at":      self.checked_at.isoformat(),
        }


# ==============================================================================
# Individual health check functions
# ==============================================================================

async def check_database_health() -> bool:
    """
    Verify that the PostgreSQL database is reachable and accepting queries.

    Delegates to app.database.db.check_database_health() which executes
    "SELECT 1" on the live async connection pool. No new connection logic
    is added here — this is a thin wrapper for a consistent calling interface.

    Returns:
        bool: True if the database responded successfully, False otherwise.
    """
    try:
        result = await _db_health_check()
        if result:
            logger.debug(
                "Database health check: OK",
                extra={"service": ServiceName.SYSTEM_HEALTH},
            )
        else:
            logger.warning(
                "Database health check: FAILED",
                extra={"service": ServiceName.SYSTEM_HEALTH},
            )
        return result
    except Exception as exc:
        logger.error(
            "Database health check raised unexpected exception",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )
        return False


def check_scheduler_health(scheduler_manager: Optional[object] = None) -> bool:
    """
    Verify that the APScheduler instance is running.

    Accepts the SchedulerManager instance from main.py. If None is passed
    (e.g. in a context where the scheduler is not injected), returns False
    rather than crashing — a missing scheduler is a real failure condition.

    Args:
        scheduler_manager: The SchedulerManager instance from
                           app.schedulers.scheduler_manager. Accepts Optional
                           to avoid a hard import cycle — pass it in from
                           admin_health_report.py at call time.

    Returns:
        bool: True if the scheduler is running, False otherwise.
    """
    try:
        if scheduler_manager is None:
            logger.warning(
                "Scheduler health check: no SchedulerManager provided — "
                "reporting as not running",
                extra={"service": ServiceName.SYSTEM_HEALTH},
            )
            return False

        is_running: bool = getattr(scheduler_manager, "is_running", False)

        if is_running:
            logger.debug(
                "Scheduler health check: OK",
                extra={"service": ServiceName.SYSTEM_HEALTH},
            )
        else:
            logger.warning(
                "Scheduler health check: FAILED — scheduler not running",
                extra={"service": ServiceName.SYSTEM_HEALTH},
            )

        return bool(is_running)

    except Exception as exc:
        logger.error(
            "Scheduler health check raised unexpected exception",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )
        return False


async def check_google_api_health() -> bool:
    """
    Verify that the Google API endpoint is reachable.

    Sends a lightweight HEAD request to the Google Maps API base URL.
    Any response in the 1xx–4xx range is treated as "reachable" — even a
    403 (authentication required) proves the API is up and routing.
    Connection errors and 5xx responses are treated as "unreachable".

    Returns:
        bool: True if Google API is reachable, False otherwise.
    """
    return await _http_health_check(
        name="Google API",
        url=_GOOGLE_HEALTH_URL,
    )


async def check_whatsapp_api_health() -> bool:
    """
    Verify that the WhatsApp Cloud API (Meta Graph API) is reachable.

    Sends a HEAD request to https://graph.facebook.com.
    A 400 or 401 response is expected and treated as healthy — the API
    is responding even though our request has no valid path or token.

    Returns:
        bool: True if WhatsApp Cloud API is reachable, False otherwise.
    """
    return await _http_health_check(
        name="WhatsApp API",
        url=_WHATSAPP_HEALTH_URL,
    )


async def check_openai_api_health() -> bool:
    """
    Verify that the OpenAI API endpoint is reachable.

    Sends a HEAD request to https://api.openai.com.
    A 401 (unauthorized) response is expected and treated as healthy.
    Connection refused or 5xx means OpenAI is having problems.

    Returns:
        bool: True if OpenAI API is reachable, False otherwise.
    """
    return await _http_health_check(
        name="OpenAI API",
        url=_OPENAI_HEALTH_URL,
    )


# ==============================================================================
# Composite check — run all checks and return SystemHealthResult
# ==============================================================================

async def run_all_checks(
    scheduler_manager: Optional[object] = None,
) -> SystemHealthResult:
    """
    Run all infrastructure health checks and return a consolidated result.

    Called by admin_health_report.run_health_report() to gather all check
    results before calling admin_notification_service.send_health_summary().

    Checks are run sequentially (not concurrently) to avoid overwhelming
    the connection pool or triggering API rate limits during health probes.
    All checks complete within 5 × HEALTH_CHECK_TIMEOUT_SECONDS ≈ 40 seconds
    in the worst case — well within the scheduler job window.

    Args:
        scheduler_manager: Optional SchedulerManager instance.
                           Pass the live instance from main.py for an accurate
                           scheduler check. Pass None if unavailable.

    Returns:
        SystemHealthResult: Bundled results for all checks.
    """
    log_extra = {"service": ServiceName.SYSTEM_HEALTH}
    logger.info("Running all system health checks", extra=log_extra)

    failures: list[str] = []

    # 1. Database
    db_ok = await check_database_health()
    if not db_ok:
        failures.append("database")

    # 2. Scheduler
    scheduler_ok = check_scheduler_health(scheduler_manager)
    if not scheduler_ok:
        failures.append("scheduler")

    # 3. Google API
    google_ok = await check_google_api_health()
    if not google_ok:
        failures.append("google_api")

    # 4. WhatsApp API
    whatsapp_ok = await check_whatsapp_api_health()
    if not whatsapp_ok:
        failures.append("whatsapp_api")

    # 5. OpenAI API
    openai_ok = await check_openai_api_health()
    if not openai_ok:
        failures.append("openai_api")

    result = SystemHealthResult(
        db_ok=db_ok,
        scheduler_ok=scheduler_ok,
        google_api_ok=google_ok,
        whatsapp_api_ok=whatsapp_ok,
        openai_api_ok=openai_ok,
        failures=failures,
    )

    if result.all_ok:
        logger.info(
            "All health checks passed",
            extra=log_extra,
        )
    else:
        logger.warning(
            "Health checks failed for: %s",
            ", ".join(failures),
            extra={**log_extra, "failures": failures},
        )

    return result


# ==============================================================================
# Internal HTTP check helper
# ==============================================================================

async def _http_health_check(name: str, url: str) -> bool:
    """
    Perform a lightweight HTTP HEAD request to verify API reachability.

    Uses httpx.AsyncClient with a strict timeout. Any response in the
    1xx–4xx range is treated as reachable. 5xx responses and connection
    errors are treated as unreachable.

    Why HEAD over GET?
    HEAD requests do not return a response body, so they consume minimal
    bandwidth and are faster. All major API providers support HEAD on
    their base URLs. If HEAD returns 405 (method not allowed), we fall
    back to GET — that 405 itself proves the server is reachable.

    Args:
        name: Human-readable name of the service (for logs).
        url:  Base URL to check.

    Returns:
        bool: True if reachable, False otherwise. Never raises.
    """
    log_extra = {
        "service": ServiceName.SYSTEM_HEALTH,
        "check_name": name,
        "url": url,
    }

    try:
        async with httpx.AsyncClient(
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            try:
                response = await client.head(url)
            except Exception:
                # HEAD not supported — try GET as fallback
                response = await client.get(url)

            reachable = response.status_code in _REACHABLE_STATUS_CODES

            if reachable:
                logger.debug(
                    "%s health check: OK (HTTP %d)",
                    name,
                    response.status_code,
                    extra={**log_extra, "status_code": response.status_code},
                )
            else:
                logger.warning(
                    "%s health check: FAILED (HTTP %d)",
                    name,
                    response.status_code,
                    extra={**log_extra, "status_code": response.status_code},
                )

            return reachable

    except httpx.TimeoutException:
        logger.warning(
            "%s health check: TIMEOUT after %.1fs",
            name,
            HEALTH_CHECK_TIMEOUT_SECONDS,
            extra={**log_extra, "reason": "timeout"},
        )
        return False

    except httpx.ConnectError as exc:
        logger.warning(
            "%s health check: CONNECTION REFUSED",
            name,
            extra={**log_extra, "reason": "connect_error", "error": str(exc)},
        )
        return False

    except Exception as exc:
        logger.error(
            "%s health check: UNEXPECTED ERROR",
            name,
            extra={**log_extra, "reason": "unexpected", "error": str(exc)},
        )
        return False