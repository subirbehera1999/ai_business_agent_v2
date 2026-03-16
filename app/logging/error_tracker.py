# ==============================================================================
# File: app/logging/error_tracker.py
# Purpose: Captures, persists, and surfaces critical errors for admin
#          visibility across the entire AI Business Agent platform.
#
#          Role in the system:
#            logger.py  → emits log lines to stdout / log files
#            error_tracker.py → stores errors in the database AND
#                               sends WhatsApp alerts for critical failures
#
#          The two work together but serve different purposes:
#            - Logs are ephemeral (rotated, cleared, not queryable long-term)
#            - Tracked errors are persisted in the database and queryable
#              by the admin dashboard
#
#          What gets tracked:
#            Every call to ErrorTracker.capture() stores a record with:
#              - error_type:    exception class name
#              - message:       human-readable description
#              - service:       which module raised the error
#              - business_id:   affected business (if applicable)
#              - severity:      LOW / MEDIUM / HIGH / CRITICAL
#              - traceback:     full Python traceback (truncated to 2000 chars)
#              - context:       arbitrary JSON dict for extra debug info
#              - fingerprint:   hash of (service + error_type + message[:50])
#                               used for deduplication
#              - occurrence_count: incremented on repeated identical errors
#              - first_seen_at / last_seen_at: timestamps
#
#          Severity levels and alerting behaviour:
#            LOW:      stored only — no alert
#            MEDIUM:   stored only — no alert
#            HIGH:     stored + admin WhatsApp alert (rate-limited)
#            CRITICAL: stored + immediate admin WhatsApp alert (always sent)
#
#          Deduplication:
#            Identical errors (same fingerprint) increment occurrence_count
#            rather than creating new rows. This prevents one broken scheduler
#            job from flooding the error table with thousands of rows.
#
#          Auto-severity detection:
#            If severity is not explicitly provided, ErrorTracker infers it
#            from the exception type and service name:
#              - PaymentError / WebhookError → CRITICAL
#              - DatabaseError / IntegrityError → HIGH
#              - ExternalAPIError / TimeoutError → MEDIUM
#              - Everything else → LOW
#
#          Cleanup policy:
#            Errors older than ERROR_RETENTION_DAYS (90) are deleted
#            by the daily cleanup job. This matches job_model.py cleanup.
#
#          Usage:
#            # In any service or scheduler:
#            from app.logging.error_tracker import ErrorTracker
#
#            await error_tracker.capture(
#                db=db,
#                message="AI reply generation failed",
#                exc=exc,
#                service=ServiceName.REVIEW_MONITOR,
#                business_id=business_id,
#                severity=ErrorSeverity.MEDIUM,
#                context={"review_id": review_id},
#            )
#
#          NOTE: capture() never raises. A broken error tracker must
#          never crash the system that is trying to report an error.
# ==============================================================================

import hashlib
import logging
import traceback as tb_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.base import Base
from app.utils.time_utils import now_utc

logger = logging.getLogger(ServiceName.ERROR_TRACKER)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ERROR_RETENTION_DAYS: int = 90          # errors older than this are deleted
MAX_TRACEBACK_LENGTH: int = 2000        # truncate long tracebacks
MAX_CONTEXT_LENGTH: int = 1000          # truncate large context dicts
MAX_SAME_ERROR_ALERTS_PER_HOUR: int = 3 # rate limit for HIGH severity alerts


# ==============================================================================
# Severity constants
# ==============================================================================

class ErrorSeverity:
    """
    Severity levels for tracked errors.

    LOW:      Minor issue, no user impact. Logged only.
    MEDIUM:   Degraded functionality. Logged only.
    HIGH:     Significant failure, business impacted. Alert sent.
    CRITICAL: Payment, data loss, or system-wide failure. Always alerted.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    ALL = (LOW, MEDIUM, HIGH, CRITICAL)

    # Severity rank for comparison
    _RANK = {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}

    @classmethod
    def rank(cls, severity: str) -> int:
        return cls._RANK.get(severity, 0)

    @classmethod
    def is_alertable(cls, severity: str) -> bool:
        """Return True if this severity level should trigger an admin alert."""
        return severity in (cls.HIGH, cls.CRITICAL)


# ==============================================================================
# ErrorLog database model
# ==============================================================================

class ErrorLog(Base):
    """
    Persisted error record in the database.

    Stores deduplicated errors with occurrence counting so the admin
    can see both individual failures and recurring patterns.

    Cleanup: rows older than ERROR_RETENTION_DAYS days are deleted
    by the daily maintenance job.
    """
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Error identity
    fingerprint = Column(String(64), nullable=False, index=True)
    error_type = Column(String(128), nullable=False)
    message = Column(Text, nullable=False)
    service = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, index=True)

    # Context
    business_id = Column(String(36), nullable=True, index=True)
    traceback = Column(Text, nullable=True)
    context = Column(JSONB, nullable=True)

    # Occurrence tracking
    occurrence_count = Column(Integer, nullable=False, default=1)
    first_seen_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_seen_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Alert tracking
    alert_sent = Column(Integer, nullable=False, default=0)
    last_alert_at = Column(DateTime(timezone=True), nullable=True)

    # Standard audit timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<ErrorLog id={self.id} "
            f"service={self.service} "
            f"severity={self.severity} "
            f"occurrences={self.occurrence_count}>"
        )


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class CaptureResult:
    """
    Result of ErrorTracker.capture().

    Attributes:
        stored:       True if error was persisted to DB.
        deduplicated: True if this was a repeat of an existing error.
        alert_sent:   True if an admin WhatsApp alert was dispatched.
        fingerprint:  Deduplication hash for this error.
        error_log_id: DB row ID (None if storage failed).
    """
    stored: bool = False
    deduplicated: bool = False
    alert_sent: bool = False
    fingerprint: str = ""
    error_log_id: Optional[int] = None


# ==============================================================================
# Error Tracker
# ==============================================================================

class ErrorTracker:
    """
    Captures, deduplicates, and persists errors across the platform.

    Responsibilities:
      - Store errors in the error_logs table
      - Deduplicate repeated identical errors
      - Send admin WhatsApp alerts for HIGH and CRITICAL errors
      - Provide query methods for the admin dashboard

    Dependency:
      Requires a WhatsApp notification service for alerting.
      Pass None to disable alerting (useful in tests).

    Usage:
        error_tracker = ErrorTracker(
            admin_whatsapp=admin_notification_service
        )

        await error_tracker.capture(
            db=db,
            message="Webhook signature verification failed",
            exc=exc,
            service=ServiceName.PAYMENTS,
            severity=ErrorSeverity.CRITICAL,
        )
    """

    def __init__(self, admin_whatsapp=None) -> None:
        """
        Args:
            admin_whatsapp: AdminNotificationService instance.
                            Pass None to disable WhatsApp alerting.
        """
        self._admin_whatsapp = admin_whatsapp
        # In-memory rate limiter for HIGH severity alerts
        # Key: fingerprint → list of alert timestamps this hour
        self._alert_timestamps: dict[str, list[datetime]] = {}

    # ------------------------------------------------------------------
    # Primary method — capture an error
    # ------------------------------------------------------------------

    async def capture(
        self,
        db: AsyncSession,
        message: str,
        exc: Optional[Exception] = None,
        service: str = "unknown",
        business_id: Optional[str] = None,
        severity: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> CaptureResult:
        """
        Capture and persist an error event.

        This method NEVER raises. If the error tracker itself fails,
        it logs to stdout and returns a failed CaptureResult.

        Args:
            db:           AsyncSession.
            message:      Human-readable error description.
            exc:          Exception instance (optional but recommended).
            service:      ServiceName constant of the reporting module.
            business_id:  Affected business UUID (optional).
            severity:     ErrorSeverity constant. Auto-detected if None.
            context:      Extra debug information as a dict (optional).

        Returns:
            CaptureResult.
        """
        result = CaptureResult()

        try:
            # ── Derive error metadata ─────────────────────────────────
            error_type = type(exc).__name__ if exc else "ApplicationError"
            tb_text = _extract_traceback(exc)
            resolved_severity = severity or _infer_severity(exc, service)
            fingerprint = _make_fingerprint(service, error_type, message)
            result.fingerprint = fingerprint

            safe_context = _sanitise_context(context)

            # ── Upsert: deduplicate or create ─────────────────────────
            existing = await _find_existing(
                db=db,
                fingerprint=fingerprint,
            )

            if existing:
                # Increment occurrence count on existing record
                existing.occurrence_count += 1
                existing.last_seen_at = now_utc()
                if business_id and not existing.business_id:
                    existing.business_id = business_id
                await db.flush()

                result.stored = True
                result.deduplicated = True
                result.error_log_id = existing.id

                logger.debug(
                    "Duplicate error captured — count=%d fingerprint=%s",
                    existing.occurrence_count,
                    fingerprint[:12],
                )
            else:
                # Create new error log row
                error_log = ErrorLog(
                    fingerprint=fingerprint,
                    error_type=error_type,
                    message=message[:500],
                    service=service,
                    severity=resolved_severity,
                    business_id=business_id,
                    traceback=tb_text,
                    context=safe_context,
                    occurrence_count=1,
                    first_seen_at=now_utc(),
                    last_seen_at=now_utc(),
                    alert_sent=0,
                )
                db.add(error_log)
                await db.flush()

                result.stored = True
                result.deduplicated = False
                result.error_log_id = error_log.id

                logger.debug(
                    "New error captured service=%s severity=%s fingerprint=%s",
                    service,
                    resolved_severity,
                    fingerprint[:12],
                )

            await db.commit()

            # ── Admin alert for HIGH / CRITICAL ───────────────────────
            if ErrorSeverity.is_alertable(resolved_severity):
                alert_sent = await self._maybe_send_alert(
                    fingerprint=fingerprint,
                    message=message,
                    service=service,
                    severity=resolved_severity,
                    business_id=business_id,
                    error_type=error_type,
                    occurrence_count=(
                        existing.occurrence_count if existing else 1
                    ),
                )
                result.alert_sent = alert_sent

        except Exception as tracker_exc:
            # The error tracker itself crashed — log to stdout only
            # Never let this propagate up to the caller
            logger.error(
                "ErrorTracker.capture() failed — error not persisted",
                extra={
                    "original_message": message[:100],
                    "tracker_error": str(tracker_exc),
                    "service": service,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Alert dispatch
    # ------------------------------------------------------------------

    async def _maybe_send_alert(
        self,
        fingerprint: str,
        message: str,
        service: str,
        severity: str,
        business_id: Optional[str],
        error_type: str,
        occurrence_count: int,
    ) -> bool:
        """
        Send admin WhatsApp alert respecting rate limits.

        CRITICAL: always sent (bypasses rate limiter).
        HIGH:     rate-limited to MAX_SAME_ERROR_ALERTS_PER_HOUR per fingerprint.

        Returns True if alert was sent.
        """
        if not self._admin_whatsapp:
            return False

        if severity == ErrorSeverity.CRITICAL:
            # Critical errors always get through
            return await self._dispatch_alert(
                message=message,
                service=service,
                severity=severity,
                business_id=business_id,
                error_type=error_type,
                occurrence_count=occurrence_count,
            )

        # HIGH severity — apply rate limit
        if not self._within_alert_rate_limit(fingerprint):
            logger.debug(
                "Alert rate limit hit for fingerprint=%s",
                fingerprint[:12],
            )
            return False

        return await self._dispatch_alert(
            message=message,
            service=service,
            severity=severity,
            business_id=business_id,
            error_type=error_type,
            occurrence_count=occurrence_count,
        )

    def _within_alert_rate_limit(self, fingerprint: str) -> bool:
        """
        Check if we are within the per-fingerprint hourly alert limit.
        Evicts timestamps older than 1 hour before checking.
        """
        now = now_utc()
        cutoff = now - timedelta(hours=1)

        timestamps = self._alert_timestamps.get(fingerprint, [])
        # Evict old timestamps
        fresh = [t for t in timestamps if t > cutoff]
        self._alert_timestamps[fingerprint] = fresh

        if len(fresh) >= MAX_SAME_ERROR_ALERTS_PER_HOUR:
            return False

        # Record this alert attempt
        self._alert_timestamps[fingerprint].append(now)
        return True

    async def _dispatch_alert(
        self,
        message: str,
        service: str,
        severity: str,
        business_id: Optional[str],
        error_type: str,
        occurrence_count: int,
    ) -> bool:
        """
        Send the actual admin WhatsApp alert. Never raises.
        """
        try:
            severity_emoji = {
                ErrorSeverity.CRITICAL: "🚨",
                ErrorSeverity.HIGH: "🔴",
            }.get(severity, "⚠️")

            biz_line = (
                f"\nBusiness: `{business_id}`" if business_id else ""
            )
            count_line = (
                f"\nOccurrences: {occurrence_count}"
                if occurrence_count > 1 else ""
            )

            alert_message = (
                f"{severity_emoji} *{severity.upper()} Error — {service}*\n\n"
                f"Type: `{error_type}`\n"
                f"Message: {message[:200]}"
                f"{biz_line}"
                f"{count_line}\n\n"
                f"_Check error_logs table for full details._"
            )

            await self._admin_whatsapp.send_critical(
                message=alert_message,
            )
            return True

        except Exception as exc:
            logger.warning(
                "ErrorTracker alert dispatch failed",
                extra={"error": str(exc)},
            )
            return False

    # ------------------------------------------------------------------
    # Query methods — used by admin dashboard and health checks
    # ------------------------------------------------------------------

    async def get_recent_errors(
        self,
        db: AsyncSession,
        severity: Optional[str] = None,
        service: Optional[str] = None,
        business_id: Optional[str] = None,
        hours: int = 24,
        limit: int = 50,
    ) -> list[ErrorLog]:
        """
        Fetch recent error records for admin review.

        Args:
            db:           AsyncSession.
            severity:     Filter by ErrorSeverity (optional).
            service:      Filter by service name (optional).
            business_id:  Filter by business UUID (optional).
            hours:        Lookback window in hours (default 24).
            limit:        Maximum records to return (default 50).

        Returns:
            List of ErrorLog records, most recent first.
        """
        try:
            cutoff = now_utc() - timedelta(hours=hours)
            stmt = (
                select(ErrorLog)
                .where(ErrorLog.last_seen_at >= cutoff)
                .order_by(ErrorLog.last_seen_at.desc())
                .limit(limit)
            )

            if severity:
                stmt = stmt.where(ErrorLog.severity == severity)
            if service:
                stmt = stmt.where(ErrorLog.service == service)
            if business_id:
                stmt = stmt.where(ErrorLog.business_id == business_id)

            result = await db.execute(stmt)
            return list(result.scalars().all())

        except Exception as exc:
            logger.error(
                "get_recent_errors query failed",
                extra={"error": str(exc)},
            )
            return []

    async def get_error_summary(
        self,
        db: AsyncSession,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Return a summary count of errors by severity for the given window.

        Used by system_health.py and admin_health_report.py.

        Returns:
            Dict with keys: total, critical, high, medium, low, by_service.
        """
        try:
            cutoff = now_utc() - timedelta(hours=hours)

            stmt = (
                select(
                    ErrorLog.severity,
                    func.count(ErrorLog.id).label("count"),
                    func.sum(ErrorLog.occurrence_count).label("occurrences"),
                )
                .where(ErrorLog.last_seen_at >= cutoff)
                .group_by(ErrorLog.severity)
            )
            rows = await db.execute(stmt)
            severity_rows = rows.all()

            by_severity = {s: 0 for s in ErrorSeverity.ALL}
            total_occurrences = 0
            for row in severity_rows:
                by_severity[row.severity] = row.count
                total_occurrences += (row.occurrences or 0)

            # By service
            service_stmt = (
                select(
                    ErrorLog.service,
                    func.count(ErrorLog.id).label("count"),
                )
                .where(ErrorLog.last_seen_at >= cutoff)
                .group_by(ErrorLog.service)
                .order_by(func.count(ErrorLog.id).desc())
                .limit(10)
            )
            service_rows = await db.execute(service_stmt)
            by_service = {
                row.service: row.count
                for row in service_rows.all()
            }

            return {
                "total_unique_errors": sum(by_severity.values()),
                "total_occurrences": total_occurrences,
                "critical": by_severity[ErrorSeverity.CRITICAL],
                "high": by_severity[ErrorSeverity.HIGH],
                "medium": by_severity[ErrorSeverity.MEDIUM],
                "low": by_severity[ErrorSeverity.LOW],
                "by_service": by_service,
                "window_hours": hours,
            }

        except Exception as exc:
            logger.error(
                "get_error_summary query failed",
                extra={"error": str(exc)},
            )
            return {
                "total_unique_errors": 0,
                "total_occurrences": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "by_service": {},
                "window_hours": hours,
            }

    async def delete_old_errors(
        self,
        db: AsyncSession,
        retention_days: int = ERROR_RETENTION_DAYS,
    ) -> int:
        """
        Delete error records older than retention_days.

        Called by the daily maintenance job to keep the table lean.

        Returns:
            Number of rows deleted.
        """
        try:
            from sqlalchemy import delete as sql_delete

            cutoff = now_utc() - timedelta(days=retention_days)
            stmt = sql_delete(ErrorLog).where(
                ErrorLog.last_seen_at < cutoff
            )
            result = await db.execute(stmt)
            await db.commit()
            deleted = result.rowcount or 0

            if deleted > 0:
                logger.info(
                    "Old error logs deleted",
                    extra={
                        "deleted": deleted,
                        "retention_days": retention_days,
                    },
                )
            return deleted

        except Exception as exc:
            logger.error(
                "delete_old_errors failed",
                extra={"error": str(exc)},
            )
            return 0


# ==============================================================================
# Pure helpers
# ==============================================================================

def _make_fingerprint(service: str, error_type: str, message: str) -> str:
    """
    Create a stable SHA-256 fingerprint for deduplication.

    Same service + error_type + first 80 chars of message
    always produces the same fingerprint — even if the traceback
    changes between occurrences.
    """
    raw = f"{service}:{error_type}:{message[:80]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_traceback(exc: Optional[Exception]) -> Optional[str]:
    """
    Extract and truncate the traceback from an exception.

    Returns None if no exception provided.
    Truncates to MAX_TRACEBACK_LENGTH to keep the DB row compact.
    """
    if exc is None:
        return None
    try:
        full_tb = tb_module.format_exc()
        if full_tb and full_tb.strip() != "NoneType: None":
            return full_tb[-MAX_TRACEBACK_LENGTH:]
        # Fall back to just the exception repr
        return repr(exc)[:MAX_TRACEBACK_LENGTH]
    except Exception:
        return repr(exc)[:MAX_TRACEBACK_LENGTH]


def _sanitise_context(context: Optional[dict]) -> Optional[dict]:
    """
    Sanitise the context dict before storing to DB.

    Removes sensitive keys and truncates large string values.
    Returns None if context is empty or None.
    """
    if not context:
        return None

    from app.config.constants import _SENSITIVE_FIELD_NAMES

    safe = {}
    for key, value in context.items():
        if key.lower() in _SENSITIVE_FIELD_NAMES:
            safe[key] = "***REDACTED***"
        elif isinstance(value, str) and len(value) > MAX_CONTEXT_LENGTH:
            safe[key] = value[:MAX_CONTEXT_LENGTH] + "...[truncated]"
        else:
            safe[key] = value

    return safe if safe else None


def _infer_severity(
    exc: Optional[Exception],
    service: str,
) -> str:
    """
    Automatically infer error severity from exception type and service.

    Rules (in priority order):
      - Payment or webhook service → CRITICAL
      - DatabaseError / IntegrityError → HIGH
      - TimeoutError / ConnectionError → MEDIUM
      - Everything else → LOW
    """
    # Service-based rules
    critical_services = {
        ServiceName.PAYMENTS,
        ServiceName.WEBHOOKS,
    }
    if service in critical_services:
        return ErrorSeverity.CRITICAL

    if exc is None:
        return ErrorSeverity.LOW

    error_type = type(exc).__name__.lower()

    # Exception-type-based rules
    if any(k in error_type for k in ("integrity", "database", "db", "sqlalchemy")):
        return ErrorSeverity.HIGH

    if any(k in error_type for k in ("timeout", "connection", "network", "api")):
        return ErrorSeverity.MEDIUM

    return ErrorSeverity.LOW


async def _find_existing(
    db: AsyncSession,
    fingerprint: str,
) -> Optional[ErrorLog]:
    """
    Look up an existing ErrorLog by fingerprint.

    Only matches records seen in the last 24 hours to prevent
    infinitely incrementing old stale records.
    """
    cutoff = now_utc() - timedelta(hours=24)
    stmt = (
        select(ErrorLog)
        .where(
            ErrorLog.fingerprint == fingerprint,
            ErrorLog.last_seen_at >= cutoff,
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()