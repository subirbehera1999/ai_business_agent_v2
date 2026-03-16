# ==============================================================================
# File: app/utils/idempotency_utils.py
# Purpose: Provides idempotency key generation and duplicate execution
#          prevention for all critical operations in the system.
#
#          Problem solved:
#            Network retries, scheduler restarts, and webhook redeliveries
#            can cause the same operation to be attempted more than once.
#            This utility ensures each operation executes exactly once,
#            regardless of how many times it is triggered.
#
#          Strategy:
#            1. Before executing a task, generate a deterministic key
#               that uniquely identifies this specific operation instance.
#            2. Check whether the key has already been executed.
#            3. If yes — skip. If no — execute and record the key.
#
#          Storage backends:
#            - Database (UsageModel / ReviewModel / PaymentModel columns)
#              → used for durable, cross-restart persistence
#            - In-memory cache (dict)
#              → used for within-process deduplication in unit tests
#
#          Key formats are defined in app/config/constants.py and are
#          deterministic — the same inputs always produce the same key.
# ==============================================================================

import hashlib
import logging
import uuid
from datetime import date, datetime
from typing import Optional

from app.config.constants import (
    IDEMPOTENCY_KEY_ALERT_SEND,
    IDEMPOTENCY_KEY_CONTENT_GEN,
    IDEMPOTENCY_KEY_PAYMENT_INIT,
    IDEMPOTENCY_KEY_PAYMENT_VERIFY,
    IDEMPOTENCY_KEY_REPORT_GEN,
    IDEMPOTENCY_KEY_REVIEW_PROCESS,
    IDEMPOTENCY_KEY_REVIEW_REPLY,
    ServiceName,
)

logger = logging.getLogger(ServiceName.API)


# ==============================================================================
# Key Generation — One function per operation type
# ==============================================================================

def make_review_reply_key(
    business_id: uuid.UUID,
    review_id: uuid.UUID,
) -> str:
    """
    Generate the idempotency key for an AI review reply generation attempt.

    This key is stored in ReviewModel.idempotency_key after a reply is
    successfully generated. If the scheduler retries the same review,
    this key already exists and the operation is skipped.

    Args:
        business_id:  UUID of the owning business.
        review_id:    UUID of the review being replied to.

    Returns:
        str: Deterministic idempotency key string.

    Example:
        key = make_review_reply_key(business_id, review_id)
        # → "BUSINESS_<uuid>_REVIEW_REPLY_REVIEWID_<uuid>"
    """
    return IDEMPOTENCY_KEY_REVIEW_REPLY.format(
        business_id=str(business_id),
        review_id=str(review_id),
    )


def make_review_process_key(
    business_id: uuid.UUID,
    review_id: uuid.UUID,
) -> str:
    """
    Generate the idempotency key for a review processing pipeline entry.

    Distinct from the reply key — this key guards the entire processing
    pipeline entry (sentiment + reply), whereas the reply key guards
    only the AI generation step.

    Args:
        business_id:  UUID of the owning business.
        review_id:    UUID of the review being processed.

    Returns:
        str: Deterministic idempotency key string.
    """
    return IDEMPOTENCY_KEY_REVIEW_PROCESS.format(
        business_id=str(business_id),
        review_id=str(review_id),
    )


def make_payment_init_key(
    business_id: uuid.UUID,
    billing_cycle: str,
) -> str:
    """
    Generate the idempotency key for a payment initiation request.

    Stored in PaymentModel.idempotency_key when the PENDING payment record
    is created. If the same business retries initiating a payment for the
    same billing cycle before completing checkout, the existing PENDING
    order is returned instead of creating a duplicate Razorpay order.

    Args:
        business_id:   UUID of the owning business.
        billing_cycle: Billing cycle string — "monthly" or "annual".

    Returns:
        str: Deterministic idempotency key string.

    Example:
        key = make_payment_init_key(business_id, "monthly")
        # → "PAYMENT_INIT_<uuid>_monthly"
    """
    return IDEMPOTENCY_KEY_PAYMENT_INIT.format(
        business_id=str(business_id),
        billing_cycle=billing_cycle,
    )


def make_payment_verify_key(
    business_id: uuid.UUID,
    payment_id: str,
) -> str:
    """
    Generate the idempotency key for a Razorpay payment verification.

    Stored in PaymentModel.idempotency_key after the webhook is processed.
    Prevents duplicate subscription activations when Razorpay delivers the
    same webhook event more than once.

    Args:
        business_id:  UUID of the owning business.
        payment_id:   Razorpay payment ID string (e.g., "pay_XXXXXX").

    Returns:
        str: Deterministic idempotency key string.
    """
    return IDEMPOTENCY_KEY_PAYMENT_VERIFY.format(
        business_id=str(business_id),
        payment_id=payment_id,
    )


def make_report_gen_key(
    business_id: uuid.UUID,
    report_type: str,
    report_date: Optional[date] = None,
) -> str:
    """
    Generate the idempotency key for a report generation job.

    Prevents duplicate weekly/monthly/quarterly reports when the scheduler
    restarts mid-cycle or a job is retried after a transient failure.

    Args:
        business_id:  UUID of the owning business.
        report_type:  Report type identifier (weekly / monthly / quarterly).
        report_date:  Date of the report period (defaults to today UTC).

    Returns:
        str: Deterministic idempotency key string.
    """
    effective_date = report_date or date.today()
    return IDEMPOTENCY_KEY_REPORT_GEN.format(
        business_id=str(business_id),
        report_type=report_type,
        date=effective_date.isoformat(),
    )


def make_content_gen_key(
    business_id: uuid.UUID,
    week_identifier: str,
) -> str:
    """
    Generate the idempotency key for a weekly social media content generation.

    The week_identifier should be a consistent string representation of the
    ISO week, e.g., "2024-W42", to ensure the same week always maps to the
    same key regardless of when within the week the job runs.

    Args:
        business_id:      UUID of the owning business.
        week_identifier:  ISO week string (e.g., "2024-W42").

    Returns:
        str: Deterministic idempotency key string.
    """
    return IDEMPOTENCY_KEY_CONTENT_GEN.format(
        business_id=str(business_id),
        week=week_identifier,
    )


def make_alert_send_key(
    business_id: uuid.UUID,
    alert_type: str,
    alert_date: Optional[date] = None,
) -> str:
    """
    Generate the idempotency key for a business alert dispatch.

    Used as AlertModel.deduplication_key for alert types that should be
    sent at most once per calendar day per business.

    Args:
        business_id:  UUID of the owning business.
        alert_type:   Alert type identifier (see AlertType enum).
        alert_date:   Calendar date of the alert (defaults to today UTC).

    Returns:
        str: Deterministic idempotency key string.
    """
    effective_date = alert_date or date.today()
    return IDEMPOTENCY_KEY_ALERT_SEND.format(
        business_id=str(business_id),
        alert_type=alert_type,
        date=effective_date.isoformat(),
    )


def make_custom_key(
    *parts: str,
    separator: str = "_",
) -> str:
    """
    Generate a custom idempotency key from arbitrary string parts.

    Use this for operation types not covered by the named constructors above.
    All parts are uppercased and joined with the separator to match the
    naming convention of the pre-defined key formats.

    Args:
        *parts:     String components of the key (e.g., "JOB", "WEEKLY", id).
        separator:  Character used to join parts (default: "_").

    Returns:
        str: Joined and uppercased key string.

    Example:
        key = make_custom_key("COMPETITOR", "SCAN", str(business_id), "2024-10-01")
        # → "COMPETITOR_SCAN_<uuid>_2024-10-01"
    """
    return separator.join(str(p).upper() for p in parts)


# ==============================================================================
# Key Hashing — for keys that exceed column length limits
# ==============================================================================

def hash_key(key: str) -> str:
    """
    Return a SHA-256 hex digest of the idempotency key.

    Use this when a generated key exceeds the 500-character column limit
    (e.g., keys containing long JSON blobs or compound identifiers).

    The hash is deterministic — the same input always produces the same
    32-character output — so idempotency semantics are preserved.

    Args:
        key: The full idempotency key string to hash.

    Returns:
        str: 64-character lowercase hex string (SHA-256 digest).

    Example:
        short_key = hash_key(very_long_key)
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def safe_key(key: str, max_length: int = 490) -> str:
    """
    Return the key as-is if within max_length, otherwise return its hash.

    This ensures all keys stored in the database stay within the column
    limit (500 chars in idempotency_key columns) without requiring callers
    to manually check key length.

    Args:
        key:        The idempotency key string.
        max_length: Maximum allowed character length (default: 490).

    Returns:
        str: Original key if within limit, SHA-256 hash otherwise.
    """
    if len(key) <= max_length:
        return key
    hashed = hash_key(key)
    logger.debug(
        "Idempotency key exceeded max length — using hash",
        extra={
            "service": ServiceName.API,
            "original_length": len(key),
            "hashed_key": hashed,
        },
    )
    return hashed


# ==============================================================================
# Week Identifier Helper
# ==============================================================================

def get_iso_week_identifier(dt: Optional[datetime] = None) -> str:
    """
    Return the ISO 8601 week identifier for a given datetime.

    Used by make_content_gen_key() to produce a stable weekly key
    regardless of the day within the week the job runs.

    Args:
        dt: Datetime to extract the week from (defaults to UTC now).

    Returns:
        str: ISO week string in format "YYYY-Www" (e.g., "2024-W42").

    Example:
        week = get_iso_week_identifier()
        # → "2024-W42"
    """
    from datetime import timezone
    effective_dt = dt or datetime.now(timezone.utc)
    iso_year, iso_week, _ = effective_dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


# ==============================================================================
# In-Memory Idempotency Store (for testing and within-process deduplication)
# ==============================================================================

class InMemoryIdempotencyStore:
    """
    A simple in-memory idempotency store for unit testing and lightweight
    within-process deduplication.

    In production, idempotency state is persisted in the database
    (ReviewModel.idempotency_key, PaymentModel.idempotency_key, etc.).
    This store is provided for contexts where a database session is not
    available or not appropriate (e.g., unit tests, one-shot scripts).

    Thread safety:
        This store is NOT thread-safe. Use only in single-threaded or
        async single-event-loop contexts. For concurrent access, use the
        database-backed approach exclusively.

    Usage:
        store = InMemoryIdempotencyStore()
        if store.has_been_executed("my_key"):
            return  # skip
        store.mark_executed("my_key")
        # ... perform operation
    """

    def __init__(self) -> None:
        self._executed: dict[str, datetime] = {}

    def has_been_executed(self, key: str) -> bool:
        """
        Check whether the given key has already been recorded as executed.

        Args:
            key: Idempotency key to check.

        Returns:
            bool: True if the key has been marked as executed.
        """
        return key in self._executed

    def mark_executed(self, key: str) -> None:
        """
        Record that the operation identified by key has been executed.

        Args:
            key: Idempotency key to record.
        """
        from datetime import timezone
        self._executed[key] = datetime.now(timezone.utc)
        logger.debug(
            "Idempotency key recorded",
            extra={
                "service": ServiceName.API,
                "key": key,
            },
        )

    def clear(self, key: Optional[str] = None) -> None:
        """
        Remove a specific key or clear the entire store.

        Args:
            key: Key to remove. If None, clears all recorded keys.
        """
        if key:
            self._executed.pop(key, None)
        else:
            self._executed.clear()

    def size(self) -> int:
        """Return the number of recorded keys in the store."""
        return len(self._executed)

    def __contains__(self, key: str) -> bool:
        """Support `key in store` syntax."""
        return self.has_been_executed(key)

    def __repr__(self) -> str:
        return f"<InMemoryIdempotencyStore keys={self.size()}>"


# ==============================================================================
# Guard Helper — Inline idempotency check for service-layer use
# ==============================================================================

def should_skip(
    existing_key: Optional[str],
    expected_key: str,
) -> bool:
    """
    Return True if a task should be skipped due to idempotency.

    Compares an existing stored key (e.g., ReviewModel.idempotency_key)
    against the expected key for the current operation. If they match,
    the operation has already been completed and should be skipped.

    Args:
        existing_key:  The key currently stored on the record (may be None).
        expected_key:  The key generated for the current operation attempt.

    Returns:
        bool: True if the operation should be skipped (already done).

    Example:
        key = make_review_reply_key(business_id, review_id)
        if should_skip(review.idempotency_key, key):
            logger.info("Skipping — reply already generated")
            return review.ai_reply
        # proceed with generation
    """
    if existing_key is None:
        return False
    matches = existing_key == expected_key
    if matches:
        logger.debug(
            "Idempotency check passed — operation already completed",
            extra={
                "service": ServiceName.API,
                "existing_key": existing_key,
            },
        )
    return matches


# ==============================================================================
# Missing key functions — added to satisfy all import contracts
# ==============================================================================

def make_job_lock_key(job_name: str, business_id: Optional[uuid.UUID] = None) -> str:
    """
    Generate the lock key for a scheduler job.

    Used by scheduler_manager.py and expiry_checker.py to prevent
    duplicate concurrent job execution.

    Args:
        job_name:    Job name string (see JobName enum in constants.py).
        business_id: Optional business scope for per-business job locks.

    Returns:
        str: Job lock key string.

    Example:
        key = make_job_lock_key("weekly_report", business_id)
        # → "JOB_LOCK_WEEKLY_REPORT_BUSINESS_<uuid>"
    """
    if business_id:
        return f"JOB_LOCK_{job_name.upper()}_BUSINESS_{str(business_id)}"
    return f"JOB_LOCK_{job_name.upper()}_GLOBAL"


def make_weekly_report_key(
    business_id: uuid.UUID,
    week_identifier: str,
) -> str:
    """
    Generate the idempotency key for a weekly report generation job.

    Args:
        business_id:      UUID of the owning business.
        week_identifier:  ISO week string (e.g. "2024-W42").

    Returns:
        str: Deterministic idempotency key.
    """
    from app.config.constants import IDEMPOTENCY_KEY_REPORT_GEN
    return IDEMPOTENCY_KEY_REPORT_GEN.format(
        business_id=str(business_id),
        report_type="weekly",
        date=week_identifier,
    )


def make_monthly_report_key(
    business_id: uuid.UUID,
    month_identifier: str,
) -> str:
    """
    Generate the idempotency key for a monthly report generation job.

    Args:
        business_id:       UUID of the owning business.
        month_identifier:  Month label string (e.g. "2024-01").

    Returns:
        str: Deterministic idempotency key.
    """
    from app.config.constants import IDEMPOTENCY_KEY_REPORT_GEN
    return IDEMPOTENCY_KEY_REPORT_GEN.format(
        business_id=str(business_id),
        report_type="monthly",
        date=month_identifier,
    )


def make_quarterly_report_key(
    business_id: uuid.UUID,
    quarter_identifier: str,
) -> str:
    """
    Generate the idempotency key for a quarterly report generation job.

    Args:
        business_id:         UUID of the owning business.
        quarter_identifier:  Quarter label string (e.g. "2024-Q1").

    Returns:
        str: Deterministic idempotency key.
    """
    from app.config.constants import IDEMPOTENCY_KEY_REPORT_GEN
    return IDEMPOTENCY_KEY_REPORT_GEN.format(
        business_id=str(business_id),
        report_type="quarterly",
        date=quarter_identifier,
    )


def make_weekly_content_key(
    business_id: uuid.UUID,
    week_identifier: str,
) -> str:
    """
    Generate the idempotency key for a weekly content generation job.

    Delegates to make_content_gen_key() for consistency.

    Args:
        business_id:      UUID of the owning business.
        week_identifier:  ISO week string (e.g. "2024-W42").

    Returns:
        str: Deterministic idempotency key.
    """
    return make_content_gen_key(
        business_id=business_id,
        week_identifier=week_identifier,
    )