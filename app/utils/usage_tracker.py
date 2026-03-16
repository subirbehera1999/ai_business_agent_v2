# ==============================================================================
# File: app/utils/usage_tracker.py
# Purpose: Records usage counters immediately after each billable operation
#          completes. This is the write-side complement to rate_limiter.py
#          (the read-side).
#
#          Rule from DATA_SAFETY_AND_RUNTIME_GUARDRAILS.txt:
#            "All AI and analytics operations must update usage counters.
#             Tracking logic must run immediately after task completion."
#
#          Tracked metrics:
#            - reviews_processed
#            - ai_replies_generated  (+ ai_replies_failed on failure)
#            - competitor_scans
#            - reports_generated
#            - content_pieces_generated
#            - whatsapp_messages_sent / failed
#            - alerts_triggered
#            - google_api_errors
#            - openai_api_errors
#            - whatsapp_api_errors
#
#          Design principles:
#            1. Tracking is non-blocking — failures are logged but never
#               propagated. A counter failure must never fail the operation.
#            2. All increments delegate to UsageRepository which uses
#               atomic server-side SQL arithmetic (no race conditions).
#            3. Convenience functions are named after the operation that
#               triggers them, not the metric column, for readability.
#            4. A single track_operation() dispatcher handles arbitrary
#               metric names for programmatic use.
# ==============================================================================

import logging
import uuid
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName, UsageMetric
from app.repositories.usage_repository import UsageRepository

logger = logging.getLogger(ServiceName.API)

# Stateless repository singleton — safe to reuse across calls
_usage_repo = UsageRepository()


# ==============================================================================
# Core Dispatcher
# ==============================================================================

async def track_operation(
    db: AsyncSession,
    business_id: uuid.UUID,
    metric: str,
    amount: int = 1,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record a usage increment for any metric after an operation completes.

    This is the generic entry point. All named helpers below delegate here.
    Failures are caught and logged — tracking must never fail the caller.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business to record usage for.
        metric:       Column name of the counter to increment.
                      Must match a valid column on UsageModel.
        amount:       Number of units to add (default: 1).
        usage_date:   Date override for the usage record (defaults to today).

    Returns:
        None. Never raises — tracking failures are logged only.
    """
    try:
        await _usage_repo._increment(
            db,
            business_id,
            column_name=metric,
            amount=amount,
            usage_date=usage_date,
        )
        logger.debug(
            "Usage tracked",
            extra={
                "service": ServiceName.API,
                "business_id": str(business_id),
                "metric": metric,
                "amount": amount,
            },
        )
    except Exception as exc:
        # Non-critical — log and continue. Never propagate tracking failures.
        logger.error(
            "Failed to track usage — operation will not be retried",
            extra={
                "service": ServiceName.API,
                "business_id": str(business_id),
                "metric": metric,
                "amount": amount,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


# ==============================================================================
# Named Trackers — One per operation type
# ==============================================================================

async def track_review_processed(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one review has been entered into the processing pipeline.

    Call immediately after a review passes validation and enters the
    sentiment + AI reply pipeline. Do not call on skipped or spam reviews.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        UsageMetric.REVIEWS_PROCESSED,
        usage_date=usage_date,
    )


async def track_ai_reply_generated(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one AI reply was successfully generated.

    Call immediately after ai_reply_service.py returns a reply string.
    This increments the plan-enforced daily AI reply counter.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        UsageMetric.AI_REPLIES_GENERATED,
        usage_date=usage_date,
    )


async def track_ai_reply_failed(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one AI reply generation attempt failed.

    Call when ai_reply_service.py raises after all retries are exhausted.
    Failed attempts do not count against the plan limit but are tracked
    for OpenAI cost exposure monitoring.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "ai_replies_failed",
        usage_date=usage_date,
    )


async def track_competitor_scan(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one competitor profile scan was performed.

    Call immediately after competitor_service.py completes a scan for
    a single competitor. This increments the plan-enforced daily scan counter.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        UsageMetric.COMPETITOR_SCANS,
        usage_date=usage_date,
    )


async def track_report_generated(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one report was successfully generated and dispatched.

    Call after reports_service.py has generated and sent the report.
    Applies to weekly, monthly, and quarterly report types.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        UsageMetric.REPORTS_GENERATED,
        usage_date=usage_date,
    )


async def track_content_generated(
    db: AsyncSession,
    business_id: uuid.UUID,
    count: int = 1,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one or more social media content pieces were generated.

    Call after content_generation_service.py returns generated posts.
    The count argument supports batch content generation where multiple
    pieces are produced in a single call.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        count:        Number of content pieces generated (default: 1).
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "content_pieces_generated",
        amount=count,
        usage_date=usage_date,
    )


async def track_whatsapp_sent(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one WhatsApp message was successfully delivered.

    Call after whatsapp_service.py confirms delivery from the API response.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "whatsapp_messages_sent",
        usage_date=usage_date,
    )


async def track_whatsapp_failed(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one WhatsApp message delivery failed after all retries.

    Call when whatsapp_client.py exhausts retry attempts without success.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "whatsapp_messages_failed",
        usage_date=usage_date,
    )


async def track_alert_triggered(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that one business event alert was triggered and dispatched.

    Call after alert_manager.py successfully creates and queues an alert.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "alerts_triggered",
        usage_date=usage_date,
    )


async def track_google_api_error(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that a Google API call failed after all retries.

    Call in google_reviews_client.py and google_sheets_client.py when
    the retry policy is exhausted and the call cannot be completed.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "google_api_errors",
        usage_date=usage_date,
    )


async def track_openai_api_error(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that an OpenAI API call failed after all retries.

    Call in ai_reply_service.py when the retry policy is exhausted
    and the AI generation cannot be completed.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "openai_api_errors",
        usage_date=usage_date,
    )


async def track_whatsapp_api_error(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record that a WhatsApp API call failed after all retries.

    Call in whatsapp_client.py when the retry policy is exhausted
    and the message cannot be delivered.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_operation(
        db, business_id,
        "whatsapp_api_errors",
        usage_date=usage_date,
    )


# ==============================================================================
# Compound Trackers — Multiple counters in one call
# ==============================================================================

async def track_review_pipeline_success(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record a complete successful review pipeline execution.

    Increments both reviews_processed and ai_replies_generated in a
    single logical call. Use when a review passes all stages
    (validation → sentiment → AI reply) without error.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_review_processed(db, business_id, usage_date)
    await track_ai_reply_generated(db, business_id, usage_date)


async def track_review_pipeline_failure(
    db: AsyncSession,
    business_id: uuid.UUID,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record a failed review pipeline execution.

    Increments reviews_processed (review entered the pipeline) and
    ai_replies_failed (AI generation could not complete). Use when
    a review was validated and processed but the AI reply failed.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        usage_date:   Date override (defaults to today UTC).
    """
    await track_review_processed(db, business_id, usage_date)
    await track_ai_reply_failed(db, business_id, usage_date)


async def track_whatsapp_outcome(
    db: AsyncSession,
    business_id: uuid.UUID,
    success: bool,
    usage_date: Optional[date] = None,
) -> None:
    """
    Record a WhatsApp send outcome — either success or failure.

    Convenience function for callers that handle both outcomes in a
    single code path and want to branch on a boolean.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        success:      True if the message was delivered, False if it failed.
        usage_date:   Date override (defaults to today UTC).
    """
    if success:
        await track_whatsapp_sent(db, business_id, usage_date)
    else:
        await track_whatsapp_failed(db, business_id, usage_date)


# ==============================================================================
# Batch Tracker — Record usage for multiple businesses at once
# ==============================================================================

async def track_batch(
    db: AsyncSession,
    entries: list[dict],
) -> None:
    """
    Record usage increments for multiple business/metric pairs in one call.

    Each entry must be a dict with keys:
        business_id (uuid.UUID): Business to update.
        metric      (str):       Counter column name.
        amount      (int):       Units to increment (default 1 if absent).
        usage_date  (date|None): Date override (default today if absent).

    Failures for individual entries are logged and skipped — the batch
    continues regardless of partial failures.

    Args:
        db:      Active async database session.
        entries: List of tracking instruction dicts.

    Example:
        await track_batch(db, [
            {"business_id": b1_id, "metric": UsageMetric.REVIEWS_PROCESSED},
            {"business_id": b2_id, "metric": UsageMetric.AI_REPLIES_GENERATED},
        ])
    """
    for entry in entries:
        business_id = entry.get("business_id")
        metric = entry.get("metric")
        amount = entry.get("amount", 1)
        usage_date = entry.get("usage_date")

        if not business_id or not metric:
            logger.warning(
                "Skipping invalid batch tracking entry — missing business_id or metric",
                extra={
                    "service": ServiceName.API,
                    "entry": str(entry),
                },
            )
            continue

        await track_operation(db, business_id, metric, amount, usage_date)


# ==============================================================================
# UsageTracker — Class wrapper used by schedulers and job modules
# ==============================================================================

class UsageTracker:
    """
    Thin class wrapper around the module-level tracking functions.

    Used by scheduler jobs that prefer dependency-injected objects
    over calling module-level functions directly.

    Usage:
        tracker = UsageTracker()
        await tracker.track_review_processed(db=db, business_id=business_id)
    """

    async def track_review_processed(self, db, business_id) -> None:
        await track_review_processed(db=db, business_id=business_id)

    async def track_ai_reply_generated(self, db, business_id) -> None:
        await track_ai_reply_generated(db=db, business_id=business_id)

    async def track_ai_reply_failed(self, db, business_id) -> None:
        await track_ai_reply_failed(db=db, business_id=business_id)

    async def track_competitor_scan(self, db, business_id) -> None:
        await track_competitor_scan(db=db, business_id=business_id)

    async def track_report_generated(self, db, business_id) -> None:
        await track_report_generated(db=db, business_id=business_id)

    async def track_content_generated(self, db, business_id) -> None:
        await track_content_generated(db=db, business_id=business_id)