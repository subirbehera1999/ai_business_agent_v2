# ==============================================================================
# File: app/jobs/review_jobs.py
# Purpose: Task functions for the review processing pipeline.
#
#          This module contains every task a scheduler job needs to call
#          when processing Google reviews for a business. All tasks accept a
#          JobContext (from job_manager.py) and return a JobResult.
#
#          Public entry points (called by review_monitor.py via job_manager):
#
#            process_reviews_for_business(ctx)
#              The full pipeline for a single business:
#                1. Rate limit check — block if daily review quota exceeded
#                2. Fetch pending reviews from database (NEW / PROCESSING)
#                3. Per-review loop:
#                   a. Validate review — skip spam / empty
#                   b. Rate limit AI reply — check quota before OpenAI call
#                   c. Sentiment analysis
#                   d. AI reply generation + persistence
#                   e. Usage tracking
#                   f. Alert dispatch for negative / positive reviews
#                4. Spike detection — fire REVIEW_SPIKE alert if volume high
#
#          Internal helpers (not called directly by schedulers):
#
#            _process_single_review(ctx, review, business)
#              Handles steps 3a–3f for one review. Returns a mini-result
#              dict: {processed, skipped, failed, error}.
#
#            _dispatch_review_alert(ctx, review, business, ai_reply_generated)
#              Builds ReviewAlertInput and calls alert_manager.dispatch_review_alert()
#              for negative and positive reviews.
#
#            _check_spike(ctx, business)
#              Counts reviews in the last SPIKE_WINDOW_HOURS window. If count
#              exceeds REVIEW_SPIKE_THRESHOLD, fires a REVIEW_SPIKE alert.
#
#          Processing rules:
#            - Only NEW or PROCESSING reviews with < 3 attempts are processed
#            - Rating-only reviews (no text) get AI replies using star rating
#            - Reviews flagged as spam are marked SKIPPED, never replied to
#            - Each review gets at most 3 AI reply attempts (review_model guard)
#            - Rate limit is checked once per batch (review quota) and once
#              per review (AI reply quota)
#            - Failure of one review never aborts processing of the others
#            - Spike detection runs after the batch loop regardless of results
#
#          Data safety:
#            - Idempotency enforced by ai_reply_service (key: business+review)
#            - review_repository.mark_processing() atomically claims a review
#            - review_repository.mark_failed() records exhausted reviews
#            - All DB state changes are flushed inside the service layer;
#              commit is controlled by run_job_for_business() in job_manager
# ==============================================================================

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.alerts.alert_manager import AlertManager, ReviewAlertInput
from app.config.constants import (
    REVIEW_SPIKE_THRESHOLD,
    AlertType,
    ReviewSentiment,
    ReviewStatus,
    ServiceName,
)
from app.database.models.business_model import BusinessModel
from app.database.models.review_model import ReviewModel
from app.jobs.job_manager import JobContext, JobResult, make_result
from app.utils.rate_limiter import check_ai_reply_limit, check_review_limit
from app.utils.usage_tracker import (
    track_review_pipeline_failure,
    track_review_pipeline_success,
)

logger = logging.getLogger(ServiceName.SCHEDULER)

# Rolling window for spike detection — count reviews in last N hours
_SPIKE_WINDOW_HOURS: int = 1

# Alert manager singleton — stateless, safe to share
_alert_manager = AlertManager()


# ==============================================================================
# Public entry point — called by review_monitor.py via job_manager
# ==============================================================================

async def process_reviews_for_business(ctx: JobContext) -> JobResult:
    """
    Run the full review processing pipeline for a single business.

    Called by review_monitor.py through run_job_for_business(). Processes
    all pending reviews (NEW / PROCESSING status, < 3 attempts) in a single
    batch, up to the daily rate limit for this business.

    Steps:
      1. Load business profile (needed for alert context and AI prompts)
      2. Check daily review processing quota — skip entire job if exhausted
      3. Fetch pending reviews (batch of 20 max, oldest first)
      4. Per-review: validate → rate-limit AI → analyse sentiment → reply
      5. After batch: spike detection against rolling 1-hour window

    Args:
        ctx: JobContext from job_manager with db session and all services.

    Returns:
        JobResult: Structured result with processed/skipped/failed counts.
        Never raises — all failures are captured in the result.
    """
    result = make_result("process_reviews", str(ctx.business_id))
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "process_reviews",
        "business_id": str(ctx.business_id),
    }

    # ------------------------------------------------------------------
    # Step 1: Load business profile
    # ------------------------------------------------------------------
    business = await ctx.business_repo.get_by_id(ctx.db, ctx.business_id)
    if not business:
        logger.warning(
            "Review job skipped — business not found",
            extra=log_extra,
        )
        result.skipped += 1
        return result

    if not business.is_active or business.is_deleted:
        logger.info(
            "Review job skipped — business inactive or deleted",
            extra={**log_extra, "business_name": business.business_name},
        )
        result.skipped += 1
        return result

    if not business.google_place_id:
        logger.info(
            "Review job skipped — no Google Place ID configured",
            extra={**log_extra, "business_name": business.business_name},
        )
        result.skipped += 1
        return result

    # ------------------------------------------------------------------
    # Step 2: Daily review processing quota check
    # ------------------------------------------------------------------
    quota = await check_review_limit(ctx.db, ctx.business_id)
    if not quota.allowed:
        logger.info(
            "Review job skipped — daily review limit reached",
            extra={
                **log_extra,
                "used": quota.current_count,
                "limit": quota.limit,
            },
        )
        result.skipped += 1
        return result

    # ------------------------------------------------------------------
    # Step 3: Fetch pending reviews
    # ------------------------------------------------------------------
    pending_reviews = await ctx.review_repo.get_pending_for_processing(
        ctx.db,
        business_id=ctx.business_id,
        limit=20,
    )

    if not pending_reviews:
        logger.info(
            "Review job: no pending reviews",
            extra={**log_extra, "business_name": business.business_name},
        )
        # Still run spike check — new reviews may have arrived via polling
        await _check_spike(ctx, business)
        return result

    logger.info(
        "Review job: processing batch",
        extra={
            **log_extra,
            "pending_count": len(pending_reviews),
            "business_name": business.business_name,
        },
    )

    # ------------------------------------------------------------------
    # Step 4: Per-review processing loop
    # ------------------------------------------------------------------
    for review in pending_reviews:
        mini = await _process_single_review(ctx, review, business)
        result.processed += mini["processed"]
        result.skipped += mini["skipped"]
        result.failed += mini["failed"]
        if mini.get("error"):
            result.errors.append(mini["error"])

    # ------------------------------------------------------------------
    # Step 5: Spike detection
    # ------------------------------------------------------------------
    await _check_spike(ctx, business)

    logger.info(
        "Review job complete",
        extra={
            **log_extra,
            "processed": result.processed,
            "skipped": result.skipped,
            "failed": result.failed,
        },
    )

    return result


# ==============================================================================
# Single review processor
# ==============================================================================

async def _process_single_review(
    ctx: JobContext,
    review: ReviewModel,
    business: BusinessModel,
) -> dict:
    """
    Process one review through the full sentiment + AI reply + alert pipeline.

    This function handles the entire lifecycle of a single review:
      a. Mark as PROCESSING (atomic claim)
      b. Validate — skip spam or invalid records
      c. Check AI reply rate limit for this business
      d. Run sentiment analysis
      e. Generate and persist AI reply
      f. Track usage
      g. Dispatch WhatsApp alert for negative/positive reviews

    Args:
        ctx:      JobContext with db, services, and repositories.
        review:   ReviewModel instance fetched from the database.
        business: BusinessModel of the owning business.

    Returns:
        dict with keys: processed (int), skipped (int), failed (int),
        error (Optional[str]). Never raises.
    """
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "process_reviews",
        "business_id": str(ctx.business_id),
        "review_id": str(review.id),
        "rating": review.rating,
    }

    # ------------------------------------------------------------------
    # a. Claim the review atomically
    # ------------------------------------------------------------------
    try:
        await ctx.review_repo.mark_processing(ctx.db, review.id)
    except Exception as exc:
        logger.error(
            "Failed to claim review for processing",
            extra={**log_extra, "error": str(exc)},
        )
        return {"processed": 0, "skipped": 0, "failed": 1, "error": str(exc)}

    # ------------------------------------------------------------------
    # b. Validate — skip spam or invalid
    # ------------------------------------------------------------------
    if review.is_spam or not review.is_valid:
        logger.info(
            "Review skipped — spam or invalid",
            extra=log_extra,
        )
        await ctx.review_repo.mark_skipped(ctx.db, review.id)
        return {"processed": 0, "skipped": 1, "failed": 0, "error": None}

    # ------------------------------------------------------------------
    # c. AI reply rate limit check
    # ------------------------------------------------------------------
    ai_quota = await check_ai_reply_limit(ctx.db, ctx.business_id)
    if not ai_quota.allowed:
        logger.info(
            "Review processing skipped — AI reply daily limit reached",
            extra={
                **log_extra,
                "used": ai_quota.current_count,
                "limit": ai_quota.limit,
            },
        )
        # Do not mark as SKIPPED — push back to NEW so it retries tomorrow
        await ctx.review_repo.mark_failed(
            ctx.db,
            review.id,
            error_message="AI reply limit reached — will retry next cycle",
        )
        return {"processed": 0, "skipped": 1, "failed": 0, "error": None}

    # ------------------------------------------------------------------
    # d. Sentiment analysis
    # ------------------------------------------------------------------
    sentiment_result = None
    try:
        sentiment_result = await ctx.sentiment_service.analyze(
            review_text=review.text_for_ai,
            star_rating=review.rating,
            business_id=str(ctx.business_id),
            business_type=business.business_type or "Local Business",
            review_id=str(review.id),
        )

        # Persist sentiment to the review record
        await ctx.review_repo.update_sentiment(
            ctx.db,
            review_id=review.id,
            sentiment=sentiment_result.label,
            sentiment_score=sentiment_result.score,
        )

        logger.debug(
            "Sentiment analysed",
            extra={
                **log_extra,
                "sentiment": sentiment_result.label,
                "score": sentiment_result.score,
            },
        )

    except Exception as exc:
        # Non-fatal — ai_reply_service will run its own internal sentiment
        # if sentiment_result is None
        logger.warning(
            "Sentiment analysis failed — proceeding without pre-analysis",
            extra={**log_extra, "error": str(exc)},
        )

    # ------------------------------------------------------------------
    # e. AI reply generation
    # ------------------------------------------------------------------
    reply_result = await ctx.ai_reply_service.generate_reply(
        db=ctx.db,
        review_id=str(review.id),
        business_id=str(ctx.business_id),
        review_text=review.text_for_ai,
        star_rating=review.rating,
        reviewer_name=review.reviewer_name or "A Google User",
        business_name=business.business_name,
        business_type=business.business_type or "Local Business",
        sentiment_result=sentiment_result,
    )

    # ------------------------------------------------------------------
    # f. Usage tracking
    # ------------------------------------------------------------------
    try:
        if reply_result.success and not reply_result.skipped:
            await track_review_pipeline_success(ctx.db, ctx.business_id)
        else:
            await track_review_pipeline_failure(ctx.db, ctx.business_id)
    except Exception as exc:
        # Usage tracking is best-effort — never fail the job for this
        logger.warning(
            "Usage tracking failed for review",
            extra={**log_extra, "error": str(exc)},
        )

    # ------------------------------------------------------------------
    # Evaluate reply result
    # ------------------------------------------------------------------
    if reply_result.skipped:
        logger.info(
            "AI reply skipped",
            extra={**log_extra, "reason": reply_result.skip_reason},
        )
        return {"processed": 0, "skipped": 1, "failed": 0, "error": None}

    if not reply_result.success:
        logger.warning(
            "AI reply generation failed",
            extra={**log_extra, "error": reply_result.error},
        )

        # Mark as failed if this was the last allowed attempt
        if review.processing_attempts >= 3:
            await ctx.review_repo.mark_failed(
                ctx.db,
                review.id,
                error_message=reply_result.error or "AI reply failed",
            )

        return {
            "processed": 0,
            "skipped": 0,
            "failed": 1,
            "error": reply_result.error,
        }

    # ------------------------------------------------------------------
    # g. WhatsApp alert dispatch for significant reviews
    # ------------------------------------------------------------------
    await _dispatch_review_alert(
        ctx=ctx,
        review=review,
        business=business,
        ai_reply_generated=True,
    )

    logger.info(
        "Review processed successfully",
        extra={
            **log_extra,
            "business_name": business.business_name,
            "sentiment": sentiment_result.label if sentiment_result else "unknown",
        },
    )

    return {"processed": 1, "skipped": 0, "failed": 0, "error": None}


# ==============================================================================
# Alert dispatcher — negative and positive reviews
# ==============================================================================

async def _dispatch_review_alert(
    ctx: JobContext,
    review: ReviewModel,
    business: BusinessModel,
    ai_reply_generated: bool,
) -> None:
    """
    Dispatch a WhatsApp alert for a new negative or positive review.

    Neutral reviews do not trigger alerts — only negative (1–2 stars) and
    positive (4–5 stars) reviews notify the business owner via WhatsApp.

    Alert deduplication is handled inside AlertManager using a key that
    includes the review_id — the same review never fires twice.

    Args:
        ctx:                JobContext with db and alert manager access.
        review:             The processed ReviewModel.
        business:           BusinessModel of the owning business.
        ai_reply_generated: Whether an AI reply was successfully generated.
    """
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "process_reviews",
        "business_id": str(ctx.business_id),
        "review_id": str(review.id),
        "rating": review.rating,
    }

    # Determine alert type from rating
    if review.rating <= 2:
        alert_type = AlertType.NEGATIVE_REVIEW
    elif review.rating >= 4:
        alert_type = AlertType.POSITIVE_REVIEW
    else:
        # Neutral — no alert
        return

    # Determine sentiment label for context
    sentiment_label = (
        review.sentiment or
        (ReviewSentiment.NEGATIVE if review.rating <= 2 else ReviewSentiment.POSITIVE)
    )

    # Build excerpt — truncated to 150 chars for WhatsApp readability
    excerpt = ""
    if review.review_text:
        excerpt = review.review_text[:150]
        if len(review.review_text) > 150:
            excerpt += "…"

    alert_input = ReviewAlertInput(
        business_id=str(ctx.business_id),
        business_name=business.business_name,
        alert_type=alert_type,
        review_id=str(review.id),
        reviewer_name=review.reviewer_name or "A Google User",
        rating=review.rating,
        review_excerpt=excerpt,
        sentiment=sentiment_label,
        ai_reply_generated=ai_reply_generated,
    )

    try:
        dispatch_result = await _alert_manager.dispatch_review_alert(
            db=ctx.db,
            input=alert_input,
            whatsapp_number=business.owner_whatsapp_number,
        )

        if dispatch_result.sent:
            logger.info(
                "Review alert dispatched",
                extra={
                    **log_extra,
                    "alert_type": alert_type,
                    "message_id": dispatch_result.message_id,
                },
            )
        elif dispatch_result.skipped:
            logger.debug(
                "Review alert skipped — dedup",
                extra={**log_extra, "alert_type": alert_type},
            )
        else:
            logger.warning(
                "Review alert delivery failed",
                extra={
                    **log_extra,
                    "alert_type": alert_type,
                    "error": dispatch_result.error,
                },
            )

    except Exception as exc:
        # Alert failure is non-fatal — review was already processed
        logger.error(
            "Review alert dispatch raised unexpected exception",
            extra={**log_extra, "alert_type": alert_type, "error": str(exc)},
        )


# ==============================================================================
# Spike detection
# ==============================================================================

async def _check_spike(
    ctx: JobContext,
    business: BusinessModel,
) -> None:
    """
    Check for a review volume spike in the last 1 hour.

    If the number of new reviews in the rolling window meets or exceeds
    REVIEW_SPIKE_THRESHOLD (5), dispatch a REVIEW_SPIKE alert to the
    business owner.

    Called once per processing cycle, after the per-review loop completes.
    Uses a dedup key scoped to today's date — the same business cannot
    receive more than one spike alert per calendar day.

    Args:
        ctx:      JobContext with db and business_id.
        business: BusinessModel of the business being checked.
    """
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "process_reviews",
        "business_id": str(ctx.business_id),
    }

    try:
        since = datetime.now(timezone.utc) - timedelta(hours=_SPIKE_WINDOW_HOURS)
        review_count = await ctx.review_repo.count_since(
            ctx.db,
            business_id=ctx.business_id,
            since=since,
        )

        if review_count < REVIEW_SPIKE_THRESHOLD:
            return

        logger.info(
            "Review spike detected",
            extra={
                **log_extra,
                "review_count": review_count,
                "window_hours": _SPIKE_WINDOW_HOURS,
                "threshold": REVIEW_SPIKE_THRESHOLD,
            },
        )

        # Build a spike alert — review_id is empty for spike-level alerts
        spike_input = ReviewAlertInput(
            business_id=str(ctx.business_id),
            business_name=business.business_name,
            alert_type=AlertType.REVIEW_SPIKE,
            review_id="spike",  # Not a specific review — used only in dedup key
            reviewer_name="",
            rating=0,
            review_excerpt="",
            sentiment=ReviewSentiment.NEUTRAL,
            ai_reply_generated=False,
            spike_count=review_count,
        )

        await _alert_manager.dispatch_review_alert(
            db=ctx.db,
            input=spike_input,
            whatsapp_number=business.owner_whatsapp_number,
        )

    except Exception as exc:
        # Spike check is non-fatal — never abort the job for this
        logger.error(
            "Spike detection failed",
            extra={**log_extra, "error": str(exc)},
        )