# ==============================================================================
# File: app/jobs/report_jobs.py
# Purpose: Task functions for weekly, monthly, and quarterly report generation.
#
#          Each task is a standalone coroutine that accepts a JobContext
#          (from job_manager.py) and returns a JobResult. All three are
#          triggered by their respective scheduler jobs:
#
#            weekly_report_job.py   → generate_weekly_report(ctx)
#            monthly_report_job.py  → generate_monthly_report(ctx)
#            quarterly_report_job.py → generate_quarterly_report(ctx)
#
#          Each task follows the same structure:
#
#            Step 1: Load business profile — skip if inactive / deleted
#            Step 2: Check daily report rate limit
#            Step 3: Gather data from repositories
#                    - Review counts and sentiment breakdown for the period
#                    - Average rating for period and prior period (trend)
#                    - Usage totals for the period
#            Step 4: Call reports_service to assemble the report messages
#            Step 5: Deliver via whatsapp_service.send_report()
#            Step 6: Track usage — increment reports_generated counter
#
#          Data gathering:
#            All data is pulled from the local database — reviews, usage
#            counters, sentiment breakdowns. No external API calls are made
#            in this file. Reports service may call OpenAI for AI insights
#            (monthly/quarterly), but that is handled inside the service.
#
#          Period windows:
#            Weekly:    Last completed Mon–Sun (via get_weekly_period())
#            Monthly:   Last completed calendar month (get_monthly_period())
#            Quarterly: Last completed calendar quarter (get_quarterly_period())
#            All period helpers return (start: date, end: date).
#
#          Failure handling:
#            - Report assembly failure (ReportError) → log, mark failed, return
#            - WhatsApp delivery failure → log warning, still mark as processed
#              (report was built — delivery failure is a notification problem,
#               not a data problem)
#            - Any unhandled exception → captured by failsafe in job_manager
#
#          Multi-tenancy:
#            All repository queries are scoped by business_id.
#            No cross-business data access occurs anywhere in this file.
# ==============================================================================

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from app.config.constants import ReportType, ServiceName
from app.database.models.business_model import BusinessModel
from app.jobs.job_manager import JobContext, JobResult, make_result
from app.services.reports_service import ReportError, ReportResult
from app.utils.rate_limiter import check_report_limit
from app.utils.time_utils import (
    get_monthly_period,
    get_quarterly_period,
    get_weekly_period,
)
from app.utils.usage_tracker import track_report_generated

logger = logging.getLogger(ServiceName.SCHEDULER)


# ==============================================================================
# Weekly report task
# ==============================================================================

async def generate_weekly_report(ctx: JobContext) -> JobResult:
    """
    Generate and deliver the weekly WhatsApp performance report.

    Covers the most recently completed Mon–Sun calendar week.
    Weekly reports are rule-based (no OpenAI) — fast and cost-free.

    Data gathered: review counts, sentiment breakdown, avg rating vs
    prior week, usage totals for the period.

    Called by weekly_report_job.py via run_job_for_business().

    Args:
        ctx: JobContext from job_manager with db session and all services.

    Returns:
        JobResult. Never raises.
    """
    result = make_result("weekly_report", str(ctx.business_id))
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "weekly_report",
        "business_id": str(ctx.business_id),
    }

    # ------------------------------------------------------------------
    # Step 1: Load and validate business
    # ------------------------------------------------------------------
    business = await _load_business(ctx, log_extra)
    if business is None:
        result.skipped += 1
        return result

    # ------------------------------------------------------------------
    # Step 2: Rate limit check
    # ------------------------------------------------------------------
    if not await _check_report_limit(ctx, log_extra, result):
        return result

    # ------------------------------------------------------------------
    # Step 3: Gather data for the period
    # ------------------------------------------------------------------
    period_start, period_end = get_weekly_period()
    prior_start = period_start - timedelta(days=7)
    prior_end = period_end - timedelta(days=7)

    data = await _gather_review_data(ctx, period_start, period_end, log_extra)
    prior_avg = await _get_prior_avg_rating(ctx, prior_start, prior_end, log_extra)

    # ------------------------------------------------------------------
    # Step 4: Assemble report
    # ------------------------------------------------------------------
    report = await ctx.reports_service.build_weekly_report(
        business_id=str(ctx.business_id),
        business_name=business.business_name,
        business_type=business.business_type or "Local Business",
        period_start=period_start,
        period_end=period_end,
        total_reviews=data["total"],
        positive_count=data["positive"],
        negative_count=data["negative"],
        neutral_count=data["neutral"],
        avg_rating=data["avg_rating"] or 0.0,
        previous_avg_rating=prior_avg,
    )

    return await _deliver_and_track(
        ctx=ctx,
        report=report,
        business=business,
        report_type=ReportType.WEEKLY,
        result=result,
        log_extra=log_extra,
    )


# ==============================================================================
# Monthly report task
# ==============================================================================

async def generate_monthly_report(ctx: JobContext) -> JobResult:
    """
    Generate and deliver the monthly WhatsApp performance report.

    Covers the most recently completed calendar month.
    Monthly reports include an AI-generated insights section (OpenAI).
    Falls back to rule-based tips if OpenAI is unavailable.

    Data gathered: review counts, sentiment breakdown, avg rating vs
    prior month, top positive/negative themes derived from sentiment counts.

    Called by monthly_report_job.py via run_job_for_business().

    Args:
        ctx: JobContext from job_manager with db session and all services.

    Returns:
        JobResult. Never raises.
    """
    result = make_result("monthly_report", str(ctx.business_id))
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "monthly_report",
        "business_id": str(ctx.business_id),
    }

    # ------------------------------------------------------------------
    # Step 1: Load and validate business
    # ------------------------------------------------------------------
    business = await _load_business(ctx, log_extra)
    if business is None:
        result.skipped += 1
        return result

    # ------------------------------------------------------------------
    # Step 2: Rate limit check
    # ------------------------------------------------------------------
    if not await _check_report_limit(ctx, log_extra, result):
        return result

    # ------------------------------------------------------------------
    # Step 3: Gather data for the period
    # ------------------------------------------------------------------
    period_start, period_end = get_monthly_period()

    # Prior period = same length ending the day before period_start
    prior_end = period_start - timedelta(days=1)
    prior_start = prior_end.replace(day=1)

    data = await _gather_review_data(ctx, period_start, period_end, log_extra)
    prior_avg = await _get_prior_avg_rating(ctx, prior_start, prior_end, log_extra)

    # Derive top themes from sentiment label (simple heuristic — service
    # uses them as prompt context, not as the full insight output)
    positive_themes = _derive_positive_themes(data["avg_rating"], data["positive"])
    negative_themes = _derive_negative_themes(data["negative"], data["total"])

    # ------------------------------------------------------------------
    # Step 4: Assemble report
    # ------------------------------------------------------------------
    report = await ctx.reports_service.build_monthly_report(
        business_id=str(ctx.business_id),
        business_name=business.business_name,
        business_type=business.business_type or "Local Business",
        period_start=period_start,
        period_end=period_end,
        total_reviews=data["total"],
        positive_count=data["positive"],
        negative_count=data["negative"],
        neutral_count=data["neutral"],
        avg_rating=data["avg_rating"] or 0.0,
        previous_avg_rating=prior_avg,
        top_positive_themes=positive_themes,
        top_negative_themes=negative_themes,
    )

    return await _deliver_and_track(
        ctx=ctx,
        report=report,
        business=business,
        report_type=ReportType.MONTHLY,
        result=result,
        log_extra=log_extra,
    )


# ==============================================================================
# Quarterly report task
# ==============================================================================

async def generate_quarterly_report(ctx: JobContext) -> JobResult:
    """
    Generate and deliver the quarterly WhatsApp strategic report.

    Covers the most recently completed calendar quarter (Q1–Q4).
    Quarterly reports include a full AI-generated 90-day strategy section.
    Falls back to rule-based analysis if OpenAI is unavailable.

    Data gathered: review counts, sentiment breakdown, avg rating vs
    prior quarter, top themes for strategic context.

    Called by quarterly_report_job.py via run_job_for_business().

    Args:
        ctx: JobContext from job_manager with db session and all services.

    Returns:
        JobResult. Never raises.
    """
    result = make_result("quarterly_report", str(ctx.business_id))
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": "quarterly_report",
        "business_id": str(ctx.business_id),
    }

    # ------------------------------------------------------------------
    # Step 1: Load and validate business
    # ------------------------------------------------------------------
    business = await _load_business(ctx, log_extra)
    if business is None:
        result.skipped += 1
        return result

    # ------------------------------------------------------------------
    # Step 2: Rate limit check
    # ------------------------------------------------------------------
    if not await _check_report_limit(ctx, log_extra, result):
        return result

    # ------------------------------------------------------------------
    # Step 3: Gather data for the period
    # ------------------------------------------------------------------
    period_start, period_end = get_quarterly_period()
    quarter_label = _quarter_label(period_start)

    # Prior quarter — go back one full quarter from period_start
    prior_end = period_start - timedelta(days=1)
    prior_start, _ = get_quarterly_period(reference=prior_end)

    data = await _gather_review_data(ctx, period_start, period_end, log_extra)
    prior_avg = await _get_prior_avg_rating(ctx, prior_start, prior_end, log_extra)

    positive_themes = _derive_positive_themes(data["avg_rating"], data["positive"])
    negative_themes = _derive_negative_themes(data["negative"], data["total"])

    # ------------------------------------------------------------------
    # Step 4: Assemble report
    # ------------------------------------------------------------------
    report = await ctx.reports_service.build_quarterly_report(
        business_id=str(ctx.business_id),
        business_name=business.business_name,
        business_type=business.business_type or "Local Business",
        period_start=period_start,
        period_end=period_end,
        quarter_label=quarter_label,
        total_reviews=data["total"],
        avg_rating=data["avg_rating"] or 0.0,
        positive_count=data["positive"],
        negative_count=data["negative"],
        neutral_count=data["neutral"],
        previous_quarter_avg_rating=prior_avg,
        top_positive_themes=positive_themes,
        top_negative_themes=negative_themes,
    )

    return await _deliver_and_track(
        ctx=ctx,
        report=report,
        business=business,
        report_type=ReportType.QUARTERLY,
        result=result,
        log_extra=log_extra,
    )


# ==============================================================================
# Shared helpers
# ==============================================================================

async def _load_business(
    ctx: JobContext,
    log_extra: dict,
) -> Optional[BusinessModel]:
    """
    Load and validate the business profile.

    Returns None (with a log line) if the business is not found,
    inactive, deleted, or has no WhatsApp number configured.

    Args:
        ctx:       JobContext with db and business_repo.
        log_extra: Structured log context dict.

    Returns:
        BusinessModel if valid and active, else None.
    """
    business = await ctx.business_repo.get_by_id(ctx.db, ctx.business_id)

    if not business:
        logger.warning(
            "Report job skipped — business not found",
            extra=log_extra,
        )
        return None

    if not business.is_active or business.is_deleted:
        logger.info(
            "Report job skipped — business inactive or deleted",
            extra={**log_extra, "business_name": business.business_name},
        )
        return None

    if not business.owner_whatsapp_number:
        logger.info(
            "Report job skipped — no WhatsApp number configured",
            extra={**log_extra, "business_name": business.business_name},
        )
        return None

    return business


async def _check_report_limit(
    ctx: JobContext,
    log_extra: dict,
    result: JobResult,
) -> bool:
    """
    Check the daily report generation quota for this business.

    Modifies result.skipped if blocked.

    Returns:
        True if allowed to proceed, False if quota exceeded.
    """
    quota = await check_report_limit(ctx.db, ctx.business_id)
    if not quota.allowed:
        logger.info(
            "Report job skipped — daily report limit reached",
            extra={
                **log_extra,
                "used": quota.current_count,
                "limit": quota.limit,
            },
        )
        result.skipped += 1
        return False
    return True


async def _gather_review_data(
    ctx: JobContext,
    period_start: date,
    period_end: date,
    log_extra: dict,
) -> dict:
    """
    Gather review counts, sentiment breakdown, and average rating
    for a date period from the review repository.

    Uses period_end + 1 day as the upper bound for count_since() which
    takes a datetime lower bound — we convert period boundaries to UTC
    datetimes covering the full calendar day range.

    Args:
        ctx:          JobContext with db and review_repo.
        period_start: Inclusive start date.
        period_end:   Inclusive end date.
        log_extra:    Structured log context dict.

    Returns:
        dict with keys: total (int), positive (int), negative (int),
        neutral (int), avg_rating (Optional[float]).
        All values default to 0 / None on error.
    """
    result = {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "avg_rating": None,
    }

    since_dt = datetime.combine(period_start, datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    until_dt = datetime.combine(
        period_end + timedelta(days=1), datetime.min.time()
    ).replace(tzinfo=timezone.utc)

    try:
        # Total review count for the period
        result["total"] = await ctx.review_repo.count_since(
            ctx.db,
            business_id=ctx.business_id,
            since=since_dt,
        )
    except Exception as exc:
        logger.warning(
            "Failed to count reviews for report period",
            extra={**log_extra, "error": str(exc)},
        )

    try:
        # Sentiment breakdown
        sentiment_counts = await ctx.review_repo.count_by_sentiment_since(
            ctx.db,
            business_id=ctx.business_id,
            since=since_dt,
        )
        result["positive"] = sentiment_counts.get("positive", 0)
        result["negative"] = sentiment_counts.get("negative", 0)
        result["neutral"] = sentiment_counts.get("neutral", 0)
    except Exception as exc:
        logger.warning(
            "Failed to get sentiment breakdown for report period",
            extra={**log_extra, "error": str(exc)},
        )

    try:
        # Average rating for the period
        result["avg_rating"] = await ctx.review_repo.get_average_rating_since(
            ctx.db,
            business_id=ctx.business_id,
            since=since_dt,
        )
    except Exception as exc:
        logger.warning(
            "Failed to get average rating for report period",
            extra={**log_extra, "error": str(exc)},
        )

    return result


async def _get_prior_avg_rating(
    ctx: JobContext,
    prior_start: date,
    prior_end: date,
    log_extra: dict,
) -> Optional[float]:
    """
    Get the average rating for the prior period (for trend comparison).

    Args:
        ctx:        JobContext with db and review_repo.
        prior_start: Start of the prior period.
        prior_end:   End of the prior period.
        log_extra:   Structured log context dict.

    Returns:
        float average rating or None if no data / error.
    """
    try:
        since_dt = datetime.combine(prior_start, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        return await ctx.review_repo.get_average_rating_since(
            ctx.db,
            business_id=ctx.business_id,
            since=since_dt,
        )
    except Exception as exc:
        logger.debug(
            "Failed to get prior period avg rating — omitting trend",
            extra={**log_extra, "error": str(exc)},
        )
        return None


async def _deliver_and_track(
    ctx: JobContext,
    report: "ReportResult | ReportError",
    business: BusinessModel,
    report_type: str,
    result: JobResult,
    log_extra: dict,
) -> JobResult:
    """
    Deliver an assembled report via WhatsApp and record usage.

    Handles both ReportResult (success) and ReportError (assembly failure).
    For successful assembly: delivers via whatsapp_service, tracks usage.
    For assembly failure: logs the error, marks result as failed.

    WhatsApp delivery failure is treated as a warning — the report was
    successfully built and the usage counter is still incremented. Delivery
    failures are transient (wrong number, API outage) and are not retried
    here — the scheduler will regenerate the report next cycle.

    Args:
        ctx:         JobContext with db and whatsapp_service.
        report:      ReportResult or ReportError from reports_service.
        business:    BusinessModel for logging context.
        report_type: ReportType constant for logging.
        result:      JobResult being built for this task.
        log_extra:   Structured log context dict.

    Returns:
        Updated JobResult.
    """
    # ------------------------------------------------------------------
    # Assembly failure
    # ------------------------------------------------------------------
    if isinstance(report, ReportError):
        logger.error(
            "Report assembly failed",
            extra={
                **log_extra,
                "report_type": report_type,
                "reason": report.reason,
                "detail": report.detail,
            },
        )
        result.success = False
        result.add_error(f"Report assembly failed: {report.reason} — {report.detail}")
        return result

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------
    logger.info(
        "Report assembled — delivering via WhatsApp",
        extra={
            **log_extra,
            "report_type": report_type,
            "messages": report.message_count,
            "chars": report.total_chars,
            "used_ai": report.used_ai_insights,
            "period": report.period_label,
        },
    )

    delivery = await ctx.whatsapp_service.send_report(
        db=ctx.db,
        business_id=str(ctx.business_id),
        report_parts=report.messages,
        report_type=report_type,
    )

    if delivery.success:
        logger.info(
            "Report delivered successfully",
            extra={
                **log_extra,
                "report_type": report_type,
                "message_id": delivery.message_id,
                "business_name": business.business_name,
            },
        )
    else:
        logger.warning(
            "Report WhatsApp delivery failed",
            extra={
                **log_extra,
                "report_type": report_type,
                "error": delivery.error,
            },
        )

    # ------------------------------------------------------------------
    # Usage tracking — always record the report was generated,
    # regardless of delivery outcome (report was built and consumed quota)
    # ------------------------------------------------------------------
    try:
        await track_report_generated(ctx.db, ctx.business_id)
    except Exception as exc:
        logger.warning(
            "Failed to track report_generated usage",
            extra={**log_extra, "error": str(exc)},
        )

    result.processed += 1
    return result


# ==============================================================================
# Quarter label helper
# ==============================================================================

def _quarter_label(period_start: date) -> str:
    """
    Derive a human-readable quarter label from a quarter start date.

    Args:
        period_start: First day of a calendar quarter.

    Returns:
        str: e.g. "Q1 2025", "Q3 2024"
    """
    quarter_number = (period_start.month - 1) // 3 + 1
    return f"Q{quarter_number} {period_start.year}"


# ==============================================================================
# Theme derivation helpers
# ==============================================================================

def _derive_positive_themes(
    avg_rating: Optional[float],
    positive_count: int,
) -> Optional[list[str]]:
    """
    Derive a list of positive theme labels for the insights prompt.

    These are high-level signals based on available data — used as context
    for the OpenAI insight prompt, not as factual claims.

    Args:
        avg_rating:     Average star rating for the period.
        positive_count: Count of positive reviews.

    Returns:
        list[str] of theme labels, or None if insufficient data.
    """
    if positive_count == 0:
        return None

    themes: list[str] = []

    if avg_rating is not None and avg_rating >= 4.5:
        themes.append("Excellent customer satisfaction")
    elif avg_rating is not None and avg_rating >= 4.0:
        themes.append("Strong customer satisfaction")

    if positive_count >= 10:
        themes.append("High volume of positive feedback")
    elif positive_count >= 5:
        themes.append("Consistent positive feedback")

    return themes if themes else None


def _derive_negative_themes(
    negative_count: int,
    total_count: int,
) -> Optional[list[str]]:
    """
    Derive a list of negative theme labels for the insights prompt.

    Args:
        negative_count: Count of negative reviews.
        total_count:    Total review count for the period.

    Returns:
        list[str] of theme labels, or None if no negative reviews.
    """
    if negative_count == 0:
        return None

    themes: list[str] = []
    negative_rate = negative_count / total_count if total_count > 0 else 0.0

    if negative_rate >= 0.3:
        themes.append("High proportion of negative reviews requires attention")
    elif negative_rate >= 0.15:
        themes.append("Notable proportion of negative feedback")

    if negative_count >= 5:
        themes.append("Multiple negative reviews in period")

    return themes if themes else None