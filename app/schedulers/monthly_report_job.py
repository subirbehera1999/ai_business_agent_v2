# ==============================================================================
# File: app/schedulers/monthly_report_job.py
# Purpose: First-of-month scheduled job that generates and delivers a
#          comprehensive monthly business performance report to every
#          active subscribed business via WhatsApp.
#
#          Schedule: 1st of every month at 08:00 UTC
#                    (registered by scheduler_manager.py)
#
#          All subscribed businesses receive all 5 report sections.
#          There is no plan gating — subscription = full access.
#
#          Report sections (all businesses receive all parts):
#            Part 1 — Monthly Overview
#                     Total reviews, avg rating, MoM rating change,
#                     volume comparison vs previous month,
#                     response rate (reviews with AI replies / total)
#
#            Part 2 — Reputation Trend
#                     Week-by-week rating trajectory across the month,
#                     best and worst performing week,
#                     cumulative trend verdict (improving/declining/stable)
#
#            Part 3 — Top Review Themes
#                     Most praised aspects (from positive reviews)
#                     Most complained aspects (from negative reviews)
#                     AI-extracted recurring themes
#
#            Part 4 — Competitor Intelligence
#                     Tracked competitors' rating changes this month
#                     Review volume comparison
#                     Competitive position summary
#
#            Part 5 — AI Strategic Insights
#                     3-4 data-driven strategic recommendations
#                     Based on reputation trend + theme analysis
#
#          Processing pipeline per business:
#            1. Verify active subscription (only check — no tier check)
#            2. Idempotency check (skip if already sent this month)
#            3. Fetch this month's and previous month's reviews
#            4. Compute monthly statistics and week-by-week breakdown
#            5. Extract themes from reviews via reports_service
#            6. Fetch competitor snapshots if any are tracked
#            7. Generate AI strategic insights via reports_service
#            8. Build and deliver multi-part WhatsApp report
#            9. Record delivery + increment usage counter
#
#          Performance contract:
#            - Businesses processed in batches of BUSINESS_BATCH_SIZE (10)
#            - One AI call per business (strategic insights only)
#            - Max reviews sent to AI: MAX_REVIEWS_FOR_INSIGHTS (20)
#            - Each business isolated — failure never blocks others
#            - Scheduler-level job lock managed by scheduler_manager.py
#
#          Idempotency:
#            Key: MONTHLY_REPORT_{business_id}_{YYYY_MM}
#            Prevents duplicate delivery on scheduler retry.
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.notifications.admin_notification_service import AdminNotificationService
from app.notifications.whatsapp_service import WhatsAppService
from app.repositories.business_repository import BusinessRepository
from app.repositories.review_repository import ReviewRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.services.reports_service import ReportsService
from app.utils.batch_utils import process_in_batches
from app.utils.formatting_utils import format_star_bar, format_percent
from app.utils.idempotency_utils import make_monthly_report_key
from app.utils.time_utils import (
    get_month_date_range,
    get_week_boundaries_in_month,
    month_label,
    today_local,
)
from app.utils.usage_tracker import UsageTracker

logger = logging.getLogger(ServiceName.MONTHLY_REPORT)

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
BUSINESS_BATCH_SIZE: int = 10        # businesses per batch cycle
MAX_REVIEWS_FOR_INSIGHTS: int = 20   # max reviews sent to AI for insights
MIN_REVIEWS_FOR_REPORT: int = 3      # skip report if fewer reviews last month
MAX_THEMES_DISPLAYED: int = 5        # max positive/negative themes shown


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class BusinessMonthlyResult:
    """
    Result of generating and delivering a monthly report for one business.

    Attributes:
        business_id:      Business UUID.
        parts_delivered:  WhatsApp parts sent.
        month_label:      Human-readable month e.g. "March 2025".
        reviews_covered:  Number of reviews included in the report.
        skipped:          True if business was intentionally skipped.
        skip_reason:      Why it was skipped.
        error:            Error if generation/delivery failed.
    """
    business_id: str
    parts_delivered: int = 0
    month_label: str = ""
    reviews_covered: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MonthlyReportJobResult:
    """
    Aggregate result of the full monthly report run.

    Attributes:
        businesses_reported: Businesses that received a full report.
        businesses_skipped:  Businesses skipped.
        businesses_errored:  Businesses that failed.
        total_parts_sent:    Total WhatsApp parts delivered.
        month_label:         Month this run covered.
    """
    businesses_reported: int = 0
    businesses_skipped: int = 0
    businesses_errored: int = 0
    total_parts_sent: int = 0
    month_label: str = ""

    def merge(self, biz: BusinessMonthlyResult) -> None:
        """Fold a single business result into the aggregate."""
        if biz.skipped:
            self.businesses_skipped += 1
            return
        if biz.error and biz.parts_delivered == 0:
            self.businesses_errored += 1
            return
        self.businesses_reported += 1
        self.total_parts_sent += biz.parts_delivered


# ==============================================================================
# Monthly Report Job
# ==============================================================================

class MonthlyReportJob:
    """
    Generates and delivers comprehensive monthly performance reports
    to every active subscribed business. No plan gating — all sections
    delivered to all businesses with an active subscription.

    Report content sourced from:
      - review_repo:      stored reviews for the month
      - reports_service:  AI theme extraction + strategic insights
      - whatsapp_service: multi-part message delivery
    """

    def __init__(
        self,
        review_repo: ReviewRepository,
        business_repo: BusinessRepository,
        subscription_repo: SubscriptionRepository,
        reports_service: ReportsService,
        whatsapp_service: WhatsAppService,
        usage_tracker: UsageTracker,
        admin_notification: AdminNotificationService,
    ) -> None:
        self._review_repo = review_repo
        self._biz_repo = business_repo
        self._sub_repo = subscription_repo
        self._reports = reports_service
        self._whatsapp = whatsapp_service
        self._usage_tracker = usage_tracker
        self._admin = admin_notification

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, db: AsyncSession) -> MonthlyReportJobResult:
        """
        Execute the monthly report cycle for all active businesses.

        Args:
            db: AsyncSession provided by scheduler_manager.py.

        Returns:
            MonthlyReportJobResult. Never raises.
        """
        today = today_local()

        # Report covers the previous completed calendar month
        report_month_start, report_month_end = get_month_date_range(
            reference_date=today, offset_months=-1
        )
        prev_month_start, prev_month_end = get_month_date_range(
            reference_date=today, offset_months=-2
        )
        m_label = month_label(report_month_start)

        aggregate = MonthlyReportJobResult(month_label=m_label)
        log_extra = {
            "service": ServiceName.MONTHLY_REPORT,
            "month": m_label,
        }

        logger.info("Monthly report job started", extra=log_extra)

        try:
            active_ids = await self._sub_repo.get_active_business_ids(db=db)

            if not active_ids:
                logger.info(
                    "No active businesses — skipping monthly report",
                    extra=log_extra,
                )
                return aggregate

            logger.info(
                "Generating monthly reports for %d businesses",
                len(active_ids),
                extra=log_extra,
            )

            async def process_batch(batch: list[str]) -> None:
                for business_id in batch:
                    result = await self._process_business(
                        db=db,
                        business_id=business_id,
                        month_start=report_month_start,
                        month_end=report_month_end,
                        prev_month_start=prev_month_start,
                        prev_month_end=prev_month_end,
                        m_label=m_label,
                    )
                    aggregate.merge(result)

            await process_in_batches(
                items=active_ids,
                batch_size=BUSINESS_BATCH_SIZE,
                processor=process_batch,
            )

        except Exception as exc:
            logger.error(
                "Monthly report job failed at top level",
                extra={**log_extra, "error": str(exc)},
            )
            await self._admin.send_job_failure(
                job_name="monthly_report_job",
                error=str(exc),
            )

        logger.info(
            "Monthly report job complete",
            extra={
                **log_extra,
                "reported": aggregate.businesses_reported,
                "skipped": aggregate.businesses_skipped,
                "errored": aggregate.businesses_errored,
                "parts_sent": aggregate.total_parts_sent,
            },
        )
        return aggregate

    # ------------------------------------------------------------------
    # Per-business processing
    # ------------------------------------------------------------------

    async def _process_business(
        self,
        db: AsyncSession,
        business_id: str,
        month_start: date,
        month_end: date,
        prev_month_start: date,
        prev_month_end: date,
        m_label: str,
    ) -> BusinessMonthlyResult:
        """
        Generate and deliver the full monthly report for one business.

        Returns:
            BusinessMonthlyResult. Never raises.
        """
        result = BusinessMonthlyResult(
            business_id=business_id,
            month_label=m_label,
        )
        log_extra = {
            "service": ServiceName.MONTHLY_REPORT,
            "business_id": business_id,
            "month": m_label,
        }

        try:
            # ── Load business ─────────────────────────────────────────
            business = await self._biz_repo.get_by_id(
                db=db, business_id=business_id
            )
            if not business:
                result.skipped = True
                result.skip_reason = "business_not_found"
                return result

            # ── Idempotency check ─────────────────────────────────────
            idempotency_key = make_monthly_report_key(
                business_id=business_id,
                month_start=month_start,
            )
            already_sent = await self._review_repo.report_already_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
            )
            if already_sent:
                result.skipped = True
                result.skip_reason = "already_sent_this_month"
                return result

            # ── Fetch review data ─────────────────────────────────────
            this_month_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=month_start,
                end_date=month_end,
            )

            if len(this_month_reviews) < MIN_REVIEWS_FOR_REPORT:
                result.skipped = True
                result.skip_reason = (
                    f"insufficient_reviews: {len(this_month_reviews)} "
                    f"(minimum {MIN_REVIEWS_FOR_REPORT})"
                )
                return result

            result.reviews_covered = len(this_month_reviews)

            prev_month_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=prev_month_start,
                end_date=prev_month_end,
            )

            # ── Week-by-week breakdown ────────────────────────────────
            weekly_breakdown = _build_weekly_breakdown(
                reviews=this_month_reviews,
                month_start=month_start,
                month_end=month_end,
            )

            # ── Build all report parts ────────────────────────────────
            parts = await self._build_report_parts(
                db=db,
                business_id=business_id,
                business=business,
                this_month=this_month_reviews,
                prev_month=prev_month_reviews,
                weekly_breakdown=weekly_breakdown,
                m_label=m_label,
                log_extra=log_extra,
            )

            if not parts:
                result.error = "report_build_returned_no_parts"
                return result

            # ── Deliver via WhatsApp ──────────────────────────────────
            delivery = await self._whatsapp.send_report(
                db=db,
                business_id=business_id,
                report_parts=parts,
                report_type="monthly",
            )

            result.parts_delivered = delivery.parts_sent

            if not delivery.success:
                result.error = f"delivery_failed: {delivery.error}"
                logger.warning(
                    "Monthly report delivery failed",
                    extra={**log_extra, "error": delivery.error},
                )
                return result

            # ── Record delivery for idempotency ──────────────────────
            await self._review_repo.record_report_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
                report_type="monthly",
            )
            await db.commit()

            # ── Increment usage counter ───────────────────────────────
            try:
                await self._usage_tracker.increment_reports_generated(
                    db=db,
                    business_id=business_id,
                )
            except Exception as exc:
                logger.warning(
                    "Usage tracking failed",
                    extra={**log_extra, "error": str(exc)},
                )

            logger.info(
                "Monthly report delivered",
                extra={
                    **log_extra,
                    "parts": result.parts_delivered,
                    "reviews": result.reviews_covered,
                },
            )

        except Exception as exc:
            logger.error(
                "Monthly report failed for business",
                extra={**log_extra, "error": str(exc)},
            )
            result.error = str(exc)

        return result

    # ------------------------------------------------------------------
    # Report builder — all 5 parts for every business
    # ------------------------------------------------------------------

    async def _build_report_parts(
        self,
        db: AsyncSession,
        business_id: str,
        business,
        this_month: list,
        prev_month: list,
        weekly_breakdown: list[dict],
        m_label: str,
        log_extra: dict,
    ) -> list[str]:
        """
        Build all 5 WhatsApp message parts for the monthly report.

        Every subscribed business receives all sections. Parts are
        built independently — failure in one omits that section
        rather than aborting the whole report.

        Returns:
            Ordered list of message strings. Empty list on total failure.
        """
        parts: list[str] = []

        # ── Part 1: Monthly Overview ──────────────────────────────────
        try:
            parts.append(_build_overview_part(
                business_name=business.business_name,
                m_label=m_label,
                this_month=this_month,
                prev_month=prev_month,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 1 (overview)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 2: Reputation Trend ──────────────────────────────────
        try:
            parts.append(_build_reputation_trend_part(
                weekly_breakdown=weekly_breakdown,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 2 (reputation trend)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 3: Top Review Themes (AI) ────────────────────────────
        try:
            part3 = await self._build_themes_part(
                business_id=business_id,
                business_name=business.business_name,
                this_month=this_month,
                log_extra=log_extra,
            )
            if part3:
                parts.append(part3)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 3 (themes)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 4: Competitor Intelligence ──────────────────────────
        try:
            part4 = await self._build_competitor_part(
                db=db,
                business_id=business_id,
                m_label=m_label,
                log_extra=log_extra,
            )
            if part4:
                parts.append(part4)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 4 (competitor)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 5: AI Strategic Insights ────────────────────────────
        try:
            part5 = await self._build_insights_part(
                business_id=business_id,
                business_name=business.business_name,
                this_month=this_month,
                weekly_breakdown=weekly_breakdown,
                m_label=m_label,
                log_extra=log_extra,
            )
            if part5:
                parts.append(part5)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 5 (AI insights)",
                extra={**log_extra, "error": str(exc)},
            )

        return parts

    # ------------------------------------------------------------------
    # Section builders requiring service calls
    # ------------------------------------------------------------------

    async def _build_themes_part(
        self,
        business_id: str,
        business_name: str,
        this_month: list,
        log_extra: dict,
    ) -> Optional[str]:
        """
        AI-extract recurring positive and negative themes from this month's reviews.
        Returns None if extraction fails or no themes found.
        """
        try:
            positive_reviews = [
                r for r in this_month
                if getattr(r, "sentiment", "neutral") == "positive"
            ][:MAX_REVIEWS_FOR_INSIGHTS // 2]

            negative_reviews = [
                r for r in this_month
                if getattr(r, "sentiment", "neutral") == "negative"
            ][:MAX_REVIEWS_FOR_INSIGHTS // 2]

            theme_result = await self._reports.extract_review_themes(
                business_id=business_id,
                business_name=business_name,
                positive_reviews=positive_reviews,
                negative_reviews=negative_reviews,
            )

            if not theme_result.success:
                return None

            lines = ["🔍 *Top Review Themes This Month*", ""]

            if theme_result.positive_themes:
                lines.append("*What customers loved:*")
                for theme in theme_result.positive_themes[:MAX_THEMES_DISPLAYED]:
                    lines.append(f"✅ {theme}")
                lines.append("")

            if theme_result.negative_themes:
                lines.append("*Areas customers mentioned:*")
                for theme in theme_result.negative_themes[:MAX_THEMES_DISPLAYED]:
                    lines.append(f"⚠️ {theme}")

            return "\n".join(lines) if len(lines) > 2 else None

        except Exception as exc:
            logger.warning(
                "Theme extraction failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None

    async def _build_competitor_part(
        self,
        db: AsyncSession,
        business_id: str,
        m_label: str,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Build competitor intelligence from stored monthly snapshots.
        Returns None if no competitors are tracked for this business.
        """
        try:
            snapshots = await self._review_repo.get_recent_competitor_snapshots(
                db=db,
                business_id=business_id,
                limit=5,
            )
            if not snapshots:
                return None

            lines = [
                f"🏆 *Competitor Intelligence — {m_label}*",
                "",
            ]
            for comp in snapshots:
                delta = comp.get("rating_delta")
                delta_str = ""
                if delta is not None:
                    arrow = "↑" if delta >= 0 else "↓"
                    delta_str = f" ({arrow}{abs(delta):.1f} this month)"
                lines.append(
                    f"• *{comp['name'][:30]}*: "
                    f"⭐ {comp.get('current_rating', 'N/A')}{delta_str} "
                    f"| {comp.get('review_count', 0)} reviews"
                )

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "Competitor intelligence build failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None

    async def _build_insights_part(
        self,
        business_id: str,
        business_name: str,
        this_month: list,
        weekly_breakdown: list[dict],
        m_label: str,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Generate 3-4 AI strategic recommendations from the month's data.
        Curates a mixed review sample to stay within token budget.
        Returns None if AI generation fails.
        """
        try:
            negative = [
                r for r in this_month
                if getattr(r, "sentiment", "neutral") in ("negative", "neutral")
            ][:MAX_REVIEWS_FOR_INSIGHTS // 2]
            positive = [
                r for r in this_month
                if getattr(r, "sentiment", "neutral") == "positive"
            ][:MAX_REVIEWS_FOR_INSIGHTS // 2]
            sample = (negative + positive)[:MAX_REVIEWS_FOR_INSIGHTS]

            trend_summary = _summarise_weekly_trend(weekly_breakdown)

            insight_result = await self._reports.generate_monthly_insights(
                business_id=business_id,
                business_name=business_name,
                reviews=sample,
                trend_summary=trend_summary,
                month_label=m_label,
            )

            if not insight_result.success or not insight_result.insights:
                return None

            lines = [f"🎯 *Strategic Insights — {m_label}*", ""]
            for i, insight in enumerate(insight_result.insights[:4], 1):
                lines.append(f"{i}. {insight}")
            lines += ["", "_AI-generated from your monthly review data._"]

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "AI strategic insights generation failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None


# ==============================================================================
# Pure section builders
# ==============================================================================

def _build_overview_part(
    business_name: str,
    m_label: str,
    this_month: list,
    prev_month: list,
) -> str:
    """Part 1: Monthly overview with MoM comparison and star breakdown."""
    total = len(this_month)
    prev_total = len(prev_month)
    avg = _avg_rating_from_list(this_month)
    prev_avg = _avg_rating_from_list(prev_month)

    # Volume change
    if prev_total > 0:
        vol_delta = total - prev_total
        vol_pct = (vol_delta / prev_total) * 100
        vol_str = (
            f"{'📈' if vol_delta >= 0 else '📉'} "
            f"{'▲' if vol_delta >= 0 else '▼'}{abs(vol_delta)} "
            f"({format_percent(abs(vol_pct))} vs last month)"
        )
    else:
        vol_str = "📊 First full month of data"

    # Rating change
    if avg and prev_avg:
        delta = avg - prev_avg
        if abs(delta) < 0.05:
            rating_trend = "➡️ Steady"
        elif delta > 0:
            rating_trend = f"📈 Up {delta:+.2f} vs last month"
        else:
            rating_trend = f"📉 Down {delta:+.2f} vs last month"
    else:
        rating_trend = "📊 No prior month data"

    # Response rate
    replied = sum(1 for r in this_month if getattr(r, "has_reply", False))
    response_rate = (replied / total * 100) if total > 0 else 0.0

    # Star breakdown
    star_counts = {i: 0 for i in range(1, 6)}
    for r in this_month:
        rating = getattr(r, "rating", 0)
        if 1 <= rating <= 5:
            star_counts[rating] += 1

    lines = [
        f"📅 *Monthly Report — {business_name}*",
        f"_{m_label}_",
        "",
        "📊 *Monthly Overview*",
        "",
        f"Total Reviews:    *{total}*   {vol_str}",
        (
            f"Average Rating:   *{avg:.2f}/5.0*   {rating_trend}"
            if avg else "Average Rating:   *N/A*"
        ),
        f"Response Rate:    *{response_rate:.0f}%* ({replied}/{total} replied)",
        "",
        "*Rating Breakdown:*",
    ]
    for stars in range(5, 0, -1):
        count = star_counts[stars]
        bar = format_star_bar(count, total)
        lines.append(f"{'⭐' * stars}  {bar} ({count})")

    return "\n".join(lines)


def _build_reputation_trend_part(weekly_breakdown: list[dict]) -> str:
    """Part 2: Week-by-week reputation trajectory with trend verdict."""
    if not weekly_breakdown:
        return (
            "📈 *Reputation Trend*\n\n"
            "_No week-by-week data available for this period._"
        )

    lines = ["📈 *Reputation Trend*", ""]
    prev_rating: Optional[float] = None

    for week in weekly_breakdown:
        label = week.get("label", "Unknown week")
        count = week.get("count", 0)
        avg = week.get("avg_rating")

        if avg is None:
            lines.append(f"  {label}:  No reviews")
            continue

        arrow = "•"
        if prev_rating is not None:
            delta = avg - prev_rating
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")

        lines.append(f"  {label}:  ⭐ {avg:.1f}  {arrow}  ({count} reviews)")
        prev_rating = avg

    # Overall trajectory
    ratings = [
        w["avg_rating"] for w in weekly_breakdown
        if w.get("avg_rating") is not None
    ]
    if len(ratings) >= 2:
        overall = ratings[-1] - ratings[0]
        if overall > 0.15:
            verdict = "📈 *Improving* — rating trended upward this month."
        elif overall < -0.15:
            verdict = "📉 *Declining* — rating trended downward this month."
        else:
            verdict = "➡️ *Stable* — rating remained consistent this month."
        lines += ["", verdict]

    return "\n".join(lines)


# ==============================================================================
# Helpers
# ==============================================================================

def _build_weekly_breakdown(
    reviews: list,
    month_start: date,
    month_end: date,
) -> list[dict]:
    """Group reviews by calendar week and compute per-week avg rating."""
    from app.utils.time_utils import get_week_boundaries_in_month

    week_ranges = get_week_boundaries_in_month(
        month_start=month_start,
        month_end=month_end,
    )
    breakdown = []
    for i, (w_start, w_end) in enumerate(week_ranges, 1):
        week_reviews = [r for r in reviews if _review_in_range(r, w_start, w_end)]
        label = f"Week {i} ({w_start.strftime('%b %-d')}–{w_end.strftime('%-d')})"
        breakdown.append({
            "label": label,
            "count": len(week_reviews),
            "avg_rating": _avg_rating_from_list(week_reviews),
        })
    return breakdown


def _summarise_weekly_trend(weekly_breakdown: list[dict]) -> str:
    """Compact text summary of weekly trend for AI prompt context."""
    if not weekly_breakdown:
        return "No weekly trend data available."
    parts = []
    for w in weekly_breakdown:
        avg = w.get("avg_rating")
        avg_str = f"{avg:.1f}" if avg else "no data"
        parts.append(
            f"{w.get('label', '')}: avg {avg_str} ({w.get('count', 0)} reviews)"
        )
    return "; ".join(parts)


def _avg_rating_from_list(reviews: list) -> Optional[float]:
    """Compute average star rating from a list of review records."""
    ratings = [
        getattr(r, "rating", None) for r in reviews
        if getattr(r, "rating", None) is not None
    ]
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 2)


def _review_in_range(review, start: date, end: date) -> bool:
    """Return True if a review's date falls within [start, end] inclusive."""
    review_date = getattr(review, "review_date", None)
    if not review_date:
        return False
    if hasattr(review_date, "date"):
        review_date = review_date.date()
    return start <= review_date <= end


# ==============================================================================
# Functional entry point (called by scheduler_manager.py)
# ==============================================================================

async def run_monthly_report(
    db: AsyncSession,
    business_repo: BusinessRepository,
    subscription_repo: SubscriptionRepository,
    admin_notification: AdminNotificationService,
    **kwargs,
) -> MonthlyReportJobResult:
    """
    Functional entry point for the monthly report job.

    Called by scheduler_manager.py.

    Returns:
        MonthlyReportJobResult. Never raises.
    """
    job = MonthlyReportJob(
        review_repo=kwargs["review_repo"],
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        reports_service=kwargs["reports_service"],
        whatsapp_service=kwargs["whatsapp_service"],
        usage_tracker=kwargs["usage_tracker"],
        admin_notification=admin_notification,
    )
    return await job.run(db=db)