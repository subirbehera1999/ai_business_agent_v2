# ==============================================================================
# File: app/schedulers/weekly_report_job.py
# Purpose: Monday morning scheduled job (06:00 UTC) that generates and
#          delivers a multi-part weekly performance report to every
#          active subscribed business via WhatsApp.
#
#          Schedule: Every Monday at 06:00 UTC (registered by scheduler_manager)
#
#          Every subscribed business receives all 4 report sections —
#          there is no plan gating. Subscription = full access.
#
#          Report sections (all businesses receive all parts):
#            Part 1 — Review Summary
#                     Total reviews, star rating breakdown,
#                     average rating vs previous week
#
#            Part 2 — Sentiment Trends
#                     Positive / negative / neutral breakdown,
#                     sentiment shift vs previous week
#
#            Part 3 — Competitor Snapshot
#                     Tracked competitors' ratings and review counts
#                     (omitted if no competitors are configured)
#
#            Part 4 — AI Improvement Suggestions
#                     2-3 actionable suggestions from review themes
#
#          Processing pipeline per business:
#            1. Verify active subscription
#            2. Idempotency check — skip if report already sent this week
#            3. Fetch this week's and last week's reviews
#            4. Build Parts 1 and 2 from review data (pure computation)
#            5. Fetch competitor snapshots for Part 3
#            6. Generate AI suggestions for Part 4 via reports_service
#            7. Deliver all parts via WhatsApp
#            8. Record delivery + increment usage counter
#
#          Performance contract:
#            - Businesses processed in batches of BUSINESS_BATCH_SIZE (10)
#            - One AI call per business (Part 4 suggestions only)
#            - Max reviews sent to AI: MAX_REVIEWS_IN_PROMPT (15)
#            - Parts built independently — one failure omits that section
#            - Scheduler-level job lock managed by scheduler_manager.py
#
#          Idempotency:
#            Key: WEEKLY_REPORT_{business_id}_{YYYY_Www}
#            Prevents duplicate delivery on scheduler retry.
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import date, timedelta
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
from app.utils.formatting_utils import format_star_bar
from app.utils.idempotency_utils import make_weekly_report_key
from app.utils.time_utils import (
    get_week_date_range,
    iso_week_label,
    today_local,
)
from app.utils.usage_tracker import UsageTracker

logger = logging.getLogger(ServiceName.WEEKLY_REPORT)

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
BUSINESS_BATCH_SIZE: int = 10
MAX_REVIEWS_IN_PROMPT: int = 15
MIN_REVIEWS_FOR_REPORT: int = 1


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class BusinessReportResult:
    """Result of generating and delivering a weekly report for one business."""
    business_id: str
    parts_delivered: int = 0
    week_label: str = ""
    reviews_covered: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WeeklyReportJobResult:
    """Aggregate result of the full weekly report run."""
    businesses_reported: int = 0
    businesses_skipped: int = 0
    businesses_errored: int = 0
    total_parts_sent: int = 0
    week_label: str = ""

    def merge(self, biz: BusinessReportResult) -> None:
        if biz.skipped:
            self.businesses_skipped += 1
            return
        if biz.error and biz.parts_delivered == 0:
            self.businesses_errored += 1
            return
        self.businesses_reported += 1
        self.total_parts_sent += biz.parts_delivered


# ==============================================================================
# Weekly Report Job
# ==============================================================================

class WeeklyReportJob:
    """
    Generates and delivers weekly performance reports to all subscribed businesses.

    No plan gating — every subscribed business receives all 4 report
    sections including competitor snapshot and AI suggestions.
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

    async def run(self, db: AsyncSession) -> WeeklyReportJobResult:
        """
        Execute the weekly report cycle for all active businesses.

        Returns:
            WeeklyReportJobResult. Never raises.
        """
        today = today_local()
        week_start, week_end = get_week_date_range(
            reference_date=today, offset_weeks=-1
        )
        week_label = iso_week_label(week_start, week_end)

        aggregate = WeeklyReportJobResult(week_label=week_label)
        log_extra = {
            "service": ServiceName.WEEKLY_REPORT,
            "week": week_label,
        }

        logger.info("Weekly report job started", extra=log_extra)

        try:
            active_ids = await self._sub_repo.get_active_business_ids(db=db)

            if not active_ids:
                logger.info(
                    "No active businesses — skipping weekly report",
                    extra=log_extra,
                )
                return aggregate

            logger.info(
                "Generating reports for %d businesses",
                len(active_ids),
                extra=log_extra,
            )

            async def process_batch(batch: list[str]) -> None:
                for business_id in batch:
                    result = await self._process_business(
                        db=db,
                        business_id=business_id,
                        week_start=week_start,
                        week_end=week_end,
                        week_label=week_label,
                    )
                    aggregate.merge(result)

            await process_in_batches(
                items=active_ids,
                batch_size=BUSINESS_BATCH_SIZE,
                processor=process_batch,
            )

        except Exception as exc:
            logger.error(
                "Weekly report job failed at top level",
                extra={**log_extra, "error": str(exc)},
            )
            await self._admin.send_job_failure(
                job_name="weekly_report_job",
                error=str(exc),
            )

        logger.info(
            "Weekly report job complete",
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
        week_start: date,
        week_end: date,
        week_label: str,
    ) -> BusinessReportResult:
        """Generate and deliver the full weekly report for one business."""
        result = BusinessReportResult(
            business_id=business_id,
            week_label=week_label,
        )
        log_extra = {
            "service": ServiceName.WEEKLY_REPORT,
            "business_id": business_id,
            "week": week_label,
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
            idempotency_key = make_weekly_report_key(
                business_id=business_id,
                week_start=week_start,
            )
            already_sent = await self._review_repo.report_already_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
            )
            if already_sent:
                result.skipped = True
                result.skip_reason = "already_sent_this_week"
                return result

            # ── Fetch review data ─────────────────────────────────────
            this_week_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=week_start,
                end_date=week_end,
            )

            if len(this_week_reviews) < MIN_REVIEWS_FOR_REPORT:
                result.skipped = True
                result.skip_reason = (
                    f"insufficient_reviews: {len(this_week_reviews)}"
                )
                return result

            result.reviews_covered = len(this_week_reviews)

            prev_week_start = week_start - timedelta(weeks=1)
            prev_week_end = week_start - timedelta(days=1)
            prev_week_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=prev_week_start,
                end_date=prev_week_end,
            )

            # ── Build all report parts ────────────────────────────────
            parts = await self._build_report_parts(
                db=db,
                business_id=business_id,
                business=business,
                this_week=this_week_reviews,
                prev_week=prev_week_reviews,
                week_label=week_label,
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
                report_type="weekly",
            )

            result.parts_delivered = delivery.parts_sent

            if not delivery.success:
                result.error = f"delivery_failed: {delivery.error}"
                logger.warning(
                    "Weekly report delivery failed",
                    extra={**log_extra, "error": delivery.error},
                )
                return result

            # ── Record delivery ───────────────────────────────────────
            await self._review_repo.record_report_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
                report_type="weekly",
            )
            await db.commit()

            # ── Usage counter ─────────────────────────────────────────
            try:
                await self._usage_tracker.increment_reports_generated(
                    db=db, business_id=business_id
                )
            except Exception as exc:
                logger.warning(
                    "Usage tracking failed",
                    extra={**log_extra, "error": str(exc)},
                )

            logger.info(
                "Weekly report delivered",
                extra={
                    **log_extra,
                    "parts": result.parts_delivered,
                    "reviews": result.reviews_covered,
                },
            )

        except Exception as exc:
            logger.error(
                "Weekly report failed for business",
                extra={**log_extra, "error": str(exc)},
            )
            result.error = str(exc)

        return result

    # ------------------------------------------------------------------
    # Report builder — all parts for every subscribed business
    # ------------------------------------------------------------------

    async def _build_report_parts(
        self,
        db: AsyncSession,
        business_id: str,
        business,
        this_week: list,
        prev_week: list,
        week_label: str,
        log_extra: dict,
    ) -> list[str]:
        """
        Build all WhatsApp message parts for the weekly report.

        Every subscribed business receives all parts. Parts are built
        independently — failure in one omits that section rather than
        aborting the whole report.
        """
        parts: list[str] = []

        # ── Part 1: Review Summary ────────────────────────────────────
        try:
            parts.append(_build_review_summary_part(
                business_name=business.business_name,
                week_label=week_label,
                this_week=this_week,
                prev_week=prev_week,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 1 (review summary)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 2: Sentiment Trends ──────────────────────────────────
        try:
            parts.append(_build_sentiment_part(
                this_week=this_week,
                prev_week=prev_week,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 2 (sentiment trends)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 3: Competitor Snapshot ───────────────────────────────
        try:
            part3 = await self._build_competitor_part(
                db=db,
                business_id=business_id,
                log_extra=log_extra,
            )
            if part3:
                parts.append(part3)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 3 (competitor snapshot)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 4: AI Improvement Suggestions ───────────────────────
        try:
            part4 = await self._build_suggestions_part(
                business_id=business_id,
                business_name=business.business_name,
                this_week=this_week,
                log_extra=log_extra,
            )
            if part4:
                parts.append(part4)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 4 (AI suggestions)",
                extra={**log_extra, "error": str(exc)},
            )

        return parts

    async def _build_competitor_part(
        self,
        db: AsyncSession,
        business_id: str,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Build competitor snapshot from stored data.
        Returns None if no competitors configured for this business.
        """
        try:
            snapshots = await self._review_repo.get_recent_competitor_snapshots(
                db=db,
                business_id=business_id,
                limit=5,
            )
            if not snapshots:
                return None

            lines = ["🏆 *Competitor Snapshot*", ""]
            for comp in snapshots:
                delta = comp.get("rating_delta")
                delta_str = ""
                if delta is not None:
                    arrow = "↑" if delta >= 0 else "↓"
                    delta_str = f" ({arrow}{abs(delta):.1f} vs last week)"
                lines.append(
                    f"• *{comp['name'][:30]}*: "
                    f"⭐ {comp.get('current_rating', 'N/A')}{delta_str} "
                    f"({comp.get('review_count', 0)} reviews)"
                )
            lines += ["", "_Competitor data refreshed weekly._"]
            return "\n".join(lines)
        except Exception as exc:
            logger.warning(
                "Competitor snapshot fetch failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None

    async def _build_suggestions_part(
        self,
        business_id: str,
        business_name: str,
        this_week: list,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Generate AI improvement suggestions from this week's reviews.
        Prioritises negative/neutral reviews for improvement signal.
        Returns None if AI generation fails.
        """
        sample = [
            r for r in this_week
            if getattr(r, "sentiment", "neutral") in ("negative", "neutral")
        ][:MAX_REVIEWS_IN_PROMPT]

        if not sample:
            sample = this_week[:3]

        try:
            suggestion_result = await self._reports.generate_weekly_suggestions(
                business_id=business_id,
                business_name=business_name,
                reviews=sample,
            )

            if not suggestion_result.success or not suggestion_result.suggestions:
                return None

            lines = ["💡 *This Week's Improvement Tips*", ""]
            for i, tip in enumerate(suggestion_result.suggestions[:3], 1):
                lines.append(f"{i}. {tip}")
            lines += ["", "_AI-generated based on your recent reviews._"]
            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "AI suggestions generation failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None


# ==============================================================================
# Pure section builders
# ==============================================================================

def _build_review_summary_part(
    business_name: str,
    week_label: str,
    this_week: list,
    prev_week: list,
) -> str:
    """Part 1: header, review count, average rating, star breakdown."""
    total = len(this_week)
    avg = _avg_rating(this_week)
    prev_avg = _avg_rating(prev_week)

    if prev_avg and avg:
        delta = avg - prev_avg
        if abs(delta) < 0.1:
            movement = "➡️ Steady"
        elif delta > 0:
            movement = f"📈 Up {delta:+.1f} vs last week"
        else:
            movement = f"📉 Down {delta:+.1f} vs last week"
    else:
        movement = "📊 First week of data"

    star_counts = {i: 0 for i in range(1, 6)}
    for r in this_week:
        rating = getattr(r, "rating", 0)
        if 1 <= rating <= 5:
            star_counts[rating] += 1

    lines = [
        f"📋 *Weekly Report — {business_name}*",
        f"_{week_label}_",
        "",
        "⭐ *Review Summary*",
        "",
        f"Total Reviews:   *{total}*",
        f"Average Rating:  *{avg:.1f}/5.0* {movement}" if avg else
        f"Average Rating:  *N/A*",
        "",
        "*Rating Breakdown:*",
    ]
    for stars in range(5, 0, -1):
        count = star_counts[stars]
        bar = format_star_bar(count, total)
        lines.append(f"{'⭐' * stars}  {bar} ({count})")

    return "\n".join(lines)


def _build_sentiment_part(this_week: list, prev_week: list) -> str:
    """Part 2: sentiment breakdown with week-over-week shift."""
    def _counts(reviews: list) -> dict[str, int]:
        c: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for r in reviews:
            s = getattr(r, "sentiment", "neutral") or "neutral"
            if s in c:
                c[s] += 1
        return c

    this = _counts(this_week)
    prev = _counts(prev_week)
    total = max(len(this_week), 1)
    prev_total = max(len(prev_week), 1)

    def _pct(count: int, t: int) -> float:
        return round((count / t) * 100, 1) if t > 0 else 0.0

    def _shift(cur: float, prv: float) -> str:
        d = cur - prv
        if not prv or abs(d) < 2.0:
            return ""
        return f" ({'↑' if d > 0 else '↓'}{abs(d):.0f}% vs last week)"

    pos_pct = _pct(this["positive"], total)
    neg_pct = _pct(this["negative"], total)
    neu_pct = _pct(this["neutral"], total)
    prev_pos_pct = _pct(prev["positive"], prev_total)
    prev_neg_pct = _pct(prev["negative"], prev_total)

    lines = [
        "💬 *Sentiment Trends*",
        "",
        f"😊 Positive:  *{pos_pct}%* ({this['positive']} reviews)"
        + _shift(pos_pct, prev_pos_pct),
        f"😐 Neutral:   *{neu_pct}%* ({this['neutral']} reviews)",
        f"😞 Negative:  *{neg_pct}%* ({this['negative']} reviews)"
        + _shift(neg_pct, prev_neg_pct),
    ]

    if neg_pct >= 30:
        lines += [
            "",
            "⚠️ *High negative sentiment this week.*",
            "Consider reviewing recent complaints and following up with customers.",
        ]
    elif pos_pct >= 70:
        lines += [
            "",
            "🎉 *Strong positive sentiment this week!*",
            "Your customers are happy — keep up the great work.",
        ]

    return "\n".join(lines)


# ==============================================================================
# Helpers
# ==============================================================================

def _avg_rating(reviews: list) -> Optional[float]:
    ratings = [
        getattr(r, "rating", None) for r in reviews
        if getattr(r, "rating", None) is not None
    ]
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 2)


# ==============================================================================
# Functional entry point
# ==============================================================================

async def run_weekly_report(
    db: AsyncSession,
    business_repo: BusinessRepository,
    subscription_repo: SubscriptionRepository,
    admin_notification: AdminNotificationService,
    **kwargs,
) -> WeeklyReportJobResult:
    """Functional entry point called by scheduler_manager.py."""
    job = WeeklyReportJob(
        review_repo=kwargs["review_repo"],
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        reports_service=kwargs["reports_service"],
        whatsapp_service=kwargs["whatsapp_service"],
        usage_tracker=kwargs["usage_tracker"],
        admin_notification=admin_notification,
    )
    return await job.run(db=db)