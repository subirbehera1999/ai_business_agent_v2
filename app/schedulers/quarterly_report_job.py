# ==============================================================================
# File: app/schedulers/quarterly_report_job.py
# Purpose: Quarterly scheduled job that generates and delivers the deepest
#          strategic performance report to every active subscribed business.
#
#          Schedule: 1st of January, April, July, October at 09:00 UTC
#                    (registered by scheduler_manager.py)
#                    Runs 1 hour after monthly_report_job on those days.
#
#          Every subscribed business receives the full quarterly report —
#          there is no plan gating. Subscription = full access.
#
#          This is the most comprehensive report in the system.
#          It covers 90 days of data and delivers strategic insights
#          that weekly and monthly reports cannot — because trends only
#          become visible at the quarter scale.
#
#          Report sections (all businesses receive all parts):
#
#            Part 1 — Quarterly Overview
#                     3-month review volume and average rating
#                     QoQ (quarter-over-quarter) comparison
#                     Response rate across the quarter
#                     Sentiment distribution summary
#
#            Part 2 — Month-by-Month Breakdown
#                     Review volume and avg rating for each of the
#                     3 months in the quarter
#                     Best and worst performing month
#                     Momentum direction (accelerating / decelerating)
#
#            Part 3 — Reputation Health Score
#                     Composite score (0–100) derived from:
#                       - Average rating (weighted 40%)
#                       - Positive sentiment ratio (weighted 30%)
#                       - Response rate (weighted 20%)
#                       - Review volume growth (weighted 10%)
#                     Score vs previous quarter
#                     Health verdict (Excellent / Good / Needs Attention / Critical)
#
#            Part 4 — Competitive Position (if competitors tracked)
#                     Quarter-end competitor ratings
#                     QoQ competitor movement
#                     Our rank among tracked competitors
#                     Competitive gap analysis
#
#            Part 5 — AI Strategic Recommendations
#                     4-5 forward-looking quarterly strategy points
#                     Based on 90-day trend data + health score
#                     Specific, actionable, business-type-aware
#
#          Processing pipeline per business:
#            1. Verify active subscription
#            2. Idempotency check (skip if already sent this quarter)
#            3. Fetch this quarter's and previous quarter's reviews
#            4. Compute quarterly stats and month-by-month breakdown
#            5. Calculate Reputation Health Score
#            6. Fetch competitor data if any tracked
#            7. Generate AI strategic recommendations
#            8. Build and deliver multi-part WhatsApp report
#            9. Record delivery + increment usage counter
#
#          Performance contract:
#            - Businesses processed in batches of BUSINESS_BATCH_SIZE (10)
#            - One AI call per business (strategic recommendations only)
#            - Max reviews sent to AI: MAX_REVIEWS_FOR_STRATEGY (25)
#            - Parts built independently — failure in one omits that section
#            - Scheduler-level job lock managed by scheduler_manager.py
#
#          Idempotency:
#            Key: QUARTERLY_REPORT_{business_id}_{YYYY_QN}
#            Example: QUARTERLY_REPORT_uuid_2025_Q1
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
from app.utils.formatting_utils import format_star_bar
from app.utils.idempotency_utils import make_quarterly_report_key
from app.utils.time_utils import (
    get_quarter_date_range,
    get_month_date_range,
    month_label,
    quarter_label,
    today_local,
)
from app.utils.usage_tracker import UsageTracker

logger = logging.getLogger(ServiceName.QUARTERLY_REPORT)

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
BUSINESS_BATCH_SIZE: int = 10
MAX_REVIEWS_FOR_STRATEGY: int = 25   # max reviews sent to AI per business
MIN_REVIEWS_FOR_REPORT: int = 5      # skip if fewer than this in the quarter

# ---------------------------------------------------------------------------
# Reputation Health Score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_AVG_RATING: float = 0.40
WEIGHT_POSITIVE_RATIO: float = 0.30
WEIGHT_RESPONSE_RATE: float = 0.20
WEIGHT_VOLUME_GROWTH: float = 0.10

# Health score thresholds
HEALTH_EXCELLENT: int = 80
HEALTH_GOOD: int = 60
HEALTH_NEEDS_ATTENTION: int = 40


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class ReputationHealthScore:
    """
    Composite reputation health score (0–100) for a business quarter.

    Attributes:
        score:              Composite score 0–100.
        prev_score:         Previous quarter's score (None if first quarter).
        avg_rating_score:   Component score from average rating (0–100).
        positive_ratio_score: Component score from positive sentiment ratio.
        response_rate_score:  Component score from AI reply response rate.
        volume_growth_score:  Component score from review volume growth.
        verdict:            Excellent / Good / Needs Attention / Critical.
        delta:              Score change vs previous quarter (None if first).
    """
    score: float
    prev_score: Optional[float]
    avg_rating_score: float
    positive_ratio_score: float
    response_rate_score: float
    volume_growth_score: float
    verdict: str
    delta: Optional[float]


@dataclass
class BusinessQuarterlyResult:
    """
    Result of generating and delivering a quarterly report for one business.

    Attributes:
        business_id:      Business UUID.
        parts_delivered:  WhatsApp parts sent.
        quarter_label:    Human-readable quarter e.g. "Q1 2025".
        reviews_covered:  Number of reviews in the quarter.
        health_score:     Computed reputation health score.
        skipped:          True if business was intentionally skipped.
        skip_reason:      Why it was skipped.
        error:            Error if generation/delivery failed.
    """
    business_id: str
    parts_delivered: int = 0
    quarter_label: str = ""
    reviews_covered: int = 0
    health_score: Optional[float] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class QuarterlyReportJobResult:
    """
    Aggregate result of the full quarterly report run.

    Attributes:
        businesses_reported: Businesses that received a full report.
        businesses_skipped:  Businesses skipped.
        businesses_errored:  Businesses that failed.
        total_parts_sent:    Total WhatsApp parts delivered.
        quarter_label:       Quarter this run covered.
    """
    businesses_reported: int = 0
    businesses_skipped: int = 0
    businesses_errored: int = 0
    total_parts_sent: int = 0
    quarter_label: str = ""

    def merge(self, biz: BusinessQuarterlyResult) -> None:
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
# Quarterly Report Job
# ==============================================================================

class QuarterlyReportJob:
    """
    Generates and delivers deep quarterly strategic reports to all
    subscribed businesses.

    The quarterly report is the most comprehensive report in the system.
    It answers questions that weekly and monthly reports cannot:
      - Is the business's reputation improving or declining over 90 days?
      - How does the health score compare to last quarter?
      - Where does the business stand against competitors long-term?
      - What strategic moves should be prioritised next quarter?

    No plan gating. Every subscribed business receives the full report.
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

    async def run(self, db: AsyncSession) -> QuarterlyReportJobResult:
        """
        Execute the quarterly report cycle for all active businesses.

        Args:
            db: AsyncSession provided by scheduler_manager.py.

        Returns:
            QuarterlyReportJobResult. Never raises.
        """
        today = today_local()

        # Report covers the previous completed quarter
        q_start, q_end = get_quarter_date_range(
            reference_date=today, offset_quarters=-1
        )
        prev_q_start, prev_q_end = get_quarter_date_range(
            reference_date=today, offset_quarters=-2
        )
        q_label = quarter_label(q_start)

        aggregate = QuarterlyReportJobResult(quarter_label=q_label)
        log_extra = {
            "service": ServiceName.QUARTERLY_REPORT,
            "quarter": q_label,
        }

        logger.info("Quarterly report job started", extra=log_extra)

        try:
            active_ids = await self._sub_repo.get_active_business_ids(db=db)

            if not active_ids:
                logger.info(
                    "No active businesses — skipping quarterly report",
                    extra=log_extra,
                )
                return aggregate

            logger.info(
                "Generating quarterly reports for %d businesses",
                len(active_ids),
                extra=log_extra,
            )

            async def process_batch(batch: list[str]) -> None:
                for business_id in batch:
                    result = await self._process_business(
                        db=db,
                        business_id=business_id,
                        q_start=q_start,
                        q_end=q_end,
                        prev_q_start=prev_q_start,
                        prev_q_end=prev_q_end,
                        q_label=q_label,
                    )
                    aggregate.merge(result)

            await process_in_batches(
                items=active_ids,
                batch_size=BUSINESS_BATCH_SIZE,
                processor=process_batch,
            )

        except Exception as exc:
            logger.error(
                "Quarterly report job failed at top level",
                extra={**log_extra, "error": str(exc)},
            )
            await self._admin.send_job_failure(
                job_name="quarterly_report_job",
                error=str(exc),
            )

        logger.info(
            "Quarterly report job complete",
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
        q_start: date,
        q_end: date,
        prev_q_start: date,
        prev_q_end: date,
        q_label: str,
    ) -> BusinessQuarterlyResult:
        """
        Generate and deliver the full quarterly report for one business.

        Returns:
            BusinessQuarterlyResult. Never raises.
        """
        result = BusinessQuarterlyResult(
            business_id=business_id,
            quarter_label=q_label,
        )
        log_extra = {
            "service": ServiceName.QUARTERLY_REPORT,
            "business_id": business_id,
            "quarter": q_label,
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
            idempotency_key = make_quarterly_report_key(
                business_id=business_id,
                quarter_start=q_start,
            )
            already_sent = await self._review_repo.report_already_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
            )
            if already_sent:
                result.skipped = True
                result.skip_reason = "already_sent_this_quarter"
                return result

            # ── Fetch quarter review data ─────────────────────────────
            this_q_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=q_start,
                end_date=q_end,
            )

            if len(this_q_reviews) < MIN_REVIEWS_FOR_REPORT:
                result.skipped = True
                result.skip_reason = (
                    f"insufficient_reviews: {len(this_q_reviews)} "
                    f"(minimum {MIN_REVIEWS_FOR_REPORT})"
                )
                return result

            result.reviews_covered = len(this_q_reviews)

            prev_q_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=prev_q_start,
                end_date=prev_q_end,
            )

            # ── Month-by-month breakdown ──────────────────────────────
            monthly_breakdown = _build_monthly_breakdown(
                reviews=this_q_reviews,
                q_start=q_start,
            )

            # ── Reputation Health Score ───────────────────────────────
            health = _compute_health_score(
                this_q=this_q_reviews,
                prev_q=prev_q_reviews,
            )
            result.health_score = health.score

            # ── Build all report parts ────────────────────────────────
            parts = await self._build_report_parts(
                db=db,
                business_id=business_id,
                business=business,
                this_q=this_q_reviews,
                prev_q=prev_q_reviews,
                monthly_breakdown=monthly_breakdown,
                health=health,
                q_label=q_label,
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
                report_type="quarterly",
            )

            result.parts_delivered = delivery.parts_sent

            if not delivery.success:
                result.error = f"delivery_failed: {delivery.error}"
                logger.warning(
                    "Quarterly report delivery failed",
                    extra={**log_extra, "error": delivery.error},
                )
                return result

            # ── Record delivery ───────────────────────────────────────
            await self._review_repo.record_report_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
                report_type="quarterly",
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
                "Quarterly report delivered",
                extra={
                    **log_extra,
                    "parts": result.parts_delivered,
                    "reviews": result.reviews_covered,
                    "health_score": health.score,
                    "verdict": health.verdict,
                },
            )

        except Exception as exc:
            logger.error(
                "Quarterly report failed for business",
                extra={**log_extra, "error": str(exc)},
            )
            result.error = str(exc)

        return result

    # ------------------------------------------------------------------
    # Report builder — all 5 parts for every subscribed business
    # ------------------------------------------------------------------

    async def _build_report_parts(
        self,
        db: AsyncSession,
        business_id: str,
        business,
        this_q: list,
        prev_q: list,
        monthly_breakdown: list[dict],
        health: ReputationHealthScore,
        q_label: str,
        log_extra: dict,
    ) -> list[str]:
        """
        Build all 5 WhatsApp message parts for the quarterly report.

        Each part built independently — failure in one omits that section
        rather than aborting the full report.

        Returns:
            Ordered list of message strings. Empty list on total failure.
        """
        parts: list[str] = []

        # ── Part 1: Quarterly Overview ────────────────────────────────
        try:
            parts.append(_build_overview_part(
                business_name=business.business_name,
                q_label=q_label,
                this_q=this_q,
                prev_q=prev_q,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 1 (overview)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 2: Month-by-Month Breakdown ─────────────────────────
        try:
            parts.append(_build_monthly_breakdown_part(
                monthly_breakdown=monthly_breakdown,
                q_label=q_label,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 2 (month breakdown)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 3: Reputation Health Score ──────────────────────────
        try:
            parts.append(_build_health_score_part(health=health))
        except Exception as exc:
            logger.warning(
                "Failed to build Part 3 (health score)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 4: Competitive Position ─────────────────────────────
        try:
            part4 = await self._build_competitive_position_part(
                db=db,
                business_id=business_id,
                this_q=this_q,
                q_label=q_label,
                log_extra=log_extra,
            )
            if part4:
                parts.append(part4)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 4 (competitive position)",
                extra={**log_extra, "error": str(exc)},
            )

        # ── Part 5: AI Strategic Recommendations ─────────────────────
        try:
            part5 = await self._build_strategy_part(
                business_id=business_id,
                business_name=business.business_name,
                business_type=getattr(business, "business_type", "local business"),
                this_q=this_q,
                health=health,
                monthly_breakdown=monthly_breakdown,
                q_label=q_label,
                log_extra=log_extra,
            )
            if part5:
                parts.append(part5)
        except Exception as exc:
            logger.warning(
                "Failed to build Part 5 (AI strategy)",
                extra={**log_extra, "error": str(exc)},
            )

        return parts

    # ------------------------------------------------------------------
    # Section builders requiring service calls
    # ------------------------------------------------------------------

    async def _build_competitive_position_part(
        self,
        db: AsyncSession,
        business_id: str,
        this_q: list,
        q_label: str,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Build competitive position section from stored competitor snapshots.

        Includes our rank among tracked competitors and the rating gap
        to the top competitor. Returns None if no competitors tracked.
        """
        try:
            snapshots = await self._review_repo.get_recent_competitor_snapshots(
                db=db,
                business_id=business_id,
                limit=10,
            )
            if not snapshots:
                return None

            our_avg = _avg_rating_from_list(this_q)
            our_avg_str = f"{our_avg:.2f}" if our_avg else "N/A"

            # Rank us among all competitors
            all_ratings: list[tuple[str, float]] = []
            if our_avg:
                all_ratings.append(("*Your Business*", our_avg))

            for comp in snapshots:
                r = comp.get("current_rating")
                if r:
                    all_ratings.append((comp["name"][:25], float(r)))

            # Sort descending by rating
            all_ratings.sort(key=lambda x: x[1], reverse=True)
            our_rank = next(
                (i + 1 for i, (name, _) in enumerate(all_ratings)
                 if name == "*Your Business*"),
                None,
            )
            total_in_market = len(all_ratings)

            lines = [
                f"🏆 *Competitive Position — {q_label}*",
                "",
                f"Your Rating:  ⭐ *{our_avg_str}*",
            ]

            if our_rank:
                lines.append(
                    f"Market Rank:  *#{our_rank} of {total_in_market}* "
                    f"tracked businesses"
                )

            lines += ["", "*Competitor Ratings:*"]
            for i, (name, rating) in enumerate(all_ratings[:6], 1):
                marker = " ← You" if name == "*Your Business*" else ""
                lines.append(f"  {i}. {name}: ⭐ {rating:.1f}{marker}")

            # Gap analysis
            if our_avg and all_ratings:
                top_name, top_rating = all_ratings[0]
                if top_name != "*Your Business*":
                    gap = top_rating - our_avg
                    if gap > 0.1:
                        lines += [
                            "",
                            f"📌 *Gap to leader:* {gap:.1f} stars "
                            f"({top_name})",
                        ]
                    else:
                        lines += [
                            "",
                            "🥇 *You lead the market this quarter.*",
                        ]

            # QoQ movement for each competitor
            lines += ["", "*QoQ Movement:*"]
            for comp in snapshots[:5]:
                delta = comp.get("rating_delta")
                if delta is not None:
                    arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
                    lines.append(
                        f"  {arrow} {comp['name'][:25]}: "
                        f"{delta:+.1f} this quarter"
                    )

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "Competitive position build failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None

    async def _build_strategy_part(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        this_q: list,
        health: ReputationHealthScore,
        monthly_breakdown: list[dict],
        q_label: str,
        log_extra: dict,
    ) -> Optional[str]:
        """
        Generate 4-5 forward-looking AI strategic recommendations.

        Combines 90-day review sample with the health score and monthly
        trend summary to produce quarterly strategy — not generic tips,
        but specific actions informed by this business's actual data.

        Token budget controlled via MAX_REVIEWS_FOR_STRATEGY.
        Returns None if AI generation fails.
        """
        try:
            # Curated sample: balanced negative + positive for full context
            negative = [
                r for r in this_q
                if getattr(r, "sentiment", "neutral") in ("negative", "neutral")
            ][:MAX_REVIEWS_FOR_STRATEGY // 2]
            positive = [
                r for r in this_q
                if getattr(r, "sentiment", "neutral") == "positive"
            ][:MAX_REVIEWS_FOR_STRATEGY - len(negative)]
            sample = (negative + positive)[:MAX_REVIEWS_FOR_STRATEGY]

            # Build compact trend context for the AI prompt
            trend_context = _build_quarterly_trend_context(
                health=health,
                monthly_breakdown=monthly_breakdown,
            )

            strategy_result = await self._reports.generate_quarterly_strategy(
                business_id=business_id,
                business_name=business_name,
                business_type=business_type,
                reviews=sample,
                trend_context=trend_context,
                health_score=health.score,
                health_verdict=health.verdict,
                quarter_label=q_label,
            )

            if not strategy_result.success or not strategy_result.recommendations:
                return None

            lines = [
                f"🎯 *Quarterly Strategy — {q_label}*",
                f"_Health Score: {health.score:.0f}/100 — {health.verdict}_",
                "",
                "*Recommended Focus Areas for Next Quarter:*",
                "",
            ]
            for i, rec in enumerate(strategy_result.recommendations[:5], 1):
                lines.append(f"{i}. {rec}")

            lines += [
                "",
                "_AI-generated from 90 days of business data._",
                "_Review with your team and adapt to your context._",
            ]
            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "AI strategy generation failed",
                extra={**log_extra, "error": str(exc)},
            )
            return None


# ==============================================================================
# Pure section builders
# ==============================================================================

def _build_overview_part(
    business_name: str,
    q_label: str,
    this_q: list,
    prev_q: list,
) -> str:
    """
    Part 1: Quarterly overview with QoQ comparison.

    Covers: review volume, average rating, sentiment distribution,
    response rate, and quarter-over-quarter movement for each metric.
    """
    total = len(this_q)
    prev_total = len(prev_q)
    avg = _avg_rating_from_list(this_q)
    prev_avg = _avg_rating_from_list(prev_q)

    # Volume QoQ
    if prev_total > 0:
        vol_delta = total - prev_total
        vol_pct = (vol_delta / prev_total) * 100
        vol_str = (
            f"{'📈' if vol_delta >= 0 else '📉'} "
            f"{'▲' if vol_delta >= 0 else '▼'}{abs(vol_delta)} "
            f"({abs(vol_pct):.0f}% vs last quarter)"
        )
    else:
        vol_str = "📊 First full quarter of data"

    # Rating QoQ
    if avg and prev_avg:
        delta = avg - prev_avg
        if abs(delta) < 0.05:
            rating_qoq = "➡️ Steady"
        elif delta > 0:
            rating_qoq = f"📈 Up {delta:+.2f} vs last quarter"
        else:
            rating_qoq = f"📉 Down {delta:+.2f} vs last quarter"
    else:
        rating_qoq = "📊 No prior quarter data"

    # Sentiment distribution
    sentiment_counts = _sentiment_counts(this_q)
    pos_pct = _pct(sentiment_counts["positive"], total)
    neg_pct = _pct(sentiment_counts["negative"], total)
    neu_pct = _pct(sentiment_counts["neutral"], total)

    # Response rate
    replied = sum(1 for r in this_q if getattr(r, "has_reply", False))
    response_rate = (replied / total * 100) if total > 0 else 0.0

    # Star breakdown
    star_counts = {i: 0 for i in range(1, 6)}
    for r in this_q:
        rating = getattr(r, "rating", 0)
        if 1 <= rating <= 5:
            star_counts[rating] += 1

    lines = [
        f"📊 *Quarterly Report — {business_name}*",
        f"_{q_label}  (90-day performance summary)_",
        "",
        "📋 *Quarter Overview*",
        "",
        f"Total Reviews:    *{total}*   {vol_str}",
        (
            f"Average Rating:   *{avg:.2f}/5.0*   {rating_qoq}"
            if avg else "Average Rating:   *N/A*"
        ),
        f"Response Rate:    *{response_rate:.0f}%* ({replied}/{total} replied)",
        "",
        "*Sentiment:*",
        f"  😊 Positive:  {pos_pct:.0f}%  "
        f"😐 Neutral: {neu_pct:.0f}%  "
        f"😞 Negative: {neg_pct:.0f}%",
        "",
        "*Rating Breakdown:*",
    ]
    for stars in range(5, 0, -1):
        count = star_counts[stars]
        bar = format_star_bar(count, total)
        lines.append(f"  {'⭐' * stars}  {bar} ({count})")

    return "\n".join(lines)


def _build_monthly_breakdown_part(
    monthly_breakdown: list[dict],
    q_label: str,
) -> str:
    """
    Part 2: Month-by-month performance within the quarter.

    Shows volume, average rating, and direction for each of the
    3 months. Identifies the best and worst performing month.
    Includes momentum verdict (accelerating / decelerating / stable).
    """
    if not monthly_breakdown:
        return (
            f"📅 *Month-by-Month Breakdown — {q_label}*\n\n"
            "_No monthly breakdown data available._"
        )

    lines = [f"📅 *Month-by-Month — {q_label}*", ""]

    prev_avg: Optional[float] = None
    for month in monthly_breakdown:
        label = month.get("label", "Unknown")
        count = month.get("count", 0)
        avg = month.get("avg_rating")

        arrow = "•"
        if prev_avg is not None and avg is not None:
            delta = avg - prev_avg
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")

        avg_str = f"{avg:.1f}" if avg else "N/A"
        lines.append(f"  {label}:  ⭐ {avg_str}  {arrow}  ({count} reviews)")
        if avg is not None:
            prev_avg = avg

    # Best and worst month
    months_with_data = [m for m in monthly_breakdown if m.get("avg_rating")]
    if months_with_data:
        best = max(months_with_data, key=lambda m: m["avg_rating"])
        worst = min(months_with_data, key=lambda m: m["avg_rating"])

        lines += [""]
        if best["label"] != worst["label"]:
            lines.append(f"🏆 Best month:   *{best['label']}* (⭐ {best['avg_rating']:.1f})")
            lines.append(f"📉 Toughest month: *{worst['label']}* (⭐ {worst['avg_rating']:.1f})")

    # Momentum verdict
    ratings = [m["avg_rating"] for m in monthly_breakdown if m.get("avg_rating")]
    if len(ratings) >= 2:
        overall = ratings[-1] - ratings[0]
        if overall > 0.2:
            verdict = "📈 *Momentum: Accelerating* — steady improvement across the quarter."
        elif overall < -0.2:
            verdict = "📉 *Momentum: Decelerating* — rating declined across the quarter."
        else:
            verdict = "➡️ *Momentum: Stable* — consistent performance across all 3 months."
        lines += ["", verdict]

    return "\n".join(lines)


def _build_health_score_part(health: ReputationHealthScore) -> str:
    """
    Part 3: Reputation Health Score with component breakdown.

    The composite score (0–100) is the most important single number
    in the quarterly report. It gives the business owner one clear
    signal: how healthy is their reputation right now?
    """
    score = health.score
    verdict = health.verdict

    # Visual score bar
    filled = round(score / 5)       # 20 segments representing 0–100
    bar = "█" * filled + "░" * (20 - filled)

    # Verdict emoji
    verdict_emoji = {
        "Excellent":       "🟢",
        "Good":            "🟡",
        "Needs Attention": "🟠",
        "Critical":        "🔴",
    }.get(verdict, "⚪")

    lines = [
        "💯 *Reputation Health Score*",
        "",
        f"  Score:   *{score:.0f} / 100*",
        f"  [{bar}]",
        f"  Verdict: {verdict_emoji} *{verdict}*",
    ]

    # QoQ change
    if health.delta is not None:
        direction = "▲" if health.delta > 0 else ("▼" if health.delta < 0 else "→")
        lines.append(
            f"  QoQ Change: {direction} {abs(health.delta):.0f} points "
            f"vs last quarter"
        )

    lines += [
        "",
        "*Score Breakdown:*",
        f"  ⭐ Average Rating:      {health.avg_rating_score:.0f}/100",
        f"  😊 Positive Sentiment:  {health.positive_ratio_score:.0f}/100",
        f"  💬 Response Rate:       {health.response_rate_score:.0f}/100",
        f"  📈 Volume Growth:       {health.volume_growth_score:.0f}/100",
        "",
        f"  _(Weights: rating {int(WEIGHT_AVG_RATING * 100)}% · "
        f"sentiment {int(WEIGHT_POSITIVE_RATIO * 100)}% · "
        f"response {int(WEIGHT_RESPONSE_RATE * 100)}% · "
        f"growth {int(WEIGHT_VOLUME_GROWTH * 100)}%)_",
    ]

    # Actionable verdict context
    if verdict == "Excellent":
        lines += ["", "🎉 Outstanding quarter. Focus on maintaining this level."]
    elif verdict == "Good":
        lines += ["", "👍 Solid performance. Small improvements can push you to Excellent."]
    elif verdict == "Needs Attention":
        lines += ["", "⚠️ Action needed. Review the strategy section for improvement areas."]
    else:
        lines += ["", "🚨 Immediate attention required. Prioritise negative review response."]

    return "\n".join(lines)


# ==============================================================================
# Helpers
# ==============================================================================

def _build_monthly_breakdown(
    reviews: list,
    q_start: date,
) -> list[dict]:
    """
    Build month-by-month stats for the 3 months of the quarter.

    Args:
        reviews:  All reviews in the quarter.
        q_start:  First day of the quarter.

    Returns:
        List of 3 dicts: label, count, avg_rating.
    """
    breakdown = []
    for i in range(3):
        month_start, month_end = get_month_date_range(
            reference_date=date(q_start.year, q_start.month, 1),
            offset_months=i,
        )
        month_reviews = [
            r for r in reviews
            if _review_in_range(r, month_start, month_end)
        ]
        breakdown.append({
            "label": month_label(month_start),
            "count": len(month_reviews),
            "avg_rating": _avg_rating_from_list(month_reviews),
        })
    return breakdown


def _compute_health_score(
    this_q: list,
    prev_q: list,
) -> ReputationHealthScore:
    """
    Compute the Reputation Health Score for this quarter.

    Four weighted components:
      1. Average rating → normalised to 0–100 (5.0 stars = 100)
      2. Positive sentiment ratio → 0–100 (100% positive = 100)
      3. Response rate → 0–100 (100% replied = 100)
      4. Volume growth → 0–100 (≥50% growth = 100, ≤-50% = 0)

    Previous quarter score computed with same formula for QoQ delta.
    """
    def _score_components(reviews: list, prev_reviews: list) -> tuple[float, float, float, float]:
        total = max(len(reviews), 1)
        prev_total = max(len(prev_reviews), 1)

        # Component 1: average rating
        avg = _avg_rating_from_list(reviews) or 0.0
        avg_score = (avg / 5.0) * 100

        # Component 2: positive sentiment ratio
        pos_count = sum(
            1 for r in reviews
            if getattr(r, "sentiment", "neutral") == "positive"
        )
        pos_ratio_score = (pos_count / total) * 100

        # Component 3: response rate
        replied = sum(1 for r in reviews if getattr(r, "has_reply", False))
        response_score = (replied / total) * 100

        # Component 4: volume growth
        growth_pct = ((len(reviews) - len(prev_reviews)) / prev_total) * 100
        # Clamp to [-50, +50] then map to [0, 100]
        clamped = max(-50.0, min(50.0, growth_pct))
        volume_growth_score = ((clamped + 50.0) / 100.0) * 100

        return avg_score, pos_ratio_score, response_score, volume_growth_score

    avg_s, pos_s, resp_s, vol_s = _score_components(this_q, prev_q)
    score = (
        avg_s * WEIGHT_AVG_RATING
        + pos_s * WEIGHT_POSITIVE_RATIO
        + resp_s * WEIGHT_RESPONSE_RATE
        + vol_s * WEIGHT_VOLUME_GROWTH
    )

    # Previous quarter score for delta (use empty list for prev-prev)
    prev_avg_s, prev_pos_s, prev_resp_s, prev_vol_s = _score_components(prev_q, [])
    prev_score = (
        prev_avg_s * WEIGHT_AVG_RATING
        + prev_pos_s * WEIGHT_POSITIVE_RATIO
        + prev_resp_s * WEIGHT_RESPONSE_RATE
        + prev_vol_s * WEIGHT_VOLUME_GROWTH
    ) if prev_q else None

    delta = round(score - prev_score, 1) if prev_score is not None else None

    # Verdict
    if score >= HEALTH_EXCELLENT:
        verdict = "Excellent"
    elif score >= HEALTH_GOOD:
        verdict = "Good"
    elif score >= HEALTH_NEEDS_ATTENTION:
        verdict = "Needs Attention"
    else:
        verdict = "Critical"

    return ReputationHealthScore(
        score=round(score, 1),
        prev_score=round(prev_score, 1) if prev_score is not None else None,
        avg_rating_score=round(avg_s, 1),
        positive_ratio_score=round(pos_s, 1),
        response_rate_score=round(resp_s, 1),
        volume_growth_score=round(vol_s, 1),
        verdict=verdict,
        delta=delta,
    )


def _build_quarterly_trend_context(
    health: ReputationHealthScore,
    monthly_breakdown: list[dict],
) -> str:
    """
    Compact text summary of quarterly trend for the AI strategy prompt.

    Gives the AI structured context without sending raw Python objects.
    """
    parts = [
        f"Health score: {health.score:.0f}/100 ({health.verdict})",
    ]
    if health.delta is not None:
        direction = "up" if health.delta > 0 else "down"
        parts.append(f"QoQ: {direction} {abs(health.delta):.0f} points")

    for m in monthly_breakdown:
        avg = m.get("avg_rating")
        avg_str = f"{avg:.1f}" if avg else "no data"
        parts.append(
            f"{m.get('label', '')}: avg {avg_str} ({m.get('count', 0)} reviews)"
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


def _sentiment_counts(reviews: list) -> dict[str, int]:
    """Count reviews by sentiment label."""
    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    for r in reviews:
        s = getattr(r, "sentiment", "neutral") or "neutral"
        if s in counts:
            counts[s] += 1
    return counts


def _pct(count: int, total: int) -> float:
    """Compute percentage safely."""
    return round((count / total) * 100, 1) if total > 0 else 0.0


def _review_in_range(review, start: date, end: date) -> bool:
    """Return True if review date falls within [start, end] inclusive."""
    review_date = getattr(review, "review_date", None)
    if not review_date:
        return False
    if hasattr(review_date, "date"):
        review_date = review_date.date()
    return start <= review_date <= end


# ==============================================================================
# Functional entry point (called by scheduler_manager.py)
# ==============================================================================

async def run_quarterly_report(
    db: AsyncSession,
    business_repo: BusinessRepository,
    subscription_repo: SubscriptionRepository,
    admin_notification: AdminNotificationService,
    **kwargs,
) -> QuarterlyReportJobResult:
    """
    Functional entry point for the quarterly report job.

    Called by scheduler_manager.py on 1st Jan, Apr, Jul, Oct at 09:00 UTC.

    Returns:
        QuarterlyReportJobResult. Never raises.
    """
    job = QuarterlyReportJob(
        review_repo=kwargs["review_repo"],
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        reports_service=kwargs["reports_service"],
        whatsapp_service=kwargs["whatsapp_service"],
        usage_tracker=kwargs["usage_tracker"],
        admin_notification=admin_notification,
    )
    return await job.run(db=db)