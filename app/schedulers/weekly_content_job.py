# ==============================================================================
# File: app/schedulers/weekly_content_job.py
# Purpose: Monday morning scheduled job (07:00 UTC) that generates 3-5
#          ready-to-post social media content pieces for every active
#          subscribed business and delivers them via WhatsApp.
#
#          Schedule: Every Monday at 07:00 UTC (registered by scheduler_manager)
#          Runs 1 hour after weekly_report_job to stagger OpenAI API load.
#
#          Every subscribed business receives social media content —
#          there is no plan gating. Subscription = full access.
#
#          Content generated per business:
#            - 3 to 5 social media posts tailored to:
#                • Business type and name
#                • Recent review sentiment and themes (14-day window)
#                • Current week context
#            - Each post includes:
#                • Post body text (Instagram / Facebook / Google Posts ready)
#                • Suggested hashtags
#                • Post type label (promotional / testimonial / tips / engagement)
#
#          Processing pipeline per business:
#            1. Verify active subscription
#            2. Idempotency check — skip if content already sent this week
#            3. Fetch last 14 days of reviews for sentiment context
#            4. Generate content via content_generation_service (one AI call)
#            5. Format posts for WhatsApp delivery
#            6. Deliver via WhatsApp as multi-part message
#            7. Record delivery for idempotency
#            8. Increment usage counter
#
#          Performance contract:
#            - Businesses processed in batches of BUSINESS_BATCH_SIZE (10)
#            - One OpenAI call per business (all posts in one batch call)
#            - Max reviews sent to AI: MAX_REVIEWS_FOR_CONTEXT (10)
#            - Each business isolated — failure never blocks others
#            - Scheduler-level lock managed by scheduler_manager.py
#
#          Idempotency:
#            Key: WEEKLY_CONTENT_{business_id}_{YYYY_Www}
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
from app.services.content_generation_service import ContentGenerationService
from app.utils.batch_utils import process_in_batches
from app.utils.idempotency_utils import make_weekly_content_key
from app.utils.time_utils import (
    get_week_date_range,
    iso_week_label,
    today_local,
)
from app.utils.usage_tracker import UsageTracker

logger = logging.getLogger(ServiceName.WEEKLY_CONTENT)

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
BUSINESS_BATCH_SIZE: int = 10
MAX_REVIEWS_FOR_CONTEXT: int = 10
MIN_POSTS_TO_DELIVER: int = 1
POSTS_PER_WEEK: int = 5


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class BusinessContentResult:
    """Result of generating and delivering weekly content for one business."""
    business_id: str
    posts_generated: int = 0
    parts_delivered: int = 0
    week_label: str = ""
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WeeklyContentJobResult:
    """Aggregate result of the full weekly content generation run."""
    businesses_served: int = 0
    businesses_skipped: int = 0
    businesses_errored: int = 0
    total_posts: int = 0
    total_parts_sent: int = 0
    week_label: str = ""

    def merge(self, biz: BusinessContentResult) -> None:
        if biz.skipped:
            self.businesses_skipped += 1
            return
        if biz.error and biz.posts_generated == 0:
            self.businesses_errored += 1
            return
        self.businesses_served += 1
        self.total_posts += biz.posts_generated
        self.total_parts_sent += biz.parts_delivered


# ==============================================================================
# Weekly Content Job
# ==============================================================================

class WeeklyContentJob:
    """
    Generates and delivers weekly social media content to all subscribed businesses.

    No plan gating — every subscribed business receives personalised
    social media posts each Monday.

    Content is personalised using:
      - Recent review sentiment (last 14 days)
      - Business type and name
      - Current week context
    """

    def __init__(
        self,
        review_repo: ReviewRepository,
        business_repo: BusinessRepository,
        subscription_repo: SubscriptionRepository,
        content_service: ContentGenerationService,
        whatsapp_service: WhatsAppService,
        usage_tracker: UsageTracker,
        admin_notification: AdminNotificationService,
    ) -> None:
        self._review_repo = review_repo
        self._biz_repo = business_repo
        self._sub_repo = subscription_repo
        self._content = content_service
        self._whatsapp = whatsapp_service
        self._usage_tracker = usage_tracker
        self._admin = admin_notification

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, db: AsyncSession) -> WeeklyContentJobResult:
        """
        Execute the weekly content generation cycle for all active businesses.

        Returns:
            WeeklyContentJobResult. Never raises.
        """
        today = today_local()
        week_start, week_end = get_week_date_range(
            reference_date=today, offset_weeks=0
        )
        week_label = iso_week_label(week_start, week_end)

        aggregate = WeeklyContentJobResult(week_label=week_label)
        log_extra = {
            "service": ServiceName.WEEKLY_CONTENT,
            "week": week_label,
        }

        logger.info("Weekly content job started", extra=log_extra)

        try:
            active_ids = await self._sub_repo.get_active_business_ids(db=db)

            if not active_ids:
                logger.info(
                    "No active businesses — skipping weekly content",
                    extra=log_extra,
                )
                return aggregate

            logger.info(
                "Generating content for %d businesses",
                len(active_ids),
                extra=log_extra,
            )

            async def process_batch(batch: list[str]) -> None:
                for business_id in batch:
                    result = await self._process_business(
                        db=db,
                        business_id=business_id,
                        week_start=week_start,
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
                "Weekly content job failed at top level",
                extra={**log_extra, "error": str(exc)},
            )
            await self._admin.send_job_failure(
                job_name="weekly_content_job",
                error=str(exc),
            )

        logger.info(
            "Weekly content job complete",
            extra={
                **log_extra,
                "served": aggregate.businesses_served,
                "skipped": aggregate.businesses_skipped,
                "errored": aggregate.businesses_errored,
                "posts": aggregate.total_posts,
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
        week_label: str,
    ) -> BusinessContentResult:
        """Generate and deliver social media content for a single business."""
        result = BusinessContentResult(
            business_id=business_id,
            week_label=week_label,
        )
        log_extra = {
            "service": ServiceName.WEEKLY_CONTENT,
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
            idempotency_key = make_weekly_content_key(
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

            # ── Fetch recent reviews for context ──────────────────────
            context_window_start = today_local() - timedelta(days=14)
            recent_reviews = await self._review_repo.get_reviews_in_range(
                db=db,
                business_id=business_id,
                start_date=context_window_start,
                end_date=today_local(),
            )

            context_reviews = _select_context_reviews(
                reviews=recent_reviews,
                max_count=MAX_REVIEWS_FOR_CONTEXT,
            )

            # ── Generate content ──────────────────────────────────────
            generation_result = await self._content.generate_weekly_posts(
                business_id=business_id,
                business_name=business.business_name,
                business_type=getattr(business, "business_type", "local business"),
                recent_reviews=context_reviews,
                post_count=POSTS_PER_WEEK,
                week_label=week_label,
            )

            if not generation_result.success:
                result.error = f"generation_failed: {generation_result.error}"
                return result

            posts = generation_result.posts or []
            if len(posts) < MIN_POSTS_TO_DELIVER:
                result.skipped = True
                result.skip_reason = (
                    f"insufficient_posts_generated: {len(posts)}"
                )
                return result

            result.posts_generated = len(posts)

            # ── Format for WhatsApp ───────────────────────────────────
            parts = _format_content_for_whatsapp(
                business_name=business.business_name,
                week_label=week_label,
                posts=posts,
            )

            # ── Deliver via WhatsApp ──────────────────────────────────
            delivery = await self._whatsapp.send_report(
                db=db,
                business_id=business_id,
                report_parts=parts,
                report_type="weekly_content",
            )

            result.parts_delivered = delivery.parts_sent

            if not delivery.success:
                result.error = f"delivery_failed: {delivery.error}"
                logger.warning(
                    "Weekly content delivery failed",
                    extra={**log_extra, "error": delivery.error},
                )
                return result

            # ── Record delivery ───────────────────────────────────────
            await self._review_repo.record_report_sent(
                db=db,
                business_id=business_id,
                idempotency_key=idempotency_key,
                report_type="weekly_content",
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
                "Weekly content delivered",
                extra={
                    **log_extra,
                    "posts": result.posts_generated,
                    "parts": result.parts_delivered,
                },
            )

        except Exception as exc:
            logger.error(
                "Weekly content failed for business",
                extra={**log_extra, "error": str(exc)},
            )
            result.error = str(exc)

        return result


# ==============================================================================
# Pure helpers
# ==============================================================================

def _select_context_reviews(reviews: list, max_count: int) -> list:
    """
    Select the most contextually useful reviews for AI content generation.

    Takes a 50/50 split of negative/neutral and positive reviews to give
    the AI both the challenges (themes to address) and wins (language to
    highlight). Never exceeds max_count for token budget control.
    """
    negative = [
        r for r in reviews
        if getattr(r, "sentiment", "neutral") in ("negative", "neutral")
    ]
    positive = [
        r for r in reviews
        if getattr(r, "sentiment", "neutral") == "positive"
    ]
    neg_budget = max_count // 2
    pos_budget = max_count - neg_budget
    return (negative[:neg_budget] + positive[:pos_budget])[:max_count]


def _format_content_for_whatsapp(
    business_name: str,
    week_label: str,
    posts: list,
) -> list[str]:
    """
    Format generated posts as WhatsApp message parts.

    Header message + one message per post for easy copy-paste.
    Each post includes type label, body, and hashtags.
    """
    parts: list[str] = []

    parts.append(
        f"✍️ *Your Weekly Content — {business_name}*\n"
        f"_{week_label}_\n\n"
        f"Here are *{len(posts)} ready-to-post* social media ideas for this week.\n"
        f"Copy, customise, and post on Instagram, Facebook, or Google Posts!\n\n"
        f"_Scroll down for all {len(posts)} posts_ 👇"
    )

    _POST_TYPE_EMOJI = {
        "promotional":   "🛍️",
        "testimonial":   "⭐",
        "tips":          "💡",
        "engagement":    "💬",
        "announcement":  "📢",
        "seasonal":      "🌟",
    }

    for i, post in enumerate(posts, 1):
        post_type = getattr(post, "post_type", "promotional")
        body = getattr(post, "body", "")
        hashtags = getattr(post, "hashtags", [])
        emoji = _POST_TYPE_EMOJI.get(post_type, "📝")

        lines = [
            f"{emoji} *Post {i} of {len(posts)}*  [{post_type.title()}]",
            "",
            body,
        ]
        if hashtags:
            lines += ["", " ".join(f"#{t.lstrip('#')}" for t in hashtags[:8])]
        lines += ["", "─────────────────────", "_Tap to copy and post_"]
        parts.append("\n".join(lines))

    return parts


# ==============================================================================
# Functional entry point
# ==============================================================================

async def run_weekly_content(
    db: AsyncSession,
    business_repo: BusinessRepository,
    subscription_repo: SubscriptionRepository,
    admin_notification: AdminNotificationService,
    **kwargs,
) -> WeeklyContentJobResult:
    """Functional entry point called by scheduler_manager.py."""
    job = WeeklyContentJob(
        review_repo=kwargs["review_repo"],
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        content_service=kwargs["content_service"],
        whatsapp_service=kwargs["whatsapp_service"],
        usage_tracker=kwargs["usage_tracker"],
        admin_notification=admin_notification,
    )
    return await job.run(db=db)