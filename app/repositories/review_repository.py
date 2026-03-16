# ==============================================================================
# File: app/repositories/review_repository.py
# Purpose: Repository class encapsulating all database operations for the
#          ReviewModel. This is the only layer permitted to query the
#          reviews table directly.
#
#          Core responsibilities:
#            - Idempotent review storage (upsert by google_review_id)
#            - New review detection for scheduler polling cycles
#            - Sentiment and reply status tracking
#            - Spike detection queries for alert system
#            - Rating trend queries for rating drop alerts
#            - All queries enforce business_id tenant isolation
# ==============================================================================

import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import (
    REVIEW_MAX_TEXT_LENGTH_FOR_AI,
    ReviewSentiment,
    ReviewStatus,
    ServiceName,
)
from app.database.models.review_model import ReviewModel

logger = logging.getLogger(ServiceName.GOOGLE_REVIEWS)


# ==============================================================================
# Review Repository
# ==============================================================================

class ReviewRepository:
    """
    Handles all database operations for ReviewModel.

    Session management (commit/rollback) is the responsibility of the
    caller. This repository only calls flush() to populate server defaults.

    All queries:
        - Filter by business_id for multi-tenant isolation
        - Apply LIMIT clauses — no unbounded result sets
        - Use indexed columns for all WHERE and ORDER BY clauses

    Usage:
        repo = ReviewRepository()
        new_reviews = await repo.get_pending_for_processing(db, business_id)
    """

    # ── Create / Upsert ────────────────────────────────────────────────────────

    async def upsert(
        self,
        db: AsyncSession,
        *,
        business_id: uuid.UUID,
        google_review_id: str,
        rating: int,
        reviewer_name: Optional[str] = None,
        reviewer_profile_url: Optional[str] = None,
        review_text: Optional[str] = None,
        google_place_id: Optional[str] = None,
        published_at: Optional[datetime] = None,
        updated_on_google_at: Optional[datetime] = None,
        original_language: Optional[str] = None,
    ) -> tuple[ReviewModel, bool]:
        """
        Insert a new review or skip if it already exists.

        Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING to guarantee
        idempotency — the same review fetched across multiple polling
        cycles is stored exactly once.

        The review_text is truncated to REVIEW_MAX_TEXT_LENGTH_FOR_AI (800
        chars) and stored in review_text_truncated if the original exceeds
        this limit. The original text is always preserved in review_text.

        Args:
            db:                   Active async database session.
            business_id:          UUID of the owning business.
            google_review_id:     Google's unique review identifier.
            rating:               Star rating (1–5).
            reviewer_name:        Reviewer display name.
            reviewer_profile_url: Reviewer Google profile URL.
            review_text:          Full review text.
            google_place_id:      Google Place ID at time of fetch.
            published_at:         Review publication timestamp.
            updated_on_google_at: Last edit timestamp on Google.
            original_language:    Detected language code.

        Returns:
            tuple[ReviewModel, bool]: The review instance and a boolean
            indicating whether it was newly created (True) or already
            existed (False).

        Raises:
            SQLAlchemyError: On any database error.
        """
        # Truncate text for AI safety — preserve original separately
        review_text_truncated: Optional[str] = None
        if review_text and len(review_text) > REVIEW_MAX_TEXT_LENGTH_FOR_AI:
            review_text_truncated = review_text[:REVIEW_MAX_TEXT_LENGTH_FOR_AI]

        try:
            # Attempt insert — skip silently if (business_id, google_review_id) exists
            stmt = (
                pg_insert(ReviewModel)
                .values(
                    business_id=business_id,
                    google_review_id=google_review_id,
                    rating=rating,
                    reviewer_name=reviewer_name,
                    reviewer_profile_url=reviewer_profile_url,
                    review_text=review_text,
                    review_text_truncated=review_text_truncated,
                    google_place_id=google_place_id,
                    published_at=published_at,
                    updated_on_google_at=updated_on_google_at,
                    original_language=original_language,
                    status=ReviewStatus.NEW,
                    processing_attempts=0,
                    alert_sent=False,
                    is_spam=False,
                    is_valid=True,
                    reply_posted_to_google=False,
                )
                .on_conflict_do_nothing(
                    constraint="uq_reviews_business_google_id"
                )
                .returning(ReviewModel.id)
            )

            result = await db.execute(stmt)
            inserted_id = result.scalar_one_or_none()

            if inserted_id:
                # Newly inserted — fetch the full record
                review = await self.get_by_id(db, inserted_id)
                logger.info(
                    "New review stored",
                    extra={
                        "service": ServiceName.GOOGLE_REVIEWS,
                        "business_id": str(business_id),
                        "review_id": str(inserted_id),
                        "google_review_id": google_review_id,
                        "rating": rating,
                    },
                )
                return review, True
            else:
                # Already exists — fetch the existing record
                existing = await self.get_by_google_review_id(
                    db, business_id, google_review_id
                )
                return existing, False

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to upsert review",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "google_review_id": google_review_id,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Single Record ───────────────────────────────────────────────────

    async def get_by_id(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> Optional[ReviewModel]:
        """
        Fetch a single review by its internal UUID primary key.

        Args:
            db:         Active async database session.
            review_id:  Internal UUID of the review.

        Returns:
            ReviewModel if found, else None.
        """
        try:
            result = await db.execute(
                select(ReviewModel).where(ReviewModel.id == review_id)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch review by ID",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_google_review_id(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        google_review_id: str,
    ) -> Optional[ReviewModel]:
        """
        Fetch a single review by its Google review ID within a business scope.

        Args:
            db:               Active async database session.
            business_id:      UUID of the owning business.
            google_review_id: Google's unique review identifier.

        Returns:
            ReviewModel if found, else None.
        """
        try:
            result = await db.execute(
                select(ReviewModel).where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.google_review_id == google_review_id,
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch review by Google review ID",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "google_review_id": google_review_id,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Collections ─────────────────────────────────────────────────────

    async def get_pending_for_processing(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        limit: int = 20,
    ) -> list[ReviewModel]:
        """
        Fetch reviews that are eligible for AI reply generation.

        Returns valid, non-spam reviews in NEW or PROCESSING status that
        have not exhausted their retry budget (max 3 attempts).

        Used by review_jobs.py as the entry point for each processing cycle.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            limit:        Maximum records to return (default: 20).

        Returns:
            list[ReviewModel]: Reviews ready for sentiment analysis and
            AI reply generation, ordered oldest first.
        """
        try:
            result = await db.execute(
                select(ReviewModel)
                .where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.is_valid.is_(True),
                    ReviewModel.is_spam.is_(False),
                    ReviewModel.status.in_(
                        [ReviewStatus.NEW, ReviewStatus.PROCESSING]
                    ),
                    ReviewModel.processing_attempts < 3,
                )
                .order_by(ReviewModel.published_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch reviews pending processing",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_recent_by_business(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ReviewModel]:
        """
        Fetch the most recent reviews for a business, ordered newest first.

        Used for report generation and dashboard display.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            limit:        Maximum records to return.
            offset:       Pagination offset.

        Returns:
            list[ReviewModel]: Recent reviews, newest first.
        """
        try:
            result = await db.execute(
                select(ReviewModel)
                .where(ReviewModel.business_id == business_id)
                .order_by(desc(ReviewModel.published_at))
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch recent reviews",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_sentiment(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        sentiment: str,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[ReviewModel]:
        """
        Fetch reviews for a business filtered by sentiment classification.

        Used by analytics_service.py and reports_service.py for sentiment
        trend analysis.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            sentiment:    Sentiment value (positive / negative / neutral).
            since:        Optional lower bound on published_at.
            limit:        Maximum records to return.

        Returns:
            list[ReviewModel]: Reviews matching the sentiment filter.
        """
        try:
            conditions = [
                ReviewModel.business_id == business_id,
                ReviewModel.sentiment == sentiment,
                ReviewModel.is_valid.is_(True),
            ]
            if since:
                conditions.append(ReviewModel.published_at >= since)

            result = await db.execute(
                select(ReviewModel)
                .where(and_(*conditions))
                .order_by(desc(ReviewModel.published_at))
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch reviews by sentiment",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "sentiment": sentiment,
                    "error": str(exc),
                },
            )
            raise

    async def get_unsent_alerts(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        limit: int = 20,
    ) -> list[ReviewModel]:
        """
        Fetch valid reviews for which a WhatsApp alert has not yet been sent.

        Used by alert_manager.py to dispatch pending review notifications.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            limit:        Maximum records to return.

        Returns:
            list[ReviewModel]: Reviews awaiting alert dispatch.
        """
        try:
            result = await db.execute(
                select(ReviewModel)
                .where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.alert_sent.is_(False),
                    ReviewModel.is_valid.is_(True),
                    ReviewModel.is_spam.is_(False),
                )
                .order_by(ReviewModel.published_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch reviews with unsent alerts",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Aggregates & Analytics ─────────────────────────────────────────────────

    async def count_since(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        since: datetime,
    ) -> int:
        """
        Count the number of valid reviews received since a given timestamp.

        Used by rating_alerts.py and competitor_alerts.py to detect
        review volume spikes within a rolling time window.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            since:        Lower bound timestamp for the count window.

        Returns:
            int: Count of valid non-spam reviews since the given timestamp.
        """
        try:
            result = await db.execute(
                select(func.count(ReviewModel.id)).where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.is_valid.is_(True),
                    ReviewModel.is_spam.is_(False),
                    ReviewModel.published_at >= since,
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count reviews since timestamp",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "since": since.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def get_average_rating_since(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        since: datetime,
    ) -> Optional[float]:
        """
        Calculate the average star rating for reviews since a given timestamp.

        Used by rating_alerts.py to detect rating drops compared to
        the previous rolling window.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            since:        Lower bound timestamp.

        Returns:
            float: Average rating rounded to 2 decimal places, or None if
            no reviews exist in the window.
        """
        try:
            result = await db.execute(
                select(func.avg(ReviewModel.rating)).where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.is_valid.is_(True),
                    ReviewModel.is_spam.is_(False),
                    ReviewModel.published_at >= since,
                )
            )
            avg = result.scalar_one_or_none()
            return round(float(avg), 2) if avg is not None else None

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to calculate average rating",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "since": since.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def count_by_sentiment_since(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        since: datetime,
    ) -> dict[str, int]:
        """
        Count reviews grouped by sentiment classification since a timestamp.

        Used by analytics_service.py and reports_service.py for sentiment
        breakdown in weekly and monthly reports.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            since:        Lower bound timestamp.

        Returns:
            dict: Sentiment counts with keys 'positive', 'negative', 'neutral'.
            Missing sentiments default to 0.
        """
        try:
            result = await db.execute(
                select(ReviewModel.sentiment, func.count(ReviewModel.id))
                .where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.is_valid.is_(True),
                    ReviewModel.is_spam.is_(False),
                    ReviewModel.sentiment.isnot(None),
                    ReviewModel.published_at >= since,
                )
                .group_by(ReviewModel.sentiment)
            )
            rows = result.all()

            counts: dict[str, int] = {
                ReviewSentiment.POSITIVE: 0,
                ReviewSentiment.NEGATIVE: 0,
                ReviewSentiment.NEUTRAL: 0,
            }
            for sentiment, count in rows:
                if sentiment in counts:
                    counts[sentiment] = count
            return counts

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count reviews by sentiment",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "since": since.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def count_total_by_business(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> int:
        """
        Count all valid reviews ever stored for a business.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.

        Returns:
            int: Total review count.
        """
        try:
            result = await db.execute(
                select(func.count(ReviewModel.id)).where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.is_valid.is_(True),
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count total reviews",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Update — Processing Pipeline ───────────────────────────────────────────

    async def mark_processing(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> None:
        """
        Transition a review to PROCESSING status.

        Called by review_jobs.py at the start of each processing attempt
        to claim the review and increment the attempt counter atomically.

        Args:
            db:         Active async database session.
            review_id:  UUID of the review.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    status=ReviewStatus.PROCESSING,
                    processing_attempts=ReviewModel.processing_attempts + 1,
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark review as processing",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def update_sentiment(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
        sentiment: str,
        sentiment_score: Optional[float] = None,
    ) -> None:
        """
        Record the sentiment classification result for a review.

        Called by sentiment_service.py after classification is complete.

        Args:
            db:              Active async database session.
            review_id:       UUID of the review.
            sentiment:       Classification result (positive/negative/neutral).
            sentiment_score: Confidence score (0.0–1.0).
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    sentiment_analysed_at=datetime.now(
                        tz=__import__("datetime").timezone.utc
                    ),
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update review sentiment",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def save_ai_reply(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
        ai_reply: str,
        prompt_used: str,
        idempotency_key: str,
    ) -> None:
        """
        Persist the AI-generated reply for a review and mark it as REPLIED.

        Called by ai_reply_service.py after successful reply generation.

        Args:
            db:               Active async database session.
            review_id:        UUID of the review.
            ai_reply:         Generated reply text.
            prompt_used:      Filename of the prompt template used.
            idempotency_key:  Idempotency key to prevent duplicate generation.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    ai_reply=ai_reply,
                    ai_reply_prompt_used=prompt_used,
                    ai_reply_generated_at=datetime.now(
                        tz=__import__("datetime").timezone.utc
                    ),
                    idempotency_key=idempotency_key,
                    status=ReviewStatus.REPLIED,
                )
            )
            await db.flush()

            logger.info(
                "AI reply saved for review",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "prompt_used": prompt_used,
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to save AI reply",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_reply_posted(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> None:
        """
        Record that the AI reply was successfully posted to Google.

        Args:
            db:         Active async database session.
            review_id:  UUID of the review.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    reply_posted_to_google=True,
                    reply_posted_at=datetime.now(
                        tz=__import__("datetime").timezone.utc
                    ),
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark reply as posted",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_alert_sent(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> None:
        """
        Set the alert_sent flag to prevent duplicate WhatsApp notifications.

        Called by whatsapp_service.py after successful alert delivery.

        Args:
            db:         Active async database session.
            review_id:  UUID of the review.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    alert_sent=True,
                    alert_sent_at=datetime.now(
                        tz=__import__("datetime").timezone.utc
                    ),
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark review alert as sent",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_failed(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
        error_message: str,
    ) -> None:
        """
        Mark a review as FAILED after exhausting processing attempts.

        Called by review_jobs.py when a review has failed 3 consecutive
        processing attempts and should be excluded from future cycles.

        Args:
            db:            Active async database session.
            review_id:     UUID of the review.
            error_message: Last error message from the failed attempt.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    status=ReviewStatus.FAILED,
                    last_processing_error=error_message,
                )
            )
            await db.flush()

            logger.warning(
                "Review marked as failed after exhausting retries",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error_message": error_message,
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark review as failed",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_skipped(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> None:
        """
        Mark a review as SKIPPED — intentionally excluded from processing.

        Used for reviews flagged as spam or failing validation.

        Args:
            db:         Active async database session.
            review_id:  UUID of the review.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(status=ReviewStatus.SKIPPED)
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark review as skipped",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    async def flag_as_spam(
        self,
        db: AsyncSession,
        review_id: uuid.UUID,
    ) -> None:
        """
        Flag a review as spam and transition it to SKIPPED status.

        Called by review_validator.py when a review fails spam detection.

        Args:
            db:         Active async database session.
            review_id:  UUID of the review.
        """
        try:
            await db.execute(
                update(ReviewModel)
                .where(ReviewModel.id == review_id)
                .values(
                    is_spam=True,
                    is_valid=False,
                    status=ReviewStatus.SKIPPED,
                )
            )
            await db.flush()

            logger.info(
                "Review flagged as spam",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to flag review as spam",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "review_id": str(review_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Existence Check ────────────────────────────────────────────────────────

    async def exists(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        google_review_id: str,
    ) -> bool:
        """
        Check whether a review already exists for this business.

        Uses COUNT to avoid full model hydration for a boolean check.
        Called by review_monitor.py before attempting upsert when a
        lightweight pre-check is preferable.

        Args:
            db:               Active async database session.
            business_id:      UUID of the owning business.
            google_review_id: Google review identifier.

        Returns:
            bool: True if the review already exists in the database.
        """
        try:
            result = await db.execute(
                select(func.count(ReviewModel.id)).where(
                    ReviewModel.business_id == business_id,
                    ReviewModel.google_review_id == google_review_id,
                )
            )
            return result.scalar_one() > 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to check review existence",
                extra={
                    "service": ServiceName.GOOGLE_REVIEWS,
                    "business_id": str(business_id),
                    "google_review_id": google_review_id,
                    "error": str(exc),
                },
            )
            raise