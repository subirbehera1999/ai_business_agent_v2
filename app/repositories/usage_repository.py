# ==============================================================================
# File: app/repositories/usage_repository.py
# Purpose: Repository class encapsulating all database operations for the
#          UsageModel. This is the only layer permitted to query the
#          usage_records table directly.
#
#          Core responsibilities:
#            - Atomic daily counter increments (no read-modify-write races)
#            - Upsert-based record creation (one record per business per day)
#            - Rate limit reads for plan enforcement
#            - Usage summary queries for health reports and admin dashboards
#            - All queries enforce business_id tenant isolation
#
#          Critical pattern:
#            All counter increments use server-side SQL arithmetic
#            (column + 1) rather than Python-side read-modify-write.
#            This prevents race conditions when multiple scheduler jobs
#            update the same business's counters concurrently.
# ==============================================================================

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName, UsageMetric
from app.database.models.usage_model import UsageModel

logger = logging.getLogger(ServiceName.API)


# ==============================================================================
# Usage Repository
# ==============================================================================

class UsageRepository:
    """
    Handles all database operations for UsageModel.

    Session management (commit/rollback) is the responsibility of the
    caller. This repository only calls flush() to populate server defaults.

    Design principle:
        All counter increments are atomic server-side SQL expressions.
        Never read a counter value into Python, modify it, then write it back.
        This class uses column + 1 patterns exclusively for all increments
        to guarantee correctness under concurrent scheduler execution.

    Usage:
        repo = UsageRepository()
        usage = await repo.get_or_create_today(db, business_id)
        await repo.increment_reviews_processed(db, business_id)
    """

    # ── Get or Create Today's Record ──────────────────────────────────────────

    async def get_or_create_today(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> UsageModel:
        """
        Fetch today's usage record for a business, creating it if absent.

        Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING to guarantee
        exactly one record per business per calendar date even under
        concurrent access from multiple scheduler workers.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date for the record (defaults to today UTC).

        Returns:
            UsageModel: The existing or newly created usage record.

        Raises:
            SQLAlchemyError: On any database error.
        """
        today = usage_date or date.today()

        try:
            # Attempt insert — silently skip if record already exists
            stmt = (
                pg_insert(UsageModel)
                .values(
                    business_id=business_id,
                    usage_date=today,
                    reviews_processed=0,
                    ai_replies_generated=0,
                    ai_replies_failed=0,
                    competitor_scans=0,
                    reports_generated=0,
                    content_pieces_generated=0,
                    whatsapp_messages_sent=0,
                    whatsapp_messages_failed=0,
                    alerts_triggered=0,
                    google_api_errors=0,
                    openai_api_errors=0,
                    whatsapp_api_errors=0,
                    rate_limit_hits=0,
                )
                .on_conflict_do_nothing(
                    constraint="uq_usage_business_date"
                )
            )
            await db.execute(stmt)
            await db.flush()

            # Fetch the record — whether just inserted or already existing
            result = await db.execute(
                select(UsageModel).where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date == today,
                )
            )
            record = result.scalar_one()
            return record

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to get or create today's usage record",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "usage_date": str(today),
                    "error": str(exc),
                },
            )
            raise

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get_by_date(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: date,
    ) -> Optional[UsageModel]:
        """
        Fetch the usage record for a specific business and date.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   The calendar date to look up.

        Returns:
            UsageModel if a record exists for that date, else None.
        """
        try:
            result = await db.execute(
                select(UsageModel).where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date == usage_date,
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch usage record by date",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "usage_date": str(usage_date),
                    "error": str(exc),
                },
            )
            raise

    async def get_range(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        start_date: date,
        end_date: date,
        limit: int = 50,
    ) -> list[UsageModel]:
        """
        Fetch usage records for a business within a date range.

        Used by analytics_service.py and reports_service.py to build
        weekly and monthly performance summaries.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            start_date:   Inclusive lower bound date.
            end_date:     Inclusive upper bound date.
            limit:        Maximum records to return (default: 50).

        Returns:
            list[UsageModel]: Usage records ordered by date ascending.
        """
        try:
            result = await db.execute(
                select(UsageModel)
                .where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date >= start_date,
                    UsageModel.usage_date <= end_date,
                )
                .order_by(UsageModel.usage_date.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch usage records for date range",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "error": str(exc),
                },
            )
            raise

    async def get_current_count(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        metric: str,
        usage_date: Optional[date] = None,
    ) -> int:
        """
        Return the current value of a single usage counter for today.

        Used by rate_limiter.py for a lightweight counter read before
        deciding whether to allow or block an operation.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            metric:       Column name of the counter (see UsageMetric enum).
            usage_date:   Date to check (defaults to today UTC).

        Returns:
            int: Current counter value, or 0 if no record exists for today.

        Raises:
            AttributeError: If the metric name is not a valid column.
        """
        today = usage_date or date.today()

        # Validate metric name against the model to prevent SQL injection
        valid_metrics = {
            UsageMetric.REVIEWS_PROCESSED,
            UsageMetric.AI_REPLIES_GENERATED,
            UsageMetric.COMPETITOR_SCANS,
            UsageMetric.REPORTS_GENERATED,
        }
        if metric not in valid_metrics:
            raise AttributeError(
                f"'{metric}' is not a valid rate-limited usage metric. "
                f"Valid metrics: {valid_metrics}"
            )

        try:
            column = getattr(UsageModel, metric)
            result = await db.execute(
                select(column).where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date == today,
                )
            )
            value = result.scalar_one_or_none()
            return int(value) if value is not None else 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch current usage count",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "metric": metric,
                    "error": str(exc),
                },
            )
            raise

    # ── Atomic Counter Increments ─────────────────────────────────────────────
    #
    # All methods below use server-side column arithmetic:
    #   UsageModel.column + 1
    # This is a single atomic SQL UPDATE — safe under concurrent access.
    # Never replace these with read-modify-write patterns.
    # ─────────────────────────────────────────────────────────────────────────

    async def _increment(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        column_name: str,
        amount: int = 1,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment a single usage counter column.

        Internal helper used by all public increment methods. Ensures the
        daily record exists before incrementing and updates last_activity_at
        in the same statement.

        Args:
            db:            Active async database session.
            business_id:   UUID of the business.
            column_name:   Name of the column to increment.
            amount:        Amount to add (default: 1).
            usage_date:    Date for the record (defaults to today UTC).
        """
        today = usage_date or date.today()

        # Ensure the record exists for today before incrementing
        await self.get_or_create_today(db, business_id, today)

        try:
            column = getattr(UsageModel, column_name)
            await db.execute(
                update(UsageModel)
                .where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date == today,
                )
                .values(
                    **{column_name: column + amount},
                    last_activity_at=datetime.now(timezone.utc),
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to increment usage counter",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "column": column_name,
                    "amount": amount,
                    "error": str(exc),
                },
            )
            raise

    async def increment_reviews_processed(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the reviews_processed counter.

        Called by review_jobs.py immediately after a review enters
        the processing pipeline.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            UsageMetric.REVIEWS_PROCESSED,
            usage_date=usage_date,
        )

    async def increment_ai_replies_generated(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the ai_replies_generated counter.

        Called by ai_reply_service.py immediately after a successful
        AI reply generation.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            UsageMetric.AI_REPLIES_GENERATED,
            usage_date=usage_date,
        )

    async def increment_ai_replies_failed(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the ai_replies_failed counter.

        Called by ai_reply_service.py when an AI generation attempt fails.
        Failed attempts count toward OpenAI cost exposure tracking even
        though they do not count against the plan limit.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "ai_replies_failed",
            usage_date=usage_date,
        )

    async def increment_competitor_scans(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the competitor_scans counter.

        Called by competitor_service.py after each competitor profile scan.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            UsageMetric.COMPETITOR_SCANS,
            usage_date=usage_date,
        )

    async def increment_reports_generated(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the reports_generated counter.

        Called by reports_service.py after each report is successfully
        generated and dispatched.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            UsageMetric.REPORTS_GENERATED,
            usage_date=usage_date,
        )

    async def increment_content_pieces_generated(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the content_pieces_generated counter.

        Called by content_generation_service.py after each social media
        content piece is successfully generated.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "content_pieces_generated",
            usage_date=usage_date,
        )

    async def increment_whatsapp_sent(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the whatsapp_messages_sent counter.

        Called by whatsapp_service.py after a successful message delivery.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "whatsapp_messages_sent",
            usage_date=usage_date,
        )

    async def increment_whatsapp_failed(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the whatsapp_messages_failed counter.

        Called by whatsapp_service.py when a message delivery fails
        after all retries are exhausted.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "whatsapp_messages_failed",
            usage_date=usage_date,
        )

    async def increment_alerts_triggered(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the alerts_triggered counter.

        Called by alert_manager.py after each business event alert
        is dispatched via WhatsApp.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "alerts_triggered",
            usage_date=usage_date,
        )

    async def increment_google_api_errors(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the google_api_errors counter.

        Called by google_reviews_client.py and google_sheets_client.py
        when an API call fails after all retries.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "google_api_errors",
            usage_date=usage_date,
        )

    async def increment_openai_api_errors(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the openai_api_errors counter.

        Called by ai_reply_service.py when an OpenAI API call fails
        after all retries are exhausted.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "openai_api_errors",
            usage_date=usage_date,
        )

    async def increment_whatsapp_api_errors(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the whatsapp_api_errors counter.

        Called by whatsapp_client.py when a WhatsApp API call fails
        after all retries are exhausted.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "whatsapp_api_errors",
            usage_date=usage_date,
        )

    async def increment_rate_limit_hits(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        usage_date: Optional[date] = None,
    ) -> None:
        """
        Atomically increment the rate_limit_hits counter.

        Called by rate_limiter.py every time an operation is blocked
        because the business has reached its daily plan limit.
        Accumulation of this counter triggers upgrade suggestion alerts.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            usage_date:   Date override (defaults to today UTC).
        """
        await self._increment(
            db, business_id,
            "rate_limit_hits",
            usage_date=usage_date,
        )

    # ── Aggregates & Summaries ────────────────────────────────────────────────

    async def get_totals_for_range(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        start_date: date,
        end_date: date,
    ) -> dict[str, int]:
        """
        Return summed usage totals across all counters for a date range.

        Used by reports_service.py to build weekly and monthly summaries
        without loading individual daily records into Python memory.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            start_date:   Inclusive start of the range.
            end_date:     Inclusive end of the range.

        Returns:
            dict: Summed totals for each usage counter column.
            All keys are present and default to 0 if no records found.
        """
        try:
            result = await db.execute(
                select(
                    func.coalesce(func.sum(UsageModel.reviews_processed), 0),
                    func.coalesce(func.sum(UsageModel.ai_replies_generated), 0),
                    func.coalesce(func.sum(UsageModel.ai_replies_failed), 0),
                    func.coalesce(func.sum(UsageModel.competitor_scans), 0),
                    func.coalesce(func.sum(UsageModel.reports_generated), 0),
                    func.coalesce(func.sum(UsageModel.content_pieces_generated), 0),
                    func.coalesce(func.sum(UsageModel.whatsapp_messages_sent), 0),
                    func.coalesce(func.sum(UsageModel.whatsapp_messages_failed), 0),
                    func.coalesce(func.sum(UsageModel.alerts_triggered), 0),
                    func.coalesce(func.sum(UsageModel.google_api_errors), 0),
                    func.coalesce(func.sum(UsageModel.openai_api_errors), 0),
                    func.coalesce(func.sum(UsageModel.whatsapp_api_errors), 0),
                    func.coalesce(func.sum(UsageModel.rate_limit_hits), 0),
                ).where(
                    UsageModel.business_id == business_id,
                    UsageModel.usage_date >= start_date,
                    UsageModel.usage_date <= end_date,
                )
            )
            row = result.one()
            return {
                "reviews_processed": int(row[0]),
                "ai_replies_generated": int(row[1]),
                "ai_replies_failed": int(row[2]),
                "competitor_scans": int(row[3]),
                "reports_generated": int(row[4]),
                "content_pieces_generated": int(row[5]),
                "whatsapp_messages_sent": int(row[6]),
                "whatsapp_messages_failed": int(row[7]),
                "alerts_triggered": int(row[8]),
                "google_api_errors": int(row[9]),
                "openai_api_errors": int(row[10]),
                "whatsapp_api_errors": int(row[11]),
                "rate_limit_hits": int(row[12]),
            }

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch usage totals for date range",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "error": str(exc),
                },
            )
            raise

    async def get_platform_totals_for_date(
        self,
        db: AsyncSession,
        usage_date: date,
    ) -> dict[str, int]:
        """
        Return summed usage totals across ALL businesses for a single date.

        Used by admin_health_report.py to generate the daily platform-wide
        health summary sent to the admin WhatsApp number.

        Args:
            db:          Active async database session.
            usage_date:  The date to summarise.

        Returns:
            dict: Platform-wide summed totals for each counter column.
        """
        try:
            result = await db.execute(
                select(
                    func.coalesce(func.sum(UsageModel.reviews_processed), 0),
                    func.coalesce(func.sum(UsageModel.ai_replies_generated), 0),
                    func.coalesce(func.sum(UsageModel.competitor_scans), 0),
                    func.coalesce(func.sum(UsageModel.reports_generated), 0),
                    func.coalesce(func.sum(UsageModel.whatsapp_messages_sent), 0),
                    func.coalesce(func.sum(UsageModel.alerts_triggered), 0),
                    func.coalesce(func.sum(UsageModel.openai_api_errors), 0),
                    func.coalesce(func.sum(UsageModel.rate_limit_hits), 0),
                    func.count(UsageModel.business_id),
                ).where(
                    UsageModel.usage_date == usage_date,
                )
            )
            row = result.one()
            return {
                "reviews_processed": int(row[0]),
                "ai_replies_generated": int(row[1]),
                "competitor_scans": int(row[2]),
                "reports_generated": int(row[3]),
                "whatsapp_messages_sent": int(row[4]),
                "alerts_triggered": int(row[5]),
                "openai_api_errors": int(row[6]),
                "rate_limit_hits": int(row[7]),
                "active_businesses": int(row[8]),
            }

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch platform usage totals",
                extra={
                    "service": ServiceName.API,
                    "usage_date": str(usage_date),
                    "error": str(exc),
                },
            )
            raise

    async def count_businesses_active_on_date(
        self,
        db: AsyncSession,
        usage_date: date,
    ) -> int:
        """
        Count the number of distinct businesses that had any activity
        on a given date.

        Used by admin health reports to track daily active business count.

        Args:
            db:          Active async database session.
            usage_date:  The date to count.

        Returns:
            int: Number of businesses with a usage record on that date.
        """
        try:
            result = await db.execute(
                select(func.count(UsageModel.business_id)).where(
                    UsageModel.usage_date == usage_date,
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count active businesses for date",
                extra={
                    "service": ServiceName.API,
                    "usage_date": str(usage_date),
                    "error": str(exc),
                },
            )
            raise