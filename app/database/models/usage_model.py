# ==============================================================================
# File: app/database/models/usage_model.py
# Purpose: SQLAlchemy ORM model for tracking daily usage counters per business.
#          This table is the single source of truth for rate limit enforcement.
#          Every AI operation, review processed, competitor scan, and report
#          generated increments the relevant counter here.
#
#          Usage flow:
#            rate_limiter.py  → reads today's counters → allow / block
#            usage_tracker.py → increments counter     → after task completes
#
#          One record exists per business per calendar date.
#          The composite unique constraint on (business_id, usage_date)
#          enforces this invariant at the database level.
#
#          Records are never deleted individually — retention policy is
#          handled at the database level (e.g., partition or scheduled purge).
# ==============================================================================

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import BaseModel

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel


# ==============================================================================
# Usage Model
# ==============================================================================

class UsageModel(BaseModel):
    """
    Tracks daily usage counters for a single business.

    One record is created per business per calendar date on first use.
    Subsequent operations on the same day increment the relevant counter
    using atomic database-level updates to prevent race conditions.

    The rate_limiter.py reads this model before every expensive operation.
    The usage_tracker.py writes to this model immediately after completion.

    Inherits from BaseModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)

    Table:
        usage_records

    Indexes:
        - ix_usage_business_id              — tenant isolation
        - ix_usage_date                     — date range queries
        - ix_usage_business_date            — composite: today's record per business
        - uq_usage_business_date            — one record per business per day
    """

    __tablename__ = "usage_records"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    # ── Usage Date ────────────────────────────────────────────────────────────

    usage_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment="Calendar date (local business date) for which these counters apply",
    )

    # ── Review Processing Counters ────────────────────────────────────────────

    reviews_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of reviews fetched and entered the processing pipeline today",
    )

    # ── AI Reply Counters ─────────────────────────────────────────────────────

    ai_replies_generated: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of AI reply generations successfully completed today",
    )

    ai_replies_failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of AI reply generation attempts that failed today",
    )

    # ── Competitor Scan Counters ──────────────────────────────────────────────

    competitor_scans: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of competitor profile scans performed today",
    )

    # ── Report Counters ───────────────────────────────────────────────────────

    reports_generated: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of reports (weekly / monthly / quarterly) generated today",
    )

    # ── Content Generation Counters ───────────────────────────────────────────

    content_pieces_generated: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of social media content pieces generated today",
    )

    # ── WhatsApp Message Counters ─────────────────────────────────────────────

    whatsapp_messages_sent: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of WhatsApp messages successfully delivered today",
    )

    whatsapp_messages_failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of WhatsApp message delivery failures today",
    )

    # ── Alert Counters ────────────────────────────────────────────────────────

    alerts_triggered: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of business event alerts triggered and sent today",
    )

    # ── API Error Counters ────────────────────────────────────────────────────

    google_api_errors: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of Google API call failures encountered today",
    )

    openai_api_errors: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of OpenAI API call failures encountered today",
    )

    whatsapp_api_errors: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of WhatsApp API call failures encountered today",
    )

    # ── Rate Limit Hit Counters ───────────────────────────────────────────────

    rate_limit_hits: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment=(
            "Number of times an operation was blocked today due to usage "
            "limits being reached — useful for upgrade prompts"
        ),
    )

    # ── Last Activity Timestamp ───────────────────────────────────────────────

    last_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the most recent usage increment for this business today (UTC)",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        back_populates="usage_records",
        lazy="selectin",
    )

    # ── Table-level Constraints and Indexes ───────────────────────────────────

    __table_args__ = (
        # Core invariant — exactly one usage record per business per day
        UniqueConstraint(
            "business_id",
            "usage_date",
            name="uq_usage_business_date",
        ),
        Index(
            "ix_usage_business_id",
            "business_id",
        ),
        Index(
            "ix_usage_date",
            "usage_date",
        ),
        # Primary lookup index — rate_limiter.py and usage_tracker.py
        # both query by (business_id, usage_date) on every operation
        Index(
            "ix_usage_business_date",
            "business_id",
            "usage_date",
        ),
    )

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def total_ai_operations(self) -> int:
        """
        Total AI API calls consumed today.

        Combines successful and failed reply generations to give the true
        OpenAI cost exposure for this business on this date.
        """
        return self.ai_replies_generated + self.ai_replies_failed

    @property
    def total_api_errors(self) -> int:
        """Total external API errors across all integrations today."""
        return (
            self.google_api_errors
            + self.openai_api_errors
            + self.whatsapp_api_errors
        )

    @property
    def total_whatsapp_attempts(self) -> int:
        """Total WhatsApp send attempts (successful + failed) today."""
        return self.whatsapp_messages_sent + self.whatsapp_messages_failed

    @property
    def whatsapp_delivery_rate(self) -> float:
        """
        WhatsApp message delivery success rate for today (0.0–1.0).

        Returns 0.0 if no messages were attempted.
        """
        total = self.total_whatsapp_attempts
        if total == 0:
            return 0.0
        return round(self.whatsapp_messages_sent / total, 4)

    @property
    def has_hit_rate_limit_today(self) -> bool:
        """True if any operation was blocked by rate limiting today."""
        return self.rate_limit_hits > 0

    def is_within_limit(self, metric: str, limit: int) -> bool:
        """
        Check whether the current counter for a given metric is within
        the allowed daily limit.

        Args:
            metric (str): Column name of the counter to check.
                          Valid values: 'reviews_processed',
                          'ai_replies_generated', 'competitor_scans',
                          'reports_generated'.
            limit  (int): The maximum allowed value for this metric.

        Returns:
            bool: True if the current value is strictly less than the limit.

        Raises:
            AttributeError: If the metric name does not correspond to a
                            valid column on this model.
        """
        current = getattr(self, metric, None)
        if current is None:
            raise AttributeError(
                f"UsageModel has no counter column named '{metric}'. "
                f"Valid metrics: reviews_processed, ai_replies_generated, "
                f"competitor_scans, reports_generated."
            )
        return int(current) < limit

    def to_summary_dict(self) -> dict:
        """
        Return a concise summary dictionary suitable for admin health reports
        and WhatsApp analytics messages.

        Returns:
            dict: Key usage metrics for this business on this date.
        """
        return {
            "business_id": str(self.business_id),
            "usage_date": self.usage_date.isoformat(),
            "reviews_processed": self.reviews_processed,
            "ai_replies_generated": self.ai_replies_generated,
            "ai_replies_failed": self.ai_replies_failed,
            "competitor_scans": self.competitor_scans,
            "reports_generated": self.reports_generated,
            "content_pieces_generated": self.content_pieces_generated,
            "whatsapp_messages_sent": self.whatsapp_messages_sent,
            "alerts_triggered": self.alerts_triggered,
            "rate_limit_hits": self.rate_limit_hits,
            "total_api_errors": self.total_api_errors,
            "last_activity_at": (
                self.last_activity_at.isoformat()
                if self.last_activity_at
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<UsageModel id={self.id} "
            f"business_id={self.business_id} "
            f"date={self.usage_date} "
            f"reviews={self.reviews_processed} "
            f"ai_replies={self.ai_replies_generated} "
            f"competitor_scans={self.competitor_scans}>"
        )