# ==============================================================================
# File: app/database/models/subscription_model.py
# Purpose: SQLAlchemy ORM model representing a business subscription.
#
#          ONE SUBSCRIPTION TIER ONLY.
#          Pay = full access to every feature.
#          The only variable is billing_cycle: "monthly" or "annual".
#
#          The subscription record is the gatekeeper for platform access —
#          plan_manager.py reads this model to enforce daily usage caps.
#          The override_* columns allow per-business cap adjustments by
#          admin — they are abuse guards, not feature gates.
# ==============================================================================

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.constants import DAILY_USAGE_LIMITS, SubscriptionStatus
from app.database.base import SoftDeletableModel

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel
    from app.database.models.payment_model import PaymentModel


# ==============================================================================
# Subscription Model
# ==============================================================================

class SubscriptionModel(SoftDeletableModel):
    """
    Represents a subscription held by a business on the platform.

    A business may have multiple subscription records over time
    (one per billing renewal), but only one may be ACTIVE at any point.
    Repositories enforce this constraint.

    One tier only — full access on any active subscription.
    The override_* columns are per-business abuse-prevention overrides
    set by admin only. NULL means the platform default from
    DAILY_USAGE_LIMITS in constants.py applies.

    Inherits from SoftDeletableModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)
        - is_deleted   (soft-delete flag)
        - deleted_at   (soft-delete timestamp)

    Table:
        subscriptions

    Indexes:
        - ix_subscriptions_business_id          — tenant isolation queries
        - ix_subscriptions_status               — filter active subscriptions
        - ix_subscriptions_business_status      — composite: active sub per business
        - ix_subscriptions_expires_at           — expiry checker job
        - ix_subscriptions_razorpay_sub_id      — webhook lookup by Razorpay ID
        - ix_subscriptions_is_deleted           — soft-delete filter
    """

    __tablename__ = "subscriptions"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    # ── Subscription State ────────────────────────────────────────────────────

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=SubscriptionStatus.PENDING,
        server_default=SubscriptionStatus.PENDING,
        comment="Current lifecycle state: active / expired / cancelled / pending / trial",
    )

    # ── Billing Cycle ─────────────────────────────────────────────────────────

    billing_cycle: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="monthly",
        server_default="monthly",
        comment="Billing cycle string: 'monthly' or 'annual'",
    )

    billing_cycle_months: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Billing duration in months (1 = monthly, 12 = annual)",
    )

    amount: Mapped[float] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        comment="Total amount charged for this subscription period (INR)",
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
        server_default="INR",
        comment="Billing currency code",
    )

    # ── Validity Window ───────────────────────────────────────────────────────

    starts_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the subscription became or becomes active (UTC)",
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the subscription expires (UTC)",
    )

    trial_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="End of trial period if this subscription started as a trial (UTC)",
    )

    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the subscription was cancelled (UTC)",
    )

    # ── Razorpay Integration ──────────────────────────────────────────────────

    razorpay_subscription_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        comment="Razorpay subscription object ID (sub_XXXXXX)",
    )

    razorpay_plan_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Razorpay plan ID linked to this subscription",
    )

    last_payment_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Razorpay payment ID of the most recent successful charge",
    )

    # ── Renewal Control ───────────────────────────────────────────────────────

    auto_renew: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="Whether this subscription auto-renews at expiry",
    )

    renewal_reminder_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the pre-expiry renewal reminder has been sent",
    )

    # ── Per-Business Usage Cap Overrides ─────────────────────────────────────
    # Set by admin only. NULL = use platform default from DAILY_USAGE_LIMITS.
    # These are abuse-prevention guards — not feature gates.

    override_max_reviews_per_day: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Per-business daily review cap override (null = platform default)",
    )

    override_max_ai_replies_per_day: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Per-business daily AI reply cap override (null = platform default)",
    )

    override_max_competitor_scans_per_day: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Per-business daily competitor scan cap override (null = platform default)",
    )

    override_max_reports_per_day: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Per-business daily report cap override (null = platform default)",
    )

    # ── Admin Notes ───────────────────────────────────────────────────────────

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Internal admin notes about this subscription (never sent to AI)",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        back_populates="subscriptions",
        lazy="selectin",
    )

    payments: Mapped[list["PaymentModel"]] = relationship(
        "PaymentModel",
        back_populates="subscription",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    # ── Table-level Indexes ───────────────────────────────────────────────────

    __table_args__ = (
        Index("ix_subscriptions_business_id",     "business_id"),
        Index("ix_subscriptions_status",          "status"),
        Index("ix_subscriptions_business_status", "business_id", "status"),
        Index("ix_subscriptions_expires_at",      "expires_at"),
        Index("ix_subscriptions_razorpay_sub_id", "razorpay_subscription_id"),
        Index("ix_subscriptions_is_deleted",      "is_deleted"),
    )

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True if the subscription is currently in ACTIVE status."""
        return self.status == SubscriptionStatus.ACTIVE

    @property
    def is_trial(self) -> bool:
        """True if the subscription is currently in TRIAL status."""
        return self.status == SubscriptionStatus.TRIAL

    @property
    def is_expired(self) -> bool:
        """True if the subscription is in EXPIRED status."""
        return self.status == SubscriptionStatus.EXPIRED

    @property
    def is_usable(self) -> bool:
        """
        True if the subscription grants access to platform features.

        Both ACTIVE and TRIAL subscriptions allow feature usage.
        EXPIRED, CANCELLED, and PENDING do not.
        """
        return self.status in {SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL}

    @property
    def effective_max_reviews_per_day(self) -> int:
        """
        Effective daily review processing cap for this subscription.

        Returns the per-business admin override if set, otherwise the
        platform-wide default from DAILY_USAGE_LIMITS.
        """
        if self.override_max_reviews_per_day is not None:
            return self.override_max_reviews_per_day
        return DAILY_USAGE_LIMITS["max_reviews_per_day"]

    @property
    def effective_max_ai_replies_per_day(self) -> int:
        """
        Effective daily AI reply generation cap for this subscription.

        Returns the per-business admin override if set, otherwise the
        platform-wide default from DAILY_USAGE_LIMITS.
        """
        if self.override_max_ai_replies_per_day is not None:
            return self.override_max_ai_replies_per_day
        return DAILY_USAGE_LIMITS["max_ai_replies_per_day"]

    @property
    def effective_max_competitor_scans_per_day(self) -> int:
        """
        Effective daily competitor scan cap for this subscription.

        Returns the per-business admin override if set, otherwise the
        platform-wide default from DAILY_USAGE_LIMITS.
        """
        if self.override_max_competitor_scans_per_day is not None:
            return self.override_max_competitor_scans_per_day
        return DAILY_USAGE_LIMITS["max_competitor_scans_per_day"]

    @property
    def effective_max_reports_per_day(self) -> int:
        """
        Effective daily report generation cap for this subscription.

        Returns the per-business admin override if set, otherwise the
        platform-wide default from DAILY_USAGE_LIMITS.
        """
        if self.override_max_reports_per_day is not None:
            return self.override_max_reports_per_day
        return DAILY_USAGE_LIMITS["max_reports_per_day"]

    def __repr__(self) -> str:
        return (
            f"<SubscriptionModel id={self.id} "
            f"business_id={self.business_id} "
            f"billing_cycle='{self.billing_cycle}' "
            f"status='{self.status}' "
            f"expires_at={self.expires_at}>"
        )