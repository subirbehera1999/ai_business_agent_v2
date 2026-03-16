# ==============================================================================
# File: app/database/models/payment_model.py
# Purpose: SQLAlchemy ORM model representing a Razorpay payment transaction.
#          Every payment attempt — successful or failed — is recorded here
#          before any subscription state is modified. This is the immutable
#          financial audit trail for the platform.
#
#          Payment flow:
#            Payment initiated → record created (INITIATED)
#            → Razorpay webhook received
#            → signature verified
#            → status updated (SUCCESS / FAILED)
#            → subscription activated (on SUCCESS only)
#
#          Server-side verification is mandatory. Frontend payment confirmation
#          is never trusted. See: app/payments/webhook_handler.py
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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.constants import PaymentEventType, PaymentStatus
from app.database.base import BaseModel

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel
    from app.database.models.subscription_model import SubscriptionModel


# ==============================================================================
# Payment Model
# ==============================================================================

class PaymentModel(BaseModel):
    """
    Represents a single Razorpay payment transaction.

    Records are created when a payment is initiated and updated when
    the Razorpay webhook confirms the outcome. The subscription is only
    activated after this record reaches SUCCESS status.

    A payment record is intentionally immutable after reaching a terminal
    state (SUCCESS, FAILED, REFUNDED). Only the webhook handler and
    payment_service.py may update payment records.

    Inherits from BaseModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)

    Note: BaseModel (not SoftDeletableModel) is used here because payment
    records must never be soft-deleted. Financial records require permanent
    retention for auditing and dispute resolution.

    Table:
        payments

    Indexes:
        - ix_payments_business_id             — tenant isolation queries
        - ix_payments_subscription_id         — load payments for a subscription
        - ix_payments_razorpay_payment_id     — webhook lookup
        - ix_payments_razorpay_order_id       — order reconciliation
        - ix_payments_status                  — filter by outcome
        - ix_payments_created_at              — chronological queries
        - uq_payments_razorpay_payment_id     — uniqueness constraint
    """

    __tablename__ = "payments"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subscriptions.id", ondelete="SET NULL"),
        nullable=True,
        comment="Foreign key to the subscription this payment activates",
    )

    # ── Razorpay Identifiers ──────────────────────────────────────────────────

    razorpay_order_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Razorpay order ID (order_XXXXXX) — created before checkout",
    )

    razorpay_payment_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        comment="Razorpay payment ID (pay_XXXXXX) — assigned after payment attempt",
    )

    razorpay_signature: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Razorpay webhook HMAC signature — stored for audit trail",
    )

    razorpay_subscription_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Razorpay subscription ID if this payment is for a recurring plan",
    )

    # ── Payment Details ───────────────────────────────────────────────────────

    amount: Mapped[float] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        comment="Payment amount in base currency units (e.g., INR)",
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
        server_default="INR",
        comment="ISO 4217 currency code",
    )

    plan: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Subscription plan being purchased (starter / growth / pro)",
    )

    billing_cycle_months: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Billing duration in months for this payment",
    )

    # ── Status & Lifecycle ────────────────────────────────────────────────────

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=PaymentStatus.INITIATED,
        server_default=PaymentStatus.INITIATED,
        comment="Current payment lifecycle state",
    )

    event_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Razorpay webhook event type that last updated this record",
    )

    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when payment was successfully captured (UTC)",
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when payment was confirmed as failed (UTC)",
    )

    refunded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when payment was refunded (UTC)",
    )

    # ── Verification ─────────────────────────────────────────────────────────

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once Razorpay webhook signature has been verified server-side",
    )

    verification_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last signature verification attempt (UTC)",
    )

    subscription_activated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the subscription was successfully activated after this payment",
    )

    # ── Failure Details ───────────────────────────────────────────────────────

    failure_reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Human-readable failure reason from Razorpay (e.g., insufficient funds)",
    )

    failure_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Machine-readable failure code from Razorpay",
    )

    # ── Refund Details ────────────────────────────────────────────────────────

    refund_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Razorpay refund ID if this payment was refunded",
    )

    refund_amount: Mapped[float | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
        comment="Amount refunded (may be partial)",
    )

    # ── Idempotency ───────────────────────────────────────────────────────────

    idempotency_key: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment="Idempotency key to prevent duplicate payment processing",
    )

    # ── Raw Webhook Payload ───────────────────────────────────────────────────

    raw_webhook_payload: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Raw JSON webhook payload from Razorpay — stored for debugging and audit",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        back_populates="payments",
        lazy="selectin",
    )

    subscription: Mapped["SubscriptionModel | None"] = relationship(
        "SubscriptionModel",
        back_populates="payments",
        lazy="selectin",
    )

    # ── Table-level Constraints and Indexes ───────────────────────────────────

    __table_args__ = (
        UniqueConstraint(
            "razorpay_payment_id",
            name="uq_payments_razorpay_payment_id",
        ),
        UniqueConstraint(
            "idempotency_key",
            name="uq_payments_idempotency_key",
        ),
        Index(
            "ix_payments_business_id",
            "business_id",
        ),
        Index(
            "ix_payments_subscription_id",
            "subscription_id",
        ),
        Index(
            "ix_payments_razorpay_payment_id",
            "razorpay_payment_id",
        ),
        Index(
            "ix_payments_razorpay_order_id",
            "razorpay_order_id",
        ),
        Index(
            "ix_payments_status",
            "status",
        ),
        Index(
            "ix_payments_business_status",
            "business_id",
            "status",
        ),
        Index(
            "ix_payments_created_at",
            "created_at",
        ),
    )

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_successful(self) -> bool:
        """True if this payment reached SUCCESS status."""
        return self.status == PaymentStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """True if this payment reached FAILED status."""
        return self.status == PaymentStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """True if this payment is still awaiting webhook confirmation."""
        return self.status in {PaymentStatus.INITIATED, PaymentStatus.PENDING}

    @property
    def is_refunded(self) -> bool:
        """True if this payment has been fully or partially refunded."""
        return self.status == PaymentStatus.REFUNDED

    @property
    def is_terminal(self) -> bool:
        """
        True if this payment has reached a terminal state.

        Terminal states are SUCCESS, FAILED, and REFUNDED.
        A terminal payment record must not be modified by webhook retries.
        """
        return self.status in {
            PaymentStatus.SUCCESS,
            PaymentStatus.FAILED,
            PaymentStatus.REFUNDED,
        }

    @property
    def can_activate_subscription(self) -> bool:
        """
        True if this payment qualifies to activate a subscription.

        Requires:
        - Payment is successful
        - Webhook signature has been verified server-side
        - Subscription has not already been activated by this payment
        """
        return (
            self.is_successful
            and self.is_verified
            and not self.subscription_activated
        )

    @property
    def amount_display(self) -> str:
        """Return a human-readable amount string (e.g., '₹999.00')."""
        symbol_map = {"INR": "₹", "USD": "$", "EUR": "€"}
        symbol = symbol_map.get(self.currency, self.currency)
        return f"{symbol}{float(self.amount):,.2f}"

    def __repr__(self) -> str:
        return (
            f"<PaymentModel id={self.id} "
            f"business_id={self.business_id} "
            f"razorpay_payment_id='{self.razorpay_payment_id}' "
            f"amount={self.amount} {self.currency} "
            f"status='{self.status}' "
            f"verified={self.is_verified}>"
        )