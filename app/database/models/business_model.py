# ==============================================================================
# File: app/database/models/business_model.py
# Purpose: SQLAlchemy ORM model representing a registered business account.
#          This is the root entity of the multi-tenant system. Every other
#          model (reviews, subscriptions, payments, usage, alerts, jobs)
#          references business_id as the tenant isolation key.
# ==============================================================================

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import SoftDeletableModel

# Avoid circular imports — models reference each other only for type hints
if TYPE_CHECKING:
    from app.database.models.subscription_model import SubscriptionModel
    from app.database.models.review_model import ReviewModel
    from app.database.models.payment_model import PaymentModel
    from app.database.models.usage_model import UsageModel
    from app.database.models.job_model import JobModel


# ==============================================================================
# Business Model
# ==============================================================================

class BusinessModel(SoftDeletableModel):
    """
    Represents a registered business account on the platform.

    Each business is an isolated tenant. All queries across the system
    must filter by business_id to prevent cross-tenant data leakage.

    Inherits from SoftDeletableModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)
        - is_deleted   (soft-delete flag)
        - deleted_at   (soft-delete timestamp)

    Table:
        businesses

    Indexes:
        - ix_businesses_owner_whatsapp  — fast lookup by WhatsApp number
        - ix_businesses_google_place_id — fast lookup by Google Place ID
        - ix_businesses_is_active       — filter active businesses in scheduler
        - ix_businesses_is_deleted      — exclude soft-deleted records efficiently
    """

    __tablename__ = "businesses"

    # ── Identity ──────────────────────────────────────────────────────────────

    business_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Registered name of the business",
    )

    business_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Business category (e.g., restaurant, clinic, salon)",
    )

    business_description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Short description of the business",
    )

    # ── Owner Contact ─────────────────────────────────────────────────────────

    owner_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Full name of the business owner",
    )

    owner_email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Owner email address — used for account identification",
    )

    owner_whatsapp_number: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        unique=True,
        comment="WhatsApp number with country code (e.g., +919876543210)",
    )

    # ── Location ──────────────────────────────────────────────────────────────

    city: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="City where the business is located",
    )

    state: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="State or province",
    )

    country: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        server_default="India",
        comment="Country of operation",
    )

    # ── Google Integration ────────────────────────────────────────────────────

    google_place_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        comment="Google Places ID for fetching reviews and business info",
    )

    google_business_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Business name as registered on Google",
    )

    google_access_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Encrypted Google OAuth access token",
    )

    google_refresh_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Encrypted Google OAuth refresh token",
    )

    google_sheets_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="URL of the connected Google Sheet for sales data",
    )

    # ── Current Rating ────────────────────────────────────────────────────────

    current_google_rating: Mapped[float | None] = mapped_column(
        nullable=True,
        comment="Most recently fetched Google rating (1.0–5.0)",
    )

    total_google_reviews: Mapped[int | None] = mapped_column(
        nullable=True,
        comment="Total number of Google reviews at last fetch",
    )

    # ── Status Flags ──────────────────────────────────────────────────────────

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="Whether this business account is active and being processed",
    )

    is_onboarding_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the business has completed onboarding setup",
    )

    is_google_connected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once Google Business account is successfully linked",
    )

    is_sheets_connected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once a Google Sheet for sales data is linked",
    )

    # ── Feedback ──────────────────────────────────────────────────────────────

    feedback_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the 30-day feedback request has been sent",
    )

    # ── Timezone ──────────────────────────────────────────────────────────────

    timezone: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
        server_default="Asia/Kolkata",
        comment="Business local timezone (IANA format) for scheduling reports",
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    # All relationships use lazy="selectin" for async-safe eager loading.
    # This avoids implicit lazy-load errors in async SQLAlchemy sessions.

    subscriptions: Mapped[list["SubscriptionModel"]] = relationship(
        "SubscriptionModel",
        back_populates="business",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    reviews: Mapped[list["ReviewModel"]] = relationship(
        "ReviewModel",
        back_populates="business",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    payments: Mapped[list["PaymentModel"]] = relationship(
        "PaymentModel",
        back_populates="business",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    usage_records: Mapped[list["UsageModel"]] = relationship(
        "UsageModel",
        back_populates="business",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    jobs: Mapped[list["JobModel"]] = relationship(
        "JobModel",
        back_populates="business",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    # ── Table-level Indexes ───────────────────────────────────────────────────

    __table_args__ = (
        Index(
            "ix_businesses_owner_whatsapp",
            "owner_whatsapp_number",
        ),
        Index(
            "ix_businesses_google_place_id",
            "google_place_id",
        ),
        Index(
            "ix_businesses_is_active",
            "is_active",
        ),
        Index(
            "ix_businesses_is_deleted",
            "is_deleted",
        ),
        Index(
            "ix_businesses_owner_email",
            "owner_email",
        ),
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        """Return the business name for use in messages and reports."""
        return self.business_name

    @property
    def whatsapp_recipient(self) -> str:
        """
        Return the WhatsApp-formatted recipient number.

        WhatsApp Cloud API requires numbers without the leading '+'.
        """
        return self.owner_whatsapp_number.lstrip("+")

    @property
    def has_google_integration(self) -> bool:
        """True if the business has a linked Google Place ID."""
        return bool(self.google_place_id)

    @property
    def has_sales_data(self) -> bool:
        """True if the business has a linked Google Sheets URL."""
        return bool(self.google_sheets_url)

    def __repr__(self) -> str:
        return (
            f"<BusinessModel id={self.id} "
            f"name='{self.business_name}' "
            f"active={self.is_active} "
            f"deleted={self.is_deleted}>"
        )