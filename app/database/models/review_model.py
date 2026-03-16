# ==============================================================================
# File: app/database/models/review_model.py
# Purpose: SQLAlchemy ORM model representing a Google Business review fetched
#          for a registered business. Stores the original review content,
#          sentiment classification, AI-generated reply, processing status,
#          and the full lifecycle of each review through the platform pipeline.
#
#          Review pipeline:
#            Google API fetch → validate → store (NEW)
#            → sentiment analysis → classify (POSITIVE / NEGATIVE / NEUTRAL)
#            → AI reply generation → store reply (REPLIED)
#            → WhatsApp alert sent to business owner
#
#          Idempotency is enforced via the composite unique constraint on
#          (business_id, google_review_id) — the same review is never
#          processed twice regardless of polling frequency.
# ==============================================================================

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.constants import ReviewSentiment, ReviewStatus
from app.database.base import BaseModel

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel


# ==============================================================================
# Review Model
# ==============================================================================

class ReviewModel(BaseModel):
    """
    Represents a single Google Business review for a registered business.

    Each review passes through sentiment analysis and AI reply generation.
    The record tracks the full lifecycle from initial fetch to replied state.

    A review is uniquely identified by the combination of business_id and
    google_review_id. Duplicate fetches of the same review are silently
    skipped by the repository layer using the unique constraint.

    Inherits from BaseModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)

    Table:
        reviews

    Indexes:
        - ix_reviews_business_id              — tenant isolation
        - ix_reviews_status                   — filter unprocessed reviews
        - ix_reviews_sentiment                — filter by sentiment type
        - ix_reviews_business_status          — composite: pending reviews per business
        - ix_reviews_google_review_id         — deduplication lookup
        - ix_reviews_reviewer_published_at    — chronological ordering
        - ix_reviews_rating                   — alert threshold queries
        - uq_reviews_business_google_id       — idempotency constraint
    """

    __tablename__ = "reviews"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    # ── Google Review Identity ────────────────────────────────────────────────

    google_review_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Unique review identifier assigned by Google — used for deduplication",
    )

    google_place_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Google Place ID of the business at time of fetch — for audit",
    )

    # ── Reviewer Information ──────────────────────────────────────────────────

    reviewer_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Display name of the reviewer as returned by Google",
    )

    reviewer_profile_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Google profile URL of the reviewer",
    )

    # ── Review Content ────────────────────────────────────────────────────────

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Star rating given by the reviewer (1–5)",
    )

    review_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full text of the review as submitted by the reviewer",
    )

    review_text_truncated: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Truncated review text sent to AI (max 800 chars per prompt safety rules). "
            "Null if review_text was within the limit."
        ),
    )

    original_language: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Detected language code of the review text (e.g., 'en', 'hi')",
    )

    # ── Timestamps from Google ────────────────────────────────────────────────

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the review was published on Google (UTC)",
    )

    updated_on_google_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last edit to this review on Google (UTC)",
    )

    # ── Sentiment Analysis ────────────────────────────────────────────────────

    sentiment: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Sentiment classification: positive / negative / neutral",
    )

    sentiment_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score of the sentiment classification (0.0–1.0)",
    )

    sentiment_analysed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when sentiment analysis was completed (UTC)",
    )

    # ── AI Reply ──────────────────────────────────────────────────────────────

    ai_reply: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated reply text for this review",
    )

    ai_reply_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the AI reply was generated (UTC)",
    )

    ai_reply_prompt_used: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Filename of the prompt template used for AI reply generation",
    )

    reply_posted_to_google: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the AI reply has been successfully posted to Google",
    )

    reply_posted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the reply was posted to Google (UTC)",
    )

    # ── Processing Status ─────────────────────────────────────────────────────

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=ReviewStatus.NEW,
        server_default=ReviewStatus.NEW,
        comment="Processing lifecycle state of the review",
    )

    processing_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of AI reply generation attempts — used to detect stuck reviews",
    )

    last_processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Last error message from a failed processing attempt",
    )

    # ── Alert Tracking ────────────────────────────────────────────────────────

    alert_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once a WhatsApp alert has been sent to the business owner",
    )

    alert_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the WhatsApp alert was sent (UTC)",
    )

    # ── Spam / Validation ─────────────────────────────────────────────────────

    is_spam: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the review was flagged as spam by review_validator.py",
    )

    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="False if the review failed validation and must be excluded from processing",
    )

    # ── Idempotency ───────────────────────────────────────────────────────────

    idempotency_key: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment="Idempotency key for reply generation — prevents duplicate AI calls",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        back_populates="reviews",
        lazy="selectin",
    )

    # ── Table-level Constraints and Indexes ───────────────────────────────────

    __table_args__ = (
        # Core idempotency constraint — same review never stored twice per business
        UniqueConstraint(
            "business_id",
            "google_review_id",
            name="uq_reviews_business_google_id",
        ),
        Index(
            "ix_reviews_business_id",
            "business_id",
        ),
        Index(
            "ix_reviews_status",
            "status",
        ),
        Index(
            "ix_reviews_sentiment",
            "sentiment",
        ),
        # Composite index for the most common scheduler query:
        # "give me all NEW reviews for this business"
        Index(
            "ix_reviews_business_status",
            "business_id",
            "status",
        ),
        Index(
            "ix_reviews_google_review_id",
            "google_review_id",
        ),
        Index(
            "ix_reviews_published_at",
            "published_at",
        ),
        Index(
            "ix_reviews_rating",
            "rating",
        ),
        # Composite index for alert threshold queries:
        # "count reviews with rating <= 2 for this business since date"
        Index(
            "ix_reviews_business_rating",
            "business_id",
            "rating",
        ),
        Index(
            "ix_reviews_is_valid_is_spam",
            "is_valid",
            "is_spam",
        ),
    )

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_positive(self) -> bool:
        """True if the review was classified as positive sentiment."""
        return self.sentiment == ReviewSentiment.POSITIVE

    @property
    def is_negative(self) -> bool:
        """True if the review was classified as negative sentiment."""
        return self.sentiment == ReviewSentiment.NEGATIVE

    @property
    def is_neutral(self) -> bool:
        """True if the review was classified as neutral sentiment."""
        return self.sentiment == ReviewSentiment.NEUTRAL

    @property
    def is_new(self) -> bool:
        """True if this review has not yet entered the processing pipeline."""
        return self.status == ReviewStatus.NEW

    @property
    def is_replied(self) -> bool:
        """True if an AI reply has been generated for this review."""
        return self.status == ReviewStatus.REPLIED

    @property
    def needs_processing(self) -> bool:
        """
        True if this review is eligible for AI reply generation.

        A review is eligible when:
        - It has not been spam-flagged
        - It passed validation
        - It has not already been replied to
        - It has not been explicitly skipped
        - It has not exhausted its processing attempt budget (max 3)
        """
        return (
            not self.is_spam
            and self.is_valid
            and self.status in {ReviewStatus.NEW, ReviewStatus.PROCESSING}
            and self.processing_attempts < 3
        )

    @property
    def has_text(self) -> bool:
        """True if the reviewer left written text alongside their rating."""
        return bool(self.review_text and len(self.review_text.strip()) > 0)

    @property
    def text_for_ai(self) -> str:
        """
        Return the review text safe for inclusion in an AI prompt.

        Returns the truncated version if truncation was applied, otherwise
        returns the original text. Returns an empty string if no text exists.
        """
        if self.review_text_truncated:
            return self.review_text_truncated
        return self.review_text or ""

    @property
    def star_display(self) -> str:
        """Return a star emoji representation of the rating (e.g., '★★★☆☆')."""
        filled = "★" * self.rating
        empty = "☆" * (5 - self.rating)
        return filled + empty

    def __repr__(self) -> str:
        return (
            f"<ReviewModel id={self.id} "
            f"business_id={self.business_id} "
            f"google_review_id='{self.google_review_id}' "
            f"rating={self.rating} "
            f"sentiment='{self.sentiment}' "
            f"status='{self.status}'>"
        )