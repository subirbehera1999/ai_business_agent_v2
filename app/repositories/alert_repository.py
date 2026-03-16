# ==============================================================================
# File: app/repositories/alert_repository.py
# Purpose: Repository class encapsulating all database operations for business
#          alerts. Alerts are system-generated events (rating drops, review
#          spikes, competitor changes, usage limits) that are stored here
#          before being dispatched via WhatsApp.
#
#          Core responsibilities:
#            - Idempotent alert creation (deduplication by business + type + date)
#            - Alert dispatch tracking (sent / failed states)
#            - Per-type and per-business alert retrieval
#            - Unsent alert querying for dispatch workers
#            - Alert history for admin dashboards and audit
#            - All queries enforce business_id tenant isolation
#
#          Alert deduplication strategy:
#            The same alert type must not be sent to a business more than
#            once per deduplication window (e.g., one RATING_DROP alert per
#            day). alert_manager.py enforces this by checking for recent
#            alerts of the same type before creating new ones.
# ==============================================================================

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import AlertSeverity, AlertType, ServiceName
from app.database.base import BaseModel

# ---------------------------------------------------------------------------
# Inline Alert Model definition
# The alert model is small enough to co-locate here as a lightweight
# alternative to a separate model file. alert_manager.py and all alert
# services import AlertModel from this module.
# ---------------------------------------------------------------------------
from sqlalchemy import Boolean, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel


class AlertModel(BaseModel):
    """
    Represents a single system-generated business alert.

    Each alert is a discrete event notification (rating drop, review spike,
    competitor change, usage limit reached, etc.) that is queued for
    WhatsApp delivery to the business owner.

    Inherits from BaseModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)

    Table:
        alerts

    Indexes:
        - ix_alerts_business_id               — tenant isolation
        - ix_alerts_alert_type                — filter by event type
        - ix_alerts_is_sent                   — unsent alert dispatch queue
        - ix_alerts_business_type_date        — deduplication window queries
        - ix_alerts_severity                  — filter critical alerts
        - ix_alerts_created_at                — chronological ordering
    """

    __tablename__ = "alerts"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    # ── Alert Classification ──────────────────────────────────────────────────

    alert_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Alert category (see AlertType enum in constants.py)",
    )

    severity: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=AlertSeverity.INFO,
        server_default=AlertSeverity.INFO,
        comment="Alert severity level: info / warning / critical",
    )

    # ── Alert Content ─────────────────────────────────────────────────────────

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Short alert headline for WhatsApp message header",
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full alert message body sent via WhatsApp",
    )

    # ── Contextual Metadata ───────────────────────────────────────────────────

    context_data: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Optional JSON string with alert-specific context. "
            "Examples: {'old_rating': 4.5, 'new_rating': 3.8}, "
            "{'review_count': 12, 'window_hours': 1}"
        ),
    )

    reference_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "Optional ID of the entity that triggered this alert. "
            "Example: review UUID for a negative review alert."
        ),
    )

    # ── Deduplication Window ──────────────────────────────────────────────────

    alert_date: Mapped[date] = mapped_column(
        nullable=False,
        comment="Calendar date of the alert — used for deduplication window queries",
    )

    deduplication_key: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment=(
            "Optional unique key to prevent duplicate alerts. "
            "Format: BUSINESS_{id}_ALERT_{type}_DATE_{date}. "
            "NULL for alert types that allow multiple per day."
        ),
    )

    # ── Dispatch State ────────────────────────────────────────────────────────

    is_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the WhatsApp message has been successfully delivered",
    )

    sent_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Timestamp when the WhatsApp alert was delivered (UTC)",
    )

    send_attempts: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of WhatsApp send attempts made for this alert",
    )

    last_send_error: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Last error message from a failed send attempt",
    )

    is_failed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if all send attempts have been exhausted without success",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        lazy="selectin",
    )

    # ── Table-level Constraints and Indexes ───────────────────────────────────

    __table_args__ = (
        UniqueConstraint(
            "deduplication_key",
            name="uq_alerts_deduplication_key",
        ),
        Index("ix_alerts_business_id", "business_id"),
        Index("ix_alerts_alert_type", "alert_type"),
        Index("ix_alerts_is_sent", "is_sent"),
        Index("ix_alerts_is_failed", "is_failed"),
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_created_at", "created_at"),
        # Primary deduplication query index:
        # "does a RATING_DROP alert already exist for this business today?"
        Index(
            "ix_alerts_business_type_date",
            "business_id",
            "alert_type",
            "alert_date",
        ),
        # Dispatch worker index:
        # "give me all unsent, non-failed alerts ordered oldest first"
        Index(
            "ix_alerts_is_sent_is_failed",
            "is_sent",
            "is_failed",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AlertModel id={self.id} "
            f"business_id={self.business_id} "
            f"type='{self.alert_type}' "
            f"severity='{self.severity}' "
            f"sent={self.is_sent}>"
        )


# Resolve forward reference for ForeignKey
from sqlalchemy import ForeignKey  # noqa: E402 — needed after AlertModel definition


# ==============================================================================
# Alert Repository
# ==============================================================================

class AlertRepository:
    """
    Handles all database operations for AlertModel.

    Session management (commit/rollback) is the responsibility of the
    caller. This repository only calls flush() to populate server defaults.

    All queries:
        - Filter by business_id for multi-tenant isolation
        - Apply LIMIT clauses — no unbounded result sets
        - Use deduplication_key to prevent duplicate alerts

    Usage:
        repo = AlertRepository()
        alert = await repo.create(db, business_id=..., alert_type=..., ...)
    """

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        db: AsyncSession,
        *,
        business_id: uuid.UUID,
        alert_type: str,
        title: str,
        message: str,
        severity: str = AlertSeverity.INFO,
        context_data: Optional[str] = None,
        reference_id: Optional[str] = None,
        alert_date: Optional[date] = None,
        deduplication_key: Optional[str] = None,
    ) -> AlertModel:
        """
        Create and persist a new alert record.

        If a deduplication_key is provided and a record with that key already
        exists, the existing record is returned without creating a duplicate.
        This ensures idempotent alert creation even when alert_manager.py
        is called multiple times within the same polling cycle.

        Args:
            db:                 Active async database session.
            business_id:        UUID of the owning business.
            alert_type:         Alert category (see AlertType enum).
            title:              Short headline for the WhatsApp message.
            message:            Full message body.
            severity:           Severity level (info/warning/critical).
            context_data:       Optional JSON string with alert context.
            reference_id:       Optional ID of the triggering entity.
            alert_date:         Alert calendar date (defaults to today UTC).
            deduplication_key:  Optional unique key to prevent duplicates.

        Returns:
            AlertModel: The newly created or existing (deduplicated) alert.

        Raises:
            SQLAlchemyError: On any database error.
        """
        today = alert_date or date.today()

        try:
            # If deduplication key provided — use upsert to prevent duplicates
            if deduplication_key:
                stmt = (
                    pg_insert(AlertModel)
                    .values(
                        business_id=business_id,
                        alert_type=alert_type,
                        title=title,
                        message=message,
                        severity=severity,
                        context_data=context_data,
                        reference_id=reference_id,
                        alert_date=today,
                        deduplication_key=deduplication_key,
                        is_sent=False,
                        send_attempts=0,
                        is_failed=False,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_alerts_deduplication_key"
                    )
                    .returning(AlertModel.id)
                )
                result = await db.execute(stmt)
                inserted_id = result.scalar_one_or_none()

                if inserted_id:
                    alert = await self.get_by_id(db, inserted_id)
                    logger.info(
                        "Alert created",
                        extra={
                            "service": ServiceName.ALERT,
                            "business_id": str(business_id),
                            "alert_id": str(inserted_id),
                            "alert_type": alert_type,
                            "severity": severity,
                        },
                    )
                    return alert
                else:
                    # Deduplicated — return existing alert
                    existing = await self.get_by_deduplication_key(
                        db, deduplication_key
                    )
                    logger.debug(
                        "Alert deduplicated — returning existing record",
                        extra={
                            "service": ServiceName.ALERT,
                            "business_id": str(business_id),
                            "alert_type": alert_type,
                            "deduplication_key": deduplication_key,
                        },
                    )
                    return existing

            # No deduplication key — plain insert (allows multiple per day)
            alert = AlertModel(
                business_id=business_id,
                alert_type=alert_type,
                title=title,
                message=message,
                severity=severity,
                context_data=context_data,
                reference_id=reference_id,
                alert_date=today,
                deduplication_key=None,
                is_sent=False,
                send_attempts=0,
                is_failed=False,
            )
            db.add(alert)
            await db.flush()

            logger.info(
                "Alert created",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "alert_id": str(alert.id),
                    "alert_type": alert_type,
                    "severity": severity,
                },
            )
            return alert

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to create alert",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "alert_type": alert_type,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Single Record ───────────────────────────────────────────────────

    async def get_by_id(
        self,
        db: AsyncSession,
        alert_id: uuid.UUID,
    ) -> Optional[AlertModel]:
        """
        Fetch a single alert by its primary key.

        Args:
            db:        Active async database session.
            alert_id:  UUID primary key of the alert.

        Returns:
            AlertModel if found, else None.
        """
        try:
            result = await db.execute(
                select(AlertModel).where(AlertModel.id == alert_id)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch alert by ID",
                extra={
                    "service": ServiceName.ALERT,
                    "alert_id": str(alert_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_deduplication_key(
        self,
        db: AsyncSession,
        deduplication_key: str,
    ) -> Optional[AlertModel]:
        """
        Fetch an alert by its deduplication key.

        Used after an ON CONFLICT DO NOTHING upsert to retrieve the
        existing record when a duplicate was detected.

        Args:
            db:                 Active async database session.
            deduplication_key:  The unique deduplication key string.

        Returns:
            AlertModel if found, else None.
        """
        try:
            result = await db.execute(
                select(AlertModel).where(
                    AlertModel.deduplication_key == deduplication_key
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch alert by deduplication key",
                extra={
                    "service": ServiceName.ALERT,
                    "deduplication_key": deduplication_key,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Collections ─────────────────────────────────────────────────────

    async def get_unsent(
        self,
        db: AsyncSession,
        business_id: Optional[uuid.UUID] = None,
        limit: int = 20,
    ) -> list[AlertModel]:
        """
        Fetch unsent, non-failed alerts awaiting WhatsApp delivery.

        Called by the alert dispatch worker to build the send queue.
        Ordered oldest first to ensure alerts are delivered in sequence.

        Args:
            db:           Active async database session.
            business_id:  Optional UUID to filter by a single business.
                          If None, returns unsent alerts platform-wide.
            limit:        Maximum records to return.

        Returns:
            list[AlertModel]: Unsent alerts ordered oldest first.
        """
        try:
            conditions = [
                AlertModel.is_sent.is_(False),
                AlertModel.is_failed.is_(False),
            ]
            if business_id:
                conditions.append(AlertModel.business_id == business_id)

            result = await db.execute(
                select(AlertModel)
                .where(and_(*conditions))
                .order_by(AlertModel.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch unsent alerts",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id) if business_id else "all",
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
    ) -> list[AlertModel]:
        """
        Fetch recent alerts for a business, ordered newest first.

        Used for admin dashboards and business alert history views.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            limit:        Maximum records to return.
            offset:       Pagination offset.

        Returns:
            list[AlertModel]: Recent alerts, newest first.
        """
        try:
            result = await db.execute(
                select(AlertModel)
                .where(AlertModel.business_id == business_id)
                .order_by(desc(AlertModel.created_at))
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch recent alerts for business",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_type_since(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        alert_type: str,
        since: datetime,
        limit: int = 20,
    ) -> list[AlertModel]:
        """
        Fetch alerts of a specific type for a business since a timestamp.

        Used by alert_manager.py to check whether a specific alert type
        was recently sent before creating a duplicate. This is the primary
        deduplication lookup for alert types that do not use a static key.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            alert_type:   Alert type to filter by (see AlertType enum).
            since:        Lower bound timestamp — only return alerts after this.
            limit:        Maximum records to return.

        Returns:
            list[AlertModel]: Matching alerts within the time window.
        """
        try:
            result = await db.execute(
                select(AlertModel)
                .where(
                    AlertModel.business_id == business_id,
                    AlertModel.alert_type == alert_type,
                    AlertModel.created_at >= since,
                )
                .order_by(desc(AlertModel.created_at))
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch alerts by type since timestamp",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "alert_type": alert_type,
                    "since": since.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def get_critical_unsent(
        self,
        db: AsyncSession,
        limit: int = 50,
    ) -> list[AlertModel]:
        """
        Fetch all unsent critical-severity alerts across all businesses.

        Used by admin_notification_service.py to escalate high-priority
        undelivered alerts to the admin WhatsApp number.

        Args:
            db:     Active async database session.
            limit:  Maximum records to return.

        Returns:
            list[AlertModel]: Unsent critical alerts, oldest first.
        """
        try:
            result = await db.execute(
                select(AlertModel)
                .where(
                    AlertModel.is_sent.is_(False),
                    AlertModel.is_failed.is_(False),
                    AlertModel.severity == AlertSeverity.CRITICAL,
                )
                .order_by(AlertModel.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch critical unsent alerts",
                extra={
                    "service": ServiceName.ALERT,
                    "error": str(exc),
                },
            )
            raise

    # ── Deduplication Check ────────────────────────────────────────────────────

    async def exists_for_type_on_date(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        alert_type: str,
        alert_date: date,
    ) -> bool:
        """
        Check whether an alert of the given type already exists for this
        business on the given calendar date.

        Used by alert_manager.py as a fast pre-check before creating
        certain alert types to prevent same-day duplicates.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            alert_type:   Alert category to check.
            alert_date:   Calendar date to scope the check.

        Returns:
            bool: True if a matching alert already exists on that date.
        """
        try:
            result = await db.execute(
                select(func.count(AlertModel.id)).where(
                    AlertModel.business_id == business_id,
                    AlertModel.alert_type == alert_type,
                    AlertModel.alert_date == alert_date,
                )
            )
            return result.scalar_one() > 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to check alert existence for date",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "alert_type": alert_type,
                    "alert_date": str(alert_date),
                    "error": str(exc),
                },
            )
            raise

    # ── Update — Dispatch Tracking ─────────────────────────────────────────────

    async def mark_sent(
        self,
        db: AsyncSession,
        alert_id: uuid.UUID,
    ) -> None:
        """
        Mark an alert as successfully delivered via WhatsApp.

        Called by whatsapp_service.py after confirmed delivery.

        Args:
            db:        Active async database session.
            alert_id:  UUID of the alert to mark as sent.
        """
        try:
            await db.execute(
                update(AlertModel)
                .where(AlertModel.id == alert_id)
                .values(
                    is_sent=True,
                    sent_at=datetime.now(timezone.utc),
                    send_attempts=AlertModel.send_attempts + 1,
                )
            )
            await db.flush()

            logger.info(
                "Alert marked as sent",
                extra={
                    "service": ServiceName.ALERT,
                    "alert_id": str(alert_id),
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark alert as sent",
                extra={
                    "service": ServiceName.ALERT,
                    "alert_id": str(alert_id),
                    "error": str(exc),
                },
            )
            raise

    async def record_send_failure(
        self,
        db: AsyncSession,
        alert_id: uuid.UUID,
        error_message: str,
        mark_failed: bool = False,
    ) -> None:
        """
        Record a failed WhatsApp send attempt for an alert.

        Increments the send_attempts counter and stores the error message.
        If mark_failed is True, sets is_failed=True to remove the alert
        from the dispatch queue after all retries are exhausted.

        Args:
            db:            Active async database session.
            alert_id:      UUID of the alert.
            error_message: Error description from the failed send attempt.
            mark_failed:   If True, marks the alert as permanently failed.
        """
        try:
            values: dict = {
                "send_attempts": AlertModel.send_attempts + 1,
                "last_send_error": error_message[:500],
            }
            if mark_failed:
                values["is_failed"] = True

            await db.execute(
                update(AlertModel)
                .where(AlertModel.id == alert_id)
                .values(**values)
            )
            await db.flush()

            log_level = logger.warning if mark_failed else logger.debug
            log_level(
                "Alert send failure recorded",
                extra={
                    "service": ServiceName.ALERT,
                    "alert_id": str(alert_id),
                    "mark_failed": mark_failed,
                    "error_message": error_message,
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to record alert send failure",
                extra={
                    "service": ServiceName.ALERT,
                    "alert_id": str(alert_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Aggregates ────────────────────────────────────────────────────────────

    async def count_by_type_since(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        alert_type: str,
        since: datetime,
    ) -> int:
        """
        Count alerts of a specific type for a business since a timestamp.

        Used by alert_manager.py for frequency-based suppression — e.g.,
        do not send more than N competitor alerts in a 24-hour window.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            alert_type:   Alert category to count.
            since:        Lower bound timestamp.

        Returns:
            int: Count of matching alerts in the time window.
        """
        try:
            result = await db.execute(
                select(func.count(AlertModel.id)).where(
                    AlertModel.business_id == business_id,
                    AlertModel.alert_type == alert_type,
                    AlertModel.created_at >= since,
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count alerts by type since timestamp",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "alert_type": alert_type,
                    "since": since.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def count_unsent_by_business(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> int:
        """
        Count the number of unsent alerts for a specific business.

        Used by admin dashboards to surface businesses with a backlog
        of undelivered alerts.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.

        Returns:
            int: Number of pending unsent alerts for the business.
        """
        try:
            result = await db.execute(
                select(func.count(AlertModel.id)).where(
                    AlertModel.business_id == business_id,
                    AlertModel.is_sent.is_(False),
                    AlertModel.is_failed.is_(False),
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count unsent alerts for business",
                extra={
                    "service": ServiceName.ALERT,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise