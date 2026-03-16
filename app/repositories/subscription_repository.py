# ==============================================================================
# File: app/repositories/subscription_repository.py
# Purpose: Repository class encapsulating all database operations for the
#          SubscriptionModel. This is the only layer permitted to query the
#          subscriptions table directly.
#
#          ONE SUBSCRIPTION TIER ONLY.
#          No plan tiers — no plan column on the model.
#          The only choice is billing_cycle: "monthly" or "annual".
#
#          Critical invariants enforced here:
#            - Only one ACTIVE or TRIAL subscription per business at any time
#            - All queries filter by business_id for tenant isolation
#            - Soft-deleted records are excluded from all standard queries
#            - Expiry checker queries use indexed expires_at for performance
# ==============================================================================

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName, SubscriptionStatus
from app.database.models.subscription_model import SubscriptionModel

logger = logging.getLogger(ServiceName.SUBSCRIPTION)


# ==============================================================================
# Subscription Repository
# ==============================================================================

class SubscriptionRepository:
    """
    Handles all database operations for SubscriptionModel.

    Session management (commit/rollback) is the caller's responsibility.
    This repository only calls flush() to populate server defaults.

    All queries:
        - Filter by business_id for multi-tenant isolation
        - Exclude soft-deleted records unless explicitly stated
        - Apply LIMIT clauses — no unbounded result sets

    Usage:
        repo = SubscriptionRepository()
        sub = await repo.get_active(db, business_id)
    """

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        db: AsyncSession,
        *,
        business_id: uuid.UUID,
        billing_cycle: str,
        billing_cycle_months: int = 1,
        amount: float = 0.0,
        currency: str = "INR",
        status: str = SubscriptionStatus.PENDING,
        starts_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        trial_ends_at: Optional[datetime] = None,
        razorpay_subscription_id: Optional[str] = None,
        razorpay_plan_id: Optional[str] = None,
        auto_renew: bool = True,
        payment_record_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> SubscriptionModel:
        """
        Create and persist a new subscription record.

        One tier only — full access on any active subscription.
        billing_cycle determines duration and price only.

        Args:
            db:                       Active async database session.
            business_id:              UUID of the owning business.
            billing_cycle:            "monthly" or "annual".
            billing_cycle_months:     Duration in months (1 or 12).
            amount:                   Billing amount charged (INR).
            currency:                 ISO 4217 currency code (default: INR).
            status:                   Initial lifecycle status (default: PENDING).
            starts_at:                Optional activation timestamp.
            expires_at:               Optional expiry timestamp.
            trial_ends_at:            Optional trial end timestamp.
            razorpay_subscription_id: Razorpay subscription object ID.
            razorpay_plan_id:         Razorpay plan ID.
            auto_renew:               Whether to auto-renew (default: True).
            payment_record_id:        Internal payment record UUID (for notes).
            notes:                    Internal admin notes (never sent to AI).

        Returns:
            SubscriptionModel: The newly created subscription instance.

        Raises:
            IntegrityError: On unique constraint violations.
            SQLAlchemyError: On any other database error.
        """
        try:
            subscription = SubscriptionModel(
                business_id=business_id,
                billing_cycle=billing_cycle,
                billing_cycle_months=billing_cycle_months,
                amount=amount,
                currency=currency,
                status=status,
                starts_at=starts_at,
                expires_at=expires_at,
                trial_ends_at=trial_ends_at,
                razorpay_subscription_id=razorpay_subscription_id,
                razorpay_plan_id=razorpay_plan_id,
                auto_renew=auto_renew,
                notes=notes,
            )
            db.add(subscription)
            await db.flush()

            logger.info(
                "Subscription record created",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "subscription_id": str(subscription.id),
                    "billing_cycle": billing_cycle,
                    "status": status,
                },
            )
            return subscription

        except IntegrityError as exc:
            logger.error(
                "Subscription creation failed — integrity error",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

        except SQLAlchemyError as exc:
            logger.error(
                "Subscription creation failed — database error",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Single Record ───────────────────────────────────────────────────

    async def get_by_id(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
    ) -> Optional[SubscriptionModel]:
        """
        Fetch a single subscription by primary key.

        Args:
            db:               Active async database session.
            subscription_id:  UUID primary key.

        Returns:
            SubscriptionModel if found and not soft-deleted, else None.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel).where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch subscription by ID",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_active(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> Optional[SubscriptionModel]:
        """
        Fetch the currently active or trial subscription for a business.

        Alias used throughout the service layer. Delegates to
        get_active_by_business_id() for full implementation.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.

        Returns:
            SubscriptionModel if an active/trial subscription exists, else None.
        """
        return await self.get_active_by_business_id(db=db, business_id=business_id)

    async def get_active_by_business_id(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> Optional[SubscriptionModel]:
        """
        Fetch the currently active or trial subscription for a business.

        Primary method called by plan_manager.py and rate_limiter.py
        before every gated operation.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.

        Returns:
            SubscriptionModel if an active/trial subscription exists, else None.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel)
                .where(
                    SubscriptionModel.business_id == business_id,
                    SubscriptionModel.is_deleted.is_(False),
                    SubscriptionModel.status.in_(
                        [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
                    ),
                )
                .order_by(SubscriptionModel.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch active subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_razorpay_subscription_id(
        self,
        db: AsyncSession,
        razorpay_subscription_id: str,
    ) -> Optional[SubscriptionModel]:
        """
        Fetch a subscription by its Razorpay subscription ID.

        Called by webhook_handler.py when processing Razorpay subscription
        events to locate the corresponding internal record.

        Args:
            db:                       Active async database session.
            razorpay_subscription_id: Razorpay subscription object ID.

        Returns:
            SubscriptionModel if found, else None.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel).where(
                    SubscriptionModel.razorpay_subscription_id == razorpay_subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch subscription by Razorpay ID",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "razorpay_subscription_id": razorpay_subscription_id,
                    "error": str(exc),
                },
            )
            raise

    async def get_latest_by_business_id(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> Optional[SubscriptionModel]:
        """
        Fetch the most recently created subscription for a business
        regardless of status. Used for history display and renewal flows.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.

        Returns:
            Most recent SubscriptionModel or None.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel)
                .where(
                    SubscriptionModel.business_id == business_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .order_by(SubscriptionModel.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch latest subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Collections ─────────────────────────────────────────────────────

    async def get_all_by_business_id(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SubscriptionModel]:
        """
        Fetch paginated subscription history for a business.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business.
            limit:        Maximum records to return.
            offset:       Pagination offset.

        Returns:
            list[SubscriptionModel]: Subscriptions ordered newest first.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel)
                .where(
                    SubscriptionModel.business_id == business_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .order_by(SubscriptionModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch subscriptions for business",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_expiring_soon(
        self,
        db: AsyncSession,
        before: datetime,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SubscriptionModel]:
        """
        Fetch active subscriptions expiring before the given timestamp.

        Used by expiry_checker.py to identify subscriptions that require
        a renewal reminder WhatsApp message.

        Args:
            db:      Active async database session.
            before:  Return subscriptions expiring before this timestamp.
            limit:   Maximum records to return per batch.
            offset:  Pagination offset.

        Returns:
            list[SubscriptionModel]: Active subscriptions expiring soon.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel)
                .where(
                    SubscriptionModel.is_deleted.is_(False),
                    SubscriptionModel.status == SubscriptionStatus.ACTIVE,
                    SubscriptionModel.expires_at.isnot(None),
                    SubscriptionModel.expires_at <= before,
                    SubscriptionModel.renewal_reminder_sent.is_(False),
                )
                .order_by(SubscriptionModel.expires_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch expiring subscriptions",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "before": before.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def get_expired_active(
        self,
        db: AsyncSession,
        as_of: datetime,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SubscriptionModel]:
        """
        Fetch subscriptions still marked ACTIVE but past their expiry date.

        Used by expiry_checker.py to transition overdue subscriptions to
        EXPIRED status and deactivate the associated business accounts.

        Args:
            db:      Active async database session.
            as_of:   Current timestamp — subscriptions with expires_at
                     before this value are considered expired.
            limit:   Maximum records to return per batch.
            offset:  Pagination offset.

        Returns:
            list[SubscriptionModel]: Overdue active subscriptions.
        """
        try:
            result = await db.execute(
                select(SubscriptionModel)
                .where(
                    SubscriptionModel.is_deleted.is_(False),
                    SubscriptionModel.status == SubscriptionStatus.ACTIVE,
                    SubscriptionModel.expires_at.isnot(None),
                    SubscriptionModel.expires_at < as_of,
                )
                .order_by(SubscriptionModel.expires_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch expired active subscriptions",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "as_of": as_of.isoformat(),
                    "error": str(exc),
                },
            )
            raise

    async def count_active(self, db: AsyncSession) -> int:
        """
        Return the total count of active and trial subscriptions.
        Used by admin health reports and system monitoring.

        Args:
            db: Active async database session.

        Returns:
            int: Total count of usable subscriptions platform-wide.
        """
        try:
            result = await db.execute(
                select(func.count(SubscriptionModel.id)).where(
                    SubscriptionModel.is_deleted.is_(False),
                    SubscriptionModel.status.in_(
                        [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
                    ),
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count active subscriptions",
                extra={"service": ServiceName.SUBSCRIPTION, "error": str(exc)},
            )
            raise

    # ── Update — Status Transitions ────────────────────────────────────────────

    async def activate(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
        starts_at: datetime,
        expires_at: datetime,
        razorpay_subscription_id: Optional[str] = None,
        last_payment_id: Optional[str] = None,
    ) -> Optional[SubscriptionModel]:
        """
        Transition a subscription to ACTIVE status.

        Called by payment_service.py after successful payment verification.
        Sets the validity window and Razorpay identifiers atomically.

        Args:
            db:                       Active async database session.
            subscription_id:          UUID of the subscription to activate.
            starts_at:                Activation timestamp.
            expires_at:               Expiry timestamp.
            razorpay_subscription_id: Razorpay subscription ID to record.
            last_payment_id:          Razorpay payment ID of triggering payment.

        Returns:
            Updated SubscriptionModel if found, else None.
        """
        fields: dict = {
            "status": SubscriptionStatus.ACTIVE,
            "starts_at": starts_at,
            "expires_at": expires_at,
        }
        if razorpay_subscription_id is not None:
            fields["razorpay_subscription_id"] = razorpay_subscription_id
        if last_payment_id is not None:
            fields["last_payment_id"] = last_payment_id

        try:
            await db.execute(
                update(SubscriptionModel)
                .where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .values(**fields)
            )
            await db.flush()
            updated = await self.get_by_id(db, subscription_id)
            logger.info(
                "Subscription activated",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "expires_at": expires_at.isoformat(),
                },
            )
            return updated

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to activate subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    async def extend_subscription(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
        new_end_date: datetime,
        payment_record_id: Optional[str] = None,
        last_payment_id: Optional[str] = None,
    ) -> Optional[SubscriptionModel]:
        """
        Extend an existing active subscription's expiry date.

        Called by payment_service.py when a renewal payment is captured.
        Resets the renewal reminder flag so a fresh reminder is sent
        before the new expiry.

        No plan change is possible — one tier only.

        Args:
            db:               Active async database session.
            subscription_id:  UUID of the subscription to extend.
            new_end_date:     New expiry timestamp after extension.
            payment_record_id: Internal payment record UUID (for audit).
            last_payment_id:  Razorpay payment ID of the renewal charge.

        Returns:
            Updated SubscriptionModel if found, else None.
        """
        fields: dict = {
            "expires_at": new_end_date,
            "status": SubscriptionStatus.ACTIVE,
            "renewal_reminder_sent": False,
        }
        if last_payment_id is not None:
            fields["last_payment_id"] = last_payment_id

        try:
            await db.execute(
                update(SubscriptionModel)
                .where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .values(**fields)
            )
            await db.flush()
            updated = await self.get_by_id(db, subscription_id)
            logger.info(
                "Subscription extended",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "new_end_date": new_end_date.isoformat(),
                },
            )
            return updated

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to extend subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_expired(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
    ) -> None:
        """
        Transition a subscription to EXPIRED status.
        Called by expiry_checker.py for subscriptions past their expires_at.
        """
        try:
            await db.execute(
                update(SubscriptionModel)
                .where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .values(status=SubscriptionStatus.EXPIRED)
            )
            await db.flush()
            logger.info(
                "Subscription marked expired",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark subscription expired",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_cancelled(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
        cancelled_at: Optional[datetime] = None,
    ) -> None:
        """
        Transition a subscription to CANCELLED status.
        """
        cancelled_at = cancelled_at or datetime.now(timezone.utc)
        try:
            await db.execute(
                update(SubscriptionModel)
                .where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .values(
                    status=SubscriptionStatus.CANCELLED,
                    cancelled_at=cancelled_at,
                    auto_renew=False,
                )
            )
            await db.flush()
            logger.info(
                "Subscription marked cancelled",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "cancelled_at": cancelled_at.isoformat(),
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to cancel subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_renewal_reminder_sent(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
    ) -> None:
        """
        Set the renewal_reminder_sent flag to prevent duplicate reminders.
        Called by expiry_checker.py after the WhatsApp reminder is confirmed sent.
        """
        try:
            await db.execute(
                update(SubscriptionModel)
                .where(
                    SubscriptionModel.id == subscription_id,
                    SubscriptionModel.is_deleted.is_(False),
                )
                .values(renewal_reminder_sent=True)
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark renewal reminder sent",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Access Validation ──────────────────────────────────────────────────────

    async def has_active_subscription(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> bool:
        """
        Check whether a business has any usable subscription.

        Uses COUNT() to avoid loading the full model for a boolean check.
        Called by plan_manager.py as a fast pre-check.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business to check.

        Returns:
            bool: True if the business has an ACTIVE or TRIAL subscription.
        """
        try:
            result = await db.execute(
                select(func.count(SubscriptionModel.id)).where(
                    SubscriptionModel.business_id == business_id,
                    SubscriptionModel.is_deleted.is_(False),
                    SubscriptionModel.status.in_(
                        [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
                    ),
                )
            )
            return result.scalar_one() > 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to check active subscription existence",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Soft Delete ────────────────────────────────────────────────────────────

    async def soft_delete(
        self,
        db: AsyncSession,
        subscription_id: uuid.UUID,
    ) -> bool:
        """
        Soft-delete a subscription record.

        Args:
            db:               Active async database session.
            subscription_id:  UUID of the subscription to soft-delete.

        Returns:
            bool: True if found and deleted, False if not found.
        """
        try:
            subscription = await self.get_by_id(db, subscription_id)
            if not subscription:
                return False

            subscription.soft_delete()
            await db.flush()
            logger.info(
                "Subscription soft-deleted",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                },
            )
            return True

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to soft-delete subscription",
                extra={
                    "service": ServiceName.SUBSCRIPTION,
                    "subscription_id": str(subscription_id),
                    "error": str(exc),
                },
            )
            raise