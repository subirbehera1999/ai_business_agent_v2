# ==============================================================================
# File: app/subscriptions/subscription_service.py
# Purpose: Manages the full subscription lifecycle for all businesses.
#
#          There is ONE subscription. Every subscribed business gets full
#          access to all features. The only variable is billing cycle:
#            - MONTHLY  (pay every month)
#            - ANNUAL   (pay once per year, discounted)
#
#          Billing cycle is a pricing choice, not a feature tier.
#          There is no upgrade/downgrade between plan tiers because
#          there is only one plan.
#
#          Responsibilities:
#            1. create_subscription()
#               - Creates a new subscription after payment confirmation
#               - Sets start_date and end_date based on billing cycle
#               - Called by payment_service after payment.captured webhook
#               - Idempotent: extends existing subscription if one exists
#
#            2. get_active_subscription()
#               - Returns the current active subscription for a business
#               - Returns None if expired or no subscription exists
#               - Used by plan_manager.py to verify subscription before
#                 checking usage limits
#
#            3. renew_subscription()
#               - Extends the end_date by one billing cycle
#               - Called by payment_service on a renewal payment
#               - Billing cycle preserved from the existing subscription
#
#            4. cancel_subscription()
#               - Marks subscription as CANCELLED
#               - Business retains full access until end_date
#               - No immediate feature cutoff on cancellation
#
#            5. check_and_expire_subscriptions()
#               - Called by expiry_checker.py daily scheduler
#               - Marks ACTIVE/CANCELLED subscriptions as EXPIRED
#                 when end_date has passed
#               - Sends renewal reminders 3 days before expiry
#               - Sends expiry notice on the day of expiry
#               - Processes in batches (performance contract)
#
#            6. get_subscription_summary()
#               - Returns human-readable status for API responses
#               - Includes days remaining, billing cycle, dates
#
#          Multi-tenant:
#            All queries filtered by business_id.
#            No cross-business subscription access permitted.
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import (
    BillingCycle,
    ServiceName,
    SubscriptionStatus,
)
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.utils.time_utils import (
    compute_subscription_end_date,
    days_until,
    now_utc,
    today_local,
)

logger = logging.getLogger(ServiceName.SUBSCRIPTIONS)

# ---------------------------------------------------------------------------
# Renewal reminder threshold — days before expiry
# ---------------------------------------------------------------------------
RENEWAL_REMINDER_DAYS: int = 3

# Batch size for expiry check processing
EXPIRY_CHECK_BATCH_SIZE: int = 20


# ==============================================================================
# Output dataclasses
# ==============================================================================

@dataclass
class SubscriptionSummary:
    """
    Human-readable subscription status for API responses.

    Attributes:
        business_id:      Business UUID.
        has_active_plan:  True if subscription is currently active.
        status:           SubscriptionStatus constant.
        billing_cycle:    "monthly" or "annual".
        start_date:       Subscription start (UTC ISO string).
        end_date:         Subscription end (UTC ISO string).
        days_remaining:   Days until expiry (0 if expired/none).
        is_expiring_soon: True if within RENEWAL_REMINDER_DAYS of expiry.
    """
    business_id: str
    has_active_plan: bool
    status: str
    billing_cycle: str
    start_date: Optional[str]
    end_date: Optional[str]
    days_remaining: int
    is_expiring_soon: bool


@dataclass
class SubscriptionOperationResult:
    """
    Result of a subscription lifecycle operation.

    Attributes:
        success:         True if the operation completed successfully.
        subscription_id: UUID of the affected subscription record.
        message:         Human-readable outcome description.
        error:           Error message if success=False.
    """
    success: bool
    subscription_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExpiryCheckResult:
    """
    Result of a batch subscription expiry check run.

    Attributes:
        checked:       Number of active subscriptions evaluated.
        expired:       Number newly marked as EXPIRED.
        reminder_sent: Number of renewal reminders dispatched.
        errors:        Number of subscriptions that failed processing.
    """
    checked: int = 0
    expired: int = 0
    reminder_sent: int = 0
    errors: int = 0


# ==============================================================================
# Subscription Service
# ==============================================================================

class SubscriptionService:
    """
    Manages the full subscription lifecycle for all businesses.

    One subscription type. Full access to all features for any
    subscribed business. Billing cycle (monthly/annual) is a
    pricing choice only — it does not affect feature access.

    Usage:
        service = SubscriptionService(
            subscription_repo=subscription_repo,
            business_repo=business_repo,
            whatsapp_service=whatsapp_service,
        )

        result = await service.create_subscription(
            db=db,
            business_id="uuid",
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id="payment-uuid",
        )
    """

    def __init__(
        self,
        subscription_repo: SubscriptionRepository,
        business_repo: BusinessRepository,
        whatsapp_service,           # WhatsAppService
    ) -> None:
        self._sub_repo = subscription_repo
        self._biz_repo = business_repo
        self._whatsapp = whatsapp_service

    # ------------------------------------------------------------------
    # 1. Create subscription
    # ------------------------------------------------------------------

    async def create_subscription(
        self,
        db: AsyncSession,
        business_id: str,
        billing_cycle: str,
        payment_record_id: str,
        start_date: Optional[datetime] = None,
    ) -> SubscriptionOperationResult:
        """
        Create a new active subscription for a business.

        Called by payment_service.py after payment.captured is verified.
        Idempotent: if the business already has an active subscription,
        extends it rather than creating a duplicate.

        Args:
            db:                AsyncSession.
            business_id:       Business UUID.
            billing_cycle:     BillingCycle.MONTHLY or ANNUAL.
            payment_record_id: UUID of the confirming payment record.
            start_date:        Override start date (default: now UTC).

        Returns:
            SubscriptionOperationResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.SUBSCRIPTIONS,
            "business_id": business_id,
            "billing_cycle": billing_cycle,
        }

        try:
            effective_start = start_date or now_utc()
            end_date = compute_subscription_end_date(
                start=effective_start,
                billing_cycle=billing_cycle,
            )

            # Idempotency: extend existing active subscription
            existing = await self._sub_repo.get_active(
                db=db, business_id=business_id
            )
            if existing:
                await self._sub_repo.extend_subscription(
                    db=db,
                    subscription_id=str(existing.id),
                    new_end_date=end_date,
                    payment_record_id=payment_record_id,
                )
                await db.commit()
                logger.info(
                    "Subscription extended",
                    extra={
                        **log_extra,
                        "subscription_id": str(existing.id),
                        "new_end_date": end_date.isoformat(),
                    },
                )
                return SubscriptionOperationResult(
                    success=True,
                    subscription_id=str(existing.id),
                    message=f"Subscription extended to {end_date.date().isoformat()}",
                )

            # Create new subscription record
            sub = await self._sub_repo.create(
                db=db,
                business_id=business_id,
                billing_cycle=billing_cycle,
                start_date=effective_start,
                end_date=end_date,
                status=SubscriptionStatus.ACTIVE,
                payment_record_id=payment_record_id,
            )
            await db.commit()

            logger.info(
                "Subscription created",
                extra={
                    **log_extra,
                    "subscription_id": str(sub.id),
                    "end_date": end_date.isoformat(),
                },
            )
            return SubscriptionOperationResult(
                success=True,
                subscription_id=str(sub.id),
                message=(
                    f"Subscription activated until "
                    f"{end_date.date().isoformat()}"
                ),
            )

        except Exception as exc:
            await db.rollback()
            logger.error(
                "create_subscription failed",
                extra={**log_extra, "error": str(exc)},
            )
            return SubscriptionOperationResult(
                success=False,
                error=f"Failed to create subscription: {exc}",
            )

    # ------------------------------------------------------------------
    # 2. Get active subscription
    # ------------------------------------------------------------------

    async def get_active_subscription(
        self,
        db: AsyncSession,
        business_id: str,
    ):
        """
        Return the current active subscription for a business.

        Returns None if:
          - No subscription record exists
          - Subscription has expired (end_date < now)
          - Subscription status is CANCELLED or EXPIRED

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.

        Returns:
            SubscriptionModel or None.
        """
        return await self._sub_repo.get_active(db=db, business_id=business_id)

    # ------------------------------------------------------------------
    # 3. Renew subscription
    # ------------------------------------------------------------------

    async def renew_subscription(
        self,
        db: AsyncSession,
        business_id: str,
        payment_record_id: str,
        billing_cycle: Optional[str] = None,
    ) -> SubscriptionOperationResult:
        """
        Renew (extend) an existing subscription after a renewal payment.

        Extends end_date by one billing cycle. If billing_cycle is not
        provided, the existing cycle is preserved. A business can switch
        between monthly and annual at renewal by providing the new cycle.

        Args:
            db:                AsyncSession.
            business_id:       Business UUID.
            payment_record_id: UUID of the renewal payment record.
            billing_cycle:     New billing cycle (optional — keeps existing).

        Returns:
            SubscriptionOperationResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.SUBSCRIPTIONS,
            "business_id": business_id,
        }

        try:
            existing = await self._sub_repo.get_active(
                db=db, business_id=business_id
            )

            if not existing:
                # No active subscription — treat as a fresh create
                cycle = billing_cycle or BillingCycle.MONTHLY
                return await self.create_subscription(
                    db=db,
                    business_id=business_id,
                    billing_cycle=cycle,
                    payment_record_id=payment_record_id,
                )

            # Use provided cycle or fall back to existing
            cycle = billing_cycle or existing.billing_cycle or BillingCycle.MONTHLY

            # Extend from the later of (now, current end_date) to avoid
            # penalising early renewals
            extend_from = max(
                now_utc(),
                existing.end_date.replace(tzinfo=timezone.utc)
                if existing.end_date and existing.end_date.tzinfo is None
                else (existing.end_date or now_utc()),
            )
            new_end_date = compute_subscription_end_date(
                start=extend_from,
                billing_cycle=cycle,
            )

            await self._sub_repo.extend_subscription(
                db=db,
                subscription_id=str(existing.id),
                new_end_date=new_end_date,
                payment_record_id=payment_record_id,
                billing_cycle=cycle,
            )
            await db.commit()

            logger.info(
                "Subscription renewed",
                extra={
                    **log_extra,
                    "subscription_id": str(existing.id),
                    "new_end_date": new_end_date.isoformat(),
                    "cycle": cycle,
                },
            )

            await self._notify_renewal_confirmed(
                business_id=business_id,
                new_end_date=new_end_date.date().isoformat(),
                billing_cycle=cycle,
            )

            return SubscriptionOperationResult(
                success=True,
                subscription_id=str(existing.id),
                message=f"Subscription renewed until {new_end_date.date().isoformat()}",
            )

        except Exception as exc:
            await db.rollback()
            logger.error(
                "renew_subscription failed",
                extra={**log_extra, "error": str(exc)},
            )
            return SubscriptionOperationResult(
                success=False,
                error=f"Renewal failed: {exc}",
            )

    # ------------------------------------------------------------------
    # 4. Cancel subscription
    # ------------------------------------------------------------------

    async def cancel_subscription(
        self,
        db: AsyncSession,
        business_id: str,
        reason: Optional[str] = None,
    ) -> SubscriptionOperationResult:
        """
        Cancel a business subscription.

        Access continues until end_date — no immediate feature cutoff.
        Status set to CANCELLED; expiry_checker marks it EXPIRED on end_date.

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.
            reason:       Optional cancellation reason for audit trail.

        Returns:
            SubscriptionOperationResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.SUBSCRIPTIONS,
            "business_id": business_id,
        }

        try:
            existing = await self._sub_repo.get_active(
                db=db, business_id=business_id
            )

            if not existing:
                return SubscriptionOperationResult(
                    success=False,
                    error="No active subscription found to cancel.",
                )

            await self._sub_repo.update_status(
                db=db,
                subscription_id=str(existing.id),
                status=SubscriptionStatus.CANCELLED,
                cancellation_reason=reason,
                cancelled_at=now_utc(),
            )
            await db.commit()

            end_label = (
                existing.end_date.date().isoformat()
                if existing.end_date else "end of cycle"
            )

            logger.info(
                "Subscription cancelled",
                extra={
                    **log_extra,
                    "subscription_id": str(existing.id),
                    "access_until": end_label,
                    "reason": reason or "not provided",
                },
            )

            await self._notify_cancellation(
                business_id=business_id,
                access_until=end_label,
            )

            return SubscriptionOperationResult(
                success=True,
                subscription_id=str(existing.id),
                message=(
                    f"Subscription cancelled. "
                    f"Full access continues until {end_label}."
                ),
            )

        except Exception as exc:
            await db.rollback()
            logger.error(
                "cancel_subscription failed",
                extra={**log_extra, "error": str(exc)},
            )
            return SubscriptionOperationResult(
                success=False,
                error=f"Cancellation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # 5. Expiry check (called by expiry_checker.py)
    # ------------------------------------------------------------------

    async def check_and_expire_subscriptions(
        self,
        db: AsyncSession,
    ) -> ExpiryCheckResult:
        """
        Scan all active/cancelled subscriptions and expire those past end_date.

        Also sends renewal reminders to businesses within
        RENEWAL_REMINDER_DAYS of expiry. Processed in batches.

        Called by expiry_checker.py on a daily schedule.

        Args:
            db: AsyncSession.

        Returns:
            ExpiryCheckResult. Never raises.
        """
        log_extra = {"service": ServiceName.SUBSCRIPTIONS}
        result = ExpiryCheckResult()
        now = now_utc()
        today = today_local()
        reminder_threshold = today

        logger.info("Subscription expiry check started", extra=log_extra)

        try:
            candidates = await self._sub_repo.get_expiry_candidates(
                db=db,
                statuses=[SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED],
            )

            result.checked = len(candidates)

            for sub in candidates:
                try:
                    await self._process_expiry_candidate(
                        db=db,
                        sub=sub,
                        now=now,
                        reminder_threshold=reminder_threshold,
                        result=result,
                    )
                except Exception as exc:
                    result.errors += 1
                    logger.error(
                        "Expiry check failed for subscription",
                        extra={
                            **log_extra,
                            "subscription_id": str(sub.id),
                            "business_id": str(sub.business_id),
                            "error": str(exc),
                        },
                    )

        except Exception as exc:
            logger.error(
                "check_and_expire_subscriptions failed",
                extra={**log_extra, "error": str(exc)},
            )

        logger.info(
            "Subscription expiry check complete",
            extra={
                **log_extra,
                "checked": result.checked,
                "expired": result.expired,
                "reminders": result.reminder_sent,
                "errors": result.errors,
            },
        )
        return result

    async def _process_expiry_candidate(
        self,
        db: AsyncSession,
        sub,
        now: datetime,
        reminder_threshold,
        result: ExpiryCheckResult,
    ) -> None:
        """Evaluate a single subscription for expiry or renewal reminder."""
        if not sub.end_date:
            return

        end_date_aware = (
            sub.end_date.replace(tzinfo=timezone.utc)
            if sub.end_date.tzinfo is None
            else sub.end_date
        )

        if end_date_aware <= now:
            await self._sub_repo.update_status(
                db=db,
                subscription_id=str(sub.id),
                status=SubscriptionStatus.EXPIRED,
            )
            await db.commit()
            result.expired += 1

            logger.info(
                "Subscription expired",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "subscription_id": str(sub.id),
                    "business_id": str(sub.business_id),
                },
            )
            await self._notify_expiry(business_id=str(sub.business_id))

        elif sub.end_date.date() <= reminder_threshold + timedelta(
            days=RENEWAL_REMINDER_DAYS
        ):
            days_left = days_until(sub.end_date)
            if days_left >= 0:
                await self._notify_renewal_reminder(
                    business_id=str(sub.business_id),
                    days_remaining=days_left,
                )
                result.reminder_sent += 1

    # ------------------------------------------------------------------
    # 6. Subscription summary
    # ------------------------------------------------------------------

    async def get_subscription_summary(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> SubscriptionSummary:
        """
        Build a human-readable subscription summary for API responses.

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.

        Returns:
            SubscriptionSummary. Never raises.
        """
        try:
            sub = await self._sub_repo.get_active(db=db, business_id=business_id)

            if not sub:
                return SubscriptionSummary(
                    business_id=business_id,
                    has_active_plan=False,
                    status=SubscriptionStatus.EXPIRED,
                    billing_cycle=BillingCycle.MONTHLY,
                    start_date=None,
                    end_date=None,
                    days_remaining=0,
                    is_expiring_soon=False,
                )

            days_left = days_until(sub.end_date) if sub.end_date else 0

            return SubscriptionSummary(
                business_id=business_id,
                has_active_plan=True,
                status=sub.status,
                billing_cycle=sub.billing_cycle or BillingCycle.MONTHLY,
                start_date=(
                    sub.start_date.isoformat() if sub.start_date else None
                ),
                end_date=(
                    sub.end_date.isoformat() if sub.end_date else None
                ),
                days_remaining=max(0, days_left),
                is_expiring_soon=0 <= days_left <= RENEWAL_REMINDER_DAYS,
            )

        except Exception as exc:
            logger.error(
                "get_subscription_summary failed",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )
            return SubscriptionSummary(
                business_id=business_id,
                has_active_plan=False,
                status=SubscriptionStatus.EXPIRED,
                billing_cycle=BillingCycle.MONTHLY,
                start_date=None,
                end_date=None,
                days_remaining=0,
                is_expiring_soon=False,
            )

    # ------------------------------------------------------------------
    # WhatsApp notification helpers — all best-effort, never raise
    # ------------------------------------------------------------------

    async def _notify_renewal_confirmed(
        self,
        business_id: str,
        new_end_date: str,
        billing_cycle: str,
    ) -> None:
        """Notify business owner that renewal was successful."""
        try:
            biz = await self._biz_repo.get_by_id_no_db(business_id)
            if not biz or not biz.whatsapp_number:
                return
            cycle_label = "monthly" if billing_cycle == BillingCycle.MONTHLY else "annual"
            msg = (
                f"✅ *Subscription Renewed*\n\n"
                f"Your {cycle_label} subscription has been renewed.\n"
                f"Access continues until *{new_end_date}*.\n\n"
                f"Thank you for staying with us! 🙏"
            )
            await self._whatsapp.send_text_message(
                to=biz.whatsapp_number, text=msg
            )
        except Exception as exc:
            logger.warning(
                "Renewal confirmation WhatsApp notification failed",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )

    async def _notify_cancellation(
        self,
        business_id: str,
        access_until: str,
    ) -> None:
        """Notify business owner that their subscription has been cancelled."""
        try:
            biz = await self._biz_repo.get_by_id_no_db(business_id)
            if not biz or not biz.whatsapp_number:
                return
            msg = (
                f"📋 *Subscription Cancelled*\n\n"
                f"Your subscription has been cancelled.\n"
                f"You will continue to have full access until *{access_until}*.\n\n"
                f"We're sorry to see you go. "
                f"You can resubscribe anytime from the dashboard."
            )
            await self._whatsapp.send_text_message(
                to=biz.whatsapp_number, text=msg
            )
        except Exception as exc:
            logger.warning(
                "Cancellation WhatsApp notification failed",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )

    async def _notify_renewal_reminder(
        self,
        business_id: str,
        days_remaining: int,
    ) -> None:
        """Send a renewal reminder N days before expiry."""
        try:
            biz = await self._biz_repo.get_by_id_no_db(business_id)
            if not biz or not biz.whatsapp_number:
                return
            day_label = (
                "today"
                if days_remaining == 0
                else f"in {days_remaining} day{'s' if days_remaining != 1 else ''}"
            )
            msg = (
                f"⏰ *Subscription Expiring {day_label.title()}*\n\n"
                f"Your subscription expires {day_label}.\n\n"
                f"Renew now to keep receiving:\n"
                f"• AI-powered review replies\n"
                f"• Business insights and alerts\n"
                f"• Weekly and monthly reports\n\n"
                f"Visit the dashboard to renew."
            )
            await self._whatsapp.send_text_message(
                to=biz.whatsapp_number, text=msg
            )
        except Exception as exc:
            logger.warning(
                "Renewal reminder WhatsApp notification failed",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )

    async def _notify_expiry(self, business_id: str) -> None:
        """Send an expiry notice on the day the subscription expires."""
        try:
            biz = await self._biz_repo.get_by_id_no_db(business_id)
            if not biz or not biz.whatsapp_number:
                return
            msg = (
                f"🔴 *Subscription Expired*\n\n"
                f"Your subscription has expired.\n\n"
                f"AI review replies, alerts, and reports have been paused.\n\n"
                f"Resubscribe from the dashboard to reactivate all features. "
                f"Your data is safely preserved."
            )
            await self._whatsapp.send_text_message(
                to=biz.whatsapp_number, text=msg
            )
        except Exception as exc:
            logger.warning(
                "Expiry WhatsApp notification failed",
                extra={
                    "service": ServiceName.SUBSCRIPTIONS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )