# ==============================================================================
# File: app/subscriptions/plan_manager.py
# Purpose: Usage limit enforcement for subscribed businesses.
#
#          There is ONE subscription tier. Every subscribed business has
#          access to ALL features. There is no plan gating, no feature
#          access matrix, and no tier comparison.
#
#          The only role of this module is to enforce daily usage caps
#          that prevent system abuse and control infrastructure costs:
#
#            MAX_REVIEWS_PER_DAY           — reviews processed per business
#            MAX_AI_REPLIES_PER_DAY        — AI replies generated per business
#            MAX_COMPETITOR_SCANS_PER_DAY  — competitor scans per business
#
#          These limits are defined in app/config/constants.py and loaded
#          from environment variables so they can be adjusted without
#          code changes.
#
#          Two methods:
#            check_usage_limit(db, business_id, metric)
#              → Returns UsageLimitResult(allowed, current, limit, remaining)
#              → Used before executing a billable operation
#
#            require_within_limit(db, business_id, metric)
#              → Same check but raises UsageLimitExceededError if over limit
#              → Used in service methods that prefer exception-based flow
#
#          Subscription check:
#            Both methods first verify the business has an active
#            subscription. A business with no subscription is denied —
#            not because of a plan tier, but because they have not paid.
#
#          Multi-tenant:
#            All queries are scoped to business_id.
#            No cross-business access possible.
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import (
    MAX_AI_REPLIES_PER_DAY,
    MAX_COMPETITOR_SCANS_PER_DAY,
    MAX_REVIEWS_PER_DAY,
    ServiceName,
    UsageMetric,
)
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository

logger = logging.getLogger(ServiceName.SUBSCRIPTIONS)


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class UsageLimitResult:
    """
    Result of a daily usage limit check.

    Attributes:
        allowed:        True if the operation is within the daily limit.
        metric:         The usage metric that was checked.
        business_id:    Business UUID.
        current_usage:  Today's usage count for this metric.
        daily_limit:    Maximum allowed per day (0 = unlimited).
        remaining:      Remaining operations today.
        reason:         Why the check passed or failed.
        no_subscription: True if denied due to no active subscription.
    """
    allowed: bool
    metric: str
    business_id: str
    current_usage: int = 0
    daily_limit: int = 0
    remaining: int = 0
    reason: str = ""
    no_subscription: bool = False

    def __str__(self) -> str:
        status = "ALLOWED" if self.allowed else "DENIED"
        return (
            f"UsageLimitResult({status} "
            f"metric={self.metric} "
            f"usage={self.current_usage}/{self.daily_limit})"
        )


# ==============================================================================
# Daily limit map — one value per tracked metric
# ==============================================================================

_DAILY_LIMITS: dict[str, int] = {
    UsageMetric.REVIEWS_PROCESSED:     MAX_REVIEWS_PER_DAY,
    UsageMetric.AI_REPLIES_GENERATED:  MAX_AI_REPLIES_PER_DAY,
    UsageMetric.COMPETITOR_SCANS:      MAX_COMPETITOR_SCANS_PER_DAY,
}


# ==============================================================================
# Plan Manager
# ==============================================================================

class PlanManager:
    """
    Enforces daily usage limits for subscribed businesses.

    No feature gating. No plan tiers. Every subscribed business can use
    every feature. This class only enforces daily volume caps to protect
    infrastructure costs and prevent abuse.

    Usage:
        plan_manager = PlanManager(
            subscription_repo=subscription_repo,
            usage_repo=usage_repo,
        )

        result = await plan_manager.check_usage_limit(
            db=db,
            business_id="uuid",
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        if not result.allowed:
            logger.info("Daily AI reply limit reached for %s", business_id)
            return
    """

    def __init__(
        self,
        subscription_repo: SubscriptionRepository,
        usage_repo: UsageRepository,
    ) -> None:
        self._sub_repo = subscription_repo
        self._usage_repo = usage_repo

    # ------------------------------------------------------------------
    # Primary check — returns result, never raises
    # ------------------------------------------------------------------

    async def check_usage_limit(
        self,
        db: AsyncSession,
        business_id: str,
        metric: str,
    ) -> UsageLimitResult:
        """
        Check whether a business is within their daily usage limit
        for a given metric.

        Steps:
          1. Verify active subscription (deny if none)
          2. Look up the daily limit for the metric
          3. Fetch today's current usage count
          4. Return allowed=True if current < limit

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.
            metric:       UsageMetric constant.

        Returns:
            UsageLimitResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.SUBSCRIPTIONS,
            "business_id": business_id,
            "metric": metric,
        }

        try:
            # Step 1: Active subscription check
            sub = await self._sub_repo.get_active(
                db=db, business_id=business_id
            )
            if not sub:
                logger.debug(
                    "Usage check denied — no active subscription",
                    extra=log_extra,
                )
                return UsageLimitResult(
                    allowed=False,
                    metric=metric,
                    business_id=business_id,
                    reason="no_active_subscription",
                    no_subscription=True,
                )

            # Step 2: Look up daily limit for this metric
            daily_limit = _DAILY_LIMITS.get(metric, 0)
            if daily_limit == 0:
                # Metric has no configured limit — allow unconditionally
                return UsageLimitResult(
                    allowed=True,
                    metric=metric,
                    business_id=business_id,
                    daily_limit=0,
                    reason="no_limit_configured",
                )

            # Step 3: Fetch today's usage
            current_usage = await self._usage_repo.get_today_count(
                db=db,
                business_id=business_id,
                metric=metric,
            )

            # Step 4: Compare
            if current_usage >= daily_limit:
                logger.debug(
                    "Daily usage limit reached",
                    extra={
                        **log_extra,
                        "current": current_usage,
                        "limit": daily_limit,
                    },
                )
                return UsageLimitResult(
                    allowed=False,
                    metric=metric,
                    business_id=business_id,
                    current_usage=current_usage,
                    daily_limit=daily_limit,
                    remaining=0,
                    reason=f"daily_limit_reached: {current_usage}/{daily_limit}",
                )

            return UsageLimitResult(
                allowed=True,
                metric=metric,
                business_id=business_id,
                current_usage=current_usage,
                daily_limit=daily_limit,
                remaining=daily_limit - current_usage,
                reason="within_limit",
            )

        except Exception as exc:
            logger.error(
                "check_usage_limit unexpected error — denying by default",
                extra={**log_extra, "error": str(exc)},
            )
            return UsageLimitResult(
                allowed=False,
                metric=metric,
                business_id=business_id,
                reason=f"check_error: {exc}",
            )

    # ------------------------------------------------------------------
    # Exception-raising variant
    # ------------------------------------------------------------------

    async def require_within_limit(
        self,
        db: AsyncSession,
        business_id: str,
        metric: str,
    ) -> UsageLimitResult:
        """
        Assert usage is within the daily limit; raise if exceeded.

        Convenience wrapper for service methods that prefer exception-based
        control flow over checking result.allowed manually.

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.
            metric:       UsageMetric constant.

        Returns:
            UsageLimitResult (always allowed — raises if not).

        Raises:
            UsageLimitExceededError if limit is reached or no subscription.
        """
        result = await self.check_usage_limit(
            db=db,
            business_id=business_id,
            metric=metric,
        )
        if not result.allowed:
            raise UsageLimitExceededError(result)
        return result


# ==============================================================================
# Exception
# ==============================================================================

class UsageLimitExceededError(Exception):
    """
    Raised by PlanManager.require_within_limit() when a daily cap is hit
    or when the business has no active subscription.

    Carries the full UsageLimitResult for structured error handling.
    """

    def __init__(self, result: UsageLimitResult) -> None:
        self.result = result
        super().__init__(result.reason)

    @property
    def is_no_subscription(self) -> bool:
        return self.result.no_subscription

    @property
    def is_limit_reached(self) -> bool:
        return not self.result.no_subscription and not self.result.allowed

    def __str__(self) -> str:
        return (
            f"UsageLimitExceededError("
            f"metric={self.result.metric} "
            f"usage={self.result.current_usage}/{self.result.daily_limit} "
            f"reason={self.result.reason})"
        )