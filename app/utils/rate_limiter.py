# ==============================================================================
# File: app/utils/rate_limiter.py
# Purpose: Enforces per-business daily usage caps.
#
#          ONE SUBSCRIPTION TIER ONLY — No plan-based limits.
#          Every paying business has the same daily caps defined in
#          DAILY_USAGE_LIMITS in constants.py.
#          These caps are abuse-prevention guards only — not feature gates.
#
#          Per-business admin overrides are supported via the
#          override_* columns on SubscriptionModel. If an override is
#          set for a business, it takes priority over the platform default.
#
#          Enforcement flow:
#            1. Load today's usage record for the business
#            2. Load the active subscription (existence check only)
#            3. Resolve effective limit:
#                 subscription.override_max_* (if set by admin)
#                   → DAILY_USAGE_LIMITS[metric] (platform default)
#            4. If within limit  → allow, return RateLimitResult(allowed=True)
#            5. If limit reached → block, increment rate_limit_hits counter,
#                                  return RateLimitResult(allowed=False)
#
#          This module never raises exceptions for limit violations —
#          it returns a structured result. The caller decides whether
#          to raise an HTTP error or silently skip the operation.
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import (
    DAILY_USAGE_LIMITS,
    ServiceName,
    UsageMetric,
)
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository

logger = logging.getLogger(ServiceName.API)

# Repository singletons — stateless, safe to reuse across calls
_subscription_repo = SubscriptionRepository()
_usage_repo = UsageRepository()


# ==============================================================================
# Result Type
# ==============================================================================

@dataclass
class RateLimitResult:
    """
    Structured result returned by every rate limit check.

    Attributes:
        allowed:       True if the operation may proceed.
        metric:        The usage metric that was checked.
        current_count: Current counter value for today.
        limit:         The effective daily limit for this metric.
        remaining:     Operations remaining before the limit. Zero when blocked.
        business_id:   UUID string of the checked business.
        reason:        Human-readable explanation when not allowed.
    """
    allowed: bool
    metric: str
    current_count: int
    limit: int
    remaining: int
    business_id: str
    reason: str = ""

    @property
    def is_blocked(self) -> bool:
        """True if the operation was blocked (inverse of allowed)."""
        return not self.allowed

    def __str__(self) -> str:
        if self.allowed:
            return (
                f"ALLOWED [{self.metric}] "
                f"business={self.business_id} "
                f"used={self.current_count}/{self.limit} "
                f"remaining={self.remaining}"
            )
        return (
            f"BLOCKED [{self.metric}] "
            f"business={self.business_id} "
            f"used={self.current_count}/{self.limit} "
            f"reason='{self.reason}'"
        )


# ==============================================================================
# Limit Resolution
# ==============================================================================

def _resolve_limit(metric: str, override: Optional[int]) -> int:
    """
    Resolve the effective daily limit for a metric.

    Priority order:
      1. Per-business admin override (if set on subscription record)
      2. Platform default from DAILY_USAGE_LIMITS

    Args:
        metric:   Usage metric name (see UsageMetric enum).
        override: Per-business override value from the subscription record.
                  None means use the platform default.

    Returns:
        int: Effective daily limit for this metric.
    """
    if override is not None:
        return override

    # Map UsageMetric values to DAILY_USAGE_LIMITS keys
    metric_key_map = {
        UsageMetric.REVIEWS_PROCESSED:    "max_reviews_per_day",
        UsageMetric.AI_REPLIES_GENERATED: "max_ai_replies_per_day",
        UsageMetric.COMPETITOR_SCANS:     "max_competitor_scans_per_day",
        UsageMetric.REPORTS_GENERATED:    "max_reports_per_day",
    }
    limit_key = metric_key_map.get(metric)
    if limit_key:
        return DAILY_USAGE_LIMITS[limit_key]

    # Unknown metric — block by returning 0
    return 0


# ==============================================================================
# Core Rate Limit Check
# ==============================================================================

async def check_rate_limit(
    db: AsyncSession,
    business_id,
    metric: str,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check whether a business is within its daily limit for a given metric.

    Loads the active subscription to verify it is usable and to read
    any per-business override set by admin. Compares today's usage
    counter against the resolved limit.

    This function is read-only — it never increments counters.
    Use usage_tracker.py to record usage after the operation completes.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business to check.
        metric:       Usage metric to check (see UsageMetric enum).
        usage_date:   Date to check against (defaults to today UTC).

    Returns:
        RateLimitResult: Structured result with allowed flag and counts.
        Never raises on limit violations — always returns a result.

    Raises:
        SQLAlchemyError: On database errors (not limit violations).
    """
    today = usage_date or date.today()
    business_id_str = str(business_id)

    # ── Load active subscription ───────────────────────────────────────────
    subscription = await _subscription_repo.get_active_by_business_id(
        db, business_id
    )

    if not subscription or not subscription.is_usable:
        logger.warning(
            "Rate limit check blocked — no active subscription",
            extra={
                "service": ServiceName.API,
                "business_id": business_id_str,
                "metric": metric,
            },
        )
        return RateLimitResult(
            allowed=False,
            metric=metric,
            current_count=0,
            limit=0,
            remaining=0,
            business_id=business_id_str,
            reason="No active subscription. Please complete payment to continue.",
        )

    # ── Resolve per-metric admin override ─────────────────────────────────
    override_map = {
        UsageMetric.REVIEWS_PROCESSED:    subscription.override_max_reviews_per_day,
        UsageMetric.AI_REPLIES_GENERATED: subscription.override_max_ai_replies_per_day,
        UsageMetric.COMPETITOR_SCANS:     subscription.override_max_competitor_scans_per_day,
        UsageMetric.REPORTS_GENERATED:    subscription.override_max_reports_per_day,
    }
    override = override_map.get(metric)

    # ── Resolve effective limit ────────────────────────────────────────────
    limit = _resolve_limit(metric, override)

    # ── Read current counter ───────────────────────────────────────────────
    current_count = await _usage_repo.get_current_count(
        db, business_id, metric, today
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    remaining = max(0, limit - current_count)
    allowed = current_count < limit

    result = RateLimitResult(
        allowed=allowed,
        metric=metric,
        current_count=current_count,
        limit=limit,
        remaining=remaining,
        business_id=business_id_str,
        reason="" if allowed else (
            f"Daily limit of {limit} {metric.replace('_', ' ')} reached. "
            f"Resets tomorrow."
        ),
    )

    log_level = logger.debug if allowed else logger.warning
    log_level(
        str(result),
        extra={
            "service": ServiceName.API,
            "business_id": business_id_str,
            "metric": metric,
            "current_count": current_count,
            "limit": limit,
            "allowed": allowed,
        },
    )

    return result


# ==============================================================================
# Enforce — Check + Record Hit
# ==============================================================================

async def enforce_rate_limit(
    db: AsyncSession,
    business_id,
    metric: str,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check the rate limit and record a hit counter if the limit is exceeded.

    Extends check_rate_limit() by incrementing the rate_limit_hits counter
    when the operation is blocked. This counter is used to detect businesses
    that are consistently hitting limits and may need admin attention.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        metric:       Usage metric to enforce.
        usage_date:   Date override (defaults to today UTC).

    Returns:
        RateLimitResult: Result with allowed flag and context.

    Raises:
        SQLAlchemyError: On database errors.
    """
    result = await check_rate_limit(db, business_id, metric, usage_date)

    if result.is_blocked:
        try:
            await _usage_repo.increment_rate_limit_hits(
                db, business_id, usage_date
            )
        except Exception as exc:
            # Non-critical — counter failure must not block the response
            logger.error(
                "Failed to increment rate_limit_hits counter",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "metric": metric,
                    "error": str(exc),
                },
            )

    return result


# ==============================================================================
# Convenience Checkers — Named per operation type
# ==============================================================================

async def check_review_limit(
    db: AsyncSession,
    business_id,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check whether the business is within its daily review processing limit.
    Called by review_jobs.py before entering a review into the pipeline.
    """
    return await enforce_rate_limit(
        db, business_id, UsageMetric.REVIEWS_PROCESSED, usage_date
    )


async def check_ai_reply_limit(
    db: AsyncSession,
    business_id,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check whether the business is within its daily AI reply generation limit.
    Called by ai_reply_service.py before invoking the OpenAI API.
    """
    return await enforce_rate_limit(
        db, business_id, UsageMetric.AI_REPLIES_GENERATED, usage_date
    )


async def check_competitor_scan_limit(
    db: AsyncSession,
    business_id,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check whether the business is within its daily competitor scan limit.
    Called by competitor_service.py before each competitor profile scan.
    """
    return await enforce_rate_limit(
        db, business_id, UsageMetric.COMPETITOR_SCANS, usage_date
    )


async def check_report_limit(
    db: AsyncSession,
    business_id,
    usage_date: Optional[date] = None,
) -> RateLimitResult:
    """
    Check whether the business is within its daily report generation limit.
    Called by reports_service.py before generating any report type.
    """
    return await enforce_rate_limit(
        db, business_id, UsageMetric.REPORTS_GENERATED, usage_date
    )


# ==============================================================================
# Batch Check — Multiple metrics in one call
# ==============================================================================

async def check_multiple_limits(
    db: AsyncSession,
    business_id,
    metrics: list[str],
    usage_date: Optional[date] = None,
) -> dict[str, RateLimitResult]:
    """
    Check multiple usage metrics for a business in a single call.

    Returns a mapping of metric name to RateLimitResult.

    Args:
        db:           Active async database session.
        business_id:  UUID of the business.
        metrics:      List of metric names to check.
        usage_date:   Date override (defaults to today UTC).

    Returns:
        dict[str, RateLimitResult]: Results keyed by metric name.
    """
    results: dict[str, RateLimitResult] = {}
    for metric in metrics:
        results[metric] = await check_rate_limit(
            db, business_id, metric, usage_date
        )
    return results


def all_limits_passed(results: dict[str, RateLimitResult]) -> bool:
    """Return True only if every result in a batch check passed."""
    return all(r.allowed for r in results.values())


def first_blocked(
    results: dict[str, RateLimitResult],
) -> Optional[RateLimitResult]:
    """Return the first blocked result from a batch check, or None if all passed."""
    for result in results.values():
        if result.is_blocked:
            return result
    return None