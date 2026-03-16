# ==============================================================================
# File: tests/test_subscriptions.py
# Purpose: Tests for the subscription management system.
#
#          Covers:
#            1. SubscriptionService — create, renew, cancel, expiry check,
#               summary, and idempotency behaviour
#            2. PlanManager — usage limit enforcement, no-subscription denial,
#               limit-reached denial, within-limit approval
#            3. Business invariants — no plan tiers anywhere, billing_cycle
#               is the only variable, full access on any active subscription
#
#          Design:
#            All repository and notification calls are mocked.
#            No real database or WhatsApp API required.
#            Tests are fully isolated — no shared mutable state.
#
#          Running:
#            pytest tests/test_subscriptions.py -v
#            pytest tests/test_subscriptions.py -v -k "expiry"
# ==============================================================================

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Env vars before any app import
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL",             "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("JWT_SECRET_KEY",           "test-secret-key-at-least-32-characters-long!!")
os.environ.setdefault("RAZORPAY_KEY_ID",          "rzp_test_key_id")
os.environ.setdefault("RAZORPAY_KEY_SECRET",      "rzp_test_secret")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET",  "rzp_test_webhook_secret")
os.environ.setdefault("WHATSAPP_API_TOKEN",       "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "test_phone_id")
os.environ.setdefault("OPENAI_API_KEY",           "sk-test-key")
os.environ.setdefault("ADMIN_WHATSAPP_NUMBER",    "9999999999")

from app.config.constants import (                      # noqa: E402
    BillingCycle,
    MAX_AI_REPLIES_PER_DAY,
    MAX_COMPETITOR_SCANS_PER_DAY,
    MAX_REVIEWS_PER_DAY,
    SubscriptionStatus,
    UsageMetric,
)
from app.subscriptions.plan_manager import (            # noqa: E402
    PlanManager,
    UsageLimitExceededError,
    UsageLimitResult,
)
from app.subscriptions.subscription_service import (    # noqa: E402
    ExpiryCheckResult,
    SubscriptionOperationResult,
    SubscriptionService,
    SubscriptionSummary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BUSINESS_ID      = str(uuid.uuid4())
_PAYMENT_REC_ID   = str(uuid.uuid4())
_SUB_ID           = str(uuid.uuid4())

_NOW_UTC = datetime.now(tz=timezone.utc)


# ==============================================================================
# Helpers
# ==============================================================================

def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.commit   = AsyncMock()
    db.rollback = AsyncMock()
    db.flush    = AsyncMock()
    return db


def _mock_subscription(
    status: str  = SubscriptionStatus.ACTIVE,
    billing_cycle: str = BillingCycle.MONTHLY,
    days_until_expiry: int = 20,
) -> MagicMock:
    sub = MagicMock()
    sub.id            = uuid.UUID(_SUB_ID)
    sub.business_id   = uuid.UUID(_BUSINESS_ID)
    sub.status        = status
    sub.billing_cycle = billing_cycle
    sub.start_date    = _NOW_UTC - timedelta(days=10)
    sub.end_date      = _NOW_UTC + timedelta(days=days_until_expiry)
    return sub


def _make_subscription_service(
    active_sub: MagicMock | None = None,
) -> SubscriptionService:
    """
    Build a SubscriptionService with all external dependencies mocked.
    Pass active_sub=None to simulate no active subscription.
    """
    sub_repo = MagicMock()
    biz_repo = MagicMock()
    whatsapp = MagicMock()

    sub_repo.get_active              = AsyncMock(return_value=active_sub)
    sub_repo.create                  = AsyncMock(return_value=_mock_subscription())
    sub_repo.extend_subscription     = AsyncMock(return_value=_mock_subscription())
    sub_repo.update_status           = AsyncMock(return_value=_mock_subscription(status=SubscriptionStatus.CANCELLED))
    sub_repo.get_expiry_candidates   = AsyncMock(return_value=[])

    biz_repo.get_by_id = AsyncMock(return_value=MagicMock(business_name="Test Biz"))

    whatsapp.send_payment_confirmation = AsyncMock()
    whatsapp.send_renewal_reminder     = AsyncMock()
    whatsapp.send_expiry_notice        = AsyncMock()

    svc = SubscriptionService(
        subscription_repo=sub_repo,
        business_repo=biz_repo,
        whatsapp_service=whatsapp,
    )
    return svc


def _make_plan_manager(
    active_sub: MagicMock | None = None,
    today_usage: int = 0,
) -> PlanManager:
    """Build a PlanManager with mocked repositories."""
    sub_repo   = MagicMock()
    usage_repo = MagicMock()

    sub_repo.get_active        = AsyncMock(return_value=active_sub)
    usage_repo.get_today_count = AsyncMock(return_value=today_usage)

    return PlanManager(
        subscription_repo=sub_repo,
        usage_repo=usage_repo,
    )


# ==============================================================================
# 1. SubscriptionService — create_subscription
# ==============================================================================

class TestCreateSubscription:
    """Tests for SubscriptionService.create_subscription()."""

    @pytest.mark.asyncio
    async def test_creates_new_subscription_when_none_exists(self) -> None:
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        result = await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert result.success is True
        assert result.subscription_id is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_extends_existing_active_subscription(self) -> None:
        """
        If a subscription already exists, create_subscription must extend it
        rather than creating a duplicate. Idempotency guard.
        """
        existing_sub = _mock_subscription()
        svc          = _make_subscription_service(active_sub=existing_sub)
        db           = _mock_db()

        result = await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert result.success is True
        # extend_subscription must be called, create must NOT be called
        svc._sub_repo.extend_subscription.assert_called_once()
        svc._sub_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_monthly_subscription_end_date_is_30_days_ahead(self) -> None:
        """Monthly subscription must end approximately 30 days after start."""
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        captured_kwargs: dict = {}

        async def _capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_subscription()

        svc._sub_repo.create = _capture_create

        await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        end_date   = captured_kwargs.get("end_date")
        start_date = captured_kwargs.get("start_date")
        assert end_date is not None
        assert start_date is not None
        delta = (end_date - start_date).days
        # Monthly = 30 or 31 days depending on calendar
        assert 28 <= delta <= 32, f"Monthly subscription delta {delta} days is unexpected"

    @pytest.mark.asyncio
    async def test_annual_subscription_end_date_is_365_days_ahead(self) -> None:
        """Annual subscription must end approximately 365 days after start."""
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        captured_kwargs: dict = {}

        async def _capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_subscription()

        svc._sub_repo.create = _capture_create

        await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.ANNUAL,
            payment_record_id=_PAYMENT_REC_ID,
        )

        end_date   = captured_kwargs.get("end_date")
        start_date = captured_kwargs.get("start_date")
        delta = (end_date - start_date).days
        # Annual = 365 or 366 days (leap year)
        assert 364 <= delta <= 367, f"Annual subscription delta {delta} days is unexpected"

    @pytest.mark.asyncio
    async def test_create_subscription_never_includes_plan_field(self) -> None:
        """
        Repository create() must never be called with a plan or plan_name
        argument. One tier only — billing_cycle is the only variable.
        """
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        call_kwargs = svc._sub_repo.create.call_args.kwargs
        assert "plan" not in call_kwargs, (
            "Subscription creation must not pass a plan field. "
            "One subscription tier — billing_cycle only."
        )
        assert "plan_name" not in call_kwargs

    @pytest.mark.asyncio
    async def test_status_is_active_on_creation(self) -> None:
        """New subscription must always be created with ACTIVE status."""
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        call_kwargs = svc._sub_repo.create.call_args.kwargs
        assert call_kwargs.get("status") == SubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_db_error_returns_failure_not_raise(self) -> None:
        """Database failure must return SubscriptionOperationResult(success=False), not raise."""
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()
        svc._sub_repo.create = AsyncMock(side_effect=RuntimeError("DB crash"))

        result = await svc.create_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            billing_cycle=BillingCycle.MONTHLY,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert isinstance(result, SubscriptionOperationResult)
        assert result.success is False
        assert result.error is not None


# ==============================================================================
# 2. SubscriptionService — renew_subscription
# ==============================================================================

class TestRenewSubscription:
    """Tests for SubscriptionService.renew_subscription()."""

    @pytest.mark.asyncio
    async def test_renew_extends_end_date(self) -> None:
        """Renewal must push the end_date forward, not backwards."""
        existing = _mock_subscription(days_until_expiry=5)
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        result = await svc.renew_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert result.success is True
        svc._sub_repo.extend_subscription.assert_called_once()

    @pytest.mark.asyncio
    async def test_renew_without_existing_sub_creates_new(self) -> None:
        """
        If there is no active subscription, renew must create a fresh one.
        This handles the case of a lapsed business re-subscribing.
        """
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        result = await svc.renew_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert result.success is True
        # Should have fallen through to create_subscription
        svc._sub_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_renew_can_switch_billing_cycle(self) -> None:
        """
        A business may switch from monthly to annual (or vice versa) at
        renewal. The new billing_cycle must be applied to the extension.
        """
        existing = _mock_subscription(billing_cycle=BillingCycle.MONTHLY)
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        await svc.renew_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            payment_record_id=_PAYMENT_REC_ID,
            billing_cycle=BillingCycle.ANNUAL,
        )

        call_kwargs = svc._sub_repo.extend_subscription.call_args.kwargs
        assert call_kwargs.get("billing_cycle") == BillingCycle.ANNUAL

    @pytest.mark.asyncio
    async def test_renew_db_error_returns_failure(self) -> None:
        """Database error during renewal must return failure, not raise."""
        existing = _mock_subscription()
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()
        svc._sub_repo.extend_subscription = AsyncMock(
            side_effect=RuntimeError("DB crash")
        )

        result = await svc.renew_subscription(
            db=db,
            business_id=_BUSINESS_ID,
            payment_record_id=_PAYMENT_REC_ID,
        )

        assert result.success is False
        assert result.error is not None


# ==============================================================================
# 3. SubscriptionService — cancel_subscription
# ==============================================================================

class TestCancelSubscription:
    """Tests for SubscriptionService.cancel_subscription()."""

    @pytest.mark.asyncio
    async def test_cancel_active_subscription_succeeds(self) -> None:
        existing = _mock_subscription(status=SubscriptionStatus.ACTIVE)
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        result = await svc.cancel_subscription(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert result.success is True
        svc._sub_repo.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_when_no_subscription_returns_failure(self) -> None:
        """Cancelling with no active subscription must return failure gracefully."""
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        result = await svc.cancel_subscription(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert result.success is False
        assert result.error is not None
        svc._sub_repo.update_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_sets_cancelled_status(self) -> None:
        """Cancellation must update status to CANCELLED, not EXPIRED or PENDING."""
        existing = _mock_subscription(status=SubscriptionStatus.ACTIVE)
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        await svc.cancel_subscription(
            db=db,
            business_id=_BUSINESS_ID,
        )

        call_kwargs = svc._sub_repo.update_status.call_args.kwargs
        assert call_kwargs.get("status") == SubscriptionStatus.CANCELLED


# ==============================================================================
# 4. SubscriptionService — check_and_expire_subscriptions
# ==============================================================================

class TestExpiryCheck:
    """Tests for SubscriptionService.check_and_expire_subscriptions()."""

    @pytest.mark.asyncio
    async def test_returns_expiry_check_result(self) -> None:
        """check_and_expire_subscriptions must always return ExpiryCheckResult."""
        svc = _make_subscription_service()
        db  = _mock_db()

        result = await svc.check_and_expire_subscriptions(db=db)

        assert isinstance(result, ExpiryCheckResult)

    @pytest.mark.asyncio
    async def test_expired_subscription_is_marked_expired(self) -> None:
        """
        A subscription whose end_date is in the past must be marked EXPIRED.
        """
        expired_sub = _mock_subscription(
            status=SubscriptionStatus.ACTIVE,
            days_until_expiry=-1,   # past
        )
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(return_value=[expired_sub])

        result = await svc.check_and_expire_subscriptions(db=db)

        assert result.checked == 1
        assert result.expired == 1
        svc._sub_repo.update_status.assert_called_once()
        call_kwargs = svc._sub_repo.update_status.call_args.kwargs
        assert call_kwargs.get("status") == SubscriptionStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_subscription_expiring_soon_triggers_reminder(self) -> None:
        """
        A subscription expiring within RENEWAL_REMINDER_DAYS must trigger
        a renewal reminder notification, not immediate expiry.
        """
        expiring_sub = _mock_subscription(
            status=SubscriptionStatus.ACTIVE,
            days_until_expiry=2,    # within 3-day reminder window
        )
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(return_value=[expiring_sub])
        svc._whatsapp.send_renewal_reminder  = AsyncMock()

        result = await svc.check_and_expire_subscriptions(db=db)

        assert result.reminder_sent == 1
        assert result.expired == 0

    @pytest.mark.asyncio
    async def test_healthy_subscription_not_affected(self) -> None:
        """A subscription with plenty of time left must not be expired or reminded."""
        healthy_sub = _mock_subscription(
            status=SubscriptionStatus.ACTIVE,
            days_until_expiry=20,
        )
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(return_value=[healthy_sub])

        result = await svc.check_and_expire_subscriptions(db=db)

        assert result.expired == 0
        assert result.reminder_sent == 0
        svc._sub_repo.update_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_in_one_sub_does_not_stop_others(self) -> None:
        """
        If processing one subscription raises an error, the loop must continue
        processing remaining subscriptions. Failure isolation is required.
        """
        sub_ok   = _mock_subscription(days_until_expiry=-1)    # will expire
        sub_ok.id = uuid.uuid4()

        sub_bad  = _mock_subscription(days_until_expiry=-1)
        sub_bad.id = uuid.uuid4()

        call_count = 0

        async def _exploding_update_status(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated DB crash on first sub")
            return MagicMock()

        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(return_value=[sub_bad, sub_ok])
        svc._sub_repo.update_status         = _exploding_update_status

        result = await svc.check_and_expire_subscriptions(db=db)

        assert result.checked  == 2
        assert result.errors   == 1
        assert result.expired  == 1   # second subscription still processed

    @pytest.mark.asyncio
    async def test_expiry_check_never_raises(self) -> None:
        """check_and_expire_subscriptions must never raise, even on repo failure."""
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(
            side_effect=RuntimeError("DB completely down")
        )

        result = await svc.check_and_expire_subscriptions(db=db)

        assert isinstance(result, ExpiryCheckResult)

    @pytest.mark.asyncio
    async def test_zero_candidates_returns_zero_counts(self) -> None:
        """When no subscriptions are candidates, all counts must be zero."""
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_expiry_candidates = AsyncMock(return_value=[])

        result = await svc.check_and_expire_subscriptions(db=db)

        assert result.checked      == 0
        assert result.expired      == 0
        assert result.reminder_sent == 0
        assert result.errors       == 0


# ==============================================================================
# 5. SubscriptionService — get_subscription_summary
# ==============================================================================

class TestSubscriptionSummary:
    """Tests for SubscriptionService.get_subscription_summary()."""

    @pytest.mark.asyncio
    async def test_active_sub_returns_has_active_plan_true(self) -> None:
        existing = _mock_subscription(days_until_expiry=15)
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert summary.has_active_plan is True
        assert summary.status == SubscriptionStatus.ACTIVE
        assert summary.days_remaining > 0

    @pytest.mark.asyncio
    async def test_no_sub_returns_has_active_plan_false(self) -> None:
        svc = _make_subscription_service(active_sub=None)
        db  = _mock_db()

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert summary.has_active_plan is False
        assert summary.days_remaining  == 0

    @pytest.mark.asyncio
    async def test_expiring_soon_flag_set_within_reminder_window(self) -> None:
        """is_expiring_soon must be True when within 3 days of expiry."""
        expiring = _mock_subscription(days_until_expiry=2)
        svc      = _make_subscription_service(active_sub=expiring)
        db       = _mock_db()

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert summary.is_expiring_soon is True

    @pytest.mark.asyncio
    async def test_expiring_soon_flag_false_when_plenty_of_time(self) -> None:
        healthy = _mock_subscription(days_until_expiry=20)
        svc     = _make_subscription_service(active_sub=healthy)
        db      = _mock_db()

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert summary.is_expiring_soon is False

    @pytest.mark.asyncio
    async def test_summary_never_raises(self) -> None:
        """get_subscription_summary must return a safe default on DB failure."""
        svc = _make_subscription_service()
        db  = _mock_db()
        svc._sub_repo.get_active = AsyncMock(
            side_effect=RuntimeError("DB crash")
        )

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        assert isinstance(summary, SubscriptionSummary)
        assert summary.has_active_plan is False

    @pytest.mark.asyncio
    async def test_summary_contains_no_plan_tier_field(self) -> None:
        """
        SubscriptionSummary must not contain a plan or plan_name field.
        One tier only — billing_cycle is the only variable.
        """
        existing = _mock_subscription()
        svc      = _make_subscription_service(active_sub=existing)
        db       = _mock_db()

        summary = await svc.get_subscription_summary(
            db=db,
            business_id=_BUSINESS_ID,
        )

        summary_dict = summary.__dict__
        assert "plan" not in summary_dict
        assert "plan_name" not in summary_dict


# ==============================================================================
# 6. PlanManager — check_usage_limit
# ==============================================================================

class TestPlanManagerUsageLimits:
    """Tests for PlanManager.check_usage_limit()."""

    @pytest.mark.asyncio
    async def test_within_limit_returns_allowed(self) -> None:
        """Usage below the daily limit must be allowed."""
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(active_sub=active_sub, today_usage=5)
        db         = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert result.allowed is True
        assert result.remaining == MAX_AI_REPLIES_PER_DAY - 5

    @pytest.mark.asyncio
    async def test_at_limit_returns_denied(self) -> None:
        """Usage equal to the daily limit must be denied."""
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(
            active_sub=active_sub,
            today_usage=MAX_AI_REPLIES_PER_DAY,
        )
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert result.allowed is False
        assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_over_limit_returns_denied(self) -> None:
        """Usage beyond the daily limit must be denied."""
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(
            active_sub=active_sub,
            today_usage=MAX_AI_REPLIES_PER_DAY + 50,
        )
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_no_active_subscription_returns_denied(self) -> None:
        """
        Businesses with no active subscription must be denied regardless
        of their usage count. Subscription check happens before usage check.
        """
        pm = _make_plan_manager(active_sub=None, today_usage=0)
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.REVIEWS_PROCESSED,
        )

        assert result.allowed is False
        assert result.no_subscription is True

    @pytest.mark.asyncio
    async def test_competitor_scans_use_correct_limit(self) -> None:
        """Each metric must use its own limit, not a shared one."""
        active_sub = _mock_subscription()
        # Just below competitor scan limit
        pm = _make_plan_manager(
            active_sub=active_sub,
            today_usage=MAX_COMPETITOR_SCANS_PER_DAY - 1,
        )
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.COMPETITOR_SCANS,
        )

        assert result.allowed is True
        assert result.daily_limit == MAX_COMPETITOR_SCANS_PER_DAY

    @pytest.mark.asyncio
    async def test_reviews_processed_use_correct_limit(self) -> None:
        active_sub = _mock_subscription()
        pm = _make_plan_manager(
            active_sub=active_sub,
            today_usage=MAX_REVIEWS_PER_DAY - 1,
        )
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.REVIEWS_PROCESSED,
        )

        assert result.allowed is True
        assert result.daily_limit == MAX_REVIEWS_PER_DAY

    @pytest.mark.asyncio
    async def test_result_includes_current_usage_and_limit(self) -> None:
        """UsageLimitResult must expose current_usage and daily_limit fields."""
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(active_sub=active_sub, today_usage=10)
        db         = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert result.current_usage == 10
        assert result.daily_limit   == MAX_AI_REPLIES_PER_DAY

    @pytest.mark.asyncio
    async def test_check_never_raises_on_db_error(self) -> None:
        """check_usage_limit must return denied result on error, never raise."""
        active_sub = _mock_subscription()
        pm = _make_plan_manager(active_sub=active_sub)
        pm._usage_repo.get_today_count = AsyncMock(
            side_effect=RuntimeError("DB crash")
        )
        db = _mock_db()

        result = await pm.check_usage_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert isinstance(result, UsageLimitResult)
        assert result.allowed is False


# ==============================================================================
# 7. PlanManager — require_within_limit
# ==============================================================================

class TestPlanManagerRequireWithinLimit:
    """Tests for PlanManager.require_within_limit() — the raising variant."""

    @pytest.mark.asyncio
    async def test_within_limit_returns_result_without_raising(self) -> None:
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(active_sub=active_sub, today_usage=0)
        db         = _mock_db()

        result = await pm.require_within_limit(
            db=db,
            business_id=_BUSINESS_ID,
            metric=UsageMetric.AI_REPLIES_GENERATED,
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_at_limit_raises_usage_limit_exceeded_error(self) -> None:
        """require_within_limit must raise UsageLimitExceededError when denied."""
        active_sub = _mock_subscription()
        pm         = _make_plan_manager(
            active_sub=active_sub,
            today_usage=MAX_AI_REPLIES_PER_DAY,
        )
        db = _mock_db()

        with pytest.raises(UsageLimitExceededError):
            await pm.require_within_limit(
                db=db,
                business_id=_BUSINESS_ID,
                metric=UsageMetric.AI_REPLIES_GENERATED,
            )

    @pytest.mark.asyncio
    async def test_no_subscription_raises_usage_limit_exceeded_error(self) -> None:
        """No active subscription must also raise UsageLimitExceededError."""
        pm = _make_plan_manager(active_sub=None)
        db = _mock_db()

        with pytest.raises(UsageLimitExceededError):
            await pm.require_within_limit(
                db=db,
                business_id=_BUSINESS_ID,
                metric=UsageMetric.REVIEWS_PROCESSED,
            )


# ==============================================================================
# 8. Business invariants — one tier, billing_cycle only
# ==============================================================================

class TestSingleTierInvariant:
    """
    Enforce the critical business rule: ONE subscription tier.
    Pay = full access to ALL features.
    billing_cycle (monthly/annual) is a pricing choice only.
    These tests document and guard this invariant.
    """

    def test_billing_cycle_enum_has_only_monthly_and_annual(self) -> None:
        """BillingCycle must only have monthly and annual — no other tiers."""
        valid_cycles = {BillingCycle.MONTHLY, BillingCycle.ANNUAL}
        all_cycles   = set(BillingCycle)
        assert all_cycles == valid_cycles, (
            f"BillingCycle has unexpected values: {all_cycles - valid_cycles}. "
            "Only 'monthly' and 'annual' are allowed. No plan tiers."
        )

    def test_subscription_status_has_no_plan_tier_values(self) -> None:
        """SubscriptionStatus must not contain plan-tier values like BASIC, PRO, PREMIUM."""
        forbidden = {"basic", "pro", "premium", "enterprise", "free", "standard"}
        statuses  = {s.value.lower() for s in SubscriptionStatus}
        found     = forbidden & statuses
        assert not found, (
            f"SubscriptionStatus contains plan tier values: {found}. "
            "Status must only reflect lifecycle state, not tier."
        )

    def test_usage_limits_are_flat_same_for_all_businesses(self) -> None:
        """
        All businesses must share the same usage limits.
        There must be no per-plan limit dict anywhere.
        """
        from app.config.constants import DAILY_USAGE_LIMITS, MAX_AI_REPLIES_PER_DAY

        # DAILY_USAGE_LIMITS is a flat dict, not nested per plan
        assert isinstance(DAILY_USAGE_LIMITS, dict)
        assert not any(
            isinstance(v, dict) for v in DAILY_USAGE_LIMITS.values()
        ), "DAILY_USAGE_LIMITS must be a flat dict, not nested per plan"

        # MAX_AI_REPLIES_PER_DAY is a single integer, not per-plan
        assert isinstance(MAX_AI_REPLIES_PER_DAY, int)

    @pytest.mark.asyncio
    async def test_plan_manager_does_not_differentiate_by_cycle(self) -> None:
        """
        PlanManager must apply the same daily limits regardless of whether
        the business is on a monthly or annual billing cycle. Billing cycle
        affects price only, not feature access or usage limits.
        """
        db = _mock_db()

        for cycle in (BillingCycle.MONTHLY, BillingCycle.ANNUAL):
            sub = _mock_subscription(billing_cycle=cycle)
            pm  = _make_plan_manager(active_sub=sub, today_usage=5)

            result = await pm.check_usage_limit(
                db=db,
                business_id=_BUSINESS_ID,
                metric=UsageMetric.AI_REPLIES_GENERATED,
            )

            assert result.daily_limit == MAX_AI_REPLIES_PER_DAY, (
                f"Billing cycle '{cycle}' produced different daily limit "
                f"{result.daily_limit} — limits must be identical for all cycles."
            )