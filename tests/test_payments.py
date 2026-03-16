# ==============================================================================
# File: tests/test_payments.py
# Purpose: Tests for the full payment workflow.
#
#          Covers:
#            1. RazorpayClient — signature verification (payment + webhook)
#            2. PaymentService — initiate_payment, process_webhook_event,
#               get_payment_status, and internal routing helpers
#            3. RazorpayWebhookHandler — end-to-end handle() flow
#            4. Private helpers — _resolve_amount, _build_receipt
#            5. Idempotency — duplicate event handling
#            6. Security — server-side verification, never trust frontend
#
#          Design:
#            All external API calls (Razorpay HTTP) are mocked.
#            All database operations (repositories) are mocked.
#            No real network or database required.
#
#          Running:
#            pytest tests/test_payments.py -v
#            pytest tests/test_payments.py -v -k "signature"
# ==============================================================================

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Set env vars before any app imports
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL",             "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("JWT_SECRET_KEY",           "test-secret-key-at-least-32-characters-long!!")
os.environ.setdefault("RAZORPAY_KEY_ID",          "rzp_test_key_id")
os.environ.setdefault("RAZORPAY_KEY_SECRET",      "rzp_test_webhook_secret")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET",  "rzp_test_webhook_secret")
os.environ.setdefault("WHATSAPP_API_TOKEN",       "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "test_phone_id")
os.environ.setdefault("OPENAI_API_KEY",           "sk-test-key")
os.environ.setdefault("ADMIN_WHATSAPP_NUMBER",    "+919999999999")

from app.config.constants import PaymentStatus                          # noqa: E402
from app.payments.payment_service import (                              # noqa: E402
    PaymentInitRequest,
    PaymentInitResult,
    PaymentService,
    WebhookProcessResult,
    _build_receipt,
    _resolve_amount,
)
from app.payments.razorpay_client import (                              # noqa: E402
    RazorpayApiResult,
    RazorpayClient,
    RazorpayOrder,
    RazorpayPayment,
)
from app.payments.webhook_handler import (                              # noqa: E402
    RazorpayWebhookHandler,
    WebhookHandlerResult,
)

# ---------------------------------------------------------------------------
# Constants re-used across tests
# ---------------------------------------------------------------------------
_BUSINESS_ID     = str(uuid.uuid4())
_ORDER_ID        = "order_TestABC123"
_PAYMENT_ID      = "pay_TestXYZ456"
_WEBHOOK_SECRET  = os.environ["RAZORPAY_WEBHOOK_SECRET"]
_MONTHLY_PAISE   = 99900
_ANNUAL_PAISE    = 999900


# ==============================================================================
# Helpers
# ==============================================================================

def _make_webhook_signature(body: bytes, secret: str = _WEBHOOK_SECRET) -> str:
    """Produce a valid Razorpay HMAC-SHA256 webhook signature."""
    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


def _make_payment_signature(order_id: str, payment_id: str, secret: str = "rzp_test_webhook_secret") -> str:
    """Produce a valid Razorpay payment (checkout) HMAC-SHA256 signature."""
    message = f"{order_id}|{payment_id}"
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _mock_db() -> AsyncMock:
    """Return a minimal AsyncSession mock."""
    db = AsyncMock()
    db.commit   = AsyncMock()
    db.rollback = AsyncMock()
    db.flush    = AsyncMock()
    return db


def _mock_razorpay_order(
    order_id: str = _ORDER_ID,
    amount: int = _MONTHLY_PAISE,
    status: str = "created",
) -> RazorpayOrder:
    return RazorpayOrder(
        order_id   = order_id,
        amount     = amount,
        currency   = "INR",
        receipt    = f"RCPT_{order_id[:8]}_MON",
        status     = status,
        created_at = int(datetime.now(tz=timezone.utc).timestamp()),
    )


def _mock_razorpay_payment(
    payment_id: str = _PAYMENT_ID,
    order_id: str   = _ORDER_ID,
    status: str     = "captured",
    captured: bool  = True,
    amount: int     = _MONTHLY_PAISE,
) -> RazorpayPayment:
    return RazorpayPayment(
        payment_id = payment_id,
        order_id   = order_id,
        amount     = amount,
        currency   = "INR",
        status     = status,
        method     = "upi",
        captured   = captured,
        email      = "test@example.com",
        contact    = "9876543210",
        created_at = int(datetime.now(tz=timezone.utc).timestamp()),
    )


def _mock_payment_record(
    status: str = PaymentStatus.PENDING,
    business_id: str = _BUSINESS_ID,
) -> MagicMock:
    rec = MagicMock()
    rec.id                   = uuid.uuid4()
    rec.business_id          = uuid.UUID(business_id)
    rec.razorpay_order_id    = _ORDER_ID
    rec.razorpay_payment_id  = None
    rec.status               = status
    rec.billing_cycle        = "monthly"
    rec.amount               = _MONTHLY_PAISE
    return rec


def _mock_business(business_id: str = _BUSINESS_ID) -> MagicMock:
    b = MagicMock()
    b.id            = uuid.UUID(business_id)
    b.business_name = "Test Business"
    b.is_active     = True
    b.is_deleted    = False
    return b


def _payment_captured_event(
    payment_id: str = _PAYMENT_ID,
    order_id: str   = _ORDER_ID,
    amount: int     = _MONTHLY_PAISE,
) -> dict:
    return {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id":       payment_id,
                    "order_id": order_id,
                    "amount":   amount,
                    "currency": "INR",
                    "status":   "captured",
                }
            }
        },
    }


def _payment_failed_event(
    payment_id: str = _PAYMENT_ID,
    order_id: str   = _ORDER_ID,
) -> dict:
    return {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id":           payment_id,
                    "order_id":     order_id,
                    "status":       "failed",
                    "error_code":   "BAD_REQUEST_ERROR",
                    "error_reason": "payment_failed",
                }
            }
        },
    }


def _make_payment_service() -> PaymentService:
    """Construct a PaymentService with all dependencies mocked."""
    svc = PaymentService.__new__(PaymentService)
    svc._razorpay       = MagicMock(spec=RazorpayClient)
    svc._business_repo  = MagicMock()
    svc._subscription_repo = MagicMock()
    svc._payment_repo   = MagicMock()
    svc._whatsapp       = MagicMock()
    svc._admin          = MagicMock()
    # Default async stubs — individual tests override as needed
    svc._whatsapp.send_payment_confirmation = AsyncMock()
    svc._admin.send_payment_received        = AsyncMock()
    svc._admin.send_payment_failed          = AsyncMock()
    svc._admin.send_alert                   = AsyncMock()
    return svc


# ==============================================================================
# 1. Private helpers
# ==============================================================================

class TestResolveAmount:
    """Unit tests for _resolve_amount() helper."""

    def test_monthly_returns_correct_paise(self) -> None:
        assert _resolve_amount("monthly") == _MONTHLY_PAISE

    def test_annual_returns_correct_paise(self) -> None:
        assert _resolve_amount("annual") == _ANNUAL_PAISE

    def test_monthly_is_cheaper_than_annual(self) -> None:
        """Annual is always higher total but we store raw paise — just confirm distinct."""
        assert _resolve_amount("monthly") != _resolve_amount("annual")

    def test_unknown_cycle_defaults_to_monthly(self) -> None:
        """Unknown billing cycle must fall back to monthly price, not crash."""
        assert _resolve_amount("quarterly") == _MONTHLY_PAISE

    def test_amounts_are_positive_integers(self) -> None:
        for cycle in ("monthly", "annual"):
            amount = _resolve_amount(cycle)
            assert isinstance(amount, int)
            assert amount > 0


class TestBuildReceipt:
    """Unit tests for _build_receipt() helper."""

    def test_receipt_contains_business_short_id(self) -> None:
        biz_id  = "550e8400-e29b-41d4-a716-446655440000"
        receipt = _build_receipt(biz_id, "monthly")
        short   = biz_id.replace("-", "")[:12]
        assert short in receipt

    def test_monthly_receipt_contains_mon(self) -> None:
        receipt = _build_receipt(_BUSINESS_ID, "monthly")
        assert "MON" in receipt

    def test_annual_receipt_contains_ann(self) -> None:
        receipt = _build_receipt(_BUSINESS_ID, "annual")
        assert "ANN" in receipt

    def test_receipt_max_40_chars(self) -> None:
        receipt = _build_receipt(_BUSINESS_ID, "monthly")
        assert len(receipt) <= 40

    def test_receipt_starts_with_rcpt(self) -> None:
        receipt = _build_receipt(_BUSINESS_ID, "annual")
        assert receipt.startswith("RCPT_")


# ==============================================================================
# 2. RazorpayClient — signature verification
# ==============================================================================

class TestRazorpayClientWebhookSignature:
    """Tests for RazorpayClient.verify_webhook_signature()."""

    def _client(self) -> RazorpayClient:
        """Build a RazorpayClient with secrets injected directly."""
        client = RazorpayClient.__new__(RazorpayClient)
        client._key_id     = "rzp_test_key_id"
        client._key_secret = _WEBHOOK_SECRET
        client._http       = None
        return client

    def test_valid_signature_returns_true(self) -> None:
        client  = self._client()
        body    = b'{"event":"payment.captured"}'
        sig     = _make_webhook_signature(body)
        assert client.verify_webhook_signature(body, sig) is True

    def test_wrong_secret_returns_false(self) -> None:
        client  = self._client()
        body    = b'{"event":"payment.captured"}'
        sig     = _make_webhook_signature(body, secret="wrong_secret")
        assert client.verify_webhook_signature(body, sig) is False

    def test_tampered_body_returns_false(self) -> None:
        client  = self._client()
        body    = b'{"event":"payment.captured"}'
        sig     = _make_webhook_signature(body)
        tampered = b'{"event":"payment.failed"}'
        assert client.verify_webhook_signature(tampered, sig) is False

    def test_empty_signature_returns_false(self) -> None:
        client = self._client()
        body   = b'{"event":"payment.captured"}'
        assert client.verify_webhook_signature(body, "") is False

    def test_empty_body_signature_different_from_nonempty(self) -> None:
        """An empty body has its own valid HMAC — not the same as any real event."""
        client      = self._client()
        real_body   = b'{"event":"payment.captured"}'
        real_sig    = _make_webhook_signature(real_body)
        # empty body's signature is different
        assert client.verify_webhook_signature(b"", real_sig) is False


class TestRazorpayClientPaymentSignature:
    """Tests for RazorpayClient.verify_payment_signature()."""

    def _client(self) -> RazorpayClient:
        client = RazorpayClient.__new__(RazorpayClient)
        client._key_id     = "rzp_test_key_id"
        client._key_secret = _WEBHOOK_SECRET
        client._http       = None
        return client

    def test_valid_payment_signature_returns_true(self) -> None:
        client = self._client()
        sig    = _make_payment_signature(_ORDER_ID, _PAYMENT_ID, _WEBHOOK_SECRET)
        assert client.verify_payment_signature(_ORDER_ID, _PAYMENT_ID, sig) is True

    def test_wrong_payment_id_returns_false(self) -> None:
        client = self._client()
        sig    = _make_payment_signature(_ORDER_ID, _PAYMENT_ID, _WEBHOOK_SECRET)
        assert client.verify_payment_signature(_ORDER_ID, "pay_wrong", sig) is False

    def test_wrong_order_id_returns_false(self) -> None:
        client = self._client()
        sig    = _make_payment_signature(_ORDER_ID, _PAYMENT_ID, _WEBHOOK_SECRET)
        assert client.verify_payment_signature("order_wrong", _PAYMENT_ID, sig) is False

    def test_swapped_order_and_payment_id_returns_false(self) -> None:
        """Signature is not symmetric — swapping IDs must fail verification."""
        client = self._client()
        sig    = _make_payment_signature(_ORDER_ID, _PAYMENT_ID, _WEBHOOK_SECRET)
        assert client.verify_payment_signature(_PAYMENT_ID, _ORDER_ID, sig) is False


# ==============================================================================
# 3. PaymentService — initiate_payment
# ==============================================================================

class TestPaymentServiceInitiate:
    """Tests for PaymentService.initiate_payment()."""

    @pytest.mark.asyncio
    async def test_success_returns_razorpay_order_id(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id = AsyncMock(return_value=_mock_business())
        svc._payment_repo.get_by_idempotency_key = AsyncMock(return_value=None)
        svc._razorpay.create_order = AsyncMock(
            return_value=RazorpayApiResult(
                success=True,
                data=_mock_razorpay_order(),
                status_code=200,
            )
        )
        svc._payment_repo.create = AsyncMock(return_value=_mock_payment_record())

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="monthly",
            ),
        )

        assert result.success is True
        assert result.razorpay_order_id == _ORDER_ID
        assert result.amount_paise == _MONTHLY_PAISE
        assert result.currency == "INR"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_annual_cycle_uses_annual_price(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id = AsyncMock(return_value=_mock_business())
        svc._payment_repo.get_by_idempotency_key = AsyncMock(return_value=None)

        captured_args = {}

        async def _capture_create_order(**kwargs):
            captured_args.update(kwargs)
            return RazorpayApiResult(
                success=True,
                data=_mock_razorpay_order(amount=_ANNUAL_PAISE),
                status_code=200,
            )

        svc._razorpay.create_order = _capture_create_order
        svc._payment_repo.create   = AsyncMock(return_value=_mock_payment_record())

        await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="annual",
            ),
        )

        assert captured_args.get("amount_paise") == _ANNUAL_PAISE

    @pytest.mark.asyncio
    async def test_unknown_business_returns_failure(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id = AsyncMock(return_value=None)

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="monthly",
            ),
        )

        assert result.success is False
        assert result.error is not None
        assert result.razorpay_order_id is None

    @pytest.mark.asyncio
    async def test_invalid_billing_cycle_returns_failure(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id = AsyncMock(return_value=_mock_business())

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="quarterly",  # not allowed
            ),
        )

        assert result.success is False
        assert result.razorpay_order_id is None

    @pytest.mark.asyncio
    async def test_razorpay_api_failure_returns_failure(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id = AsyncMock(return_value=_mock_business())
        svc._payment_repo.get_by_idempotency_key = AsyncMock(return_value=None)
        svc._razorpay.create_order = AsyncMock(
            return_value=RazorpayApiResult(
                success=False,
                error="Razorpay unavailable",
                status_code=503,
            )
        )

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="monthly",
            ),
        )

        assert result.success is False
        assert "Razorpay" in (result.error or "")

    @pytest.mark.asyncio
    async def test_idempotent_returns_existing_pending_order(self) -> None:
        """
        If a pending payment record already exists for the same
        business + billing_cycle, the existing order must be returned
        without creating a new Razorpay order.
        """
        svc             = _make_payment_service()
        db              = _mock_db()
        existing_record = _mock_payment_record(status=PaymentStatus.PENDING)
        existing_record.razorpay_order_id = _ORDER_ID

        svc._business_repo.get_by_id         = AsyncMock(return_value=_mock_business())
        svc._payment_repo.get_by_idempotency_key = AsyncMock(return_value=existing_record)

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="monthly",
            ),
        )

        assert result.success is True
        # Razorpay create_order must NOT have been called
        svc._razorpay.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_never_creates_plan_tier(self) -> None:
        """
        initiate_payment must never reference a plan name or tier.
        The result must contain only billing_cycle — no plan field.
        One tier only — pay = full access.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        svc._business_repo.get_by_id         = AsyncMock(return_value=_mock_business())
        svc._payment_repo.get_by_idempotency_key = AsyncMock(return_value=None)
        svc._razorpay.create_order = AsyncMock(
            return_value=RazorpayApiResult(
                success=True,
                data=_mock_razorpay_order(),
                status_code=200,
            )
        )
        svc._payment_repo.create = AsyncMock(return_value=_mock_payment_record())

        result = await svc.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id=_BUSINESS_ID,
                billing_cycle="monthly",
            ),
        )

        # PaymentInitResult has no plan field — confirm by checking __dataclass_fields__
        result_fields = result.__dataclass_fields__.keys() if hasattr(result, "__dataclass_fields__") else vars(result).keys()
        assert "plan" not in result_fields
        assert "plan_name" not in result_fields


# ==============================================================================
# 4. PaymentService — process_webhook_event (payment.captured)
# ==============================================================================

class TestWebhookPaymentCaptured:
    """Tests for PaymentService.process_webhook_event() with payment.captured."""

    @pytest.mark.asyncio
    async def test_captured_event_returns_success(self) -> None:
        svc = _make_payment_service()
        db  = _mock_db()

        rz_payment = _mock_razorpay_payment()
        rz_result  = RazorpayApiResult(success=True, data=rz_payment, status_code=200)

        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=None)
        svc._razorpay.fetch_payment                  = AsyncMock(return_value=rz_result)

        payment_rec = _mock_payment_record(status=PaymentStatus.PENDING)
        svc._payment_repo.get_by_order_id_unscoped = AsyncMock(return_value=payment_rec)
        svc._payment_repo.update_status            = AsyncMock(return_value=payment_rec)

        mock_sub = MagicMock()
        mock_sub.id = uuid.uuid4()
        svc._subscription_repo.get_active      = AsyncMock(return_value=None)
        svc._subscription_repo.create          = AsyncMock(return_value=mock_sub)
        svc._business_repo.get_by_id           = AsyncMock(return_value=_mock_business())

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        assert result.success is True
        assert result.event_type == "payment.captured"
        assert result.payment_id == _PAYMENT_ID
        assert result.already_processed is False

    @pytest.mark.asyncio
    async def test_already_captured_is_idempotent(self) -> None:
        """
        A payment event received twice must be recognised and skipped,
        not re-processed. already_processed=True, success=True.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        # Simulate payment record already CAPTURED from first delivery
        existing = _mock_payment_record(status=PaymentStatus.CAPTURED)
        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=existing)

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        assert result.success is True
        assert result.already_processed is True
        # Razorpay fetch must NOT be called — we short-circuit at idempotency check
        svc._razorpay.fetch_payment.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_payment_id_in_payload_returns_failure(self) -> None:
        """Malformed payload missing payment id must return failure, not raise."""
        svc = _make_payment_service()
        db  = _mock_db()

        bad_payload = {
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {}  # no id, no order_id
                }
            },
        }

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=bad_payload,
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_razorpay_verification_failure_returns_failure(self) -> None:
        """
        Server-side fetch of payment fails — must return failure.
        This guards against spoofed webhook events that pass
        HMAC signature but reference a fake payment ID.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=None)
        svc._razorpay.fetch_payment = AsyncMock(
            return_value=RazorpayApiResult(
                success=False,
                error="Payment not found",
                status_code=404,
            )
        )

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_payment_not_captured_on_razorpay_returns_failure(self) -> None:
        """
        Payment status on Razorpay is not 'captured' — must reject,
        never activate subscription. Prevents accepting authorized-only payments.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        # Payment exists but is only "authorized", not "captured"
        rz_payment = _mock_razorpay_payment(status="authorized", captured=False)
        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=None)
        svc._razorpay.fetch_payment = AsyncMock(
            return_value=RazorpayApiResult(success=True, data=rz_payment, status_code=200)
        )

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        assert result.success is False
        # Subscription must NOT have been created
        svc._subscription_repo.create.assert_not_called()


# ==============================================================================
# 5. PaymentService — process_webhook_event (payment.failed)
# ==============================================================================

class TestWebhookPaymentFailed:
    """Tests for process_webhook_event() with payment.failed events."""

    @pytest.mark.asyncio
    async def test_failed_event_returns_success_true(self) -> None:
        """
        payment.failed must return success=True — the event was handled,
        even though the payment itself failed. success here means the
        system processed the webhook correctly, not that payment succeeded.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        payment_rec = _mock_payment_record(status=PaymentStatus.PENDING)
        svc._payment_repo.get_by_order_id_unscoped = AsyncMock(return_value=payment_rec)
        svc._payment_repo.update_status            = AsyncMock(return_value=payment_rec)
        svc._business_repo.get_by_id              = AsyncMock(return_value=_mock_business())

        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.failed",
            event_payload=_payment_failed_event(),
        )

        assert result.success is True
        assert result.event_type == "payment.failed"

    @pytest.mark.asyncio
    async def test_failed_event_does_not_activate_subscription(self) -> None:
        """Subscription must never be activated on payment.failed."""
        svc = _make_payment_service()
        db  = _mock_db()

        payment_rec = _mock_payment_record(status=PaymentStatus.PENDING)
        svc._payment_repo.get_by_order_id_unscoped = AsyncMock(return_value=payment_rec)
        svc._payment_repo.update_status            = AsyncMock(return_value=payment_rec)
        svc._business_repo.get_by_id              = AsyncMock(return_value=_mock_business())

        await svc.process_webhook_event(
            db=db,
            event_type="payment.failed",
            event_payload=_payment_failed_event(),
        )

        svc._subscription_repo.create.assert_not_called()
        svc._subscription_repo.extend_subscription.assert_not_called()


# ==============================================================================
# 6. PaymentService — process_webhook_event (unrecognised event)
# ==============================================================================

class TestWebhookUnrecognisedEvent:
    """Tests for unrecognised / informational Razorpay event types."""

    @pytest.mark.asyncio
    async def test_unrecognised_event_returns_success(self) -> None:
        """
        Events we don't handle must be acknowledged (success=True) without
        error so Razorpay does not retry indefinitely.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        result = await svc.process_webhook_event(
            db=db,
            event_type="subscription.charged",
            event_payload={"event": "subscription.charged", "payload": {}},
        )

        assert result.success is True
        assert result.event_type == "subscription.charged"

    @pytest.mark.asyncio
    async def test_unrecognised_event_does_not_touch_db(self) -> None:
        """Unhandled events must not modify any database records."""
        svc = _make_payment_service()
        db  = _mock_db()

        await svc.process_webhook_event(
            db=db,
            event_type="order.paid",
            event_payload={"event": "order.paid", "payload": {}},
        )

        svc._payment_repo.update_status.assert_not_called()
        svc._subscription_repo.create.assert_not_called()


# ==============================================================================
# 7. RazorpayWebhookHandler — handle()
# ==============================================================================

class TestRazorpayWebhookHandler:
    """End-to-end tests for RazorpayWebhookHandler.handle()."""

    def _make_handler(self) -> RazorpayWebhookHandler:
        razorpay_client  = MagicMock(spec=RazorpayClient)
        payment_service  = MagicMock(spec=PaymentService)
        return RazorpayWebhookHandler(
            razorpay_client=razorpay_client,
            payment_service=payment_service,
        )

    def _make_request(
        self,
        body: bytes,
        signature: str,
    ) -> MagicMock:
        """Build a mock FastAPI Request with controllable body and headers."""
        req = MagicMock()
        req.body = AsyncMock(return_value=body)
        req.headers = {
            "x-razorpay-signature": signature,
            "user-agent": "Razorpay/1.0",
        }
        req.client = MagicMock()
        req.client.host = "127.0.0.1"
        return req

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self) -> None:
        handler  = self._make_handler()
        payload  = json.dumps(_payment_captured_event()).encode()
        sig      = _make_webhook_signature(payload)
        request  = self._make_request(payload, sig)
        db       = _mock_db()

        # Signature verification passes
        handler._razorpay.verify_webhook_signature.return_value = True
        handler._payment_service.process_webhook_event = AsyncMock(
            return_value=WebhookProcessResult(
                success=True,
                event_type="payment.captured",
                payment_id=_PAYMENT_ID,
            )
        )

        result = await handler.handle(request=request, db=db)

        assert result.accepted is True
        assert result.http_status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected_with_400(self) -> None:
        handler = self._make_handler()
        payload = json.dumps(_payment_captured_event()).encode()
        request = self._make_request(payload, "invalid_signature")
        db      = _mock_db()

        handler._razorpay.verify_webhook_signature.return_value = False

        result = await handler.handle(request=request, db=db)

        assert result.accepted is False
        assert result.http_status_code == 400

    @pytest.mark.asyncio
    async def test_missing_signature_rejected_with_400(self) -> None:
        handler = self._make_handler()
        payload = json.dumps(_payment_captured_event()).encode()
        request = self._make_request(payload, "")  # empty sig
        db      = _mock_db()

        handler._razorpay.verify_webhook_signature.return_value = False

        result = await handler.handle(request=request, db=db)

        assert result.accepted is False
        assert result.http_status_code == 400

    @pytest.mark.asyncio
    async def test_empty_body_rejected_with_400(self) -> None:
        handler = self._make_handler()
        request = self._make_request(b"", "any_sig")
        db      = _mock_db()

        result = await handler.handle(request=request, db=db)

        assert result.accepted is False
        assert result.http_status_code == 400

    @pytest.mark.asyncio
    async def test_internal_processing_error_still_returns_200(self) -> None:
        """
        If payment_service raises an internal error AFTER valid signature,
        the handler must still return 200 to prevent Razorpay retry storm.
        The error is logged internally.
        """
        handler = self._make_handler()
        payload = json.dumps(_payment_captured_event()).encode()
        sig     = _make_webhook_signature(payload)
        request = self._make_request(payload, sig)
        db      = _mock_db()

        handler._razorpay.verify_webhook_signature.return_value = True
        handler._payment_service.process_webhook_event = AsyncMock(
            side_effect=RuntimeError("Unexpected DB crash")
        )

        result = await handler.handle(request=request, db=db)

        # Signature was valid so Razorpay gets 200
        assert result.http_status_code == 200

    @pytest.mark.asyncio
    async def test_process_webhook_called_with_correct_event_type(self) -> None:
        """Handler must pass the correct event_type to payment_service."""
        handler = self._make_handler()
        payload = json.dumps(_payment_captured_event()).encode()
        sig     = _make_webhook_signature(payload)
        request = self._make_request(payload, sig)
        db      = _mock_db()

        handler._razorpay.verify_webhook_signature.return_value = True
        handler._payment_service.process_webhook_event = AsyncMock(
            return_value=WebhookProcessResult(
                success=True,
                event_type="payment.captured",
            )
        )

        await handler.handle(request=request, db=db)

        handler._payment_service.process_webhook_event.assert_awaited_once()
        call_kwargs = handler._payment_service.process_webhook_event.call_args.kwargs
        assert call_kwargs.get("event_type") == "payment.captured"


# ==============================================================================
# 8. Security invariants
# ==============================================================================

class TestPaymentSecurityInvariants:
    """
    Critical security properties that must always hold.
    These tests document and enforce non-negotiable payment safety rules.
    """

    @pytest.mark.asyncio
    async def test_subscription_never_activated_without_server_verification(self) -> None:
        """
        Subscription must only be activated after server-side Razorpay
        fetch confirms captured status. Frontend confirmation alone is
        insufficient and must never bypass this check.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        # Razorpay fetch fails — server cannot verify
        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=None)
        svc._razorpay.fetch_payment = AsyncMock(
            return_value=RazorpayApiResult(
                success=False,
                error="Verification failed",
                status_code=500,
            )
        )

        await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        # Subscription creation must NOT have been called
        svc._subscription_repo.create.assert_not_called()
        svc._subscription_repo.extend_subscription.assert_not_called()

    def test_webhook_signature_uses_constant_time_comparison(self) -> None:
        """
        Verify Razorpay signature comparison uses hmac.compare_digest
        (constant-time) — not == or string comparison — to prevent
        timing-based signature forgery attacks.
        """
        import inspect
        import app.payments.razorpay_client as rc_module
        source = inspect.getsource(rc_module.RazorpayClient.verify_webhook_signature)
        assert "compare_digest" in source, (
            "verify_webhook_signature must use hmac.compare_digest "
            "for constant-time comparison to prevent timing attacks."
        )

    def test_payment_signature_uses_constant_time_comparison(self) -> None:
        """verify_payment_signature must also use constant-time comparison."""
        import inspect
        import app.payments.razorpay_client as rc_module
        source = inspect.getsource(rc_module.RazorpayClient.verify_payment_signature)
        assert "compare_digest" in source, (
            "verify_payment_signature must use hmac.compare_digest."
        )

    @pytest.mark.asyncio
    async def test_process_webhook_never_raises(self) -> None:
        """
        process_webhook_event must NEVER raise an exception — it must
        always return a WebhookProcessResult. Raising would cause the
        webhook handler to return a 500, triggering Razorpay retries.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        # Simulate a catastrophic unexpected error deep in the call stack
        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(
            side_effect=Exception("Catastrophic DB failure")
        )

        # Must not raise
        result = await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        assert isinstance(result, WebhookProcessResult)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_plan_tier_in_subscription_creation(self) -> None:
        """
        Subscription created after payment.captured must not include
        a plan tier. One tier only — billing_cycle is the only variable.
        """
        svc = _make_payment_service()
        db  = _mock_db()

        rz_payment = _mock_razorpay_payment()
        svc._payment_repo.get_by_razorpay_payment_id = AsyncMock(return_value=None)
        svc._razorpay.fetch_payment = AsyncMock(
            return_value=RazorpayApiResult(success=True, data=rz_payment, status_code=200)
        )

        payment_rec = _mock_payment_record(status=PaymentStatus.PENDING)
        payment_rec.billing_cycle = "monthly"
        svc._payment_repo.get_by_order_id_unscoped = AsyncMock(return_value=payment_rec)
        svc._payment_repo.update_status            = AsyncMock(return_value=payment_rec)
        svc._business_repo.get_by_id              = AsyncMock(return_value=_mock_business())

        created_kwargs: dict = {}

        async def _capture_create(**kwargs):
            created_kwargs.update(kwargs)
            mock = MagicMock()
            mock.id = uuid.uuid4()
            return mock

        svc._subscription_repo.get_active = AsyncMock(return_value=None)
        svc._subscription_repo.create     = _capture_create

        await svc.process_webhook_event(
            db=db,
            event_type="payment.captured",
            event_payload=_payment_captured_event(),
        )

        # plan or plan_name must NOT appear in subscription creation args
        assert "plan" not in created_kwargs, (
            "Subscription creation must not include a plan tier field. "
            "One tier only — billing_cycle is the only variable."
        )
        assert "plan_name" not in created_kwargs