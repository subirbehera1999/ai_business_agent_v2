# ==============================================================================
# File: tests/test_api.py
# Purpose: Integration-style tests for all HTTP API endpoints.
#
#          Tests cover:
#            - Health endpoints (liveness + readiness)
#            - Onboarding endpoints (register, profile CRUD, google connect)
#            - Payment endpoints (initiate, status, subscription)
#            - Webhook endpoints (razorpay receiver, ping)
#
#          Test strategy:
#            - Uses FastAPI TestClient (synchronous HTTPX wrapper)
#            - Database operations are mocked via pytest monkeypatch /
#              unittest.mock.AsyncMock so tests run without a real database
#            - JWT tokens are generated inline using the same algorithm and
#              secret as production (JWT_SECRET_KEY set in test env)
#            - Each test is independent — no shared mutable state
#
#          Running tests:
#            pytest tests/test_api.py -v
#            pytest tests/test_api.py -v -k "health"   # run only health tests
#
#          Environment:
#            Tests set required env variables at module level before importing
#            any app module, because settings are loaded at import time.
# ==============================================================================

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from collections.abc import Iterator
import jwt
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Set required environment variables BEFORE importing any app module.
# app/config/settings.py validates at import time.
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL",         "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("JWT_SECRET_KEY",       "test-secret-key-at-least-32-characters-long!!")
os.environ.setdefault("RAZORPAY_KEY_ID",      "rzp_test_key_id")
os.environ.setdefault("RAZORPAY_KEY_SECRET",  "rzp_test_key_secret")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET", "rzp_test_webhook_secret")
os.environ.setdefault("WHATSAPP_API_TOKEN",   "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "test_phone_number_id")
os.environ.setdefault("OPENAI_API_KEY",       "sk-test-openai-key")
os.environ.setdefault("ADMIN_WHATSAPP_NUMBER","+919999999999")

from app.main import app  # noqa: E402 — must come after env setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JWT_SECRET    = os.environ["JWT_SECRET_KEY"]
_JWT_ALGORITHM = "HS256"

_VALID_BUSINESS_ID   = str(uuid.uuid4())
_VALID_BUSINESS_NAME = "Sunrise Cafe"
_VALID_WHATSAPP      = "9876543210"
_VALID_LOCATION_ID   = "ChIJN1t_tDeuEmsRUsoyG83frY4"
_VALID_BUSINESS_TYPE = "restaurant"


# ==============================================================================
# Helpers
# ==============================================================================

def _make_access_token(
    business_id: str = _VALID_BUSINESS_ID,
    expires_in_minutes: int = 60,
) -> str:
    """
    Generate a valid test JWT access token.

    Uses the same algorithm and secret as production so auth_middleware
    accepts it without modification.
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub":  business_id,
        "jti":  str(uuid.uuid4()),
        "type": "access",
        "iat":  int(now.timestamp()),
        "exp":  int((now + timedelta(minutes=expires_in_minutes)).timestamp()),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _auth_headers(business_id: str = _VALID_BUSINESS_ID) -> dict:
    """Return Authorization header dict for authenticated requests."""
    return {"Authorization": f"Bearer {_make_access_token(business_id)}"}


def _expired_auth_headers() -> dict:
    """Return an Authorization header with an already-expired token."""
    return {"Authorization": f"Bearer {_make_access_token(expires_in_minutes=-5)}"}


def _razorpay_signature(payload_bytes: bytes, secret: str = "rzp_test_key_secret") -> str:
    """Compute a valid Razorpay HMAC-SHA256 webhook signature."""
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def _mock_business() -> MagicMock:
    """Return a minimal mock BusinessModel instance."""
    b = MagicMock()
    b.id              = uuid.UUID(_VALID_BUSINESS_ID)
    b.business_name   = _VALID_BUSINESS_NAME
    b.whatsapp_number = _VALID_WHATSAPP
    b.is_active       = True
    b.is_deleted      = False
    b.onboarding_complete = True
    return b


def _mock_subscription() -> MagicMock:
    """Return a minimal mock SubscriptionModel instance."""
    s = MagicMock()
    s.id              = uuid.uuid4()
    s.business_id     = uuid.UUID(_VALID_BUSINESS_ID)
    s.billing_cycle   = "monthly"
    s.status          = "active"
    s.expires_at      = datetime.now(tz=timezone.utc) + timedelta(days=25)
    s.amount          = 999.0
    return s


# ==============================================================================
# Fixtures
# ==============================================================================



@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with (
        patch("app.database.db.connect_database",    new_callable=AsyncMock),
        patch("app.database.db.disconnect_database", new_callable=AsyncMock),
        patch("app.schedulers.scheduler_manager.SchedulerManager.start",    return_value=None),
        patch("app.schedulers.scheduler_manager.SchedulerManager.shutdown", return_value=None),
        patch("app.schedulers.scheduler_manager.SchedulerManager.get_job_count", return_value=9),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def auth_mock_business():
    """
    Patch require_auth to return a mock business without hitting the database.
    Also patches subscription check so all authenticated requests pass.
    """
    mock_biz = _mock_business()
    mock_sub = _mock_subscription()

    with (
        patch(
            "app.security.auth_middleware.require_auth",
            return_value=mock_biz,
        ),
        patch(
            "app.repositories.subscription_repository.SubscriptionRepository.get_active",
            new_callable=AsyncMock,
            return_value=mock_sub,
        ),
        patch(
            "app.repositories.business_repository.BusinessRepository.get_by_id",
            new_callable=AsyncMock,
            return_value=mock_biz,
        ),
    ):
        yield mock_biz


# ==============================================================================
# 1. Health Endpoints
# ==============================================================================

class TestHealthEndpoints:
    """Tests for GET /api/v1/health and GET /api/v1/health/detailed."""

    def test_liveness_returns_200(self, client: TestClient) -> None:
        """Liveness probe must return 200 with no auth required."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_liveness_response_structure(self, client: TestClient) -> None:
        """Liveness response must include status and service fields."""
        response = client.get("/api/v1/health")
        body = response.json()
        assert body["status"] == "ok"
        assert "service" in body or "message" in body

    def test_liveness_no_auth_required(self, client: TestClient) -> None:
        """Liveness probe must succeed without an Authorization header."""
        response = client.get("/api/v1/health")
        assert response.status_code != 401
        assert response.status_code != 403

    def test_detailed_health_no_auth_required(self, client: TestClient) -> None:
        """Detailed health check must be publicly accessible."""
        with patch(
            "app.utils.system_health.run_all_health_checks",
            new_callable=AsyncMock,
            return_value={
                "database": True,
                "scheduler": True,
                "google_api": True,
                "whatsapp_api": True,
                "openai_api": True,
            },
        ):
            response = client.get("/api/v1/health/detailed")
            assert response.status_code in (200, 503)

    def test_detailed_health_503_when_db_down(self, client: TestClient) -> None:
        """Detailed health must return 503 when any check fails."""
        with patch(
            "app.utils.system_health.run_all_health_checks",
            new_callable=AsyncMock,
            return_value={
                "database": False,
                "scheduler": True,
                "google_api": True,
                "whatsapp_api": True,
                "openai_api": True,
            },
        ):
            response = client.get("/api/v1/health/detailed")
            assert response.status_code == 503

    def test_root_endpoint_returns_200(self, client: TestClient) -> None:
        """Root / endpoint must return 200 with service metadata."""
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "service" in body
        assert "version" in body


# ==============================================================================
# 2. Onboarding Endpoints
# ==============================================================================

class TestOnboardingRegister:
    """Tests for POST /api/v1/onboarding/register."""

    _URL = "/api/v1/onboarding/register"

    def _valid_payload(self) -> dict:
        return {
            "business_name":    _VALID_BUSINESS_NAME,
            "whatsapp_number":  _VALID_WHATSAPP,
            "google_location_id": _VALID_LOCATION_ID,
            "business_type":    _VALID_BUSINESS_TYPE,
        }

    def test_register_returns_201_on_success(self, client: TestClient) -> None:
        """Successful registration must return HTTP 201."""
        mock_biz = _mock_business()
        with (
            patch(
                "app.repositories.business_repository.BusinessRepository.get_by_whatsapp",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.repositories.business_repository.BusinessRepository.create",
                new_callable=AsyncMock,
                return_value=mock_biz,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.post(self._URL, json=self._valid_payload())
            assert response.status_code == 201

    def test_register_response_has_business_id(self, client: TestClient) -> None:
        """Registration response must include the new business_id."""
        mock_biz = _mock_business()
        with (
            patch(
                "app.repositories.business_repository.BusinessRepository.get_by_whatsapp",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.repositories.business_repository.BusinessRepository.create",
                new_callable=AsyncMock,
                return_value=mock_biz,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.post(self._URL, json=self._valid_payload())
            body = response.json()
            assert "data" in body or "business_id" in str(body)

    def test_register_missing_business_name_returns_422(self, client: TestClient) -> None:
        """Missing required field must return HTTP 422 Unprocessable Entity."""
        payload = self._valid_payload()
        del payload["business_name"]
        response = client.post(self._URL, json=payload)
        assert response.status_code == 422

    def test_register_missing_whatsapp_returns_422(self, client: TestClient) -> None:
        """Missing whatsapp_number must return HTTP 422."""
        payload = self._valid_payload()
        del payload["whatsapp_number"]
        response = client.post(self._URL, json=payload)
        assert response.status_code == 422

    def test_register_invalid_business_type_returns_422(self, client: TestClient) -> None:
        """Invalid business_type must be rejected with 422."""
        payload = self._valid_payload()
        payload["business_type"] = "invalid_type_xyz"
        response = client.post(self._URL, json=payload)
        assert response.status_code == 422

    def test_register_invalid_whatsapp_format_returns_422(self, client: TestClient) -> None:
        """Non-numeric / wrong-length WhatsApp number must return 422."""
        payload = self._valid_payload()
        payload["whatsapp_number"] = "not-a-number"
        response = client.post(self._URL, json=payload)
        assert response.status_code == 422

    def test_register_empty_payload_returns_422(self, client: TestClient) -> None:
        """Empty request body must return 422."""
        response = client.post(self._URL, json={})
        assert response.status_code == 422

    def test_register_no_content_type_returns_422(self, client: TestClient) -> None:
        """Request without JSON body must be rejected."""
        response = client.post(self._URL, data="not-json")
        assert response.status_code in (400, 422)


class TestOnboardingProfile:
    """Tests for GET /api/v1/onboarding/profile and PATCH /api/v1/onboarding/profile."""

    _GET_URL   = "/api/v1/onboarding/profile"
    _PATCH_URL = "/api/v1/onboarding/profile"

    def test_get_profile_requires_auth(self, client: TestClient) -> None:
        """GET profile without a token must return 401."""
        response = client.get(self._GET_URL)
        assert response.status_code == 401

    def test_get_profile_returns_200_with_valid_token(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """GET profile with valid auth must return 200."""
        with patch("app.database.db.get_db_session"):
            response = client.get(
                self._GET_URL,
                headers=_auth_headers(),
            )
            assert response.status_code == 200

    def test_get_profile_expired_token_returns_401(self, client: TestClient) -> None:
        """Expired JWT must be rejected with 401."""
        response = client.get(
            self._GET_URL,
            headers=_expired_auth_headers(),
        )
        assert response.status_code == 401

    def test_patch_profile_requires_auth(self, client: TestClient) -> None:
        """PATCH profile without token must return 401."""
        response = client.patch(
            self._PATCH_URL,
            json={"city": "Mumbai"},
        )
        assert response.status_code == 401

    def test_patch_profile_returns_200_on_valid_update(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """PATCH profile with a valid field update must return 200."""
        mock_biz = _mock_business()
        with (
            patch(
                "app.repositories.business_repository.BusinessRepository.update",
                new_callable=AsyncMock,
                return_value=mock_biz,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.patch(
                self._PATCH_URL,
                json={"city": "Mumbai"},
                headers=_auth_headers(),
            )
            assert response.status_code == 200

    def test_patch_profile_empty_body_returns_422(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """PATCH with an empty body must be rejected (at least one field required)."""
        with patch("app.database.db.get_db_session"):
            response = client.patch(
                self._PATCH_URL,
                json={},
                headers=_auth_headers(),
            )
            assert response.status_code == 422


# ==============================================================================
# 3. Payment Endpoints
# ==============================================================================

class TestPaymentInitiate:
    """Tests for POST /api/v1/payments/initiate."""

    _URL = "/api/v1/payments/initiate"

    def _valid_payload(self) -> dict:
        return {"billing_cycle": "monthly"}

    def test_initiate_requires_auth(self, client: TestClient) -> None:
        """Payment initiation without token must return 401."""
        response = client.post(self._URL, json=self._valid_payload())
        assert response.status_code == 401

    def test_initiate_returns_201_on_success(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Successful payment initiation must return 201."""
        mock_result = MagicMock()
        mock_result.success         = True
        mock_result.razorpay_order_id = "order_test123"
        mock_result.amount_paise    = 99900
        mock_result.currency        = "INR"
        mock_result.billing_cycle   = "monthly"

        with (
            patch(
                "app.payments.payment_service.PaymentService.initiate_payment",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.post(
                self._URL,
                json=self._valid_payload(),
                headers=_auth_headers(),
            )
            assert response.status_code == 201

    def test_initiate_invalid_billing_cycle_returns_422(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Invalid billing_cycle value must return 422."""
        with patch("app.database.db.get_db_session"):
            response = client.post(
                self._URL,
                json={"billing_cycle": "weekly"},
                headers=_auth_headers(),
            )
            assert response.status_code == 422

    def test_initiate_missing_billing_cycle_returns_422(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Missing billing_cycle field must return 422."""
        with patch("app.database.db.get_db_session"):
            response = client.post(
                self._URL,
                json={},
                headers=_auth_headers(),
            )
            assert response.status_code == 422

    def test_initiate_annual_billing_cycle_accepted(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """'annual' billing_cycle must be a valid input."""
        mock_result = MagicMock()
        mock_result.success           = True
        mock_result.razorpay_order_id = "order_annual123"
        mock_result.amount_paise      = 999900
        mock_result.currency          = "INR"
        mock_result.billing_cycle     = "annual"

        with (
            patch(
                "app.payments.payment_service.PaymentService.initiate_payment",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.post(
                self._URL,
                json={"billing_cycle": "annual"},
                headers=_auth_headers(),
            )
            assert response.status_code == 201


class TestPaymentStatus:
    """Tests for GET /api/v1/payments/status/{razorpay_order_id}."""

    _BASE_URL = "/api/v1/payments/status"

    def test_status_requires_auth(self, client: TestClient) -> None:
        """Payment status check without token must return 401."""
        response = client.get(f"{self._BASE_URL}/order_test123")
        assert response.status_code == 401

    def test_status_returns_200_for_existing_order(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Status check for an existing order must return 200."""
        mock_payment = MagicMock()
        mock_payment.id                  = uuid.uuid4()
        mock_payment.business_id         = uuid.UUID(_VALID_BUSINESS_ID)
        mock_payment.razorpay_order_id   = "order_test123"
        mock_payment.razorpay_payment_id = "pay_test123"
        mock_payment.status              = "captured"
        mock_payment.amount              = 99900
        mock_payment.billing_cycle       = "monthly"

        with (
            patch(
                "app.payments.payment_service.PaymentService.get_payment_status",
                new_callable=AsyncMock,
                return_value=mock_payment,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.get(
                f"{self._BASE_URL}/order_test123",
                headers=_auth_headers(),
            )
            assert response.status_code == 200

    def test_status_returns_404_for_unknown_order(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Status check for an unknown order must return 404."""
        with (
            patch(
                "app.payments.payment_service.PaymentService.get_payment_status",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.get(
                f"{self._BASE_URL}/nonexistent_order",
                headers=_auth_headers(),
            )
            assert response.status_code == 404


class TestSubscriptionStatus:
    """Tests for GET /api/v1/payments/subscription."""

    _URL = "/api/v1/payments/subscription"

    def test_subscription_requires_auth(self, client: TestClient) -> None:
        """Subscription status without token must return 401."""
        response = client.get(self._URL)
        assert response.status_code == 401

    def test_subscription_returns_200_with_active_subscription(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """Active subscription fetch must return 200."""
        mock_sub = _mock_subscription()
        with (
            patch(
                "app.repositories.subscription_repository.SubscriptionRepository.get_active",
                new_callable=AsyncMock,
                return_value=mock_sub,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.get(
                self._URL,
                headers=_auth_headers(),
            )
            assert response.status_code == 200

    def test_subscription_returns_404_when_no_subscription(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """No active subscription must return 404."""
        with (
            patch(
                "app.repositories.subscription_repository.SubscriptionRepository.get_active",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.get(
                self._URL,
                headers=_auth_headers(),
            )
            assert response.status_code == 404


# ==============================================================================
# 4. Webhook Endpoints
# ==============================================================================

class TestWebhookEndpoints:
    """Tests for POST /api/v1/webhooks/razorpay and GET /api/v1/webhooks/razorpay/ping."""

    _WEBHOOK_URL = "/api/v1/webhooks/razorpay"
    _PING_URL    = "/api/v1/webhooks/razorpay/ping"

    def _payment_captured_payload(self) -> dict:
        return {
            "event": "payment.captured",
            "payload": {
                "payment": {
                    "entity": {
                        "id":       "pay_test123",
                        "order_id": "order_test123",
                        "amount":   99900,
                        "currency": "INR",
                        "status":   "captured",
                    }
                }
            },
        }

    def test_ping_returns_200(self, client: TestClient) -> None:
        """Webhook ping must return 200 without auth."""
        response = client.get(self._PING_URL)
        assert response.status_code == 200

    def test_ping_response_structure(self, client: TestClient) -> None:
        """Webhook ping must return expected status field."""
        response = client.get(self._PING_URL)
        body = response.json()
        assert body.get("status") == "ok"

    def test_webhook_no_signature_returns_400(self, client: TestClient) -> None:
        """Webhook request with no signature header must return 400."""
        payload = self._payment_captured_payload()
        response = client.post(
            self._WEBHOOK_URL,
            json=payload,
        )
        assert response.status_code == 400

    def test_webhook_invalid_signature_returns_400(self, client: TestClient) -> None:
        """Webhook request with wrong signature must return 400."""
        payload_bytes = json.dumps(self._payment_captured_payload()).encode()
        response = client.post(
            self._WEBHOOK_URL,
            content=payload_bytes,
            headers={
                "Content-Type":         "application/json",
                "X-Razorpay-Signature": "invalid_signature_string",
            },
        )
        assert response.status_code == 400

    def test_webhook_valid_signature_returns_200(self, client: TestClient) -> None:
        """Webhook request with valid HMAC signature must return 200."""
        payload_dict  = self._payment_captured_payload()
        payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode()
        signature     = _razorpay_signature(payload_bytes)

        mock_result = MagicMock()
        mock_result.accepted         = True
        mock_result.http_status_code = 200
        mock_result.event_type       = "payment.captured"
        mock_result.event_id         = "evt_test123"
        mock_result.process_result   = MagicMock(success=True)

        with (
            patch(
                "app.payments.webhook_handler.RazorpayWebhookHandler.handle",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.post(
                self._WEBHOOK_URL,
                content=payload_bytes,
                headers={
                    "Content-Type":         "application/json",
                    "X-Razorpay-Signature": signature,
                },
            )
            assert response.status_code == 200

    def test_webhook_does_not_require_jwt_auth(self, client: TestClient) -> None:
        """
        Webhook endpoint must never require a JWT Authorization header.
        Security is HMAC-only. This test ensures the endpoint is not
        accidentally protected by AuthMiddleware.
        """
        payload_bytes = json.dumps(self._payment_captured_payload()).encode()
        signature     = _razorpay_signature(payload_bytes)

        # Sending without any Authorization header — should not get 401
        response = client.post(
            self._WEBHOOK_URL,
            content=payload_bytes,
            headers={
                "Content-Type":         "application/json",
                "X-Razorpay-Signature": signature,
            },
        )
        assert response.status_code != 401

    def test_webhook_empty_body_returns_400(self, client: TestClient) -> None:
        """Webhook with empty body must return 400."""
        response = client.post(
            self._WEBHOOK_URL,
            content=b"",
            headers={
                "Content-Type":         "application/json",
                "X-Razorpay-Signature": "some_signature",
            },
        )
        assert response.status_code == 400

    def test_webhook_malformed_json_returns_400(self, client: TestClient) -> None:
        """Webhook with malformed JSON (even with valid signature) must return 400."""
        bad_payload = b"{not valid json"
        signature   = _razorpay_signature(bad_payload)

        response = client.post(
            self._WEBHOOK_URL,
            content=bad_payload,
            headers={
                "Content-Type":         "application/json",
                "X-Razorpay-Signature": signature,
            },
        )
        assert response.status_code == 400


# ==============================================================================
# 5. Authentication Edge Cases
# ==============================================================================

class TestAuthenticationEdgeCases:
    """Cross-cutting authentication behaviour across all protected endpoints."""

    _PROTECTED_ENDPOINTS = [
        ("GET",   "/api/v1/onboarding/profile"),
        ("PATCH", "/api/v1/onboarding/profile"),
        ("POST",  "/api/v1/payments/initiate"),
        ("GET",   "/api/v1/payments/subscription"),
    ]

    def test_all_protected_endpoints_reject_missing_token(
        self,
        client: TestClient,
    ) -> None:
        """Every protected endpoint must return 401 with no Authorization header."""
        for method, url in self._PROTECTED_ENDPOINTS:
            response = client.request(method, url)
            assert response.status_code == 401, (
                f"Expected 401 for {method} {url}, got {response.status_code}"
            )

    def test_all_protected_endpoints_reject_expired_token(
        self,
        client: TestClient,
    ) -> None:
        """Every protected endpoint must return 401 with an expired token."""
        headers = _expired_auth_headers()
        for method, url in self._PROTECTED_ENDPOINTS:
            response = client.request(method, url, headers=headers)
            assert response.status_code == 401, (
                f"Expected 401 for {method} {url} with expired token, "
                f"got {response.status_code}"
            )

    def test_all_protected_endpoints_reject_malformed_token(
        self,
        client: TestClient,
    ) -> None:
        """Malformed token string must be rejected with 401."""
        headers = {"Authorization": "Bearer this.is.not.a.jwt"}
        for method, url in self._PROTECTED_ENDPOINTS:
            response = client.request(method, url, headers=headers)
            assert response.status_code == 401, (
                f"Expected 401 for {method} {url} with bad token, "
                f"got {response.status_code}"
            )

    def test_public_endpoints_never_require_token(
        self,
        client: TestClient,
    ) -> None:
        """Public endpoints must return non-401 without any Authorization header."""
        public_endpoints = [
            ("GET",  "/api/v1/health"),
            ("GET",  "/api/v1/health/detailed"),
            ("POST", "/api/v1/onboarding/register"),
            ("GET",  "/api/v1/webhooks/razorpay/ping"),
        ]
        for method, url in public_endpoints:
            response = client.request(method, url)
            assert response.status_code != 401, (
                f"Public endpoint {method} {url} should not require auth, "
                f"got {response.status_code}"
            )


# ==============================================================================
# 6. Response Structure Contracts
# ==============================================================================

class TestResponseStructure:
    """Verify that all successful responses follow the standard envelope."""

    def test_health_response_is_json(self, client: TestClient) -> None:
        """Health endpoint must return JSON content-type."""
        response = client.get("/api/v1/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_422_response_has_detail_field(self, client: TestClient) -> None:
        """FastAPI validation errors (422) must include a 'detail' field."""
        response = client.post(
            "/api/v1/onboarding/register",
            json={},
        )
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body

    def test_unhandled_error_returns_500_not_traceback(
        self,
        client: TestClient,
        auth_mock_business,
    ) -> None:
        """
        Unhandled exceptions must return 500 with a safe message.
        Internal error details (tracebacks, model paths) must never
        appear in the response body.
        """
        with (
            patch(
                "app.repositories.subscription_repository.SubscriptionRepository.get_active",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Simulated DB crash"),
            ),
            patch("app.database.db.get_db_session"),
        ):
            response = client.get(
                "/api/v1/payments/subscription",
                headers=_auth_headers(),
            )
            assert response.status_code in (500, 503)
            body_text = response.text
            # Must not leak internal paths or class names
            assert "Traceback" not in body_text
            assert "app/repositories" not in body_text