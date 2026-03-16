# ==============================================================================
# File: app/payments/razorpay_client.py
# Purpose: Low-level client for the Razorpay API.
#          Handles all direct Razorpay API communication:
#
#            - create_order()          → create a new payment order
#            - fetch_payment()         → retrieve payment details by ID
#            - fetch_order()           → retrieve order details by ID
#            - verify_payment_signature() → HMAC-SHA256 webhook/callback
#                                          signature verification
#            - create_refund()         → initiate a refund for a payment
#
#          Architecture:
#            This is a pure integration client — no business logic.
#            payment_service.py orchestrates the payment workflow and
#            calls this client for all Razorpay operations.
#            webhook_handler.py calls verify_payment_signature() directly
#            for webhook event authentication.
#
#          Authentication:
#            HTTP Basic Auth using RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.
#            All requests are made over HTTPS.
#            Secret is NEVER logged, included in error messages, or
#            exposed outside this module.
#
#          Payment data protection:
#            Per DATA_SAFETY_AND_RUNTIME_GUARDRAILS:
#              - Payment verification is always server-side
#              - Frontend payment confirmation is never trusted
#              - Signature verification happens before any subscription update
#              - All amounts are in paise (smallest INR unit) — integer only
#
#          Retry policy:
#            Uses with_razorpay_retry (3 attempts, exponential backoff).
#            Idempotency keys are passed on order creation to prevent
#            duplicate orders on network retry.
#
#          HTTP client: httpx (sync) — Razorpay's official Python SDK
#            uses requests internally; we use httpx directly to stay
#            async-compatible and avoid the SDK's lack of async support.
# ==============================================================================

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_razorpay_retry

logger = logging.getLogger(ServiceName.PAYMENTS)
settings = get_settings()

# ---------------------------------------------------------------------------
# Razorpay API base URL
# ---------------------------------------------------------------------------
_RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
_ORDERS_URL    = f"{_RAZORPAY_BASE_URL}/orders"
_PAYMENTS_URL  = f"{_RAZORPAY_BASE_URL}/payments"
_REFUNDS_URL   = f"{_RAZORPAY_BASE_URL}/refunds"

# Currency for all transactions
_CURRENCY = "INR"


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class RazorpayOrder:
    """
    A Razorpay order created via the Orders API.

    Attributes:
        order_id:       Razorpay order ID e.g. "order_MXf9k3..."
        amount:         Amount in paise (e.g. 49900 = ₹499.00).
        currency:       Always "INR".
        receipt:        Merchant-provided receipt ID (our idempotency key).
        status:         "created", "attempted", "paid".
        created_at:     Unix timestamp of order creation.
    """
    order_id: str
    amount: int
    currency: str
    receipt: str
    status: str
    created_at: int

    @property
    def amount_rupees(self) -> float:
        return self.amount / 100


@dataclass
class RazorpayPayment:
    """
    A Razorpay payment record.

    Attributes:
        payment_id:  Razorpay payment ID e.g. "pay_MXf9k3..."
        order_id:    Associated Razorpay order ID.
        amount:      Amount captured in paise.
        currency:    Always "INR".
        status:      "created", "authorized", "captured", "refunded", "failed".
        method:      Payment method: "card", "upi", "netbanking", "wallet".
        captured:    True if payment is captured (funds secured).
        email:       Payer email (may be empty — not required for UPI).
        contact:     Payer phone number.
        created_at:  Unix timestamp.
        error_code:  Razorpay error code if status="failed".
        error_desc:  Human-readable failure reason if status="failed".
    """
    payment_id: str
    order_id: str
    amount: int
    currency: str
    status: str
    method: str
    captured: bool
    email: str
    contact: str
    created_at: int
    error_code: Optional[str] = None
    error_desc: Optional[str] = None

    @property
    def is_captured(self) -> bool:
        return self.status == "captured" and self.captured

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def amount_rupees(self) -> float:
        return self.amount / 100


@dataclass
class RazorpayApiResult:
    """
    Wrapper for all Razorpay API responses.

    Attributes:
        success:      True if the API call succeeded.
        data:         Typed result object (RazorpayOrder, RazorpayPayment, etc.)
        error:        Human-readable error message if success=False.
        error_code:   Razorpay error code string if available.
        status_code:  HTTP status code.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    status_code: Optional[int] = None

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403)

    @property
    def is_not_found(self) -> bool:
        return self.status_code == 404


# ==============================================================================
# Razorpay Client
# ==============================================================================

class RazorpayClient:
    """
    Async client for the Razorpay REST API.

    Uses HTTP Basic Auth with RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET.
    Instantiated once per application and shared across payment services.

    Usage:
        client = RazorpayClient()
        await client.initialise()

        result = await client.create_order(
            amount_paise=49900,
            receipt="RECEIPT_BIZ_42_PLAN_PRO",
            idempotency_key="IDEM_BIZ_42_PLAN_PRO_20250315",
        )

        if result.success:
            order = result.data  # RazorpayOrder
            print(order.order_id)

        await client.close()
    """

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None
        self._key_id: str = settings.RAZORPAY_KEY_ID
        self._key_secret: str = settings.RAZORPAY_KEY_SECRET

    async def initialise(self) -> None:
        """
        Initialise the shared HTTP client with Basic Auth credentials.
        Must be called before any API methods are used.
        """
        self._http = httpx.AsyncClient(
            auth=(self._key_id, self._key_secret),
            timeout=httpx.Timeout(settings.EXTERNAL_API_TIMEOUT_SECONDS),
            follow_redirects=False,
            headers={
                "User-Agent": "AIBusinessAgent/1.0",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "RazorpayClient initialised",
            extra={
                "service": ServiceName.PAYMENTS,
                "key_id": self._key_id[:8] + "...",   # log prefix only
            },
        )

    async def close(self) -> None:
        """Close the shared HTTP client. Called from app lifespan shutdown."""
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info(
            "RazorpayClient closed",
            extra={"service": ServiceName.PAYMENTS},
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @with_razorpay_retry
    async def create_order(
        self,
        amount_paise: int,
        receipt: str,
        idempotency_key: str,
        notes: Optional[dict[str, str]] = None,
    ) -> RazorpayApiResult:
        """
        Create a new Razorpay payment order.

        The order represents the intent to collect a specific amount.
        The frontend uses the order_id to render the Razorpay checkout
        widget. Payment capture happens after the user pays and the
        webhook confirms it.

        Idempotency:
            The receipt field acts as a merchant-side idempotency key.
            If create_order is called twice with the same receipt,
            Razorpay returns the existing order rather than creating a
            duplicate. This protects against network retries.

        Args:
            amount_paise:     Amount in paise (integer). ₹499 = 49900 paise.
            receipt:          Merchant receipt ID — must be unique per order.
                              Format: "RCPT_{business_id}_{plan}_{timestamp}"
            idempotency_key:  Idempotency key sent in the Razorpay-Idempotency-Key
                              header. Prevents duplicate orders on retry.
            notes:            Optional key-value metadata attached to the order.
                              Visible in Razorpay Dashboard. Max 15 key-value pairs.
                              Keys and values are strings, max 256 chars each.

        Returns:
            RazorpayApiResult with data=RazorpayOrder.
        """
        self._ensure_initialised()

        payload: dict[str, Any] = {
            "amount": amount_paise,
            "currency": _CURRENCY,
            "receipt": receipt[:40],    # Razorpay receipt max 40 chars
        }
        if notes:
            # Razorpay notes: max 15 pairs, keys/values truncated to 256 chars
            payload["notes"] = {
                str(k)[:256]: str(v)[:256]
                for k, v in list(notes.items())[:15]
            }

        headers = {"Razorpay-Idempotency-Key": idempotency_key}

        log_extra = {
            "service": ServiceName.PAYMENTS,
            "amount_paise": amount_paise,
            "receipt": receipt,
        }

        try:
            response = await self._http.post(
                _ORDERS_URL,
                json=payload,
                headers=headers,
            )

            if response.status_code in (200, 201):
                body = response.json()
                order = _parse_order(body)
                logger.info(
                    "Razorpay order created",
                    extra={**log_extra, "order_id": order.order_id},
                )
                return RazorpayApiResult(
                    success=True,
                    data=order,
                    status_code=response.status_code,
                )

            return _handle_razorpay_error(response, log_extra, "create_order")

        except httpx.TimeoutException as exc:
            logger.error("create_order timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("create_order HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    @with_razorpay_retry
    async def fetch_order(
        self,
        order_id: str,
    ) -> RazorpayApiResult:
        """
        Fetch a Razorpay order by its order_id.

        Used to verify order status before activating a subscription.
        An order must have status="paid" before the subscription is activated.

        Args:
            order_id: Razorpay order ID e.g. "order_MXf9k3..."

        Returns:
            RazorpayApiResult with data=RazorpayOrder.
        """
        self._ensure_initialised()
        log_extra = {
            "service": ServiceName.PAYMENTS,
            "order_id": order_id,
        }

        try:
            response = await self._http.get(f"{_ORDERS_URL}/{order_id}")

            if response.status_code == 200:
                body = response.json()
                order = _parse_order(body)
                logger.debug(
                    "Razorpay order fetched",
                    extra={**log_extra, "status": order.status},
                )
                return RazorpayApiResult(
                    success=True,
                    data=order,
                    status_code=200,
                )

            return _handle_razorpay_error(response, log_extra, "fetch_order")

        except httpx.TimeoutException as exc:
            logger.error("fetch_order timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("fetch_order HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    # ------------------------------------------------------------------
    # Payment management
    # ------------------------------------------------------------------

    @with_razorpay_retry
    async def fetch_payment(
        self,
        payment_id: str,
    ) -> RazorpayApiResult:
        """
        Fetch a Razorpay payment by its payment_id.

        Used during webhook processing to independently verify payment
        details server-side. Never trust client-reported payment status.

        Args:
            payment_id: Razorpay payment ID e.g. "pay_MXf9k3..."

        Returns:
            RazorpayApiResult with data=RazorpayPayment.
        """
        self._ensure_initialised()
        log_extra = {
            "service": ServiceName.PAYMENTS,
            "payment_id": payment_id,
        }

        try:
            response = await self._http.get(f"{_PAYMENTS_URL}/{payment_id}")

            if response.status_code == 200:
                body = response.json()
                payment = _parse_payment(body)
                logger.info(
                    "Razorpay payment fetched",
                    extra={
                        **log_extra,
                        "status": payment.status,
                        "amount_paise": payment.amount,
                        "captured": payment.captured,
                    },
                )
                return RazorpayApiResult(
                    success=True,
                    data=payment,
                    status_code=200,
                )

            return _handle_razorpay_error(response, log_extra, "fetch_payment")

        except httpx.TimeoutException as exc:
            logger.error("fetch_payment timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("fetch_payment HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    # ------------------------------------------------------------------
    # Refunds
    # ------------------------------------------------------------------

    @with_razorpay_retry
    async def create_refund(
        self,
        payment_id: str,
        amount_paise: Optional[int] = None,
        notes: Optional[dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> RazorpayApiResult:
        """
        Initiate a refund for a captured payment.

        Supports full and partial refunds.
        Full refund: omit amount_paise (Razorpay refunds the full amount).
        Partial refund: specify amount_paise <= original payment amount.

        Args:
            payment_id:       Razorpay payment ID to refund.
            amount_paise:     Refund amount in paise. None = full refund.
            notes:            Optional metadata for the refund record.
            idempotency_key:  Prevents duplicate refunds on retry.

        Returns:
            RazorpayApiResult with data=dict of refund record.
        """
        self._ensure_initialised()

        payload: dict[str, Any] = {}
        if amount_paise is not None:
            payload["amount"] = amount_paise
        if notes:
            payload["notes"] = {
                str(k)[:256]: str(v)[:256]
                for k, v in list(notes.items())[:15]
            }

        request_headers: dict[str, str] = {}
        if idempotency_key:
            request_headers["Razorpay-Idempotency-Key"] = idempotency_key

        log_extra = {
            "service": ServiceName.PAYMENTS,
            "payment_id": payment_id,
            "amount_paise": amount_paise or "full",
        }

        try:
            response = await self._http.post(
                f"{_PAYMENTS_URL}/{payment_id}/refund",
                json=payload,
                headers=request_headers,
            )

            if response.status_code in (200, 201):
                body = response.json()
                refund_id = body.get("id", "")
                logger.info(
                    "Razorpay refund initiated",
                    extra={**log_extra, "refund_id": refund_id},
                )
                return RazorpayApiResult(
                    success=True,
                    data=body,
                    status_code=response.status_code,
                )

            return _handle_razorpay_error(response, log_extra, "create_refund")

        except httpx.TimeoutException as exc:
            logger.error("create_refund timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("create_refund HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    # ------------------------------------------------------------------
    # Signature verification — synchronous, no HTTP call
    # ------------------------------------------------------------------

    def verify_payment_signature(
        self,
        order_id: str,
        payment_id: str,
        signature: str,
    ) -> bool:
        """
        Verify the Razorpay payment signature from the checkout callback.

        Called after the user completes payment in the Razorpay widget.
        The frontend receives order_id, payment_id, and signature from
        Razorpay and forwards all three to our backend for server-side
        verification. NEVER activate a subscription based on frontend
        confirmation alone.

        Verification algorithm (Razorpay specification):
            expected = HMAC-SHA256(
                key    = RAZORPAY_KEY_SECRET,
                message = "{order_id}|{payment_id}"
            )
            valid = (expected == signature)

        This is a constant-time comparison to prevent timing attacks.

        Args:
            order_id:   Razorpay order ID from checkout.
            payment_id: Razorpay payment ID from checkout.
            signature:  HMAC signature from checkout callback.

        Returns:
            bool: True if signature is valid, False otherwise.
        """
        message = f"{order_id}|{payment_id}"
        expected = hmac.new(
            key=self._key_secret.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(expected, signature)

        logger.info(
            "Payment signature verification",
            extra={
                "service": ServiceName.PAYMENTS,
                "order_id": order_id,
                "payment_id": payment_id,
                "valid": is_valid,
            },
        )

        return is_valid

    def verify_webhook_signature(
        self,
        webhook_body: bytes,
        webhook_signature: str,
    ) -> bool:
        """
        Verify the Razorpay webhook signature from the X-Razorpay-Signature header.

        Called by webhook_handler.py for every incoming Razorpay webhook.
        The raw request body (bytes) must be used — not a parsed JSON dict,
        as JSON re-serialisation can alter key ordering and break the signature.

        Verification algorithm (Razorpay webhook specification):
            expected = HMAC-SHA256(
                key    = RAZORPAY_KEY_SECRET,
                message = raw_request_body_bytes
            )
            valid = (expected == webhook_signature)

        Args:
            webhook_body:      Raw HTTP request body as bytes.
            webhook_signature: Value of X-Razorpay-Signature header.

        Returns:
            bool: True if webhook is authentic, False if tampered/spoofed.
        """
        expected = hmac.new(
            key=self._key_secret.encode("utf-8"),
            msg=webhook_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(expected, webhook_signature)

        if not is_valid:
            logger.warning(
                "Webhook signature verification FAILED — possible spoofed request",
                extra={"service": ServiceName.PAYMENTS},
            )
        else:
            logger.debug(
                "Webhook signature verified",
                extra={"service": ServiceName.PAYMENTS},
            )

        return is_valid

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Raise if the HTTP client has not been initialised."""
        if self._http is None:
            raise RuntimeError(
                "RazorpayClient has not been initialised. "
                "Call await client.initialise() before use."
            )


# ==============================================================================
# Parse helpers
# ==============================================================================

def _parse_order(body: dict) -> RazorpayOrder:
    """
    Parse a raw Razorpay Orders API response into a RazorpayOrder.

    Args:
        body: Raw JSON dict from the Orders API.

    Returns:
        RazorpayOrder with typed fields.
    """
    return RazorpayOrder(
        order_id=body.get("id", ""),
        amount=body.get("amount", 0),
        currency=body.get("currency", _CURRENCY),
        receipt=body.get("receipt", ""),
        status=body.get("status", "created"),
        created_at=body.get("created_at", 0),
    )


def _parse_payment(body: dict) -> RazorpayPayment:
    """
    Parse a raw Razorpay Payments API response into a RazorpayPayment.

    Args:
        body: Raw JSON dict from the Payments API.

    Returns:
        RazorpayPayment with typed fields.
    """
    error_desc = body.get("error_description") or body.get("error_reason")
    return RazorpayPayment(
        payment_id=body.get("id", ""),
        order_id=body.get("order_id", ""),
        amount=body.get("amount", 0),
        currency=body.get("currency", _CURRENCY),
        status=body.get("status", ""),
        method=body.get("method", ""),
        captured=body.get("captured", False),
        email=body.get("email", ""),
        contact=body.get("contact", ""),
        created_at=body.get("created_at", 0),
        error_code=body.get("error_code"),
        error_desc=error_desc,
    )


def _handle_razorpay_error(
    response: httpx.Response,
    log_extra: dict,
    method_name: str,
) -> RazorpayApiResult:
    """
    Parse a non-2xx Razorpay API response into a RazorpayApiResult.

    Razorpay error response format:
        {
            "error": {
                "code": "BAD_REQUEST_ERROR",
                "description": "The order id is invalid.",
                "field": "order_id",
                "source": "business",
                "step": "payment_initiation",
                "reason": "input_validation_failed"
            }
        }

    Args:
        response:    The failed httpx.Response.
        log_extra:   Structured log context.
        method_name: Calling method name for log context.

    Returns:
        RazorpayApiResult with success=False and structured error info.
    """
    status_code = response.status_code

    try:
        body = response.json()
        error_block = body.get("error", {})
        error_code = error_block.get("code", "UNKNOWN_ERROR")
        error_desc = error_block.get("description", "No description provided")
        error_field = error_block.get("field", "")
        error_reason = error_block.get("reason", "")
    except Exception:
        error_code = "PARSE_ERROR"
        error_desc = response.text[:200]
        error_field = ""
        error_reason = ""

    error_msg = error_desc
    if error_field:
        error_msg += f" (field: {error_field})"
    if error_reason:
        error_msg += f" (reason: {error_reason})"

    logger.error(
        f"Razorpay {method_name} failed",
        extra={
            **log_extra,
            "status_code": status_code,
            "error_code": error_code,
            "error_desc": error_desc,
        },
    )

    return RazorpayApiResult(
        success=False,
        error=error_msg,
        error_code=error_code,
        status_code=status_code,
    )