# ==============================================================================
# File: app/payments/webhook_handler.py
# Purpose: Receives and authenticates Razorpay webhook HTTP requests.
#
#          Responsibilities:
#            1. Signature verification
#               - Reads raw request body BEFORE any JSON parsing
#               - Verifies X-Razorpay-Signature header using HMAC-SHA256
#               - Rejects unauthenticated requests with 400 (not 401/403,
#                 to avoid leaking that the endpoint exists)
#               - Uses razorpay_client.verify_webhook_signature() for
#                 constant-time comparison (no timing attacks)
#
#            2. Event parsing and routing
#               - Parses JSON payload after signature is verified
#               - Extracts event type from payload["event"]
#               - Delegates to payment_service.process_webhook_event()
#               - Always returns 200 OK if signature is valid, even if
#                 event processing fails internally — Razorpay interprets
#                 non-2xx as "retry needed" and will keep retrying
#
#            3. Idempotency guard
#               - Each webhook carries a unique event ID
#               - Duplicate events are acknowledged (200) without re-processing
#               - payment_service handles deeper idempotency per payment_id
#
#            4. Error isolation
#               - Signature failure      → 400 (reject immediately)
#               - JSON parse failure     → 400 (malformed payload)
#               - Processing exception  → 200 (acknowledged, logged as error)
#               - Returning 500 would cause Razorpay to retry indefinitely
#
#          Security contract (from DATA_SAFETY_AND_RUNTIME_GUARDRAILS):
#            - Signature is verified against the RAW request body bytes
#            - JSON parsing happens ONLY after signature passes
#            - No subscription activation occurs before signature check
#            - The raw body is never logged (may contain sensitive payment data)
#
#          Integration:
#            This handler is mounted as a FastAPI dependency at:
#              POST /api/v1/webhooks/razorpay
#            It is the ONLY entry point for Razorpay webhook events.
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.payments.payment_service import PaymentService, WebhookProcessResult
from app.payments.razorpay_client import RazorpayClient
import json

logger = logging.getLogger(ServiceName.PAYMENTS)

# Header name Razorpay uses for the HMAC-SHA256 signature
_RAZORPAY_SIGNATURE_HEADER = "X-Razorpay-Signature"

# Razorpay event types this handler explicitly recognises
_KNOWN_EVENT_TYPES = frozenset({
    "payment.captured",
    "payment.failed",
    "payment.authorized",
    "order.paid",
    "refund.processed",
    "refund.failed",
    "subscription.activated",
    "subscription.charged",
    "subscription.cancelled",
})


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class WebhookHandlerResult:
    """
    Result of handling a single Razorpay webhook request.

    Attributes:
        accepted:         True if the signature was valid and the request
                          was accepted for processing.
        http_status_code: HTTP status code to return to Razorpay.
        event_type:       Parsed event type string (if available).
        event_id:         Razorpay event ID for traceability.
        process_result:   Result from payment_service (if processing ran).
        rejection_reason: Human-readable reason if accepted=False.
    """
    accepted: bool
    http_status_code: int
    event_type: Optional[str] = None
    event_id: Optional[str] = None
    process_result: Optional[WebhookProcessResult] = None
    rejection_reason: Optional[str] = None

    @property
    def should_return_200(self) -> bool:
        """
        True if Razorpay should receive a 200 response.

        Returns 200 for:
          - Successfully processed events
          - Events processed idempotently (already seen)
          - Events we don't handle (unrecognised type) but signature was valid
          - Internal processing errors (to prevent infinite Razorpay retries)

        Returns non-200 only for:
          - Invalid or missing signature        → 400
          - Completely malformed JSON payload   → 400
        """
        return self.http_status_code == status.HTTP_200_OK


# ==============================================================================
# Webhook Handler
# ==============================================================================

class RazorpayWebhookHandler:
    """
    Authenticates and routes Razorpay webhook events.

    Instantiated once per application and shared across requests.
    All state is in injected dependencies — this class is stateless.

    Usage (called from webhook_route.py):

        handler = RazorpayWebhookHandler(
            razorpay_client=razorpay_client,
            payment_service=payment_service,
        )

        result = await handler.handle(request=request, db=db)

        return Response(
            status_code=result.http_status_code,
            content="OK" if result.accepted else result.rejection_reason,
        )
    """

    def __init__(
        self,
        razorpay_client: RazorpayClient,
        payment_service: PaymentService,
    ) -> None:
        self._razorpay = razorpay_client
        self._payment_service = payment_service

    async def handle(
        self,
        request: Request,
        db: AsyncSession,
    ) -> WebhookHandlerResult:
        """
        Process an incoming Razorpay webhook HTTP request.

        This is the single entry point called by webhook_route.py.
        All logic is contained here to keep the route thin.

        Steps:
            1. Read raw body bytes (before any parsing)
            2. Extract and validate the signature header
            3. Verify HMAC signature against raw body
            4. Parse JSON payload
            5. Extract event_id and event_type
            6. Delegate to payment_service

        Args:
            request: FastAPI Request object (raw body read here).
            db:      AsyncSession injected by FastAPI dependency.

        Returns:
            WebhookHandlerResult with the appropriate HTTP status code.
            Never raises — all exceptions are caught and logged.
        """
        log_extra = {
            "service": ServiceName.PAYMENTS,
            "remote_ip": _get_client_ip(request),
        }

        # ------------------------------------------------------------------
        # Step 1: Read raw body BEFORE any parsing
        # ------------------------------------------------------------------
        try:
            raw_body: bytes = await request.body()
        except Exception as exc:
            logger.error(
                "Webhook: failed to read request body",
                extra={**log_extra, "error": str(exc)},
            )
            return WebhookHandlerResult(
                accepted=False,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                rejection_reason="Failed to read request body",
            )

        if not raw_body:
            logger.warning("Webhook: empty request body", extra=log_extra)
            return WebhookHandlerResult(
                accepted=False,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                rejection_reason="Empty request body",
            )

        # ------------------------------------------------------------------
        # Step 2: Extract signature header
        # ------------------------------------------------------------------
        signature = request.headers.get(_RAZORPAY_SIGNATURE_HEADER, "").strip()

        if not signature:
            logger.warning(
                "Webhook: missing signature header",
                extra={**log_extra, "header": _RAZORPAY_SIGNATURE_HEADER},
            )
            return WebhookHandlerResult(
                accepted=False,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                rejection_reason="Missing signature header",
            )

        # ------------------------------------------------------------------
        # Step 3: Verify HMAC signature against RAW body bytes
        # ------------------------------------------------------------------
        is_valid = self._razorpay.verify_webhook_signature(
            webhook_body=raw_body,
            webhook_signature=signature,
        )

        if not is_valid:
            logger.warning(
                "Webhook: signature verification FAILED — rejecting request",
                extra={
                    **log_extra,
                    "signature_prefix": signature[:8] + "...",
                },
            )
            return WebhookHandlerResult(
                accepted=False,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                rejection_reason="Invalid webhook signature",
            )

        # ------------------------------------------------------------------
        # Step 4: Parse JSON payload (only after signature verified)
        # ------------------------------------------------------------------
        try:
            import json
            payload: dict = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "Webhook: JSON parse failed after valid signature",
                extra={**log_extra, "error": str(exc)},
            )
            return WebhookHandlerResult(
                accepted=False,
                http_status_code=status.HTTP_400_BAD_REQUEST,
                rejection_reason="Malformed JSON payload",
            )

        # ------------------------------------------------------------------
        # Step 5: Extract event metadata
        # ------------------------------------------------------------------
        event_type: str = payload.get("event", "unknown")
        event_id: str = payload.get("id", "")       # Razorpay event UUID
        account_id: str = payload.get("account_id", "")

        log_extra = {
            **log_extra,
            "event_type": event_type,
            "event_id": event_id,
            "account_id": account_id,
        }

        logger.info("Webhook received and authenticated", extra=log_extra)

        # Log unknown event types at debug — do not fail them
        if event_type not in _KNOWN_EVENT_TYPES:
            logger.debug(
                "Webhook: unrecognised event type — acknowledged without processing",
                extra=log_extra,
            )
            return WebhookHandlerResult(
                accepted=True,
                http_status_code=status.HTTP_200_OK,
                event_type=event_type,
                event_id=event_id,
            )

        # ------------------------------------------------------------------
        # Step 6: Delegate to payment_service
        # ------------------------------------------------------------------
        try:
            process_result = await self._payment_service.process_webhook_event(
                db=db,
                event_type=event_type,
                event_payload=payload,
            )

            if process_result.already_processed:
                logger.info(
                    "Webhook: event already processed — acknowledged",
                    extra={**log_extra, "payment_id": process_result.payment_id},
                )
            elif not process_result.success:
                # Processing failed internally — still return 200 so Razorpay
                # does not retry. The failure is logged for manual investigation.
                logger.error(
                    "Webhook: event processing failed — returning 200 to prevent retry",
                    extra={
                        **log_extra,
                        "error": process_result.error,
                        "payment_id": process_result.payment_id,
                    },
                )
            else:
                logger.info(
                    "Webhook: event processed successfully",
                    extra={
                        **log_extra,
                        "payment_id": process_result.payment_id,
                        "subscription_updated": process_result.subscription_updated,
                    },
                )

            return WebhookHandlerResult(
                accepted=True,
                http_status_code=status.HTTP_200_OK,
                event_type=event_type,
                event_id=event_id,
                process_result=process_result,
            )

        except Exception as exc:
            # Catch-all: processing raised unexpectedly.
            # Return 200 to prevent Razorpay retries — log at ERROR for
            # manual investigation.
            logger.error(
                "Webhook: unhandled exception during event processing",
                extra={**log_extra, "error": str(exc), "error_type": type(exc).__name__},
            )
            return WebhookHandlerResult(
                accepted=True,
                http_status_code=status.HTTP_200_OK,
                event_type=event_type,
                event_id=event_id,
                rejection_reason=None,   # accepted=True — we return 200
            )


# ==============================================================================
# Private helpers
# ==============================================================================

def _get_client_ip(request: Request) -> str:
    """
    Extract the client IP address from the request.

    Checks X-Forwarded-For first (set by load balancers / reverse proxies)
    then falls back to the direct client host.

    Returns the IP address string, or "unknown" if unavailable.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        # X-Forwarded-For may contain a comma-separated list — take the first
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"