# ==============================================================================
# File: app/api/routes/webhook_route.py
# Purpose: Razorpay webhook receiver endpoint.
#
#          This route accepts POST requests from Razorpay's webhook delivery
#          system and delegates all processing to RazorpayWebhookHandler.
#
#          SECURITY — Critical rules enforced here:
#            1. Raw request body is read BEFORE any JSON parsing.
#               Razorpay computes its HMAC signature over the raw bytes.
#               Any re-serialisation would produce a different byte sequence
#               and break signature verification.
#            2. HMAC-SHA256 signature verification happens inside the handler
#               before any business logic executes.
#            3. This endpoint is intentionally UNAUTHENTICATED — it has no
#               JWT requirement because the caller is Razorpay's server,
#               not a business user. The webhook secret is the auth mechanism.
#            4. The endpoint ALWAYS returns HTTP 200 for valid signatures,
#               even when internal processing fails. This prevents Razorpay
#               from endlessly retrying events that will never succeed due to
#               business logic errors. Only signature failures return 400.
#
#          Idempotency:
#            Razorpay may deliver the same event more than once (retries on
#            timeout or non-2xx response). payment_service.py uses idempotency
#            keys to ensure each payment event is processed exactly once.
#
#          Event types handled (inside webhook_handler.py / payment_service.py):
#            - payment.captured   → activates or extends subscription
#            - payment.failed     → logs failure, notifies admin
#            - refund.processed   → logs refund, notifies admin
#            - subscription.*     → handled as informational events
#
#          Response contract:
#            200 OK       → signature valid, event accepted (may have failed
#                           internally — internal errors are swallowed to
#                           prevent Razorpay retry storms)
#            400 Bad Req  → missing or invalid signature, or malformed JSON
#            405 Method   → not POST (handled automatically by FastAPI)
#
#          Request logging:
#            Every incoming webhook is logged with its event_id and
#            event_type so that the admin can trace any payment event
#            through the system logs.
# ==============================================================================

import logging

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.db import get_db_session
from app.payments.razorpay_client import RazorpayClient
from app.payments.payment_service import PaymentService
from app.payments.webhook_handler import RazorpayWebhookHandler

logger = logging.getLogger(ServiceName.PAYMENTS)

router = APIRouter(
    prefix="/api/v1/webhooks",
    tags=["Webhooks"],
)

# ==============================================================================
# Module-level singletons
# ==============================================================================
# All dependencies are stateless — safe to instantiate once and reuse across
# requests. The handler itself is a thin routing shell; all state lives in
# the injected services and the database session.

_razorpay_client = RazorpayClient()
_payment_service = PaymentService()
_webhook_handler = RazorpayWebhookHandler(
    razorpay_client=_razorpay_client,
    payment_service=_payment_service,
)


# ==============================================================================
# Webhook endpoint
# ==============================================================================

@router.post(
    "/razorpay",
    status_code=status.HTTP_200_OK,
    summary="Razorpay webhook receiver",
    description=(
        "Receives and processes Razorpay payment webhook events. "
        "Verifies HMAC-SHA256 signature before any processing. "
        "Returns 200 for all valid-signature events (including internal errors) "
        "to prevent Razorpay retry storms. Returns 400 only for invalid "
        "or missing signatures."
    ),
    response_description="Empty body with appropriate HTTP status code.",
    include_in_schema=True,
)
async def razorpay_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    """
    Razorpay webhook POST receiver.

    This route is intentionally unauthenticated (no JWT required).
    Security is enforced exclusively through HMAC-SHA256 signature
    verification using the Razorpay webhook secret.

    The raw request body is passed to the handler without modification.
    Never parse JSON before signature verification — doing so invalidates
    the byte-level HMAC check.

    Args:
        request: Raw FastAPI Request. Body is read inside the handler.
        db:      Async database session injected by FastAPI.

    Returns:
        Response:
            - 200 OK  on valid signature (processing may have failed internally)
            - 400 Bad Request on invalid/missing signature or malformed JSON
    """
    log_extra = {
        "service": ServiceName.PAYMENTS,
        "remote_ip": _get_client_ip(request),
        "user_agent": request.headers.get("user-agent", "unknown"),
    }

    logger.info(
        "Razorpay webhook received",
        extra=log_extra,
    )

    # Delegate all processing — signature verification, parsing, routing —
    # to the handler. The route stays thin; no business logic here.
    result = await _webhook_handler.handle(request=request, db=db)

    # Log outcome for traceability
    if result.accepted:
        logger.info(
            "Webhook accepted",
            extra={
                **log_extra,
                "event_type": result.event_type,
                "event_id": result.event_id,
                "process_success": (
                    result.process_result.success
                    if result.process_result
                    else None
                ),
            },
        )
    else:
        logger.warning(
            "Webhook rejected",
            extra={
                **log_extra,
                "reason": result.rejection_reason,
                "http_status": result.http_status_code,
            },
        )

    # Return a minimal response — Razorpay only cares about the status code.
    # Body content is intentionally kept short for security (no internal detail
    # is exposed in rejection messages).
    return Response(
        status_code=result.http_status_code,
        content="OK" if result.accepted else result.rejection_reason or "rejected",
        media_type="text/plain",
    )


# ==============================================================================
# Health probe for webhook endpoint (used by Razorpay dashboard connectivity
# test and internal monitoring)
# ==============================================================================

@router.get(
    "/razorpay/ping",
    status_code=status.HTTP_200_OK,
    summary="Webhook endpoint health probe",
    description=(
        "Lightweight GET probe confirming the webhook endpoint is reachable. "
        "Used by Razorpay dashboard connectivity tests and internal monitoring. "
        "Returns a static JSON response — no database access."
    ),
    include_in_schema=True,
)
async def razorpay_webhook_ping() -> dict:
    """
    Static health probe for the webhook receiver endpoint.

    Razorpay's dashboard connectivity test sends a GET request to the
    configured webhook URL. This endpoint satisfies that probe without
    requiring database access or authentication.

    Returns:
        dict: Static status confirmation payload.
    """
    return {
        "status": "ok",
        "endpoint": "razorpay_webhook",
        "message": "Webhook receiver is reachable.",
    }


# ==============================================================================
# Private helpers
# ==============================================================================

def _get_client_ip(request: Request) -> str:
    """
    Extract the originating client IP address from the request.

    Checks X-Forwarded-For first (set by reverse proxies / load balancers)
    then falls back to the direct connection IP.

    Args:
        request: FastAPI Request object.

    Returns:
        str: Best-effort client IP address string.
    """
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        # X-Forwarded-For may contain a comma-separated list; leftmost is origin
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"