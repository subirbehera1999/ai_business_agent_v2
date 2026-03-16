# ==============================================================================
# File: app/api/routes/payment_route.py
# Purpose: Razorpay payment initiation and status check endpoints.
#
#          Endpoints:
#
#          POST /api/v1/payments/initiate
#          ──────────────────────────────
#          Initiates a Razorpay payment order for subscription purchase.
#          Returns the Razorpay order_id and publishable key_id for the
#          frontend checkout widget.
#
#          The business_id is extracted from the authenticated JWT — a
#          business cannot initiate payment on behalf of another account.
#          The billing_cycle (monthly / annual) comes from the request body.
#
#          Flow:
#            1. Validate billing_cycle via Pydantic (PaymentInitiationRequest)
#            2. Build PaymentInitRequest from JWT business_id + payload
#            3. Call payment_service.initiate_payment()
#            4. On success: return Razorpay order details for checkout widget
#            5. On failure: return 400/500 with safe error message
#
#          GET /api/v1/payments/status/{razorpay_order_id}
#          ─────────────────────────────────────────────────
#          Poll the payment status for a specific Razorpay order.
#          Used by the frontend after checkout to confirm payment outcome.
#          Scoped by the authenticated business_id for multi-tenant safety.
#
#          GET /api/v1/payments/subscription
#          ──────────────────────────────────
#          Returns the current active subscription details for the
#          authenticated business. Returns 404 if no active subscription.
#
#          Auth:
#            All three endpoints require Bearer JWT via require_auth.
#            Listed in _PUBLIC_PATH_PREFIXES: /api/v1/payments — the
#            prefix match means /initiate is also considered public by
#            the middleware. However, require_auth is explicitly applied
#            here as a dependency to enforce authentication at the route
#            level regardless of middleware config.
#
#          Security rule (enforced here):
#            business_id ALWAYS comes from the JWT (require_auth).
#            It is NEVER accepted from the request body or query params.
#            This prevents a business from paying as another business.
#
#          Important note on PlanName:
#            This system uses ONE subscription tier only (one price, full
#            access). There is no plan selection. The payment_service
#            internally uses a hardcoded plan constant. The API only
#            accepts billing_cycle (monthly / annual).
# ==============================================================================

import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.db import get_db_session
from app.payments.payment_service import PaymentInitRequest, PaymentService
from app.repositories.subscription_repository import SubscriptionRepository
from app.security.auth_middleware import require_auth
from app.validators.input_validator import PaymentInitiationRequest

logger = logging.getLogger(ServiceName.API)

router = APIRouter(
    prefix="/api/v1/payments",
    tags=["Payments"],
)

# Module-level singletons — stateless, safe to share across requests
_payment_service = PaymentService()
_subscription_repo = SubscriptionRepository()

# Internal plan constant — one tier only, full access on any paid subscription
_PLATFORM_PLAN = "platform"


# ==============================================================================
# POST /api/v1/payments/initiate  — protected
# ==============================================================================

@router.post(
    "/initiate",
    summary="Initiate a Razorpay payment order",
    status_code=status.HTTP_201_CREATED,
)
async def initiate_payment(
    payload: PaymentInitiationRequest,
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Initiate a Razorpay subscription payment for the authenticated business.

    Creates a Razorpay order and persists a PENDING payment record.
    Returns the order_id and publishable key_id for the frontend checkout
    widget (Razorpay.js or React Native SDK).

    The business_id is extracted from the authenticated JWT — it is never
    accepted from the request body. The billing_cycle (monthly / annual)
    determines pricing and subscription duration.

    Idempotency: If a PENDING order already exists for the same business
    and billing cycle, the existing order details are returned without
    creating a duplicate Razorpay order.

    Args:
        payload:  Validated PaymentInitiationRequest (billing_cycle only).
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        201 Created with Razorpay order details on success.
        400 Bad Request if the payment service rejects the request.
        500 on unexpected server error.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "POST /api/v1/payments/initiate",
        "business_id": str(business.id),
        "billing_cycle": payload.billing_cycle,
    }

    logger.info("Payment initiation requested", extra=log_extra)

    # Build the internal request — business_id from JWT, not from body
    init_request = PaymentInitRequest(
        business_id=str(business.id),
        billing_cycle=payload.billing_cycle,
    )

    result = await _payment_service.initiate_payment(db, init_request)

    if not result.success:
        logger.warning(
            "Payment initiation failed",
            extra={**log_extra, "error": result.error},
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": result.error or "Payment initiation failed.",
                "data": None,
            },
        )

    logger.info(
        "Payment order created",
        extra={
            **log_extra,
            "razorpay_order_id": result.razorpay_order_id,
            "payment_record_id": result.payment_record_id,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "ok",
            "message": "Payment order created. Complete checkout using the details below.",
            "data": {
                "razorpay_order_id":  result.razorpay_order_id,
                "amount_paise":       result.amount_paise,
                "currency":           result.currency,
                "key_id":             result.key_id,
                "payment_record_id":  result.payment_record_id,
                "billing_cycle":      payload.billing_cycle,
            },
        },
    )


# ==============================================================================
# GET /api/v1/payments/status/{razorpay_order_id}  — protected
# ==============================================================================

@router.get(
    "/status/{razorpay_order_id}",
    summary="Poll payment status for a Razorpay order",
    status_code=status.HTTP_200_OK,
)
async def get_payment_status(
    razorpay_order_id: str,
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Return the current payment status for a specific Razorpay order.

    Used by the frontend after the Razorpay checkout widget closes to
    confirm whether payment was captured, failed, or is still pending.

    The query is always scoped by the authenticated business_id — a
    business cannot look up payment records belonging to another account.

    Args:
        razorpay_order_id: Razorpay order ID from the initiate response.
        business:          BusinessModel from require_auth dependency.
        db:                Request-scoped async database session.

    Returns:
        200 OK with payment status and details.
        404 if no matching payment record is found for this business.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": f"GET /api/v1/payments/status/{razorpay_order_id}",
        "business_id": str(business.id),
        "razorpay_order_id": razorpay_order_id,
    }

    if not razorpay_order_id or not razorpay_order_id.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "razorpay_order_id is required.",
                "data": None,
            },
        )

    payment = await _payment_service.get_payment_status(
        db=db,
        business_id=str(business.id),
        razorpay_order_id=razorpay_order_id.strip(),
    )

    if not payment:
        logger.info(
            "Payment status check — record not found",
            extra=log_extra,
        )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "status": "error",
                "message": "Payment record not found.",
                "data": None,
            },
        )

    logger.info(
        "Payment status retrieved",
        extra={**log_extra, "payment_status": payment.status},
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Payment status retrieved.",
            "data": _serialise_payment(payment),
        },
    )


# ==============================================================================
# GET /api/v1/payments/subscription  — protected
# ==============================================================================

@router.get(
    "/subscription",
    summary="Get current subscription status",
    status_code=status.HTTP_200_OK,
)
async def get_subscription_status(
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Return the current active subscription details for the authenticated business.

    Returns the active or trial subscription if one exists. Returns 404
    if the business has no active subscription — this indicates payment
    has not been completed or the subscription has expired.

    Args:
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        200 OK with subscription details.
        404 if no active subscription exists.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "GET /api/v1/payments/subscription",
        "business_id": str(business.id),
    }

    subscription = await _subscription_repo.get_active_by_business_id(
        db=db,
        business_id=business.id,
    )

    if not subscription:
        logger.info(
            "Subscription status check — no active subscription",
            extra=log_extra,
        )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "status": "error",
                "message": "No active subscription found. "
                           "Please complete payment to activate your account.",
                "data": None,
            },
        )

    logger.info(
        "Subscription status retrieved",
        extra={
            **log_extra,
            "subscription_id": str(subscription.id),
            "sub_status": subscription.status,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Subscription details retrieved.",
            "data": _serialise_subscription(subscription),
        },
    )


# ==============================================================================
# Serialisation helpers
# ==============================================================================

def _serialise_payment(payment) -> dict:
    """
    Serialise a PaymentModel to a safe API response dict.

    Excludes the raw Razorpay signature — it is an internal audit field
    and must never be exposed to the client.

    Args:
        payment: PaymentModel instance.

    Returns:
        dict safe for JSON serialisation.
    """
    return {
        "payment_record_id":    str(payment.id),
        "razorpay_order_id":    payment.razorpay_order_id,
        "razorpay_payment_id":  payment.razorpay_payment_id,
        "amount":               float(payment.amount) if payment.amount else None,
        "currency":             payment.currency,
        "status":               payment.status,
        "billing_cycle_months": payment.billing_cycle_months,
        "paid_at":              payment.paid_at.isoformat() if payment.paid_at else None,
        "created_at":           payment.created_at.isoformat(),
    }


def _serialise_subscription(subscription) -> dict:
    """
    Serialise a SubscriptionModel to a safe API response dict.

    Excludes Razorpay internal IDs and per-business override limits
    which are internal system fields not relevant to the client.

    Args:
        subscription: SubscriptionModel instance.

    Returns:
        dict safe for JSON serialisation.
    """
    billing_label = (
        "annual" if subscription.billing_cycle_months == 12 else "monthly"
    )

    return {
        "subscription_id":     str(subscription.id),
        "status":              subscription.status,
        "billing_cycle":       billing_label,
        "billing_cycle_months": subscription.billing_cycle_months,
        "amount":              float(subscription.amount) if subscription.amount else None,
        "currency":            subscription.currency,
        "starts_at":           subscription.starts_at.isoformat() if subscription.starts_at else None,
        "expires_at":          subscription.expires_at.isoformat() if subscription.expires_at else None,
        "trial_ends_at":       subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
        "created_at":          subscription.created_at.isoformat(),
    }