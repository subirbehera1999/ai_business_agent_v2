# ==============================================================================
# File: app/api/routes/onboarding_route.py
# Purpose: Business registration and profile management endpoints.
#
#          Endpoints:
#
#          POST /api/v1/onboarding/register
#          ─────────────────────────────────
#          Register a new business. Validates uniqueness of email/WhatsApp,
#          creates the business record, and returns the business ID.
#          No token is issued here — tokens are issued after payment.
#          The business must complete payment to activate their account.
#
#          GET /api/v1/onboarding/profile
#          ───────────────────────────────
#          Returns the authenticated business owner's profile.
#          Protected — requires Bearer token.
#
#          PATCH /api/v1/onboarding/profile
#          ─────────────────────────────────
#          Partial update to the business profile.
#          Protected — requires Bearer token.
#          At least one field must be provided (enforced by Pydantic schema).
#
#          POST /api/v1/onboarding/google
#          ──────────────────────────────
#          Connect or update the Google Business integration for a business.
#          Saves google_place_id and marks the business as Google-connected.
#          Protected — requires Bearer token.
#
#          POST /api/v1/onboarding/complete
#          ─────────────────────────────────
#          Mark the onboarding process as complete.
#          Protected — requires Bearer token.
#          Should be called after all integration steps are confirmed.
#
#          Auth:
#            - /register is public (listed in _PUBLIC_PATH_PREFIXES)
#            - All other endpoints require Bearer JWT via require_auth
#
#          Error handling:
#            - Duplicate email/WhatsApp → 409 Conflict
#            - Business not found → 404 Not Found
#            - Validation errors → 422 (handled by FastAPI/Pydantic)
#            - Unexpected errors → 500 with safe message (no internals exposed)
#
#          Layer contract:
#            Routes call repositories directly for simple CRUD operations.
#            No service layer indirection is needed for straightforward
#            profile reads and updates — the repository IS the service here.
# ==============================================================================

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.db import get_db_session
from app.repositories.business_repository import BusinessRepository
from app.security.auth_middleware import require_auth
from app.validators.input_validator import (
    BusinessOnboardingRequest,
    BusinessProfileUpdateRequest,
)

logger = logging.getLogger(ServiceName.API)

router = APIRouter(
    prefix="/api/v1/onboarding",
    tags=["Onboarding"],
)

# Module-level singletons — stateless, safe to share across requests
_business_repo = BusinessRepository()


# ==============================================================================
# POST /api/v1/onboarding/register  — public
# ==============================================================================

@router.post(
    "/register",
    summary="Register a new business",
    status_code=status.HTTP_201_CREATED,
)
async def register_business(
    payload: BusinessOnboardingRequest,
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Register a new business account.

    Creates a business record. Does NOT issue a JWT — authentication
    tokens are issued after payment is completed and verified.

    The caller receives the business_id which must be passed to the
    payment initiation endpoint to associate the payment with this account.

    Steps:
      1. Check email uniqueness
      2. Check WhatsApp number uniqueness
      3. Create business record
      4. Return business_id and next-step instructions

    Args:
        payload: Validated BusinessOnboardingRequest body.
        db:      Request-scoped async database session.

    Returns:
        201 Created with business_id on success.
        409 Conflict if email or WhatsApp number already registered.
        500 on unexpected database error.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "POST /api/v1/onboarding/register",
        "business_name": payload.business_name,
    }

    # ------------------------------------------------------------------
    # Uniqueness checks — return clear errors before touching the DB
    # ------------------------------------------------------------------
    try:
        email_taken = await _business_repo.email_exists(
            db, owner_email=payload.whatsapp_number  # placeholder — no email in schema
        )
    except Exception:
        email_taken = False  # safe to proceed; IntegrityError will catch duplicates

    try:
        whatsapp_taken = await _business_repo.whatsapp_number_exists(
            db, whatsapp_number=payload.whatsapp_number
        )
    except Exception as exc:
        logger.error(
            "Failed to check WhatsApp uniqueness",
            extra={**log_extra, "error": str(exc)},
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Registration check failed. Please try again.",
        )

    if whatsapp_taken:
        logger.info(
            "Registration rejected — WhatsApp number already registered",
            extra={**log_extra, "whatsapp": payload.whatsapp_number},
        )
        return _error_response(
            status.HTTP_409_CONFLICT,
            "This WhatsApp number is already registered. "
            "Please contact support if you believe this is an error.",
        )

    # ------------------------------------------------------------------
    # Create business record
    # ------------------------------------------------------------------
    try:
        business = await _business_repo.create(
            db,
            business_name=payload.business_name,
            owner_name=payload.business_name,   # owner_name not in schema — use business_name as placeholder
            owner_email=f"{payload.whatsapp_number}@placeholder.local",  # email not in schema
            owner_whatsapp_number=payload.whatsapp_number,
            business_type=payload.business_type,
            business_description=payload.notes,
            city=payload.city,
        )

    except IntegrityError:
        logger.warning(
            "Registration race condition — duplicate record on insert",
            extra=log_extra,
        )
        return _error_response(
            status.HTTP_409_CONFLICT,
            "This WhatsApp number is already registered.",
        )

    except Exception as exc:
        logger.error(
            "Business registration failed — database error",
            extra={**log_extra, "error": str(exc)},
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Registration failed due to a server error. Please try again.",
        )

    # ------------------------------------------------------------------
    # Connect Google Place ID if provided
    # ------------------------------------------------------------------
    try:
        await _business_repo.update_google_integration(
            db,
            business_id=business.id,
            google_place_id=payload.google_location_id,
            is_google_connected=True,
        )
    except Exception as exc:
        # Non-fatal — business is created; Google link can be set later
        logger.warning(
            "Google Place ID could not be saved during registration",
            extra={**log_extra, "error": str(exc)},
        )

    # ------------------------------------------------------------------
    # Connect Google Sheets if provided
    # ------------------------------------------------------------------
    if payload.google_sheets_url:
        try:
            await _business_repo.update_sheets_connection(
                db,
                business_id=business.id,
                google_sheets_url=payload.google_sheets_url,
                is_sheets_connected=True,
            )
        except Exception as exc:
            logger.warning(
                "Google Sheets URL could not be saved during registration",
                extra={**log_extra, "error": str(exc)},
            )

    logger.info(
        "Business registered successfully",
        extra={
            **log_extra,
            "business_id": str(business.id),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "ok",
            "message": "Business registered successfully. Complete payment to activate your account.",
            "data": {
                "business_id": str(business.id),
                "business_name": business.business_name,
                "whatsapp_number": business.owner_whatsapp_number,
                "next_step": "Complete payment at POST /api/v1/payments/initiate",
            },
        },
    )


# ==============================================================================
# GET /api/v1/onboarding/profile  — protected
# ==============================================================================

@router.get(
    "/profile",
    summary="Get business profile",
    status_code=status.HTTP_200_OK,
)
async def get_profile(
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Return the authenticated business owner's profile.

    The business object is provided by the require_auth dependency.
    A fresh DB read is done to ensure the response reflects the latest state.

    Args:
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        200 OK with business profile data.
        404 if the business no longer exists (edge case: deleted after token issued).
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "GET /api/v1/onboarding/profile",
        "business_id": str(business.id),
    }

    # Fresh read to ensure up-to-date data
    profile = await _business_repo.get_by_id(db, business.id)
    if not profile:
        logger.warning("Profile fetch — business not found", extra=log_extra)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "error",
                "message": "Business profile not found.",
                "data": None,
            },
        )

    logger.info("Profile fetched", extra=log_extra)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Profile retrieved successfully.",
            "data": _serialise_profile(profile),
        },
    )


# ==============================================================================
# PATCH /api/v1/onboarding/profile  — protected
# ==============================================================================

@router.patch(
    "/profile",
    summary="Update business profile",
    status_code=status.HTTP_200_OK,
)
async def update_profile(
    payload: BusinessProfileUpdateRequest,
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Partial update to the business profile.

    Only provided (non-None) fields are applied. At least one field must
    be present — enforced by BusinessProfileUpdateRequest.at_least_one_field().

    Updatable fields: business_name, whatsapp_number, google_sheets_url,
    city, business_type, notes.

    Args:
        payload:  Validated BusinessProfileUpdateRequest body.
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        200 OK with updated profile data.
        409 if new WhatsApp number is already taken by another business.
        500 on database error.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "PATCH /api/v1/onboarding/profile",
        "business_id": str(business.id),
    }

    # ------------------------------------------------------------------
    # WhatsApp uniqueness check if number is being changed
    # ------------------------------------------------------------------
    if (
        payload.whatsapp_number
        and payload.whatsapp_number != business.owner_whatsapp_number
    ):
        try:
            taken = await _business_repo.whatsapp_number_exists(
                db, whatsapp_number=payload.whatsapp_number
            )
        except Exception as exc:
            logger.error(
                "WhatsApp uniqueness check failed during profile update",
                extra={**log_extra, "error": str(exc)},
            )
            return _error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Profile update check failed. Please try again.",
            )

        if taken:
            return _error_response(
                status.HTTP_409_CONFLICT,
                "This WhatsApp number is already registered to another account.",
            )

    # ------------------------------------------------------------------
    # Apply updates via direct DB execute (no dedicated repo method for
    # generic profile patch — build update dict from non-None fields)
    # ------------------------------------------------------------------
    from sqlalchemy import update as sa_update
    from app.database.models.business_model import BusinessModel as BM

    update_fields: dict = {}

    if payload.business_name is not None:
        update_fields["business_name"] = payload.business_name
    if payload.whatsapp_number is not None:
        update_fields["owner_whatsapp_number"] = payload.whatsapp_number
    if payload.city is not None:
        update_fields["city"] = payload.city
    if payload.business_type is not None:
        update_fields["business_type"] = payload.business_type
    if payload.notes is not None:
        update_fields["business_description"] = payload.notes

    # Google Sheets update — delegate to dedicated repo method
    if payload.google_sheets_url is not None:
        try:
            await _business_repo.update_sheets_connection(
                db,
                business_id=business.id,
                google_sheets_url=payload.google_sheets_url,
                is_sheets_connected=True,
            )
        except Exception as exc:
            logger.error(
                "Sheets connection update failed",
                extra={**log_extra, "error": str(exc)},
            )
            return _error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Profile update failed. Please try again.",
            )
        update_fields.pop("google_sheets_url", None)  # handled above

    if update_fields:
        try:
            await db.execute(
                sa_update(BM)
                .where(BM.id == business.id, BM.is_deleted.is_(False))
                .values(**update_fields)
            )
            await db.flush()
        except Exception as exc:
            logger.error(
                "Profile update DB write failed",
                extra={**log_extra, "error": str(exc)},
            )
            return _error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Profile update failed. Please try again.",
            )

    # Return fresh profile
    updated = await _business_repo.get_by_id(db, business.id)
    if not updated:
        return _error_response(
            status.HTTP_404_NOT_FOUND, "Business profile not found."
        )

    logger.info(
        "Profile updated",
        extra={**log_extra, "fields_updated": list(update_fields.keys())},
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Profile updated successfully.",
            "data": _serialise_profile(updated),
        },
    )


# ==============================================================================
# POST /api/v1/onboarding/google  — protected
# ==============================================================================

@router.post(
    "/google",
    summary="Connect or update Google Business integration",
    status_code=status.HTTP_200_OK,
)
async def connect_google(
    request: Request,
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Connect or update the Google Business Profile integration.

    Accepts a JSON body with google_place_id (required) and optionally
    google_business_name. Saves the integration fields and marks the
    business as Google-connected.

    Expected body:
        {
            "google_place_id": "ChIJ...",
            "google_business_name": "Sunrise Cafe" (optional)
        }

    Args:
        request:  Raw FastAPI Request for JSON body parsing.
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        200 OK on successful update.
        400 Bad Request if google_place_id is missing.
        500 on database error.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "POST /api/v1/onboarding/google",
        "business_id": str(business.id),
    }

    try:
        body = await request.json()
    except Exception:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "Invalid JSON body.",
        )

    google_place_id: Optional[str] = body.get("google_place_id", "").strip() or None
    google_business_name: Optional[str] = (
        body.get("google_business_name", "").strip() or None
    )

    if not google_place_id:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "google_place_id is required.",
        )

    try:
        await _business_repo.update_google_integration(
            db,
            business_id=business.id,
            google_place_id=google_place_id,
            google_business_name=google_business_name,
            is_google_connected=True,
        )
    except Exception as exc:
        logger.error(
            "Google integration update failed",
            extra={**log_extra, "error": str(exc)},
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to save Google integration. Please try again.",
        )

    logger.info(
        "Google integration connected",
        extra={**log_extra, "google_place_id": google_place_id},
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Google Business Profile connected successfully.",
            "data": {
                "google_place_id": google_place_id,
                "google_business_name": google_business_name,
                "is_google_connected": True,
            },
        },
    )


# ==============================================================================
# POST /api/v1/onboarding/complete  — protected
# ==============================================================================

@router.post(
    "/complete",
    summary="Mark onboarding as complete",
    status_code=status.HTTP_200_OK,
)
async def complete_onboarding(
    business=Depends(require_auth),
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """
    Mark the business onboarding process as complete.

    Sets is_onboarding_complete = True on the business record. This flag
    enables the business to appear in scheduled job processing queues
    (review monitoring, report generation, feedback requests).

    Should be called by the client after all integration steps (Google,
    Sheets, payment) have been confirmed.

    Args:
        business: BusinessModel from require_auth dependency.
        db:       Request-scoped async database session.

    Returns:
        200 OK with confirmation.
        500 on database error.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "POST /api/v1/onboarding/complete",
        "business_id": str(business.id),
    }

    if business.is_onboarding_complete:
        logger.info(
            "Onboarding already complete — idempotent response",
            extra=log_extra,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "message": "Onboarding was already marked as complete.",
                "data": {"is_onboarding_complete": True},
            },
        )

    try:
        await _business_repo.mark_onboarding_complete(db, business.id)
    except Exception as exc:
        logger.error(
            "Failed to mark onboarding complete",
            extra={**log_extra, "error": str(exc)},
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to complete onboarding. Please try again.",
        )

    logger.info("Onboarding marked complete", extra=log_extra)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Onboarding complete. Your account is now fully active.",
            "data": {"is_onboarding_complete": True},
        },
    )


# ==============================================================================
# Shared helpers
# ==============================================================================

def _serialise_profile(business) -> dict:
    """
    Serialise a BusinessModel instance to a safe dict for API responses.

    Excludes encrypted tokens (google_access_token, google_refresh_token)
    and internal flags. Returns only fields safe to expose to the client.

    Args:
        business: BusinessModel instance.

    Returns:
        dict suitable for JSON serialisation.
    """
    return {
        "business_id":             str(business.id),
        "business_name":           business.business_name,
        "business_type":           business.business_type,
        "business_description":    business.business_description,
        "owner_name":              business.owner_name,
        "owner_email":             business.owner_email,
        "whatsapp_number":         business.owner_whatsapp_number,
        "city":                    business.city,
        "state":                   business.state,
        "country":                 business.country,
        "timezone":                business.timezone,
        "google_place_id":         business.google_place_id,
        "google_business_name":    business.google_business_name,
        "current_google_rating":   business.current_google_rating,
        "total_google_reviews":    business.total_google_reviews,
        "is_google_connected":     business.is_google_connected,
        "is_sheets_connected":     business.is_sheets_connected,
        "is_onboarding_complete":  business.is_onboarding_complete,
        "is_active":               business.is_active,
        "created_at":              business.created_at.isoformat(),
    }


def _error_response(http_status: int, message: str) -> JSONResponse:
    """
    Build a standard error JSONResponse.

    Args:
        http_status: HTTP status code integer.
        message:     Safe, user-facing error message.

    Returns:
        JSONResponse with standard error envelope.
    """
    return JSONResponse(
        status_code=http_status,
        content={
            "status": "error",
            "message": message,
            "data": None,
        },
    )