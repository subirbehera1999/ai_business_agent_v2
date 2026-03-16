# ==============================================================================
# File: app/validators/input_validator.py
# Purpose: Validates all incoming API request data before it reaches
#          the service layer.
#
#          This file defines Pydantic request models (schemas) for every
#          API endpoint in the system, plus standalone validation helpers
#          used by route handlers.
#
#          Why validate at the API boundary?
#            The service layer assumes data is already clean and typed.
#            If a route handler passes raw unvalidated request data to a
#            service, the service might:
#              - Crash on unexpected types (None where str expected)
#              - Store malformed phone numbers in the database
#              - Pass invalid business IDs to AI prompts
#              - Accept SQL injection attempts in text fields
#
#            Pydantic validates and coerces types at the boundary so
#            services always receive clean, typed Python objects.
#
#          Request models defined here:
#            1. BusinessOnboardingRequest  — register a new business
#            2. PaymentInitiationRequest   — start a Razorpay payment
#            3. CompetitorRegistrationRequest — add a competitor to track
#            4. BusinessProfileUpdateRequest — partial profile update
#            5. TokenRefreshRequest        — exchange refresh token
#
#          Standalone helpers:
#            validate_uuid()        — validate UUID path parameters
#            validate_phone()       — validate and normalise phone numbers
#            validate_pagination()  — validate page/limit query params
#            validate_date_range()  — validate date range query params
# ==============================================================================

import logging
import re
import uuid
from datetime import date as dt_date, timedelta
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.constants import ServiceName

logger = logging.getLogger(ServiceName.INPUT_VALIDATOR)

# ---------------------------------------------------------------------------
# Field constraints
# ---------------------------------------------------------------------------
MIN_BUSINESS_NAME_LEN: int = 2
MAX_BUSINESS_NAME_LEN: int = 100
MAX_CITY_LEN: int = 100
MAX_COMPETITOR_NAME_LEN: int = 100
MAX_NOTES_LEN: int = 500

# Indian mobile number: 10 digits, starts with 6-9
_MOBILE_DIGITS_RE = re.compile(r"^[6-9]\d{9}$")

# Google Sheets URL pattern
_SHEETS_URL_RE = re.compile(
    r"^https://docs\.google\.com/spreadsheets/d/[\w-]+",
    re.IGNORECASE,
)

# Google location ID: alphanumeric, 8-100 chars
_LOCATION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,100}$")

# UUID pattern
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# HTML tag stripper for XSS protection
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Allowed business types
ALLOWED_BUSINESS_TYPES = frozenset({
    "restaurant", "cafe", "clinic", "salon", "spa",
    "gym", "retail_store", "pharmacy", "hotel", "bakery", "other",
})

# Allowed billing cycles
ALLOWED_BILLING_CYCLES = frozenset({"monthly", "annual"})

# Pagination limits
MAX_PAGE_LIMIT: int = 100
DEFAULT_PAGE_LIMIT: int = 20


# ==============================================================================
# Shared field helpers
# ==============================================================================

def _strip_html(value: str) -> str:
    """Remove HTML tags from a string. Basic XSS protection."""
    return _HTML_TAG_RE.sub("", value).strip()


def _normalise_phone(raw: str) -> str:
    """
    Normalise a phone number to E.164 format (+91XXXXXXXXXX).

    Accepts:
      "9876543210"        -> "+919876543210"
      "+919876543210"     -> "+919876543210"
      "09876543210"       -> "+919876543210"
      "+91 98765 43210"   -> "+919876543210"

    Raises ValueError if number is not a valid Indian mobile.
    """
    digits_only = re.sub(r"[^\d+]", "", raw.strip())

    if digits_only.startswith("0"):
        digits_only = digits_only[1:]

    if digits_only.startswith("+91"):
        digits_only = digits_only[3:]
    elif digits_only.startswith("91") and len(digits_only) == 12:
        digits_only = digits_only[2:]

    if not _MOBILE_DIGITS_RE.match(digits_only):
        raise ValueError(
            f"Invalid Indian mobile number. "
            f"Must be 10 digits starting with 6, 7, 8, or 9. Got: '{raw}'"
        )

    return f"+91{digits_only}"


# ==============================================================================
# 1. Business Onboarding Request
# ==============================================================================

class BusinessOnboardingRequest(BaseModel):
    """
    Request body for POST /api/v1/onboarding/register

    All required fields must be valid for onboarding to proceed.
    Optional fields provide richer AI context but are not mandatory.
    """

    business_name: str = Field(
        ...,
        min_length=MIN_BUSINESS_NAME_LEN,
        max_length=MAX_BUSINESS_NAME_LEN,
        description="Legal or trading name of the business",
        examples=["Sunrise Cafe", "Dr. Sharma Clinic"],
    )
    whatsapp_number: str = Field(
        ...,
        description="Business owner WhatsApp number (Indian mobile, 10 digits)",
        examples=["9876543210", "+919876543210"],
    )
    google_location_id: str = Field(
        ...,
        description="Google Business Profile location ID",
        examples=["ChIJN1t_tDeuEmsRUsoyG83frY4"],
    )
    business_type: str = Field(
        ...,
        description=f"Type of business. Allowed: {', '.join(sorted(ALLOWED_BUSINESS_TYPES))}",
        examples=["restaurant", "clinic", "salon"],
    )
    google_sheets_url: Optional[str] = Field(
        default=None,
        description="Google Sheets URL containing sales data (optional)",
        examples=["https://docs.google.com/spreadsheets/d/SHEET_ID/edit"],
    )
    city: Optional[str] = Field(
        default=None,
        max_length=MAX_CITY_LEN,
        description="City or area where the business operates (optional)",
        examples=["Mumbai", "Bhubaneswar"],
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=MAX_NOTES_LEN,
        description="Any additional context about the business (optional)",
    )

    @field_validator("business_name")
    @classmethod
    def validate_business_name(cls, v: str) -> str:
        cleaned = _strip_html(v)
        if len(cleaned) < MIN_BUSINESS_NAME_LEN:
            raise ValueError(
                f"Business name must be at least {MIN_BUSINESS_NAME_LEN} characters."
            )
        return cleaned

    @field_validator("whatsapp_number")
    @classmethod
    def validate_whatsapp_number(cls, v: str) -> str:
        return _normalise_phone(v)

    @field_validator("google_location_id")
    @classmethod
    def validate_google_location_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not _LOCATION_ID_RE.match(cleaned):
            raise ValueError(
                "Invalid Google location ID format. "
                "Should be the alphanumeric ID from your Google Business Profile URL."
            )
        return cleaned

    @field_validator("business_type")
    @classmethod
    def validate_business_type(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in ALLOWED_BUSINESS_TYPES:
            raise ValueError(
                f"Invalid business type '{v}'. "
                f"Allowed values: {', '.join(sorted(ALLOWED_BUSINESS_TYPES))}"
            )
        return normalised

    @field_validator("google_sheets_url")
    @classmethod
    def validate_google_sheets_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        if not _SHEETS_URL_RE.match(cleaned):
            raise ValueError(
                "Invalid Google Sheets URL. "
                "Must start with: https://docs.google.com/spreadsheets/d/"
            )
        return cleaned

    @field_validator("city")
    @classmethod
    def validate_city(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = _strip_html(v)
        return cleaned if cleaned else None

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _strip_html(v)[:MAX_NOTES_LEN]


# ==============================================================================
# 2. Payment Initiation Request
# ==============================================================================

class PaymentInitiationRequest(BaseModel):
    """
    Request body for POST /api/v1/payments/initiate

    Initiates a Razorpay payment order for subscription purchase.
    The business_id comes from the authenticated JWT — not from this
    body — so a business cannot pay on behalf of another.
    """

    billing_cycle: str = Field(
        ...,
        description="Billing cycle: 'monthly' or 'annual'",
        examples=["monthly", "annual"],
    )

    @field_validator("billing_cycle")
    @classmethod
    def validate_billing_cycle(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in ALLOWED_BILLING_CYCLES:
            raise ValueError(
                f"Invalid billing cycle '{v}'. Allowed values: monthly, annual"
            )
        return normalised


# ==============================================================================
# 3. Competitor Registration Request
# ==============================================================================

class CompetitorRegistrationRequest(BaseModel):
    """
    Request body for POST /api/v1/onboarding/competitors

    Registers a competitor for tracking by competitor_service.py.
    """

    competitor_name: str = Field(
        ...,
        min_length=2,
        max_length=MAX_COMPETITOR_NAME_LEN,
        description="Display name for the competitor",
        examples=["Blue Star Cafe", "City Dental Clinic"],
    )
    google_location_id: str = Field(
        ...,
        description="Competitor Google Business Profile location ID",
        examples=["ChIJN1t_tDeuEmsRUsoyG83frY4"],
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=MAX_NOTES_LEN,
        description="Optional notes about this competitor",
    )

    @field_validator("competitor_name")
    @classmethod
    def validate_competitor_name(cls, v: str) -> str:
        cleaned = _strip_html(v)
        if len(cleaned) < 2:
            raise ValueError("Competitor name must be at least 2 characters.")
        return cleaned

    @field_validator("google_location_id")
    @classmethod
    def validate_google_location_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not _LOCATION_ID_RE.match(cleaned):
            raise ValueError("Invalid Google location ID format for competitor.")
        return cleaned

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _strip_html(v)[:MAX_NOTES_LEN]


# ==============================================================================
# 4. Business Profile Update Request
# ==============================================================================

class BusinessProfileUpdateRequest(BaseModel):
    """
    Request body for PATCH /api/v1/onboarding/profile

    Partial update — all fields optional. Only provided fields are updated.
    At least one field must be present (enforced by model_validator).
    """

    business_name: Optional[str] = Field(
        default=None,
        min_length=MIN_BUSINESS_NAME_LEN,
        max_length=MAX_BUSINESS_NAME_LEN,
    )
    whatsapp_number: Optional[str] = Field(default=None)
    google_sheets_url: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None, max_length=MAX_CITY_LEN)
    business_type: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None, max_length=MAX_NOTES_LEN)

    @model_validator(mode="after")
    def at_least_one_field(self) -> "BusinessProfileUpdateRequest":
        provided = [
            f for f in (
                self.business_name, self.whatsapp_number,
                self.google_sheets_url, self.city,
                self.business_type, self.notes,
            )
            if f is not None
        ]
        if not provided:
            raise ValueError(
                "At least one field must be provided for a profile update."
            )
        return self

    @field_validator("business_name")
    @classmethod
    def validate_business_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = _strip_html(v)
        if len(cleaned) < MIN_BUSINESS_NAME_LEN:
            raise ValueError(
                f"Business name must be at least {MIN_BUSINESS_NAME_LEN} characters."
            )
        return cleaned

    @field_validator("whatsapp_number")
    @classmethod
    def validate_whatsapp_number(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _normalise_phone(v)

    @field_validator("google_sheets_url")
    @classmethod
    def validate_google_sheets_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        if not _SHEETS_URL_RE.match(cleaned):
            raise ValueError(
                "Invalid Google Sheets URL. "
                "Must start with: https://docs.google.com/spreadsheets/d/"
            )
        return cleaned

    @field_validator("business_type")
    @classmethod
    def validate_business_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalised = v.strip().lower()
        if normalised not in ALLOWED_BUSINESS_TYPES:
            raise ValueError(
                f"Invalid business type. "
                f"Allowed: {', '.join(sorted(ALLOWED_BUSINESS_TYPES))}"
            )
        return normalised

    @field_validator("city")
    @classmethod
    def validate_city(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _strip_html(v) or None

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _strip_html(v)[:MAX_NOTES_LEN]


# ==============================================================================
# 5. Token Refresh Request
# ==============================================================================

class TokenRefreshRequest(BaseModel):
    """
    Request body for POST /api/v1/auth/refresh

    Exchanges a valid refresh token for a new access + refresh token pair.
    """

    refresh_token: str = Field(
        ...,
        min_length=10,
        description="The refresh token issued at login or last refresh",
    )

    @field_validator("refresh_token")
    @classmethod
    def validate_refresh_token(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("refresh_token must not be empty.")
        parts = cleaned.split(".")
        if len(parts) != 3:
            raise ValueError(
                "Invalid token format. "
                "Expected a JWT with three dot-separated segments."
            )
        return cleaned


# ==============================================================================
# Standalone helpers — used by route handlers for path/query params
# ==============================================================================

def validate_uuid(value: str, field_name: str = "id") -> str:
    """
    Validate that a string is a valid UUID.

    Used for path parameters like /api/v1/businesses/{business_id}.

    Args:
        value:      The raw path parameter string.
        field_name: Field name used in error messages.

    Returns:
        Lowercase UUID string.

    Raises:
        ValueError: If value is not a valid UUID format.
    """
    cleaned = value.strip().lower()
    if not _UUID_RE.match(cleaned):
        raise ValueError(
            f"Invalid {field_name}: '{value}' is not a valid UUID. "
            f"Expected: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        )
    try:
        uuid.UUID(cleaned)
    except ValueError:
        raise ValueError(f"Invalid {field_name}: '{value}'.")
    return cleaned


def validate_phone(raw: str) -> str:
    """
    Validate and normalise an Indian mobile number to E.164 format.

    Standalone helper for use outside Pydantic models.

    Returns: "+91XXXXXXXXXX"
    Raises:  ValueError if not a valid Indian mobile number.
    """
    return _normalise_phone(raw)


def validate_pagination(page: int, limit: int) -> tuple[int, int]:
    """
    Validate and normalise pagination query parameters.

    Args:
        page:  Page number (1-based, minimum 1).
        limit: Records per page (1 to MAX_PAGE_LIMIT).

    Returns:
        Tuple (page, limit).

    Raises:
        ValueError: If page < 1, limit < 1, or limit > MAX_PAGE_LIMIT.
    """
    if page < 1:
        raise ValueError("page must be 1 or greater.")
    if limit < 1:
        raise ValueError("limit must be 1 or greater.")
    if limit > MAX_PAGE_LIMIT:
        raise ValueError(
            f"limit must not exceed {MAX_PAGE_LIMIT}. Got {limit}."
        )
    return page, limit


def validate_date_range(
    start_date: str,
    end_date: str,
    max_range_days: int = 365,
) -> tuple[dt_date, dt_date]:
    """
    Validate a date range string pair for analytics/report query params.

    Both dates must be in YYYY-MM-DD format.
    start_date must not be after end_date.
    Range must not exceed max_range_days.

    Args:
        start_date:     String in YYYY-MM-DD format.
        end_date:       String in YYYY-MM-DD format.
        max_range_days: Maximum allowed range in days (default 365).

    Returns:
        Tuple of (datetime.date, datetime.date).

    Raises:
        ValueError: If dates are invalid, in wrong order, or range too large.
    """
    try:
        start = dt_date.fromisoformat(start_date)
    except ValueError:
        raise ValueError(
            f"Invalid start_date '{start_date}'. Expected format: YYYY-MM-DD"
        )

    try:
        end = dt_date.fromisoformat(end_date)
    except ValueError:
        raise ValueError(
            f"Invalid end_date '{end_date}'. Expected format: YYYY-MM-DD"
        )

    if start > end:
        raise ValueError(
            f"start_date ({start_date}) must not be after end_date ({end_date})."
        )

    if (end - start).days > max_range_days:
        raise ValueError(
            f"Date range too large: {(end - start).days} days. "
            f"Maximum allowed is {max_range_days} days."
        )

    return start, end


# ==============================================================================
# Standard API validation error formatter
# ==============================================================================

def build_validation_error_response(errors: list) -> dict:
    """
    Convert Pydantic ValidationError details into the standard API error format.

    Used by FastAPI exception handlers to return consistent validation
    error responses across all endpoints.

    Args:
        errors: List of error dicts from ValidationError.errors()

    Returns:
        Standard error response:
          {
            "status": "error",
            "message": "Validation failed. Please check the fields below.",
            "data": {"field_name": "error message", ...}
          }
    """
    field_errors: dict[str, str] = {}

    for error in errors:
        loc = error.get("loc", ())
        # Skip leading "body" element from Pydantic location tuples
        field_parts = [str(p) for p in loc if p != "body"]
        field_name = ".".join(field_parts) if field_parts else "request"
        field_errors[field_name] = error.get("msg", "Invalid value")

    return {
        "status": "error",
        "message": "Validation failed. Please check the fields below.",
        "data": field_errors,
    }