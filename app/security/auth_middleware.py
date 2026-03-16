# ==============================================================================
# File: app/security/auth_middleware.py
# Purpose: FastAPI authentication middleware that protects all API endpoints.
#
#          Two components are defined here:
#
#          1. AuthMiddleware (Starlette BaseHTTPMiddleware)
#             Runs on EVERY incoming HTTP request before it reaches any route.
#             Responsibilities:
#               - Inject a unique request ID into every request/response
#               - Bind request ID to the logging context (logger.py)
#               - Log all incoming requests with timing
#               - Log all outgoing responses with status code and duration
#               - Catch unhandled exceptions and return safe JSON errors
#               - Record request timing for performance monitoring
#
#          2. require_auth (FastAPI dependency)
#             Used on individual protected routes as a dependency:
#               @router.get("/reviews")
#               async def get_reviews(business = Depends(require_auth)):
#             Responsibilities:
#               - Extract token from Authorization header
#               - Verify token via TokenManager
#               - Load the authenticated business from the database
#               - Return the business object to the route handler
#               - Raise HTTP 401 if token is missing, invalid, or expired
#               - Raise HTTP 403 if business has no active subscription
#
#          Public routes (no auth required):
#            The middleware does NOT block unauthenticated requests —
#            that is the job of the require_auth dependency on each route.
#            Public routes (health check, onboarding, webhook, payment
#            initiation) simply do not use the require_auth dependency.
#
#          Why middleware + dependency, not middleware alone?
#            Middleware runs on ALL requests including public ones.
#            If you enforce auth in middleware, public routes break.
#            The dependency pattern lets each route opt-in to auth
#            explicitly — no route accidentally skips auth.
#
#          Request ID:
#            Every request gets a UUID injected as X-Request-ID.
#            If the client sends X-Request-ID, that value is used instead.
#            This enables end-to-end request tracing across logs and
#            WhatsApp notifications.
#
#          Rate limiting at middleware level:
#            This middleware enforces a global IP-level rate limit to
#            protect against brute-force attacks on the auth endpoint.
#            Per-business rate limiting is handled by plan_manager.py.
#
#          Security headers:
#            Every response gets standard security headers added:
#              X-Content-Type-Options: nosniff
#              X-Frame-Options: DENY
#              X-XSS-Protection: 1; mode=block
#              Strict-Transport-Security: max-age=31536000
#              Cache-Control: no-store (for API responses)
# ==============================================================================

import logging
import time
import uuid
from typing import Callable, Optional

import redis as redis_lib

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config.constants import ServiceName
from app.logging.logger import RequestContext
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.security.token_manager import TokenManager
from app.database.db import get_db_session

logger = logging.getLogger(ServiceName.SECURITY)

# ---------------------------------------------------------------------------
# Public routes — these paths bypass auth requirement in require_auth
# Health, onboarding, payment initiation, and webhooks are public.
# Webhooks use their own signature verification (webhook_handler.py).
# ---------------------------------------------------------------------------
_PUBLIC_PATH_PREFIXES = (
    "/api/v1/health",
    "/api/v1/onboarding",
    "/api/v1/payments",
    "/api/v1/webhooks",
    "/docs",
    "/openapi.json",
    "/redoc",
)


# ---------------------------------------------------------------------------
# IP-level brute force protection
# Uses Redis for shared state across all Gunicorn workers.
# Falls back to in-memory if Redis is unavailable (development / no Redis).
# ---------------------------------------------------------------------------
_MAX_FAILED_ATTEMPTS: int = 10        # per IP per window
_FAILED_ATTEMPT_WINDOW_SECONDS: int = 300  # 5-minute window

# Redis client — initialised once at module load, None if unavailable
_redis_client: Optional[redis_lib.Redis] = None

def _get_redis() -> Optional[redis_lib.Redis]:
    """
    Return a Redis client, initialising it on first call.
    Returns None if Redis is not configured or unreachable.
    Failures are silent — falls back to in-memory protection.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        _redis_client = redis_lib.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,   # fast fail — don't block requests
            socket_timeout=1,
        )
        _redis_client.ping()            # verify connection works
        logger.info(
            "Redis connected for brute-force protection",
            extra={"service": ServiceName.SECURITY},
        )
        return _redis_client
    except Exception as exc:
        logger.warning(
            "Redis unavailable — falling back to in-memory brute-force tracking",
            extra={"service": ServiceName.SECURITY, "error": str(exc)},
        )
        _redis_client = None
        return None

# Fallback in-memory dict (used only when Redis is unavailable)
_failed_attempts_memory: dict[str, list[float]] = {}

# ---------------------------------------------------------------------------
# Security response headers added to every response
# ---------------------------------------------------------------------------
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


# ==============================================================================
# Auth Middleware
# ==============================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that runs on every HTTP request.

    Does NOT enforce authentication — that is done by require_auth().
    Instead this middleware handles cross-cutting concerns:
      - Request ID injection and logging context binding
      - Structured request/response logging with timing
      - Global security headers on all responses
      - Safe top-level exception handling

    Registered in main.py:
        app.add_middleware(AuthMiddleware)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process every incoming request.

        1. Extract or generate request ID
        2. Bind request ID to logging context
        3. Log the incoming request
        4. Forward to route handler
        5. Log the response with timing
        6. Attach security headers to response
        7. Return response

        Any unhandled exception from the route is caught and returned
        as a safe JSON error — never exposing internal details.
        """
        start_time = time.monotonic()

        # ── Request ID ────────────────────────────────────────────────
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # ── Logging context ───────────────────────────────────────────
        async with RequestContext(request_id=request_id):
            log_extra = {
                "service": ServiceName.SECURITY,
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": _get_client_ip(request),
            }

            logger.info("Request received", extra=log_extra)

            # ── Forward to route ──────────────────────────────────────
            try:
                response: Response = await call_next(request)

            except Exception as exc:
                # Catch any unhandled exception from route handlers
                # Never expose internal error details to the client
                duration_ms = round((time.monotonic() - start_time) * 1000, 2)
                logger.error(
                    "Unhandled exception in request handler",
                    extra={
                        **log_extra,
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
                error_response = JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "status": "error",
                        "message": "An internal error occurred.",
                        "data": None,
                        "request_id": request_id,
                    },
                )
                _attach_security_headers(error_response, request_id)
                return error_response

            # ── Log response ──────────────────────────────────────────
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            logger.log(
                log_level,
                "Request completed",
                extra={
                    **log_extra,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

            # ── Attach security headers ───────────────────────────────
            _attach_security_headers(response, request_id)

            return response


# ==============================================================================
# require_auth — FastAPI dependency for protected routes
# ==============================================================================

# HTTPBearer extracts the token from "Authorization: Bearer <token>"
# auto_error=False means we handle the 401 ourselves with a structured response
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db=Depends(get_db_session),
) -> object:
    """
    FastAPI dependency that enforces authentication on protected routes.

    Usage:
        @router.get("/reviews")
        async def list_reviews(business=Depends(require_auth)):
            # business is the authenticated BusinessModel object
            return await review_repo.get_all(business_id=business.id)

    Checks performed (in order):
      1. Authorization header present and contains a Bearer token
      2. Token signature valid and not expired
      3. Token not blacklisted (revoked)
      4. Business exists in the database
      5. Business has an active subscription

    Raises:
        HTTP 401: Missing token, invalid token, expired token, revoked token
        HTTP 403: Business found but subscription is inactive/expired
        HTTP 500: Unexpected error during verification

    Returns:
        BusinessModel instance for the authenticated business.
    """
    client_ip = _get_client_ip(request)
    request_id = request.headers.get("X-Request-ID", "")

    # ── Check IP brute-force protection ──────────────────────────────
    if _is_ip_rate_limited(client_ip):
        logger.warning(
            "IP rate limit exceeded on auth endpoint",
            extra={
                "service": ServiceName.SECURITY,
                "client_ip": client_ip,
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "status": "error",
                "message": "Too many failed authentication attempts. "
                           "Please try again in 5 minutes.",
                "data": None,
            },
        )

    # ── Token presence check ──────────────────────────────────────────
    if not credentials or not credentials.credentials:
        _record_failed_attempt(client_ip)
        logger.warning(
            "Request missing Authorization token",
            extra={
                "service": ServiceName.SECURITY,
                "client_ip": client_ip,
                "path": request.url.path,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Authentication required. "
                           "Provide a Bearer token in the Authorization header.",
                "data": None,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_token = credentials.credentials

    try:
        # ── Token verification ────────────────────────────────────────
        token_manager = TokenManager()
        verification = await token_manager.verify_access_token(
            db=db,
            token=raw_token,
        )

        if not verification.valid:
            _record_failed_attempt(client_ip)
            logger.warning(
                "Token verification failed",
                extra={
                    "service": ServiceName.SECURITY,
                    "client_ip": client_ip,
                    "reason": verification.error,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "status": "error",
                    "message": "Invalid or expired token. Please log in again.",
                    "data": None,
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        business_id = verification.payload.business_id

        # ── Business existence check ──────────────────────────────────
        business_repo = BusinessRepository()
        business = await business_repo.get_by_id(db=db, business_id=business_id)

        if not business:
            logger.warning(
                "Authenticated token references non-existent business",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": business_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "status": "error",
                    "message": "Business account not found.",
                    "data": None,
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # ── Active subscription check ─────────────────────────────────
        sub_repo = SubscriptionRepository()
        active_sub = await sub_repo.get_active(db=db, business_id=business_id)

        if not active_sub:
            logger.info(
                "Authenticated request from business with no active subscription",
                extra={
                    "service": ServiceName.SECURITY,
                    "business_id": business_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "error",
                    "message": "No active subscription. "
                               "Please renew your subscription to continue.",
                    "data": None,
                },
            )

        # ── Success ───────────────────────────────────────────────────
        logger.debug(
            "Request authenticated",
            extra={
                "service": ServiceName.SECURITY,
                "business_id": business_id,
            },
        )

        return business

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions unchanged
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error in require_auth",
            extra={
                "service": ServiceName.SECURITY,
                "client_ip": client_ip,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Authentication service unavailable.",
                "data": None,
            },
        )


# ==============================================================================
# Optional auth — for routes that work both authenticated and unauthenticated
# ==============================================================================

async def optional_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db=Depends(get_db_session),
) -> Optional[object]:
    """
    FastAPI dependency that attempts auth but does not require it.

    Returns the authenticated BusinessModel if a valid token is provided,
    or None if no token is present or the token is invalid.

    Use for routes that behave differently for authenticated vs anonymous
    users (e.g. public status pages with extra data for logged-in users).

    Usage:
        @router.get("/status")
        async def status(business=Depends(optional_auth)):
            if business:
                # Return detailed status for authenticated business
            else:
                # Return public status only
    """
    if not credentials or not credentials.credentials:
        return None

    try:
        token_manager = TokenManager()
        verification = await token_manager.verify_access_token(
            db=db,
            token=credentials.credentials,
        )
        if not verification.valid:
            return None

        business_repo = BusinessRepository()
        return await business_repo.get_by_id(
            db=db,
            business_id=verification.payload.business_id,
        )
    except Exception:
        return None


# ==============================================================================
# Pure helpers
# ==============================================================================

def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP address from the request.

    Checks X-Forwarded-For (set by reverse proxies like Nginx)
    before falling back to the direct connection IP.
    Only trust X-Forwarded-For if your deployment uses a trusted proxy.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For: client, proxy1, proxy2
        # First entry is the original client IP
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _attach_security_headers(response: Response, request_id: str) -> None:
    """
    Attach security headers and request ID to the response.

    Called on every response — both successful and error responses.
    """
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    response.headers["X-Request-ID"] = request_id


def _is_ip_rate_limited(client_ip: str) -> bool:
    """
    Check if an IP has exceeded the failed authentication attempt limit.

    Uses Redis sliding window counter when available (safe for multi-worker).
    Falls back to in-memory sliding window when Redis is unavailable.
    """
    redis = _get_redis()

    if redis is not None:
        # Redis path — shared across all workers
        try:
            key = f"brute_force:{client_ip}"
            count = redis.get(key)
            return int(count) >= _MAX_FAILED_ATTEMPTS if count else False
        except Exception as exc:
            logger.warning(
                "Redis read failed in rate limit check — using memory fallback",
                extra={"service": ServiceName.SECURITY, "error": str(exc)},
            )
            # Fall through to memory path below

    # Memory fallback path
    now = time.monotonic()
    cutoff = now - _FAILED_ATTEMPT_WINDOW_SECONDS
    attempts = _failed_attempts_memory.get(client_ip, [])
    fresh = [t for t in attempts if t > cutoff]
    _failed_attempts_memory[client_ip] = fresh
    return len(fresh) >= _MAX_FAILED_ATTEMPTS


def _record_failed_attempt(client_ip: str) -> None:
    """
    Record a failed authentication attempt for the given IP.

    Uses Redis INCR + EXPIRE when available (safe for multi-worker).
    Falls back to in-memory list when Redis is unavailable.
    """
    redis = _get_redis()

    if redis is not None:
        try:
            key = f"brute_force:{client_ip}"
            pipe = redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, _FAILED_ATTEMPT_WINDOW_SECONDS)
            pipe.execute()
            return
        except Exception as exc:
            logger.warning(
                "Redis write failed in record attempt — using memory fallback",
                extra={"service": ServiceName.SECURITY, "error": str(exc)},
            )
            # Fall through to memory path below

    # Memory fallback path
    now = time.monotonic()
    if client_ip not in _failed_attempts_memory:
        _failed_attempts_memory[client_ip] = []
    _failed_attempts_memory[client_ip].append(now)


def _is_public_path(path: str) -> bool:
    """
    Return True if the path is in the public routes list.

    Used internally — individual routes manage their own auth
    via require_auth dependency rather than path matching.
    """
    return any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)