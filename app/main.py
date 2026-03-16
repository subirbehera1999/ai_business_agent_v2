# ==============================================================================
# File: app/main.py
# Purpose: FastAPI application entrypoint.
#
#          This module is the root of the application. It:
#            1. Configures structured logging at startup
#            2. Creates the FastAPI application instance
#            3. Registers CORS and authentication middleware
#            4. Mounts the master API router
#            5. Manages application lifespan (startup / shutdown hooks)
#               - Opens the database connection pool
#               - Starts the background job scheduler
#               - Shuts both down cleanly on process termination
#
#          Lifespan pattern:
#            FastAPI's @asynccontextmanager lifespan is used instead of the
#            deprecated on_event("startup") / on_event("shutdown") handlers.
#            Code before `yield` runs at startup; code after runs at shutdown.
#
#          Deployment:
#            Run with:
#              uvicorn app.main:app --host 0.0.0.0 --port 8000
#
#            Production (with multiple workers):
#              gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
#
#          Worker scaling note:
#            The scheduler is started inside the lifespan, which means each
#            Gunicorn worker starts its own scheduler. For multi-worker
#            deployments, ensure job locking (scheduler_manager.py) is active
#            so duplicate job execution is prevented across workers.
# ==============================================================================

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.router import api_router
from app.config.constants import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION_PREFIX,
    ServiceName,
)
from app.config.settings import get_settings
from app.database.db import connect_database, disconnect_database
from app.logging.logger import configure_logging
from app.notifications.admin_notification_service import AdminNotificationService
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository
from app.schedulers.scheduler_manager import SchedulerContext, SchedulerManager
from app.security.auth_middleware import AuthMiddleware
from app.subscriptions.subscription_service import SubscriptionService
from app.integrations.whatsapp_client import WhatsAppClient

# Configure structured logging before anything else runs.
# This must be the very first call so all subsequent loggers inherit the config.
configure_logging()

logger = logging.getLogger(ServiceName.API)
settings = get_settings()


# ==============================================================================
# Application lifespan — startup and shutdown hooks
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle: startup and shutdown.

    Startup (before yield):
      1. Connect database connection pool
      2. Build shared service dependencies
      3. Start background job scheduler

    Shutdown (after yield):
      4. Shut down scheduler gracefully (waits for running jobs)
      5. Close database connection pool

    Any exception during startup propagates and prevents the server from
    starting, which is intentional — a misconfigured system should not serve
    traffic.
    """
    # ------------------------------------------------------------------
    # STARTUP
    # ------------------------------------------------------------------
    logger.info(
        "AI Business Agent starting up",
        extra={
            "service": ServiceName.API,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        },
    )

    # 1. Database connection pool
    await connect_database()
    logger.info("Database connection pool opened", extra={"service": ServiceName.API})

    # 2. Build shared dependencies for the scheduler
    #    These are stateless repository/service instances — safe to share.
    whatsapp_client       = WhatsAppClient()
    admin_notification    = AdminNotificationService(whatsapp_client=whatsapp_client)
    business_repo         = BusinessRepository()
    subscription_repo     = SubscriptionRepository()
    usage_repo            = UsageRepository()
    subscription_service  = SubscriptionService()

    scheduler_context = SchedulerContext(
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        usage_repo=usage_repo,
        subscription_service=subscription_service,
        admin_notification=admin_notification,
    )

    # 3. Start scheduler
    scheduler = SchedulerManager(context=scheduler_context)
    scheduler.start()

    # Store on app.state so health endpoints can inspect scheduler status
    app.state.scheduler = scheduler

    logger.info(
        "Scheduler started",
        extra={
            "service": ServiceName.API,
            "job_count": scheduler.get_job_count(),
        },
    )

    logger.info(
        "AI Business Agent startup complete — ready to serve requests",
        extra={"service": ServiceName.API},
    )

    # ------------------------------------------------------------------
    # YIELD — application serves requests here
    # ------------------------------------------------------------------
    yield

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------
    logger.info(
        "AI Business Agent shutting down",
        extra={"service": ServiceName.API},
    )

    # 4. Shut down scheduler — wait=True ensures running jobs complete
    scheduler.shutdown()
    logger.info("Scheduler shut down", extra={"service": ServiceName.API})

    # 5. Close database connection pool
    await disconnect_database()
    logger.info("Database connection pool closed", extra={"service": ServiceName.API})

    logger.info(
        "AI Business Agent shutdown complete",
        extra={"service": ServiceName.API},
    )


# ==============================================================================
# FastAPI application instance
# ==============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=f"{API_VERSION_PREFIX}/docs" if settings.APP_DEBUG else None,
    redoc_url=f"{API_VERSION_PREFIX}/redoc" if settings.APP_DEBUG else None,
    openapi_url=f"{API_VERSION_PREFIX}/openapi.json" if settings.APP_DEBUG else None,
    lifespan=lifespan,
)


# ==============================================================================
# Middleware registration
# Order matters — middleware is applied in reverse registration order,
# so the last registered runs first on incoming requests.
# ==============================================================================

# 1. CORS — must be registered before auth middleware so preflight OPTIONS
#    requests are handled without triggering auth checks.
_cors_origins = settings.allowed_origins_list or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# 2. Authentication — validates JWT tokens on all protected routes.
#    Public paths (health check, webhook receiver) are explicitly excluded
#    inside AuthMiddleware._is_public_path().
app.add_middleware(AuthMiddleware)


# ==============================================================================
# Router registration
# ==============================================================================

app.include_router(api_router)


# ==============================================================================
# Global exception handler
# ==============================================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception that escapes service layer.

    Returns a safe 500 response — never exposes internal error details to
    the client. The full traceback is logged internally for debugging.

    This handler is a last resort. All expected errors should be caught
    and handled within their respective service or route layers.
    """
    logger.error(
        "Unhandled exception — %s: %s",
        type(exc).__name__,
        str(exc),
        exc_info=True,
        extra={
            "service": ServiceName.API,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred. Please try again later.",
            "data": None,
        },
    )


# ==============================================================================
# Root redirect — convenience for browser access
# ==============================================================================

@app.get(
    "/",
    include_in_schema=False,
    status_code=status.HTTP_200_OK,
)
async def root() -> dict:
    """
    Root endpoint. Returns a minimal identification payload.

    Not included in OpenAPI schema — this is a convenience probe only.
    All real endpoints are under /api/v1/.
    """
    return {
        "status": "ok",
        "service": API_TITLE,
        "version": settings.APP_VERSION,
        "docs": f"{API_VERSION_PREFIX}/docs" if settings.APP_DEBUG else "disabled",
    }