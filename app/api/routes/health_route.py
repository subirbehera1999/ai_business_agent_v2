# ==============================================================================
# File: app/api/routes/health_route.py
# Purpose: Health check endpoints for server monitoring and uptime probes.
#
#          Two endpoints are exposed:
#
#          GET /api/v1/health
#          ─────────────────
#          Lightweight liveness probe. Returns immediately with a 200 OK
#          and basic app metadata. Does NOT check the database or external
#          services. Used by load balancers, container orchestrators (ECS,
#          Kubernetes), and uptime monitors to verify the process is alive.
#
#          GET /api/v1/health/detailed
#          ───────────────────────────
#          Full infrastructure readiness probe. Runs all health checks:
#            - PostgreSQL database connectivity
#            - APScheduler running status
#            - Google API reachability
#            - WhatsApp Cloud API reachability
#            - OpenAI API reachability
#
#          Returns HTTP 200 if all checks pass, HTTP 503 if any fail.
#          The response body always includes per-component status so
#          monitoring systems can pinpoint which service is degraded.
#
#          Authentication:
#            Both endpoints are public — listed in _PUBLIC_PATH_PREFIXES
#            in auth_middleware.py. No token required. This is intentional:
#            load balancers and uptime monitors cannot send auth headers.
#            The detailed endpoint does not expose sensitive data — only
#            boolean pass/fail per component.
#
#          Response format (all endpoints):
#            {
#              "status": "ok" | "error",
#              "message": "...",
#              "data": { ... }
#            }
# ==============================================================================

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.system_health import run_all_checks

logger = logging.getLogger(ServiceName.API)

router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health"],
)


# ==============================================================================
# GET /api/v1/health  — liveness probe
# ==============================================================================

@router.get(
    "",
    summary="Liveness probe",
    description=(
        "Lightweight liveness check. Returns 200 immediately if the process "
        "is running. Does not check the database or external services."
    ),
    status_code=status.HTTP_200_OK,
)
async def health_check(request: Request) -> JSONResponse:
    """
    Liveness probe — confirms the API process is running.

    Intended for load balancers and container health checks that need
    a fast, low-cost response. Does not touch the database or any
    external API. Always returns 200 if the process is alive.

    Returns:
        200 OK with app name, version, and UTC timestamp.
    """
    settings = get_settings()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "message": "Service is running.",
            "data": {
                "app": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


# ==============================================================================
# GET /api/v1/health/detailed  — readiness / deep probe
# ==============================================================================

@router.get(
    "/detailed",
    summary="Readiness probe — full infrastructure check",
    description=(
        "Runs all infrastructure health checks: database, scheduler, "
        "Google API, WhatsApp API, OpenAI API. "
        "Returns 200 if all pass, 503 if any fail."
    ),
)
async def health_check_detailed(request: Request) -> JSONResponse:
    """
    Readiness probe — verifies all infrastructure dependencies are reachable.

    Runs the following checks via system_health.run_all_checks():
      - PostgreSQL database connectivity (SELECT 1)
      - APScheduler running status
      - Google API endpoint reachability
      - WhatsApp Cloud API endpoint reachability
      - OpenAI API endpoint reachability

    The scheduler_manager is not passed here (no access to the live
    scheduler instance from this layer) — scheduler_ok reflects whether
    the scheduler process started, not whether jobs are currently running.

    HTTP response codes:
      200  All checks passed — system is fully ready.
      503  One or more checks failed — system is degraded.

    The response body always includes per-component breakdown so
    monitoring tools can pinpoint the failing component.

    Returns:
        200 or 503 JSON response with per-component health status.
    """
    log_extra = {
        "service": ServiceName.API,
        "endpoint": "/api/v1/health/detailed",
    }

    logger.info("Detailed health check requested", extra=log_extra)

    health = await run_all_checks(scheduler_manager=None)

    http_status = (
        status.HTTP_200_OK
        if health.all_ok
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    if health.all_ok:
        logger.info("Detailed health check passed", extra=log_extra)
    else:
        logger.warning(
            "Detailed health check — degraded components detected",
            extra={**log_extra, "failures": health.failures},
        )

    settings = get_settings()

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ok" if health.all_ok else "degraded",
            "message": (
                "All systems operational."
                if health.all_ok
                else f"Degraded: {', '.join(health.failures)}"
            ),
            "data": {
                "app": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "timestamp": health.checked_at.isoformat(),
                "components": {
                    "database":     health.db_ok,
                    "scheduler":    health.scheduler_ok,
                    "google_api":   health.google_api_ok,
                    "whatsapp_api": health.whatsapp_api_ok,
                    "openai_api":   health.openai_api_ok,
                },
                "all_ok": health.all_ok,
                "failures": health.failures,
            },
        },
    )