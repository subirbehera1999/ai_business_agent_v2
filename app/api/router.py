# ==============================================================================
# File: app/api/router.py
# Purpose: Central API router aggregator.
#
#          This module is the single registration point for all route modules.
#          It imports each route's APIRouter and includes it into one master
#          router that is mounted onto the FastAPI app in main.py.
#
#          Adding a new route module:
#            1. Create the route file in app/api/routes/
#            2. Define `router = APIRouter(prefix=..., tags=[...])` in that file
#            3. Import the router here and call `api_router.include_router()`
#            4. No changes needed in main.py
#
#          Route prefix convention:
#            All routes are versioned under /api/v1/
#            Each sub-router owns its own prefix (set in the route file itself).
#            This router does NOT add an additional prefix — sub-routers already
#            carry their full path.
#
#          Current registered routes:
#            GET  /api/v1/health                     Health check
#            GET  /api/v1/health/detailed             Detailed system health
#            POST /api/v1/onboarding/register         Business registration
#            GET  /api/v1/onboarding/profile          Fetch business profile
#            PUT  /api/v1/onboarding/profile          Update business profile
#            POST /api/v1/payments/initiate           Initiate Razorpay order
#            GET  /api/v1/payments/status/{order_id}  Payment status check
#            GET  /api/v1/payments/subscription       Active subscription detail
#            POST /api/v1/webhooks/razorpay           Razorpay webhook receiver
#            GET  /api/v1/webhooks/razorpay/ping      Webhook endpoint probe
# ==============================================================================

import logging

from fastapi import APIRouter

from app.config.constants import API_VERSION_PREFIX, ServiceName
from app.api.routes.health_route import router as health_router
from app.api.routes.onboarding_route import router as onboarding_router
from app.api.routes.payment_route import router as payment_router
from app.api.routes.webhook_route import router as webhook_router

logger = logging.getLogger(ServiceName.API)

# ==============================================================================
# Master API Router
# ==============================================================================

api_router = APIRouter()

# ------------------------------------------------------------------------------
# Route registration
# Sub-routers carry their own prefix — no additional prefix added here.
# Registration order determines the order in the OpenAPI docs.
# ------------------------------------------------------------------------------

api_router.include_router(health_router)
api_router.include_router(onboarding_router)
api_router.include_router(payment_router)
api_router.include_router(webhook_router)

logger.debug(
    "API router assembled",
    extra={
        "service": ServiceName.API,
        "version_prefix": API_VERSION_PREFIX,
        "route_count": len(api_router.routes),
    },
)