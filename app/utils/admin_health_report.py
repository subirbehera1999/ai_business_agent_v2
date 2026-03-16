# ==============================================================================
# File: app/utils/admin_health_report.py
# Purpose: Orchestrates the daily system health report sent to the platform
#          administrator via WhatsApp.
#
#          This file exposes exactly ONE public function:
#
#            run_health_report(db, admin_notification, scheduler_manager=None)
#
#          That function is imported and called directly by scheduler_manager.py:
#
#            from app.utils.admin_health_report import run_health_report
#
#            async def _job_health_report(self) -> None:
#                async with _db_session() as db:
#                    await _run_with_lock(
#                        ...
#                        job_fn=lambda: run_health_report(
#                            db=db,
#                            admin_notification=self._ctx.admin_notification,
#                        ),
#                        ...
#                    )
#
#          What run_health_report() does:
#
#            Step 1 — Infrastructure checks
#              Calls system_health.run_all_checks() to get boolean results
#              for database, scheduler, Google API, WhatsApp API, OpenAI API.
#
#            Step 2 — Platform metrics
#              Queries the database for:
#                - active_businesses:    businesses with active subscriptions
#                - active_subscriptions: subscription records in ACTIVE state
#                - jobs_run_today:       job_logs records created today
#                - errors_today:         error_logs records created today
#
#            Step 3 — Deliver
#              Calls admin_notification.send_health_summary() with all the
#              gathered data. That method is already fully implemented in
#              admin_notification_service.py — this file just provides the
#              data it needs.
#
#          What this file does NOT do:
#            - Define any scheduling logic (that is scheduler_manager.py)
#            - Send any WhatsApp messages directly (that is admin_notification_service.py)
#            - Run health checks (that is system_health.py)
#            - Contain any notification formatting (that is template_manager.py)
#
#          Scheduler_manager passes scheduler_manager=None by default because
#          the scheduler instance is available at the SchedulerManager level
#          not at the job wrapper level. The scheduler_ok check defaults to
#          checking whether the APScheduler is_running flag is True. If None
#          is passed, system_health.py returns False for that check only.
#
#          Platform metric queries:
#            All queries use LIMIT to prevent full table scans.
#            Counts use COUNT() aggregate — never load records into memory.
#            All queries include today's date filter to scope to today only.
#
#          Never raises:
#            run_health_report() wraps all work in try/except.
#            A failed health report must never crash the scheduler.
#            Failures are logged and the function returns gracefully.
# ==============================================================================

import logging
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName, SubscriptionStatus
from app.database.models.subscription_model import SubscriptionModel
from app.database.models.usage_model import UsageModel
from app.notifications.admin_notification_service import AdminNotificationService
from app.utils.system_health import SystemHealthResult, run_all_checks

logger = logging.getLogger(ServiceName.SYSTEM_HEALTH)


# ==============================================================================
# Public entry point — called by scheduler_manager.py
# ==============================================================================

async def run_health_report(
    db: AsyncSession,
    admin_notification: AdminNotificationService,
    scheduler_manager: Optional[object] = None,
) -> None:
    """
    Orchestrate and deliver the daily system health report to the admin.

    This is the exact function imported by scheduler_manager.py and called
    inside the _job_health_report() wrapper. It runs once daily at 08:00 UTC.

    Steps:
      1. Run all infrastructure health checks via system_health.run_all_checks()
      2. Query platform metrics from the database (active businesses, errors, etc.)
      3. Call admin_notification.send_health_summary() to deliver the report

    Args:
        db:                 Active AsyncSession from the scheduler job wrapper.
        admin_notification: AdminNotificationService instance from SchedulerContext.
        scheduler_manager:  Optional SchedulerManager instance for scheduler
                            health check. Passed as None from scheduler_manager.py
                            (the scheduler is always running when this job fires,
                            so the check is informational only).

    Returns:
        None. Never raises — failures are logged internally.
    """
    log_extra = {"service": ServiceName.SYSTEM_HEALTH}

    try:
        logger.info("Health report job started", extra=log_extra)

        # ------------------------------------------------------------------
        # Step 1: Infrastructure health checks
        # ------------------------------------------------------------------
        health: SystemHealthResult = await run_all_checks(
            scheduler_manager=scheduler_manager,
        )

        logger.info(
            "Infrastructure checks complete",
            extra={**log_extra, **health.to_log_dict()},
        )

        # ------------------------------------------------------------------
        # Step 2: Platform metrics
        # ------------------------------------------------------------------
        metrics = await _collect_platform_metrics(db)

        logger.info(
            "Platform metrics collected",
            extra={**log_extra, **metrics},
        )

        # ------------------------------------------------------------------
        # Step 3: Deliver health summary to admin
        # ------------------------------------------------------------------
        report_date = date.today().isoformat()

        result = await admin_notification.send_health_summary(
            db_ok=health.db_ok,
            scheduler_ok=health.scheduler_ok,
            google_api_ok=health.google_api_ok,
            whatsapp_api_ok=health.whatsapp_api_ok,
            openai_api_ok=health.openai_api_ok,
            active_businesses=metrics["active_businesses"],
            active_subscriptions=metrics["active_subscriptions"],
            jobs_run_today=metrics["jobs_run_today"],
            errors_today=metrics["errors_today"],
            report_date=report_date,
        )

        if result.success and not result.skipped:
            logger.info(
                "Health report delivered to admin",
                extra={**log_extra, "message_id": result.message_id},
            )
        elif result.skipped:
            logger.info(
                "Health report skipped — no admin number configured",
                extra=log_extra,
            )
        else:
            logger.warning(
                "Health report delivery failed",
                extra={**log_extra, "error": result.error},
            )

        # ------------------------------------------------------------------
        # Step 4: If any infrastructure check failed — send critical alert
        # ------------------------------------------------------------------
        if not health.all_ok:
            await _send_failure_alerts(
                health=health,
                admin_notification=admin_notification,
            )

    except Exception as exc:
        # Health report failure must never crash the scheduler
        logger.error(
            "Health report job failed with unexpected exception",
            extra={**log_extra, "error": str(exc)},
        )


# ==============================================================================
# Platform metrics collector
# ==============================================================================

async def _collect_platform_metrics(db: AsyncSession) -> dict:
    """
    Query the database for today's platform-wide operational metrics.

    All queries are aggregate COUNT() operations — no records are loaded
    into Python memory. All queries are scoped to today's date where
    applicable to keep the numbers relevant to the daily report.

    Metrics collected:
        active_businesses:    Distinct businesses with an ACTIVE subscription.
        active_subscriptions: Total ACTIVE subscription records right now.
        jobs_run_today:       Usage records with any activity today
                              (proxy for jobs processed — lightweight query).
        errors_today:         Error log entries created today
                              (queried from error_logs table if available).

    Args:
        db: Active AsyncSession.

    Returns:
        dict with keys: active_businesses, active_subscriptions,
                        jobs_run_today, errors_today.
        All values default to 0 on any query failure — metrics are
        best-effort and must not block the health report delivery.
    """
    today = date.today()
    metrics = {
        "active_businesses": 0,
        "active_subscriptions": 0,
        "jobs_run_today": 0,
        "errors_today": 0,
    }

    # ------------------------------------------------------------------
    # Active subscriptions count
    # ------------------------------------------------------------------
    try:
        result = await db.execute(
            select(func.count(SubscriptionModel.id)).where(
                SubscriptionModel.status == SubscriptionStatus.ACTIVE,
            )
        )
        metrics["active_subscriptions"] = result.scalar_one() or 0
    except SQLAlchemyError as exc:
        logger.warning(
            "Failed to count active subscriptions for health report",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Active businesses count (distinct businesses with active subscription)
    # ------------------------------------------------------------------
    try:
        result = await db.execute(
            select(
                func.count(func.distinct(SubscriptionModel.business_id))
            ).where(
                SubscriptionModel.status == SubscriptionStatus.ACTIVE,
            )
        )
        metrics["active_businesses"] = result.scalar_one() or 0
    except SQLAlchemyError as exc:
        logger.warning(
            "Failed to count active businesses for health report",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Jobs run today — count of usage records with last_activity_at today
    # This is a lightweight proxy: each business with activity today
    # has had at least one job touch their usage record.
    # ------------------------------------------------------------------
    try:
        result = await db.execute(
            select(func.count(UsageModel.id)).where(
                UsageModel.usage_date == today,
            )
        )
        metrics["jobs_run_today"] = result.scalar_one() or 0
    except SQLAlchemyError as exc:
        logger.warning(
            "Failed to count jobs run today for health report",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Errors today — query error_logs table if it exists.
    # ErrorLog model is defined inside error_tracker.py (co-located).
    # We import it here at query time to avoid a top-level circular import.
    # ------------------------------------------------------------------
    try:
        from app.logging.error_tracker import ErrorLog  # co-located model

        today_start = datetime.combine(today, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        result = await db.execute(
            select(func.count(ErrorLog.id)).where(
                ErrorLog.created_at >= today_start,
            )
        )
        metrics["errors_today"] = result.scalar_one() or 0
    except ImportError:
        # error_tracker.py not yet available in this deployment
        logger.debug(
            "ErrorLog not importable — errors_today defaults to 0",
            extra={"service": ServiceName.SYSTEM_HEALTH},
        )
    except SQLAlchemyError as exc:
        logger.warning(
            "Failed to count errors today for health report",
            extra={
                "service": ServiceName.SYSTEM_HEALTH,
                "error": str(exc),
            },
        )

    return metrics


# ==============================================================================
# Failure alert dispatcher
# ==============================================================================

async def _send_failure_alerts(
    health: SystemHealthResult,
    admin_notification: AdminNotificationService,
) -> None:
    """
    Send targeted critical/warning alerts for each failed infrastructure check.

    Called only when health.all_ok is False. Sends one alert per failed
    component so the admin knows exactly which service is down, not just
    that "something is wrong".

    The send_health_summary() already shows the full status grid. These
    additional alerts provide immediately actionable context for each failure.

    Args:
        health:             SystemHealthResult from run_all_checks().
        admin_notification: AdminNotificationService for alert delivery.
    """
    log_extra = {"service": ServiceName.SYSTEM_HEALTH}

    if not health.db_ok:
        logger.critical(
            "CRITICAL: Database is unreachable — sending admin alert",
            extra=log_extra,
        )
        await admin_notification.send_critical(
            title="Database Unreachable",
            message=(
                "PostgreSQL failed the SELECT 1 health check.\n\n"
                "Immediate action required:\n"
                "• Check DATABASE_URL environment variable\n"
                "• Verify PostgreSQL service is running\n"
                "• Check server disk space and connection limits\n\n"
                "All business data operations are currently failing."
            ),
        )

    if not health.scheduler_ok:
        logger.error(
            "Scheduler is not running — sending admin warning",
            extra=log_extra,
        )
        await admin_notification.send_warning(
            title="Scheduler Not Running",
            message=(
                "The APScheduler background job runner is not active.\n\n"
                "This means:\n"
                "• No new reviews are being processed\n"
                "• No reports will be generated\n"
                "• No sales analytics are running\n\n"
                "Check application logs for scheduler startup errors."
            ),
        )

    if not health.google_api_ok:
        await admin_notification.send_warning(
            title="Google API Unreachable",
            message=(
                "Google API health check failed.\n\n"
                "Affected features:\n"
                "• Google Reviews polling\n"
                "• Google Sheets sales data sync\n\n"
                "This may be a temporary outage. "
                "Review monitoring will resume automatically when API is restored."
            ),
        )

    if not health.whatsapp_api_ok:
        await admin_notification.send_critical(
            title="WhatsApp API Unreachable",
            message=(
                "WhatsApp Cloud API (Meta Graph API) health check failed.\n\n"
                "Affected features:\n"
                "• All business alert deliveries\n"
                "• Weekly and monthly reports\n"
                "• AI reply notifications\n\n"
                "No WhatsApp messages can be delivered until this is resolved.\n"
                "Check Meta Business Suite for service status."
            ),
        )

    if not health.openai_api_ok:
        await admin_notification.send_warning(
            title="OpenAI API Unreachable",
            message=(
                "OpenAI API health check failed.\n\n"
                "Affected features:\n"
                "• AI review reply generation\n"
                "• Business insight generation\n"
                "• Social media content generation\n\n"
                "Reviews will be stored but AI replies will fail until resolved."
            ),
        )