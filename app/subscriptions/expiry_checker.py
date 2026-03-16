# ==============================================================================
# File: app/subscriptions/expiry_checker.py
# Purpose: Scheduler-facing module that runs the daily subscription expiry
#          check with job locking, failsafe execution, and idempotency.
#
#          This module is the thin orchestration layer between the
#          scheduler (scheduler_manager.py) and the business logic
#          (subscription_service.check_and_expire_subscriptions()).
#
#          Responsibilities:
#            1. Job locking
#               - Acquires a distributed job lock before running
#               - Prevents concurrent execution if scheduler fires twice
#                 (e.g. after a server restart during a run)
#               - Lock key: JOB_LOCK_EXPIRY_CHECK_{iso_date}
#               - Lock expiry: 30 minutes (generous for large deployments)
#
#            2. Idempotency
#               - Uses the ISO date as part of the lock key
#               - Running twice on the same calendar day is safe:
#                 second run acquires lock → finds today's lock already
#                 exists → skips gracefully
#
#            3. Failsafe execution
#               - Wrapped in failsafe_runner to prevent scheduler crashes
#               - Any uncaught exception is logged and swallowed
#               - Scheduler continues operating after a failed expiry run
#
#            4. Result reporting
#               - Logs a structured summary after every run
#               - Admin is notified if any errors occurred during processing
#
#          Schedule:
#            Runs once per day at 00:05 UTC (5 minutes after midnight)
#            to ensure all same-day expirations are caught.
#            Scheduled by scheduler_manager.py.
#
#          Performance contract:
#            - Never processes all businesses in a single query
#            - Delegates batching to subscription_service.py
#            - Maximum job execution budget: 2 minutes
#            - If run takes longer, lock expiry prevents double execution
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import JobName, ServiceName
from app.subscriptions.subscription_service import ExpiryCheckResult, SubscriptionService
from app.utils.failsafe_runner import run_job
from app.utils.idempotency_utils import make_job_lock_key
from app.utils.time_utils import today_local

logger = logging.getLogger(ServiceName.SUBSCRIPTIONS)

# Lock duration in seconds — 30 minutes
_LOCK_TTL_SECONDS: int = 30 * 60

# Job name constant for lock keys and logging
_JOB_NAME = JobName.EXPIRY_CHECK


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class ExpiryCheckRunResult:
    """
    Result of a single expiry checker invocation.

    Attributes:
        ran:              True if the job actually executed (not skipped).
        skipped:          True if skipped due to lock already held.
        skip_reason:      Why it was skipped ("lock_held", "already_ran_today").
        check_result:     ExpiryCheckResult from subscription_service (if ran).
        error:            Top-level error message if the run itself failed.
        run_date:         The calendar date this run was for.
    """
    ran: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    check_result: Optional[ExpiryCheckResult] = None
    error: Optional[str] = None
    run_date: Optional[date] = None

    @property
    def success(self) -> bool:
        return self.ran and self.error is None

    @property
    def had_processing_errors(self) -> bool:
        return (
            self.check_result is not None
            and self.check_result.errors > 0
        )

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"ExpiryCheckRunResult("
                f"skipped=True reason={self.skip_reason})"
            )
        if not self.ran:
            return f"ExpiryCheckRunResult(ran=False error={self.error})"
        cr = self.check_result
        return (
            f"ExpiryCheckRunResult("
            f"ran=True "
            f"checked={cr.checked if cr else 0} "
            f"expired={cr.expired if cr else 0} "
            f"reminders={cr.reminder_sent if cr else 0} "
            f"errors={cr.errors if cr else 0})"
        )


# ==============================================================================
# Expiry Checker
# ==============================================================================

class ExpiryChecker:
    """
    Orchestrates the daily subscription expiry check job.

    Thin coordinator between the scheduler and SubscriptionService.
    Handles job locking, idempotency, and failsafe wrapping.

    Usage (from scheduler_manager.py):

        checker = ExpiryChecker(
            subscription_service=subscription_service,
            job_repo=job_repo,
            admin_notification=admin_notification,
        )

        result = await checker.run(db=db)
    """

    def __init__(
        self,
        subscription_service: SubscriptionService,
        job_repo,               # JobRepository
        admin_notification,     # AdminNotificationService
    ) -> None:
        self._subscription_service = subscription_service
        self._job_repo = job_repo
        self._admin = admin_notification

    async def run(
        self,
        db: AsyncSession,
        run_date: Optional[date] = None,
    ) -> ExpiryCheckRunResult:
        """
        Execute the daily subscription expiry check.

        Acquires a date-scoped job lock before running to prevent
        concurrent or duplicate execution. Delegates all business logic
        to subscription_service.check_and_expire_subscriptions().

        Args:
            db:        AsyncSession.
            run_date:  Override the run date (for testing / backfill).
                       Defaults to today in local timezone.

        Returns:
            ExpiryCheckRunResult. Never raises.
        """
        today = run_date or today_local()
        lock_key = make_job_lock_key(
            job_name=_JOB_NAME,
            date_str=today.isoformat(),
        )
        log_extra = {
            "service": ServiceName.SUBSCRIPTIONS,
            "job": _JOB_NAME,
            "run_date": today.isoformat(),
            "lock_key": lock_key,
        }

        # ------------------------------------------------------------------
        # Attempt to acquire job lock
        # ------------------------------------------------------------------
        lock_acquired = await self._try_acquire_lock(
            db=db,
            lock_key=lock_key,
            log_extra=log_extra,
        )

        if not lock_acquired:
            logger.info(
                "Expiry check skipped — lock already held",
                extra=log_extra,
            )
            return ExpiryCheckRunResult(
                ran=False,
                skipped=True,
                skip_reason="lock_held",
                run_date=today,
            )

        # ------------------------------------------------------------------
        # Execute with failsafe wrapper
        # ------------------------------------------------------------------
        async def _do_run() -> ExpiryCheckResult:
            return await self._subscription_service.check_and_expire_subscriptions(
                db=db,
            )

        check_result: Optional[ExpiryCheckResult] = None
        async with run_job(job_type=_JOB_NAME) as job_result:
            check_result = await _do_run()

        # ------------------------------------------------------------------
        # Release lock and build result
        # ------------------------------------------------------------------
        await self._release_lock(
            db=db,
            lock_key=lock_key,
            log_extra=log_extra,
        )

        if job_result.failed:
            error_msg = job_result.error_message or "unknown error"
            logger.error(
                "Expiry check job failed",
                extra={**log_extra, "error": error_msg},
            )
            await self._admin.send_alert(
                title="🔴 Subscription expiry check failed",
                message=(
                    f"The daily subscription expiry check failed on "
                    f"{today.isoformat()}.\n"
                    f"Error: {error_msg}\n"
                    f"Some subscriptions may not have been expired. "
                    f"Manual review recommended."
                ),
            )
            return ExpiryCheckRunResult(
                ran=True,
                error=error_msg,
                run_date=today,
            )

        assert check_result is not None  # set when job_result.succeeded

        # Notify admin if any individual subscription processing errors occurred
        if check_result.errors > 0:
            await self._admin.send_alert(
                title="⚠️ Expiry check completed with errors",
                message=(
                    f"Subscription expiry check on {today.isoformat()} "
                    f"completed with {check_result.errors} error(s).\n"
                    f"Checked: {check_result.checked} | "
                    f"Expired: {check_result.expired} | "
                    f"Reminders sent: {check_result.reminder_sent}\n"
                    f"Check logs for affected subscription IDs."
                ),
            )

        logger.info(
            "Expiry check completed successfully",
            extra={
                **log_extra,
                "checked": check_result.checked,
                "expired": check_result.expired,
                "reminders": check_result.reminder_sent,
                "errors": check_result.errors,
            },
        )

        return ExpiryCheckRunResult(
            ran=True,
            check_result=check_result,
            run_date=today,
        )

    # ------------------------------------------------------------------
    # Job lock helpers
    # ------------------------------------------------------------------

    async def _try_acquire_lock(
        self,
        db: AsyncSession,
        lock_key: str,
        log_extra: dict,
    ) -> bool:
        """
        Attempt to acquire the job lock for this run.

        Creates a job_log record with the lock_key. If the record already
        exists, the lock is considered held and False is returned.

        Args:
            db:        AsyncSession.
            lock_key:  Unique lock identifier for this job + date.
            log_extra: Structured log context.

        Returns:
            True if lock acquired, False if already held.
        """
        try:
            acquired = await self._job_repo.acquire_lock(
                db=db,
                lock_key=lock_key,
                job_name=_JOB_NAME,
                ttl_seconds=_LOCK_TTL_SECONDS,
            )
            if acquired:
                logger.debug(
                    "Job lock acquired",
                    extra=log_extra,
                )
            return acquired
        except Exception as exc:
            # If lock acquisition itself fails, run anyway to avoid
            # missing expirations — log the failure prominently
            logger.error(
                "Failed to acquire job lock — running without lock",
                extra={**log_extra, "error": str(exc)},
            )
            return True   # proceed without lock rather than silently skip

    async def _release_lock(
        self,
        db: AsyncSession,
        lock_key: str,
        log_extra: dict,
    ) -> None:
        """
        Release the job lock after the run completes.

        Lock release failure is logged but never raised — the TTL
        (_LOCK_TTL_SECONDS) acts as the backstop to prevent the lock
        from being held indefinitely.

        Args:
            db:        AsyncSession.
            lock_key:  Lock key to release.
            log_extra: Structured log context.
        """
        try:
            await self._job_repo.release_lock(
                db=db,
                lock_key=lock_key,
            )
            logger.debug("Job lock released", extra=log_extra)
        except Exception as exc:
            logger.warning(
                "Failed to release job lock — TTL will clear it",
                extra={**log_extra, "error": str(exc), "ttl": _LOCK_TTL_SECONDS},
            )


# ==============================================================================
# Standalone runner — for direct invocation from scheduler_manager
# ==============================================================================

async def run_expiry_check(
    db: AsyncSession,
    subscription_service: SubscriptionService,
    job_repo,
    admin_notification,
    run_date: Optional[date] = None,
) -> ExpiryCheckRunResult:
    """
    Functional entry point for the expiry check job.

    Convenience wrapper that instantiates ExpiryChecker and runs it.
    Suitable for use in scheduler_manager.py without requiring the
    caller to manage the ExpiryChecker instance.

    Args:
        db:                   AsyncSession.
        subscription_service: Injected SubscriptionService.
        job_repo:             Injected JobRepository.
        admin_notification:   Injected AdminNotificationService.
        run_date:             Override run date (for testing / backfill).

    Returns:
        ExpiryCheckRunResult. Never raises.
    """
    checker = ExpiryChecker(
        subscription_service=subscription_service,
        job_repo=job_repo,
        admin_notification=admin_notification,
    )
    return await checker.run(db=db, run_date=run_date)