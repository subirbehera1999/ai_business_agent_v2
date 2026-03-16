# ==============================================================================
# File: app/schedulers/scheduler_manager.py
# Purpose: Central scheduler controller for the AI Business Agent platform.
#
#          Manages all background jobs via APScheduler (AsyncIOScheduler).
#          Boots alongside the FastAPI application and shuts down cleanly
#          on application exit.
#
#          Registered jobs and schedules:
#
#            Job                        Schedule          Description
#            ─────────────────────────────────────────────────────────
#            review_monitor             Every 30 min      Poll new Google reviews
#            sales_analysis_job         Daily 01:00 UTC   Run sales analytics
#            weekly_report_job          Mon 06:00 UTC     Weekly performance report
#            weekly_content_job         Mon 07:00 UTC     Weekly social media content
#            monthly_report_job         1st 08:00 UTC     Monthly business report
#            quarterly_report_job       1st Jan/Apr/Jul/Oct 09:00 UTC
#            expiry_checker             Daily 00:05 UTC   Expire subscriptions
#            health_report              Daily 08:00 UTC   System health summary
#
#          Architecture principles enforced here:
#
#            1. Job locking (scheduler-level)
#               Each job acquires a distributed lock before executing.
#               Lock key: JOB_LOCK_{JOB_NAME}_{ISO_DATE}
#               If the lock is held, the job is skipped with a log entry.
#               Locks are stored in the database (job_model.py).
#               TTL: 15 minutes for standard jobs, 30 min for report jobs.
#
#            2. Failsafe execution
#               Every job is wrapped in failsafe_runner.run_job().
#               An uncaught exception in any job never crashes the scheduler.
#
#            3. Load distribution
#               Jobs iterate businesses in batches of 10-20.
#               No job processes all businesses in a single pass.
#               Each job module handles its own batching.
#
#            4. Job isolation
#               Failure in one business's job never blocks others.
#               Handled inside individual job modules.
#
#            5. Maximum execution time
#               Jobs are designed to complete within 2 minutes.
#               The lock TTL acts as the hard guard against runaway jobs.
#
#          Dependency injection:
#            SchedulerManager receives all service/repo dependencies at
#            construction time. It injects them into job functions at
#            registration time. No global state is used in job functions.
#
#          Startup / shutdown:
#            start()   → called from FastAPI lifespan startup
#            shutdown() → called from FastAPI lifespan shutdown
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.config.constants import JobName, ServiceName
from app.database.db import get_session
from app.notifications.admin_notification_service import AdminNotificationService
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository
from app.schedulers.review_monitor import run_review_monitor
from app.schedulers.sales_analysis_job import run_sales_analysis
from app.schedulers.weekly_report_job import run_weekly_report
from app.schedulers.weekly_content_job import run_weekly_content
from app.schedulers.monthly_report_job import run_monthly_report
from app.schedulers.quarterly_report_job import run_quarterly_report
from app.subscriptions.expiry_checker import run_expiry_check
from app.subscriptions.subscription_service import SubscriptionService
from app.utils.admin_health_report import run_health_report
from app.utils.failsafe_runner import run_job
from app.utils.idempotency_utils import make_job_lock_key
from app.utils.time_utils import today_local

logger = logging.getLogger(ServiceName.SCHEDULER)

# ---------------------------------------------------------------------------
# Job lock TTLs in seconds
# ---------------------------------------------------------------------------
_STANDARD_LOCK_TTL: int = 15 * 60    # 15 minutes
_REPORT_LOCK_TTL:   int = 30 * 60    # 30 minutes — reports take longer
_EXPIRY_LOCK_TTL:   int = 30 * 60    # 30 minutes

# ---------------------------------------------------------------------------
# Timezone for all cron schedules
# ---------------------------------------------------------------------------
_SCHEDULER_TIMEZONE: str = "UTC"


# ==============================================================================
# Scheduler context — injected dependencies
# ==============================================================================

@dataclass
class SchedulerContext:
    """
    Container for all dependencies injected into scheduled jobs.

    Passed to each job function so jobs remain pure functions with
    explicit dependencies — no global singletons inside job modules.
    """
    business_repo:          BusinessRepository
    subscription_repo:      SubscriptionRepository
    usage_repo:             UsageRepository
    subscription_service:   SubscriptionService
    admin_notification:     AdminNotificationService
    # Additional service dependencies are added as the scheduler
    # grows. Each job function declares only what it needs.


# ==============================================================================
# Scheduler Manager
# ==============================================================================

class SchedulerManager:
    """
    Central controller for all APScheduler background jobs.

    Instantiated once at application startup. Registers all jobs and
    manages the AsyncIOScheduler lifecycle.

    Usage (from FastAPI lifespan):

        scheduler = SchedulerManager(context=scheduler_context)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            scheduler.start()
            yield
            scheduler.shutdown()
    """

    def __init__(self, context: SchedulerContext) -> None:
        self._ctx = context
        self._scheduler = AsyncIOScheduler(timezone=_SCHEDULER_TIMEZONE)
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Register all jobs and start the scheduler.

        Called from FastAPI lifespan startup. Jobs begin executing
        immediately according to their configured schedules.
        """
        if self._running:
            logger.warning(
                "Scheduler already running — start() called twice",
                extra={"service": ServiceName.SCHEDULER},
            )
            return

        self._register_all_jobs()
        self._scheduler.start()
        self._running = True

        logger.info(
            "Scheduler started — %d jobs registered",
            len(self._scheduler.get_jobs()),
            extra={"service": ServiceName.SCHEDULER},
        )

    def shutdown(self) -> None:
        """
        Gracefully shut down the scheduler.

        Waits for currently running jobs to complete before stopping.
        Called from FastAPI lifespan shutdown.
        """
        if not self._running:
            return

        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info(
            "Scheduler shut down cleanly",
            extra={"service": ServiceName.SCHEDULER},
        )

    @property
    def is_running(self) -> bool:
        """Return True if the scheduler is active."""
        return self._running

    def get_job_count(self) -> int:
        """Return the number of registered jobs."""
        return len(self._scheduler.get_jobs())

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_all_jobs(self) -> None:
        """Register all background jobs with the APScheduler instance."""

        # ── Review monitor — every 30 minutes ──────────────────────────
        _settings = get_settings()
        self._scheduler.add_job(
            func=self._job_review_monitor,
            trigger=IntervalTrigger(minutes=_settings.SCHEDULER_REVIEW_POLL_INTERVAL_MINUTES),
            id=JobName.REVIEW_MONITOR,
            name="Google Reviews Monitor",
            replace_existing=True,
            max_instances=1,    # APScheduler-level guard: never run twice
        )

        # ── Daily sales analysis — 01:00 UTC ───────────────────────────
        self._scheduler.add_job(
            func=self._job_sales_analysis,
            trigger=CronTrigger(hour=1, minute=0),
            id=JobName.SALES_ANALYSIS,
            name="Daily Sales Analysis",
            replace_existing=True,
            max_instances=1,
        )

        # ── Weekly report — Monday 06:00 UTC ───────────────────────────
        self._scheduler.add_job(
            func=self._job_weekly_report,
            trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
            id=JobName.WEEKLY_REPORT,
            name="Weekly Performance Report",
            replace_existing=True,
            max_instances=1,
        )

        # ── Weekly content — Monday 07:00 UTC ──────────────────────────
        self._scheduler.add_job(
            func=self._job_weekly_content,
            trigger=CronTrigger(day_of_week="mon", hour=7, minute=0),
            id=JobName.WEEKLY_CONTENT,
            name="Weekly Social Media Content",
            replace_existing=True,
            max_instances=1,
        )

        # ── Monthly report — 1st of month 08:00 UTC ────────────────────
        self._scheduler.add_job(
            func=self._job_monthly_report,
            trigger=CronTrigger(day=1, hour=8, minute=0),
            id=JobName.MONTHLY_REPORT,
            name="Monthly Business Report",
            replace_existing=True,
            max_instances=1,
        )

        # ── Quarterly report — 1st of Jan/Apr/Jul/Oct 09:00 UTC ────────
        self._scheduler.add_job(
            func=self._job_quarterly_report,
            trigger=CronTrigger(month="1,4,7,10", day=1, hour=9, minute=0),
            id=JobName.QUARTERLY_REPORT,
            name="Quarterly Strategic Report",
            replace_existing=True,
            max_instances=1,
        )

        # ── Subscription expiry check — daily 00:05 UTC ────────────────
        self._scheduler.add_job(
            func=self._job_expiry_check,
            trigger=CronTrigger(hour=0, minute=5),
            id=JobName.EXPIRY_CHECK,
            name="Subscription Expiry Checker",
            replace_existing=True,
            max_instances=1,
        )

        # ── System health report — daily 08:00 UTC ─────────────────────
        self._scheduler.add_job(
            func=self._job_health_report,
            trigger=CronTrigger(hour=8, minute=0),
            id=JobName.HEALTH_REPORT,
            name="System Health Report",
            replace_existing=True,
            max_instances=1,
        )

        logger.info(
            "All jobs registered",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_count": len(self._scheduler.get_jobs()),
            },
        )

    # ------------------------------------------------------------------
    # Job wrapper functions
    # Each wrapper:
    #   1. Opens a database session
    #   2. Acquires a job lock (skip if held)
    #   3. Delegates to the job module
    #   4. Releases the lock
    #   5. Handles all exceptions — never propagates
    # ------------------------------------------------------------------

    async def _job_review_monitor(self) -> None:
        """Wrapper: Google Reviews Monitor (every 30 min)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.REVIEW_MONITOR,
                lock_ttl=_STANDARD_LOCK_TTL,
                job_fn=lambda: run_review_monitor(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_sales_analysis(self) -> None:
        """Wrapper: Daily Sales Analysis (01:00 UTC)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.SALES_ANALYSIS,
                lock_ttl=_STANDARD_LOCK_TTL,
                job_fn=lambda: run_sales_analysis(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    usage_repo=self._ctx.usage_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_weekly_report(self) -> None:
        """Wrapper: Weekly Performance Report (Mon 06:00 UTC)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.WEEKLY_REPORT,
                lock_ttl=_REPORT_LOCK_TTL,
                job_fn=lambda: run_weekly_report(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_weekly_content(self) -> None:
        """Wrapper: Weekly Social Media Content (Mon 07:00 UTC)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.WEEKLY_CONTENT,
                lock_ttl=_STANDARD_LOCK_TTL,
                job_fn=lambda: run_weekly_content(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_monthly_report(self) -> None:
        """Wrapper: Monthly Business Report (1st 08:00 UTC)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.MONTHLY_REPORT,
                lock_ttl=_REPORT_LOCK_TTL,
                job_fn=lambda: run_monthly_report(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_quarterly_report(self) -> None:
        """Wrapper: Quarterly Strategic Report (1st Jan/Apr/Jul/Oct 09:00 UTC)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.QUARTERLY_REPORT,
                lock_ttl=_REPORT_LOCK_TTL,
                job_fn=lambda: run_quarterly_report(
                    db=db,
                    business_repo=self._ctx.business_repo,
                    subscription_repo=self._ctx.subscription_repo,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_expiry_check(self) -> None:
        """Wrapper: Subscription Expiry Check (00:05 UTC daily)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.EXPIRY_CHECK,
                lock_ttl=_EXPIRY_LOCK_TTL,
                job_fn=lambda: run_expiry_check(
                    db=db,
                    subscription_service=self._ctx.subscription_service,
                    job_repo=None,          # lock managed here at scheduler level
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )

    async def _job_health_report(self) -> None:
        """Wrapper: System Health Report (08:00 UTC daily)."""
        async with _db_session() as db:
            await _run_with_lock(
                db=db,
                job_name=JobName.HEALTH_REPORT,
                lock_ttl=_STANDARD_LOCK_TTL,
                job_fn=lambda: run_health_report(
                    db=db,
                    admin_notification=self._ctx.admin_notification,
                ),
                admin=self._ctx.admin_notification,
            )


# ==============================================================================
# Module-level helpers
# ==============================================================================

async def _run_with_lock(
    db: AsyncSession,
    job_name: str,
    lock_ttl: int,
    job_fn,
    admin: AdminNotificationService,
) -> None:
    """
    Acquire a date-scoped job lock, execute the job, then release the lock.

    Lock key format: JOB_LOCK_{JOB_NAME}_{ISO_DATE}
    Using the date as part of the key ensures:
      - Duplicate runs on the same day are blocked
      - The job runs fresh the next day without stale lock cleanup

    Lock acquisition failure → run anyway (see expiry_checker.py rationale).
    Job execution failure    → log + admin alert, never propagate.
    Lock release failure     → log warning, TTL is the backstop.

    Args:
        db:        AsyncSession for lock persistence.
        job_name:  JobName constant — used in lock key and log context.
        lock_ttl:  Lock TTL in seconds.
        job_fn:    Async callable — the actual job coroutine factory.
        admin:     AdminNotificationService for failure alerting.
    """
    today = today_local().isoformat()
    lock_key = make_job_lock_key(job_name=job_name, date_str=today)
    log_extra = {
        "service": ServiceName.SCHEDULER,
        "job": job_name,
        "lock_key": lock_key,
    }

    # ── Acquire lock ──────────────────────────────────────────────────
    lock_acquired = await _try_acquire_db_lock(
        db=db,
        lock_key=lock_key,
        job_name=job_name,
        ttl_seconds=lock_ttl,
        log_extra=log_extra,
    )

    if not lock_acquired:
        logger.info(
            "Job skipped — lock already held",
            extra=log_extra,
        )
        return

    # ── Execute with failsafe ─────────────────────────────────────────
    logger.info("Job starting", extra=log_extra)

    async with run_job(job_type=job_name) as result:
        await job_fn()

    if result.failed:
        logger.error(
            "Job failed",
            extra={**log_extra, "error": result.error_message},
        )
        await admin.send_job_failure(
            job_name=job_name,
            error=result.error_message or "unknown error",
        )
    else:
        logger.info("Job completed successfully", extra=log_extra)

    # ── Release lock ──────────────────────────────────────────────────
    await _release_db_lock(
        db=db,
        lock_key=lock_key,
        log_extra=log_extra,
    )


async def _try_acquire_db_lock(
    db: AsyncSession,
    lock_key: str,
    job_name: str,
    ttl_seconds: int,
    log_extra: dict,
) -> bool:
    """
    Attempt to insert a job lock record into the database.

    Uses INSERT ... ON CONFLICT DO NOTHING — if the key already exists,
    the insert is silently skipped and we return False (lock held).
    If the key doesn't exist, the insert succeeds and we return True.

    Expired locks (created_at + ttl < now) are treated as not held —
    a separate cleanup sweep runs on startup to clear stale locks.

    Returns True on acquisition failure (fail-open) so the job still
    runs if the database itself is unavailable — missing expirations
    is worse than a duplicate run for idempotent jobs.
    """
    from app.database.models.job_model import JobLockModel
    from app.utils.time_utils import now_utc
    from datetime import timedelta
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    try:
        expires_at = now_utc() + timedelta(seconds=ttl_seconds)

        stmt = (
            pg_insert(JobLockModel)
            .values(
                lock_key=lock_key,
                job_name=job_name,
                expires_at=expires_at,
            )
            .on_conflict_do_nothing(index_elements=["lock_key"])
        )
        result = await db.execute(stmt)
        await db.commit()

        # rowcount == 1 means the insert succeeded → lock acquired
        acquired = result.rowcount == 1
        return acquired

    except Exception as exc:
        logger.error(
            "Lock acquisition failed — proceeding without lock",
            extra={**log_extra, "error": str(exc)},
        )
        return True   # fail-open: run the job rather than silently skip


async def _release_db_lock(
    db: AsyncSession,
    lock_key: str,
    log_extra: dict,
) -> None:
    """
    Delete the job lock record from the database.

    Failure is logged as WARNING only — the TTL acts as the backstop
    to prevent a lock from being permanently held after a process crash.
    """
    from app.database.models.job_model import JobLockModel
    from sqlalchemy import delete

    try:
        await db.execute(
            delete(JobLockModel).where(JobLockModel.lock_key == lock_key)
        )
        await db.commit()
    except Exception as exc:
        logger.warning(
            "Lock release failed — TTL will clear it",
            extra={**log_extra, "error": str(exc), "ttl_note": "lock expires automatically"},
        )


def _db_session():
    """
    Return a context manager that yields an AsyncSession.

    Uses get_session() from app/database/db.py (for use outside FastAPI
    request lifecycle). Each job gets its own isolated session.
    """
    return get_session()