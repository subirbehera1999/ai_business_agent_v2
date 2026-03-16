# ==============================================================================
# File: app/jobs/job_manager.py
# Purpose: Central job execution infrastructure shared by all job modules
#          in app/jobs/. Provides the execution context, database session
#          management, and shared utilities that review_jobs.py,
#          report_jobs.py, and content_jobs.py depend on.
#
#          What this file provides:
#
#            JobContext — dataclass carrying all dependencies a job task needs:
#              - db session factory
#              - service instances (ai_reply, sentiment, reports, content)
#              - repository instances (review, usage, business, subscription)
#              - whatsapp_service for alert delivery
#              - rate_limiter reference
#
#            JobResult — structured return type for all job task functions:
#              - success: bool
#              - processed: int (records successfully handled)
#              - skipped: int (records skipped — rate limit, already done, etc.)
#              - failed: int (records that errored)
#              - errors: list[str] (error messages for logging)
#              - duration_seconds: float
#
#            run_job_for_business() — the standard execution wrapper:
#              - Accepts a coroutine factory (job function + business)
#              - Opens a dedicated db session per business
#              - Wraps execution in failsafe_runner
#              - Commits on success, rolls back on failure
#              - Logs structured result
#              - Returns JobResult
#
#            run_jobs_for_all_businesses() — batch runner:
#              - Loads all active businesses in controlled batches (20 per cycle)
#              - Calls run_job_for_business() for each
#              - Enforces failure isolation: one business failure never
#                blocks the rest
#              - Returns aggregate JobResult
#
#          Architecture contract:
#            - Schedulers call job modules (review_jobs.py, etc.)
#            - Job modules call run_job_for_business() from this file
#            - Job modules contain the actual task logic
#            - This file never contains task logic — only execution infrastructure
#
#          Multi-tenancy guarantee:
#            Each business runs in its own database session with its own
#            commit/rollback cycle. Business A's failure never rolls back
#            Business B's work.
#
#          Session lifecycle:
#            - Session is opened by run_job_for_business()
#            - Session is passed to job task via JobContext
#            - Session is committed after job task returns successfully
#            - Session is rolled back if job task raises
#            - Session is always closed in the finally block
# ==============================================================================

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.db import AsyncSessionFactory
from app.repositories.business_repository import BusinessRepository
from app.repositories.review_repository import ReviewRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository
from app.services.ai_reply_service import AIReplyService
from app.services.content_generation_service import ContentGenerationService
from app.services.reports_service import ReportsService
from app.services.sentiment_service import SentimentService
from app.notifications.whatsapp_service import WhatsAppService
from app.utils.failsafe_runner import run_safely

logger = logging.getLogger(ServiceName.SCHEDULER)


# ==============================================================================
# Job Result — structured return from every job task
# ==============================================================================

@dataclass
class JobResult:
    """
    Structured result returned by every job task function and batch runner.

    Aggregated by run_jobs_for_all_businesses() across all businesses.
    Logged by run_job_for_business() after each task completes.

    Attributes:
        success:          True if the job completed without a fatal error.
                          Individual record failures do not set this to False
                          — only exceptions that aborted the entire job do.
        processed:        Number of records successfully processed.
        skipped:          Number of records skipped (rate limit, already done,
                          invalid, etc.). Not an error — expected behaviour.
        failed:           Number of records that failed processing.
        errors:           List of error message strings for log context.
        duration_seconds: Wall clock time for this job execution.
        business_id:      UUID string of the business this result is for.
                          Empty string for aggregate results.
        job_name:         Name of the job that produced this result.
    """

    success: bool = True
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    business_id: str = ""
    job_name: str = ""

    def add_error(self, message: str) -> None:
        """Append an error message and increment the failed counter."""
        self.errors.append(message)
        self.failed += 1

    def merge(self, other: "JobResult") -> None:
        """
        Merge another result into this one.

        Used by run_jobs_for_all_businesses() to accumulate individual
        business results into a platform-wide aggregate.

        Args:
            other: JobResult from a single business execution.
        """
        self.processed += other.processed
        self.skipped += other.skipped
        self.failed += other.failed
        self.errors.extend(other.errors)
        self.duration_seconds += other.duration_seconds
        # Aggregate success is False if any individual business failed fatally
        if not other.success:
            self.success = False

    def to_log_dict(self) -> dict:
        """Return a structured dict safe for structured logging extra= field."""
        return {
            "job_name": self.job_name,
            "business_id": self.business_id,
            "success": self.success,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "error_count": len(self.errors),
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{self.job_name}] {status} | "
            f"business={self.business_id} | "
            f"processed={self.processed} skipped={self.skipped} "
            f"failed={self.failed} | "
            f"duration={self.duration_seconds:.2f}s"
        )


# ==============================================================================
# Job Context — dependency container passed to every job task
# ==============================================================================

@dataclass
class JobContext:
    """
    Dependency container passed into every job task function.

    Carries the database session and all service/repository instances needed
    to execute the task. Built fresh for each business execution by
    run_job_for_business() and passed directly to the task coroutine.

    The db session in the context is the session for this specific business.
    It must not be shared across businesses.

    Attributes:
        db:                  AsyncSession for this business's execution.
        business_id:         UUID of the business being processed.
        ai_reply_service:    Service for AI reply generation.
        sentiment_service:   Service for review sentiment classification.
        reports_service:     Service for weekly/monthly/quarterly reports.
        content_service:     Service for social media content generation.
        review_repo:         Repository for review data operations.
        usage_repo:          Repository for usage counter operations.
        business_repo:       Repository for business profile queries.
        subscription_repo:   Repository for subscription validation.
        whatsapp_service:    Service for WhatsApp message delivery.
    """

    db: AsyncSession
    business_id: uuid.UUID

    # Services
    ai_reply_service: AIReplyService
    sentiment_service: SentimentService
    reports_service: ReportsService
    content_service: ContentGenerationService

    # Repositories
    review_repo: ReviewRepository
    usage_repo: UsageRepository
    business_repo: BusinessRepository
    subscription_repo: SubscriptionRepository

    # Delivery
    whatsapp_service: WhatsAppService


# ==============================================================================
# Shared service/repository singletons
# — stateless classes, safe to share across all job executions
# ==============================================================================

_ai_reply_service = AIReplyService()
_sentiment_service = SentimentService()
_reports_service = ReportsService()
_content_service = ContentGenerationService()

_review_repo = ReviewRepository()
_usage_repo = UsageRepository()
_business_repo = BusinessRepository()
_subscription_repo = SubscriptionRepository()

_whatsapp_service = WhatsAppService()


# ==============================================================================
# Core execution wrapper — single business
# ==============================================================================

async def run_job_for_business(
    business_id: uuid.UUID,
    job_name: str,
    task_fn: Callable[["JobContext"], Coroutine[Any, Any, JobResult]],
) -> JobResult:
    """
    Execute a single job task for one business with full lifecycle management.

    Opens a dedicated database session for this business, builds a JobContext,
    runs the task inside a failsafe wrapper, commits on success, rolls back on
    failure, and always closes the session in the finally block.

    This is the standard execution wrapper used by all job modules. Every
    job task must conform to the signature:

        async def my_task(ctx: JobContext) -> JobResult:
            ...

    Args:
        business_id: UUID of the business to process.
        job_name:    Human-readable job name for logging and result labelling.
        task_fn:     Coroutine factory — accepts a JobContext, returns JobResult.

    Returns:
        JobResult: Structured result with success flag, counts, and duration.
        Never raises — all exceptions are caught and returned as a failed result.

    Example:
        result = await run_job_for_business(
            business_id=business.id,
            job_name="process_reviews",
            task_fn=lambda ctx: process_reviews_for_business(ctx),
        )
    """
    start = time.monotonic()
    result = JobResult(
        success=True,
        business_id=str(business_id),
        job_name=job_name,
    )

    db: Optional[AsyncSession] = None

    try:
        # Open a dedicated session for this business
        async with AsyncSessionFactory() as db:
            # Build the job context
            ctx = JobContext(
                db=db,
                business_id=business_id,
                ai_reply_service=_ai_reply_service,
                sentiment_service=_sentiment_service,
                reports_service=_reports_service,
                content_service=_content_service,
                review_repo=_review_repo,
                usage_repo=_usage_repo,
                business_repo=_business_repo,
                subscription_repo=_subscription_repo,
                whatsapp_service=_whatsapp_service,
            )

            # Execute the task inside a failsafe wrapper
            task_result: Optional[JobResult] = await run_safely(
                coro=task_fn(ctx),
                job_name=job_name,
                business_id=str(business_id),
            )

            if task_result is None:
                # run_safely caught an exception — job failed
                result.success = False
                result.add_error(
                    f"Job '{job_name}' for business {business_id} "
                    f"failed with an unhandled exception."
                )
                await db.rollback()
            else:
                # Task completed — merge result and commit
                task_result.business_id = str(business_id)
                task_result.job_name = job_name
                result = task_result
                result.business_id = str(business_id)
                result.job_name = job_name

                if result.success:
                    await db.commit()
                else:
                    # Task returned a failed result — still rollback
                    await db.rollback()

    except Exception as exc:
        # Session open failure or unexpected error outside task
        result.success = False
        result.add_error(
            f"Job infrastructure failure for '{job_name}' "
            f"business={business_id}: {exc}"
        )
        logger.error(
            "Job infrastructure failure",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_name": job_name,
                "business_id": str(business_id),
                "error": str(exc),
            },
        )
        if db is not None:
            try:
                await db.rollback()
            except Exception:
                pass

    finally:
        result.duration_seconds = time.monotonic() - start

    # Log structured result
    log_level = logger.info if result.success else logger.warning
    log_level(str(result), extra={
        "service": ServiceName.SCHEDULER,
        **result.to_log_dict(),
    })

    return result


# ==============================================================================
# Batch execution wrapper — all active businesses
# ==============================================================================

async def run_jobs_for_all_businesses(
    job_name: str,
    task_fn: Callable[["JobContext"], Coroutine[Any, Any, JobResult]],
    batch_size: int = 20,
) -> JobResult:
    """
    Execute a job task for every active business in controlled batches.

    Loads active businesses from the database in batches of `batch_size`
    (default 20, per the performance contract). Calls run_job_for_business()
    for each. Failure for one business never blocks the rest.

    Uses a separate database session just for loading the business list —
    each business then gets its own session from run_job_for_business().

    Args:
        job_name:   Human-readable job name for logging.
        task_fn:    Coroutine factory for the task to run per business.
        batch_size: Maximum businesses to load per DB query cycle (default 20).

    Returns:
        JobResult: Aggregate result across all businesses.
        Never raises — all failures are captured in the result.

    Example:
        aggregate = await run_jobs_for_all_businesses(
            job_name="weekly_report",
            task_fn=lambda ctx: generate_weekly_report(ctx),
        )
    """
    aggregate = JobResult(job_name=job_name, business_id="all")
    start = time.monotonic()

    logger.info(
        f"Batch job starting: {job_name}",
        extra={
            "service": ServiceName.SCHEDULER,
            "job_name": job_name,
            "batch_size": batch_size,
        },
    )

    try:
        # Load all active business IDs in one query
        # (IDs only — lightweight, no full model hydration)
        business_ids: list[uuid.UUID] = []

        async with AsyncSessionFactory() as db:
            active_businesses = await _business_repo.get_all_active(
                db, limit=1000
            )
            business_ids = [b.id for b in active_businesses]

        if not business_ids:
            logger.info(
                f"Batch job '{job_name}': no active businesses found",
                extra={"service": ServiceName.SCHEDULER, "job_name": job_name},
            )
            aggregate.duration_seconds = time.monotonic() - start
            return aggregate

        logger.info(
            f"Batch job '{job_name}': processing {len(business_ids)} businesses",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_name": job_name,
                "total_businesses": len(business_ids),
            },
        )

        # Process in batches — never all at once
        for batch_start in range(0, len(business_ids), batch_size):
            batch = business_ids[batch_start: batch_start + batch_size]

            for business_id in batch:
                result = await run_job_for_business(
                    business_id=business_id,
                    job_name=job_name,
                    task_fn=task_fn,
                )
                aggregate.merge(result)

            # Brief yield between batches to prevent scheduler thread starvation
            await asyncio.sleep(0)

    except Exception as exc:
        aggregate.success = False
        aggregate.add_error(
            f"Batch job '{job_name}' failed during business loading: {exc}"
        )
        logger.error(
            f"Batch job infrastructure failure: {job_name}",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_name": job_name,
                "error": str(exc),
            },
        )

    finally:
        aggregate.duration_seconds = time.monotonic() - start

    # Log aggregate summary
    log_level = logger.info if aggregate.success else logger.warning
    log_level(
        f"Batch job complete: {aggregate}",
        extra={
            "service": ServiceName.SCHEDULER,
            **aggregate.to_log_dict(),
        },
    )

    return aggregate


# ==============================================================================
# Convenience builder — create empty result for a known job
# ==============================================================================

def make_result(job_name: str, business_id: str = "") -> JobResult:
    """
    Create a fresh JobResult pre-populated with job and business identifiers.

    Used by job task functions at the start of execution to have a result
    object ready to populate as processing proceeds.

    Args:
        job_name:    Name of the job.
        business_id: UUID string of the business (optional for batch results).

    Returns:
        JobResult: Fresh result with success=True and all counters at 0.
    """
    return JobResult(
        success=True,
        job_name=job_name,
        business_id=business_id,
    )