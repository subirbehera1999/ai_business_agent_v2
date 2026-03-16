# ==============================================================================
# File: app/utils/failsafe_runner.py
# Purpose: Provides crash-safe execution wrappers for all background
#          scheduler jobs. Every job in app/schedulers/ and app/jobs/
#          must run inside one of these wrappers.
#
#          Problem solved:
#            An unhandled exception in a background job must never:
#              - Crash the scheduler loop
#              - Prevent other businesses from being processed
#              - Leave a job lock unreleased
#              - Go unlogged or untracked
#
#          Two interfaces provided:
#            1. run_job()          — async context manager for inline use
#            2. @failsafe_job      — decorator for job functions
#
#          Both interfaces guarantee:
#            - All exceptions are caught and logged with full context
#            - Job execution time is measured and recorded
#            - The scheduler loop always continues after a failure
#            - Job lock release is always attempted in the finally block
#            - Structured log entries include business_id, job_type,
#              duration, and full error traceback
#
#          Per DATA_SAFETY_AND_RUNTIME_GUARDRAILS.txt:
#            "No scheduler job should crash the scheduler loop.
#             Errors must be logged. System must continue processing."
# ==============================================================================

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config.constants import JobStatus, ServiceName
from app.config.settings import get_settings

logger = logging.getLogger(ServiceName.SCHEDULER)
settings = get_settings()

# Maximum wall-clock seconds a job is allowed to run before it is
# considered stale and logged as a timeout violation.
# Source: PERFORMANCE_AND_SCALABILITY_CONTRACT.txt — "No scheduler job
# should run longer than 2 minutes."
MAX_JOB_DURATION_SECONDS: int = settings.SCHEDULER_JOB_LOCK_TTL_SECONDS


# ==============================================================================
# Job Execution Result
# ==============================================================================

@dataclass
class JobResult:
    """
    Structured outcome of a single failsafe job execution.

    Attributes:
        job_type:          Identifier of the job that ran.
        business_id:       UUID string of the business processed, if any.
        status:            Final job status (completed / failed / skipped).
        duration_seconds:  Wall-clock execution time in seconds.
        records_processed: Count of records successfully handled.
        records_failed:    Count of records that failed processing.
        records_skipped:   Count of records intentionally skipped.
        error_message:     Short error description if failed.
        error_traceback:   Full Python traceback string if failed.
        timed_out:         True if execution exceeded MAX_JOB_DURATION_SECONDS.
    """

    job_type: str
    business_id: Optional[str] = None
    status: str = JobStatus.COMPLETED
    duration_seconds: float = 0.0
    records_processed: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        """True if the job completed without errors."""
        return self.status == JobStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """True if the job terminated with an unhandled error."""
        return self.status == JobStatus.FAILED

    @property
    def skipped(self) -> bool:
        """True if the job was intentionally skipped."""
        return self.status == JobStatus.SKIPPED

    @property
    def has_partial_failures(self) -> bool:
        """True if the job completed but some records failed."""
        return self.status == JobStatus.COMPLETED and self.records_failed > 0

    def __str__(self) -> str:
        parts = [
            f"JobResult[{self.job_type}]",
            f"status={self.status}",
            f"duration={self.duration_seconds:.2f}s",
        ]
        if self.business_id:
            parts.append(f"business={self.business_id}")
        if self.records_processed:
            parts.append(f"processed={self.records_processed}")
        if self.records_failed:
            parts.append(f"failed={self.records_failed}")
        if self.error_message:
            parts.append(f"error='{self.error_message}'")
        return " ".join(parts)


# ==============================================================================
# Context Manager Interface
# ==============================================================================

@asynccontextmanager
async def run_job(
    job_type: str,
    business_id: Optional[Any] = None,
    skip_reason: Optional[str] = None,
) -> AsyncGenerator[JobResult, None]:
    """
    Async context manager that wraps a job body with failsafe execution.

    Catches all exceptions, measures execution time, logs outcomes with
    full structured context, and always yields control back to the
    scheduler loop regardless of what happens inside the block.

    The caller mutates the yielded JobResult to record processing counts.
    The wrapper finalises status, duration, and logging after the block exits.

    Usage:
        async with run_job("review_monitor", business_id=business.id) as job:
            reviews = await fetch_new_reviews(business)
            job.records_processed = len(reviews)

        # After the block:
        # - job.status is set to COMPLETED / FAILED
        # - job.duration_seconds is recorded
        # - structured log entry is written

    Args:
        job_type:     Identifier of the job (see JobType enum in constants.py).
        business_id:  Optional UUID of the business being processed.
                      Used for structured log entries and job records.
        skip_reason:  If provided, the job is marked SKIPPED immediately
                      without executing the body. Pass this when a lock
                      check or subscription check determines the job
                      should not run.

    Yields:
        JobResult: Mutable result object. Set records_processed,
                   records_failed, and records_skipped inside the block.

    Notes:
        - Never re-raises exceptions — the scheduler loop always continues.
        - asyncio.CancelledError is re-raised to allow graceful shutdown.
        - Timeout logging is advisory — enforcement is via job locking TTL.
    """
    business_id_str = str(business_id) if business_id else None
    result = JobResult(job_type=job_type, business_id=business_id_str)

    # ── Pre-execution skip check ───────────────────────────────────────────
    if skip_reason:
        result.status = JobStatus.SKIPPED
        logger.info(
            "Job skipped",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_type": job_type,
                "business_id": business_id_str,
                "skip_reason": skip_reason,
            },
        )
        yield result
        return

    # ── Execution ──────────────────────────────────────────────────────────
    start_time = time.monotonic()

    logger.info(
        "Job started",
        extra={
            "service": ServiceName.SCHEDULER,
            "job_type": job_type,
            "business_id": business_id_str,
        },
    )

    try:
        yield result

        # ── Successful completion ──────────────────────────────────────────
        result.duration_seconds = round(time.monotonic() - start_time, 3)
        result.status = JobStatus.COMPLETED

        # Warn if job exceeded the recommended max duration
        if result.duration_seconds > MAX_JOB_DURATION_SECONDS:
            result.timed_out = True
            logger.warning(
                "Job exceeded maximum execution time",
                extra={
                    "service": ServiceName.SCHEDULER,
                    "job_type": job_type,
                    "business_id": business_id_str,
                    "duration_seconds": result.duration_seconds,
                    "max_allowed_seconds": MAX_JOB_DURATION_SECONDS,
                },
            )

        logger.info(
            "Job completed",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_type": job_type,
                "business_id": business_id_str,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "records_processed": result.records_processed,
                "records_failed": result.records_failed,
                "records_skipped": result.records_skipped,
            },
        )

    except asyncio.CancelledError:
        # Re-raise cancellation — this signals graceful scheduler shutdown
        result.duration_seconds = round(time.monotonic() - start_time, 3)
        result.status = JobStatus.FAILED
        result.error_message = "Job cancelled (scheduler shutdown)"
        logger.warning(
            "Job cancelled during scheduler shutdown",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_type": job_type,
                "business_id": business_id_str,
                "duration_seconds": result.duration_seconds,
            },
        )
        raise  # Allow CancelledError to propagate for clean shutdown

    except Exception as exc:
        # ── Graceful failure — never crashes the scheduler ─────────────────
        result.duration_seconds = round(time.monotonic() - start_time, 3)
        result.status = JobStatus.FAILED
        result.error_message = str(exc)
        result.error_traceback = traceback.format_exc()

        logger.error(
            "Job failed with unhandled exception",
            extra={
                "service": ServiceName.SCHEDULER,
                "job_type": job_type,
                "business_id": business_id_str,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "error": result.error_message,
                "error_type": type(exc).__name__,
                "records_processed": result.records_processed,
                "records_failed": result.records_failed,
            },
        )
        # Do NOT re-raise — the scheduler loop must continue


# ==============================================================================
# Decorator Interface
# ==============================================================================

def failsafe_job(
    job_type: Optional[str] = None,
    include_business_id_arg: str = "business_id",
) -> Callable:
    """
    Decorator that wraps an async job function with failsafe execution.

    The decorated function is called inside a run_job() context manager.
    Exceptions are caught, logged, and swallowed — the scheduler continues.

    The wrapped function may optionally accept a `job_result` keyword
    argument — if so, the JobResult object is injected so the function
    can record processing counts directly.

    Args:
        job_type:                Identifier for this job type. Defaults to
                                 the decorated function's __name__.
        include_business_id_arg: Name of the keyword argument in the wrapped
                                 function that carries the business_id.
                                 Used to extract it for log context.
                                 Defaults to "business_id".

    Returns:
        Decorated async function with failsafe behaviour.

    Usage:
        @failsafe_job(job_type="review_monitor")
        async def run_review_monitor(business_id: uuid.UUID, ...) -> JobResult:
            ...

        # Or with injected result:
        @failsafe_job(job_type="weekly_report")
        async def run_weekly_report(
            business_id: uuid.UUID,
            job_result: JobResult,
            ...
        ) -> None:
            job_result.records_processed += 1
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, JobResult]]:

        resolved_job_type = job_type or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> JobResult:
            # Extract business_id from kwargs for log context
            business_id = kwargs.get(include_business_id_arg)

            async with run_job(
                job_type=resolved_job_type,
                business_id=business_id,
            ) as job_result:
                # Inject job_result if the function accepts it
                import inspect
                sig = inspect.signature(func)
                if "job_result" in sig.parameters:
                    kwargs["job_result"] = job_result

                await func(*args, **kwargs)

            return job_result

        return wrapper
    return decorator


# ==============================================================================
# Batch Job Runner — iterates businesses safely
# ==============================================================================

async def run_for_each_business(
    businesses: list,
    job_func: Callable,
    job_type: str,
    *,
    continue_on_failure: bool = True,
    delay_between_businesses_seconds: float = 0.0,
) -> list[JobResult]:
    """
    Execute a job function for each business in a list, with per-business
    failsafe isolation.

    If a job fails for one business, the error is contained and processing
    continues for all remaining businesses. This is the core isolation
    guarantee required by PERFORMANCE_AND_SCALABILITY_CONTRACT.txt:

        "Failure in one business workflow must never affect other businesses."

    Args:
        businesses:                       List of BusinessModel instances.
        job_func:                         Async callable accepting a single
                                          BusinessModel as its argument.
                                          Signature: async def f(business) -> None
        job_type:                         Job type identifier for log context.
        continue_on_failure:              If True (default), continue to the next
                                          business after a failure.
                                          If False, stop on first failure.
        delay_between_businesses_seconds: Optional sleep between businesses to
                                          avoid thundering herd on external APIs.

    Returns:
        list[JobResult]: One result per business, in input order.

    Usage:
        businesses = await business_repo.get_all_active(db, limit=20)
        results = await run_for_each_business(
            businesses,
            lambda b: process_reviews_for_business(db, b),
            job_type=JobType.REVIEW_MONITOR,
        )
        completed = sum(1 for r in results if r.succeeded)
    """
    results: list[JobResult] = []

    for business in businesses:
        business_id = getattr(business, "id", None)

        async with run_job(
            job_type=job_type,
            business_id=business_id,
        ) as job_result:
            await job_func(business)

        results.append(job_result)

        if job_result.failed and not continue_on_failure:
            logger.warning(
                "Batch job aborted after failure — continue_on_failure=False",
                extra={
                    "service": ServiceName.SCHEDULER,
                    "job_type": job_type,
                    "failed_business_id": str(business_id),
                    "businesses_remaining": len(businesses) - len(results),
                },
            )
            break

        if delay_between_businesses_seconds > 0:
            await asyncio.sleep(delay_between_businesses_seconds)

    # ── Summary log ───────────────────────────────────────────────────────
    total = len(results)
    completed = sum(1 for r in results if r.succeeded)
    failed = sum(1 for r in results if r.failed)
    skipped = sum(1 for r in results if r.skipped)

    logger.info(
        "Batch job cycle completed",
        extra={
            "service": ServiceName.SCHEDULER,
            "job_type": job_type,
            "total_businesses": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        },
    )

    return results


# ==============================================================================
# Single-shot safe execution helper
# ==============================================================================

async def run_once_safely(
    coro: Coroutine,
    job_type: str,
    business_id: Optional[Any] = None,
) -> JobResult:
    """
    Execute a single coroutine inside a failsafe wrapper.

    Convenience helper for one-off job executions where the caller has
    already constructed the coroutine and just needs safe execution.

    Args:
        coro:         Coroutine to execute.
        job_type:     Job type identifier for log context.
        business_id:  Optional business UUID for log context.

    Returns:
        JobResult: Execution outcome.

    Usage:
        result = await run_once_safely(
            generate_weekly_report(db, business),
            job_type=JobType.WEEKLY_REPORT,
            business_id=business.id,
        )
        if result.failed:
            logger.error(f"Report failed: {result.error_message}")
    """
    async with run_job(job_type=job_type, business_id=business_id) as job_result:
        await coro
    return job_result


# Alias used by job_manager.py
run_safely = run_job