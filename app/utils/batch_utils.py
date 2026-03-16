# ==============================================================================
# File: app/utils/batch_utils.py
# Purpose: Provides safe batch processing utilities for all large-scale
#          operations in the system. Every bulk operation — review scanning,
#          competitor data collection, analytics recalculation, report
#          generation — must process records in controlled batches.
#
#          Rules from PERFORMANCE_AND_SCALABILITY_CONTRACT.txt:
#            "Heavy workloads must always be processed in batches."
#            "Recommended batch size: 20–50 records per batch."
#            "Large datasets must never be loaded entirely into memory."
#
#          Rules from DATA_SAFETY_AND_RUNTIME_GUARDRAILS.txt:
#            "Large operations must be processed in batches."
#            "Recommended batch size: 20–50 records per batch."
#
#          Utilities provided:
#            - chunk()              Split any list into fixed-size chunks
#            - paginate_query()     Async generator over paginated DB queries
#            - BatchProcessor       Class-based processor with progress tracking
#            - process_in_batches() Functional batch runner with callbacks
#            - safe_batch_size()    Clamp a requested size to safe bounds
# ==============================================================================

import asyncio
import logging
import math
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.config.settings import get_settings

logger = logging.getLogger(ServiceName.API)
settings = get_settings()

T = TypeVar("T")

# System-wide batch size bounds from the guardrails contract
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 50
DEFAULT_BATCH_SIZE: int = settings.BATCH_SIZE_RECORDS       # from .env (default 50)
DEFAULT_BUSINESS_BATCH_SIZE: int = settings.BATCH_SIZE_BUSINESSES  # from .env (default 20)


# ==============================================================================
# Batch Size Guard
# ==============================================================================

def safe_batch_size(
    requested: int,
    min_size: int = MIN_BATCH_SIZE,
    max_size: int = MAX_BATCH_SIZE,
) -> int:
    """
    Clamp a requested batch size to the safe operating range.

    Prevents callers from accidentally using batch sizes that are either
    too small (causing excessive round-trips) or too large (causing memory
    pressure and scheduler timeouts).

    Args:
        requested: The batch size the caller wants to use.
        min_size:  Minimum allowed batch size (default: 1).
        max_size:  Maximum allowed batch size (default: 50).

    Returns:
        int: The clamped batch size within [min_size, max_size].

    Example:
        size = safe_batch_size(200)   # → 50
        size = safe_batch_size(0)     # → 1
        size = safe_batch_size(20)    # → 20
    """
    clamped = max(min_size, min(requested, max_size))
    if clamped != requested:
        logger.debug(
            "Batch size clamped to safe range",
            extra={
                "service": ServiceName.API,
                "requested": requested,
                "clamped": clamped,
                "min_size": min_size,
                "max_size": max_size,
            },
        )
    return clamped


# ==============================================================================
# List Chunking
# ==============================================================================

def chunk(
    items: list[T],
    size: int = DEFAULT_BATCH_SIZE,
) -> list[list[T]]:
    """
    Split a list into fixed-size chunks.

    The last chunk may be smaller than size if the list length is not
    evenly divisible. Never returns empty chunks.

    Args:
        items: The list to split.
        size:  Maximum number of items per chunk. Clamped to safe range.

    Returns:
        list[list[T]]: List of chunks, each containing at most size items.

    Example:
        chunks = chunk([1, 2, 3, 4, 5], size=2)
        # → [[1, 2], [3, 4], [5]]
    """
    effective_size = safe_batch_size(size)
    if not items:
        return []
    return [
        items[i: i + effective_size]
        for i in range(0, len(items), effective_size)
    ]


def chunk_count(total: int, size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Calculate how many batches are needed to process a total number of items.

    Args:
        total: Total number of items.
        size:  Batch size (clamped to safe range).

    Returns:
        int: Number of batches required (0 if total is 0).

    Example:
        n = chunk_count(105, size=50)  # → 3
    """
    effective_size = safe_batch_size(size)
    if total <= 0 or effective_size <= 0:
        return 0
    return math.ceil(total / effective_size)


# ==============================================================================
# Async Paginated Query Generator
# ==============================================================================

async def paginate_query(
    fetch_func: Callable[[int, int], Coroutine[Any, Any, list[T]]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: Optional[int] = None,
    delay_between_batches: float = 0.0,
) -> AsyncGenerator[list[T], None]:
    """
    Async generator that yields paginated batches from a database query.

    Calls fetch_func(limit, offset) repeatedly, incrementing offset by
    batch_size each iteration. Stops when the fetch returns fewer records
    than batch_size (indicating the last page) or when max_batches is reached.

    This ensures large datasets are never loaded entirely into memory.

    Args:
        fetch_func:             Async callable with signature
                                (limit: int, offset: int) -> list[T].
                                Must be a bound method from a repository.
        batch_size:             Records per page (clamped to safe range).
        max_batches:            Optional hard stop on number of batches.
                                Prevents runaway loops on unexpectedly large tables.
        delay_between_batches:  Optional sleep in seconds between batches
                                to throttle external API pressure.

    Yields:
        list[T]: Each batch of records from the paginated query.

    Example:
        async for batch in paginate_query(
            lambda limit, offset: review_repo.get_recent_by_business(
                db, business_id, limit=limit, offset=offset
            ),
            batch_size=50,
        ):
            for review in batch:
                await process(review)
    """
    effective_size = safe_batch_size(batch_size)
    offset = 0
    batch_number = 0

    while True:
        if max_batches is not None and batch_number >= max_batches:
            logger.debug(
                "paginate_query reached max_batches limit",
                extra={
                    "service": ServiceName.API,
                    "batch_number": batch_number,
                    "max_batches": max_batches,
                    "offset": offset,
                },
            )
            break

        batch = await fetch_func(effective_size, offset)

        if not batch:
            break

        logger.debug(
            "paginate_query yielding batch",
            extra={
                "service": ServiceName.API,
                "batch_number": batch_number + 1,
                "batch_size": len(batch),
                "offset": offset,
            },
        )

        yield batch

        batch_number += 1
        offset += len(batch)

        # Stop if this was the last page
        if len(batch) < effective_size:
            break

        if delay_between_batches > 0:
            await asyncio.sleep(delay_between_batches)


# ==============================================================================
# Batch Processing Result
# ==============================================================================

@dataclass
class BatchResult:
    """
    Aggregated outcome of a batch processing run.

    Attributes:
        total_items:      Total records passed into the processor.
        processed:        Records successfully handled.
        failed:           Records that raised an exception during processing.
        skipped:          Records intentionally skipped (e.g., already done).
        total_batches:    Number of batches executed.
        errors:           List of (item_index, error_message) for failures.
    """

    total_items: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    total_batches: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Ratio of processed records to total attempted (0.0–1.0)."""
        attempted = self.processed + self.failed
        if attempted == 0:
            return 0.0
        return round(self.processed / attempted, 4)

    @property
    def has_failures(self) -> bool:
        """True if any records failed during processing."""
        return self.failed > 0

    @property
    def all_succeeded(self) -> bool:
        """True if every record was processed without failure."""
        return self.failed == 0 and self.processed == self.total_items

    def __str__(self) -> str:
        return (
            f"BatchResult("
            f"total={self.total_items} "
            f"processed={self.processed} "
            f"failed={self.failed} "
            f"skipped={self.skipped} "
            f"batches={self.total_batches} "
            f"success_rate={self.success_rate:.1%})"
        )


# ==============================================================================
# Functional Batch Runner
# ==============================================================================

async def process_in_batches(
    items: list[T],
    handler: Callable[[T], Coroutine[Any, Any, None]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    continue_on_error: bool = True,
    delay_between_batches: float = 0.0,
    delay_between_items: float = 0.0,
    service_name: str = ServiceName.API,
    operation_name: str = "batch_operation",
) -> BatchResult:
    """
    Process a list of items in safe batches, calling handler for each.

    Each item is processed individually inside each batch. Exceptions from
    handler are caught per-item — a single item failure does not abort the
    batch or the run (when continue_on_error=True).

    Args:
        items:                    List of items to process.
        handler:                  Async callable to invoke per item.
                                  Signature: async def handler(item: T) -> None
        batch_size:               Items per batch (clamped to safe range).
        continue_on_error:        If True (default), log errors and continue.
                                  If False, abort on first item failure.
        delay_between_batches:    Sleep in seconds between batches.
        delay_between_items:      Sleep in seconds between individual items.
                                  Use for API-rate-sensitive operations.
        service_name:             Service identifier for log entries.
        operation_name:           Human-readable name for log entries.

    Returns:
        BatchResult: Aggregated processing outcome.

    Example:
        result = await process_in_batches(
            reviews,
            handler=lambda r: process_single_review(db, r),
            batch_size=20,
            operation_name="review_processing",
        )
        logger.info(f"Done: {result}")
    """
    effective_size = safe_batch_size(batch_size)
    batches = chunk(items, effective_size)
    result = BatchResult(total_items=len(items), total_batches=len(batches))

    logger.info(
        "Batch processing started",
        extra={
            "service": service_name,
            "operation": operation_name,
            "total_items": len(items),
            "batch_size": effective_size,
            "total_batches": len(batches),
        },
    )

    global_index = 0

    for batch_num, batch in enumerate(batches, start=1):
        for item in batch:
            try:
                await handler(item)
                result.processed += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append((global_index, str(exc)))
                logger.error(
                    "Batch item processing failed",
                    extra={
                        "service": service_name,
                        "operation": operation_name,
                        "batch_number": batch_num,
                        "item_index": global_index,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                if not continue_on_error:
                    logger.warning(
                        "Batch processing aborted — continue_on_error=False",
                        extra={
                            "service": service_name,
                            "operation": operation_name,
                            "aborted_at_index": global_index,
                        },
                    )
                    return result

            global_index += 1

            if delay_between_items > 0:
                await asyncio.sleep(delay_between_items)

        logger.debug(
            "Batch completed",
            extra={
                "service": service_name,
                "operation": operation_name,
                "batch_number": batch_num,
                "total_batches": len(batches),
                "running_processed": result.processed,
                "running_failed": result.failed,
            },
        )

        if delay_between_batches > 0 and batch_num < len(batches):
            await asyncio.sleep(delay_between_batches)

    logger.info(
        "Batch processing completed",
        extra={
            "service": service_name,
            "operation": operation_name,
            "total_items": result.total_items,
            "processed": result.processed,
            "failed": result.failed,
            "skipped": result.skipped,
            "success_rate": f"{result.success_rate:.1%}",
        },
    )

    return result


# ==============================================================================
# Class-Based Batch Processor — for stateful operations
# ==============================================================================

class BatchProcessor:
    """
    Stateful batch processor for operations that need shared context
    across all items in a run (e.g., a shared database session or
    a shared result accumulator).

    Usage:
        processor = BatchProcessor(
            batch_size=20,
            service_name=ServiceName.GOOGLE_REVIEWS,
            operation_name="review_pipeline",
        )

        async for batch in processor.iter_batches(reviews):
            for review in batch:
                await process_review(db, review)
                processor.mark_processed()

        result = processor.result
        logger.info(str(result))
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        service_name: str = ServiceName.API,
        operation_name: str = "batch_operation",
        continue_on_error: bool = True,
    ) -> None:
        self.batch_size = safe_batch_size(batch_size)
        self.service_name = service_name
        self.operation_name = operation_name
        self.continue_on_error = continue_on_error
        self._result = BatchResult()

    @property
    def result(self) -> BatchResult:
        """Return the current aggregated result."""
        return self._result

    def mark_processed(self, count: int = 1) -> None:
        """Increment the processed counter."""
        self._result.processed += count

    def mark_failed(self, index: int = 0, error: str = "") -> None:
        """Increment the failed counter and record the error."""
        self._result.failed += 1
        self._result.errors.append((index, error))

    def mark_skipped(self, count: int = 1) -> None:
        """Increment the skipped counter."""
        self._result.skipped += count

    async def iter_batches(
        self,
        items: list[T],
        delay_between_batches: float = 0.0,
    ) -> AsyncGenerator[list[T], None]:
        """
        Async generator yielding batches from the provided list.

        Updates total_items and total_batches on the result before
        yielding the first batch.

        Args:
            items:                  List of items to split and yield.
            delay_between_batches:  Sleep in seconds between batches.

        Yields:
            list[T]: Each batch of at most batch_size items.
        """
        batches = chunk(items, self.batch_size)
        self._result.total_items = len(items)
        self._result.total_batches = len(batches)

        logger.info(
            "BatchProcessor started",
            extra={
                "service": self.service_name,
                "operation": self.operation_name,
                "total_items": len(items),
                "batch_size": self.batch_size,
                "total_batches": len(batches),
            },
        )

        for batch_num, batch in enumerate(batches, start=1):
            yield batch

            logger.debug(
                "BatchProcessor batch yielded",
                extra={
                    "service": self.service_name,
                    "operation": self.operation_name,
                    "batch_number": batch_num,
                    "total_batches": len(batches),
                },
            )

            if delay_between_batches > 0 and batch_num < len(batches):
                await asyncio.sleep(delay_between_batches)

        logger.info(
            "BatchProcessor completed",
            extra={
                "service": self.service_name,
                "operation": self.operation_name,
                "processed": self._result.processed,
                "failed": self._result.failed,
                "skipped": self._result.skipped,
                "success_rate": f"{self._result.success_rate:.1%}",
            },
        )

    def reset(self) -> None:
        """Reset the result state for reuse of this processor instance."""
        self._result = BatchResult()