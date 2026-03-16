# ==============================================================================
# File: app/database/models/job_model.py
# Purpose: SQLAlchemy ORM model for tracking the execution history of all
#          background scheduler jobs. Every job run — successful, failed, or
#          skipped — is recorded here. This table also serves as the
#          distributed job lock store, preventing duplicate concurrent
#          execution of the same job for the same business.
#
#          Job lifecycle:
#            Scheduler triggers job → lock acquired → record created (RUNNING)
#            → job completes       → record updated (COMPLETED) → lock released
#            → job fails           → record updated (FAILED)    → lock released
#            → lock already exists → record created (SKIPPED)
#
#          Retention policy:
#            Records older than 90 days are purged automatically by a
#            scheduled maintenance task to prevent unbounded table growth.
#            See: JOB_LOG_RETENTION_DAYS in app/config/constants.py
# ==============================================================================

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.constants import JobStatus, JobType, JOB_LOG_RETENTION_DAYS
from app.database.base import BaseModel

if TYPE_CHECKING:
    from app.database.models.business_model import BusinessModel


# ==============================================================================
# Job Model
# ==============================================================================

class JobModel(BaseModel):
    """
    Records the execution history of every background scheduler job.

    Each row represents a single execution attempt of a named job for a
    specific business. The lock_key column doubles as a distributed lock
    mechanism — the scheduler_manager checks for RUNNING records with a
    non-expired lock before starting a new execution.

    Retention Policy:
        Records where created_at < NOW() - INTERVAL '90 days' are eligible
        for purge. The scheduler runs a nightly maintenance task that deletes
        stale records in batches to avoid table lock contention.
        Constant: JOB_LOG_RETENTION_DAYS = 90

    Inherits from BaseModel which provides:
        - id           (UUID v4, primary key)
        - created_at   (timestamp, set on insert)
        - updated_at   (timestamp, updated on every write)

    Table:
        job_logs

    Indexes:
        - ix_job_logs_business_id             — tenant isolation
        - ix_job_logs_job_type                — filter by job type
        - ix_job_logs_status                  — filter running / failed jobs
        - ix_job_logs_business_job_type       — composite: active lock lookup
        - ix_job_logs_lock_key                — lock acquisition check
        - ix_job_logs_lock_expires_at         — expired lock cleanup
        - ix_job_logs_created_at              — retention purge queries
        - uq_job_logs_lock_key                — one active lock per job per business
    """

    __tablename__ = "job_logs"

    # ── Tenant Reference ──────────────────────────────────────────────────────

    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the owning business — tenant isolation key",
    )

    # ── Job Identity ──────────────────────────────────────────────────────────

    job_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Identifier for the job type (see JobType enum in constants.py)",
    )

    job_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable name of the job for logs and admin dashboards",
    )

    # ── Execution Status ──────────────────────────────────────────────────────

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=JobStatus.PENDING,
        server_default=JobStatus.PENDING,
        comment="Current execution state of this job run",
    )

    # ── Execution Timing ──────────────────────────────────────────────────────

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when job execution began (UTC)",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when job execution finished — success or failure (UTC)",
    )

    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total wall-clock execution time in seconds",
    )

    # ── Job Lock ──────────────────────────────────────────────────────────────

    lock_key: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment=(
            "Distributed lock key held during job execution. "
            "Format: JOB_LOCK_{JOB_TYPE}_BUSINESS_{BUSINESS_ID}. "
            "Cleared on completion or failure to release the lock."
        ),
    )

    lock_acquired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the job lock was acquired (UTC)",
    )

    lock_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "Timestamp after which the lock is considered stale and may be "
            "forcibly released. Set to acquired_at + JOB_LOCK_TTL_SECONDS."
        ),
    )

    # ── Execution Results ─────────────────────────────────────────────────────

    records_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of records (reviews, reports, etc.) processed in this run",
    )

    records_failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of records that failed to process in this run",
    )

    records_skipped: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of records skipped (e.g., duplicates, invalid data) in this run",
    )

    # ── Error Tracking ────────────────────────────────────────────────────────

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Short error message if the job failed or was partially successful",
    )

    error_traceback: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full Python traceback captured on failure — stored for debugging",
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times this job run has been retried after failure",
    )

    # ── Skip Reason ───────────────────────────────────────────────────────────

    skip_reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment=(
            "Reason the job was skipped without executing. "
            "Examples: lock already held, business inactive, limits exceeded."
        ),
    )

    # ── Trigger Source ────────────────────────────────────────────────────────

    triggered_by: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="scheduler",
        server_default="scheduler",
        comment=(
            "Source that triggered this job run. "
            "Values: 'scheduler', 'manual', 'webhook', 'admin'."
        ),
    )

    # ── Metadata ──────────────────────────────────────────────────────────────

    job_metadata: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Optional JSON string with job-specific metadata. "
            "Example: report type, date range, competitor count scanned."
        ),
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    business: Mapped["BusinessModel"] = relationship(
        "BusinessModel",
        back_populates="jobs",
        lazy="selectin",
    )

    # ── Table-level Constraints and Indexes ───────────────────────────────────

    __table_args__ = (
        # One active lock per job type per business — prevents concurrent execution.
        # lock_key is set to NULL on completion so the constraint only applies
        # while a job is RUNNING, allowing historical records to coexist.
        UniqueConstraint(
            "lock_key",
            name="uq_job_logs_lock_key",
        ),
        Index(
            "ix_job_logs_business_id",
            "business_id",
        ),
        Index(
            "ix_job_logs_job_type",
            "job_type",
        ),
        Index(
            "ix_job_logs_status",
            "status",
        ),
        # Primary lock check index — scheduler queries:
        # "is there a RUNNING job of this type for this business?"
        Index(
            "ix_job_logs_business_job_type",
            "business_id",
            "job_type",
        ),
        # Composite status index for dashboard queries:
        # "show all FAILED jobs for this business"
        Index(
            "ix_job_logs_business_status",
            "business_id",
            "status",
        ),
        Index(
            "ix_job_logs_lock_key",
            "lock_key",
        ),
        # Stale lock cleanup index — scheduler queries:
        # "find all RUNNING jobs where lock_expires_at < NOW()"
        Index(
            "ix_job_logs_lock_expires_at",
            "lock_expires_at",
        ),
        # Retention purge index — nightly cleanup queries:
        # "DELETE FROM job_logs WHERE created_at < NOW() - INTERVAL '90 days'"
        Index(
            "ix_job_logs_created_at",
            "created_at",
        ),
    )

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if this job is currently executing and holds a lock."""
        return self.status == JobStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """True if this job finished successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """True if this job terminated with an error."""
        return self.status == JobStatus.FAILED

    @property
    def is_skipped(self) -> bool:
        """True if this job was skipped without executing."""
        return self.status == JobStatus.SKIPPED

    @property
    def is_terminal(self) -> bool:
        """
        True if this job has reached a terminal state.

        Terminal jobs (COMPLETED, FAILED, SKIPPED) no longer hold a lock.
        A terminal record must not be re-activated.
        """
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SKIPPED,
        }

    @property
    def has_errors(self) -> bool:
        """True if the job recorded any errors or failed records."""
        return bool(self.error_message) or self.records_failed > 0

    @property
    def success_rate(self) -> float:
        """
        Ratio of successfully processed records to total attempted (0.0–1.0).

        Returns 0.0 if no records were attempted.
        """
        total_attempted = self.records_processed + self.records_failed
        if total_attempted == 0:
            return 0.0
        return round(self.records_processed / total_attempted, 4)

    @property
    def retention_days(self) -> int:
        """
        Number of days this record is retained before becoming eligible for purge.

        Returns the system-wide constant JOB_LOG_RETENTION_DAYS (90 days).
        """
        return JOB_LOG_RETENTION_DAYS

    def to_summary_dict(self) -> dict:
        """
        Return a concise summary dictionary for admin health reports
        and system dashboards.

        Returns:
            dict: Key execution metrics for this job run.
        """
        return {
            "id": str(self.id),
            "business_id": str(self.business_id),
            "job_type": self.job_type,
            "job_name": self.job_name,
            "status": self.status,
            "triggered_by": self.triggered_by,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "records_skipped": self.records_skipped,
            "retry_count": self.retry_count,
            "has_errors": self.has_errors,
            "success_rate": self.success_rate,
            "error_message": self.error_message,
            "skip_reason": self.skip_reason,
        }

    def __repr__(self) -> str:
        return (
            f"<JobModel id={self.id} "
            f"business_id={self.business_id} "
            f"job_type='{self.job_type}' "
            f"status='{self.status}' "
            f"duration={self.duration_seconds}s>"
        )


# ==============================================================================
# JobLockModel — Lightweight distributed lock table for scheduler jobs
# ==============================================================================

class JobLockModel(BaseModel):
    """
    Stores active job locks to prevent duplicate concurrent scheduler runs.

    A lock record is created when a job starts and deleted when it completes
    or fails. If a lock already exists for a given job+business combination,
    the scheduler skips the run for that cycle.

    Lock keys use the format defined in make_job_lock_key() in idempotency_utils.py.

    Retention: Locks are ephemeral — they are always deleted after the job
    completes. Stale locks (from crashed jobs) expire after JOB_LOCK_TTL_MINUTES
    and are cleaned up by the scheduler health check.
    """

    __tablename__ = "job_locks"
    __table_args__ = (
        UniqueConstraint("lock_key", name="uq_job_lock_key"),
        Index("ix_job_locks_lock_key",   "lock_key"),
        Index("ix_job_locks_expires_at", "expires_at"),
    )

    lock_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
        comment="Unique lock key — format: JOB_LOCK_<JOB_NAME>_BUSINESS_<id>",
    )

    job_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Job name for logging and debugging.",
    )

    business_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=True,
        comment="Business scope for per-business locks. NULL for global locks.",
    )

    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the lock was acquired.",
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When this lock automatically expires (stale lock cleanup).",
    )

    def __repr__(self) -> str:
        return f"<JobLockModel key={self.lock_key!r}>"