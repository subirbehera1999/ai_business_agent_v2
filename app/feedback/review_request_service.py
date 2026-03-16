# ==============================================================================
# File: app/feedback/review_request_service.py
# Purpose: Sends a one-time testimonial/feedback request to businesses that
#          have been active for 30+ days and have never received this message.
#
#          This service is the core of the feedback collection system. It is
#          triggered by a scheduled job (e.g. daily at a fixed time) and
#          processes eligible businesses in small batches.
#
# Eligibility rules (enforced by business_repository):
#   - Business is active and not deleted
#   - Onboarding is complete
#   - feedback_requested flag is False (never sent before)
#   - Business created_at <= (now - FEEDBACK_SEND_AFTER_DAYS)
#
# Processing pipeline per eligible business:
#   1. Build the feedback request message with the Google Form link
#      (sourced from form_link_manager)
#   2. Send the message via WhatsApp (send_text_message)
#   3. On delivery success: call business_repo.mark_feedback_requested()
#      to permanently flag the business — this is idempotent by design
#   4. On delivery failure: log a warning, do NOT mark as requested
#      (the job will retry on the next run)
#
# Idempotency guarantee:
#   The feedback_requested flag on the BusinessModel is a permanent one-way
#   flag. Once set, business_repo.get_pending_feedback_request() will never
#   return that business again. No additional idempotency key is needed
#   because the flag itself acts as the dedup guard.
#
# Why mark only on delivery success:
#   If WhatsApp delivery fails (API outage, wrong number), the business
#   should still receive the message in a future run. Marking on failure
#   would permanently skip businesses that never received the message.
#
# Batch safety:
#   The service processes up to BATCH_SIZE businesses per run (default 20).
#   This matches the scheduler batch contract and prevents memory spikes.
#
# Session contract:
#   The caller (scheduler) opens and commits the AsyncSession.
#   This service calls db.flush() inside mark_feedback_requested() but
#   never commits directly.
# ==============================================================================

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import FEEDBACK_SEND_AFTER_DAYS, ServiceName
from app.config.settings import get_settings
from app.feedback.form_link_manager import FormLinkManager
from app.notifications.whatsapp_service import WhatsAppService
from app.repositories.business_repository import BusinessRepository

logger = logging.getLogger(ServiceName.SCHEDULER)

# Default batch size per run — aligned with scheduler batch contract
_DEFAULT_BATCH_SIZE: int = 20


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class FeedbackRequestRunResult:
    """
    Summary result for one execution of the feedback request service.

    Attributes:
        sent:     Number of feedback messages successfully delivered.
        skipped:  Number of businesses skipped (no WhatsApp number, etc.).
        failed:   Number of businesses where delivery failed.
        errors:   List of error strings for failed businesses.
    """
    sent: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def total_processed(self) -> int:
        return self.sent + self.skipped + self.failed

    def __str__(self) -> str:
        return (
            f"FeedbackRequestRunResult("
            f"sent={self.sent} "
            f"skipped={self.skipped} "
            f"failed={self.failed})"
        )


# ==============================================================================
# Service
# ==============================================================================

class ReviewRequestService:
    """
    Sends one-time feedback/testimonial requests to eligible businesses.

    Businesses become eligible 30 days after onboarding. Once a message is
    delivered, the business is permanently flagged and never contacted again.

    Usage (from scheduler):

        service = ReviewRequestService()
        async with AsyncSessionFactory() as db:
            result = await service.run(db)
            await db.commit()
    """

    def __init__(
        self,
        business_repo: Optional[BusinessRepository] = None,
        whatsapp_service: Optional[WhatsAppService] = None,
        form_link_manager: Optional[FormLinkManager] = None,
    ) -> None:
        self._business_repo = business_repo or BusinessRepository()
        self._whatsapp = whatsapp_service or WhatsAppService()
        self._form_link = form_link_manager or FormLinkManager()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        db: AsyncSession,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        reference_time: Optional[datetime] = None,
    ) -> FeedbackRequestRunResult:
        """
        Process one batch of eligible businesses for feedback requests.

        Fetches up to batch_size businesses that:
          - Are active and onboarding-complete
          - Have never received a feedback request
          - Registered more than FEEDBACK_SEND_AFTER_DAYS days ago

        For each eligible business, sends a WhatsApp message containing
        the Google Form link and marks the business on successful delivery.

        Args:
            db:             AsyncSession provided by the caller (scheduler).
            batch_size:     Maximum businesses to process in this run.
            reference_time: Override for "now" (for testing). Defaults to
                            current UTC time.

        Returns:
            FeedbackRequestRunResult. Never raises.
        """
        result = FeedbackRequestRunResult()
        log_extra = {"service": ServiceName.SCHEDULER, "job": "feedback_request"}

        now = reference_time or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=FEEDBACK_SEND_AFTER_DAYS)

        # ------------------------------------------------------------------
        # Fetch eligible businesses
        # ------------------------------------------------------------------
        try:
            businesses = await self._business_repo.get_pending_feedback_request(
                db,
                registered_before=cutoff,
                limit=batch_size,
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch businesses for feedback request",
                extra={**log_extra, "error": str(exc)},
            )
            result.add_error(f"fetch_failed: {exc}")
            return result

        if not businesses:
            logger.info(
                "Feedback request job: no eligible businesses found",
                extra=log_extra,
            )
            return result

        logger.info(
            "Feedback request job: processing batch",
            extra={**log_extra, "batch_size": len(businesses)},
        )

        # ------------------------------------------------------------------
        # Get the Google Form link once — shared across all messages
        # ------------------------------------------------------------------
        form_link = self._form_link.get_form_link()

        # ------------------------------------------------------------------
        # Per-business processing
        # ------------------------------------------------------------------
        for business in businesses:
            business_log = {
                **log_extra,
                "business_id": str(business.id),
                "business_name": business.business_name,
            }

            # Guard: must have a WhatsApp number
            if not business.owner_whatsapp_number:
                logger.info(
                    "Feedback request skipped — no WhatsApp number",
                    extra=business_log,
                )
                result.skipped += 1
                continue

            # Build and send the message
            await self._send_feedback_request(
                db=db,
                business_id=business.id,
                business_name=business.business_name,
                owner_name=business.owner_name,
                form_link=form_link,
                result=result,
                log_extra=business_log,
            )

        logger.info(
            "Feedback request job complete",
            extra={
                **log_extra,
                "sent": result.sent,
                "skipped": result.skipped,
                "failed": result.failed,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Single business — send + mark
    # ------------------------------------------------------------------

    async def _send_feedback_request(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        business_name: str,
        owner_name: str,
        form_link: str,
        result: FeedbackRequestRunResult,
        log_extra: dict,
    ) -> None:
        """
        Build the feedback message, deliver it, and mark the business.

        Marks the business as feedback_requested ONLY if WhatsApp delivery
        succeeds. On failure, leaves the flag unset so the scheduler retries
        on the next run.

        Args:
            db:            AsyncSession.
            business_id:   UUID of the business.
            business_name: Business display name.
            owner_name:    Owner's name for the greeting.
            form_link:     Google Form URL from FormLinkManager.
            result:        FeedbackRequestRunResult being accumulated.
            log_extra:     Structured log context for this business.
        """
        message = _build_feedback_message(
            owner_name=owner_name,
            business_name=business_name,
            form_link=form_link,
        )

        # Deliver via WhatsApp
        delivery = await self._whatsapp.send_text_message(
            db=db,
            business_id=str(business_id),
            text=message,
        )

        if not delivery.success:
            logger.warning(
                "Feedback request delivery failed — will retry next run",
                extra={
                    **log_extra,
                    "error": delivery.error,
                },
            )
            result.failed += 1
            result.add_error(
                f"business={business_id} error={delivery.error}"
            )
            return

        # Mark as requested — permanent one-way flag
        try:
            await self._business_repo.mark_feedback_requested(db, business_id)
        except Exception as exc:
            # Delivery succeeded but marking failed — log as warning.
            # The business will receive the message again on the next run,
            # which is acceptable (duplicate notification) but better than
            # silently skipping the mark.
            logger.error(
                "Feedback delivered but mark_feedback_requested failed",
                extra={**log_extra, "error": str(exc)},
            )
            result.failed += 1
            result.add_error(
                f"business={business_id} mark_failed: {exc}"
            )
            return

        logger.info(
            "Feedback request sent and marked",
            extra={
                **log_extra,
                "message_id": delivery.message_id,
            },
        )
        result.sent += 1


# ==============================================================================
# Message builder
# ==============================================================================

def _build_feedback_message(
    owner_name: str,
    business_name: str,
    form_link: str,
) -> str:
    """
    Build the WhatsApp feedback request message.

    The message is warm and brief — it thanks the owner for using the
    platform, explains what the form is for, and provides the link.
    Designed to feel personal, not automated.

    Args:
        owner_name:    Business owner's name.
        business_name: Business display name.
        form_link:     Google Form URL for collecting the testimonial.

    Returns:
        str: Formatted WhatsApp message body.
    """
    greeting = f"Hi {owner_name}," if owner_name else "Hi,"

    return (
        f"{greeting}\n\n"
        f"It's been 30 days since *{business_name}* joined our platform — "
        f"and we hope you've been seeing real value in your AI-powered review "
        f"replies and business insights! 🙌\n\n"
        f"We'd love to hear about your experience. Your feedback helps us "
        f"improve and helps other business owners understand what's possible.\n\n"
        f"It only takes 2 minutes 👇\n"
        f"{form_link}\n\n"
        f"Thank you for trusting us with your business. We're here whenever "
        f"you need us. 💬"
    )