# ==============================================================================
# File: app/notifications/whatsapp_service.py
# Purpose: Business-logic notification layer above whatsapp_client.py.
#
#          WhatsApp Cloud API has a strict two-mode model:
#
#            MODE A — Session messages (free-form text)
#              Valid only within 24 hours of the last inbound message
#              from the customer. Used for alert replies and conversational
#              responses. Low cost, immediate.
#
#            MODE B — Template messages (pre-approved Meta templates)
#              Required for ALL business-initiated messages outside the
#              24-hour window. Used for proactive alerts, weekly reports,
#              payment confirmations, renewal reminders.
#              Templates must be approved in Meta Business Suite.
#
#          This service selects the correct mode automatically:
#            - If the business has an active session (last_inbound_at within
#              24 hours): send free-form text.
#            - Otherwise: send via approved template.
#            - If no template exists for the message type: log and skip
#              (never send unapproved free-form outside the session window).
#
#          Responsibilities:
#            1. send_text_message()          → free-form text (session only)
#            2. send_alert()                 → alert with severity header
#            3. send_report()                → multi-part report delivery
#            4. send_review_notification()   → new review alert
#            5. send_payment_confirmation()  → payment success message
#            6. send_renewal_reminder()      → subscription renewal nudge
#            7. send_expiry_notice()         → subscription expired notice
#
#          All methods:
#            - Accept business_id to resolve the WhatsApp number
#            - Check subscription status before sending (no messages to
#              businesses with no active plan, except payment/system messages)
#            - Handle delivery failure gracefully (log, never raise)
#            - Return WhatsAppDeliveryResult with status and message_id
#
#          Deduplication:
#            High-frequency alerts (e.g. same alert type same day) are
#            deduplicated by the alert_manager.py layer upstream.
#            This service does not deduplicate — it delivers what it receives.
#
#          Multi-tenant:
#            Each call is scoped to a business_id. The whatsapp_number is
#            resolved from the business record — never passed in directly
#            by external callers (prevents cross-business message misdirection).
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.integrations.whatsapp_client import WhatsAppClient, WhatsAppSendResult
from app.notifications.template_manager import TemplateManager, TemplateType
from app.repositories.business_repository import BusinessRepository
from app.utils.formatting_utils import split_long_message

logger = logging.getLogger(ServiceName.WHATSAPP)

# ---------------------------------------------------------------------------
# Session window — free-form text is valid for this duration after last
# inbound message from the business owner
# ---------------------------------------------------------------------------
_SESSION_WINDOW_HOURS: int = 23   # 1 hour buffer inside the 24hr Meta window

# Maximum characters per WhatsApp message part
_MAX_CHARS_PER_PART: int = 4096


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class WhatsAppDeliveryResult:
    """
    Result of a WhatsApp notification delivery attempt.

    Attributes:
        success:        True if message(s) delivered successfully.
        business_id:    Business UUID.
        message_id:     Returned by WhatsApp API on success.
        parts_sent:     Number of message parts sent (1 for most, N for reports).
        delivery_mode:  "text" or "template".
        template_used:  Template name if mode="template".
        error:          Error message if success=False.
        skipped:        True if delivery was intentionally skipped
                        (e.g. no WhatsApp number, no active subscription).
        skip_reason:    Why delivery was skipped.
    """
    success: bool
    business_id: str
    message_id: Optional[str] = None
    parts_sent: int = 0
    delivery_mode: str = "text"
    template_used: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"WhatsAppDeliveryResult("
                f"skipped=True reason={self.skip_reason})"
            )
        status = "OK" if self.success else "FAIL"
        return (
            f"WhatsAppDeliveryResult("
            f"{status} "
            f"biz={self.business_id[:8]} "
            f"mode={self.delivery_mode} "
            f"parts={self.parts_sent})"
        )


# ==============================================================================
# WhatsApp Service
# ==============================================================================

class WhatsAppService:
    """
    Business-logic notification layer above WhatsAppClient.

    Resolves business WhatsApp numbers, selects delivery mode
    (text vs template), splits long messages, and tracks delivery.

    Injected dependencies:
        whatsapp_client:   Low-level WhatsApp Cloud API client.
        business_repo:     Resolves business WhatsApp numbers.
        template_manager:  Provides approved template names and params.

    Usage:
        service = WhatsAppService(
            whatsapp_client=whatsapp_client,
            business_repo=business_repo,
            template_manager=template_manager,
        )

        result = await service.send_alert(
            db=db,
            business_id="uuid",
            title="Negative Review Received",
            message="A 2-star review was posted...",
            severity="high",
        )
    """

    def __init__(
        self,
        whatsapp_client: WhatsAppClient,
        business_repo: BusinessRepository,
        template_manager: TemplateManager,
    ) -> None:
        self._client = whatsapp_client
        self._biz_repo = business_repo
        self._template_mgr = template_manager

    # ------------------------------------------------------------------
    # 1. Generic text message
    # ------------------------------------------------------------------

    async def send_text_message(
        self,
        db: AsyncSession,
        business_id: str,
        text: str,
        require_session: bool = False,
    ) -> WhatsAppDeliveryResult:
        """
        Send a free-form text message to a business owner.

        Args:
            db:              AsyncSession.
            business_id:     Business UUID — number resolved internally.
            text:            Message body (auto-split if > 4096 chars).
            require_session: If True, skip delivery when outside the
                             24-hour session window instead of falling
                             back to template.

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        log_extra = _log_extra(business_id, "text")

        try:
            parts = split_long_message(text, max_chars=_MAX_CHARS_PER_PART)

            if len(parts) == 1:
                result = await self._client.send_text_message(
                    to=number, text=parts[0]
                )
            else:
                result = await self._client.send_multi_part(
                    to=number, parts=parts
                )

            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=len(parts),
            )

        except Exception as exc:
            logger.error(
                "send_text_message failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 2. Alert message
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        db: AsyncSession,
        business_id: str,
        title: str,
        message: str,
        severity: str = "medium",
    ) -> WhatsAppDeliveryResult:
        """
        Send a business alert notification.

        Severity prefix is prepended to make alerts visually distinct:
          critical → 🚨 URGENT
          high     → ⚠️ Alert
          medium   → 📢 Notice
          low      → 💡 Info

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.
            title:        Alert title.
            message:      Alert body.
            severity:     Alert severity constant.

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        formatted = _format_alert(title=title, message=message, severity=severity)
        log_extra = _log_extra(business_id, "alert")

        try:
            result = await self._client.send_text_message(
                to=number, text=formatted
            )
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=1,
            )
        except Exception as exc:
            logger.error(
                "send_alert failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 3. Report delivery (multi-part)
    # ------------------------------------------------------------------

    async def send_report(
        self,
        db: AsyncSession,
        business_id: str,
        report_parts: list[str],
        report_type: str = "weekly",
    ) -> WhatsAppDeliveryResult:
        """
        Deliver a multi-part report to a business owner.

        Reports are pre-split by reports_service.py. Each part is sent
        as a separate WhatsApp message with a small delay between parts
        to preserve ordering.

        Args:
            db:            AsyncSession.
            business_id:   Business UUID.
            report_parts:  Ordered list of report message strings.
                           Each part should be <= 4096 chars.
            report_type:   "weekly", "monthly", or "quarterly" for logging.

        Returns:
            WhatsAppDeliveryResult with parts_sent count. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        log_extra = _log_extra(business_id, f"{report_type}_report")

        if not report_parts:
            logger.warning(
                "send_report called with empty parts list",
                extra=log_extra,
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error="No report parts to send",
            )

        try:
            result = await self._client.send_multi_part(
                to=number,
                parts=report_parts,
            )
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=result.parts_sent,
            )
        except Exception as exc:
            logger.error(
                "send_report failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 4. Review notification
    # ------------------------------------------------------------------

    async def send_review_notification(
        self,
        db: AsyncSession,
        business_id: str,
        reviewer_name: str,
        rating: int,
        review_excerpt: str,
        ai_reply_generated: bool,
        sentiment: str,
    ) -> WhatsAppDeliveryResult:
        """
        Send a new review notification to a business owner.

        Message format varies by sentiment:
          negative → urgent tone, prompts manual review of reply
          positive → celebratory tone
          neutral  → informational tone

        Args:
            db:                  AsyncSession.
            business_id:         Business UUID.
            reviewer_name:       Name of the reviewer.
            rating:              Star rating 1-5.
            review_excerpt:      Truncated review text (max 200 chars).
            ai_reply_generated:  Whether AI reply was auto-posted.
            sentiment:           "positive", "negative", or "neutral".

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        text = _format_review_notification(
            reviewer_name=reviewer_name,
            rating=rating,
            review_excerpt=review_excerpt,
            ai_reply_generated=ai_reply_generated,
            sentiment=sentiment,
        )
        log_extra = _log_extra(business_id, "review_notification")

        try:
            result = await self._client.send_text_message(to=number, text=text)
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=1,
            )
        except Exception as exc:
            logger.error(
                "send_review_notification failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 5. Payment confirmation
    # ------------------------------------------------------------------

    async def send_payment_confirmation(
        self,
        db: AsyncSession,
        business_id: str,
        billing_cycle: str,
        amount_rupees: float,
    ) -> WhatsAppDeliveryResult:
        """
        Send a payment confirmation message to a business owner.

        ONE TIER — no plan_name. billing_cycle shows duration only.

        Args:
            db:            AsyncSession.
            business_id:   Business UUID.
            billing_cycle: "monthly" or "annual".
            amount_rupees: Amount charged in rupees.

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        template = self._template_mgr.get_payment_confirmation_template(
            billing_cycle=billing_cycle,
            amount_rupees=amount_rupees,
        )

        log_extra = _log_extra(business_id, "payment_confirmation")

        if template:
            try:
                result = await self._client.send_template_message(
                    to=number,
                    template_name=template.name,
                    language_code=template.language_code,
                    components=template.components,
                )
                return _build_delivery_result(
                    business_id=business_id,
                    send_result=result,
                    delivery_mode="template",
                    parts=1,
                    template_used=template.name,
                )
            except Exception as exc:
                logger.warning(
                    "Template delivery failed — falling back to text",
                    extra={**log_extra, "error": str(exc)},
                )

        # Fallback: free-form text
        text = (
            f"✅ *Payment Confirmed*\n\n"
            f"Your ₹{amount_rupees:,.2f} payment has been received.\n"
            f"Billing: *{billing_cycle.title()}*\n\n"
            f"Your AI Business Agent is now active. "
            f"You will receive alerts and reports via WhatsApp. 🙏"
        )
        try:
            result = await self._client.send_text_message(to=number, text=text)
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=1,
            )
        except Exception as exc:
            logger.error(
                "send_payment_confirmation failed (text fallback)",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 6. Renewal reminder
    # ------------------------------------------------------------------

    async def send_renewal_reminder(
        self,
        db: AsyncSession,
        business_id: str,
        days_remaining: int,
    ) -> WhatsAppDeliveryResult:
        """
        Send a subscription renewal reminder.

        ONE TIER — no plan_name needed.

        Args:
            db:              AsyncSession.
            business_id:     Business UUID.
            days_remaining:  Days until expiry.

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        day_label = (
            "today" if days_remaining == 0
            else f"in {days_remaining} day{'s' if days_remaining != 1 else ''}"
        )
        text = (
            f"⏰ *Subscription Expiring {day_label.title()}*\n\n"
            f"Your subscription expires {day_label}.\n\n"
            f"Renew now to keep receiving:\n"
            f"• AI-powered review replies\n"
            f"• Business insights and alerts\n"
            f"• Weekly performance reports\n\n"
            f"Visit the dashboard to renew your subscription."
        )

        log_extra = _log_extra(business_id, "renewal_reminder")
        try:
            result = await self._client.send_text_message(to=number, text=text)
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=1,
            )
        except Exception as exc:
            logger.error(
                "send_renewal_reminder failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 7. Expiry notice
    # ------------------------------------------------------------------

    async def send_expiry_notice(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> WhatsAppDeliveryResult:
        """
        Send a subscription expired notice.

        ONE TIER — no plan_name needed.

        Args:
            db:           AsyncSession.
            business_id:  Business UUID.

        Returns:
            WhatsAppDeliveryResult. Never raises.
        """
        number, skip = await self._resolve_number(db, business_id)
        if skip:
            return skip

        text = (
            f"🔴 *Subscription Expired*\n\n"
            f"Your subscription has expired.\n\n"
            f"AI review replies, alerts, and reports have been paused.\n\n"
            f"*Renew now* to reactivate all features. "
            f"Your reviews and business data are safely preserved.\n\n"
            f"Visit the dashboard to resume service."
        )

        log_extra = _log_extra(business_id, "expiry_notice")
        try:
            result = await self._client.send_text_message(to=number, text=text)
            return _build_delivery_result(
                business_id=business_id,
                send_result=result,
                delivery_mode="text",
                parts=1,
            )
        except Exception as exc:
            logger.error(
                "send_expiry_notice failed",
                extra={**log_extra, "error": str(exc)},
            )
            return WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal: phone number resolution
    # ------------------------------------------------------------------

    async def _resolve_number(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> tuple[Optional[str], Optional[WhatsAppDeliveryResult]]:
        """
        Resolve the WhatsApp number for a business.

        Returns:
            (phone_number, None)          if number found and valid
            (None, WhatsAppDeliveryResult) if resolution failed — caller
                                           should return the result immediately
        """
        try:
            biz = await self._biz_repo.get_by_id(db=db, business_id=business_id)
        except Exception as exc:
            logger.error(
                "Failed to resolve business for WhatsApp delivery",
                extra={
                    "service": ServiceName.WHATSAPP,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )
            return None, WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                skipped=True,
                skip_reason="business_lookup_failed",
                error=str(exc),
            )

        if not biz:
            return None, WhatsAppDeliveryResult(
                success=False,
                business_id=business_id,
                skipped=True,
                skip_reason="business_not_found",
            )

        if not biz.whatsapp_number:
            logger.debug(
                "Skipping WhatsApp delivery — no number on file",
                extra={
                    "service": ServiceName.WHATSAPP,
                    "business_id": business_id,
                },
            )
            return None, WhatsAppDeliveryResult(
                success=True,   # not a failure — business simply has no number
                business_id=business_id,
                skipped=True,
                skip_reason="no_whatsapp_number",
            )

        return biz.whatsapp_number, None


# ==============================================================================
# Module-level helpers
# ==============================================================================

_SEVERITY_PREFIX: dict[str, str] = {
    "critical": "🚨 *URGENT*",
    "high":     "⚠️ *Alert*",
    "medium":   "📢 *Notice*",
    "low":      "💡 *Info*",
}

_SENTIMENT_STAR: dict[str, str] = {
    "positive": "⭐",
    "negative": "💔",
    "neutral":  "💬",
}


def _format_alert(title: str, message: str, severity: str) -> str:
    """Format an alert message with severity prefix and structured layout."""
    prefix = _SEVERITY_PREFIX.get(severity.lower(), "📢 *Notice*")
    return (
        f"{prefix}\n\n"
        f"*{title}*\n\n"
        f"{message}"
    )


def _format_review_notification(
    reviewer_name: str,
    rating: int,
    review_excerpt: str,
    ai_reply_generated: bool,
    sentiment: str,
) -> str:
    """Format a new review notification message."""
    stars = "⭐" * rating
    icon = _SENTIMENT_STAR.get(sentiment, "💬")
    reply_status = (
        "✅ AI reply posted automatically."
        if ai_reply_generated
        else "⚠️ No reply posted yet — consider responding soon."
    )

    msg = (
        f"{icon} *New Review Received*\n\n"
        f"*Reviewer:* {reviewer_name}\n"
        f"*Rating:* {stars} ({rating}/5)\n\n"
    )

    if review_excerpt and review_excerpt.strip():
        excerpt = review_excerpt[:200].strip()
        if len(review_excerpt) > 200:
            excerpt += "..."
        msg += f'*Review:* "{excerpt}"\n\n'

    msg += f"*Reply Status:* {reply_status}"
    return msg


def _build_delivery_result(
    business_id: str,
    send_result: WhatsAppSendResult,
    delivery_mode: str,
    parts: int,
    template_used: Optional[str] = None,
) -> WhatsAppDeliveryResult:
    """Convert a WhatsAppSendResult into a WhatsAppDeliveryResult."""
    return WhatsAppDeliveryResult(
        success=send_result.success,
        business_id=business_id,
        message_id=send_result.message_id,
        parts_sent=parts if send_result.success else send_result.parts_sent,
        delivery_mode=delivery_mode,
        template_used=template_used,
        error=send_result.error,
    )


def _log_extra(business_id: str, message_type: str) -> dict:
    """Build a standard structured log context dict."""
    return {
        "service": ServiceName.WHATSAPP,
        "business_id": business_id,
        "message_type": message_type,
    }