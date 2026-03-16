# ==============================================================================
# File: app/notifications/admin_notification_service.py
# Purpose: Sends operational alerts, payment events, system health summaries,
#          and abuse detection notices to the platform administrator via
#          WhatsApp.
#
#          ONE SUBSCRIPTION TIER ONLY.
#          No plan_name parameter exists anywhere in this file.
#          The billing_cycle ("monthly" / "annual") is the only variant.
#
#          This is a SYSTEM-LEVEL notification service — it notifies the
#          platform owner/admin, not individual business owners.
#          Business owner notifications are handled by whatsapp_service.py.
#
#          Admin WhatsApp number is loaded from:
#            settings.ADMIN_WHATSAPP_NUMBER
#          If not set, all notifications are logged but not delivered.
#
#          Notification categories:
#            1. System alerts     → send_alert / send_critical / send_warning
#            2. Payment events    → send_payment_received / send_payment_failed
#                                   / send_refund_processed
#            3. Business events   → send_new_business_alert / send_abuse_alert
#                                   / send_subscription_expiry_summary
#            4. System health     → send_health_summary / send_job_failure
#                                   / send_integration_failure
#
#          Rate limiting:
#            Same alert type is suppressed after MAX_SAME_ALERTS_PER_HOUR
#            using an in-memory counter keyed by (alert_type, hour_bucket).
#            Counts reset on restart — this is intentional.
#
#          Never raises:
#            All public methods are wrapped in try/except. A failed admin
#            alert is logged only and never propagates to the caller.
# ==============================================================================

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.integrations.whatsapp_client import WhatsAppClient

logger = logging.getLogger(ServiceName.ADMIN)
settings = get_settings()

# ---------------------------------------------------------------------------
# Rate limiting: max same-type alerts per hour before suppression kicks in
# ---------------------------------------------------------------------------
_MAX_SAME_ALERTS_PER_HOUR: int = 5

# ---------------------------------------------------------------------------
# Severity display labels
# ---------------------------------------------------------------------------
_SEVERITY_LABEL: dict[str, str] = {
    "critical": "🚨 *CRITICAL*",
    "high":     "🔴 *HIGH*",
    "warning":  "⚠️ *WARNING*",
    "info":     "ℹ️ *INFO*",
    "success":  "✅ *SUCCESS*",
}


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class AdminNotificationResult:
    """
    Result of an admin notification delivery attempt.

    Attributes:
        success:    True if message delivered (or skipped intentionally).
        message_id: WhatsApp message ID returned on success.
        suppressed: True if suppressed by rate limiter.
        skipped:    True if no admin number configured.
        error:      Error string if delivery failed.
    """
    success: bool
    message_id: Optional[str] = None
    suppressed: bool = False
    skipped: bool = False
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.skipped:
            return "AdminNotificationResult(skipped=no_admin_number)"
        if self.suppressed:
            return "AdminNotificationResult(suppressed=rate_limited)"
        status = "OK" if self.success else "FAIL"
        return f"AdminNotificationResult({status} id={self.message_id})"


# ==============================================================================
# Admin Notification Service
# ==============================================================================

class AdminNotificationService:
    """
    Sends operational notifications to the platform administrator.

    ONE TIER ONLY — no plan_name on any method.
    billing_cycle ("monthly" / "annual") is used where relevant.

    Stateful only in the in-memory rate limiter counters.
    Counters reset on process restart (intentional).

    Usage:
        admin = AdminNotificationService(whatsapp_client=client)

        await admin.send_critical(
            title="Database connection lost",
            message="PostgreSQL unreachable after 3 retries.",
        )

        await admin.send_payment_received(
            business_name="Raj Restaurant",
            amount_rupees=999.0,
            billing_cycle="monthly",
        )
    """

    def __init__(self, whatsapp_client: WhatsAppClient) -> None:
        self._client = whatsapp_client
        self._admin_number: Optional[str] = settings.ADMIN_WHATSAPP_NUMBER or None

        # Rate limiter: {(alert_type, hour_bucket): count}
        self._rate_counts: dict[tuple[str, str], int] = defaultdict(int)

        if not self._admin_number:
            logger.warning(
                "ADMIN_WHATSAPP_NUMBER not configured — "
                "admin notifications will be logged only",
                extra={"service": ServiceName.ADMIN},
            )

    # ------------------------------------------------------------------
    # 1. Generic alerts
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        alert_type: Optional[str] = None,
    ) -> AdminNotificationResult:
        """
        Send a generic admin alert with severity prefix.

        Args:
            title:      Alert headline.
            message:    Alert body detail.
            severity:   "critical" / "high" / "warning" / "info" / "success".
            alert_type: Key for rate limiting (defaults to title[:40]).

        Returns:
            AdminNotificationResult. Never raises.
        """
        key = alert_type or title[:40]
        text = _format_alert(title=title, message=message, severity=severity)
        return await self._deliver(text=text, alert_type=key)

    async def send_critical(
        self,
        title: str,
        message: str,
    ) -> AdminNotificationResult:
        """
        Send a critical system alert. Never rate-limited — always delivered.

        Use for events requiring immediate admin action:
          - database unreachable
          - scheduler crash
          - payment webhook signature failures

        Args:
            title:   Short description of the critical event.
            message: Detail and recommended action.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"🚨 *CRITICAL — ACTION REQUIRED*\n\n"
            f"*{title}*\n\n"
            f"{message}\n\n"
            f"_{_timestamp()}_"
        )
        return await self._deliver(
            text=text,
            alert_type="critical",
            bypass_rate_limit=True,
        )

    async def send_warning(
        self,
        title: str,
        message: str,
    ) -> AdminNotificationResult:
        """
        Send a warning-level admin alert.

        Use for events that need attention but are not immediately critical.

        Returns:
            AdminNotificationResult. Never raises.
        """
        return await self.send_alert(
            title=title,
            message=message,
            severity="warning",
            alert_type=f"warning_{title[:30]}",
        )

    # ------------------------------------------------------------------
    # 2. Payment events
    # ------------------------------------------------------------------

    async def send_payment_received(
        self,
        business_name: str,
        amount_rupees: float,
        billing_cycle: str,
        business_id: Optional[str] = None,
    ) -> AdminNotificationResult:
        """
        Notify admin of a successful subscription payment.

        Called by payment_service.py after payment.captured webhook
        is verified and subscription is activated.

        ONE TIER — no plan_name. billing_cycle shows duration only.

        Args:
            business_name:  Display name of the business.
            amount_rupees:  Amount received in rupees.
            billing_cycle:  "monthly" or "annual".
            business_id:    Optional UUID for log traceability.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"✅ *Payment Received*\n\n"
            f"*Business:* {business_name}\n"
            f"*Billing:* {billing_cycle.title()}\n"
            f"*Amount:* ₹{amount_rupees:,.2f}\n"
            + (f"*Business ID:* `{business_id}`\n" if business_id else "")
            + f"\n_{_timestamp()}_"
        )
        return await self._deliver(text=text, alert_type="payment_received")

    async def send_payment_failed(
        self,
        business_name: str,
        amount_rupees: float,
        billing_cycle: str,
        reason: Optional[str] = None,
        business_id: Optional[str] = None,
    ) -> AdminNotificationResult:
        """
        Notify admin of a failed payment attempt.

        ONE TIER — no plan_name.

        Args:
            business_name:  Display name of the business.
            amount_rupees:  Amount that failed.
            billing_cycle:  "monthly" or "annual".
            reason:         Razorpay failure reason string if available.
            business_id:    Optional UUID.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"🔴 *Payment Failed*\n\n"
            f"*Business:* {business_name}\n"
            f"*Billing:* {billing_cycle.title()}\n"
            f"*Amount:* ₹{amount_rupees:,.2f}\n"
            + (f"*Reason:* {reason}\n" if reason else "")
            + (f"*Business ID:* `{business_id}`\n" if business_id else "")
            + f"\n_{_timestamp()}_"
        )
        return await self._deliver(text=text, alert_type="payment_failed")

    async def send_refund_processed(
        self,
        business_name: str,
        amount_rupees: float,
        refund_id: Optional[str] = None,
        business_id: Optional[str] = None,
    ) -> AdminNotificationResult:
        """
        Notify admin that a refund was processed.

        Args:
            business_name:  Display name of the business.
            amount_rupees:  Refund amount in rupees.
            refund_id:      Razorpay refund ID.
            business_id:    Optional UUID.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"↩️ *Refund Processed*\n\n"
            f"*Business:* {business_name}\n"
            f"*Amount:* ₹{amount_rupees:,.2f}\n"
            + (f"*Refund ID:* `{refund_id}`\n" if refund_id else "")
            + (f"*Business ID:* `{business_id}`\n" if business_id else "")
            + f"\n_{_timestamp()}_"
        )
        return await self._deliver(text=text, alert_type="refund_processed")

    # ------------------------------------------------------------------
    # 3. Business events
    # ------------------------------------------------------------------

    async def send_new_business_alert(
        self,
        business_name: str,
        whatsapp_number: str,
        business_id: str,
    ) -> AdminNotificationResult:
        """
        Notify admin when a new business registers on the platform.

        ONE TIER — no plan_name. All businesses get the same full access.

        Args:
            business_name:    Display name of the new business.
            whatsapp_number:  Registered WhatsApp number.
            business_id:      UUID of the new business record.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"🎉 *New Business Registered*\n\n"
            f"*Name:* {business_name}\n"
            f"*WhatsApp:* {whatsapp_number}\n"
            f"*Business ID:* `{business_id}`\n"
            f"\n_{_timestamp()}_"
        )
        return await self._deliver(text=text, alert_type="new_business")

    async def send_abuse_alert(
        self,
        business_name: str,
        business_id: str,
        feature: str,
        usage_count: int,
        daily_limit: int,
    ) -> AdminNotificationResult:
        """
        Notify admin when a business repeatedly hits usage limits.

        Called by rate_limiter.py when daily limits are exceeded.
        Limits are abuse guards only — not plan gates.

        Args:
            business_name:  Display name of the business.
            business_id:    Business UUID.
            feature:        Feature being over-used.
            usage_count:    Current usage count today.
            daily_limit:    Allowed daily limit.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"⚠️ *Usage Limit Exceeded*\n\n"
            f"*Business:* {business_name}\n"
            f"*Feature:* {feature}\n"
            f"*Usage:* {usage_count}/{daily_limit} today\n"
            f"*Business ID:* `{business_id}`\n\n"
            f"Review for potential abuse or contact business.\n"
            f"\n_{_timestamp()}_"
        )
        return await self._deliver(
            text=text,
            alert_type=f"abuse_{business_id[:8]}_{feature}",
        )

    async def send_subscription_expiry_summary(
        self,
        checked: int,
        expired: int,
        reminders_sent: int,
        errors: int,
        run_date: str,
    ) -> AdminNotificationResult:
        """
        Send admin a daily summary of subscription expiry check results.

        Called by expiry_checker.py after the daily expiry run completes
        when there are notable events (expired > 0 or errors > 0).

        Args:
            checked:        Total subscriptions evaluated.
            expired:        Number newly expired.
            reminders_sent: Renewal reminders dispatched.
            errors:         Processing errors encountered.
            run_date:       ISO date string of the run.

        Returns:
            AdminNotificationResult. Never raises.
        """
        if expired == 0 and errors == 0 and reminders_sent == 0:
            return AdminNotificationResult(success=True, skipped=True)

        icon = "⚠️" if errors > 0 else "📋"
        text = (
            f"{icon} *Subscription Expiry Summary — {run_date}*\n\n"
            f"Checked:          {checked}\n"
            f"Expired:          {expired}\n"
            f"Reminders sent:   {reminders_sent}\n"
            f"Errors:           {errors}\n"
        )
        if errors > 0:
            text += "\n⚠️ Some subscriptions may not have been processed. Check logs."

        text += f"\n_{_timestamp()}_"
        return await self._deliver(text=text, alert_type="expiry_summary")

    # ------------------------------------------------------------------
    # 4. System health
    # ------------------------------------------------------------------

    async def send_health_summary(
        self,
        db_ok: bool,
        scheduler_ok: bool,
        google_api_ok: bool,
        whatsapp_api_ok: bool,
        openai_api_ok: bool,
        active_businesses: int,
        active_subscriptions: int,
        jobs_run_today: int,
        errors_today: int,
        report_date: str,
    ) -> AdminNotificationResult:
        """
        Send the periodic system health summary to admin.

        Called by admin_health_report.py on a daily or weekly schedule.

        Args:
            db_ok:                 Database connectivity check passed.
            scheduler_ok:          Scheduler is running.
            google_api_ok:         Google API reachable.
            whatsapp_api_ok:       WhatsApp Cloud API reachable.
            openai_api_ok:         OpenAI API reachable.
            active_businesses:     Businesses with active subscriptions.
            active_subscriptions:  Active subscription record count.
            jobs_run_today:        Background jobs executed today.
            errors_today:          Logged errors in the last 24 hours.
            report_date:           ISO date string of the report.

        Returns:
            AdminNotificationResult. Never raises.
        """
        def _status(ok: bool) -> str:
            return "✅ OK" if ok else "🔴 DOWN"

        all_ok = all([db_ok, scheduler_ok, google_api_ok, whatsapp_api_ok, openai_api_ok])
        header = "✅ *System Health — All OK*" if all_ok else "⚠️ *System Health — Issues Detected*"

        text = (
            f"{header}\n"
            f"_{report_date}_\n\n"
            f"*Infrastructure*\n"
            f"Database:      {_status(db_ok)}\n"
            f"Scheduler:     {_status(scheduler_ok)}\n"
            f"Google API:    {_status(google_api_ok)}\n"
            f"WhatsApp API:  {_status(whatsapp_api_ok)}\n"
            f"OpenAI API:    {_status(openai_api_ok)}\n\n"
            f"*Platform*\n"
            f"Active businesses:     {active_businesses}\n"
            f"Active subscriptions:  {active_subscriptions}\n"
            f"Jobs run today:        {jobs_run_today}\n"
            f"Errors today:          {errors_today}\n"
        )

        if not all_ok:
            text += "\n🔴 One or more services are DOWN. Immediate attention required."

        text += f"\n\n_{_timestamp()}_"

        return await self._deliver(
            text=text,
            alert_type="health_summary",
            bypass_rate_limit=True,
        )

    async def send_job_failure(
        self,
        job_name: str,
        error: str,
        business_id: Optional[str] = None,
    ) -> AdminNotificationResult:
        """
        Notify admin of a background job failure.

        Called by failsafe_runner.py or individual job handlers when
        a job fails after all retries are exhausted.

        Args:
            job_name:    Name of the failed job.
            error:       Error message or exception string.
            business_id: Business context if failure was business-scoped.

        Returns:
            AdminNotificationResult. Never raises.
        """
        text = (
            f"🔴 *Job Failure*\n\n"
            f"*Job:* {job_name}\n"
            + (f"*Business:* `{business_id}`\n" if business_id else "")
            + f"*Error:* {error[:400]}\n"
            f"\n_{_timestamp()}_"
        )
        return await self._deliver(
            text=text,
            alert_type=f"job_failure_{job_name}",
        )

    async def send_integration_failure(
        self,
        integration_name: str,
        error: str,
        retries_exhausted: bool = False,
    ) -> AdminNotificationResult:
        """
        Notify admin of an external integration failure.

        Args:
            integration_name:   e.g. "google_reviews", "whatsapp", "openai".
            error:              Error detail from the last attempt.
            retries_exhausted:  True if all retry attempts were spent.

        Returns:
            AdminNotificationResult. Never raises.
        """
        retry_note = " All retries exhausted." if retries_exhausted else ""
        text = (
            f"🔴 *Integration Failure*\n\n"
            f"*Service:* {integration_name}\n"
            f"*Error:* {error[:400]}\n"
            f"{retry_note}\n"
            f"\n_{_timestamp()}_"
        )
        return await self._deliver(
            text=text,
            alert_type=f"integration_{integration_name}",
        )

    # ------------------------------------------------------------------
    # Internal delivery with rate limiting
    # ------------------------------------------------------------------

    async def _deliver(
        self,
        text: str,
        alert_type: str,
        bypass_rate_limit: bool = False,
    ) -> AdminNotificationResult:
        """
        Deliver a message to the admin WhatsApp number.

        Applies in-memory rate limiting to prevent flood alerts.
        Logs the notification regardless of delivery outcome.

        Args:
            text:               Formatted message body.
            alert_type:         Rate-limit key for this alert category.
            bypass_rate_limit:  If True, skip rate limit check (critical only).

        Returns:
            AdminNotificationResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.ADMIN,
            "alert_type": alert_type,
        }

        logger.info(
            "Admin notification: %s",
            alert_type,
            extra={**log_extra, "preview": text[:120]},
        )

        if not self._admin_number:
            return AdminNotificationResult(success=True, skipped=True)

        if not bypass_rate_limit and self._is_rate_limited(alert_type):
            logger.debug(
                "Admin notification suppressed by rate limiter",
                extra=log_extra,
            )
            return AdminNotificationResult(success=True, suppressed=True)

        try:
            result = await self._client.send_text_message(
                to=self._admin_number,
                text=text,
            )

            if result.success:
                self._increment_rate_counter(alert_type)
                return AdminNotificationResult(
                    success=True,
                    message_id=result.message_id,
                )
            else:
                logger.warning(
                    "Admin notification delivery failed",
                    extra={**log_extra, "error": result.error},
                )
                return AdminNotificationResult(
                    success=False,
                    error=result.error,
                )

        except Exception as exc:
            logger.error(
                "Admin notification unexpected error",
                extra={**log_extra, "error": str(exc)},
            )
            return AdminNotificationResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # In-memory rate limiter
    # ------------------------------------------------------------------

    def _rate_bucket(self, alert_type: str) -> tuple[str, str]:
        """Build the rate limit key for an alert type in the current hour."""
        now = datetime.now(tz=timezone.utc)
        hour_bucket = now.strftime("%Y-%m-%d-%H")
        return (alert_type, hour_bucket)

    def _is_rate_limited(self, alert_type: str) -> bool:
        """Return True if this alert type has hit the per-hour cap."""
        key = self._rate_bucket(alert_type)
        return self._rate_counts[key] >= _MAX_SAME_ALERTS_PER_HOUR

    def _increment_rate_counter(self, alert_type: str) -> None:
        """Increment the per-hour delivery counter for this alert type."""
        key = self._rate_bucket(alert_type)
        self._rate_counts[key] += 1
        self._evict_stale_counters()

    def _evict_stale_counters(self) -> None:
        """
        Remove counters from previous hours to prevent unbounded memory growth.
        Called after every increment.
        """
        now = datetime.now(tz=timezone.utc)
        current_hour = now.strftime("%Y-%m-%d-%H")
        stale = [k for k in self._rate_counts if k[1] != current_hour]
        for key in stale:
            del self._rate_counts[key]


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _format_alert(title: str, message: str, severity: str) -> str:
    """Format a generic alert with severity prefix, title, and body."""
    prefix = _SEVERITY_LABEL.get(severity.lower(), "ℹ️ *INFO*")
    return (
        f"{prefix}\n\n"
        f"*{title}*\n\n"
        f"{message}\n\n"
        f"_{_timestamp()}_"
    )


def _timestamp() -> str:
    """Return a human-readable UTC timestamp string."""
    return datetime.now(tz=timezone.utc).strftime("%d %b %Y %H:%M UTC")