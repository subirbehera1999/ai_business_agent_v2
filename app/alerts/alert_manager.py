# ==============================================================================
# File: app/alerts/alert_manager.py
# Purpose: Central controller for the business alert system.
#
#          The alert manager is the single orchestration point for all
#          business event detection and notification delivery. It:
#
#            1. Accepts structured inputs from detector modules:
#                 - rating_alerts.py   → RatingDropInput
#                 - sales_alerts.py    → SalesAnomalyInput
#                 - competitor_alerts.py → CompetitorAlertInput
#                 - review pipeline    → ReviewAlertInput (negative/positive/spike)
#
#            2. For each candidate alert:
#                 a. Checks deduplication — was the same alert already sent
#                    for this business today? (uses alert_repository.py)
#                 b. Evaluates severity — CRITICAL / HIGH / MEDIUM / LOW
#                 c. Persists the alert record to the database
#                 d. Dispatches to whatsapp_service.py for delivery
#                 e. Marks the alert as sent (or failed)
#
#            3. Returns AlertDispatchResult — a structured summary of
#               what was sent, skipped (dedup), and failed, so the
#               calling scheduler can log accurately.
#
#          Deduplication strategy:
#            Static key: one alert of a given type per business per day.
#            Key format: ALERT_{alert_type}_{business_id}_{iso_date}
#            For review-level alerts: key includes review_id so the same
#            review never triggers two notifications.
#
#          Delivery contract:
#            - Alert manager never raises to its caller.
#            - WhatsApp delivery failure is caught, logged, and the alert
#              is marked as failed in the database for retry.
#            - A failed delivery does NOT prevent other alerts from firing.
#
#          Multi-tenant:
#            Every operation is scoped to business_id.
#            alert_repository queries always filter by business_id.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import AlertSeverity, AlertType, ServiceName
from app.repositories.alert_repository import AlertModel, AlertRepository
from app.utils.formatting_utils import format_alert_message, format_review_alert
from app.utils.time_utils import today_local

logger = logging.getLogger(ServiceName.ALERTS)


# ==============================================================================
# Input dataclasses — one per alert category
# ==============================================================================

@dataclass(frozen=True)
class ReviewAlertInput:
    """
    Input for a review-triggered alert (negative, positive, or spike).

    Attributes:
        business_id:      Business UUID.
        business_name:    Business display name.
        alert_type:       AlertType.NEGATIVE_REVIEW, POSITIVE_REVIEW,
                          or REVIEW_SPIKE.
        review_id:        UUID of the triggering review (for dedup key).
        reviewer_name:    Name of the reviewer.
        rating:           Star rating 1–5.
        review_excerpt:   Truncated review text for the notification.
        sentiment:        ReviewSentiment constant.
        ai_reply_generated: Whether an AI reply was auto-posted.
        spike_count:      Number of new reviews (for REVIEW_SPIKE only).
    """
    business_id: str
    business_name: str
    alert_type: str
    review_id: str
    reviewer_name: str
    rating: int
    review_excerpt: str
    sentiment: str
    ai_reply_generated: bool = False
    spike_count: Optional[int] = None


@dataclass(frozen=True)
class RatingDropInput:
    """
    Input for a rating drop alert.

    Attributes:
        business_id:     Business UUID.
        business_name:   Business display name.
        current_rating:  Latest average rating.
        previous_rating: Rating from the previous measurement.
        drop_amount:     Magnitude of the drop (positive float).
        period_label:    Time window e.g. "last 7 days".
    """
    business_id: str
    business_name: str
    current_rating: float
    previous_rating: float
    drop_amount: float
    period_label: str = "recent period"


@dataclass(frozen=True)
class SalesAnomalyInput:
    """
    Input for a sales trend alert (spike up or significant drop).

    Attributes:
        business_id:     Business UUID.
        business_name:   Business display name.
        alert_type:      AlertType.SALES_TREND or AlertType.OPPORTUNITY.
        change_pct:      Revenue change as a float e.g. 0.35 = +35%.
        current_revenue: Revenue for the current period (paise).
        period_label:    Period description e.g. "today vs 7-day avg".
        detail:          Human-readable description of the anomaly.
    """
    business_id: str
    business_name: str
    alert_type: str
    change_pct: float
    current_revenue: int       # paise
    period_label: str
    detail: str


@dataclass(frozen=True)
class CompetitorAlertInput:
    """
    Input for a competitor rating change or review spike alert.

    Attributes:
        business_id:          Business UUID.
        business_name:        Business display name.
        alert_type:           AlertType.COMPETITOR_RATING or
                              AlertType.COMPETITOR_REVIEW_SPIKE.
        competitor_name:      Name of the competitor.
        competitor_place_id:  Google Place ID (used in dedup key).
        detail:               Human-readable description of the change.
        their_current_rating: Competitor's current rating (optional).
        their_review_count:   Competitor's total review count (optional).
        new_reviews:          Reviews gained since last scan (optional).
    """
    business_id: str
    business_name: str
    alert_type: str
    competitor_name: str
    competitor_place_id: str
    detail: str
    their_current_rating: Optional[float] = None
    their_review_count: Optional[int] = None
    new_reviews: Optional[int] = None


# ==============================================================================
# Output dataclasses
# ==============================================================================

@dataclass
class AlertDispatchResult:
    """
    Summary of an alert dispatch run for one business.

    Attributes:
        business_id:   Business UUID.
        sent:          Number of alerts successfully delivered via WhatsApp.
        skipped:       Number of alerts skipped due to deduplication.
        failed:        Number of alerts that failed during delivery.
        alert_ids:     Database IDs of persisted alert records (sent + failed).
        skip_reasons:  List of (alert_type, reason) for skipped alerts.
        errors:        List of (alert_type, error_message) for failures.
    """
    business_id: str
    sent: int = 0
    skipped: int = 0
    failed: int = 0
    alert_ids: list[str] = field(default_factory=list)
    skip_reasons: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.sent + self.skipped + self.failed

    @property
    def has_failures(self) -> bool:
        return self.failed > 0

    def __str__(self) -> str:
        return (
            f"AlertDispatchResult("
            f"business={self.business_id} "
            f"sent={self.sent} "
            f"skipped={self.skipped} "
            f"failed={self.failed})"
        )


# ==============================================================================
# Alert Manager
# ==============================================================================

class AlertManager:
    """
    Central controller for all business alert detection and delivery.

    Injected dependencies:
      - alert_repo:      Persists and deduplicates alert records.
      - whatsapp_service: Delivers alert messages via WhatsApp Cloud API.

    The alert manager coordinates between the persistence layer and the
    delivery layer. It never calls Google APIs or OpenAI directly.

    Usage:
        manager = AlertManager(
            alert_repo=alert_repo,
            whatsapp_service=whatsapp_service,
        )

        result = await manager.dispatch_review_alert(
            db=db,
            input=ReviewAlertInput(...),
            whatsapp_number="+919876543210",
        )
    """

    def __init__(
        self,
        alert_repo: AlertRepository,
        whatsapp_service,          # app.notifications.whatsapp_service.WhatsAppService
    ) -> None:
        self._alert_repo = alert_repo
        self._whatsapp = whatsapp_service

    # ------------------------------------------------------------------
    # Review Alerts
    # ------------------------------------------------------------------

    async def dispatch_review_alert(
        self,
        db: AsyncSession,
        input: ReviewAlertInput,
        whatsapp_number: str,
    ) -> AlertDispatchResult:
        """
        Dispatch a review-triggered alert (negative, positive, or spike).

        Dedup key includes review_id so the same review never fires twice.

        Args:
            db:               AsyncSession.
            input:            ReviewAlertInput with review context.
            whatsapp_number:  Destination WhatsApp number for delivery.

        Returns:
            AlertDispatchResult. Never raises.
        """
        result = AlertDispatchResult(business_id=input.business_id)
        today = today_local()

        dedup_key = _make_review_alert_key(
            input.business_id, input.alert_type, input.review_id, today
        )
        severity = _review_alert_severity(input.alert_type, input.rating)

        title, message, extra_lines = _build_review_alert_content(input)

        await self._process_alert(
            db=db,
            business_id=input.business_id,
            alert_type=input.alert_type,
            severity=severity,
            title=title,
            message=message,
            dedup_key=dedup_key,
            reference_id=input.review_id,
            context_data=_review_context(input),
            whatsapp_number=whatsapp_number,
            business_name=input.business_name,
            extra_lines=extra_lines,
            result=result,
            today=today,
        )
        return result

    # ------------------------------------------------------------------
    # Rating Drop Alerts
    # ------------------------------------------------------------------

    async def dispatch_rating_drop_alert(
        self,
        db: AsyncSession,
        input: RatingDropInput,
        whatsapp_number: str,
    ) -> AlertDispatchResult:
        """
        Dispatch a rating drop alert.

        One alert per business per day — subsequent drops on the same day
        are deduplicated.

        Args:
            db:               AsyncSession.
            input:            RatingDropInput with rating change context.
            whatsapp_number:  Destination WhatsApp number.

        Returns:
            AlertDispatchResult. Never raises.
        """
        result = AlertDispatchResult(business_id=input.business_id)
        today = today_local()

        dedup_key = _make_daily_key(
            input.business_id, AlertType.RATING_DROP, today
        )
        severity = _rating_drop_severity(input.drop_amount)

        title = f"Your rating dropped by {input.drop_amount:.1f} stars"
        message = (
            f"Your Google rating has dropped from "
            f"{input.previous_rating:.1f} to {input.current_rating:.1f} "
            f"over the {input.period_label}. "
            f"Review your recent negative feedback and respond promptly."
        )

        await self._process_alert(
            db=db,
            business_id=input.business_id,
            alert_type=AlertType.RATING_DROP,
            severity=severity,
            title=title,
            message=message,
            dedup_key=dedup_key,
            reference_id=None,
            context_data={
                "current_rating": input.current_rating,
                "previous_rating": input.previous_rating,
                "drop_amount": input.drop_amount,
                "period_label": input.period_label,
            },
            whatsapp_number=whatsapp_number,
            business_name=input.business_name,
            extra_lines=[
                f"Current Rating: {input.current_rating:.1f} / 5.0",
                f"Previous Rating: {input.previous_rating:.1f} / 5.0",
            ],
            result=result,
            today=today,
        )
        return result

    # ------------------------------------------------------------------
    # Sales Anomaly Alerts
    # ------------------------------------------------------------------

    async def dispatch_sales_alert(
        self,
        db: AsyncSession,
        input: SalesAnomalyInput,
        whatsapp_number: str,
    ) -> AlertDispatchResult:
        """
        Dispatch a sales trend or opportunity alert.

        One alert per alert_type per business per day.

        Args:
            db:               AsyncSession.
            input:            SalesAnomalyInput with revenue context.
            whatsapp_number:  Destination WhatsApp number.

        Returns:
            AlertDispatchResult. Never raises.
        """
        result = AlertDispatchResult(business_id=input.business_id)
        today = today_local()

        dedup_key = _make_daily_key(input.business_id, input.alert_type, today)
        severity = _sales_alert_severity(input.change_pct, input.alert_type)

        sign = "+" if input.change_pct >= 0 else ""
        pct_display = f"{sign}{input.change_pct * 100:.1f}%"

        if input.alert_type == AlertType.OPPORTUNITY:
            title = f"Revenue opportunity detected ({pct_display})"
        else:
            title = f"Sales trend alert: {pct_display} vs prior period"

        await self._process_alert(
            db=db,
            business_id=input.business_id,
            alert_type=input.alert_type,
            severity=severity,
            title=title,
            message=input.detail,
            dedup_key=dedup_key,
            reference_id=None,
            context_data={
                "change_pct": input.change_pct,
                "period_label": input.period_label,
            },
            whatsapp_number=whatsapp_number,
            business_name=input.business_name,
            extra_lines=[f"Period: {input.period_label}"],
            result=result,
            today=today,
        )
        return result

    # ------------------------------------------------------------------
    # Competitor Alerts
    # ------------------------------------------------------------------

    async def dispatch_competitor_alert(
        self,
        db: AsyncSession,
        input: CompetitorAlertInput,
        whatsapp_number: str,
    ) -> AlertDispatchResult:
        """
        Dispatch a competitor rating change or review spike alert.

        Dedup key includes competitor_place_id so separate competitors
        each get their own dedup window.

        Args:
            db:               AsyncSession.
            input:            CompetitorAlertInput with competitor context.
            whatsapp_number:  Destination WhatsApp number.

        Returns:
            AlertDispatchResult. Never raises.
        """
        result = AlertDispatchResult(business_id=input.business_id)
        today = today_local()

        dedup_key = _make_competitor_key(
            input.business_id,
            input.alert_type,
            input.competitor_place_id,
            today,
        )
        severity = AlertSeverity.MEDIUM

        title = (
            f"Competitor update: {input.competitor_name}"
            if input.alert_type == AlertType.COMPETITOR_RATING
            else f"Competitor review spike: {input.competitor_name}"
        )

        extra_lines: list[str] = []
        if input.their_current_rating is not None:
            extra_lines.append(
                f"Their Rating: {input.their_current_rating:.1f} / 5.0"
            )
        if input.new_reviews is not None:
            extra_lines.append(f"New Reviews: +{input.new_reviews}")

        await self._process_alert(
            db=db,
            business_id=input.business_id,
            alert_type=input.alert_type,
            severity=severity,
            title=title,
            message=input.detail,
            dedup_key=dedup_key,
            reference_id=input.competitor_place_id,
            context_data={
                "competitor_name": input.competitor_name,
                "competitor_place_id": input.competitor_place_id,
                "their_rating": input.their_current_rating,
                "new_reviews": input.new_reviews,
            },
            whatsapp_number=whatsapp_number,
            business_name=input.business_name,
            extra_lines=extra_lines,
            result=result,
            today=today,
        )
        return result

    # ------------------------------------------------------------------
    # Batch dispatch — process multiple alerts in one call
    # ------------------------------------------------------------------

    async def dispatch_all_pending(
        self,
        db: AsyncSession,
        business_id: str,
        whatsapp_number: str,
        max_alerts: int = 10,
    ) -> AlertDispatchResult:
        """
        Fetch all unsent alerts for a business and dispatch them.

        Used by the review_monitor scheduler to flush any alerts that
        were persisted but not yet delivered (e.g. after a WhatsApp
        outage window).

        Args:
            db:               AsyncSession.
            business_id:      Business UUID.
            whatsapp_number:  Destination WhatsApp number.
            max_alerts:       Maximum alerts to dispatch in one call.

        Returns:
            AlertDispatchResult. Never raises.
        """
        result = AlertDispatchResult(business_id=business_id)

        try:
            unsent = await self._alert_repo.get_unsent(
                db=db,
                business_id=business_id,
                limit=max_alerts,
            )

            if not unsent:
                logger.debug(
                    "No unsent alerts found",
                    extra={
                        "service": ServiceName.ALERTS,
                        "business_id": business_id,
                    },
                )
                return result

            for alert in unsent:
                await self._deliver_persisted_alert(
                    db=db,
                    alert=alert,
                    whatsapp_number=whatsapp_number,
                    result=result,
                )

        except Exception as exc:
            logger.error(
                "dispatch_all_pending failed",
                extra={
                    "service": ServiceName.ALERTS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )

        return result

    # ------------------------------------------------------------------
    # Core internal orchestration
    # ------------------------------------------------------------------

    async def _process_alert(
        self,
        db: AsyncSession,
        business_id: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        dedup_key: str,
        reference_id: Optional[str],
        context_data: dict,
        whatsapp_number: str,
        business_name: str,
        extra_lines: list[str],
        result: AlertDispatchResult,
        today: date,
    ) -> None:
        """
        Core pipeline: dedup → persist → deliver → mark sent/failed.

        This private method is called by every public dispatch method.
        Centralising the pipeline here ensures every alert type goes
        through identical dedup, persistence, and delivery logic.

        Args:
            All fields required to evaluate, persist and deliver one alert.
            result: Mutated in place with sent/skipped/failed counts.
        """
        log_extra = {
            "service": ServiceName.ALERTS,
            "business_id": business_id,
            "alert_type": alert_type,
            "dedup_key": dedup_key,
        }

        # ------ Step 1: Deduplication check ------
        try:
            existing = await self._alert_repo.get_by_deduplication_key(
                db=db, deduplication_key=dedup_key
            )
            if existing is not None:
                logger.debug(
                    "Alert skipped — duplicate key exists",
                    extra={**log_extra, "existing_alert_id": str(existing.id)},
                )
                result.skipped += 1
                result.skip_reasons.append((alert_type, "duplicate_key"))
                return
        except Exception as exc:
            logger.error(
                "Alert dedup check failed — proceeding without dedup",
                extra={**log_extra, "error": str(exc)},
            )

        # ------ Step 2: Persist alert record ------
        try:
            import json as _json
            alert = await self._alert_repo.create(
                db=db,
                business_id=business_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                deduplication_key=dedup_key,
                reference_id=reference_id,
                context_data=_json.dumps(context_data),
                alert_date=today,
            )
            await db.commit()
            result.alert_ids.append(str(alert.id))
        except Exception as exc:
            logger.error(
                "Alert persistence failed — skipping delivery",
                extra={**log_extra, "error": str(exc)},
            )
            await db.rollback()
            result.failed += 1
            result.errors.append((alert_type, f"persist_failed: {exc}"))
            return

        # ------ Step 3: Build WhatsApp message ------
        whatsapp_text = format_alert_message(
            business_name=business_name,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            extra_lines=extra_lines or None,
        )

        # ------ Step 4: Deliver via WhatsApp ------
        await self._deliver_alert(
            db=db,
            alert=alert,
            whatsapp_number=whatsapp_number,
            whatsapp_text=whatsapp_text,
            result=result,
            log_extra=log_extra,
        )

    async def _deliver_alert(
        self,
        db: AsyncSession,
        alert: AlertModel,
        whatsapp_number: str,
        whatsapp_text: str,
        result: AlertDispatchResult,
        log_extra: dict,
    ) -> None:
        """Attempt WhatsApp delivery and mark the alert record accordingly."""
        try:
            await self._whatsapp.send_text_message(
                to=whatsapp_number,
                text=whatsapp_text,
            )
            await self._alert_repo.mark_sent(db=db, alert_id=str(alert.id))
            await db.commit()
            result.sent += 1
            logger.info(
                "Alert sent successfully",
                extra={
                    **log_extra,
                    "alert_id": str(alert.id),
                    "whatsapp_to": whatsapp_number[-4:],  # log last 4 digits only
                },
            )
        except Exception as exc:
            await self._alert_repo.record_send_failure(
                db=db,
                alert_id=str(alert.id),
                error_message=str(exc),
            )
            await db.commit()
            result.failed += 1
            result.errors.append((str(alert.alert_type), str(exc)))
            logger.error(
                "Alert WhatsApp delivery failed",
                extra={
                    **log_extra,
                    "alert_id": str(alert.id),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    async def _deliver_persisted_alert(
        self,
        db: AsyncSession,
        alert: AlertModel,
        whatsapp_number: str,
        result: AlertDispatchResult,
    ) -> None:
        """Re-attempt delivery of a previously persisted but undelivered alert."""
        whatsapp_text = format_alert_message(
            business_name="",   # name not stored; title+message are sufficient
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
        )
        log_extra = {
            "service": ServiceName.ALERTS,
            "business_id": str(alert.business_id),
            "alert_type": alert.alert_type,
            "alert_id": str(alert.id),
        }
        await self._deliver_alert(
            db=db,
            alert=alert,
            whatsapp_number=whatsapp_number,
            whatsapp_text=whatsapp_text,
            result=result,
            log_extra=log_extra,
        )


# ==============================================================================
# Deduplication key builders
# ==============================================================================

def _make_review_alert_key(
    business_id: str,
    alert_type: str,
    review_id: str,
    today: date,
) -> str:
    """
    Dedup key for review-level alerts.
    Includes review_id so each review gets at most one alert.
    """
    return f"ALERT_{alert_type}_{business_id}_{review_id}_{today.isoformat()}"


def _make_daily_key(
    business_id: str,
    alert_type: str,
    today: date,
) -> str:
    """
    Dedup key for daily-level alerts (rating drop, sales trend).
    One alert per type per business per day.
    """
    return f"ALERT_{alert_type}_{business_id}_{today.isoformat()}"


def _make_competitor_key(
    business_id: str,
    alert_type: str,
    competitor_place_id: str,
    today: date,
) -> str:
    """
    Dedup key for competitor alerts.
    One alert per competitor per type per business per day.
    """
    return (
        f"ALERT_{alert_type}_{business_id}"
        f"_{competitor_place_id}_{today.isoformat()}"
    )


# ==============================================================================
# Severity classifiers
# ==============================================================================

def _review_alert_severity(alert_type: str, rating: int) -> str:
    """Classify severity of a review alert from its type and star rating."""
    if alert_type == AlertType.NEGATIVE_REVIEW:
        return AlertSeverity.CRITICAL if rating == 1 else AlertSeverity.HIGH
    if alert_type == AlertType.REVIEW_SPIKE:
        return AlertSeverity.MEDIUM
    # POSITIVE_REVIEW
    return AlertSeverity.LOW


def _rating_drop_severity(drop_amount: float) -> str:
    """Classify severity of a rating drop by magnitude."""
    if drop_amount >= 0.5:
        return AlertSeverity.CRITICAL
    if drop_amount >= 0.3:
        return AlertSeverity.HIGH
    return AlertSeverity.MEDIUM


def _sales_alert_severity(change_pct: float, alert_type: str) -> str:
    """Classify severity of a sales anomaly."""
    if alert_type == AlertType.OPPORTUNITY:
        return AlertSeverity.LOW
    # SALES_TREND (drop)
    if change_pct <= -0.30:
        return AlertSeverity.HIGH
    return AlertSeverity.MEDIUM


# ==============================================================================
# Alert content builders
# ==============================================================================

def _build_review_alert_content(
    input: ReviewAlertInput,
) -> tuple[str, str, list[str]]:
    """
    Build title, message body, and extra lines for a review alert.

    Returns:
        tuple[title, message, extra_lines]
    """
    if input.alert_type == AlertType.REVIEW_SPIKE:
        title = f"{input.spike_count or 'Multiple'} new reviews received"
        message = (
            f"Your business received {input.spike_count or 'multiple'} new reviews "
            f"in a short period. Review them promptly and respond to any negative feedback."
        )
        extra_lines: list[str] = []

    elif input.alert_type == AlertType.NEGATIVE_REVIEW:
        title = f"Negative review from {input.reviewer_name} ({input.rating}★)"
        message = (
            f"A {input.rating}-star review has been posted. "
            f"A professional response has been "
            + ("auto-generated and posted." if input.ai_reply_generated
               else "not yet posted — please respond promptly.")
        )
        extra_lines = [f'"{input.review_excerpt}"'] if input.review_excerpt else []

    else:
        # POSITIVE_REVIEW
        title = f"Great review from {input.reviewer_name} ({input.rating}★)"
        message = (
            f"A {input.rating}-star review was posted. "
            + ("An AI thank-you reply has been posted automatically." 
               if input.ai_reply_generated
               else "Consider posting a thank-you reply.")
        )
        extra_lines = [f'"{input.review_excerpt}"'] if input.review_excerpt else []

    return title, message, extra_lines


def _review_context(input: ReviewAlertInput) -> dict:
    """Build the context_data dict for a review alert record."""
    return {
        "review_id": input.review_id,
        "reviewer_name": input.reviewer_name,
        "rating": input.rating,
        "sentiment": input.sentiment,
        "ai_reply_generated": input.ai_reply_generated,
        "spike_count": input.spike_count,
    }