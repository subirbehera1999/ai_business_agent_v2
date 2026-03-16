# ==============================================================================
# File: app/alerts/competitor_alerts.py
# Purpose: Processes CompetitorScanResult deltas and produces
#          CompetitorAlertInput objects for AlertManager.
#
#          Detection signals (evaluated per competitor):
#
#            1. COMPETITOR RATING CHANGE
#               A competitor's rating changed by >= RATING_CHANGE_THRESHOLD
#               since the last scan.
#               Direction matters:
#                 - Competitor rating goes UP   → strategic warning
#                   "A competitor improved — you may lose relative position"
#                 - Competitor rating goes DOWN → opportunity signal
#                   "A competitor weakened — good time to attract their customers"
#               Alert type: AlertType.COMPETITOR_RATING
#
#            2. COMPETITOR REVIEW SPIKE
#               A competitor gained significantly more reviews than usual
#               in a single scan window (already detected by competitor_service.py
#               as CompetitorDelta.is_review_spike=True).
#               Interpretation: they may be running a review campaign,
#               a promotion, or receiving viral attention.
#               Alert type: AlertType.COMPETITOR_REVIEW_SPIKE
#
#            3. COMPETITOR NOW LEADS (rating overtake)
#               A competitor's rating has crossed above our own for the
#               first time (or after a period of us leading).
#               This is a higher-priority variant of COMPETITOR_RATING.
#               Severity: HIGH (vs MEDIUM for normal rating change).
#
#          Output:
#            CompetitorAlertDetectionResult — contains zero or more
#            CompetitorAlertInput objects, one per (competitor, signal_type).
#            Multiple competitors can trigger alerts in the same run.
#
#          Inputs:
#            CompetitorScanResult from competitor_service.py.
#            Fully synchronous — no DB calls, no async, no OpenAI.
#
#          Deduplication:
#            Each CompetitorAlertInput carries a competitor_place_id which
#            AlertManager uses to build a per-competitor-per-day dedup key.
#            Multiple signals for the same competitor on the same day are
#            each deduplicated independently (different alert_type = different key).
#
#          Multi-tenant:
#            business_id flows through every output object and log entry.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from app.alerts.alert_manager import CompetitorAlertInput
from app.config.constants import AlertSeverity, AlertType, ServiceName
from app.services.competitor_service import (
    CompetitorDelta,
    CompetitorScanResult,
)

logger = logging.getLogger(ServiceName.ALERTS)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Minimum rating change to trigger a COMPETITOR_RATING alert
RATING_CHANGE_THRESHOLD: float = 0.2

# Rating change that escalates alert severity from MEDIUM to HIGH
HIGH_SEVERITY_RATING_CHANGE: float = 0.4

# Minimum new reviews for a spike to generate a COMPETITOR_REVIEW_SPIKE alert
# (competitor_service.py already validated is_review_spike=True, but we
# apply a second floor here to avoid alerting on tiny spikes)
MIN_NEW_REVIEWS_FOR_ALERT: int = 3


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class CompetitorSignal:
    """
    A single detected competitor alert signal for one competitor.

    Attributes:
        signal_type:          One of: "rating_change_up", "rating_change_down",
                              "review_spike", "rating_overtake".
        competitor_place_id:  Google Place ID of the triggering competitor.
        competitor_name:      Display name of the competitor.
        alert_type:           AlertType constant for the AlertManager.
        severity:             AlertSeverity constant.
        detail:               Actionable description for WhatsApp notification.
        their_current_rating: Competitor's rating after the change.
        their_previous_rating:Competitor's rating before the change.
        rating_change:        Delta (positive = they improved).
        new_reviews:          Number of new reviews (for spike signals).
        our_rating:           Our business's current rating (for context).
    """
    signal_type: str
    competitor_place_id: str
    competitor_name: str
    alert_type: str
    severity: str
    detail: str
    their_current_rating: Optional[float] = None
    their_previous_rating: Optional[float] = None
    rating_change: Optional[float] = None
    new_reviews: Optional[int] = None
    our_rating: Optional[float] = None

    def to_alert_input(self, business_id: str, business_name: str) -> CompetitorAlertInput:
        """Convert this signal into a CompetitorAlertInput for the AlertManager."""
        return CompetitorAlertInput(
            business_id=business_id,
            business_name=business_name,
            alert_type=self.alert_type,
            competitor_name=self.competitor_name,
            competitor_place_id=self.competitor_place_id,
            detail=self.detail,
            their_current_rating=self.their_current_rating,
            their_review_count=None,
            new_reviews=self.new_reviews,
        )


@dataclass
class CompetitorAlertDetectionResult:
    """
    Complete output of a competitor alert detection run for one business.

    Attributes:
        business_id:      Business UUID.
        detection_date:   Date the detection ran.
        signals:          All signals that crossed thresholds.
        alert_inputs:     CompetitorAlertInput objects ready for AlertManager.
        competitors_checked: Number of competitor deltas evaluated.
        skipped:          True if detection was skipped (no scan data, etc.).
        skip_reason:      Human-readable reason if skipped.
    """
    business_id: str
    detection_date: date
    signals: list[CompetitorSignal] = field(default_factory=list)
    alert_inputs: list[CompetitorAlertInput] = field(default_factory=list)
    competitors_checked: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None

    @property
    def has_alerts(self) -> bool:
        return len(self.alert_inputs) > 0

    @property
    def rating_change_signals(self) -> list[CompetitorSignal]:
        return [s for s in self.signals if s.alert_type == AlertType.COMPETITOR_RATING]

    @property
    def review_spike_signals(self) -> list[CompetitorSignal]:
        return [s for s in self.signals if s.alert_type == AlertType.COMPETITOR_REVIEW_SPIKE]

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"CompetitorAlertDetectionResult("
                f"business={self.business_id} "
                f"skipped=True reason={self.skip_reason})"
            )
        return (
            f"CompetitorAlertDetectionResult("
            f"business={self.business_id} "
            f"competitors_checked={self.competitors_checked} "
            f"signals={len(self.signals)} "
            f"alerts={len(self.alert_inputs)})"
        )


# ==============================================================================
# Competitor Alerts Detector
# ==============================================================================

class CompetitorAlertsDetector:
    """
    Processes a CompetitorScanResult and detects alertable competitor events.

    Fully synchronous and stateless — no DB calls, no async, no OpenAI.
    The caller (scheduler) is responsible for running the competitor scan
    upstream and passing the result here.

    Usage:
        detector = CompetitorAlertsDetector()

        result = detector.detect(
            scan_result=competitor_scan_result,
            business_id="uuid",
            business_name="Raj Restaurant",
        )

        for alert_input in result.alert_inputs:
            await alert_manager.dispatch_competitor_alert(
                db=db,
                input=alert_input,
                whatsapp_number="+919876543210",
            )
    """

    def detect(
        self,
        scan_result: CompetitorScanResult,
        business_id: str,
        business_name: str,
        detection_date: Optional[date] = None,
    ) -> CompetitorAlertDetectionResult:
        """
        Run all competitor alert signals against a CompetitorScanResult.

        Evaluates each CompetitorDelta in the scan result independently.
        Multiple competitors can produce signals in a single run.

        Args:
            scan_result:    Output from competitor_service.CompetitorService.scan_competitors().
            business_id:    Business UUID for logging and output objects.
            business_name:  Business display name for alert content.
            detection_date: Override today's date (for testing).

        Returns:
            CompetitorAlertDetectionResult. Never raises.
        """
        today = detection_date or date.today()
        log_extra = {
            "service": ServiceName.ALERTS,
            "business_id": business_id,
        }

        # Guard: scan was rate-limited or empty
        if scan_result.rate_limited:
            logger.debug(
                "Competitor alert detection skipped — scan was rate-limited",
                extra=log_extra,
            )
            return CompetitorAlertDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason="scan_rate_limited",
            )

        if not scan_result.deltas:
            logger.debug(
                "Competitor alert detection skipped — no scan deltas",
                extra=log_extra,
            )
            return CompetitorAlertDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason="no_competitor_deltas",
            )

        our_rating = scan_result.our_avg_rating
        signals: list[CompetitorSignal] = []

        for delta in scan_result.deltas:
            # Skip first-scan deltas — no prior data to compare against
            if delta.is_first_scan:
                logger.debug(
                    "Competitor delta skipped — first scan (no baseline)",
                    extra={**log_extra, "competitor": delta.name},
                )
                continue

            # Signal 1 & 3: Rating change (and potential overtake)
            rating_signal = _evaluate_rating_change(
                delta=delta,
                our_rating=our_rating,
                log_extra=log_extra,
            )
            if rating_signal:
                signals.append(rating_signal)

            # Signal 2: Review spike
            spike_signal = _evaluate_review_spike(
                delta=delta,
                log_extra=log_extra,
            )
            if spike_signal:
                signals.append(spike_signal)

        # Convert to alert inputs
        alert_inputs = [
            s.to_alert_input(business_id, business_name)
            for s in signals
        ]

        result = CompetitorAlertDetectionResult(
            business_id=business_id,
            detection_date=today,
            signals=signals,
            alert_inputs=alert_inputs,
            competitors_checked=len(scan_result.deltas),
        )

        log_level = logging.INFO if result.has_alerts else logging.DEBUG
        logger.log(
            log_level,
            "Competitor alert detection complete",
            extra={
                **log_extra,
                "competitors_checked": result.competitors_checked,
                "signals_detected": len(signals),
                "signal_types": [s.signal_type for s in signals],
            },
        )

        return result


# ==============================================================================
# Detection functions — pure, synchronous, individually testable
# ==============================================================================

def _evaluate_rating_change(
    delta: CompetitorDelta,
    our_rating: Optional[float],
    log_extra: dict,
) -> Optional[CompetitorSignal]:
    """
    Evaluate whether a competitor's rating change crosses the alert threshold.

    Handles three sub-cases:
      a. Competitor improved + crossed above our rating (overtake) → HIGH
      b. Competitor improved (no overtake)                         → MEDIUM
      c. Competitor declined                                        → LOW (opportunity)

    Args:
        delta:      CompetitorDelta from the current scan.
        our_rating: Our business's current average rating (for overtake check).
        log_extra:  Structured log context.

    Returns:
        CompetitorSignal or None.
    """
    if delta.rating_change is None:
        return None

    if abs(delta.rating_change) < RATING_CHANGE_THRESHOLD:
        return None

    change = delta.rating_change  # positive = competitor improved

    # Determine signal sub-type and message
    if change > 0:
        # Competitor rating went UP
        overtake = _check_overtake(
            delta=delta,
            our_rating=our_rating,
        )

        if overtake:
            signal_type = "rating_overtake"
            severity = AlertSeverity.HIGH
            detail = (
                f"{delta.name} has overtaken your rating. "
                f"Their rating improved from {delta.previous_rating:.1f} to "
                f"{delta.current_rating:.1f} (▲{change:.1f}). "
                f"They now rank higher than you in local search. "
                f"Focus on increasing your review volume and quality this week."
            )
            if our_rating:
                detail += f" Your current rating: {our_rating:.1f}."
        else:
            signal_type = "rating_change_up"
            severity = (
                AlertSeverity.HIGH
                if change >= HIGH_SEVERITY_RATING_CHANGE
                else AlertSeverity.MEDIUM
            )
            detail = (
                f"{delta.name}'s rating improved by ▲{change:.1f} "
                f"({delta.previous_rating:.1f} → {delta.current_rating:.1f}). "
                f"They are getting more positive reviews. "
                f"Monitor their activity and ensure your service quality stays strong."
            )
    else:
        # Competitor rating went DOWN — opportunity signal
        signal_type = "rating_change_down"
        severity = AlertSeverity.LOW
        abs_change = abs(change)
        detail = (
            f"{delta.name}'s rating dropped by ▼{abs_change:.1f} "
            f"({delta.previous_rating:.1f} → {delta.current_rating:.1f}). "
            f"This is an opportunity — their dissatisfied customers may be "
            f"looking for an alternative. Consider a targeted promotion "
            f"or highlight what sets you apart this week."
        )

    logger.debug(
        "Competitor rating change signal detected",
        extra={
            **log_extra,
            "competitor": delta.name,
            "signal_type": signal_type,
            "rating_change": change,
            "severity": severity,
        },
    )

    return CompetitorSignal(
        signal_type=signal_type,
        competitor_place_id=delta.place_id,
        competitor_name=delta.name,
        alert_type=AlertType.COMPETITOR_RATING,
        severity=severity,
        detail=detail,
        their_current_rating=delta.current_rating,
        their_previous_rating=delta.previous_rating,
        rating_change=change,
        our_rating=our_rating,
    )


def _evaluate_review_spike(
    delta: CompetitorDelta,
    log_extra: dict,
) -> Optional[CompetitorSignal]:
    """
    Evaluate whether a competitor's review count spike warrants an alert.

    competitor_service.py already validated is_review_spike=True using
    both absolute (>=3) and percentage (>=15%) criteria. We apply a
    second absolute floor here (MIN_NEW_REVIEWS_FOR_ALERT) to catch
    any cases where the service threshold was configured differently.

    Args:
        delta:    CompetitorDelta from the current scan.
        log_extra: Structured log context.

    Returns:
        CompetitorSignal or None.
    """
    if not delta.is_review_spike:
        return None

    new_reviews = delta.new_reviews or 0
    if new_reviews < MIN_NEW_REVIEWS_FOR_ALERT:
        return None

    detail = (
        f"{delta.name} received {new_reviews} new reviews recently "
        f"(total: {delta.current_reviews:,}). "
        f"This spike may indicate a marketing campaign, a promotion, "
        f"or viral attention. Monitor their reviews to understand "
        f"what is driving the engagement and consider your own response strategy."
    )

    logger.debug(
        "Competitor review spike signal detected",
        extra={
            **log_extra,
            "competitor": delta.name,
            "new_reviews": new_reviews,
        },
    )

    return CompetitorSignal(
        signal_type="review_spike",
        competitor_place_id=delta.place_id,
        competitor_name=delta.name,
        alert_type=AlertType.COMPETITOR_REVIEW_SPIKE,
        severity=AlertSeverity.MEDIUM,
        detail=detail,
        their_current_rating=delta.current_rating,
        new_reviews=new_reviews,
    )


def _check_overtake(
    delta: CompetitorDelta,
    our_rating: Optional[float],
) -> bool:
    """
    Determine if a competitor has overtaken our rating with this improvement.

    An overtake is detected when:
      1. We have our own rating to compare
      2. The competitor's previous rating was <= ours (they were behind or equal)
      3. The competitor's current rating is now > ours (they are ahead)

    Args:
        delta:      CompetitorDelta with current and previous ratings.
        our_rating: Our business's current average rating.

    Returns:
        bool: True if the competitor just crossed above our rating.
    """
    if our_rating is None:
        return False
    if delta.previous_rating is None or delta.current_rating is None:
        return False

    was_behind_or_equal = delta.previous_rating <= our_rating
    is_now_ahead = delta.current_rating > our_rating

    return was_behind_or_equal and is_now_ahead


# ==============================================================================
# Convenience: batch detection for scheduler use
# ==============================================================================

def detect_competitor_alerts_for_businesses(
    detector: CompetitorAlertsDetector,
    scan_results: dict[str, CompetitorScanResult],
    business_names: dict[str, str],
    detection_date: Optional[date] = None,
) -> list[CompetitorAlertDetectionResult]:
    """
    Run competitor alert detection for a batch of businesses.

    Fully synchronous — the scheduler awaits all competitor scans upstream
    and passes the pre-computed results here as a dict.

    Each business is processed independently. A failure for one business
    never affects others (failure isolation contract).

    Args:
        detector:       Shared CompetitorAlertsDetector instance.
        scan_results:   Dict mapping business_id → CompetitorScanResult.
        business_names: Dict mapping business_id → business_name.
        detection_date: Override date for testing.

    Returns:
        list[CompetitorAlertDetectionResult]: One result per business.
    """
    results: list[CompetitorAlertDetectionResult] = []
    today = detection_date or date.today()

    for business_id, scan_result in scan_results.items():
        business_name = business_names.get(business_id, "Unknown Business")
        try:
            result = detector.detect(
                scan_result=scan_result,
                business_id=business_id,
                business_name=business_name,
                detection_date=today,
            )
        except Exception as exc:
            logger.error(
                "Competitor alert detection error for business — skipping",
                extra={
                    "service": ServiceName.ALERTS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )
            result = CompetitorAlertDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason=f"unhandled_error: {exc}",
            )
        results.append(result)

    alert_count = sum(1 for r in results if r.has_alerts)
    logger.info(
        "Batch competitor alert detection complete",
        extra={
            "service": ServiceName.ALERTS,
            "businesses_checked": len(scan_results),
            "alerts_detected": alert_count,
        },
    )

    return results