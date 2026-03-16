# ==============================================================================
# File: app/alerts/sales_alerts.py
# Purpose: Detects revenue anomalies in a business's sales data and produces
#          SalesAnomalyInput objects for AlertManager.dispatch_sales_alert().
#
#          Detection signals (all independent — any can trigger an alert):
#
#            1. SUSTAINED DROP
#               Current window revenue is significantly below the baseline.
#               Threshold: revenue_change_pct <= -SUSTAINED_DROP_THRESHOLD
#               Severity: HIGH if >= 30% drop, MEDIUM otherwise.
#               This is the most actionable signal — revenue has been
#               trending down for a full window period, not just one bad day.
#
#            2. SHARP DAILY DROP
#               Today's revenue is significantly below the rolling daily average.
#               Threshold: today_revenue < daily_avg * SHARP_DROP_MULTIPLIER
#               Uses: analytics.daily_sales[-1] vs analytics.avg_daily_revenue
#               Catches: single-day crashes (system down, bad weather, etc.)
#
#            3. REVENUE OPPORTUNITY (positive spike)
#               Current window revenue is significantly above the baseline.
#               Threshold: revenue_change_pct >= OPPORTUNITY_THRESHOLD
#               Alert type: AlertType.OPPORTUNITY (LOW severity)
#               Purpose: Let the business know what's working so they can
#               repeat it (a promotion, a new menu item, a seasonal event).
#
#            4. ZERO REVENUE DAY
#               A day within the detection window had zero recorded transactions.
#               Only fires if adjacent days had transactions (rules out weekday
#               closures — a business closed on Sundays shouldn't alert every
#               Sunday).
#               Alert type: AlertType.SALES_TREND (HIGH severity)
#
#          Output:
#            SalesAnomalyDetectionResult — contains zero or more
#            SalesAnomalyInput objects for the AlertManager.
#            Multiple signals can fire in the same run but deduplication
#            in AlertManager enforces one alert per type per day.
#
#          Inputs:
#            SalesAnalyticsResult from analytics_service.py.
#            This detector is purely computational — no DB calls,
#            no external API calls, no async operations.
#            The scheduler fetches the analytics result and passes it here.
#
#          Multi-tenant:
#            business_id is carried through all log entries and output objects.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

from app.alerts.alert_manager import SalesAnomalyInput
from app.config.constants import AlertType, ServiceName
from app.services.analytics_service import AnalyticsError, SalesAnalyticsResult

logger = logging.getLogger(ServiceName.ALERTS)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Sustained revenue drop vs prior period to trigger SALES_TREND alert
SUSTAINED_DROP_THRESHOLD: float = 0.20        # 20% revenue decline

# Sustained drop magnitude that escalates severity to HIGH
HIGH_SEVERITY_DROP_THRESHOLD: float = 0.30    # 30% revenue decline

# Daily revenue must be below this fraction of the rolling average to
# qualify as a "sharp daily drop"
SHARP_DROP_MULTIPLIER: float = 0.40           # today < 40% of rolling avg

# Positive revenue spike threshold to trigger OPPORTUNITY alert
OPPORTUNITY_THRESHOLD: float = 0.25           # 25% revenue increase

# Minimum absolute revenue (paise) for detection to be meaningful.
# Prevents false alerts on very small revenue bases (e.g. day 1 of onboarding).
MIN_REVENUE_FOR_DETECTION_PAISE: int = 10_000  # ₹100 minimum baseline

# Minimum transaction count for the window to be considered valid
MIN_TRANSACTIONS_FOR_DETECTION: int = 5

# Minimum days of data required before detecting sustained drop/opportunity
MIN_DAYS_FOR_SUSTAINED_DETECTION: int = 5

# Minimum days of data required for sharp daily drop detection
MIN_DAYS_FOR_DAILY_DETECTION: int = 3


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class SalesSignal:
    """
    A single detected sales anomaly signal.

    Attributes:
        signal_type:      One of: "sustained_drop", "sharp_daily_drop",
                          "opportunity", "zero_revenue_day".
        alert_type:       AlertType constant for the AlertManager.
        change_pct:       Revenue change as a float (negative = drop).
        current_revenue:  Revenue for the current period in paise.
        period_label:     Human-readable description of the comparison period.
        detail:           Actionable description for WhatsApp notification.
        severity:         "high", "medium", or "low".
        trigger_date:     The specific date that triggered this signal
                          (for zero_revenue_day, the gap date).
    """
    signal_type: str
    alert_type: str
    change_pct: float
    current_revenue: int          # paise
    period_label: str
    detail: str
    severity: str
    trigger_date: Optional[date] = None

    @property
    def is_positive(self) -> bool:
        """True if this is an opportunity signal (revenue up)."""
        return self.alert_type == AlertType.OPPORTUNITY

    def to_alert_input(self, business_id: str, business_name: str) -> SalesAnomalyInput:
        """Convert this signal into a SalesAnomalyInput for the AlertManager."""
        return SalesAnomalyInput(
            business_id=business_id,
            business_name=business_name,
            alert_type=self.alert_type,
            change_pct=self.change_pct,
            current_revenue=self.current_revenue,
            period_label=self.period_label,
            detail=self.detail,
        )


@dataclass
class SalesAnomalyDetectionResult:
    """
    Complete output of a sales anomaly detection run for one business.

    Attributes:
        business_id:      Business UUID.
        detection_date:   Date the detection ran.
        signals:          All signals that crossed their thresholds.
        alert_inputs:     SalesAnomalyInput objects ready for AlertManager.
        skipped:          True if detection was skipped due to insufficient data.
        skip_reason:      Human-readable reason if skipped.
        analytics_used:   Summary of the analytics data used for detection.
    """
    business_id: str
    detection_date: date
    signals: list[SalesSignal] = field(default_factory=list)
    alert_inputs: list[SalesAnomalyInput] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    analytics_used: Optional[dict] = None

    @property
    def has_alerts(self) -> bool:
        return len(self.alert_inputs) > 0

    @property
    def has_drop_alert(self) -> bool:
        return any(
            s.alert_type == AlertType.SALES_TREND for s in self.signals
        )

    @property
    def has_opportunity_alert(self) -> bool:
        return any(
            s.alert_type == AlertType.OPPORTUNITY for s in self.signals
        )

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"SalesAnomalyDetectionResult("
                f"business={self.business_id} "
                f"skipped=True reason={self.skip_reason})"
            )
        return (
            f"SalesAnomalyDetectionResult("
            f"business={self.business_id} "
            f"signals={len(self.signals)} "
            f"alerts={len(self.alert_inputs)})"
        )


# ==============================================================================
# Sales Alerts Detector
# ==============================================================================

class SalesAlertsDetector:
    """
    Detects revenue anomalies from a SalesAnalyticsResult.

    Fully synchronous and stateless — no DB calls, no async.
    The caller (scheduler) is responsible for fetching the
    SalesAnalyticsResult before calling detect().

    Usage:
        detector = SalesAlertsDetector()

        result = detector.detect(
            analytics=analytics_result,
            business_id="uuid",
            business_name="Raj Restaurant",
        )

        for alert_input in result.alert_inputs:
            await alert_manager.dispatch_sales_alert(
                db=db,
                input=alert_input,
                whatsapp_number="+919876543210",
            )
    """

    def detect(
        self,
        analytics: SalesAnalyticsResult | AnalyticsError,
        business_id: str,
        business_name: str,
        detection_date: Optional[date] = None,
    ) -> SalesAnomalyDetectionResult:
        """
        Run all sales anomaly detection signals against the analytics result.

        Args:
            analytics:      Output from analytics_service.AnalyticsService.analyse().
                            If this is an AnalyticsError, detection is skipped.
            business_id:    Business UUID for logging and output objects.
            business_name:  Business display name for alert content.
            detection_date: Override today's date (for testing).

        Returns:
            SalesAnomalyDetectionResult. Never raises.
        """
        today = detection_date or date.today()
        log_extra = {
            "service": ServiceName.ALERTS,
            "business_id": business_id,
        }

        # Guard: analytics failed upstream — skip detection
        if isinstance(analytics, AnalyticsError):
            logger.debug(
                "Sales anomaly detection skipped — analytics unavailable",
                extra={**log_extra, "reason": analytics.reason},
            )
            return SalesAnomalyDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason=f"analytics_error: {analytics.reason}",
            )

        # Guard: insufficient data for meaningful detection
        skip_reason = _check_minimum_data(analytics)
        if skip_reason:
            logger.debug(
                "Sales anomaly detection skipped — insufficient data",
                extra={**log_extra, "skip_reason": skip_reason},
            )
            return SalesAnomalyDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason=skip_reason,
                analytics_used=_analytics_summary(analytics),
            )

        # Run all detection signals
        signals: list[SalesSignal] = []

        # Signal 1: Sustained revenue drop
        sustained_signal = _detect_sustained_drop(analytics, log_extra)
        if sustained_signal:
            signals.append(sustained_signal)

        # Signal 2: Sharp single-day drop
        sharp_signal = _detect_sharp_daily_drop(analytics, log_extra)
        if sharp_signal:
            signals.append(sharp_signal)

        # Signal 3: Revenue opportunity (positive spike)
        opportunity_signal = _detect_opportunity(analytics, log_extra)
        if opportunity_signal:
            signals.append(opportunity_signal)

        # Signal 4: Zero revenue day
        zero_signal = _detect_zero_revenue_day(analytics, log_extra)
        if zero_signal:
            signals.append(zero_signal)

        # Convert signals to alert inputs
        alert_inputs = [
            s.to_alert_input(business_id, business_name)
            for s in signals
        ]

        result = SalesAnomalyDetectionResult(
            business_id=business_id,
            detection_date=today,
            signals=signals,
            alert_inputs=alert_inputs,
            analytics_used=_analytics_summary(analytics),
        )

        log_level = logging.INFO if result.has_alerts else logging.DEBUG
        logger.log(
            log_level,
            "Sales anomaly detection complete",
            extra={
                **log_extra,
                "signals_detected": len(signals),
                "signal_types": [s.signal_type for s in signals],
            },
        )

        return result


# ==============================================================================
# Detection functions — pure, synchronous, individually testable
# ==============================================================================

def _check_minimum_data(analytics: SalesAnalyticsResult) -> Optional[str]:
    """
    Verify the analytics result has sufficient data for detection.

    Returns a skip_reason string if detection should be skipped,
    or None if detection should proceed.
    """
    if analytics.total_transactions < MIN_TRANSACTIONS_FOR_DETECTION:
        return (
            f"insufficient_transactions: "
            f"{analytics.total_transactions} < {MIN_TRANSACTIONS_FOR_DETECTION}"
        )

    # Convert paise to rupees for the baseline check
    baseline_rupees = int(analytics.prev_period_revenue * 100)
    if analytics.prev_period_revenue == Decimal("0"):
        return "no_baseline_revenue"

    baseline_paise = int(analytics.prev_period_revenue * 100)
    if baseline_paise < MIN_REVENUE_FOR_DETECTION_PAISE:
        return (
            f"baseline_revenue_too_low: "
            f"₹{analytics.prev_period_revenue:.2f} < "
            f"₹{MIN_REVENUE_FOR_DETECTION_PAISE / 100:.2f}"
        )

    return None


def _detect_sustained_drop(
    analytics: SalesAnalyticsResult,
    log_extra: dict,
) -> Optional[SalesSignal]:
    """
    Detect a sustained revenue drop over the full analysis window.

    Compares current window revenue vs prior window revenue using
    the pre-computed revenue_change_pct from the analytics result.

    Returns:
        SalesSignal if drop threshold crossed, else None.
    """
    if len(analytics.daily_sales) < MIN_DAYS_FOR_SUSTAINED_DETECTION:
        return None

    change_pct = analytics.revenue_change_pct

    if change_pct > -SUSTAINED_DROP_THRESHOLD:
        return None

    # Determine severity
    severity = (
        "high" if abs(change_pct) >= HIGH_SEVERITY_DROP_THRESHOLD
        else "medium"
    )

    pct_display = f"{abs(change_pct) * 100:.1f}%"
    current_paise = int(analytics.total_revenue * 100)
    prev_paise = int(analytics.prev_period_revenue * 100)

    detail = (
        f"Revenue dropped by {pct_display} compared to the previous period. "
        f"Current: ₹{analytics.total_revenue:,.2f} | "
        f"Previous: ₹{analytics.prev_period_revenue:,.2f}. "
        f"Review your pricing, promotions, and customer feedback to "
        f"understand what may have changed."
    )

    logger.debug(
        "Sustained revenue drop detected",
        extra={
            **log_extra,
            "change_pct": change_pct,
            "severity": severity,
        },
    )

    return SalesSignal(
        signal_type="sustained_drop",
        alert_type=AlertType.SALES_TREND,
        change_pct=change_pct,
        current_revenue=current_paise,
        period_label=analytics.period_label,
        detail=detail,
        severity=severity,
    )


def _detect_sharp_daily_drop(
    analytics: SalesAnalyticsResult,
    log_extra: dict,
) -> Optional[SalesSignal]:
    """
    Detect a sharp single-day revenue drop vs the rolling daily average.

    Only checks the most recent day in the analytics window.
    Requires at least MIN_DAYS_FOR_DAILY_DETECTION days of data to
    establish a meaningful rolling average.

    Returns:
        SalesSignal if today's revenue is critically low, else None.
    """
    if len(analytics.daily_sales) < MIN_DAYS_FOR_DAILY_DETECTION:
        return None

    if analytics.avg_daily_revenue <= 0:
        return None

    # Most recent day in the window
    latest_day = analytics.daily_sales[-1]
    threshold = analytics.avg_daily_revenue * Decimal(str(SHARP_DROP_MULTIPLIER))

    if latest_day.total_revenue >= threshold:
        return None

    # Avoid firing if the latest day simply has no transactions yet
    # (e.g. detection runs mid-morning before any sales)
    if latest_day.transaction_count == 0:
        return None

    change_pct = float(
        (latest_day.total_revenue - analytics.avg_daily_revenue)
        / analytics.avg_daily_revenue
    )

    pct_display = f"{abs(change_pct) * 100:.1f}%"
    current_paise = int(latest_day.total_revenue * 100)

    detail = (
        f"Today's revenue (₹{latest_day.total_revenue:,.2f}) is {pct_display} "
        f"below your daily average (₹{analytics.avg_daily_revenue:,.2f}). "
        f"Check for any service disruptions, staff availability, "
        f"or external factors affecting footfall today."
    )

    logger.debug(
        "Sharp daily revenue drop detected",
        extra={
            **log_extra,
            "today_revenue": str(latest_day.total_revenue),
            "daily_avg": str(analytics.avg_daily_revenue),
            "change_pct": change_pct,
        },
    )

    return SalesSignal(
        signal_type="sharp_daily_drop",
        alert_type=AlertType.SALES_TREND,
        change_pct=change_pct,
        current_revenue=current_paise,
        period_label=f"{latest_day.date_label} vs daily average",
        detail=detail,
        severity="high",
        trigger_date=latest_day.date,
    )


def _detect_opportunity(
    analytics: SalesAnalyticsResult,
    log_extra: dict,
) -> Optional[SalesSignal]:
    """
    Detect a positive revenue spike as a business opportunity signal.

    A significant revenue increase is worth notifying the business about
    so they can understand what drove it and repeat the behaviour.

    Returns:
        SalesSignal if opportunity threshold crossed, else None.
    """
    if len(analytics.daily_sales) < MIN_DAYS_FOR_SUSTAINED_DETECTION:
        return None

    change_pct = analytics.revenue_change_pct

    if change_pct < OPPORTUNITY_THRESHOLD:
        return None

    pct_display = f"+{change_pct * 100:.1f}%"
    current_paise = int(analytics.total_revenue * 100)

    detail = (
        f"Revenue is up {pct_display} compared to the previous period! "
        f"Current: ₹{analytics.total_revenue:,.2f} | "
        f"Previous: ₹{analytics.prev_period_revenue:,.2f}. "
        f"Identify what drove this growth — a promotion, seasonal demand, "
        f"or positive word-of-mouth — and double down on it."
    )

    # Include peak day context if available
    if analytics.peak_days:
        best = analytics.peak_days[0]
        detail += (
            f" Your best day was {best.day_of_week} "
            f"({best.date.strftime('%d %b')}) with "
            f"₹{best.revenue:,.2f} in revenue."
        )

    logger.debug(
        "Revenue opportunity detected",
        extra={
            **log_extra,
            "change_pct": change_pct,
        },
    )

    return SalesSignal(
        signal_type="opportunity",
        alert_type=AlertType.OPPORTUNITY,
        change_pct=change_pct,
        current_revenue=current_paise,
        period_label=analytics.period_label,
        detail=detail,
        severity="low",
    )


def _detect_zero_revenue_day(
    analytics: SalesAnalyticsResult,
    log_extra: dict,
) -> Optional[SalesSignal]:
    """
    Detect a zero-transaction day within the analytics window.

    Only fires if:
      1. There is a day in the window with zero transactions.
      2. At least one adjacent day (day before or after) had transactions.
         This prevents alerting on a legitimate day-off / closure.
      3. There are at least 3 days of data (prevents false alerts when
         data collection has just begun).

    Returns:
        SalesSignal for the most recent zero-revenue day, or None.
    """
    daily = analytics.daily_sales
    if len(daily) < MIN_DAYS_FOR_DAILY_DETECTION:
        return None

    # Find zero-transaction days with adjacent activity
    zero_days = []
    for i, day in enumerate(daily):
        if day.transaction_count > 0:
            continue
        # Check adjacent days
        prev_active = i > 0 and daily[i - 1].transaction_count > 0
        next_active = i < len(daily) - 1 and daily[i + 1].transaction_count > 0
        if prev_active or next_active:
            zero_days.append(day)

    if not zero_days:
        return None

    # Alert on the most recent zero-revenue day
    gap_day = zero_days[-1]
    current_paise = int(analytics.avg_daily_revenue * 100)

    detail = (
        f"No transactions were recorded on {gap_day.date_label}. "
        f"This is unusual given activity on adjacent days. "
        f"Check your point-of-sale system, Google Sheets sync, "
        f"or contact your team to verify sales were not missed."
    )

    logger.debug(
        "Zero revenue day detected",
        extra={
            **log_extra,
            "gap_date": str(gap_day.date),
        },
    )

    return SalesSignal(
        signal_type="zero_revenue_day",
        alert_type=AlertType.SALES_TREND,
        change_pct=-1.0,
        current_revenue=0,
        period_label=gap_day.date_label,
        detail=detail,
        severity="high",
        trigger_date=gap_day.date,
    )


# ==============================================================================
# Helpers
# ==============================================================================

def _analytics_summary(analytics: SalesAnalyticsResult) -> dict:
    """Build a compact summary of the analytics data used for detection."""
    return {
        "total_revenue": str(analytics.total_revenue),
        "prev_period_revenue": str(analytics.prev_period_revenue),
        "revenue_change_pct": analytics.revenue_change_pct,
        "total_transactions": analytics.total_transactions,
        "days_in_window": len(analytics.daily_sales),
        "avg_daily_revenue": str(analytics.avg_daily_revenue),
        "period_label": analytics.period_label,
    }


# ==============================================================================
# Convenience: batch detection for scheduler use
# ==============================================================================

def detect_sales_anomalies_for_businesses(
    detector: SalesAlertsDetector,
    analytics_map: dict[str, SalesAnalyticsResult | AnalyticsError],
    business_names: dict[str, str],
    detection_date: Optional[date] = None,
) -> list[SalesAnomalyDetectionResult]:
    """
    Run sales anomaly detection for a batch of businesses.

    Fully synchronous — the scheduler awaits analytics fetches upstream
    and passes the pre-computed results here.

    Each business is processed independently. A failure for one business
    never affects the others (failure isolation contract).

    Args:
        detector:       Shared SalesAlertsDetector instance.
        analytics_map:  Dict mapping business_id → SalesAnalyticsResult
                        or AnalyticsError.
        business_names: Dict mapping business_id → business_name.
        detection_date: Override date for testing.

    Returns:
        list[SalesAnomalyDetectionResult]: One result per business.
    """
    detector_instance = detector
    results: list[SalesAnomalyDetectionResult] = []
    today = detection_date or date.today()

    for business_id, analytics in analytics_map.items():
        business_name = business_names.get(business_id, "Unknown Business")
        try:
            result = detector_instance.detect(
                analytics=analytics,
                business_id=business_id,
                business_name=business_name,
                detection_date=today,
            )
        except Exception as exc:
            logger.error(
                "Sales anomaly detection error for business — skipping",
                extra={
                    "service": ServiceName.ALERTS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )
            result = SalesAnomalyDetectionResult(
                business_id=business_id,
                detection_date=today,
                skipped=True,
                skip_reason=f"unhandled_error: {exc}",
            )
        results.append(result)

    alert_count = sum(1 for r in results if r.has_alerts)
    logger.info(
        "Batch sales anomaly detection complete",
        extra={
            "service": ServiceName.ALERTS,
            "businesses_checked": len(analytics_map),
            "alerts_detected": alert_count,
        },
    )

    return results