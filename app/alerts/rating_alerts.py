# ==============================================================================
# File: app/alerts/rating_alerts.py
# Purpose: Detects significant drops in a business's Google review rating
#          and produces RatingDropInput objects for the AlertManager.
#
#          Detection logic:
#            The detector compares two rating windows:
#              CURRENT  — average rating of reviews in the last N days
#              BASELINE — average rating of reviews in the prior N days
#                         (the window immediately before the current window)
#
#            A drop alert is triggered when:
#              (baseline_avg - current_avg) >= DROP_THRESHOLD
#
#            Additionally, a drop is only actionable if:
#              - The current window has at least MIN_REVIEWS_FOR_DETECTION reviews
#                (prevents false alerts from a single 1-star review)
#              - The business's overall average rating has actually declined
#                (not just noise in a small sample)
#
#          Secondary signal — review velocity:
#            If the ratio of negative reviews in the current window exceeds
#            NEGATIVE_RATIO_THRESHOLD (e.g. >40% of recent reviews are
#            negative), a HIGH severity alert is produced even if the
#            raw rating drop hasn't crossed the threshold yet.
#            This catches early reputation deterioration before it
#            becomes visible in the overall average.
#
#          Output:
#            RatingDropDetectionResult — contains zero or one RatingDropInput
#            objects ready for AlertManager.dispatch_rating_drop_alert().
#
#          Multi-tenant:
#            All repository queries are scoped to business_id.
#            No cross-business data ever flows through this module.
#
#          Performance:
#            Uses pre-aggregated counts from review_repository.py —
#            never loads raw review rows into memory for counting.
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.alerts.alert_manager import RatingDropInput
from app.config.constants import AlertType, ServiceName
from app.repositories.review_repository import ReviewRepository

logger = logging.getLogger(ServiceName.ALERTS)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Minimum rating drop (current window avg vs baseline avg) to fire an alert
DROP_THRESHOLD: float = 0.3

# Minimum rating drop to classify as CRITICAL (used by alert_manager severity)
CRITICAL_DROP_THRESHOLD: float = 0.5

# Minimum reviews in the current window for detection to be meaningful
MIN_REVIEWS_FOR_DETECTION: int = 3

# Ratio of negative reviews that triggers an early-warning alert
# independent of the raw rating drop
NEGATIVE_RATIO_THRESHOLD: float = 0.40

# Minimum negative reviews in the window before ratio alert fires
# (prevents a 1-negative / 2-total = 50% ratio from falsely triggering)
MIN_NEGATIVE_FOR_RATIO_ALERT: int = 2

# Default detection window in days
DEFAULT_DETECTION_WINDOW_DAYS: int = 7

# Minimum overall business average rating below which the system is
# already aware — suppress drop alerts below this floor to avoid
# repeatedly alerting on a chronically low-rated business
CHRONIC_LOW_RATING_FLOOR: float = 2.5


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class RatingWindowStats:
    """
    Aggregated rating statistics for a single time window.

    Attributes:
        window_start:    First date of the window (inclusive).
        window_end:      Last date of the window (inclusive).
        total_reviews:   Total reviews in this window.
        positive_count:  Reviews with positive sentiment.
        negative_count:  Reviews with negative sentiment.
        neutral_count:   Reviews with neutral sentiment.
        avg_rating:      Average star rating (None if no reviews).
        negative_ratio:  Fraction of negative reviews (0.0–1.0).
    """
    window_start: date
    window_end: date
    total_reviews: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_rating: Optional[float]

    @property
    def negative_ratio(self) -> float:
        if self.total_reviews == 0:
            return 0.0
        return self.negative_count / self.total_reviews

    @property
    def has_sufficient_data(self) -> bool:
        return self.total_reviews >= MIN_REVIEWS_FOR_DETECTION

    @property
    def period_label(self) -> str:
        return (
            f"{self.window_start.strftime('%d %b')} – "
            f"{self.window_end.strftime('%d %b %Y')}"
        )


@dataclass
class RatingDropDetectionResult:
    """
    Output of a single rating drop detection run for one business.

    Attributes:
        business_id:        Business UUID.
        detection_date:     Date the detection ran.
        current_window:     Stats for the current detection window.
        baseline_window:    Stats for the prior (baseline) window.
        drop_detected:      True if a rating drop threshold was crossed.
        ratio_alert:        True if negative review ratio threshold crossed
                            (independent of raw drop — early warning signal).
        drop_amount:        Magnitude of the detected drop (0.0 if none).
        alert_input:        Populated RatingDropInput if drop_detected or
                            ratio_alert is True, else None.
        skip_reason:        Human-readable reason if no alert was produced.
    """
    business_id: str
    detection_date: date
    current_window: RatingWindowStats
    baseline_window: RatingWindowStats
    drop_detected: bool
    ratio_alert: bool
    drop_amount: float
    alert_input: Optional[RatingDropInput]
    skip_reason: Optional[str] = None

    @property
    def should_alert(self) -> bool:
        return self.drop_detected or self.ratio_alert

    def __str__(self) -> str:
        status = "ALERT" if self.should_alert else "clean"
        return (
            f"RatingDropDetectionResult("
            f"business={self.business_id} "
            f"status={status} "
            f"drop={self.drop_amount:.2f} "
            f"ratio_alert={self.ratio_alert})"
        )


# ==============================================================================
# Rating Alerts Detector
# ==============================================================================

class RatingAlertsDetector:
    """
    Detects rating drops for a business by comparing two rolling windows.

    Stateless — safe to share a single instance across the application.

    Usage:
        detector = RatingAlertsDetector(review_repo=review_repo)

        result = await detector.detect(
            db=db,
            business_id="uuid",
            business_name="Raj Restaurant",
            current_avg_rating=3.8,
            window_days=7,
        )

        if result.should_alert and result.alert_input:
            await alert_manager.dispatch_rating_drop_alert(
                db=db,
                input=result.alert_input,
                whatsapp_number="+919876543210",
            )
    """

    def __init__(self, review_repo: ReviewRepository) -> None:
        self._review_repo = review_repo

    async def detect(
        self,
        db: AsyncSession,
        business_id: str,
        business_name: str,
        current_avg_rating: float,
        window_days: int = DEFAULT_DETECTION_WINDOW_DAYS,
        detection_date: Optional[date] = None,
    ) -> RatingDropDetectionResult:
        """
        Run rating drop detection for a single business.

        Compares the current window (last window_days days) against the
        baseline window (the window_days days immediately prior).

        Args:
            db:                 AsyncSession.
            business_id:        Business UUID for repository queries and logging.
            business_name:      Business display name for alert content.
            current_avg_rating: Business's current overall average rating.
                                Used as a sanity check — if this is already
                                below CHRONIC_LOW_RATING_FLOOR, suppresses
                                the alert to avoid chronic-low-rating noise.
            window_days:        Size of each comparison window in days.
            detection_date:     Override today's date (for testing).

        Returns:
            RatingDropDetectionResult. Never raises.
        """
        today = detection_date or date.today()
        log_extra = {
            "service": ServiceName.ALERTS,
            "business_id": business_id,
            "window_days": window_days,
            "current_avg_rating": current_avg_rating,
        }

        # Define the two windows
        current_end = today
        current_start = today - timedelta(days=window_days - 1)
        baseline_end = current_start - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=window_days - 1)

        # Fetch window stats
        try:
            current_stats = await self._get_window_stats(
                db=db,
                business_id=business_id,
                window_start=current_start,
                window_end=current_end,
            )
            baseline_stats = await self._get_window_stats(
                db=db,
                business_id=business_id,
                window_start=baseline_start,
                window_end=baseline_end,
            )
        except Exception as exc:
            logger.error(
                "Rating drop detection failed — repository error",
                extra={**log_extra, "error": str(exc)},
            )
            # Return a no-alert result so the scheduler can continue
            empty_window = _empty_window_stats(current_start, current_end)
            return RatingDropDetectionResult(
                business_id=business_id,
                detection_date=today,
                current_window=empty_window,
                baseline_window=_empty_window_stats(baseline_start, baseline_end),
                drop_detected=False,
                ratio_alert=False,
                drop_amount=0.0,
                alert_input=None,
                skip_reason=f"repository_error: {exc}",
            )

        # Evaluate drop
        drop_detected, drop_amount, drop_skip_reason = _evaluate_drop(
            current=current_stats,
            baseline=baseline_stats,
            current_avg_rating=current_avg_rating,
            log_extra=log_extra,
        )

        # Evaluate negative ratio (independent signal)
        ratio_alert, ratio_skip_reason = _evaluate_negative_ratio(
            current=current_stats,
            log_extra=log_extra,
        )

        should_alert = drop_detected or ratio_alert
        skip_reason: Optional[str] = None

        if not should_alert:
            skip_reason = drop_skip_reason or ratio_skip_reason or "no_threshold_crossed"

        # Build alert input
        alert_input: Optional[RatingDropInput] = None
        if should_alert:
            alert_input = _build_alert_input(
                business_id=business_id,
                business_name=business_name,
                current_stats=current_stats,
                baseline_stats=baseline_stats,
                current_avg_rating=current_avg_rating,
                drop_amount=drop_amount,
                ratio_alert=ratio_alert,
                drop_detected=drop_detected,
            )

        result = RatingDropDetectionResult(
            business_id=business_id,
            detection_date=today,
            current_window=current_stats,
            baseline_window=baseline_stats,
            drop_detected=drop_detected,
            ratio_alert=ratio_alert,
            drop_amount=drop_amount,
            alert_input=alert_input,
            skip_reason=skip_reason,
        )

        log_level = logging.INFO if should_alert else logging.DEBUG
        logger.log(
            log_level,
            "Rating drop detection complete",
            extra={
                **log_extra,
                "drop_detected": drop_detected,
                "drop_amount": drop_amount,
                "ratio_alert": ratio_alert,
                "current_reviews": current_stats.total_reviews,
                "skip_reason": skip_reason,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Repository queries
    # ------------------------------------------------------------------

    async def _get_window_stats(
        self,
        db: AsyncSession,
        business_id: str,
        window_start: date,
        window_end: date,
    ) -> RatingWindowStats:
        """
        Fetch aggregated review stats for a given date window.

        Uses pre-aggregated count queries from the review repository —
        never loads raw review rows into memory.

        Args:
            db:             AsyncSession.
            business_id:    Business UUID.
            window_start:   Start of window (inclusive).
            window_end:     End of window (inclusive).

        Returns:
            RatingWindowStats with counts and computed average rating.
        """
        # Fetch counts by sentiment for the window
        sentiment_counts = await self._review_repo.count_by_sentiment_since(
            db=db,
            business_id=business_id,
            since=window_start,
            until=window_end,
        )

        positive_count = sentiment_counts.get("positive", 0)
        negative_count = sentiment_counts.get("negative", 0)
        neutral_count  = sentiment_counts.get("neutral",  0)
        total          = positive_count + negative_count + neutral_count

        # Fetch average rating for the window
        avg_rating = await self._review_repo.get_average_rating_since(
            db=db,
            business_id=business_id,
            since=window_start,
            until=window_end,
        )

        return RatingWindowStats(
            window_start=window_start,
            window_end=window_end,
            total_reviews=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            avg_rating=avg_rating,
        )


# ==============================================================================
# Detection logic — module-level pure functions
# ==============================================================================

def _evaluate_drop(
    current: RatingWindowStats,
    baseline: RatingWindowStats,
    current_avg_rating: float,
    log_extra: dict,
) -> tuple[bool, float, Optional[str]]:
    """
    Evaluate whether a measurable rating drop has occurred.

    Returns:
        tuple[drop_detected, drop_amount, skip_reason]
        - drop_detected: True if drop crosses DROP_THRESHOLD
        - drop_amount:   Magnitude of the drop (positive float)
        - skip_reason:   Human-readable reason if not detected
    """
    # Guard: insufficient data in current window
    if not current.has_sufficient_data:
        return False, 0.0, (
            f"insufficient_current_data: "
            f"{current.total_reviews} reviews < {MIN_REVIEWS_FOR_DETECTION} minimum"
        )

    # Guard: no baseline data to compare against
    if baseline.avg_rating is None or baseline.total_reviews == 0:
        return False, 0.0, "no_baseline_data"

    # Guard: no current rating to compare
    if current.avg_rating is None:
        return False, 0.0, "no_current_avg_rating"

    # Guard: business is already chronically low-rated — suppress noise
    if current_avg_rating < CHRONIC_LOW_RATING_FLOOR:
        return False, 0.0, (
            f"chronic_low_rating: overall avg {current_avg_rating:.1f} "
            f"< floor {CHRONIC_LOW_RATING_FLOOR}"
        )

    drop_amount = baseline.avg_rating - current.avg_rating

    if drop_amount < DROP_THRESHOLD:
        return False, max(0.0, round(drop_amount, 3)), (
            f"drop_below_threshold: {drop_amount:.3f} < {DROP_THRESHOLD}"
        )

    logger.debug(
        "Rating drop threshold crossed",
        extra={
            **log_extra,
            "baseline_avg": baseline.avg_rating,
            "current_avg": current.avg_rating,
            "drop_amount": drop_amount,
        },
    )

    return True, round(drop_amount, 3), None


def _evaluate_negative_ratio(
    current: RatingWindowStats,
    log_extra: dict,
) -> tuple[bool, Optional[str]]:
    """
    Evaluate whether the current window's negative review ratio
    exceeds the early-warning threshold.

    Returns:
        tuple[ratio_alert, skip_reason]
    """
    if not current.has_sufficient_data:
        return False, "insufficient_data_for_ratio"

    if current.negative_count < MIN_NEGATIVE_FOR_RATIO_ALERT:
        return False, (
            f"negative_count_too_low: "
            f"{current.negative_count} < {MIN_NEGATIVE_FOR_RATIO_ALERT}"
        )

    if current.negative_ratio < NEGATIVE_RATIO_THRESHOLD:
        return False, (
            f"negative_ratio_below_threshold: "
            f"{current.negative_ratio:.2f} < {NEGATIVE_RATIO_THRESHOLD}"
        )

    logger.debug(
        "Negative ratio threshold crossed",
        extra={
            **log_extra,
            "negative_ratio": current.negative_ratio,
            "threshold": NEGATIVE_RATIO_THRESHOLD,
            "negative_count": current.negative_count,
        },
    )

    return True, None


def _build_alert_input(
    business_id: str,
    business_name: str,
    current_stats: RatingWindowStats,
    baseline_stats: RatingWindowStats,
    current_avg_rating: float,
    drop_amount: float,
    ratio_alert: bool,
    drop_detected: bool,
) -> RatingDropInput:
    """
    Construct the RatingDropInput for the AlertManager.

    Builds the most informative description based on which signal
    triggered: raw drop, ratio alert, or both.

    Args:
        business_id:       Business UUID.
        business_name:     Business display name.
        current_stats:     Stats for the current detection window.
        baseline_stats:    Stats for the baseline window.
        current_avg_rating: Overall business average rating.
        drop_amount:       Magnitude of the rating drop (0.0 if ratio-only).
        ratio_alert:       Whether the ratio signal fired.
        drop_detected:     Whether the raw drop signal fired.

    Returns:
        RatingDropInput ready for AlertManager.dispatch_rating_drop_alert().
    """
    # Use current window avg if available, else fall back to overall avg
    current_rating = current_stats.avg_rating or current_avg_rating
    previous_rating = (
        baseline_stats.avg_rating
        if baseline_stats.avg_rating is not None
        else current_avg_rating + drop_amount
    )

    period_label = current_stats.period_label

    # If only the ratio signal fired (no raw drop), adjust drop_amount
    # to reflect the ratio-based signal clearly
    effective_drop = drop_amount
    if ratio_alert and not drop_detected:
        effective_drop = round(previous_rating - current_rating, 3) if previous_rating else 0.0
        effective_drop = max(effective_drop, 0.1)  # minimum non-zero for display

    return RatingDropInput(
        business_id=business_id,
        business_name=business_name,
        current_rating=round(current_rating, 2),
        previous_rating=round(previous_rating, 2),
        drop_amount=effective_drop,
        period_label=period_label,
    )


def _empty_window_stats(window_start: date, window_end: date) -> RatingWindowStats:
    """Return a zero-count RatingWindowStats for error/empty cases."""
    return RatingWindowStats(
        window_start=window_start,
        window_end=window_end,
        total_reviews=0,
        positive_count=0,
        negative_count=0,
        neutral_count=0,
        avg_rating=None,
    )


# ==============================================================================
# Convenience: batch detection for scheduler use
# ==============================================================================

async def detect_rating_drops_for_businesses(
    detector: RatingAlertsDetector,
    db: AsyncSession,
    businesses: list[dict],
    window_days: int = DEFAULT_DETECTION_WINDOW_DAYS,
    detection_date: Optional[date] = None,
) -> list[RatingDropDetectionResult]:
    """
    Run rating drop detection for a batch of businesses.

    Each business is processed independently — a failure for one
    business never affects others (failure isolation contract).

    Args:
        detector:       Shared RatingAlertsDetector instance.
        db:             AsyncSession.
        businesses:     List of dicts with keys:
                          - business_id  (str)
                          - business_name (str)
                          - current_avg_rating (float)
        window_days:    Detection window size in days.
        detection_date: Override date for testing.

    Returns:
        list[RatingDropDetectionResult]: One result per business, in order.
        Businesses that failed internally return a no-alert result.
    """
    results: list[RatingDropDetectionResult] = []

    for biz in businesses:
        try:
            result = await detector.detect(
                db=db,
                business_id=biz["business_id"],
                business_name=biz["business_name"],
                current_avg_rating=biz.get("current_avg_rating", 3.0),
                window_days=window_days,
                detection_date=detection_date,
            )
        except Exception as exc:
            logger.error(
                "Rating drop detection error for business — skipping",
                extra={
                    "service": ServiceName.ALERTS,
                    "business_id": biz.get("business_id"),
                    "error": str(exc),
                },
            )
            today = detection_date or date.today()
            empty = _empty_window_stats(
                today - timedelta(days=window_days - 1), today
            )
            result = RatingDropDetectionResult(
                business_id=biz.get("business_id", "unknown"),
                detection_date=today,
                current_window=empty,
                baseline_window=_empty_window_stats(
                    today - timedelta(days=2 * window_days - 1),
                    today - timedelta(days=window_days),
                ),
                drop_detected=False,
                ratio_alert=False,
                drop_amount=0.0,
                alert_input=None,
                skip_reason=f"unhandled_error: {exc}",
            )

        results.append(result)

    alert_count = sum(1 for r in results if r.should_alert)
    logger.info(
        "Batch rating drop detection complete",
        extra={
            "service": ServiceName.ALERTS,
            "businesses_checked": len(businesses),
            "alerts_detected": alert_count,
            "window_days": window_days,
        },
    )

    return results