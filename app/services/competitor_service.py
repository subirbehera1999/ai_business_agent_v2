# ==============================================================================
# File: app/services/competitor_service.py
# Purpose: Fetches competitor business data from Google Places API,
#          computes rating deltas, detects review spikes, and produces
#          structured competitive intelligence consumed by:
#            - reports_service.py    (weekly/monthly competitor snapshot)
#            - alerts/competitor_alerts.py (spike and rating change alerts)
#            - seo_service.py        (competitor rating context)
#
#          Competitor data flow:
#            1. Business registers up to N competitor Google Place IDs
#               during onboarding (stored in business_model competitors field)
#            2. Scheduler (competitor_scan_job) calls this service daily
#            3. Service fetches current rating + review count from Places API
#            4. Computes deltas vs last stored snapshot
#            5. Returns CompetitorScanResult — never raises
#
#          Rate limiting:
#            - MAX_COMPETITOR_SCANS_PER_DAY enforced before each scan
#            - Google Places API calls use retry + backoff (retry_utils.py)
#            - Batch processing prevents simultaneous API floods
#
#          Data safety:
#            - Every scan is scoped to business_id (multi-tenant isolation)
#            - No competitor data crosses business boundaries
#            - Idempotency: same place_id + date = skip if already scanned today
#
#          External API used:
#            - Google Places API (Details endpoint)
#            - Handled via app/integrations/google_reviews_client.py
#            - This service does NOT call HTTP directly — delegates to client
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.repositories.subscription_repository import SubscriptionRepository
from app.repositories.usage_repository import UsageRepository
from app.utils.rate_limiter import enforce_rate_limit
from app.utils.usage_tracker import track_competitor_scan

logger = logging.getLogger(ServiceName.COMPETITOR)
settings = get_settings()

# Maximum competitors tracked per business (enforced at scan time)
MAX_COMPETITORS_PER_BUSINESS = 5

# Minimum rating delta to classify as a "significant change" worth alerting
RATING_CHANGE_THRESHOLD = 0.2

# Review count increase percentage that qualifies as a "spike"
REVIEW_SPIKE_THRESHOLD_PCT = 0.15   # 15% increase in review count

# Minimum absolute review increase to trigger a spike alert
REVIEW_SPIKE_MIN_NEW_REVIEWS = 3


# ==============================================================================
# Data structures
# ==============================================================================

@dataclass
class CompetitorSnapshot:
    """
    A point-in-time snapshot of a single competitor's Google Business data.

    Attributes:
        place_id:        Google Place ID for the competitor.
        name:            Competitor business name from Places API.
        avg_rating:      Current average star rating (1.0–5.0).
        review_count:    Total review count at time of snapshot.
        scan_date:       Date this snapshot was taken.
        address:         Formatted address (optional — for display).
        is_open:         Whether the business is currently open (optional).
    """
    place_id: str
    name: str
    avg_rating: float
    review_count: int
    scan_date: date
    address: Optional[str] = None
    is_open: Optional[bool] = None

    @property
    def rating_display(self) -> str:
        return f"{self.avg_rating:.1f} / 5.0"


@dataclass
class CompetitorDelta:
    """
    Comparison between a competitor's current snapshot and their previous one.

    Attributes:
        place_id:             Google Place ID.
        name:                 Competitor name.
        current_rating:       Rating from today's scan.
        previous_rating:      Rating from last scan (None if first scan).
        rating_change:        Difference (current - previous), None if first.
        current_reviews:      Review count from today's scan.
        previous_reviews:     Review count from last scan (None if first).
        new_reviews:          Increase in review count since last scan.
        is_rating_spike:      True if rating changed by >= RATING_CHANGE_THRESHOLD.
        is_review_spike:      True if new reviews qualify as a spike.
        rating_direction:     "up", "down", "stable", or "new" (first scan).
        scan_date:            Date of current scan.
    """
    place_id: str
    name: str
    current_rating: float
    previous_rating: Optional[float]
    rating_change: Optional[float]
    current_reviews: int
    previous_reviews: Optional[int]
    new_reviews: Optional[int]
    is_rating_spike: bool
    is_review_spike: bool
    rating_direction: str
    scan_date: date

    @property
    def is_first_scan(self) -> bool:
        return self.previous_rating is None

    @property
    def rating_change_display(self) -> str:
        if self.rating_change is None:
            return "First scan"
        sign = "+" if self.rating_change > 0 else ""
        return f"{sign}{self.rating_change:.1f}"

    @property
    def requires_alert(self) -> bool:
        """True if this delta should trigger a competitor alert."""
        return self.is_rating_spike or self.is_review_spike


@dataclass
class CompetitorScanResult:
    """
    Complete output of a competitor scan run for one business.

    Attributes:
        business_id:       The business whose competitors were scanned.
        scan_date:         Date the scan ran.
        snapshots:         Fresh snapshots from today's Places API calls.
        deltas:            Computed deltas vs previous snapshots.
        competitors_scanned: Number of competitors successfully scanned.
        competitors_failed:  Number of competitors that failed to fetch.
        rate_limited:       True if scan was blocked by daily limit.
        alerts_needed:      Deltas that require alert generation.
        our_avg_rating:     Business's own current average rating (for comparison).
        best_competitor:    Competitor with highest current rating.
        worst_competitor:   Competitor with lowest current rating.
        scan_errors:        List of (place_id, error_message) for failures.
    """
    business_id: str
    scan_date: date
    snapshots: list[CompetitorSnapshot]
    deltas: list[CompetitorDelta]
    competitors_scanned: int
    competitors_failed: int
    rate_limited: bool
    alerts_needed: list[CompetitorDelta]
    our_avg_rating: Optional[float]
    best_competitor: Optional[CompetitorSnapshot]
    worst_competitor: Optional[CompetitorSnapshot]
    scan_errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts_needed) > 0

    @property
    def total_competitors(self) -> int:
        return self.competitors_scanned + self.competitors_failed

    @property
    def we_are_leading(self) -> bool:
        """True if our rating is higher than all tracked competitors."""
        if not self.our_avg_rating or not self.snapshots:
            return False
        return all(self.our_avg_rating > s.avg_rating for s in self.snapshots)

    @property
    def rating_vs_best(self) -> Optional[float]:
        """Our rating minus best competitor's rating (negative = behind)."""
        if not self.our_avg_rating or not self.best_competitor:
            return None
        return round(self.our_avg_rating - self.best_competitor.avg_rating, 2)

    def __str__(self) -> str:
        return (
            f"CompetitorScanResult("
            f"business={self.business_id} "
            f"scanned={self.competitors_scanned} "
            f"failed={self.competitors_failed} "
            f"alerts={len(self.alerts_needed)} "
            f"rate_limited={self.rate_limited})"
        )


@dataclass(frozen=True)
class CompetitorScanError:
    """Returned when the entire scan could not run."""
    business_id: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return (
            f"CompetitorScanError("
            f"business={self.business_id} "
            f"reason={self.reason}: {self.detail})"
        )


# ==============================================================================
# Competitor Service
# ==============================================================================

class CompetitorService:
    """
    Fetches and analyses competitor data for a business.

    This service is the orchestration layer. It does not call the
    Google Places API directly — that is the responsibility of
    google_reviews_client.py (injected as `places_client`).

    Usage:
        service = CompetitorService(
            places_client=google_reviews_client,
            usage_repo=usage_repo,
            subscription_repo=subscription_repo,
        )

        result = await service.scan_competitors(
            db=db,
            business_id="uuid",
            competitor_place_ids=["ChIJ...", "ChIJ..."],
            our_avg_rating=4.2,
            previous_snapshots=last_scan_data,
        )

        if isinstance(result, CompetitorScanError):
            logger.warning(str(result))
        else:
            for delta in result.alerts_needed:
                await alert_manager.trigger_competitor_alert(delta)
    """

    def __init__(
        self,
        places_client,                        # google_reviews_client.GoogleReviewsClient
        usage_repo: UsageRepository,
        subscription_repo: SubscriptionRepository,
    ) -> None:
        self._places_client = places_client
        self._usage_repo = usage_repo
        self._subscription_repo = subscription_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_competitors(
        self,
        db,
        business_id: str,
        competitor_place_ids: list[str],
        our_avg_rating: Optional[float],
        previous_snapshots: Optional[list[CompetitorSnapshot]] = None,
        scan_date: Optional[date] = None,
    ) -> CompetitorScanResult | CompetitorScanError:
        """
        Scan all competitor Place IDs and compute rating/review deltas.

        Args:
            db:                     AsyncSession for DB operations.
            business_id:            Business UUID (multi-tenant isolation).
            competitor_place_ids:   List of Google Place IDs to scan.
                                    Capped at MAX_COMPETITORS_PER_BUSINESS.
            our_avg_rating:         Our current average rating (for comparison).
            previous_snapshots:     Prior scan snapshots for delta computation.
                                    None on first scan — all deltas will be "new".
            scan_date:              Override today's date (for testing).

        Returns:
            CompetitorScanResult on success, CompetitorScanError on failure.
            Never raises.
        """
        today = scan_date or date.today()
        log_extra = {
            "service": ServiceName.COMPETITOR,
            "business_id": business_id,
            "competitor_count": len(competitor_place_ids),
            "scan_date": str(today),
        }

        # ----------------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------------
        if not competitor_place_ids:
            logger.info(
                "Competitor scan skipped — no competitor place IDs configured",
                extra=log_extra,
            )
            return CompetitorScanError(
                business_id=business_id,
                reason="no_competitors_configured",
                detail="No competitor Place IDs registered for this business.",
            )

        # Cap to maximum allowed competitors
        place_ids_to_scan = competitor_place_ids[:MAX_COMPETITORS_PER_BUSINESS]
        if len(competitor_place_ids) > MAX_COMPETITORS_PER_BUSINESS:
            logger.warning(
                "Competitor scan capped to maximum",
                extra={
                    **log_extra,
                    "requested": len(competitor_place_ids),
                    "capped_to": MAX_COMPETITORS_PER_BUSINESS,
                },
            )

        # ----------------------------------------------------------
        # Rate limit check (per competitor scan, not per competitor)
        # ----------------------------------------------------------
        rate_result = await enforce_rate_limit(
            db=db,
            business_id=business_id,
            metric="competitor_scans",
            usage_repo=self._usage_repo,
            subscription_repo=self._subscription_repo,
        )

        if not rate_result.allowed:
            logger.warning(
                "Competitor scan blocked — daily limit reached",
                extra={
                    **log_extra,
                    "limit": rate_result.limit,
                    "current": rate_result.current_count,
                },
            )
            return CompetitorScanResult(
                business_id=business_id,
                scan_date=today,
                snapshots=[],
                deltas=[],
                competitors_scanned=0,
                competitors_failed=0,
                rate_limited=True,
                alerts_needed=[],
                our_avg_rating=our_avg_rating,
                best_competitor=None,
                worst_competitor=None,
            )

        # ----------------------------------------------------------
        # Build previous snapshot lookup map
        # ----------------------------------------------------------
        prev_map: dict[str, CompetitorSnapshot] = {}
        if previous_snapshots:
            for snap in previous_snapshots:
                prev_map[snap.place_id] = snap

        # ----------------------------------------------------------
        # Fetch current data for each competitor
        # ----------------------------------------------------------
        snapshots: list[CompetitorSnapshot] = []
        scan_errors: list[tuple[str, str]] = []

        for place_id in place_ids_to_scan:
            try:
                snapshot = await self._fetch_competitor_snapshot(
                    place_id=place_id,
                    scan_date=today,
                    log_extra=log_extra,
                )
                snapshots.append(snapshot)
                logger.debug(
                    "Competitor fetched",
                    extra={
                        **log_extra,
                        "place_id": place_id,
                        "name": snapshot.name,
                        "rating": snapshot.avg_rating,
                        "reviews": snapshot.review_count,
                    },
                )
            except Exception as exc:
                scan_errors.append((place_id, str(exc)))
                logger.error(
                    "Failed to fetch competitor data",
                    extra={
                        **log_extra,
                        "place_id": place_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

        # ----------------------------------------------------------
        # Compute deltas
        # ----------------------------------------------------------
        deltas = [
            _compute_delta(snapshot, prev_map.get(snapshot.place_id), today)
            for snapshot in snapshots
        ]

        alerts_needed = [d for d in deltas if d.requires_alert]

        # ----------------------------------------------------------
        # Compute comparison stats
        # ----------------------------------------------------------
        best_competitor = (
            max(snapshots, key=lambda s: s.avg_rating) if snapshots else None
        )
        worst_competitor = (
            min(snapshots, key=lambda s: s.avg_rating) if snapshots else None
        )

        # ----------------------------------------------------------
        # Track usage — one scan unit per business per run
        # ----------------------------------------------------------
        await track_competitor_scan(db, business_id, self._usage_repo)

        result = CompetitorScanResult(
            business_id=business_id,
            scan_date=today,
            snapshots=snapshots,
            deltas=deltas,
            competitors_scanned=len(snapshots),
            competitors_failed=len(scan_errors),
            rate_limited=False,
            alerts_needed=alerts_needed,
            our_avg_rating=our_avg_rating,
            best_competitor=best_competitor,
            worst_competitor=worst_competitor,
            scan_errors=scan_errors,
        )

        logger.info(
            "Competitor scan completed",
            extra={
                **log_extra,
                "scanned": result.competitors_scanned,
                "failed": result.competitors_failed,
                "alerts_needed": len(alerts_needed),
                "we_are_leading": result.we_are_leading,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Internal: fetch single competitor
    # ------------------------------------------------------------------

    async def _fetch_competitor_snapshot(
        self,
        place_id: str,
        scan_date: date,
        log_extra: dict,
    ) -> CompetitorSnapshot:
        """
        Fetch current rating and review count for a competitor from Google Places.

        Delegates to google_reviews_client which handles:
          - Authentication
          - Retry with exponential backoff
          - Response parsing

        Args:
            place_id:   Google Place ID.
            scan_date:  Date to stamp on the snapshot.
            log_extra:  Structured log context.

        Returns:
            CompetitorSnapshot with current data.

        Raises:
            Exception: Any network or API error (caller handles per-competitor).
        """
        api_result = await self._places_client.get_place_details(
        place_id=place_id,
        fields=["name", "rating", "user_ratings_total", "formatted_address", "opening_hours"],
        )

        if not api_result.success:
            raise RuntimeError(
                f"Places API error for place_id={place_id}: {api_result.error}"
            )

        place_data = api_result.data  # This is the actual dict

        return CompetitorSnapshot(
            place_id=place_id,
            name=place_data.get("name", "Unknown Competitor"),
            avg_rating=float(place_data.get("rating", 0.0)),
            review_count=int(place_data.get("user_ratings_total", 0)),
            scan_date=scan_date,
            address=place_data.get("formatted_address"),
            is_open=_extract_is_open(place_data),
        )

    # ------------------------------------------------------------------
    # Comparison utilities (used by seo_service and reports_service)
    # ------------------------------------------------------------------

    def compare_with_us(
        self,
        our_rating: float,
        snapshots: list[CompetitorSnapshot],
    ) -> dict:
        """
        Produce a structured comparison of our rating vs all competitors.

        Used by seo_service.py and reports_service.py to build the
        competitor snapshot section of a report.

        Args:
            our_rating:  Our current average rating.
            snapshots:   Current competitor snapshots.

        Returns:
            dict with keys:
              - our_rating      (float)
              - avg_competitor_rating (float)
              - we_are_leading  (bool)
              - rating_gap      (float — positive = we lead)
              - competitors     (list[dict] sorted by rating desc)
        """
        if not snapshots:
            return {
                "our_rating": our_rating,
                "avg_competitor_rating": None,
                "we_are_leading": None,
                "rating_gap": None,
                "competitors": [],
            }

        ratings = [s.avg_rating for s in snapshots]
        avg_comp = sum(ratings) / len(ratings)
        gap = round(our_rating - avg_comp, 2)

        competitors_sorted = sorted(snapshots, key=lambda s: s.avg_rating, reverse=True)

        return {
            "our_rating": our_rating,
            "avg_competitor_rating": round(avg_comp, 2),
            "we_are_leading": our_rating > avg_comp,
            "rating_gap": gap,
            "competitors": [
                {
                    "name": s.name,
                    "rating": s.avg_rating,
                    "review_count": s.review_count,
                    "address": s.address,
                }
                for s in competitors_sorted
            ],
        }

    def get_strategic_position(
        self,
        our_rating: float,
        snapshots: list[CompetitorSnapshot],
    ) -> str:
        """
        Return a one-line strategic position summary for WhatsApp reports.

        Args:
            our_rating:  Our current average rating.
            snapshots:   Current competitor snapshots.

        Returns:
            str: Human-readable position summary.

        Example:
            "You are the highest-rated business among 3 competitors. 🏆"
            "2 of 3 competitors have a higher rating. Focus on reviews."
        """
        if not snapshots:
            return "No competitor data available for comparison."

        ahead = sum(1 for s in snapshots if our_rating > s.avg_rating)
        behind = sum(1 for s in snapshots if our_rating < s.avg_rating)
        equal = sum(1 for s in snapshots if our_rating == s.avg_rating)
        total = len(snapshots)

        if behind == 0 and (ahead > 0 or equal > 0):
            return (
                f"You are the highest-rated business among "
                f"{total} competitor{'s' if total > 1 else ''}. 🏆"
            )
        if ahead == 0:
            return (
                f"All {total} tracked competitor{'s are' if total > 1 else ' is'} "
                f"rated higher. Focus on increasing review volume and quality."
            )
        return (
            f"You are ahead of {ahead} out of {total} competitors. "
            f"{behind} competitor{'s have' if behind > 1 else ' has'} a higher rating."
        )


# ==============================================================================
# Module-level computation helpers
# ==============================================================================

def _compute_delta(
    current: CompetitorSnapshot,
    previous: Optional[CompetitorSnapshot],
    scan_date: date,
) -> CompetitorDelta:
    """
    Compute the delta between a current and previous competitor snapshot.

    Handles first-scan case (previous=None) by marking direction as "new".

    Args:
        current:    Today's snapshot.
        previous:   Last stored snapshot (None if first scan).
        scan_date:  Date of the current scan.

    Returns:
        CompetitorDelta with all comparison fields populated.
    """
    if previous is None:
        return CompetitorDelta(
            place_id=current.place_id,
            name=current.name,
            current_rating=current.avg_rating,
            previous_rating=None,
            rating_change=None,
            current_reviews=current.review_count,
            previous_reviews=None,
            new_reviews=None,
            is_rating_spike=False,
            is_review_spike=False,
            rating_direction="new",
            scan_date=scan_date,
        )

    rating_change = round(current.avg_rating - previous.avg_rating, 2)
    new_reviews = current.review_count - previous.review_count

    # Rating direction
    if abs(rating_change) < 0.05:
        direction = "stable"
    elif rating_change > 0:
        direction = "up"
    else:
        direction = "down"

    # Rating spike detection
    is_rating_spike = abs(rating_change) >= RATING_CHANGE_THRESHOLD

    # Review spike detection
    is_review_spike = _is_review_spike(
        previous_count=previous.review_count,
        new_reviews=new_reviews,
    )

    return CompetitorDelta(
        place_id=current.place_id,
        name=current.name,
        current_rating=current.avg_rating,
        previous_rating=previous.avg_rating,
        rating_change=rating_change,
        current_reviews=current.review_count,
        previous_reviews=previous.review_count,
        new_reviews=new_reviews,
        is_rating_spike=is_rating_spike,
        is_review_spike=is_review_spike,
        rating_direction=direction,
        scan_date=scan_date,
    )


def _is_review_spike(previous_count: int, new_reviews: int) -> bool:
    """
    Determine if the increase in review count qualifies as a spike.

    Two conditions must both be met:
      1. Absolute increase >= REVIEW_SPIKE_MIN_NEW_REVIEWS
      2. Percentage increase >= REVIEW_SPIKE_THRESHOLD_PCT

    This prevents a business with 3 reviews getting a 2-review increase
    (67% — technically large) from triggering a spike alert, while also
    preventing a large business gaining 100 reviews from being missed
    if the percentage is just under the threshold.

    Args:
        previous_count:  Review count at last scan.
        new_reviews:     Increase since last scan.

    Returns:
        bool: True if both spike conditions are met.
    """
    if new_reviews < REVIEW_SPIKE_MIN_NEW_REVIEWS:
        return False
    if previous_count <= 0:
        return False
    pct_increase = new_reviews / previous_count
    return pct_increase >= REVIEW_SPIKE_THRESHOLD_PCT


def _extract_is_open(place_data: dict) -> Optional[bool]:
    """
    Safely extract the current open/closed status from a Places API response.

    The `opening_hours` field is often absent for businesses that haven't
    set their hours. Returns None rather than False in that case.

    Args:
        place_data: Raw Places API response dict.

    Returns:
        bool | None: True if open, False if closed, None if unknown.
    """
    opening_hours = place_data.get("opening_hours")
    if not opening_hours:
        return None
    return opening_hours.get("open_now")


def build_competitor_summary_lines(
    scan_result: CompetitorScanResult,
) -> list[str]:
    """
    Build WhatsApp-ready summary lines from a CompetitorScanResult.

    Used by reports_service.py to build the competitor section of
    weekly and monthly reports without coupling the report generator
    to CompetitorScanResult internals.

    Args:
        scan_result: The result of a competitor scan run.

    Returns:
        list[str]: Lines ready for whatsapp_section rendering.
    """
    lines: list[str] = []

    if scan_result.rate_limited:
        lines.append("ℹ️  Competitor scan was skipped (daily limit reached).")
        return lines

    if not scan_result.snapshots:
        lines.append("ℹ️  No competitor data available.")
        return lines

    for snap in scan_result.snapshots:
        delta = next(
            (d for d in scan_result.deltas if d.place_id == snap.place_id), None
        )
        change_str = ""
        if delta and delta.rating_change is not None:
            sign = "+" if delta.rating_change > 0 else ""
            change_str = f" ({sign}{delta.rating_change:.1f} this period)"

        lines.append(f"• {snap.name}: ⭐ {snap.avg_rating:.1f}{change_str}")

    if scan_result.our_avg_rating is not None:
        service = CompetitorService.__new__(CompetitorService)
        position = service.get_strategic_position(
            scan_result.our_avg_rating, scan_result.snapshots
        )
        lines.append("")
        lines.append(position)

    return lines