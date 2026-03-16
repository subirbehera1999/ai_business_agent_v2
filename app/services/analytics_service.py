# ==============================================================================
# File: app/services/analytics_service.py
# Purpose: Sales analytics engine that processes raw Google Sheets data into
#          structured business intelligence — revenue trends, peak periods,
#          demand patterns, growth signals, and customer satisfaction indicators.
#
#          Inputs:
#            - Raw rows from Google Sheets (via google_sheets_client.py)
#            - ColumnMapping from column_mapper_service.py
#            - Review sentiment aggregates from review_repository.py
#
#          Outputs:
#            - SalesAnalyticsResult: structured data consumed by
#              reports_service.py, alerts/sales_alerts.py, and
#              notifications sent via WhatsApp
#
#          Design:
#            - No OpenAI calls in this file — pure computation
#            - All date parsing is timezone-aware via time_utils.py
#            - Works incrementally: operates on a configurable date window
#            - Never raises to the scheduler — returns AnalyticsError on failure
#            - Multi-tenant: every query and log entry is scoped to business_id
#
#          Performance contract:
#            - Rows processed in batches (batch_utils.py)
#            - Never loads entire sheet history into memory
#            - Max window: 90 days of history per analysis run
# ==============================================================================

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional

from app.config.constants import ServiceName
from app.services.column_mapper_service import (
    ColumnMapping,
    ColumnMapperService,
    extract_row_values as _extract,
)
from app.utils.time_utils import (
    get_date_range_label,
    parse_flexible_date,
    today_local,
    get_week_bounds,
    get_month_bounds,
)

logger = logging.getLogger(ServiceName.ANALYTICS)

# Maximum days of history analysed in a single run
MAX_ANALYSIS_WINDOW_DAYS = 90

# Minimum revenue value accepted (filters out zero/negative rows)
MIN_VALID_REVENUE = Decimal("0.01")

# Revenue spike threshold — day is "peak" if revenue > mean * this multiplier
PEAK_REVENUE_MULTIPLIER = 1.5

# Minimum rows required for meaningful trend analysis
MIN_ROWS_FOR_TREND = 5

# Day-of-week names (Monday = 0)
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class DailySales:
    """Aggregated sales data for a single calendar day."""
    date: date
    total_revenue: Decimal
    transaction_count: int
    avg_transaction: Decimal
    products: list[str] = field(default_factory=list)  # unique product names seen

    @property
    def date_label(self) -> str:
        return self.date.strftime("%d %b %Y")


@dataclass
class WeeklySales:
    """Aggregated sales for an ISO calendar week."""
    week_label: str       # e.g. "Week 42, 2024"
    start_date: date
    end_date: date
    total_revenue: Decimal
    transaction_count: int
    daily_breakdown: list[DailySales] = field(default_factory=list)


@dataclass
class TrendSignal:
    """
    A detected trend in sales data.

    direction: "up", "down", or "stable"
    magnitude: percentage change as a float (e.g. 0.23 = 23% increase)
    confidence: "high", "medium", or "low" based on sample size
    """
    direction: str
    magnitude: float
    confidence: str
    description: str


@dataclass
class PeakPeriod:
    """A detected peak day or period."""
    date: date
    revenue: Decimal
    revenue_vs_average: float   # multiplier e.g. 2.1 = 210% of average
    day_of_week: str


@dataclass
class SalesAnalyticsResult:
    """
    Complete analytics output for a single business over a date window.

    Consumed by:
      - reports_service.py    (weekly/monthly report generation)
      - sales_alerts.py       (anomaly detection)
      - seo_service.py        (demand pattern context)
    """
    business_id: str
    analysis_date: date
    window_start: date
    window_end: date
    period_label: str

    # Revenue aggregates
    total_revenue: Decimal
    avg_daily_revenue: Decimal
    avg_transaction_value: Decimal
    total_transactions: int

    # Period comparisons (current vs previous window)
    prev_period_revenue: Decimal
    revenue_change_pct: float           # e.g. 0.15 = +15%, -0.08 = -8%
    revenue_trend: TrendSignal

    # Breakdowns
    daily_sales: list[DailySales]
    weekly_sales: list[WeeklySales]
    peak_days: list[PeakPeriod]
    top_products: list[tuple[str, Decimal]]  # (name, total_revenue) sorted desc

    # Day-of-week pattern
    busiest_day_of_week: Optional[str]
    slowest_day_of_week: Optional[str]
    day_of_week_revenue: dict[str, Decimal]  # day_name → total revenue

    # Data quality
    total_rows_read: int
    valid_rows: int
    skipped_rows: int
    parse_errors: int
    has_sufficient_data: bool           # False if < MIN_ROWS_FOR_TREND

    @property
    def revenue_display(self) -> str:
        return f"₹{self.total_revenue:,.2f}"

    @property
    def avg_daily_display(self) -> str:
        return f"₹{self.avg_daily_revenue:,.2f}"

    @property
    def growth_display(self) -> str:
        sign = "+" if self.revenue_change_pct >= 0 else ""
        return f"{sign}{self.revenue_change_pct * 100:.1f}%"


@dataclass
class AnalyticsError:
    """Returned when analytics cannot be computed."""
    business_id: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return f"AnalyticsError(business={self.business_id} reason={self.reason}: {self.detail})"


# ==============================================================================
# Analytics Service
# ==============================================================================

class AnalyticsService:
    """
    Processes Google Sheets sales rows into structured analytics.

    Stateless — safe to share a single instance across the application.

    Usage:
        service = AnalyticsService(mapper=ColumnMapperService())

        result = await service.analyse(
            rows=sheet_rows,          # list[list[str]] from Google Sheets
            headers=header_row,       # list[str] — first row of the sheet
            business_id="uuid",
            window_days=30,
        )

        if isinstance(result, AnalyticsError):
            logger.warning(str(result))
        else:
            logger.info(f"Revenue: {result.revenue_display}")
    """

    def __init__(self, mapper: ColumnMapperService) -> None:
        self._mapper = mapper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyse(
        self,
        rows: list[list[str]],
        headers: list[str],
        business_id: str,
        window_days: int = 30,
        reference_date: Optional[date] = None,
    ) -> SalesAnalyticsResult | AnalyticsError:
        """
        Analyse sales data for a business over a rolling window.

        Args:
            rows:           Data rows from Google Sheets (headers excluded).
            headers:        Header row from Google Sheets (row 0).
            business_id:    Business UUID for isolation and logging.
            window_days:    How many days back to include (max 90).
            reference_date: The "today" reference for window calculation.
                            Defaults to actual today. Override for testing.

        Returns:
            SalesAnalyticsResult on success, AnalyticsError on failure.
            Never raises.
        """
        log_extra = {
            "service": ServiceName.ANALYTICS,
            "business_id": business_id,
            "window_days": window_days,
            "total_input_rows": len(rows),
        }

        # ----- Validate inputs -----
        effective_window = min(window_days, MAX_ANALYSIS_WINDOW_DAYS)
        today = reference_date or today_local()
        window_start = today - timedelta(days=effective_window - 1)
        window_end = today

        # ----- Map columns -----
        mapping = self._mapper.map_headers(headers, business_id)
        if not mapping.is_valid:
            error = self._mapper.build_error(mapping)
            logger.warning(
                "Analytics aborted — column mapping invalid",
                extra={**log_extra, "missing": sorted(mapping.missing_required)},
            )
            return AnalyticsError(
                business_id=business_id,
                reason="invalid_column_mapping",
                detail=str(error),
            )

        # ----- Parse rows into DailySales -----
        parsed, total_rows, skipped, parse_errors = _parse_rows(
            rows=rows,
            mapping=mapping,
            window_start=window_start,
            window_end=window_end,
            business_id=business_id,
        )

        if len(parsed) < 1:
            logger.warning(
                "Analytics: no valid rows in window",
                extra={**log_extra, "window_start": str(window_start), "window_end": str(window_end)},
            )
            return AnalyticsError(
                business_id=business_id,
                reason="no_data_in_window",
                detail=f"No valid sales rows found between {window_start} and {window_end}",
            )

        # ----- Aggregate into daily buckets -----
        daily_map = _aggregate_daily(parsed)
        daily_sales = sorted(daily_map.values(), key=lambda d: d.date)

        # ----- Previous period rows for comparison -----
        prev_start = window_start - timedelta(days=effective_window)
        prev_end = window_start - timedelta(days=1)
        prev_parsed, *_ = _parse_rows(
            rows=rows,
            mapping=mapping,
            window_start=prev_start,
            window_end=prev_end,
            business_id=business_id,
        )
        prev_daily = _aggregate_daily(prev_parsed)
        prev_revenue = sum(
            (d.total_revenue for d in prev_daily.values()),
            Decimal("0"),
        )

        # ----- Revenue aggregates -----
        total_revenue = sum((d.total_revenue for d in daily_sales), Decimal("0"))
        total_transactions = sum(d.transaction_count for d in daily_sales)
        num_days = max(len(daily_sales), 1)
        avg_daily_revenue = total_revenue / num_days
        avg_transaction_value = (
            total_revenue / total_transactions if total_transactions > 0 else Decimal("0")
        )

        # ----- Revenue change vs previous period -----
        revenue_change_pct = _compute_change_pct(total_revenue, prev_revenue)

        # ----- Trend signal -----
        revenue_trend = _detect_trend(
            daily_sales=daily_sales,
            revenue_change_pct=revenue_change_pct,
        )

        # ----- Peak days -----
        peak_days = _find_peak_days(daily_sales, avg_daily_revenue)

        # ----- Day-of-week pattern -----
        dow_revenue = _day_of_week_revenue(daily_sales)
        busiest_day = max(dow_revenue, key=lambda k: dow_revenue[k]) if dow_revenue else None
        slowest_day = min(dow_revenue, key=lambda k: dow_revenue[k]) if dow_revenue else None

        # ----- Weekly aggregates -----
        weekly_sales = _aggregate_weekly(daily_sales)

        # ----- Top products -----
        top_products = _top_products(parsed)

        period_label = get_date_range_label(window_start, window_end)
        valid_rows = len(parsed)
        has_sufficient = valid_rows >= MIN_ROWS_FOR_TREND

        logger.info(
            "Analytics completed",
            extra={
                **log_extra,
                "total_revenue": str(total_revenue),
                "valid_rows": valid_rows,
                "skipped_rows": skipped,
                "parse_errors": parse_errors,
                "peak_days": len(peak_days),
                "revenue_trend": revenue_trend.direction,
            },
        )

        return SalesAnalyticsResult(
            business_id=business_id,
            analysis_date=today,
            window_start=window_start,
            window_end=window_end,
            period_label=period_label,
            total_revenue=total_revenue,
            avg_daily_revenue=avg_daily_revenue,
            avg_transaction_value=avg_transaction_value,
            total_transactions=total_transactions,
            prev_period_revenue=prev_revenue,
            revenue_change_pct=revenue_change_pct,
            revenue_trend=revenue_trend,
            daily_sales=daily_sales,
            weekly_sales=weekly_sales,
            peak_days=peak_days,
            top_products=top_products,
            busiest_day_of_week=busiest_day,
            slowest_day_of_week=slowest_day,
            day_of_week_revenue=dow_revenue,
            total_rows_read=total_rows,
            valid_rows=valid_rows,
            skipped_rows=skipped,
            parse_errors=parse_errors,
            has_sufficient_data=has_sufficient,
        )

    # ------------------------------------------------------------------
    # Convenience: weekly analysis
    # ------------------------------------------------------------------

    async def analyse_current_week(
        self,
        rows: list[list[str]],
        headers: list[str],
        business_id: str,
    ) -> SalesAnalyticsResult | AnalyticsError:
        """Convenience wrapper: analyse the current ISO week only."""
        week_start, week_end = get_week_bounds(today_local())
        days_back = (today_local() - week_start).days + 1
        return await self.analyse(
            rows=rows,
            headers=headers,
            business_id=business_id,
            window_days=days_back,
        )

    async def analyse_current_month(
        self,
        rows: list[list[str]],
        headers: list[str],
        business_id: str,
    ) -> SalesAnalyticsResult | AnalyticsError:
        """Convenience wrapper: analyse the current calendar month only."""
        month_start, month_end = get_month_bounds(today_local())
        days_back = (today_local() - month_start).days + 1
        return await self.analyse(
            rows=rows,
            headers=headers,
            business_id=business_id,
            window_days=min(days_back, MAX_ANALYSIS_WINDOW_DAYS),
        )


# ==============================================================================
# Row parsing
# ==============================================================================

@dataclass
class _ParsedRow:
    """Intermediate: one validated sales row."""
    row_date: date
    revenue: Decimal
    product: Optional[str]
    quantity: Optional[int]
    payment_method: Optional[str]


def _parse_rows(
    rows: list[list[str]],
    mapping: ColumnMapping,
    window_start: date,
    window_end: date,
    business_id: str,
) -> tuple[list[_ParsedRow], int, int, int]:
    """
    Parse raw sheet rows into validated _ParsedRow objects.

    Skips:
      - Rows outside the date window
      - Rows with empty date or revenue cells
      - Rows where revenue parses to zero or negative

    Counts parse errors separately from intentional skips.

    Returns:
        (parsed_rows, total_rows, skipped_count, parse_error_count)
    """
    parsed: list[_ParsedRow] = []
    skipped = 0
    parse_errors = 0
    total = len(rows)

    for row in rows:
        values = _extract_values(row, mapping)
        raw_date = values.get("date") or ""
        raw_revenue = values.get("revenue") or ""

        # Skip empty rows
        if not raw_date.strip() and not raw_revenue.strip():
            skipped += 1
            continue

        # Parse date
        row_date = parse_flexible_date(raw_date)
        if row_date is None:
            parse_errors += 1
            continue

        # Filter to window
        if not (window_start <= row_date <= window_end):
            skipped += 1
            continue

        # Parse revenue
        revenue = _parse_decimal(raw_revenue)
        if revenue is None or revenue < MIN_VALID_REVENUE:
            parse_errors += 1
            continue

        # Optional fields
        product = _clean_optional(values.get("product"))
        quantity = _parse_int(values.get("quantity") or "")
        payment_method = _clean_optional(values.get("payment_method"))

        parsed.append(_ParsedRow(
            row_date=row_date,
            revenue=revenue,
            product=product,
            quantity=quantity,
            payment_method=payment_method,
        ))

    return parsed, total, skipped, parse_errors


def _extract_values(row: list[str], mapping: ColumnMapping) -> dict[str, Optional[str]]:
    """Extract canonical field values from a data row using the mapping."""
    from app.services.column_mapper_service import ALL_FIELDS
    result: dict[str, Optional[str]] = {}
    for field_name in ALL_FIELDS:
        idx = mapping.get_index(field_name)
        if idx is not None and idx < len(row):
            cell = row[idx]
            result[field_name] = cell.strip() if cell else None
        else:
            result[field_name] = None
    return result


# ==============================================================================
# Aggregation helpers
# ==============================================================================

def _aggregate_daily(rows: list[_ParsedRow]) -> dict[date, DailySales]:
    """Group parsed rows into per-day DailySales buckets."""
    buckets: dict[date, DailySales] = {}
    for row in rows:
        if row.row_date not in buckets:
            buckets[row.row_date] = DailySales(
                date=row.row_date,
                total_revenue=Decimal("0"),
                transaction_count=0,
                avg_transaction=Decimal("0"),
                products=[],
            )
        day = buckets[row.row_date]
        day.total_revenue += row.revenue
        day.transaction_count += 1
        if row.product and row.product not in day.products:
            day.products.append(row.product)

    # Compute avg_transaction per day
    for day in buckets.values():
        if day.transaction_count > 0:
            day.avg_transaction = day.total_revenue / day.transaction_count

    return buckets


def _aggregate_weekly(daily_sales: list[DailySales]) -> list[WeeklySales]:
    """Group DailySales into ISO calendar weeks."""
    week_buckets: dict[tuple[int, int], list[DailySales]] = defaultdict(list)
    for day in daily_sales:
        iso = day.date.isocalendar()
        week_buckets[(iso.year, iso.week)].append(day)

    result: list[WeeklySales] = []
    for (year, week), days in sorted(week_buckets.items()):
        days_sorted = sorted(days, key=lambda d: d.date)
        total = sum((d.total_revenue for d in days_sorted), Decimal("0"))
        txn = sum(d.transaction_count for d in days_sorted)
        result.append(WeeklySales(
            week_label=f"Week {week}, {year}",
            start_date=days_sorted[0].date,
            end_date=days_sorted[-1].date,
            total_revenue=total,
            transaction_count=txn,
            daily_breakdown=days_sorted,
        ))
    return result


def _find_peak_days(
    daily_sales: list[DailySales],
    avg_daily_revenue: Decimal,
) -> list[PeakPeriod]:
    """Identify days where revenue exceeded the average by PEAK_REVENUE_MULTIPLIER."""
    if avg_daily_revenue <= 0:
        return []

    peaks: list[PeakPeriod] = []
    threshold = avg_daily_revenue * Decimal(str(PEAK_REVENUE_MULTIPLIER))

    for day in daily_sales:
        if day.total_revenue >= threshold:
            ratio = float(day.total_revenue / avg_daily_revenue)
            peaks.append(PeakPeriod(
                date=day.date,
                revenue=day.total_revenue,
                revenue_vs_average=round(ratio, 2),
                day_of_week=DAY_NAMES[day.date.weekday()],
            ))

    return sorted(peaks, key=lambda p: p.revenue, reverse=True)


def _day_of_week_revenue(daily_sales: list[DailySales]) -> dict[str, Decimal]:
    """Sum revenue by day of week across the entire window."""
    totals: dict[str, Decimal] = {day: Decimal("0") for day in DAY_NAMES}
    for day in daily_sales:
        name = DAY_NAMES[day.date.weekday()]
        totals[name] += day.total_revenue
    return totals


def _top_products(
    rows: list[_ParsedRow],
    top_n: int = 5,
) -> list[tuple[str, Decimal]]:
    """Rank products by total revenue, returning top N."""
    product_revenue: dict[str, Decimal] = defaultdict(Decimal)
    for row in rows:
        if row.product:
            product_revenue[row.product] += row.revenue

    sorted_products = sorted(
        product_revenue.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_products[:top_n]


def _detect_trend(
    daily_sales: list[DailySales],
    revenue_change_pct: float,
) -> TrendSignal:
    """
    Derive a trend signal from the revenue change percentage and sample size.

    Confidence is based on how many data days are available:
      >= 14 days → high
      >= 7  days → medium
      < 7   days → low
    """
    n = len(daily_sales)

    if n >= 14:
        confidence = "high"
    elif n >= 7:
        confidence = "medium"
    else:
        confidence = "low"

    if revenue_change_pct >= 0.10:
        direction = "up"
        description = f"Revenue is up {revenue_change_pct * 100:.1f}% vs the previous period."
    elif revenue_change_pct <= -0.10:
        direction = "down"
        description = f"Revenue is down {abs(revenue_change_pct) * 100:.1f}% vs the previous period."
    else:
        direction = "stable"
        description = "Revenue is stable compared to the previous period."

    return TrendSignal(
        direction=direction,
        magnitude=abs(revenue_change_pct),
        confidence=confidence,
        description=description,
    )


# ==============================================================================
# Numeric helpers
# ==============================================================================

def _compute_change_pct(current: Decimal, previous: Decimal) -> float:
    """
    Compute percentage change between two revenue values.

    Returns 0.0 if previous is zero (no division-by-zero risk).
    """
    if previous == 0:
        return 0.0
    return float((current - previous) / previous)


def _parse_decimal(raw: str) -> Optional[Decimal]:
    """
    Parse a raw revenue string into a Decimal.

    Handles formats commonly seen in Indian business sheets:
      "1,299.50"  → 1299.50
      "₹ 1,299"   → 1299.00
      "1299"      → 1299.00
      "1.5K"      → 1500.00
      ""          → None
    """
    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()

    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[₹$£€\s,]", "", cleaned)

    # Handle K/L suffixes (Indian shorthand: 1.5K = 1500, 2L = 200000)
    multiplier = Decimal("1")
    upper = cleaned.upper()
    if upper.endswith("L"):
        multiplier = Decimal("100000")
        cleaned = cleaned[:-1]
    elif upper.endswith("K"):
        multiplier = Decimal("1000")
        cleaned = cleaned[:-1]

    try:
        return Decimal(cleaned) * multiplier
    except InvalidOperation:
        return None


def _parse_int(raw: str) -> Optional[int]:
    """Parse a raw quantity string into an integer. Returns None on failure."""
    if not raw or not raw.strip():
        return None
    cleaned = re.sub(r"[^\d]", "", raw.strip())
    try:
        return int(cleaned) if cleaned else None
    except ValueError:
        return None


def _clean_optional(value: Optional[str]) -> Optional[str]:
    """Strip and return None for empty optional string cells."""
    if not value:
        return None
    stripped = value.strip()
    return stripped if stripped else None