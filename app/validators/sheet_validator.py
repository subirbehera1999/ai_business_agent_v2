# ==============================================================================
# File: app/validators/sheet_validator.py
# Purpose: Validates Google Sheets sales data before it enters the
#          analytics processing pipeline.
#
#          Called by sales_analysis_job.py after fetching raw rows from
#          google_sheets_client.py and after column_mapper_service.py
#          has identified which columns map to which fields.
#
#          Why validate Sheets data?
#            Google Sheets is edited manually by business owners.
#            Real-world spreadsheets contain:
#              - Empty rows (spacer rows, section dividers)
#              - Rows with partial data (date filled, revenue missing)
#              - Wrong data types (text in a number column)
#              - Revenue entered as "₹1,500" or "1500.00" or "1,500.50"
#              - Dates in various formats (DD/MM/YYYY, MM-DD-YYYY, etc.)
#              - Negative revenue (refunds — valid but need flagging)
#              - Future-dated rows (pre-entered upcoming sales)
#              - Header rows accidentally included in the data range
#
#            Letting bad rows into the analytics engine produces wrong
#            revenue totals, incorrect trend detection, and misleading
#            insights sent to the business owner via WhatsApp.
#
#          Validation pipeline per row:
#            1. Empty row check     — skip rows with no usable data
#            2. Date extraction     — parse date from the mapped column
#            3. Date validation     — valid date, not future, not too old
#            4. Revenue extraction  — parse currency string to Decimal
#            5. Revenue validation  — not NaN, within sane range
#            6. Optional fields     — quantity, category, notes (best-effort)
#            7. Row classification  — VALID / SKIPPED / INVALID
#
#          Two-level outcome:
#            VALID:    Row passes all checks — include in analytics
#            SKIPPED:  Row has recoverable issues — exclude from analytics
#                      but do not count as an error (empty rows, headers)
#            INVALID:  Row has data errors — log for admin visibility
#
#          Currency parsing:
#            Revenue values are normalised to Python Decimal to avoid
#            floating-point rounding errors in revenue calculations.
#            Handles: "1500", "1,500", "₹1500", "₹ 1,500.50", "1500.00"
#            Rejects:  "N/A", "pending", "", None, negative beyond threshold
#
#          Date parsing:
#            Attempts multiple common date formats before rejecting.
#            Supported formats include DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD,
#            DD-MM-YYYY, DD MMM YYYY, and Excel serial date numbers.
#
#          Batch result:
#            SheetValidationResult contains:
#              - valid_rows:    List of ValidatedRow (ready for analytics)
#              - skipped_count: Rows skipped (empty, header-like)
#              - invalid_count: Rows with data errors
#              - error_summary: Dict of error type → count
#              - date_range:    Earliest and latest date in valid rows
#              - total_revenue: Sum of all valid revenue values
# ==============================================================================

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from app.config.constants import ServiceName

logger = logging.getLogger(ServiceName.SHEET_VALIDATOR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FUTURE_DAYS: int = 1           # allow rows dated 1 day ahead (timezone buffer)
MAX_PAST_DAYS: int = 730           # reject rows older than 2 years
MAX_REVENUE_VALUE: Decimal = Decimal("100_000_000")  # 10 crore — sanity ceiling
MIN_REVENUE_VALUE: Decimal = Decimal("-1_000_000")   # allow refunds up to 10 lakh
MAX_ROWS_PER_BATCH: int = 500      # enforced by sales_analysis_job.py

# Date formats tried in order (most common first)
_DATE_FORMATS = (
    "%d/%m/%Y",    # 31/12/2024  — most common in India
    "%Y-%m-%d",    # 2024-12-31  — ISO format
    "%d-%m-%Y",    # 31-12-2024
    "%m/%d/%Y",    # 12/31/2024  — US format
    "%d %b %Y",    # 31 Dec 2024
    "%d %B %Y",    # 31 December 2024
    "%b %d, %Y",   # Dec 31, 2024
    "%B %d, %Y",   # December 31, 2024
    "%d/%m/%y",    # 31/12/24    — 2-digit year
    "%Y/%m/%d",    # 2024/12/31
)

# Currency symbols to strip before parsing
_CURRENCY_SYMBOLS = re.compile(r"[₹$€£¥₩,\s]")

# Patterns that indicate a non-data row (header-like)
_HEADER_PATTERNS = re.compile(
    r"^(date|day|month|revenue|sales|amount|total|sr\.?\s*no|s\.?\s*no|#|"
    r"income|earning|sale|qty|quantity|item|product|category|note|description)$",
    re.IGNORECASE,
)


# ==============================================================================
# Row outcome enums
# ==============================================================================

class RowOutcome:
    """Row-level validation outcome constants."""
    VALID = "valid"
    SKIPPED = "skipped"
    INVALID = "invalid"


class RowSkipReason:
    """Why a row was skipped (not an error — expected situations)."""
    EMPTY_ROW = "empty_row"
    HEADER_ROW = "header_row"
    FUTURE_DATE = "future_date"
    DATE_TOO_OLD = "date_too_old"
    ZERO_REVENUE = "zero_revenue"


class RowInvalidReason:
    """Why a row is invalid (unexpected data error)."""
    MISSING_DATE = "missing_date"
    UNPARSEABLE_DATE = "unparseable_date"
    MISSING_REVENUE = "missing_revenue"
    UNPARSEABLE_REVENUE = "unparseable_revenue"
    REVENUE_OUT_OF_RANGE = "revenue_out_of_range"
    NEGATIVE_REVENUE = "negative_revenue"


# ==============================================================================
# Validated row dataclass
# ==============================================================================

@dataclass
class ValidatedRow:
    """
    A single sales row that has passed all validation checks.

    All fields are typed Python values — no raw strings.
    This is what analytics_service.py receives and processes.

    Attributes:
        row_index:    Original row number in the sheet (1-based).
        sale_date:    Parsed Python date object.
        revenue:      Decimal revenue value (never None for valid rows).
        quantity:     Integer quantity sold (None if not in sheet).
        category:     Product/service category (None if not in sheet).
        notes:        Free-text notes (None if not in sheet).
        raw_row:      Original dict from the sheet (for debugging).
    """
    row_index: int
    sale_date: date
    revenue: Decimal
    quantity: Optional[int] = None
    category: Optional[str] = None
    notes: Optional[str] = None
    raw_row: dict = field(default_factory=dict)

    @property
    def is_refund(self) -> bool:
        """Return True if this row represents a refund (negative revenue)."""
        return self.revenue < Decimal("0")


@dataclass
class RowValidationDetail:
    """
    Per-row validation detail for debugging and admin reporting.

    Only populated for SKIPPED and INVALID rows.
    """
    row_index: int
    outcome: str
    reason: str
    raw_date_value: Optional[str] = None
    raw_revenue_value: Optional[str] = None


# ==============================================================================
# Batch result dataclass
# ==============================================================================

@dataclass
class SheetValidationResult:
    """
    Result of validating an entire sheet data batch.

    Attributes:
        valid_rows:      Rows that passed all checks — ready for analytics.
        skipped_count:   Rows intentionally skipped (empty, header, future).
        invalid_count:   Rows with data errors.
        error_summary:   Dict of error reason → count.
        skip_summary:    Dict of skip reason → count.
        total_revenue:   Sum of revenue across all valid rows.
        date_range_start: Earliest sale date in valid rows.
        date_range_end:   Latest sale date in valid rows.
        row_details:     Per-row detail for skipped/invalid rows (debug).
        total_rows_processed: Total rows examined.
    """
    valid_rows: list[ValidatedRow] = field(default_factory=list)
    skipped_count: int = 0
    invalid_count: int = 0
    error_summary: dict[str, int] = field(default_factory=dict)
    skip_summary: dict[str, int] = field(default_factory=dict)
    total_revenue: Decimal = Decimal("0")
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    row_details: list[RowValidationDetail] = field(default_factory=list)
    total_rows_processed: int = 0

    @property
    def valid_count(self) -> int:
        return len(self.valid_rows)

    @property
    def success_rate(self) -> float:
        if self.total_rows_processed == 0:
            return 0.0
        return round(self.valid_count / self.total_rows_processed * 100, 1)

    @property
    def has_valid_data(self) -> bool:
        return len(self.valid_rows) > 0

    def _record_skip(self, row_index: int, reason: str, **kwargs) -> None:
        self.skipped_count += 1
        self.skip_summary[reason] = self.skip_summary.get(reason, 0) + 1
        self.row_details.append(RowValidationDetail(
            row_index=row_index,
            outcome=RowOutcome.SKIPPED,
            reason=reason,
            **kwargs,
        ))

    def _record_invalid(self, row_index: int, reason: str, **kwargs) -> None:
        self.invalid_count += 1
        self.error_summary[reason] = self.error_summary.get(reason, 0) + 1
        self.row_details.append(RowValidationDetail(
            row_index=row_index,
            outcome=RowOutcome.INVALID,
            reason=reason,
            **kwargs,
        ))

    def _add_valid(self, row: ValidatedRow) -> None:
        self.valid_rows.append(row)
        self.total_revenue += row.revenue
        # Update date range
        if self.date_range_start is None or row.sale_date < self.date_range_start:
            self.date_range_start = row.sale_date
        if self.date_range_end is None or row.sale_date > self.date_range_end:
            self.date_range_end = row.sale_date


# ==============================================================================
# Sheet Validator
# ==============================================================================

class SheetValidator:
    """
    Validates Google Sheets sales data rows before analytics processing.

    Stateless — the same instance is safe to use for all businesses.
    All currency and date parsing happens in pure Python with no I/O.

    Usage:
        validator = SheetValidator()
        result = validator.validate_batch(
            rows=raw_rows,                    # list of dicts from sheets client
            date_column="Date",               # from column_mapper_service.py
            revenue_column="Revenue",         # from column_mapper_service.py
            quantity_column="Qty",            # optional
            category_column="Category",       # optional
        )
        if result.has_valid_data:
            analytics_service.analyse(result.valid_rows)
    """

    def validate_batch(
        self,
        rows: list[dict[str, Any]],
        date_column: str,
        revenue_column: str,
        quantity_column: Optional[str] = None,
        category_column: Optional[str] = None,
        notes_column: Optional[str] = None,
        business_id: Optional[str] = None,
    ) -> SheetValidationResult:
        """
        Validate a batch of raw sheet rows.

        Args:
            rows:             List of row dicts from google_sheets_client.py.
                              Each dict maps column header → cell value.
            date_column:      Column name containing sale dates.
            revenue_column:   Column name containing revenue values.
            quantity_column:  Column name for quantity (optional).
            category_column:  Column name for product category (optional).
            notes_column:     Column name for notes (optional).
            business_id:      For structured logging only.

        Returns:
            SheetValidationResult with valid_rows ready for analytics.
        """
        result = SheetValidationResult()
        today = date.today()

        log_extra = {
            "service": ServiceName.SHEET_VALIDATOR,
            "business_id": business_id or "unknown",
            "total_rows": len(rows),
            "date_column": date_column,
            "revenue_column": revenue_column,
        }
        logger.debug("Sheet validation started", extra=log_extra)

        for row_index, raw_row in enumerate(rows, start=1):
            result.total_rows_processed += 1

            # ── Step 1: Empty row check ───────────────────────────────
            if _is_empty_row(raw_row):
                result._record_skip(row_index, RowSkipReason.EMPTY_ROW)
                continue

            # ── Step 2: Header row detection ─────────────────────────
            raw_date_val = str(raw_row.get(date_column, "") or "").strip()
            if _looks_like_header(raw_date_val):
                result._record_skip(row_index, RowSkipReason.HEADER_ROW)
                continue

            # ── Step 3: Date parsing ──────────────────────────────────
            if not raw_date_val:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.MISSING_DATE,
                    raw_date_value=raw_date_val,
                )
                continue

            parsed_date = _parse_date(raw_date_val)
            if parsed_date is None:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.UNPARSEABLE_DATE,
                    raw_date_value=raw_date_val[:30],
                )
                continue

            # ── Step 4: Date range checks ─────────────────────────────
            if parsed_date > today + timedelta(days=MAX_FUTURE_DAYS):
                result._record_skip(
                    row_index,
                    RowSkipReason.FUTURE_DATE,
                    raw_date_value=raw_date_val,
                )
                continue

            if parsed_date < today - timedelta(days=MAX_PAST_DAYS):
                result._record_skip(
                    row_index,
                    RowSkipReason.DATE_TOO_OLD,
                    raw_date_value=raw_date_val,
                )
                continue

            # ── Step 5: Revenue extraction ────────────────────────────
            raw_rev_val = str(raw_row.get(revenue_column, "") or "").strip()
            if not raw_rev_val:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.MISSING_REVENUE,
                    raw_revenue_value=raw_rev_val,
                )
                continue

            parsed_revenue = _parse_currency(raw_rev_val)
            if parsed_revenue is None:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.UNPARSEABLE_REVENUE,
                    raw_revenue_value=raw_rev_val[:30],
                )
                continue

            # ── Step 6: Revenue range check ───────────────────────────
            if parsed_revenue < MIN_REVENUE_VALUE:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.REVENUE_OUT_OF_RANGE,
                    raw_revenue_value=raw_rev_val[:30],
                )
                continue

            if parsed_revenue > MAX_REVENUE_VALUE:
                result._record_invalid(
                    row_index,
                    RowInvalidReason.REVENUE_OUT_OF_RANGE,
                    raw_revenue_value=raw_rev_val[:30],
                )
                continue

            # ── Step 7: Zero revenue ──────────────────────────────────
            # Zero-revenue rows are skipped — they add no analytics value
            # but are not data errors (business may have had a no-sale day)
            if parsed_revenue == Decimal("0"):
                result._record_skip(row_index, RowSkipReason.ZERO_REVENUE)
                continue

            # ── Step 8: Optional fields ───────────────────────────────
            quantity = _parse_quantity(raw_row.get(quantity_column))
            category = _parse_text_field(raw_row.get(category_column), max_len=100)
            notes = _parse_text_field(raw_row.get(notes_column), max_len=500)

            # ── All checks passed ─────────────────────────────────────
            validated_row = ValidatedRow(
                row_index=row_index,
                sale_date=parsed_date,
                revenue=parsed_revenue,
                quantity=quantity,
                category=category,
                notes=notes,
                raw_row=raw_row,
            )
            result._add_valid(validated_row)

        # ── Log batch summary ─────────────────────────────────────────
        logger.info(
            "Sheet validation complete",
            extra={
                **log_extra,
                "valid": result.valid_count,
                "skipped": result.skipped_count,
                "invalid": result.invalid_count,
                "success_rate_pct": result.success_rate,
                "total_revenue": str(result.total_revenue),
                "error_summary": result.error_summary,
            },
        )

        return result

    def validate_column_mapping(
        self,
        sample_rows: list[dict[str, Any]],
        date_column: str,
        revenue_column: str,
    ) -> tuple[bool, str]:
        """
        Quick pre-flight check that the mapped columns actually contain
        parseable data before running the full batch validation.

        Samples up to 5 non-empty rows to verify:
          - The date column exists and contains parseable dates
          - The revenue column exists and contains parseable numbers

        Called by sales_analysis_job.py after column_mapper_service.py
        returns a mapping but before full batch processing.

        Args:
            sample_rows:    A sample of raw rows (first 10–20 is enough).
            date_column:    Mapped date column name.
            revenue_column: Mapped revenue column name.

        Returns:
            Tuple (is_valid: bool, error_message: str).
            error_message is empty string if valid.
        """
        date_successes = 0
        revenue_successes = 0
        rows_checked = 0

        for row in sample_rows[:20]:
            if _is_empty_row(row):
                continue

            raw_date = str(row.get(date_column, "") or "").strip()
            raw_rev = str(row.get(revenue_column, "") or "").strip()

            if not raw_date or _looks_like_header(raw_date):
                continue

            rows_checked += 1

            if _parse_date(raw_date) is not None:
                date_successes += 1

            if raw_rev and _parse_currency(raw_rev) is not None:
                revenue_successes += 1

            if rows_checked >= 5:
                break

        if rows_checked == 0:
            return False, "No non-empty data rows found in sample"

        date_rate = date_successes / rows_checked
        revenue_rate = revenue_successes / rows_checked

        if date_rate < 0.5:
            return (
                False,
                f"Date column '{date_column}' could not be parsed in "
                f"{rows_checked - date_successes}/{rows_checked} sample rows. "
                f"Check the date format.",
            )

        if revenue_rate < 0.5:
            return (
                False,
                f"Revenue column '{revenue_column}' could not be parsed in "
                f"{rows_checked - revenue_successes}/{rows_checked} sample rows. "
                f"Check that it contains numeric values.",
            )

        return True, ""


# ==============================================================================
# Pure helper functions
# ==============================================================================

def _is_empty_row(row: dict[str, Any]) -> bool:
    """
    Return True if a row contains no meaningful data.

    A row is empty if all values are None, empty string, or whitespace.
    """
    if not row:
        return True
    return all(
        not str(v).strip() for v in row.values()
        if v is not None
    )


def _looks_like_header(value: str) -> bool:
    """
    Return True if the value looks like a column header rather than data.

    Matches common header names like "Date", "Revenue", "S.No", etc.
    Used to skip accidentally included header rows in the data range.
    """
    return bool(_HEADER_PATTERNS.match(value.strip()))


def _parse_date(raw: str) -> Optional[date]:
    """
    Parse a date string into a Python date object.

    Tries all formats in _DATE_FORMATS.
    Also handles Excel serial date numbers (integer days since 1900-01-01).

    Args:
        raw: Raw date string from the spreadsheet cell.

    Returns:
        Python date object, or None if all formats fail.
    """
    cleaned = raw.strip()
    if not cleaned:
        return None

    # Try standard date format strings
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    # Try Excel serial date number (integer like "45678")
    try:
        serial = int(float(cleaned))
        if 1000 < serial < 100000:   # plausible Excel serial range
            # Excel serial 1 = 1900-01-01
            # Python epoch is 1970-01-01
            excel_epoch = date(1899, 12, 30)
            return excel_epoch + timedelta(days=serial)
    except (ValueError, OverflowError):
        pass

    return None


def _parse_currency(raw: str) -> Optional[Decimal]:
    """
    Parse a currency string into a Python Decimal.

    Handles:
      "1500"       → Decimal("1500")
      "1,500"      → Decimal("1500")
      "₹1,500.50"  → Decimal("1500.50")
      "₹ 1500"     → Decimal("1500")
      "-500"       → Decimal("-500")   (refund)
      "1500.00"    → Decimal("1500.00")

    Rejects:
      "N/A", "pending", "", "abc" → None

    Args:
        raw: Raw cell value string.

    Returns:
        Decimal, or None if the value cannot be parsed as a number.
    """
    cleaned = raw.strip()
    if not cleaned:
        return None

    # Strip currency symbols, spaces, and commas
    numeric_str = _CURRENCY_SYMBOLS.sub("", cleaned)

    # Handle parentheses as negative: (1500) → -1500
    if numeric_str.startswith("(") and numeric_str.endswith(")"):
        numeric_str = "-" + numeric_str[1:-1]

    if not numeric_str:
        return None

    # Reject clearly non-numeric values
    if re.search(r"[a-zA-Z]", numeric_str):
        return None

    try:
        return Decimal(numeric_str)
    except InvalidOperation:
        return None


def _parse_quantity(raw: Any) -> Optional[int]:
    """
    Parse a quantity value to int. Returns None on failure.

    Quantities must be non-negative integers.
    Floats are rounded down (3.7 units → 3).
    """
    if raw is None:
        return None
    try:
        val = int(float(str(raw).strip()))
        return val if val >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_text_field(raw: Any, max_len: int = 200) -> Optional[str]:
    """
    Sanitise a free-text field for storage.

    Strips whitespace and control characters.
    Truncates to max_len characters.
    Returns None if empty after stripping.
    """
    if raw is None:
        return None
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", str(raw)).strip()
    cleaned = cleaned[:max_len]
    return cleaned if cleaned else None