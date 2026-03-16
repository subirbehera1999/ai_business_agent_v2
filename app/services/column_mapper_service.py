# ==============================================================================
# File: app/services/column_mapper_service.py
# Purpose: Dynamically maps Google Sheets column headers to the system's
#          internal field names for sales analytics processing.
#
#          Problem this solves:
#            Different businesses set up their Google Sheets differently.
#            One business may name a column "Sale Date", another uses
#            "Transaction Date", another uses "Date". The analytics engine
#            needs a canonical field set regardless of how each business
#            named their columns.
#
#          Mapping strategy (three levels, in priority order):
#            1. EXACT match    — "date" == "date" (case-insensitive)
#            2. ALIAS match    — "transaction date" → FIELD_DATE
#            3. FUZZY match    — token overlap scoring for typos/variations
#
#          Required fields (analytics engine cannot run without these):
#            - date        : transaction/sale date
#            - revenue     : sale amount / total value
#
#          Optional fields (enriches analytics if present):
#            - quantity    : units sold
#            - product     : product or service name
#            - category    : product/service category
#            - customer    : customer name or ID
#            - payment_method : cash / card / UPI etc.
#            - notes       : free-text notes column
#
#          Output:
#            ColumnMapping — a frozen dataclass that maps each canonical
#            field name to its actual column index in the sheet, or None
#            if that field was not found.
#
#          The sheet_validator.py uses this service to validate incoming
#          sheets before they enter the analytics pipeline.
# ==============================================================================

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config.constants import ServiceName

logger = logging.getLogger(ServiceName.API)

# ---------------------------------------------------------------------------
# Canonical field names — these are the names analytics_service.py uses
# ---------------------------------------------------------------------------
FIELD_DATE           = "date"
FIELD_REVENUE        = "revenue"
FIELD_QUANTITY       = "quantity"
FIELD_PRODUCT        = "product"
FIELD_CATEGORY       = "category"
FIELD_CUSTOMER       = "customer"
FIELD_PAYMENT_METHOD = "payment_method"
FIELD_NOTES          = "notes"

# Fields the analytics engine cannot function without
REQUIRED_FIELDS: frozenset[str] = frozenset({FIELD_DATE, FIELD_REVENUE})

# All recognised canonical fields
ALL_FIELDS: tuple[str, ...] = (
    FIELD_DATE,
    FIELD_REVENUE,
    FIELD_QUANTITY,
    FIELD_PRODUCT,
    FIELD_CATEGORY,
    FIELD_CUSTOMER,
    FIELD_PAYMENT_METHOD,
    FIELD_NOTES,
)

# ---------------------------------------------------------------------------
# Alias table — maps known header variations to canonical field names.
# Keys are lowercase, stripped. Add more aliases without touching service logic.
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # Date aliases
    "date":               FIELD_DATE,
    "sale date":          FIELD_DATE,
    "sales date":         FIELD_DATE,
    "transaction date":   FIELD_DATE,
    "order date":         FIELD_DATE,
    "invoice date":       FIELD_DATE,
    "billing date":       FIELD_DATE,
    "purchase date":      FIELD_DATE,
    "booking date":       FIELD_DATE,
    "visit date":         FIELD_DATE,
    "service date":       FIELD_DATE,
    "dt":                 FIELD_DATE,

    # Revenue aliases
    "revenue":            FIELD_REVENUE,
    "amount":             FIELD_REVENUE,
    "total":              FIELD_REVENUE,
    "total amount":       FIELD_REVENUE,
    "sale amount":        FIELD_REVENUE,
    "sales amount":       FIELD_REVENUE,
    "price":              FIELD_REVENUE,
    "value":              FIELD_REVENUE,
    "sale value":         FIELD_REVENUE,
    "billing amount":     FIELD_REVENUE,
    "invoice amount":     FIELD_REVENUE,
    "payment amount":     FIELD_REVENUE,
    "gross":              FIELD_REVENUE,
    "gross amount":       FIELD_REVENUE,
    "net":                FIELD_REVENUE,
    "net amount":         FIELD_REVENUE,
    "income":             FIELD_REVENUE,
    "earnings":           FIELD_REVENUE,
    "inr":                FIELD_REVENUE,
    "rs":                 FIELD_REVENUE,
    "rupees":             FIELD_REVENUE,

    # Quantity aliases
    "quantity":           FIELD_QUANTITY,
    "qty":                FIELD_QUANTITY,
    "units":              FIELD_QUANTITY,
    "count":              FIELD_QUANTITY,
    "no of items":        FIELD_QUANTITY,
    "number of items":    FIELD_QUANTITY,
    "items":              FIELD_QUANTITY,
    "pieces":             FIELD_QUANTITY,
    "pcs":                FIELD_QUANTITY,

    # Product aliases
    "product":            FIELD_PRODUCT,
    "item":               FIELD_PRODUCT,
    "item name":          FIELD_PRODUCT,
    "product name":       FIELD_PRODUCT,
    "service":            FIELD_PRODUCT,
    "service name":       FIELD_PRODUCT,
    "description":        FIELD_PRODUCT,
    "menu item":          FIELD_PRODUCT,
    "treatment":          FIELD_PRODUCT,

    # Category aliases
    "category":           FIELD_CATEGORY,
    "cat":                FIELD_CATEGORY,
    "type":               FIELD_CATEGORY,
    "product type":       FIELD_CATEGORY,
    "service type":       FIELD_CATEGORY,
    "department":         FIELD_CATEGORY,
    "section":            FIELD_CATEGORY,

    # Customer aliases
    "customer":           FIELD_CUSTOMER,
    "customer name":      FIELD_CUSTOMER,
    "client":             FIELD_CUSTOMER,
    "client name":        FIELD_CUSTOMER,
    "patient":            FIELD_CUSTOMER,
    "patient name":       FIELD_CUSTOMER,
    "buyer":              FIELD_CUSTOMER,
    "member":             FIELD_CUSTOMER,
    "guest":              FIELD_CUSTOMER,

    # Payment method aliases
    "payment method":     FIELD_PAYMENT_METHOD,
    "payment":            FIELD_PAYMENT_METHOD,
    "payment mode":       FIELD_PAYMENT_METHOD,
    "mode of payment":    FIELD_PAYMENT_METHOD,
    "pay mode":           FIELD_PAYMENT_METHOD,
    "pay method":         FIELD_PAYMENT_METHOD,
    "transaction type":   FIELD_PAYMENT_METHOD,
    "method":             FIELD_PAYMENT_METHOD,

    # Notes aliases
    "notes":              FIELD_NOTES,
    "note":               FIELD_NOTES,
    "remarks":            FIELD_NOTES,
    "remark":             FIELD_NOTES,
    "comments":           FIELD_NOTES,
    "comment":            FIELD_NOTES,
    "details":            FIELD_NOTES,
    "info":               FIELD_NOTES,
    "additional info":    FIELD_NOTES,
}

# Minimum fuzzy score (0.0–1.0) to accept a match
_FUZZY_MIN_SCORE: float = 0.60


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass(frozen=True)
class ColumnMapping:
    """
    Maps canonical field names to their column index (0-based) in the sheet.

    A value of None means the field was not found in the header row.
    Required fields (date, revenue) must not be None for the mapping to
    be considered valid — check is_valid before using.

    Attributes:
        date:           Column index for the date field.
        revenue:        Column index for the revenue field.
        quantity:       Column index for the quantity field (optional).
        product:        Column index for the product field (optional).
        category:       Column index for the category field (optional).
        customer:       Column index for the customer field (optional).
        payment_method: Column index for the payment method field (optional).
        notes:          Column index for the notes field (optional).
        raw_headers:    Original header row as received from Google Sheets.
        missing_required: Set of required field names that were not found.
        match_log:      List of (canonical_field, raw_header, match_type)
                        for debug traceability.
    """

    date:           Optional[int]
    revenue:        Optional[int]
    quantity:       Optional[int]
    product:        Optional[int]
    category:       Optional[int]
    customer:       Optional[int]
    payment_method: Optional[int]
    notes:          Optional[int]
    raw_headers:    tuple[str, ...]
    missing_required: frozenset[str]
    match_log:      tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        """True if all required fields were found."""
        return len(self.missing_required) == 0

    @property
    def has_quantity(self) -> bool:
        return self.quantity is not None

    @property
    def has_product(self) -> bool:
        return self.product is not None

    @property
    def has_category(self) -> bool:
        return self.category is not None

    @property
    def has_customer(self) -> bool:
        return self.customer is not None

    @property
    def has_payment_method(self) -> bool:
        return self.payment_method is not None

    @property
    def has_notes(self) -> bool:
        return self.notes is not None

    def get_index(self, field_name: str) -> Optional[int]:
        """
        Retrieve a column index by canonical field name.

        Useful for generic row-reading code in analytics_service.py:
            idx = mapping.get_index("revenue")
            value = row[idx] if idx is not None else None

        Args:
            field_name: One of the FIELD_* constants.

        Returns:
            int | None: Column index, or None if field was not found.
        """
        return getattr(self, field_name, None)

    def to_dict(self) -> dict[str, Optional[int]]:
        """Return the field→index mapping as a plain dict."""
        return {f: self.get_index(f) for f in ALL_FIELDS}

    def __str__(self) -> str:
        found = [f for f in ALL_FIELDS if self.get_index(f) is not None]
        missing = list(self.missing_required)
        return (
            f"ColumnMapping("
            f"valid={self.is_valid} "
            f"found={found} "
            f"missing_required={missing})"
        )


@dataclass(frozen=True)
class MappingError:
    """
    Describes why a header row could not be mapped.

    Returned alongside a ColumnMapping when is_valid=False so callers
    have a human-readable explanation for logging and WhatsApp alerts.
    """

    missing_fields: frozenset[str]
    suggestions:    tuple[str, ...]   # Closest raw headers to each missing field
    raw_headers:    tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"MappingError(missing={sorted(self.missing_fields)} "
            f"suggestions={self.suggestions})"
        )


# ==============================================================================
# Column Mapper Service
# ==============================================================================

class ColumnMapperService:
    """
    Maps a Google Sheets header row to the system's canonical field names.

    Stateless — a single instance can be shared across the application.

    Usage:
        mapper = ColumnMapperService()

        mapping = mapper.map_headers(
            headers=["Date", "Product Name", "Total Amount", "Qty"],
            business_id="uuid-here",
        )

        if not mapping.is_valid:
            error = mapper.build_error(mapping)
            logger.warning(str(error))
        else:
            date_col    = mapping.date      # e.g. 0
            revenue_col = mapping.revenue   # e.g. 2
    """

    def map_headers(
        self,
        headers: list[str],
        business_id: str,
    ) -> ColumnMapping:
        """
        Map a list of raw column headers to canonical field names.

        Processing steps:
          1. Normalise each header (lowercase, strip, collapse whitespace)
          2. For each canonical field, attempt exact → alias → fuzzy match
          3. Each column index can only be assigned to ONE canonical field
             (first-match-wins prevents double-assignment)
          4. Identify which required fields were not found
          5. Return a ColumnMapping with all results and a match log

        Args:
            headers:     Raw header row from Google Sheets row 1.
                         Example: ["Date", "Product", "Sale Amount", "Qty"]
            business_id: Business ID for log traceability.

        Returns:
            ColumnMapping: Always returns — never raises.
                           Check mapping.is_valid before using.
        """
        if not headers:
            logger.warning(
                "Column mapping received empty headers",
                extra={"service": ServiceName.API, "business_id": business_id},
            )
            return _empty_mapping(headers)

        normalised = [_normalise_header(h) for h in headers]
        assigned_indices: set[int] = set()
        result: dict[str, Optional[int]] = {f: None for f in ALL_FIELDS}
        match_log: list[tuple[str, str, str]] = []

        for canonical_field in ALL_FIELDS:
            idx, match_type = _find_column(
                canonical_field=canonical_field,
                normalised_headers=normalised,
                assigned_indices=assigned_indices,
            )
            if idx is not None:
                result[canonical_field] = idx
                assigned_indices.add(idx)
                raw_matched = headers[idx] if idx < len(headers) else ""
                match_log.append((canonical_field, raw_matched, match_type))
                logger.debug(
                    "Column mapped",
                    extra={
                        "service": ServiceName.API,
                        "business_id": business_id,
                        "canonical_field": canonical_field,
                        "raw_header": raw_matched,
                        "match_type": match_type,
                        "column_index": idx,
                    },
                )
            else:
                logger.debug(
                    "Column not found",
                    extra={
                        "service": ServiceName.API,
                        "business_id": business_id,
                        "canonical_field": canonical_field,
                    },
                )

        missing_required = frozenset(
            f for f in REQUIRED_FIELDS if result[f] is None
        )

        mapping = ColumnMapping(
            date=result[FIELD_DATE],
            revenue=result[FIELD_REVENUE],
            quantity=result[FIELD_QUANTITY],
            product=result[FIELD_PRODUCT],
            category=result[FIELD_CATEGORY],
            customer=result[FIELD_CUSTOMER],
            payment_method=result[FIELD_PAYMENT_METHOD],
            notes=result[FIELD_NOTES],
            raw_headers=tuple(headers),
            missing_required=missing_required,
            match_log=tuple(match_log),
        )

        if mapping.is_valid:
            logger.info(
                "Column mapping succeeded",
                extra={
                    "service": ServiceName.API,
                    "business_id": business_id,
                    "mapped_fields": [f for f in ALL_FIELDS if result[f] is not None],
                    "total_headers": len(headers),
                },
            )
        else:
            logger.warning(
                "Column mapping incomplete — required fields missing",
                extra={
                    "service": ServiceName.API,
                    "business_id": business_id,
                    "missing_required": sorted(missing_required),
                    "raw_headers": headers,
                },
            )

        return mapping

    def build_error(self, mapping: ColumnMapping) -> MappingError:
        """
        Build a MappingError from an invalid ColumnMapping.

        Finds the closest raw header to each missing required field
        to help businesses understand which column was not recognised.

        Args:
            mapping: A ColumnMapping where is_valid=False.

        Returns:
            MappingError with suggestions for each missing field.
        """
        suggestions: list[str] = []
        for missing_field in mapping.missing_required:
            best = _find_closest_header(missing_field, list(mapping.raw_headers))
            if best:
                suggestions.append(f'For "{missing_field}", did you mean: "{best}"?')

        return MappingError(
            missing_fields=mapping.missing_required,
            suggestions=tuple(suggestions),
            raw_headers=mapping.raw_headers,
        )

    def extract_row_values(
        self,
        row: list[str],
        mapping: ColumnMapping,
    ) -> dict[str, Optional[str]]:
        """
        Extract canonical field values from a single data row.

        Uses the mapping to pull values by index. Returns None for
        fields that were not mapped or whose index is out of bounds.

        Args:
            row:     A single data row from Google Sheets.
            mapping: A valid ColumnMapping from map_headers().

        Returns:
            dict[str, Optional[str]]: Canonical field → raw cell value.

        Example:
            values = mapper.extract_row_values(row, mapping)
            date_str    = values["date"]      # "2024-10-15"
            revenue_str = values["revenue"]   # "1299.50"
        """
        extracted: dict[str, Optional[str]] = {}
        for canonical_field in ALL_FIELDS:
            idx = mapping.get_index(canonical_field)
            if idx is not None and idx < len(row):
                cell = row[idx]
                extracted[canonical_field] = cell.strip() if cell else None
            else:
                extracted[canonical_field] = None
        return extracted


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _normalise_header(header: str) -> str:
    """
    Normalise a raw header string for comparison.

    Steps:
      1. Strip leading/trailing whitespace
      2. Lowercase
      3. Collapse multiple spaces/tabs to single space
      4. Remove non-alphanumeric characters except spaces
         (handles headers like "Sale_Amount", "Sale-Amount", "Sale#Amount")

    Args:
        header: Raw column header from Google Sheets.

    Returns:
        str: Normalised header string.
    """
    text = header.strip().lower()
    text = re.sub(r"[_\-#/\\|]+", " ", text)   # Replace separators with space
    text = re.sub(r"[^\w\s]", "", text)          # Remove remaining punctuation
    text = re.sub(r"\s+", " ", text)             # Collapse whitespace
    return text.strip()


def _find_column(
    canonical_field: str,
    normalised_headers: list[str],
    assigned_indices: set[int],
) -> tuple[Optional[int], str]:
    """
    Find the best matching column index for a canonical field.

    Tries three strategies in order:
      1. Exact match: normalised header == canonical field name exactly
      2. Alias match: normalised header is in the alias table for this field
      3. Fuzzy match: token overlap score >= _FUZZY_MIN_SCORE

    Skips column indices that are already assigned to another field.

    Args:
        canonical_field:    The system field name to find (e.g. "revenue").
        normalised_headers: Pre-normalised header list.
        assigned_indices:   Already-claimed column indices (skip these).

    Returns:
        tuple[Optional[int], str]: (column_index, match_type) or (None, "").
        match_type is one of: "exact", "alias", "fuzzy".
    """
    # Build set of aliases that map to this canonical field
    field_aliases: set[str] = {
        alias for alias, target in _ALIASES.items() if target == canonical_field
    }
    # Always include the canonical name itself as an exact alias
    field_aliases.add(canonical_field)

    best_fuzzy_score = 0.0
    best_fuzzy_idx: Optional[int] = None

    for idx, norm_header in enumerate(normalised_headers):
        if idx in assigned_indices or not norm_header:
            continue

        # Strategy 1: Exact match
        if norm_header == canonical_field:
            return idx, "exact"

        # Strategy 2: Alias match
        if norm_header in field_aliases:
            return idx, "alias"

        # Strategy 3: Fuzzy — compute token overlap score
        score = _fuzzy_score(norm_header, field_aliases)
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_idx = idx

    if best_fuzzy_score >= _FUZZY_MIN_SCORE and best_fuzzy_idx is not None:
        return best_fuzzy_idx, "fuzzy"

    return None, ""


def _fuzzy_score(header: str, aliases: set[str]) -> float:
    """
    Score a header against a set of aliases using token overlap.

    For each alias, compute the Jaccard similarity of word tokens between
    the header and the alias. Returns the highest score found.

    Jaccard similarity = |intersection| / |union|

    This handles variations like:
      "total sale amount" vs "total amount" → score = 2/3 = 0.67 ✓
      "qty sold" vs "qty" → score = 1/2 = 0.50 ✗ (below threshold)

    Args:
        header:  Normalised header string to score.
        aliases: Set of normalised alias strings for the target field.

    Returns:
        float: Highest similarity score found (0.0–1.0).
    """
    header_tokens = set(header.split())
    best = 0.0

    for alias in aliases:
        alias_tokens = set(alias.split())
        union = header_tokens | alias_tokens
        if not union:
            continue
        intersection = header_tokens & alias_tokens
        score = len(intersection) / len(union)
        if score > best:
            best = score

    return best


def _find_closest_header(canonical_field: str, raw_headers: list[str]) -> Optional[str]:
    """
    Find the raw header that is closest to a canonical field name.

    Used to build helpful error suggestions for businesses whose
    column names were not recognised.

    Args:
        canonical_field: The canonical field that was not found.
        raw_headers:     Original raw header list.

    Returns:
        str | None: The closest raw header, or None if headers is empty.
    """
    if not raw_headers:
        return None

    field_aliases: set[str] = {
        alias for alias, target in _ALIASES.items() if target == canonical_field
    }
    field_aliases.add(canonical_field)

    best_score = 0.0
    best_header: Optional[str] = None

    for raw in raw_headers:
        norm = _normalise_header(raw)
        score = _fuzzy_score(norm, field_aliases)
        if score > best_score:
            best_score = score
            best_header = raw

    return best_header if best_score > 0.2 else None


def _empty_mapping(headers: list[str]) -> ColumnMapping:
    """Return a ColumnMapping with all fields as None (empty header case)."""
    return ColumnMapping(
        date=None,
        revenue=None,
        quantity=None,
        product=None,
        category=None,
        customer=None,
        payment_method=None,
        notes=None,
        raw_headers=tuple(headers),
        missing_required=REQUIRED_FIELDS,
        match_log=(),
    )


# ==============================================================================
# extract_row_values — convenience function used by analytics_service.py
# ==============================================================================

def extract_row_values(
    row: list,
    mapping: "ColumnMapping",
) -> dict:
    """
    Extract named field values from a raw spreadsheet row using a ColumnMapping.

    Iterates over every mapped field in the ColumnMapping and reads the
    value at the mapped column index. Returns a dict keyed by field name.
    Fields with no mapping or out-of-range indexes are returned as None.

    Args:
        row:     Raw list of cell values from a Google Sheets row.
        mapping: ColumnMapping produced by ColumnMapperService.map_headers().

    Returns:
        dict[str, Any]: Field name → cell value (or None if unmapped).
    """
    result: dict = {}
    for field_name, col_idx in mapping.field_to_index.items():
        if col_idx is not None and col_idx < len(row):
            result[field_name] = row[col_idx]
        else:
            result[field_name] = None
    return result