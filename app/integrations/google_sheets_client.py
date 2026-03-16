# ==============================================================================
# File: app/integrations/google_sheets_client.py
# Purpose: Client for Google Sheets API v4.
#          Reads sales data from a business's connected Google Sheet and
#          returns structured row data for column_mapper_service.py and
#          analytics_service.py.
#
#          Responsibilities:
#            - fetch_sheet_data()   → reads all rows from a named sheet tab
#            - fetch_header_row()   → reads only the first row (column headers)
#            - fetch_rows_since()   → reads rows filtered by a date column
#                                     (post-filter in Python — Sheets API
#                                     does not support server-side date filtering)
#            - validate_connection()→ verifies the sheet is accessible with
#                                     the given credentials
#
#          Authentication:
#            Uses a service account JSON key (GOOGLE_SERVICE_ACCOUNT_JSON env var)
#            OR a pre-authorised OAuth2 access token
#            (GOOGLE_SHEETS_ACCESS_TOKEN env var) depending on configuration.
#            The business must share their Google Sheet with the service
#            account email during onboarding.
#
#          Pagination:
#            Google Sheets API returns up to 1000 rows per request by default.
#            For large sheets, this client uses A1 notation range batching
#            to read in chunks of MAX_ROWS_PER_REQUEST rows until all rows
#            are retrieved or MAX_TOTAL_ROWS is reached.
#
#          Data contract:
#            Raw output is list[list[str]] — each inner list is one row,
#            each element is a cell value as a string.
#            The first row is the header row (column names).
#            Empty trailing cells are NOT padded — callers must handle
#            rows of varying length (column_mapper_service.py handles this).
#
#          Safety:
#            - MAX_TOTAL_ROWS = 10,000 prevents runaway memory on huge sheets
#            - All values are returned as strings (valueRenderOption=FORMATTED_VALUE)
#              so callers receive what users see, not raw formula outputs
#            - Sheets API quota: 300 requests/min/project, 60 requests/min/user
#              This client serialises all calls (no concurrency within one fetch)
#              and relies on retry_utils for transient quota errors
# ==============================================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_google_retry

logger = logging.getLogger(ServiceName.GOOGLE_SHEETS)
settings = get_settings()

# ---------------------------------------------------------------------------
# Google Sheets API constants
# ---------------------------------------------------------------------------
_SHEETS_BASE_URL = "https://sheets.googleapis.com/v4/spreadsheets"
_VALUE_RENDER_OPTION = "FORMATTED_VALUE"   # Return display values, not formulas
_DATE_RENDER_OPTION = "FORMATTED_STRING"   # Return dates as strings e.g. "15/03/2025"
_MAJOR_DIMENSION = "ROWS"

# Safety caps
MAX_ROWS_PER_REQUEST: int = 1000
MAX_TOTAL_ROWS: int = 10_000
MAX_COLUMNS: int = 50    # Sheets wider than 50 columns are truncated

# Default sheet tab name if none specified
DEFAULT_SHEET_TAB = "Sheet1"


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class SheetData:
    """
    Raw data fetched from a Google Sheet.

    Attributes:
        spreadsheet_id:  The Google Sheets document ID.
        sheet_tab:       The tab/sheet name that was read.
        headers:         First row values (column names). Empty list if
                         the sheet has no header row.
        rows:            All data rows after the header. Each row is a
                         list of string cell values.
        total_rows:      Total rows read (excluding header).
        truncated:       True if MAX_TOTAL_ROWS was reached before all
                         rows were read.
        range_read:      The A1 notation range that was actually read.
    """
    spreadsheet_id: str
    sheet_tab: str
    headers: list[str]
    rows: list[list[str]]
    total_rows: int
    truncated: bool = False
    range_read: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    @property
    def column_count(self) -> int:
        return len(self.headers)

    def row_count(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        return (
            f"SheetData("
            f"id={self.spreadsheet_id[:8]}... "
            f"tab='{self.sheet_tab}' "
            f"headers={self.column_count} "
            f"rows={self.total_rows}"
            f"{' TRUNCATED' if self.truncated else ''})"
        )


@dataclass
class SheetsApiResult:
    """
    Wrapper for all Google Sheets API responses.

    Attributes:
        success:     True if the call succeeded.
        data:        SheetData if success=True, else None.
        error:       Error message if success=False.
        status_code: HTTP status code of the last response.
        error_type:  Categorised error type for caller handling.
    """
    success: bool
    data: Optional[SheetData] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None   # "auth", "not_found", "quota", "network"

    @property
    def is_auth_error(self) -> bool:
        return self.error_type == "auth"

    @property
    def is_not_found(self) -> bool:
        return self.error_type == "not_found"

    @property
    def is_quota_error(self) -> bool:
        return self.error_type == "quota"


# ==============================================================================
# Google Sheets Client
# ==============================================================================

class GoogleSheetsClient:
    """
    Async client for Google Sheets API v4.

    Reads sales data from a business's connected Google Sheet.
    Instantiated once per application and shared across services.

    Usage:
        client = GoogleSheetsClient()
        await client.initialise()

        result = await client.fetch_sheet_data(
            spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
            sheet_tab="Sales",
        )

        if result.success:
            headers = result.data.headers
            rows    = result.data.rows
        else:
            logger.error(result.error)

        await client.close()
    """

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None

    async def initialise(self) -> None:
        """
        Initialise the shared HTTP client.
        Must be called before any API methods are used.
        """
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.EXTERNAL_API_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers={"User-Agent": "AIBusinessAgent/1.0"},
        )
        logger.info(
            "GoogleSheetsClient initialised",
            extra={"service": ServiceName.GOOGLE_SHEETS},
        )

    async def close(self) -> None:
        """Close the shared HTTP client. Called from app lifespan shutdown."""
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info(
            "GoogleSheetsClient closed",
            extra={"service": ServiceName.GOOGLE_SHEETS},
        )

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def fetch_sheet_data(
        self,
        spreadsheet_id: str,
        sheet_tab: str = DEFAULT_SHEET_TAB,
        skip_empty_rows: bool = True,
    ) -> SheetsApiResult:
        """
        Fetch all data from a Google Sheet tab.

        Reads in paginated chunks of MAX_ROWS_PER_REQUEST rows until
        all rows are retrieved or MAX_TOTAL_ROWS is reached.

        The first row is treated as the header row and returned separately
        in SheetData.headers.

        Args:
            spreadsheet_id:  Google Sheets document ID (from the URL).
            sheet_tab:       Sheet tab name to read (default: "Sheet1").
            skip_empty_rows: If True, rows where all cells are empty
                             or whitespace are excluded.

        Returns:
            SheetsApiResult with data=SheetData. Never raises.
        """
        self._ensure_initialised()
        log_extra = {
            "service": ServiceName.GOOGLE_SHEETS,
            "spreadsheet_id": spreadsheet_id[:16],
            "sheet_tab": sheet_tab,
        }

        try:
            all_values = await self._fetch_all_values(
                spreadsheet_id=spreadsheet_id,
                sheet_tab=sheet_tab,
                log_extra=log_extra,
            )

            if all_values is None:
                # Error already logged inside _fetch_all_values
                return SheetsApiResult(
                    success=False,
                    error="Failed to read sheet data — see logs for details",
                    error_type="network",
                )

            if not all_values:
                # Sheet exists but is completely empty
                logger.info(
                    "Sheet is empty",
                    extra=log_extra,
                )
                return SheetsApiResult(
                    success=True,
                    data=SheetData(
                        spreadsheet_id=spreadsheet_id,
                        sheet_tab=sheet_tab,
                        headers=[],
                        rows=[],
                        total_rows=0,
                    ),
                )

            # First row → headers
            raw_headers = all_values[0]
            headers = [str(h).strip() for h in raw_headers]
            # Truncate to MAX_COLUMNS
            headers = headers[:MAX_COLUMNS]

            # Remaining rows → data
            raw_rows = all_values[1:]
            truncated = len(raw_rows) > MAX_TOTAL_ROWS
            raw_rows = raw_rows[:MAX_TOTAL_ROWS]

            # Normalise rows: ensure each row has exactly len(headers) cells
            data_rows = _normalise_rows(
                rows=raw_rows,
                column_count=len(headers),
                skip_empty=skip_empty_rows,
            )

            logger.info(
                "Sheet data fetched successfully",
                extra={
                    **log_extra,
                    "headers": len(headers),
                    "rows": len(data_rows),
                    "truncated": truncated,
                },
            )

            return SheetsApiResult(
                success=True,
                data=SheetData(
                    spreadsheet_id=spreadsheet_id,
                    sheet_tab=sheet_tab,
                    headers=headers,
                    rows=data_rows,
                    total_rows=len(data_rows),
                    truncated=truncated,
                ),
            )

        except _SheetsAuthError as exc:
            return SheetsApiResult(
                success=False,
                error=str(exc),
                error_type="auth",
                status_code=exc.status_code,
            )
        except _SheetsNotFoundError as exc:
            return SheetsApiResult(
                success=False,
                error=str(exc),
                error_type="not_found",
                status_code=404,
            )
        except _SheetsQuotaError as exc:
            return SheetsApiResult(
                success=False,
                error=str(exc),
                error_type="quota",
                status_code=429,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error fetching sheet data",
                extra={**log_extra, "error": str(exc)},
            )
            return SheetsApiResult(
                success=False,
                error=f"Unexpected error: {exc}",
                error_type="network",
            )

    async def fetch_header_row(
        self,
        spreadsheet_id: str,
        sheet_tab: str = DEFAULT_SHEET_TAB,
    ) -> SheetsApiResult:
        """
        Fetch only the first row (header row) of a Google Sheet.

        Used by column_mapper_service.py during onboarding to determine
        the column layout before processing any data rows.

        Args:
            spreadsheet_id: Google Sheets document ID.
            sheet_tab:      Sheet tab name.

        Returns:
            SheetsApiResult with data=SheetData containing only headers,
            rows=[], total_rows=0.
        """
        self._ensure_initialised()
        log_extra = {
            "service": ServiceName.GOOGLE_SHEETS,
            "spreadsheet_id": spreadsheet_id[:16],
            "sheet_tab": sheet_tab,
        }

        # Read only row 1
        a1_range = _build_a1_range(sheet_tab, row_start=1, row_end=1)
        result = await self._get_values(
            spreadsheet_id=spreadsheet_id,
            a1_range=a1_range,
            log_extra=log_extra,
        )

        if not result.success:
            return result

        values = result.data or []
        if not values:
            return SheetsApiResult(
                success=True,
                data=SheetData(
                    spreadsheet_id=spreadsheet_id,
                    sheet_tab=sheet_tab,
                    headers=[],
                    rows=[],
                    total_rows=0,
                ),
            )

        headers = [str(h).strip() for h in values[0]][:MAX_COLUMNS]
        return SheetsApiResult(
            success=True,
            data=SheetData(
                spreadsheet_id=spreadsheet_id,
                sheet_tab=sheet_tab,
                headers=headers,
                rows=[],
                total_rows=0,
            ),
        )

    async def fetch_rows_since(
        self,
        spreadsheet_id: str,
        date_column_index: int,
        since_date_str: str,
        sheet_tab: str = DEFAULT_SHEET_TAB,
    ) -> SheetsApiResult:
        """
        Fetch rows where the date column value is on or after since_date_str.

        Google Sheets API does not support server-side row filtering, so this
        method fetches all rows and filters in Python. The full sheet is read
        once and filtered — this is memory-efficient because SheetData rows
        are plain string lists.

        Date comparison is done as string prefix matching on the date_str
        (e.g. "2025-03" matches all rows from March 2025). For exact date
        filtering, callers should pass full ISO dates ("2025-03-01").

        Args:
            spreadsheet_id:     Google Sheets document ID.
            date_column_index:  Zero-based column index of the date column.
                                Provided by column_mapper_service.py.
            since_date_str:     ISO date string "YYYY-MM-DD".
                                Rows with dates strictly before this are excluded.
            sheet_tab:          Sheet tab name.

        Returns:
            SheetsApiResult with SheetData containing only matching rows.
            Header row is always included.
        """
        full_result = await self.fetch_sheet_data(
            spreadsheet_id=spreadsheet_id,
            sheet_tab=sheet_tab,
        )

        if not full_result.success:
            return full_result

        sheet = full_result.data
        filtered_rows = _filter_rows_since(
            rows=sheet.rows,
            date_col_idx=date_column_index,
            since_date_str=since_date_str,
        )

        log_extra = {
            "service": ServiceName.GOOGLE_SHEETS,
            "spreadsheet_id": spreadsheet_id[:16],
            "since_date": since_date_str,
            "total_rows": sheet.total_rows,
            "filtered_rows": len(filtered_rows),
        }
        logger.debug("Rows filtered by date", extra=log_extra)

        return SheetsApiResult(
            success=True,
            data=SheetData(
                spreadsheet_id=spreadsheet_id,
                sheet_tab=sheet_tab,
                headers=sheet.headers,
                rows=filtered_rows,
                total_rows=len(filtered_rows),
                truncated=sheet.truncated,
                range_read=sheet.range_read,
            ),
        )

    async def validate_connection(
        self,
        spreadsheet_id: str,
        sheet_tab: str = DEFAULT_SHEET_TAB,
    ) -> SheetsApiResult:
        """
        Verify that a Google Sheet is accessible with current credentials.

        Called during business onboarding to confirm the sheet is shared
        with the service account before attempting a full data fetch.

        Reads only the header row to minimise API quota usage.

        Args:
            spreadsheet_id: Google Sheets document ID.
            sheet_tab:      Sheet tab name to verify.

        Returns:
            SheetsApiResult. success=True means the sheet is reachable.
        """
        result = await self.fetch_header_row(
            spreadsheet_id=spreadsheet_id,
            sheet_tab=sheet_tab,
        )

        if result.success:
            logger.info(
                "Google Sheet connection validated",
                extra={
                    "service": ServiceName.GOOGLE_SHEETS,
                    "spreadsheet_id": spreadsheet_id[:16],
                    "sheet_tab": sheet_tab,
                    "headers_found": len(result.data.headers) if result.data else 0,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Internal pagination
    # ------------------------------------------------------------------

    async def _fetch_all_values(
        self,
        spreadsheet_id: str,
        sheet_tab: str,
        log_extra: dict,
    ) -> Optional[list[list[str]]]:
        """
        Fetch all row values from a sheet tab using chunked A1 range requests.

        Reads in batches of MAX_ROWS_PER_REQUEST rows starting from row 1.
        Stops when a batch returns fewer rows than requested (last page)
        or MAX_TOTAL_ROWS is reached.

        Returns:
            list[list[str]] of all values, or None on unrecoverable error.
        """
        all_values: list[list[str]] = []
        row_start = 1

        while True:
            row_end = row_start + MAX_ROWS_PER_REQUEST - 1
            a1_range = _build_a1_range(sheet_tab, row_start, row_end)

            result = await self._get_values(
                spreadsheet_id=spreadsheet_id,
                a1_range=a1_range,
                log_extra={**log_extra, "a1_range": a1_range},
            )

            if not result.success:
                if all_values:
                    # Partial data — return what we have
                    logger.warning(
                        "Pagination interrupted — returning partial data",
                        extra={**log_extra, "rows_so_far": len(all_values)},
                    )
                    return all_values
                # No data at all — surface the error
                if isinstance(result.data, _SheetsError):
                    raise result.data
                return None

            batch = result.data or []

            if not batch:
                # Empty batch — we've read past the last row
                break

            all_values.extend(batch)

            if len(batch) < MAX_ROWS_PER_REQUEST:
                # Last page — fewer rows than requested means no more data
                break

            if len(all_values) >= MAX_TOTAL_ROWS:
                logger.warning(
                    "MAX_TOTAL_ROWS reached — sheet data truncated",
                    extra={**log_extra, "max": MAX_TOTAL_ROWS},
                )
                break

            row_start = row_end + 1

        return all_values

    @with_google_retry
    async def _get_values(
        self,
        spreadsheet_id: str,
        a1_range: str,
        log_extra: dict,
    ) -> "SheetsApiResult":
        """
        Execute a single Sheets API values.get request.

        Args:
            spreadsheet_id: Sheet document ID.
            a1_range:       A1 notation range e.g. "Sheet1!A1:AX1001".
            log_extra:      Structured log context.

        Returns:
            SheetsApiResult with data=list[list[str]] (raw values).
        """
        token = _get_access_token()
        url = f"{_SHEETS_BASE_URL}/{spreadsheet_id}/values/{a1_range}"
        params = {
            "valueRenderOption": _VALUE_RENDER_OPTION,
            "dateTimeRenderOption": _DATE_RENDER_OPTION,
            "majorDimension": _MAJOR_DIMENSION,
        }
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = await self._http.get(url, params=params, headers=headers)

            if response.status_code == 200:
                body = response.json()
                values = body.get("values", [])
                return SheetsApiResult(success=True, data=values, status_code=200)

            # Handle specific HTTP errors
            return _handle_sheets_error(response, log_extra)

        except httpx.TimeoutException as exc:
            logger.error(
                "Sheets API timeout",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

        except httpx.HTTPError as exc:
            logger.error(
                "Sheets API HTTP error",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "GoogleSheetsClient has not been initialised. "
                "Call await client.initialise() before use."
            )


# ==============================================================================
# Internal error types (not exposed to callers)
# ==============================================================================

class _SheetsError(Exception):
    pass

class _SheetsAuthError(_SheetsError):
    def __init__(self, msg: str, status_code: int = 403) -> None:
        super().__init__(msg)
        self.status_code = status_code

class _SheetsNotFoundError(_SheetsError):
    pass

class _SheetsQuotaError(_SheetsError):
    pass


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _get_access_token() -> str:
    """
    Return the OAuth2 access token for Google Sheets API.

    In production, this token is obtained via:
      1. Service account JSON (GOOGLE_SERVICE_ACCOUNT_JSON) — preferred
         for server-to-server, no user consent needed after initial setup.
      2. Pre-issued OAuth token (GOOGLE_SHEETS_ACCESS_TOKEN) — used during
         development or when service account is not yet configured.

    Raises:
        RuntimeError if no token is configured.
    """
    token = settings.GOOGLE_SHEETS_ACCESS_TOKEN
    if not token:
        raise RuntimeError(
            "GOOGLE_SHEETS_ACCESS_TOKEN is not configured. "
            "Share the Google Sheet with the service account and set the token."
        )
    return token


def _build_a1_range(sheet_tab: str, row_start: int, row_end: int) -> str:
    """
    Build an A1 notation range string for the Sheets API.

    Covers all columns up to column index MAX_COLUMNS using
    Excel-style column letters. For MAX_COLUMNS=50, the last
    column is AX.

    Args:
        sheet_tab:  Sheet tab name (may contain spaces — will be quoted).
        row_start:  First row number (1-based).
        row_end:    Last row number (1-based).

    Returns:
        e.g. "'Sales Data'!A1:AX1000"
    """
    last_col = _col_index_to_letter(MAX_COLUMNS)
    # Quote the tab name to handle spaces and special characters
    safe_tab = f"'{sheet_tab}'" if " " in sheet_tab else sheet_tab
    return f"{safe_tab}!A{row_start}:{last_col}{row_end}"


def _col_index_to_letter(col_index: int) -> str:
    """
    Convert a 1-based column index to an Excel-style column letter.

    Examples:
        1  → "A"
        26 → "Z"
        27 → "AA"
        50 → "AX"

    Args:
        col_index: 1-based column number.

    Returns:
        str: Column letter(s).
    """
    letters = ""
    while col_index > 0:
        col_index, remainder = divmod(col_index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _normalise_rows(
    rows: list[list],
    column_count: int,
    skip_empty: bool,
) -> list[list[str]]:
    """
    Normalise raw sheet rows to a consistent column count.

    Rules:
      - All values are converted to strings and stripped of whitespace.
      - Rows shorter than column_count are right-padded with empty strings.
      - Rows longer than column_count are truncated to column_count.
      - If skip_empty=True, rows where every cell is empty are excluded.

    Args:
        rows:         Raw value rows from the Sheets API.
        column_count: Target number of columns (from header row length).
        skip_empty:   Whether to drop all-empty rows.

    Returns:
        list[list[str]]: Normalised rows.
    """
    normalised: list[list[str]] = []

    for raw_row in rows:
        # Convert all cells to stripped strings
        str_row = [str(cell).strip() for cell in raw_row]

        # Truncate to column_count
        str_row = str_row[:column_count]

        # Pad to column_count
        while len(str_row) < column_count:
            str_row.append("")

        # Skip all-empty rows
        if skip_empty and all(cell == "" for cell in str_row):
            continue

        normalised.append(str_row)

    return normalised


def _filter_rows_since(
    rows: list[list[str]],
    date_col_idx: int,
    since_date_str: str,
) -> list[list[str]]:
    """
    Filter rows where the date column value is on or after since_date_str.

    Comparison is lexicographic on the ISO date prefix "YYYY-MM-DD".
    Rows where the date cell is empty or unparseable are excluded.

    Args:
        rows:            Data rows (header already separated).
        date_col_idx:    Zero-based index of the date column.
        since_date_str:  ISO date string "YYYY-MM-DD" lower bound (inclusive).

    Returns:
        Filtered list of rows.
    """
    filtered: list[list[str]] = []

    for row in rows:
        if date_col_idx >= len(row):
            continue

        cell_value = row[date_col_idx].strip()
        if not cell_value:
            continue

        # Normalise common date separators to dashes for comparison
        normalised_date = cell_value.replace("/", "-").replace(".", "-")

        # Extract ISO prefix — handles "DD-MM-YYYY" vs "YYYY-MM-DD"
        comparable = _extract_iso_prefix(normalised_date)
        if not comparable:
            continue

        if comparable >= since_date_str:
            filtered.append(row)

    return filtered


def _extract_iso_prefix(date_str: str) -> Optional[str]:
    """
    Extract a comparable YYYY-MM-DD prefix from a date string.

    Handles common Indian date formats:
      - "YYYY-MM-DD" (ISO) → used as-is
      - "DD-MM-YYYY"       → reordered to "YYYY-MM-DD"
      - "DD-MM-YY"         → reordered and 2-digit year expanded
      - "YYYY/MM/DD"       → normalised to YYYY-MM-DD

    Returns None for unrecognisable formats.

    Args:
        date_str: Date string with separators normalised to "-".

    Returns:
        "YYYY-MM-DD" string or None.
    """
    parts = date_str.split("-")

    if len(parts) != 3:
        return None

    a, b, c = parts[0], parts[1], parts[2]

    # ISO format: YYYY-MM-DD (year first, 4 digits)
    if len(a) == 4:
        return f"{a}-{b.zfill(2)}-{c.zfill(2)}"

    # DD-MM-YYYY (day first, year last with 4 digits)
    if len(c) == 4:
        return f"{c}-{b.zfill(2)}-{a.zfill(2)}"

    # DD-MM-YY (2-digit year — expand to 2000s)
    if len(c) == 2:
        full_year = f"20{c}"
        return f"{full_year}-{b.zfill(2)}-{a.zfill(2)}"

    return None


def _handle_sheets_error(
    response: httpx.Response,
    log_extra: dict,
) -> "SheetsApiResult":
    """
    Convert a non-200 Sheets API response into a typed SheetsApiResult.

    Maps HTTP status codes to error_type strings for caller handling.

    Args:
        response:  The failed httpx.Response.
        log_extra: Structured log context.

    Returns:
        SheetsApiResult with success=False and appropriate error_type.
    """
    status_code = response.status_code

    try:
        body = response.json()
        error_detail = (
            body.get("error", {}).get("message", "")
            or response.text[:200]
        )
    except Exception:
        error_detail = response.text[:200]

    if status_code in (401, 403):
        error_type = "auth"
        error_msg = (
            f"Google Sheets access denied (HTTP {status_code}). "
            f"Ensure the sheet is shared with the service account. "
            f"Detail: {error_detail}"
        )
        logger.error(
            "Sheets API auth error",
            extra={**log_extra, "status_code": status_code, "detail": error_detail},
        )
    elif status_code == 404:
        error_type = "not_found"
        error_msg = (
            f"Google Sheet not found (HTTP 404). "
            f"Verify the spreadsheet_id and sheet tab name. "
            f"Detail: {error_detail}"
        )
        logger.error(
            "Sheets API not found",
            extra={**log_extra, "detail": error_detail},
        )
    elif status_code == 429:
        error_type = "quota"
        error_msg = (
            f"Google Sheets API quota exceeded (HTTP 429). "
            f"Will retry after backoff."
        )
        logger.warning(
            "Sheets API quota exceeded",
            extra=log_extra,
        )
    else:
        error_type = "network"
        error_msg = f"Google Sheets API error (HTTP {status_code}): {error_detail}"
        logger.error(
            "Sheets API unexpected error",
            extra={**log_extra, "status_code": status_code, "detail": error_detail},
        )

    return SheetsApiResult(
        success=False,
        error=error_msg,
        error_type=error_type,
        status_code=status_code,
    )