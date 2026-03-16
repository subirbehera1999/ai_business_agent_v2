# ==============================================================================
# File: app/utils/time_utils.py
# Purpose: Timezone-aware date and time utilities used across schedulers,
#          report generators, alert detectors, and analytics services.
#
#          All timestamps in this system are stored in UTC.
#          All display formatting uses the business's local timezone.
#          All report period calculations are deterministic — the same
#          inputs always produce the same period boundaries.
#
#          Key responsibilities:
#            - UTC now / today helpers
#            - Timezone-aware conversion for business local time
#            - Report period boundary calculation (weekly/monthly/quarterly)
#            - Rolling window helpers (last N hours/days)
#            - Scheduling window checks (is this job due?)
#            - Human-readable date formatting for WhatsApp messages
# ==============================================================================

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config.constants import DATE_FORMAT, DATETIME_FORMAT, DISPLAY_DATE_FORMAT, ServiceName

logger = logging.getLogger(ServiceName.API)

# UTC timezone singleton — used throughout instead of timezone.utc
# for consistency and to allow ZoneInfo-based operations
UTC = timezone.utc

# Default fallback timezone when a business has no timezone configured
DEFAULT_TIMEZONE = "Asia/Kolkata"


# ==============================================================================
# UTC Helpers
# ==============================================================================

def utc_now() -> datetime:
    """
    Return the current UTC datetime with timezone info attached.

    Always use this instead of datetime.utcnow() — the latter returns
    a naive datetime that causes comparison errors with timezone-aware
    database timestamps.

    Returns:
        datetime: Current UTC datetime (timezone-aware).
    """
    return datetime.now(UTC)


def utc_today() -> date:
    """
    Return today's date in UTC.

    Returns:
        date: Current UTC calendar date.
    """
    return datetime.now(UTC).date()


def utc_timestamp() -> str:
    """
    Return the current UTC datetime as an ISO 8601 string.

    Format: "2024-10-15T08:30:00Z"

    Returns:
        str: ISO 8601 UTC timestamp string.
    """
    return datetime.now(UTC).strftime(DATETIME_FORMAT)


def as_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware and in UTC.

    If the datetime is naive (no tzinfo), it is assumed to be UTC and
    made aware. If it already has timezone info, it is converted to UTC.

    Args:
        dt: Any datetime object.

    Returns:
        datetime: Timezone-aware UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ==============================================================================
# Timezone Conversion
# ==============================================================================

def get_timezone(tz_name: str) -> ZoneInfo:
    """
    Safely load a ZoneInfo timezone by name.

    Falls back to the default timezone if the name is invalid.
    Logs a warning on fallback so misconfigured business timezones
    are visible in operational logs.

    Args:
        tz_name: IANA timezone string (e.g., "Asia/Kolkata", "America/New_York").

    Returns:
        ZoneInfo: The requested timezone, or the default on error.
    """
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning(
            "Invalid timezone name — falling back to default",
            extra={
                "service": ServiceName.API,
                "requested_timezone": tz_name,
                "fallback_timezone": DEFAULT_TIMEZONE,
            },
        )
        return ZoneInfo(DEFAULT_TIMEZONE)


def to_local_time(dt: datetime, tz_name: str) -> datetime:
    """
    Convert a UTC datetime to a business's local timezone.

    Used for formatting report delivery times and scheduling decisions
    that depend on the business's local calendar day.

    Args:
        dt:      UTC datetime (timezone-aware).
        tz_name: IANA timezone name of the target timezone.

    Returns:
        datetime: Datetime in the specified local timezone.

    Example:
        local = to_local_time(utc_now(), "Asia/Kolkata")
        # Returns IST datetime (UTC+5:30)
    """
    tz = get_timezone(tz_name)
    return as_utc(dt).astimezone(tz)


def to_utc(dt: datetime, tz_name: str) -> datetime:
    """
    Interpret a naive datetime as being in the given timezone and
    convert it to UTC.

    Used when accepting business-local dates from user input or
    configuration and storing them as UTC.

    Args:
        dt:      Naive datetime in the business's local timezone.
        tz_name: IANA timezone name to interpret the datetime in.

    Returns:
        datetime: Timezone-aware UTC datetime.

    Example:
        utc_dt = to_utc(datetime(2024, 10, 15, 9, 0), "Asia/Kolkata")
    """
    tz = get_timezone(tz_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(UTC)


def local_date_for_business(
    business_timezone: str,
    reference_utc: Optional[datetime] = None,
) -> date:
    """
    Return the current local calendar date for a business.

    A business in "Asia/Kolkata" crossing midnight locally should have
    its usage counters reset for the new local day — not UTC midnight.
    This function provides the correct local date for that determination.

    Args:
        business_timezone:  IANA timezone of the business.
        reference_utc:      UTC datetime to convert (defaults to now).

    Returns:
        date: The local calendar date for the business.
    """
    ref = reference_utc or utc_now()
    local_dt = to_local_time(as_utc(ref), business_timezone)
    return local_dt.date()


# ==============================================================================
# Rolling Window Helpers
# ==============================================================================

def hours_ago(n: int) -> datetime:
    """
    Return the UTC datetime exactly n hours ago.

    Used for spike detection windows (e.g., "reviews in last hour").

    Args:
        n: Number of hours to look back.

    Returns:
        datetime: UTC datetime n hours before now.
    """
    return utc_now() - timedelta(hours=n)


def days_ago(n: int) -> datetime:
    """
    Return the UTC datetime exactly n days ago (same time of day).

    Used for trend analysis windows (e.g., "rating over last 7 days").

    Args:
        n: Number of days to look back.

    Returns:
        datetime: UTC datetime n days before now.
    """
    return utc_now() - timedelta(days=n)


def days_ago_date(n: int) -> date:
    """
    Return the UTC calendar date exactly n days ago.

    Args:
        n: Number of days to look back.

    Returns:
        date: UTC date n days before today.
    """
    return utc_today() - timedelta(days=n)


def start_of_day_utc(d: Optional[date] = None) -> datetime:
    """
    Return midnight UTC for a given date (or today).

    Args:
        d: Calendar date (defaults to today UTC).

    Returns:
        datetime: Midnight UTC at the start of the given date.
    """
    target = d or utc_today()
    return datetime(target.year, target.month, target.day, 0, 0, 0, tzinfo=UTC)


def end_of_day_utc(d: Optional[date] = None) -> datetime:
    """
    Return 23:59:59.999999 UTC for a given date (or today).

    Args:
        d: Calendar date (defaults to today UTC).

    Returns:
        datetime: End of day UTC for the given date.
    """
    target = d or utc_today()
    return datetime(
        target.year, target.month, target.day,
        23, 59, 59, 999999, tzinfo=UTC,
    )


# ==============================================================================
# Report Period Calculators
# ==============================================================================

def get_weekly_period(
    reference: Optional[date] = None,
) -> tuple[date, date]:
    """
    Return the start and end dates of the most recently completed
    calendar week (Monday–Sunday).

    The "most recently completed week" is defined as the 7-day period
    that ended on the Sunday immediately before the reference date.
    This ensures weekly reports always cover a full, closed period.

    Args:
        reference: Reference date (defaults to today UTC).

    Returns:
        tuple[date, date]: (week_start, week_end) — Monday to Sunday.

    Example:
        start, end = get_weekly_period(date(2024, 10, 15))
        # Tuesday → previous Mon–Sun: (2024-10-07, 2024-10-13)
    """
    ref = reference or utc_today()
    # Find the most recently completed Sunday
    days_since_monday = ref.weekday()  # Monday = 0, Sunday = 6
    # End = last Sunday (day before the most recent Monday)
    last_sunday = ref - timedelta(days=days_since_monday + 1)
    last_monday = last_sunday - timedelta(days=6)
    return last_monday, last_sunday


def get_monthly_period(
    reference: Optional[date] = None,
) -> tuple[date, date]:
    """
    Return the start and end dates of the most recently completed
    calendar month.

    Args:
        reference: Reference date (defaults to today UTC).

    Returns:
        tuple[date, date]: (month_start, month_end) — first to last day.

    Example:
        start, end = get_monthly_period(date(2024, 10, 15))
        # → (2024-09-01, 2024-09-30)
    """
    ref = reference or utc_today()
    # First day of the current month
    first_of_current = ref.replace(day=1)
    # Last day of previous month = day before first of current
    last_of_prev = first_of_current - timedelta(days=1)
    # First day of previous month
    first_of_prev = last_of_prev.replace(day=1)
    return first_of_prev, last_of_prev


def get_quarterly_period(
    reference: Optional[date] = None,
) -> tuple[date, date]:
    """
    Return the start and end dates of the most recently completed
    calendar quarter.

    Quarters:
        Q1: January   1  – March    31
        Q2: April     1  – June     30
        Q3: July      1  – September 30
        Q4: October   1  – December 31

    Args:
        reference: Reference date (defaults to today UTC).

    Returns:
        tuple[date, date]: (quarter_start, quarter_end).

    Example:
        start, end = get_quarterly_period(date(2024, 10, 15))
        # Q4 reference → Q3: (2024-07-01, 2024-09-30)
    """
    ref = reference or utc_today()
    current_quarter = (ref.month - 1) // 3 + 1  # 1–4

    if current_quarter == 1:
        # We're in Q1 → previous completed quarter is Q4 of last year
        prev_q_start = date(ref.year - 1, 10, 1)
        prev_q_end = date(ref.year - 1, 12, 31)
    elif current_quarter == 2:
        prev_q_start = date(ref.year, 1, 1)
        prev_q_end = date(ref.year, 3, 31)
    elif current_quarter == 3:
        prev_q_start = date(ref.year, 4, 1)
        prev_q_end = date(ref.year, 6, 30)
    else:  # Q4
        prev_q_start = date(ref.year, 7, 1)
        prev_q_end = date(ref.year, 9, 30)

    return prev_q_start, prev_q_end


def get_current_week_period(
    reference: Optional[date] = None,
) -> tuple[date, date]:
    """
    Return the start and end dates of the current (in-progress) calendar week.

    Unlike get_weekly_period() which returns the last *completed* week,
    this returns the current week including today.

    Args:
        reference: Reference date (defaults to today UTC).

    Returns:
        tuple[date, date]: (week_start, week_end) — Monday to Sunday.
    """
    ref = reference or utc_today()
    days_since_monday = ref.weekday()
    week_start = ref - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def get_days_in_period(start: date, end: date) -> int:
    """
    Return the number of calendar days in an inclusive date range.

    Args:
        start: Start date (inclusive).
        end:   End date (inclusive).

    Returns:
        int: Number of days from start to end inclusive.
    """
    return (end - start).days + 1


# ==============================================================================
# Scheduling Window Helpers
# ==============================================================================

def is_within_hours_of_day(
    start_hour: int,
    end_hour: int,
    tz_name: str = DEFAULT_TIMEZONE,
) -> bool:
    """
    Check whether the current local time is within a scheduling window.

    Used by schedulers to avoid sending WhatsApp messages at inappropriate
    hours (e.g., 2am) even if the scheduler triggers at that time.

    Args:
        start_hour: Start of the allowed window (0–23, inclusive).
        end_hour:   End of the allowed window (0–23, inclusive).
        tz_name:    Timezone to evaluate the current hour in.

    Returns:
        bool: True if the current local hour is within [start_hour, end_hour].

    Example:
        is_within_hours_of_day(8, 21, "Asia/Kolkata")
        # Returns True if local IST time is between 8:00 and 21:59
    """
    local_now = to_local_time(utc_now(), tz_name)
    current_hour = local_now.hour
    return start_hour <= current_hour <= end_hour


def seconds_until(target_dt: datetime) -> float:
    """
    Return the number of seconds between now (UTC) and a future datetime.

    Returns 0 if the target is in the past.

    Args:
        target_dt: Target UTC datetime (must be timezone-aware).

    Returns:
        float: Seconds until the target, or 0 if already past.
    """
    delta = as_utc(target_dt) - utc_now()
    return max(0.0, delta.total_seconds())


def is_overdue(dt: datetime) -> bool:
    """
    Return True if the given UTC datetime is in the past.

    Used by expiry_checker.py to test whether a subscription's
    expires_at timestamp has passed.

    Args:
        dt: UTC datetime to check (timezone-aware).

    Returns:
        bool: True if dt is before now (UTC).
    """
    return as_utc(dt) < utc_now()


def is_within_days(dt: datetime, days: int) -> bool:
    """
    Return True if the given datetime falls within the next N days from now.

    Used by expiry_checker.py to find subscriptions expiring soon.

    Args:
        dt:   UTC datetime to check.
        days: Number of days ahead to test against.

    Returns:
        bool: True if dt is between now and now + days.
    """
    now = utc_now()
    future = now + timedelta(days=days)
    aware_dt = as_utc(dt)
    return now <= aware_dt <= future


# ==============================================================================
# Human-Readable Formatting
# ==============================================================================

def format_date(d: date) -> str:
    """
    Format a date as "YYYY-MM-DD".

    Args:
        d: Date to format.

    Returns:
        str: Date string in ISO format.
    """
    return d.strftime(DATE_FORMAT)


def format_datetime(dt: datetime) -> str:
    """
    Format a datetime as an ISO 8601 UTC string.

    Format: "2024-10-15T08:30:00Z"

    Args:
        dt: Datetime to format (timezone-aware recommended).

    Returns:
        str: ISO 8601 UTC datetime string.
    """
    return as_utc(dt).strftime(DATETIME_FORMAT)


def format_display_date(d: date) -> str:
    """
    Format a date for human-readable display in WhatsApp messages and reports.

    Format: "15 October 2024"

    Args:
        d: Date to format.

    Returns:
        str: Long-form display date string.
    """
    return d.strftime(DISPLAY_DATE_FORMAT)


def format_display_datetime(
    dt: datetime,
    tz_name: str = DEFAULT_TIMEZONE,
) -> str:
    """
    Format a UTC datetime for human display in the business's local timezone.

    Format: "15 October 2024, 02:00 PM IST"

    Args:
        dt:      UTC datetime to format.
        tz_name: Business's IANA timezone for local conversion.

    Returns:
        str: Human-readable local datetime string with timezone abbreviation.
    """
    local_dt = to_local_time(as_utc(dt), tz_name)
    return local_dt.strftime(f"{DISPLAY_DATE_FORMAT}, %I:%M %p %Z")


def format_period(start: date, end: date) -> str:
    """
    Format a date range as a human-readable period string.

    Format: "07 October 2024 – 13 October 2024"

    Args:
        start: Start date of the period.
        end:   End date of the period.

    Returns:
        str: Human-readable period string.
    """
    return f"{format_display_date(start)} – {format_display_date(end)}"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Examples:
        0.45   → "0.45s"
        65.3   → "1m 5s"
        3725.0 → "1h 2m 5s"

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Human-readable duration string.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


# ==============================================================================
# ISO Week
# ==============================================================================

def get_iso_week(d: Optional[date] = None) -> tuple[int, int]:
    """
    Return the ISO year and week number for a given date.

    ISO weeks start on Monday. Week 1 is the first week containing
    a Thursday. This is the same calculation used by idempotency_utils.py.

    Args:
        d: Date to get the ISO week for (defaults to today UTC).

    Returns:
        tuple[int, int]: (iso_year, iso_week_number).

    Example:
        year, week = get_iso_week(date(2024, 12, 31))
        # → (2025, 1)  — Dec 31 2024 belongs to ISO week 1 of 2025
    """
    target = d or utc_today()
    iso_year, iso_week, _ = target.isocalendar()
    return iso_year, iso_week


def iso_week_string(d: Optional[date] = None) -> str:
    """
    Return the ISO 8601 week identifier string for a given date.

    Format: "YYYY-Www" (e.g., "2024-W42").

    Args:
        d: Date (defaults to today UTC).

    Returns:
        str: ISO week identifier string.
    """
    iso_year, iso_week = get_iso_week(d)
    return f"{iso_year}-W{iso_week:02d}"


# ==============================================================================
# Missing exports — added to satisfy all import contracts across the system
# ==============================================================================

from datetime import timezone as _tz


def now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware). Alias for utc_now()."""
    return utc_now()


def today_local(tz_name: str = "UTC") -> date:
    """
    Return today's date in the given timezone (defaults to UTC).

    Args:
        tz_name: IANA timezone name (e.g. "Asia/Kolkata"). Default: "UTC".

    Returns:
        date: Today's date in the specified timezone.
    """
    tz = get_timezone(tz_name)
    return datetime.now(tz=tz).date()


def compute_subscription_end_date(
    start: datetime,
    billing_cycle_months: int,
) -> datetime:
    """
    Compute the subscription expiry datetime given a start date and duration.

    Adds billing_cycle_months months to start. Handles month-end edge cases
    (e.g. Jan 31 + 1 month → Feb 28/29).

    Args:
        start:                Subscription activation datetime (UTC).
        billing_cycle_months: Duration in months (1 or 12).

    Returns:
        datetime: Expiry datetime in UTC.
    """
    month = start.month - 1 + billing_cycle_months
    year  = start.year + month // 12
    month = month % 12 + 1
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    day = min(start.day, last_day)
    return start.replace(year=year, month=month, day=day)


def days_until(target: datetime) -> int:
    """
    Return the number of whole days until the target datetime from now (UTC).

    Returns negative values if target is in the past.

    Args:
        target: Future datetime (UTC-aware).

    Returns:
        int: Whole days until target.
    """
    delta = target - datetime.now(tz=_tz.utc)
    return delta.days


def parse_flexible_date(value: str) -> Optional[date]:
    """
    Attempt to parse a date string in multiple common formats.

    Tries: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY in that order.

    Args:
        value: Raw date string from an external source.

    Returns:
        date if parsed successfully, None otherwise.
    """
    from typing import Optional as _Opt
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def get_date_range_label(start: date, end: date) -> str:
    """
    Return a human-readable date range label.

    Example: "01 Jan 2024 – 31 Jan 2024"

    Args:
        start: Range start date.
        end:   Range end date.

    Returns:
        str: Formatted date range string.
    """
    fmt = "%d %b %Y"
    return f"{start.strftime(fmt)} – {end.strftime(fmt)}"


def get_week_bounds(reference: Optional[date] = None) -> tuple[date, date]:
    """
    Return (Monday, Sunday) of the ISO week containing reference date.

    Args:
        reference: Date within the week. Defaults to today UTC.

    Returns:
        tuple[date, date]: (week_start, week_end).
    """
    d = reference or utc_today()
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def get_month_bounds(reference: Optional[date] = None) -> tuple[date, date]:
    """
    Return (first_day, last_day) of the month containing reference date.

    Args:
        reference: Date within the month. Defaults to today UTC.

    Returns:
        tuple[date, date]: (month_start, month_end).
    """
    import calendar
    d = reference or utc_today()
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last


def get_week_date_range(reference: Optional[date] = None) -> tuple[date, date]:
    """Alias for get_week_bounds(). Returns (Monday, Sunday)."""
    return get_week_bounds(reference)


def get_month_date_range(reference: Optional[date] = None) -> tuple[date, date]:
    """Alias for get_month_bounds(). Returns (first_day, last_day) of month."""
    return get_month_bounds(reference)


def get_quarter_date_range(reference: Optional[date] = None) -> tuple[date, date]:
    """
    Return (first_day, last_day) of the calendar quarter containing reference.

    Quarters: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec.

    Args:
        reference: Date within the quarter. Defaults to today UTC.

    Returns:
        tuple[date, date]: (quarter_start, quarter_end).
    """
    import calendar
    d = reference or utc_today()
    q_start_month = ((d.month - 1) // 3) * 3 + 1
    q_end_month   = q_start_month + 2
    q_start = d.replace(month=q_start_month, day=1)
    last_day = calendar.monthrange(d.year, q_end_month)[1]
    q_end = d.replace(month=q_end_month, day=last_day)
    return q_start, q_end


def get_week_boundaries_in_month(reference: Optional[date] = None) -> list[tuple[date, date]]:
    """
    Return a list of (week_start, week_end) tuples for all ISO weeks in the month.

    Each tuple is clipped to the month boundaries so no dates fall outside
    the month. Used by monthly report jobs to break down data week-by-week.

    Args:
        reference: Date within the month. Defaults to today UTC.

    Returns:
        list[tuple[date, date]]: Week boundaries, clipped to the month.
    """
    import calendar
    d = reference or utc_today()
    month_start, month_end = get_month_bounds(d)

    weeks = []
    current = month_start
    while current <= month_end:
        week_start, week_end = get_week_bounds(current)
        clipped_start = max(week_start, month_start)
        clipped_end   = min(week_end, month_end)
        if (clipped_start, clipped_end) not in weeks:
            weeks.append((clipped_start, clipped_end))
        current = week_end + timedelta(days=1)
    return weeks


def iso_week_label(reference: Optional[date] = None) -> str:
    """
    Return the ISO week label string for the week containing reference.

    Format: "YYYY-Www"  e.g. "2024-W42"

    Args:
        reference: Date within the week. Defaults to today UTC.

    Returns:
        str: ISO week label string.
    """
    d = reference or utc_today()
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def month_label(reference: Optional[date] = None) -> str:
    """
    Return a human-readable month label.

    Format: "January 2024"

    Args:
        reference: Date within the month. Defaults to today UTC.

    Returns:
        str: Month label string.
    """
    d = reference or utc_today()
    return d.strftime("%B %Y")


def quarter_label(reference: Optional[date] = None) -> str:
    """
    Return a human-readable quarter label.

    Format: "Q1 2024"

    Args:
        reference: Date within the quarter. Defaults to today UTC.

    Returns:
        str: Quarter label string.
    """
    d = reference or utc_today()
    q = ((d.month - 1) // 3) + 1
    return f"Q{q} {d.year}"


def get_current_week_label() -> str:
    """Return the ISO week label for the current UTC week. e.g. '2024-W42'."""
    return iso_week_label()


def get_current_month_label() -> str:
    """Return the month label for the current UTC month. e.g. 'January 2024'."""
    return month_label()