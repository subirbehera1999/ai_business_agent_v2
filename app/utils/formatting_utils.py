# ==============================================================================
# File: app/utils/formatting_utils.py
# Purpose: Message and data formatting utilities used across notification
#          services, report generators, and alert dispatchers.
#
#          All WhatsApp messages are plain text with minimal punctuation
#          symbols. WhatsApp Cloud API supports basic formatting:
#            *bold*       → bold text
#            _italic_     → italic text
#            ~strikethrough~
#            ```code```
#
#          Responsibilities:
#            - Review text truncation (800-char cap from review_model.py)
#            - Star rating display (★★★☆☆)
#            - Sentiment label formatting
#            - WhatsApp section block builder
#            - Alert message formatter
#            - Report summary formatter (weekly / monthly / quarterly)
#            - Business greeting builder
#            - Number and percentage formatters
#            - Currency formatter (INR default — target market is India)
# ==============================================================================

import re
import textwrap
from typing import Optional

from app.config.constants import (
    REVIEW_TEXT_MAX_LENGTH,
    AlertSeverity,
    AlertType,
    ReviewSentiment,
    ServiceName,
)

# Maximum characters for a single WhatsApp message before it must be split
WHATSAPP_MESSAGE_MAX_CHARS = 4096

# Filled and empty star characters
STAR_FILLED = "★"
STAR_EMPTY = "☆"
TOTAL_STARS = 5

# Sentiment display labels
SENTIMENT_LABELS: dict[str, str] = {
    ReviewSentiment.POSITIVE: "😊 Positive",
    ReviewSentiment.NEGATIVE: "😞 Negative",
    ReviewSentiment.NEUTRAL: "😐 Neutral",
}

# Alert severity display prefix
SEVERITY_PREFIX: dict[str, str] = {
    AlertSeverity.CRITICAL: "🚨 URGENT",
    AlertSeverity.HIGH:     "⚠️  Alert",
    AlertSeverity.MEDIUM:   "📢 Notice",
    AlertSeverity.LOW:      "💡 Info",
}

# Alert type display names
ALERT_TYPE_LABELS: dict[str, str] = {
    AlertType.NEGATIVE_REVIEW:        "Negative Review Received",
    AlertType.POSITIVE_REVIEW:        "Positive Review Received",
    AlertType.REVIEW_SPIKE:           "Review Spike Detected",
    AlertType.RATING_DROP:            "Rating Drop Detected",
    AlertType.COMPETITOR_RATING:      "Competitor Rating Change",
    AlertType.COMPETITOR_REVIEW_SPIKE:"Competitor Review Spike",
    AlertType.SALES_TREND:            "Sales Trend Alert",
    AlertType.OPPORTUNITY:            "Business Opportunity",
    AlertType.SUBSCRIPTION_EXPIRY:    "Subscription Expiring Soon",
    AlertType.USAGE_LIMIT:            "Usage Limit Reached",
    AlertType.SYSTEM_HEALTH:          "System Health Issue",
}


# ==============================================================================
# Text Truncation
# ==============================================================================

def truncate_review_text(text: str) -> str:
    """
    Truncate review text to the system maximum (800 characters).

    Truncation preserves whole words where possible and appends an
    ellipsis to indicate truncation. Matches the review_text_truncated
    column behaviour defined in review_model.py.

    Args:
        text: Raw review text from Google Reviews API.

    Returns:
        str: Truncated text, at most REVIEW_TEXT_MAX_LENGTH characters.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= REVIEW_TEXT_MAX_LENGTH:
        return text
    # Truncate at last word boundary before the limit
    truncated = text[: REVIEW_TEXT_MAX_LENGTH - 3]
    last_space = truncated.rfind(" ")
    if last_space > REVIEW_TEXT_MAX_LENGTH // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


def truncate_for_whatsapp(text: str, max_chars: int = 200) -> str:
    """
    Shorten a text snippet for embedding inside a WhatsApp message.

    Used when a review excerpt is included in an alert — we do not
    want to repeat the full review text inside a notification.

    Args:
        text:      Text to shorten.
        max_chars: Maximum characters (default 200).

    Returns:
        str: Shortened text with ellipsis if truncated.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def clean_text(text: str) -> str:
    """
    Remove excess whitespace and normalize line endings.

    Args:
        text: Raw input text.

    Returns:
        str: Cleaned text with single spaces and Unix line endings.
    """
    if not text:
        return ""
    text = text.strip()
    # Normalize multiple spaces to single space (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ==============================================================================
# Rating Display
# ==============================================================================

def format_star_rating(rating: int) -> str:
    """
    Convert a numeric rating (1–5) to a star display string.

    Args:
        rating: Integer rating between 1 and 5.

    Returns:
        str: Star string e.g. "★★★☆☆" for rating=3.

    Example:
        format_star_rating(4) → "★★★★☆"
        format_star_rating(1) → "★☆☆☆☆"
    """
    clamped = max(1, min(rating, TOTAL_STARS))
    return STAR_FILLED * clamped + STAR_EMPTY * (TOTAL_STARS - clamped)


def format_rating_change(old_rating: float, new_rating: float) -> str:
    """
    Format a rating change as a delta string with direction indicator.

    Args:
        old_rating: Previous rating (float).
        new_rating: Current rating (float).

    Returns:
        str: Formatted change string e.g. "4.2 → 3.9 (▼ 0.3)"

    Example:
        format_rating_change(4.2, 3.9) → "4.2 → 3.9 (▼ 0.3)"
        format_rating_change(3.8, 4.1) → "3.8 → 4.1 (▲ 0.3)"
    """
    delta = new_rating - old_rating
    direction = "▲" if delta > 0 else "▼" if delta < 0 else "–"
    return f"{old_rating:.1f} → {new_rating:.1f} ({direction} {abs(delta):.1f})"


def format_sentiment(sentiment: str) -> str:
    """
    Return the display label for a sentiment value.

    Args:
        sentiment: ReviewSentiment constant (positive / negative / neutral).

    Returns:
        str: Emoji-prefixed sentiment label, or the raw value if unknown.
    """
    return SENTIMENT_LABELS.get(sentiment, sentiment)


# ==============================================================================
# Number and Currency Formatters
# ==============================================================================

def format_number(value: int | float, decimal_places: int = 0) -> str:
    """
    Format a number with thousand separators.

    Args:
        value:          Number to format.
        decimal_places: Decimal places to show (default 0 for integers).

    Returns:
        str: Formatted number string e.g. "1,234" or "1,234.50".
    """
    if decimal_places == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format a ratio (0.0–1.0) as a percentage string.

    Args:
        value:          Ratio between 0.0 and 1.0.
        decimal_places: Decimal places (default 1).

    Returns:
        str: Percentage string e.g. "87.3%".
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_currency_inr(paise: int) -> str:
    """
    Format an amount in paise as a human-readable INR string.

    Razorpay stores and returns all amounts in paise (1 INR = 100 paise).
    This converts to rupees for display.

    Args:
        paise: Amount in paise (smallest INR unit).

    Returns:
        str: Formatted INR string e.g. "₹1,299.00".
    """
    rupees = paise / 100
    return f"₹{rupees:,.2f}"


# ==============================================================================
# WhatsApp Message Builders
# ==============================================================================

def whatsapp_bold(text: str) -> str:
    """Wrap text in WhatsApp bold markers."""
    return f"*{text}*"


def whatsapp_italic(text: str) -> str:
    """Wrap text in WhatsApp italic markers."""
    return f"_{text}_"


def whatsapp_divider() -> str:
    """Return a visual divider line for WhatsApp messages."""
    return "─" * 30


def build_whatsapp_section(
    title: str,
    lines: list[str],
    include_divider: bool = True,
) -> str:
    """
    Build a formatted WhatsApp message section block.

    Structure:
        ─────────────────────────────
        *SECTION TITLE*
        Line 1
        Line 2

    Args:
        title:           Section heading (will be bolded).
        lines:           Content lines for the section body.
        include_divider: Whether to prepend a divider line.

    Returns:
        str: Formatted section block ready for WhatsApp delivery.
    """
    parts: list[str] = []
    if include_divider:
        parts.append(whatsapp_divider())
    parts.append(whatsapp_bold(title.upper()))
    parts.extend(line for line in lines if line is not None)
    return "\n".join(parts)


def build_business_greeting(business_name: str) -> str:
    """
    Build the opening greeting line for any WhatsApp message.

    Args:
        business_name: Name of the business receiving the message.

    Returns:
        str: Greeting line e.g. "Hello *Raj Restaurant* 👋"
    """
    return f"Hello {whatsapp_bold(business_name)} 👋"


def split_long_message(message: str, max_chars: int = WHATSAPP_MESSAGE_MAX_CHARS) -> list[str]:
    """
    Split a long WhatsApp message into deliverable chunks.

    Splits at paragraph boundaries (double newlines) wherever possible
    to avoid cutting mid-sentence. Falls back to hard wrapping if a
    single paragraph exceeds max_chars.

    Args:
        message:   Full message text.
        max_chars: Maximum characters per chunk (default: 4096).

    Returns:
        list[str]: List of message chunks, each within max_chars.
    """
    if len(message) <= max_chars:
        return [message]

    paragraphs = message.split("\n\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # If a single paragraph is too long, hard-wrap it
            if len(paragraph) > max_chars:
                wrapped = textwrap.wrap(paragraph, max_chars)
                chunks.extend(wrapped[:-1])
                current = wrapped[-1] if wrapped else ""
            else:
                current = paragraph

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ==============================================================================
# Alert Message Formatter
# ==============================================================================

def format_alert_message(
    business_name: str,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    extra_lines: Optional[list[str]] = None,
) -> str:
    """
    Build a complete WhatsApp alert message for a business event.

    Structure:
        Hello *Business Name* 👋

        ⚠️  Alert · Negative Review Received

        *Review Drop Detected*
        Your rating has dropped from 4.2 to 3.9 in the past 24 hours.

        Action: Check your recent reviews and respond promptly.

        ─────────────────────────────
        Powered by AI Business Agent

    Args:
        business_name: Name of the business.
        alert_type:    AlertType constant for label lookup.
        severity:      AlertSeverity constant for prefix lookup.
        title:         Alert headline (bolded).
        message:       Alert body paragraph.
        extra_lines:   Optional additional lines appended after the body.

    Returns:
        str: Complete formatted WhatsApp alert message.
    """
    severity_label = SEVERITY_PREFIX.get(severity, "📢 Notice")
    type_label = ALERT_TYPE_LABELS.get(alert_type, alert_type.replace("_", " ").title())

    lines: list[str] = [
        build_business_greeting(business_name),
        "",
        f"{severity_label} · {type_label}",
        "",
        whatsapp_bold(title),
        message,
    ]

    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)

    lines.extend([
        "",
        whatsapp_divider(),
        whatsapp_italic("Powered by AI Business Agent"),
    ])

    return "\n".join(lines)


# ==============================================================================
# Review Alert Formatter
# ==============================================================================

def format_review_alert(
    business_name: str,
    reviewer_name: str,
    rating: int,
    review_excerpt: str,
    sentiment: str,
    ai_reply_generated: bool = False,
) -> str:
    """
    Build a WhatsApp notification for a new review.

    Args:
        business_name:        Business receiving the review.
        reviewer_name:        Name of the reviewer.
        rating:               Star rating (1–5).
        review_excerpt:       Truncated review text for the notification.
        sentiment:            ReviewSentiment constant.
        ai_reply_generated:   Whether an AI reply was auto-generated.

    Returns:
        str: Formatted review alert message.
    """
    sentiment_label = format_sentiment(sentiment)
    star_display = format_star_rating(rating)
    excerpt = truncate_for_whatsapp(review_excerpt, max_chars=180)

    reply_line = (
        "✅ AI reply has been posted automatically."
        if ai_reply_generated
        else "ℹ️  Review reply is pending your attention."
    )

    lines: list[str] = [
        build_business_greeting(business_name),
        "",
        "📝 *New Google Review Received*",
        "",
        f"*Reviewer:* {reviewer_name}",
        f"*Rating:* {star_display} ({rating}/5)",
        f"*Sentiment:* {sentiment_label}",
        "",
        f'_{excerpt}_' if excerpt else "",
        "",
        reply_line,
        "",
        whatsapp_divider(),
        whatsapp_italic("Powered by AI Business Agent"),
    ]

    return "\n".join(line for line in lines if line is not None)


# ==============================================================================
# Report Summary Formatter
# ==============================================================================

def format_report_header(
    business_name: str,
    report_type: str,
    period_label: str,
) -> str:
    """
    Build the header block for a weekly/monthly/quarterly report.

    Args:
        business_name: Business the report is for.
        report_type:   Human-readable report type e.g. "Weekly Report".
        period_label:  Period string e.g. "07 Oct 2024 – 13 Oct 2024".

    Returns:
        str: Header block (not a complete message — combine with sections).
    """
    return "\n".join([
        build_business_greeting(business_name),
        "",
        whatsapp_bold(f"📊 {report_type.upper()}"),
        f"Period: {period_label}",
        whatsapp_divider(),
    ])


def format_review_summary_section(
    total_reviews: int,
    positive: int,
    negative: int,
    neutral: int,
    average_rating: float,
) -> str:
    """
    Build the review summary section for a report message.

    Args:
        total_reviews:  Total reviews in the period.
        positive:       Count of positive reviews.
        negative:       Count of negative reviews.
        neutral:        Count of neutral reviews.
        average_rating: Average star rating (float).

    Returns:
        str: Formatted review summary section block.
    """
    lines = [
        f"Total Reviews:    {format_number(total_reviews)}",
        f"😊 Positive:      {format_number(positive)}",
        f"😞 Negative:      {format_number(negative)}",
        f"😐 Neutral:       {format_number(neutral)}",
        f"⭐ Avg Rating:    {average_rating:.1f} / 5.0",
    ]
    return build_whatsapp_section("Review Summary", lines)


def format_competitor_summary_section(
    competitor_name: str,
    their_rating: float,
    your_rating: float,
    their_review_count: int,
) -> str:
    """
    Build a competitor comparison section for a report.

    Args:
        competitor_name:    Name of the competitor business.
        their_rating:       Competitor's current average rating.
        your_rating:        Our business's current average rating.
        their_review_count: Competitor's review count in the period.

    Returns:
        str: Formatted competitor section block.
    """
    delta = your_rating - their_rating
    if delta > 0:
        comparison = f"✅ You are rated *{delta:.1f} stars higher*"
    elif delta < 0:
        comparison = f"⚠️  Competitor is rated *{abs(delta):.1f} stars higher*"
    else:
        comparison = "📊 Ratings are equal"

    lines = [
        f"Competitor:       {competitor_name}",
        f"Their Rating:     {their_rating:.1f} / 5.0",
        f"Your Rating:      {your_rating:.1f} / 5.0",
        comparison,
        f"Their Reviews:    {format_number(their_review_count)} this period",
    ]
    return build_whatsapp_section("Competitor Snapshot", lines)


def format_report_footer() -> str:
    """
    Build the standard report footer block.

    Returns:
        str: Footer block for appending to any report message.
    """
    return "\n".join([
        whatsapp_divider(),
        whatsapp_italic("Generated by AI Business Agent"),
        whatsapp_italic("Reply HELP for support options."),
    ])


# ==============================================================================
# Subscription Notification Formatters
# ==============================================================================

def format_subscription_expiry_warning(
    business_name: str,
    plan: str,
    days_remaining: int,
    renewal_amount: int,
) -> str:
    """
    Build a subscription expiry warning WhatsApp message.

    Args:
        business_name:    Business name.
        plan:             Current subscription plan name.
        days_remaining:   Days until the subscription expires.
        renewal_amount:   Renewal amount in paise.

    Returns:
        str: Formatted expiry warning message.
    """
    urgency = "🚨" if days_remaining <= 3 else "⚠️"
    amount_display = format_currency_inr(renewal_amount)

    lines: list[str] = [
        build_business_greeting(business_name),
        "",
        f"{urgency} *Subscription Expiring in {days_remaining} Day{'s' if days_remaining != 1 else ''}*",
        "",
        f"Your *{plan.upper()}* plan is expiring soon.",
        f"Renewal amount: {amount_display}",
        "",
        "To continue receiving AI review replies and analytics,",
        "please renew your subscription before it expires.",
        "",
        whatsapp_divider(),
        whatsapp_italic("Powered by AI Business Agent"),
    ]
    return "\n".join(lines)


def format_payment_confirmation(
    business_name: str,
    plan: str,
    amount_paise: int,
    valid_until: str,
) -> str:
    """
    Build a payment success confirmation WhatsApp message.

    Args:
        business_name:  Business name.
        plan:           Activated subscription plan name.
        amount_paise:   Amount paid in paise.
        valid_until:    Human-readable date until which the plan is active.

    Returns:
        str: Formatted payment confirmation message.
    """
    lines: list[str] = [
        build_business_greeting(business_name),
        "",
        "✅ *Payment Confirmed — Subscription Activated*",
        "",
        f"Plan:        {whatsapp_bold(plan.upper())}",
        f"Amount Paid: {format_currency_inr(amount_paise)}",
        f"Valid Until: {valid_until}",
        "",
        "Your AI business agent is now active. 🚀",
        "",
        whatsapp_divider(),
        whatsapp_italic("Powered by AI Business Agent"),
    ]
    return "\n".join(lines)


# ==============================================================================
# Aliases used by report job modules
# ==============================================================================

def format_star_bar(rating: float, max_stars: int = 5) -> str:
    """
    Return a visual star bar string for a rating value.

    Example: format_star_bar(3.5) → "★★★½☆"

    Args:
        rating:    Numeric rating (0.0 – 5.0).
        max_stars: Maximum stars in the bar (default: 5).

    Returns:
        str: Star bar using full/half/empty star characters.
    """
    filled   = int(rating)
    half     = 1 if (rating - filled) >= 0.5 else 0
    empty    = max_stars - filled - half
    return "★" * filled + ("½" if half else "") + "☆" * empty


def format_percent(value: float, decimal_places: int = 1) -> str:
    """
    Format a float as a percentage string.

    Alias for format_percentage() used by report jobs.

    Args:
        value:          Float value (e.g. 0.75 → "75.0%").
        decimal_places: Decimal precision (default: 1).

    Returns:
        str: Formatted percentage string.
    """
    return format_percentage(value=value, decimal_places=decimal_places)