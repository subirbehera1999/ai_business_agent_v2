# ==============================================================================
# File: app/validators/review_validator.py
# Purpose: Validates and filters Google reviews before they enter the
#          AI processing pipeline.
#
#          Called by review_monitor.py for every new review fetched
#          from the Google Business API before any processing occurs.
#
#          Why validation before processing?
#            Raw data from Google's API is not always clean:
#              - Reviews can be empty (star rating only, no text)
#              - Reviews can contain spam, repeated characters, or gibberish
#              - Bot-generated reviews can be detected and filtered
#              - Extremely short reviews carry no signal for AI replies
#              - Reviews with invalid ratings (outside 1-5) corrupt analytics
#
#            Filtering bad reviews before they reach the AI:
#              - Saves OpenAI API costs (no AI call for spam)
#              - Prevents garbage AI replies going live on Google
#              - Keeps the analytics database clean
#              - Protects sentiment analysis from noise
#
#          Validation pipeline (in order, cheapest first):
#            1. Field presence check  — review_id and rating must exist
#            2. Rating range check    — must be 1 to 5 inclusive
#            3. Reviewer name check   — basic sanitisation
#            4. Empty comment check   — no text is valid (handled separately)
#            5. Minimum length check  — comments shorter than MIN_COMMENT_CHARS
#                                       are flagged as low-signal
#            6. Spam pattern check    — repeated characters, all caps, URLs
#            7. Gibberish detection   — character variety ratio
#            8. Profanity/abuse check — blocks severely abusive content
#
#          Two-level outcome:
#            VALID:         Process normally — generate AI reply
#            VALID_NO_REPLY: Store and notify, but skip AI reply
#                            (empty comment, very short, borderline spam)
#            INVALID:       Reject entirely — do not store or process
#
#          Sentiment routing (informational, used by ai_reply_service.py):
#            After validation passes, the validator assigns a preliminary
#            routing hint based on the star rating:
#              5 stars → positive  → positive_review_reply_prompt.txt
#              3–4 stars → neutral  → neutral_review_reply_prompt.txt
#              1–2 stars → negative → negative_review_reply_prompt.txt
#
#            This is a fast heuristic. The actual sentiment is determined
#            by sentiment_service.py using NLP — the routing hint is only
#            used as a fallback if NLP fails.
#
#          Multi-tenant:
#            The validator is stateless — it does not query the database.
#            The same validator instance is used for all businesses.
# ==============================================================================

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.config.constants import ServiceName

logger = logging.getLogger(ServiceName.REVIEW_VALIDATOR)

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
MIN_COMMENT_CHARS: int = 3          # below this → VALID_NO_REPLY
MAX_COMMENT_CHARS: int = 5000       # above this → truncate (not reject)
MIN_CHAR_VARIETY_RATIO: float = 0.1 # unique chars / total chars — below = gibberish
MAX_REPEATED_CHAR_RATIO: float = 0.6 # single char repeated more than 60% → spam
MAX_CAPS_RATIO: float = 0.85        # more than 85% uppercase → spam
MAX_URLS_ALLOWED: int = 0           # reviews with URLs are flagged as spam
MIN_RATING: int = 1
MAX_RATING: int = 5


# ==============================================================================
# Outcome enums
# ==============================================================================

class ValidationOutcome(str, Enum):
    """
    Result of review validation.

    VALID:          Review passes all checks. Store + generate AI reply.
    VALID_NO_REPLY: Review is acceptable but low-signal. Store + notify,
                    but skip AI reply generation (save API costs).
    INVALID:        Review fails hard validation. Reject entirely.
    """
    VALID = "valid"
    VALID_NO_REPLY = "valid_no_reply"
    INVALID = "invalid"


class RejectionReason(str, Enum):
    """
    Why a review was rejected or downgraded.
    Used for logging and admin visibility.
    """
    MISSING_REVIEW_ID   = "missing_review_id"
    MISSING_RATING      = "missing_rating"
    INVALID_RATING      = "invalid_rating_range"
    EMPTY_COMMENT       = "empty_comment"
    COMMENT_TOO_SHORT   = "comment_too_short"
    SPAM_REPEATED_CHARS = "spam_repeated_characters"
    SPAM_ALL_CAPS       = "spam_all_caps"
    SPAM_CONTAINS_URL   = "spam_contains_url"
    GIBBERISH_TEXT      = "gibberish_low_char_variety"
    ABUSIVE_CONTENT     = "abusive_content"
    INVALID_REVIEWER    = "invalid_reviewer_name"
    NONE                = "none"


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class ValidationResult:
    """
    Result of validating a single review.

    Attributes:
        is_valid:           True if review should be stored.
        should_reply:       True if AI reply should be generated.
        outcome:            VALID / VALID_NO_REPLY / INVALID.
        rejection_reason:   Why the review was rejected or downgraded.
        sentiment_hint:     Preliminary routing hint from star rating.
        sanitised_comment:  Cleaned comment text (truncated if too long).
        sanitised_name:     Cleaned reviewer name.
    """
    is_valid: bool
    should_reply: bool
    outcome: ValidationOutcome
    rejection_reason: RejectionReason = RejectionReason.NONE
    sentiment_hint: str = "neutral"
    sanitised_comment: Optional[str] = None
    sanitised_name: Optional[str] = None

    @property
    def is_rejected(self) -> bool:
        return self.outcome == ValidationOutcome.INVALID

    @property
    def is_low_signal(self) -> bool:
        return self.outcome == ValidationOutcome.VALID_NO_REPLY


# ==============================================================================
# Review Validator
# ==============================================================================

class ReviewValidator:
    """
    Stateless validator for Google Business reviews.

    Validates reviews before they enter the AI processing pipeline.
    Cheap checks run first (field presence, rating range) before
    expensive regex-based spam detection.

    Usage:
        validator = ReviewValidator()
        result = validator.validate(review)
        if not result.is_valid:
            return  # skip this review entirely
        if result.should_reply:
            # generate AI reply
    """

    def validate(self, review) -> ValidationResult:
        """
        Run the full validation pipeline on a single review object.

        The review object is a GoogleReview dataclass from
        google_reviews_client.py with fields:
          review_id, reviewer_name, rating, comment, create_time

        Validation order (cheapest to most expensive):
          1. review_id presence
          2. rating presence and range
          3. reviewer name sanitisation
          4. comment presence (empty = VALID_NO_REPLY)
          5. comment length (too short = VALID_NO_REPLY)
          6. spam patterns
          7. gibberish detection
          8. abusive content

        Args:
            review: GoogleReview dataclass instance.

        Returns:
            ValidationResult. Never raises.
        """
        try:
            return self._run_pipeline(review)
        except Exception as exc:
            # Validator itself failed — log and reject the review
            # to prevent bad data entering the pipeline
            logger.error(
                "ReviewValidator encountered unexpected error — rejecting review",
                extra={
                    "service": ServiceName.REVIEW_VALIDATOR,
                    "error": str(exc),
                    "review_id": getattr(review, "review_id", "unknown"),
                },
            )
            return ValidationResult(
                is_valid=False,
                should_reply=False,
                outcome=ValidationOutcome.INVALID,
                rejection_reason=RejectionReason.MISSING_REVIEW_ID,
            )

    def _run_pipeline(self, review) -> ValidationResult:
        """Execute all validation steps in order."""

        review_id = getattr(review, "review_id", None)
        rating = getattr(review, "rating", None)
        raw_comment = getattr(review, "comment", None) or ""
        raw_name = getattr(review, "reviewer_name", None) or ""

        # ── Step 1: review_id presence ────────────────────────────────
        if not review_id or not str(review_id).strip():
            return self._reject(RejectionReason.MISSING_REVIEW_ID)

        # ── Step 2: rating presence ───────────────────────────────────
        if rating is None:
            return self._reject(RejectionReason.MISSING_RATING)

        # ── Step 3: rating range ──────────────────────────────────────
        try:
            rating_int = int(rating)
        except (TypeError, ValueError):
            return self._reject(RejectionReason.INVALID_RATING)

        if not (MIN_RATING <= rating_int <= MAX_RATING):
            return self._reject(RejectionReason.INVALID_RATING)

        # ── Step 4: Reviewer name sanitisation ────────────────────────
        sanitised_name = _sanitise_name(raw_name)

        # ── Step 5: Comment presence check ───────────────────────────
        comment = raw_comment.strip()
        if not comment:
            # Star-only review — valid to store, but no AI reply needed
            # (nothing to respond to beyond "thank you for the stars")
            sentiment_hint = _rating_to_sentiment(rating_int)
            return ValidationResult(
                is_valid=True,
                should_reply=False,
                outcome=ValidationOutcome.VALID_NO_REPLY,
                rejection_reason=RejectionReason.EMPTY_COMMENT,
                sentiment_hint=sentiment_hint,
                sanitised_comment=None,
                sanitised_name=sanitised_name,
            )

        # ── Step 6: Minimum length check ─────────────────────────────
        if len(comment) < MIN_COMMENT_CHARS:
            sentiment_hint = _rating_to_sentiment(rating_int)
            return ValidationResult(
                is_valid=True,
                should_reply=False,
                outcome=ValidationOutcome.VALID_NO_REPLY,
                rejection_reason=RejectionReason.COMMENT_TOO_SHORT,
                sentiment_hint=sentiment_hint,
                sanitised_comment=comment[:MAX_COMMENT_CHARS],
                sanitised_name=sanitised_name,
            )

        # ── Step 7: Truncate very long comments ───────────────────────
        # Truncate before expensive checks to avoid regex on 50KB strings
        if len(comment) > MAX_COMMENT_CHARS:
            comment = comment[:MAX_COMMENT_CHARS]

        # ── Step 8: Spam pattern checks ───────────────────────────────
        spam_result = _check_spam_patterns(comment)
        if spam_result is not None:
            logger.debug(
                "Review flagged as spam",
                extra={
                    "service": ServiceName.REVIEW_VALIDATOR,
                    "review_id": review_id,
                    "reason": spam_result.value,
                },
            )
            return ValidationResult(
                is_valid=True,        # store spam reviews for admin visibility
                should_reply=False,   # but do not generate AI reply
                outcome=ValidationOutcome.VALID_NO_REPLY,
                rejection_reason=spam_result,
                sentiment_hint=_rating_to_sentiment(rating_int),
                sanitised_comment=comment,
                sanitised_name=sanitised_name,
            )

        # ── Step 9: Gibberish detection ───────────────────────────────
        if _is_gibberish(comment):
            logger.debug(
                "Review flagged as gibberish",
                extra={
                    "service": ServiceName.REVIEW_VALIDATOR,
                    "review_id": review_id,
                },
            )
            return ValidationResult(
                is_valid=True,
                should_reply=False,
                outcome=ValidationOutcome.VALID_NO_REPLY,
                rejection_reason=RejectionReason.GIBBERISH_TEXT,
                sentiment_hint=_rating_to_sentiment(rating_int),
                sanitised_comment=comment,
                sanitised_name=sanitised_name,
            )

        # ── Step 10: Abusive content check ───────────────────────────
        if _contains_severe_abuse(comment):
            logger.warning(
                "Review contains abusive content — flagged, no AI reply",
                extra={
                    "service": ServiceName.REVIEW_VALIDATOR,
                    "review_id": review_id,
                },
            )
            return ValidationResult(
                is_valid=True,        # store for admin awareness
                should_reply=False,   # do not auto-reply to abusive content
                outcome=ValidationOutcome.VALID_NO_REPLY,
                rejection_reason=RejectionReason.ABUSIVE_CONTENT,
                sentiment_hint="negative",
                sanitised_comment=comment,
                sanitised_name=sanitised_name,
            )

        # ── All checks passed — VALID ─────────────────────────────────
        sentiment_hint = _rating_to_sentiment(rating_int)

        logger.debug(
            "Review validated successfully",
            extra={
                "service": ServiceName.REVIEW_VALIDATOR,
                "review_id": review_id,
                "rating": rating_int,
                "sentiment_hint": sentiment_hint,
                "comment_length": len(comment),
            },
        )

        return ValidationResult(
            is_valid=True,
            should_reply=True,
            outcome=ValidationOutcome.VALID,
            rejection_reason=RejectionReason.NONE,
            sentiment_hint=sentiment_hint,
            sanitised_comment=comment,
            sanitised_name=sanitised_name,
        )

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(reason: RejectionReason) -> ValidationResult:
        """Build a hard-rejection result."""
        return ValidationResult(
            is_valid=False,
            should_reply=False,
            outcome=ValidationOutcome.INVALID,
            rejection_reason=reason,
        )


# ==============================================================================
# Batch validation
# ==============================================================================

def validate_reviews_batch(
    reviews: list,
    validator: Optional[ReviewValidator] = None,
) -> tuple[list, list, list]:
    """
    Validate a list of reviews and split by outcome.

    Args:
        reviews:   List of GoogleReview objects.
        validator: ReviewValidator instance (creates one if None).

    Returns:
        Tuple of three lists:
          (valid_with_reply, valid_no_reply, invalid)

        valid_with_reply: Reviews to store AND generate AI reply for.
        valid_no_reply:   Reviews to store but NOT generate AI reply for.
        invalid:          Reviews to discard entirely.
    """
    v = validator or ReviewValidator()
    valid_with_reply = []
    valid_no_reply = []
    invalid = []

    for review in reviews:
        result = v.validate(review)
        # Attach the validation result to the review for downstream use
        review._validation = result

        if result.outcome == ValidationOutcome.VALID:
            valid_with_reply.append(review)
        elif result.outcome == ValidationOutcome.VALID_NO_REPLY:
            valid_no_reply.append(review)
        else:
            invalid.append(review)

    logger.debug(
        "Batch validation complete",
        extra={
            "service": ServiceName.REVIEW_VALIDATOR,
            "total": len(reviews),
            "valid_with_reply": len(valid_with_reply),
            "valid_no_reply": len(valid_no_reply),
            "invalid": len(invalid),
        },
    )

    return valid_with_reply, valid_no_reply, invalid


# ==============================================================================
# Pure helper functions
# ==============================================================================

def _rating_to_sentiment(rating: int) -> str:
    """
    Map a star rating to a preliminary sentiment routing hint.

      5 stars        → positive  (thank-you reply prompt)
      3–4 stars      → neutral   (balanced response prompt)
      1–2 stars      → negative  (apology/recovery reply prompt)
    """
    if rating >= 5:
        return "positive"
    if rating >= 3:
        return "neutral"
    return "negative"


def _sanitise_name(name: str) -> str:
    """
    Sanitise reviewer name for safe storage and display.

    - Strips leading/trailing whitespace
    - Removes control characters
    - Truncates to 100 characters
    - Returns "Anonymous" if the result is empty
    """
    if not name:
        return "Anonymous"
    # Remove control characters (tab, newline, etc.)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    cleaned = cleaned[:100]
    return cleaned if cleaned else "Anonymous"


def _check_spam_patterns(comment: str) -> Optional[RejectionReason]:
    """
    Check a comment for known spam patterns.

    Returns the RejectionReason if spam is detected, None otherwise.

    Checks (in order):
      1. URLs — reviews with links are almost always spam
      2. Repeated single character — "aaaaaaaaaa" style
      3. All caps — "THIS PLACE IS THE WORST!!!"
    """
    # 1. URL detection — any http/https/www link
    if re.search(r"(https?://|www\.)\S+", comment, re.IGNORECASE):
        return RejectionReason.SPAM_CONTAINS_URL

    # Strip whitespace and punctuation for character ratio checks
    letters_only = re.sub(r"[^a-zA-Z\u0900-\u097F\u0980-\u09FF]", "", comment)

    if len(letters_only) < MIN_COMMENT_CHARS:
        # Not enough letters to analyse — not spam, just short
        return None

    # 2. Single repeated character ratio
    # Count the most frequent character
    if letters_only:
        most_common_char = max(set(letters_only.lower()), key=letters_only.lower().count)
        repeat_ratio = letters_only.lower().count(most_common_char) / len(letters_only)
        if repeat_ratio > MAX_REPEATED_CHAR_RATIO:
            return RejectionReason.SPAM_REPEATED_CHARS

    # 3. All-caps ratio (only meaningful for ASCII text)
    ascii_letters = re.sub(r"[^a-zA-Z]", "", comment)
    if len(ascii_letters) >= 10:
        caps_ratio = sum(1 for c in ascii_letters if c.isupper()) / len(ascii_letters)
        if caps_ratio > MAX_CAPS_RATIO:
            return RejectionReason.SPAM_ALL_CAPS

    return None


def _is_gibberish(comment: str) -> bool:
    """
    Detect gibberish text by measuring character variety.

    A real review in any language uses a reasonable variety of characters.
    "asdfjkl asdfjkl asdfjkl" or "xzxzxzxzxz" are gibberish.

    Method: unique characters / total characters
    If this ratio is below MIN_CHAR_VARIETY_RATIO, it is likely gibberish.

    This is intentionally lenient to avoid false positives on
    short reviews, reviews in non-Latin scripts, or reviews
    that simply repeat a word ("good good good very good").
    """
    # Only analyse comments long enough to be meaningful
    cleaned = re.sub(r"\s", "", comment)
    if len(cleaned) < 15:
        return False

    unique_chars = len(set(cleaned.lower()))
    variety_ratio = unique_chars / len(cleaned)

    return variety_ratio < MIN_CHAR_VARIETY_RATIO


def _contains_severe_abuse(comment: str) -> bool:
    """
    Detect severely abusive content that should not receive an automated reply.

    This check uses a conservative list of patterns to avoid false positives.
    The goal is to catch:
      - Threats of violence
      - Extreme hate speech patterns
      - Personal targeted abuse

    We do NOT try to detect all offensive language — the AI reply service
    is capable of handling negative reviews with professional responses.
    Only reviews requiring human review (not automated reply) are flagged here.

    Businesses can review flagged reviews in their dashboard.
    """
    # Conservative pattern set — only clear threats and extreme abuse
    _SEVERE_PATTERNS = (
        r"\b(i will kill|i will hurt|death threat|bomb|shoot you)\b",
        r"\b(go die|kill yourself|kys)\b",
    )

    comment_lower = comment.lower()
    for pattern in _SEVERE_PATTERNS:
        if re.search(pattern, comment_lower, re.IGNORECASE):
            return True

    return False