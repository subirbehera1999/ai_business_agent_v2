# ==============================================================================
# File: app/services/sentiment_service.py
# Purpose: Classifies the sentiment of Google review text into one of three
#          categories — POSITIVE, NEGATIVE, or NEUTRAL — and produces a
#          confidence score between -1.0 and 1.0.
#
#          This service is a dependency of ai_reply_service.py, which uses
#          the sentiment result to select the correct prompt template:
#            POSITIVE  → positive_review_reply_prompt.txt
#            NEGATIVE  → negative_review_reply_prompt.txt
#            NEUTRAL   → neutral_review_reply_prompt.txt
#
#          Also referenced in review_validator.py for the inline comment:
#            "Positive review  → Thank you reply"
#            "Negative review  → Apology reply"
#            "Neutral review   → Balanced response"
#
#          Design decisions:
#            1. PRIMARY path  — OpenAI API with a structured JSON prompt.
#                               Produces both a sentiment label and a
#                               float confidence score.
#            2. FALLBACK path — Star-rating heuristic when:
#                               (a) OpenAI call fails after all retries, or
#                               (b) review has no text (rating-only review).
#            3. NEVER raises  — always returns a SentimentResult. Errors are
#                               logged and the fallback fires silently so
#                               the review pipeline is never blocked.
#            4. Idempotent    — calling twice with the same review produces
#                               the same result; no side effects.
#            5. Multi-tenant  — business_id is carried through every log
#                               entry; no cross-business data flows.
#
#          Prompt safety (DATA_SAFETY_AND_RUNTIME_GUARDRAILS §9):
#            Only review_text and business_type are sent to OpenAI.
#            No phone numbers, emails, payment data, or internal fields.
# ==============================================================================

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError

from app.config.constants import ReviewSentiment, ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_openai_retry

logger = logging.getLogger(ServiceName.API)
settings = get_settings()

# ---------------------------------------------------------------------------
# Sentiment score boundaries
# ---------------------------------------------------------------------------
# Scores map to sentiments as follows:
#   >= POSITIVE_THRESHOLD  → POSITIVE
#   <= NEGATIVE_THRESHOLD  → NEGATIVE
#   between               → NEUTRAL
#
# OpenAI is instructed to return a float in [-1.0, 1.0].
# Star-rating fallback scores are mapped deterministically (see below).
# ---------------------------------------------------------------------------
POSITIVE_THRESHOLD: float = 0.25
NEGATIVE_THRESHOLD: float = -0.25

# Star-rating → (sentiment, score) when OpenAI is unavailable
_STAR_FALLBACK: dict[int, tuple[str, float]] = {
    5: (ReviewSentiment.POSITIVE, 0.95),
    4: (ReviewSentiment.POSITIVE, 0.55),
    3: (ReviewSentiment.NEUTRAL,  0.00),
    2: (ReviewSentiment.NEGATIVE, -0.55),
    1: (ReviewSentiment.NEGATIVE, -0.95),
}

# System prompt sent to OpenAI — kept minimal for token efficiency
_SYSTEM_PROMPT = (
    "You are a sentiment analysis engine for business reviews. "
    "Analyze the review and respond ONLY with a valid JSON object. "
    "No preamble, no explanation, no markdown. "
    "JSON must contain exactly two keys: "
    '"sentiment" (string: "positive", "negative", or "neutral") and '
    '"score" (float between -1.0 and 1.0, where '
    "1.0 = very positive, -1.0 = very negative, 0.0 = neutral). "
    "Base your analysis primarily on the review text. "
    "Use the star rating as secondary context only."
)

# User prompt template — business_type provides context without exposing
# any sensitive business data (guardrails §9)
_USER_PROMPT_TEMPLATE = (
    "Business type: {business_type}\n"
    "Star rating: {rating}/5\n"
    "Review text: {review_text}\n\n"
    "Respond with JSON only."
)


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass(frozen=True)
class SentimentResult:
    """
    Immutable result of a sentiment analysis operation.

    Attributes:
        sentiment:     One of ReviewSentiment.POSITIVE / NEGATIVE / NEUTRAL.
        score:         Confidence float in [-1.0, 1.0].
                       Positive = positive sentiment, negative = negative.
        used_fallback: True if the star-rating heuristic was used instead
                       of OpenAI (e.g. API failure or no review text).
        raw_response:  Raw JSON string returned by OpenAI (None if fallback).
    """

    sentiment: str
    score: float
    used_fallback: bool = False
    raw_response: Optional[str] = None

    @property
    def is_positive(self) -> bool:
        return self.sentiment == ReviewSentiment.POSITIVE

    @property
    def is_negative(self) -> bool:
        return self.sentiment == ReviewSentiment.NEGATIVE

    @property
    def is_neutral(self) -> bool:
        return self.sentiment == ReviewSentiment.NEUTRAL

    @property
    def score_display(self) -> str:
        """Human-readable score for logging e.g. '+0.82' or '-0.45'."""
        sign = "+" if self.score >= 0 else ""
        return f"{sign}{self.score:.2f}"

    def __str__(self) -> str:
        source = "fallback" if self.used_fallback else "openai"
        return (
            f"SentimentResult("
            f"sentiment={self.sentiment} "
            f"score={self.score_display} "
            f"source={source})"
        )


# ==============================================================================
# Sentiment Service
# ==============================================================================

class SentimentService:
    """
    Classifies Google review text into POSITIVE / NEGATIVE / NEUTRAL
    with a confidence score.

    This service is stateless. A single instance can be shared across
    the application (instantiate once, inject via dependency injection
    or pass directly into the review pipeline).

    Usage:
        service = SentimentService()
        result = await service.analyze(
            review_text="The food was amazing, will come again!",
            star_rating=5,
            business_id="uuid-here",
            business_type="Restaurant",
        )
        print(result.sentiment)   # "positive"
        print(result.score)       # 0.88
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        review_text: Optional[str],
        star_rating: int,
        business_id: str,
        business_type: str = "Local Business",
        review_id: Optional[str] = None,
    ) -> SentimentResult:
        """
        Analyze the sentiment of a review.

        Tries OpenAI first. Falls back to the star-rating heuristic if:
          - review_text is empty or None
          - OpenAI returns an unparseable response
          - OpenAI raises any exception after all retries

        This method never raises. All errors are caught, logged, and
        resolved via the fallback path.

        Args:
            review_text:    Text body of the review (may be None or empty
                            for rating-only reviews).
            star_rating:    Star rating 1–5.
            business_id:    ID of the business (for log traceability).
            business_type:  Type of business for prompt context
                            e.g. "Restaurant", "Clinic", "Salon".
            review_id:      Optional review identifier for log traceability.

        Returns:
            SentimentResult: Always returns a valid result.
        """
        log_extra = {
            "service": ServiceName.API,
            "business_id": business_id,
            "review_id": review_id or "unknown",
            "star_rating": star_rating,
            "has_text": bool(review_text and review_text.strip()),
        }

        # Path 1: No text — skip OpenAI, use star-rating heuristic directly
        if not review_text or not review_text.strip():
            logger.info(
                "Sentiment analysis: no review text — using star-rating fallback",
                extra=log_extra,
            )
            return self._fallback(star_rating)

        # Path 2: Use OpenAI with retry wrapper
        try:
            result = await self._analyze_with_openai(
                review_text=review_text.strip(),
                star_rating=star_rating,
                business_type=business_type,
                log_extra=log_extra,
            )
            logger.info(
                "Sentiment analysis completed via OpenAI",
                extra={
                    **log_extra,
                    "sentiment": result.sentiment,
                    "score": result.score_display,
                },
            )
            return result

        except Exception as exc:
            # Path 3: OpenAI failed after all retries — use heuristic
            logger.error(
                "Sentiment analysis OpenAI call failed — using star-rating fallback",
                extra={
                    **log_extra,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            return self._fallback(star_rating)

    async def analyze_batch(
        self,
        reviews: list[dict],
        business_id: str,
        business_type: str = "Local Business",
    ) -> list[SentimentResult]:
        """
        Analyze sentiment for a batch of reviews.

        Each review is processed individually. A failure on one review
        does not affect the others — it falls back to heuristic silently.

        Args:
            reviews:       List of dicts, each with keys:
                             - review_text (str | None)
                             - star_rating (int)
                             - review_id   (str | None, optional)
            business_id:   Business ID for log traceability.
            business_type: Business type for OpenAI context.

        Returns:
            list[SentimentResult]: One result per input review, in order.
        """
        results: list[SentimentResult] = []

        for review in reviews:
            result = await self.analyze(
                review_text=review.get("review_text"),
                star_rating=review.get("star_rating", 3),
                business_id=business_id,
                business_type=business_type,
                review_id=review.get("review_id"),
            )
            results.append(result)

        logger.info(
            "Sentiment batch analysis complete",
            extra={
                "service": ServiceName.API,
                "business_id": business_id,
                "total": len(reviews),
                "positive": sum(1 for r in results if r.is_positive),
                "negative": sum(1 for r in results if r.is_negative),
                "neutral": sum(1 for r in results if r.is_neutral),
                "used_fallback": sum(1 for r in results if r.used_fallback),
            },
        )

        return results

    # ------------------------------------------------------------------
    # Internal: OpenAI call
    # ------------------------------------------------------------------

    @with_openai_retry
    async def _analyze_with_openai(
        self,
        review_text: str,
        star_rating: int,
        business_type: str,
        log_extra: dict,
    ) -> SentimentResult:
        """
        Call OpenAI to classify sentiment.

        Decorated with @with_openai_retry (3 attempts, exponential backoff).
        Raises on unrecoverable errors so the caller can trigger fallback.

        Args:
            review_text:   Stripped review text (guaranteed non-empty).
            star_rating:   Star rating 1–5.
            business_type: Context for the prompt.
            log_extra:     Structured log context dict.

        Returns:
            SentimentResult with used_fallback=False.

        Raises:
            ValueError:        If OpenAI returns a non-parseable response.
            RateLimitError:    Propagated to retry wrapper.
            APITimeoutError:   Propagated to retry wrapper.
            APIError:          Propagated to retry wrapper.
        """
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            business_type=business_type,
            rating=star_rating,
            review_text=_safe_truncate_for_prompt(review_text),
        )

        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=100,   # Sentiment JSON is tiny — hard cap for cost control
            temperature=0.0,  # Deterministic — same review always same result
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS,
        )

        raw_text = response.choices[0].message.content or ""
        raw_text = raw_text.strip()

        logger.debug(
            "OpenAI sentiment raw response",
            extra={**log_extra, "raw_response": raw_text[:200]},
        )

        return _parse_openai_response(raw_text, star_rating)

    # ------------------------------------------------------------------
    # Internal: star-rating heuristic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(star_rating: int) -> SentimentResult:
        """
        Determine sentiment from star rating alone.

        Used when OpenAI is unavailable or review has no text.
        Clamped to 1–5 to handle any unexpected rating values.

        Args:
            star_rating: Raw star rating integer.

        Returns:
            SentimentResult with used_fallback=True.
        """
        clamped = max(1, min(star_rating, 5))
        sentiment, score = _STAR_FALLBACK[clamped]
        return SentimentResult(
            sentiment=sentiment,
            score=score,
            used_fallback=True,
            raw_response=None,
        )


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _safe_truncate_for_prompt(text: str, max_chars: int = 600) -> str:
    """
    Truncate review text before sending to OpenAI.

    Reviews are already capped at 800 chars in the database, but we
    apply a tighter cap here (600 chars) per PERFORMANCE_AND_SCALABILITY_CONTRACT
    §10: "Prompt Size Control — avoid including excessive raw data."

    Args:
        text:      Review text (stripped, non-empty).
        max_chars: Maximum characters to send (default 600).

    Returns:
        str: Truncated text with ellipsis if shortened.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def _parse_openai_response(raw: str, star_rating: int) -> SentimentResult:
    """
    Parse the OpenAI JSON response into a SentimentResult.

    Handles common failure modes:
      - JSON wrapped in markdown code fences
      - Unexpected sentiment label casing
      - Score outside [-1.0, 1.0] range
      - Missing keys

    Args:
        raw:         Raw text from OpenAI completion.
        star_rating: Used to derive fallback score if parse fails.

    Returns:
        SentimentResult with used_fallback=False.

    Raises:
        ValueError: If the response cannot be parsed into a valid result.
    """
    # Strip markdown fences if model wraps response despite instructions
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OpenAI sentiment response is not valid JSON: {raw!r}"
        ) from exc

    # Validate sentiment label
    raw_sentiment = str(data.get("sentiment", "")).lower().strip()
    sentiment_map = {
        "positive": ReviewSentiment.POSITIVE,
        "negative": ReviewSentiment.NEGATIVE,
        "neutral":  ReviewSentiment.NEUTRAL,
    }
    if raw_sentiment not in sentiment_map:
        raise ValueError(
            f"OpenAI returned unknown sentiment label: {raw_sentiment!r}"
        )
    sentiment = sentiment_map[raw_sentiment]

    # Validate and clamp score
    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"OpenAI sentiment response missing or invalid 'score': {data!r}"
        ) from exc

    score = max(-1.0, min(1.0, score))  # Clamp to valid range

    # Cross-validate: if OpenAI returns "positive" but score is negative,
    # trust the label and correct the score to be consistent.
    if sentiment == ReviewSentiment.POSITIVE and score < POSITIVE_THRESHOLD:
        score = POSITIVE_THRESHOLD
    elif sentiment == ReviewSentiment.NEGATIVE and score > NEGATIVE_THRESHOLD:
        score = NEGATIVE_THRESHOLD

    return SentimentResult(
        sentiment=sentiment,
        score=round(score, 4),
        used_fallback=False,
        raw_response=raw,
    )


def score_to_sentiment(score: float) -> str:
    """
    Convert a float score to a sentiment label.

    Useful for re-deriving a label from a stored score without
    re-calling the service.

    Args:
        score: Float in [-1.0, 1.0].

    Returns:
        str: ReviewSentiment constant.
    """
    if score >= POSITIVE_THRESHOLD:
        return ReviewSentiment.POSITIVE
    if score <= NEGATIVE_THRESHOLD:
        return ReviewSentiment.NEGATIVE
    return ReviewSentiment.NEUTRAL