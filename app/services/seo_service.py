# ==============================================================================
# File: app/services/seo_service.py
# Purpose: Generates local SEO keyword suggestions and actionable improvement
#          tips for small businesses based on their review content, sentiment
#          trends, and business context.
#
#          Local SEO for small businesses is about appearing in Google's
#          "near me" searches and Google Maps results. The two biggest
#          levers are:
#            1. Review quantity and quality (rating + recency)
#            2. Keyword presence in the business's own review replies
#               and Google Business Profile description
#
#          This service produces:
#            - SeoSuggestionResult: structured keyword and tip list consumed
#              by reports_service.py and delivered via WhatsApp reports
#
#          Inputs consumed:
#            - Recent review texts (positive ones especially)
#            - Average star rating and trend direction
#            - Business type and name
#            - Competitor comparison context (optional)
#
#          Approach:
#            1. Extract keyword candidates from positive review texts
#               using a lightweight term frequency approach (no ML needed)
#            2. Call OpenAI with a structured prompt to generate:
#               (a) 5–8 local SEO keyword suggestions
#               (b) 3–5 actionable SEO improvement tips specific to the
#                   business's current rating and review pattern
#            3. If OpenAI fails, fall back to a rule-based suggestion set
#               derived purely from business type and rating tier
#
#          Prompt safety (guardrails §9):
#            Only review_texts (stripped, no reviewer names), business_type,
#            business_name, average_rating, and trend are sent to OpenAI.
#            No owner contact details, payment data, or internal IDs.
# ==============================================================================

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_openai_retry

logger = logging.getLogger(ServiceName.API)
settings = get_settings()

# Maximum positive review texts sent to OpenAI for keyword extraction
_MAX_REVIEW_TEXTS_FOR_SEO = 15

# Maximum characters per review text sent to OpenAI (prompt size control §10)
_MAX_CHARS_PER_REVIEW = 200

# Maximum tokens for SEO suggestions — structured JSON response
_SEO_MAX_TOKENS = 600

# Minimum word length for keyword extraction
_MIN_KEYWORD_LENGTH = 4

# Common stopwords to exclude from keyword frequency counting
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "that", "this", "with", "have", "from",
    "they", "will", "been", "were", "their", "there", "what", "when",
    "very", "good", "great", "nice", "best", "also", "just", "really",
    "would", "could", "should", "always", "never", "every", "place",
    "here", "come", "came", "went", "got", "get", "like", "love",
    "amazing", "awesome", "excellent", "highly", "recommend",
})

# -----------------------------------------------------------------------
# Rule-based fallback suggestions by business type
# -----------------------------------------------------------------------
_FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "restaurant": [
        "best restaurant near me", "home delivery food", "dine in restaurant",
        "family restaurant", "local food near me",
    ],
    "clinic": [
        "doctor near me", "best clinic near me", "health checkup",
        "general physician", "medical consultation",
    ],
    "salon": [
        "best salon near me", "hair salon", "beauty salon",
        "haircut near me", "bridal makeup",
    ],
    "spa": [
        "spa near me", "full body massage", "relaxation spa",
        "body treatment", "couple spa",
    ],
    "gym": [
        "gym near me", "fitness centre", "personal trainer",
        "weight loss gym", "yoga classes near me",
    ],
    "retail": [
        "shop near me", "local store", "best prices near me",
        "buy online", "quality products",
    ],
    "default": [
        "best service near me", "local business", "trusted service",
        "affordable near me", "top rated near me",
    ],
}

_FALLBACK_TIPS: dict[str, list[str]] = {
    "high_rating": [   # avg >= 4.5
        "Encourage happy customers to mention specific services in their reviews — keyword-rich reviews improve local search ranking.",
        "Reply to every review using natural language that includes your business type and location.",
        "Update your Google Business Profile description with your top 3–5 service keywords.",
        "Post weekly updates on your Google Business Profile — active profiles rank higher.",
    ],
    "mid_rating": [    # avg 3.5–4.4
        "Respond promptly to negative reviews with a professional apology — this signals trustworthiness to Google.",
        "Ask satisfied customers to leave a review — increasing review volume helps offset lower ratings.",
        "Include local neighbourhood name in your Google Business Profile (e.g. 'Salon in Koramangala, Bangalore').",
        "Add your top services as Google Business Profile 'Products' to appear in more search queries.",
    ],
    "low_rating": [    # avg < 3.5
        "Focus on resolving the most common complaints in reviews before running any SEO campaign.",
        "Respond to every negative review individually — Google rewards businesses that engage.",
        "Request reviews only from genuinely satisfied customers to rebuild rating momentum.",
        "Ensure your Google Business Profile has accurate hours, photos, and a complete description.",
    ],
}


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass(frozen=True)
class SeoKeyword:
    """
    A single local SEO keyword suggestion.

    Attributes:
        keyword:     The suggested keyword phrase (2–5 words typically).
        relevance:   "high", "medium", or "low" — model's confidence.
        source:      "ai" if generated by OpenAI, "fallback" if rule-based,
                     "extracted" if derived from review frequency analysis.
    """
    keyword: str
    relevance: str = "medium"
    source: str = "ai"


@dataclass(frozen=True)
class SeoTip:
    """
    A single actionable SEO improvement tip.

    Attributes:
        tip:         The actionable recommendation text.
        category:    One of: "profile", "reviews", "content", "keywords".
        priority:    "high", "medium", or "low".
        source:      "ai" or "fallback".
    """
    tip: str
    category: str = "reviews"
    priority: str = "medium"
    source: str = "ai"


@dataclass
class SeoSuggestionResult:
    """
    Complete SEO suggestion output for a business.

    Consumed by reports_service.py for inclusion in weekly/monthly reports.

    Attributes:
        business_id:        Business UUID.
        business_name:      Business name.
        keywords:           List of keyword suggestions (5–8).
        tips:               List of actionable improvement tips (3–5).
        extracted_terms:    Top terms extracted from positive reviews
                            (used as context for the keyword list).
        used_fallback:      True if OpenAI was unavailable and rule-based
                            suggestions were used instead.
        avg_rating:         The average rating used as context input.
        rating_tier:        "high", "mid", or "low" based on avg_rating.
    """
    business_id: str
    business_name: str
    keywords: list[SeoKeyword]
    tips: list[SeoTip]
    extracted_terms: list[str]
    used_fallback: bool
    avg_rating: float
    rating_tier: str

    @property
    def keyword_list(self) -> list[str]:
        """Flat list of keyword strings for easy template rendering."""
        return [k.keyword for k in self.keywords]

    @property
    def tip_list(self) -> list[str]:
        """Flat list of tip strings for easy template rendering."""
        return [t.tip for t in self.tips]

    @property
    def high_priority_tips(self) -> list[str]:
        """Tips marked as high priority."""
        return [t.tip for t in self.tips if t.priority == "high"]

    def __str__(self) -> str:
        return (
            f"SeoSuggestionResult("
            f"business={self.business_name!r} "
            f"keywords={len(self.keywords)} "
            f"tips={len(self.tips)} "
            f"rating={self.avg_rating:.1f} "
            f"fallback={self.used_fallback})"
        )


# ==============================================================================
# SEO Service
# ==============================================================================

class SeoService:
    """
    Generates local SEO keyword suggestions and actionable tips for a business.

    Stateless — safe to share a single instance across the application.

    Usage:
        service = SeoService()

        result = await service.generate_suggestions(
            business_id="uuid",
            business_name="Raj Restaurant",
            business_type="Restaurant",
            avg_rating=4.2,
            rating_trend="up",
            positive_review_texts=[
                "The biryani was amazing, very fresh ingredients...",
                "Best dosa in the area, quick service...",
            ],
        )

        for keyword in result.keyword_list:
            print(keyword)
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_suggestions(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        avg_rating: float,
        rating_trend: str = "stable",
        positive_review_texts: Optional[list[str]] = None,
        competitor_avg_rating: Optional[float] = None,
        location_hint: Optional[str] = None,
    ) -> SeoSuggestionResult:
        """
        Generate SEO keyword suggestions and improvement tips.

        Always returns a valid SeoSuggestionResult — falls back to
        rule-based suggestions if OpenAI is unavailable.

        Args:
            business_id:            Business UUID for logging.
            business_name:          Name of the business.
            business_type:          Type e.g. "Restaurant", "Clinic".
            avg_rating:             Current average star rating (1.0–5.0).
            rating_trend:           "up", "down", or "stable".
            positive_review_texts:  List of positive review text strings.
                                    Used for keyword extraction context.
            competitor_avg_rating:  Competitor's average rating (optional).
                                    Adds competitive context to tips.
            location_hint:          City/neighbourhood hint (optional).
                                    Injected into keyword suggestions when
                                    provided e.g. "Koramangala, Bangalore".

        Returns:
            SeoSuggestionResult: Always returns — never raises.
        """
        log_extra = {
            "service": ServiceName.API,
            "business_id": business_id,
            "business_type": business_type,
            "avg_rating": avg_rating,
            "rating_trend": rating_trend,
        }

        rating_tier = _rating_tier(avg_rating)
        review_texts = positive_review_texts or []

        # Step 1: Extract high-frequency terms from positive reviews
        extracted_terms = _extract_terms_from_reviews(review_texts)

        logger.debug(
            "SEO: extracted terms from reviews",
            extra={**log_extra, "extracted_terms": extracted_terms},
        )

        # Step 2: Try OpenAI for rich, context-aware suggestions
        try:
            keywords, tips = await self._generate_with_openai(
                business_name=business_name,
                business_type=business_type,
                avg_rating=avg_rating,
                rating_trend=rating_trend,
                rating_tier=rating_tier,
                extracted_terms=extracted_terms,
                competitor_avg_rating=competitor_avg_rating,
                location_hint=location_hint,
                log_extra=log_extra,
            )

            logger.info(
                "SEO suggestions generated via OpenAI",
                extra={
                    **log_extra,
                    "keyword_count": len(keywords),
                    "tip_count": len(tips),
                },
            )

            return SeoSuggestionResult(
                business_id=business_id,
                business_name=business_name,
                keywords=keywords,
                tips=tips,
                extracted_terms=extracted_terms,
                used_fallback=False,
                avg_rating=avg_rating,
                rating_tier=rating_tier,
            )

        except Exception as exc:
            logger.error(
                "SEO OpenAI call failed — using rule-based fallback",
                extra={**log_extra, "error": str(exc), "error_type": type(exc).__name__},
            )

        # Step 3: Fallback — rule-based suggestions
        keywords, tips = _build_fallback_suggestions(
            business_type=business_type,
            rating_tier=rating_tier,
            extracted_terms=extracted_terms,
            location_hint=location_hint,
        )

        logger.info(
            "SEO suggestions generated via fallback",
            extra={**log_extra, "keyword_count": len(keywords), "tip_count": len(tips)},
        )

        return SeoSuggestionResult(
            business_id=business_id,
            business_name=business_name,
            keywords=keywords,
            tips=tips,
            extracted_terms=extracted_terms,
            used_fallback=True,
            avg_rating=avg_rating,
            rating_tier=rating_tier,
        )

    # ------------------------------------------------------------------
    # OpenAI call
    # ------------------------------------------------------------------

    @with_openai_retry
    async def _generate_with_openai(
        self,
        business_name: str,
        business_type: str,
        avg_rating: float,
        rating_trend: str,
        rating_tier: str,
        extracted_terms: list[str],
        competitor_avg_rating: Optional[float],
        location_hint: Optional[str],
        log_extra: dict,
    ) -> tuple[list[SeoKeyword], list[SeoTip]]:
        """
        Call OpenAI to generate SEO keywords and tips.

        Returns:
            tuple[list[SeoKeyword], list[SeoTip]]

        Raises:
            Exception: Any OpenAI error (propagated to caller for fallback).
        """
        prompt = _build_seo_prompt(
            business_name=business_name,
            business_type=business_type,
            avg_rating=avg_rating,
            rating_trend=rating_trend,
            rating_tier=rating_tier,
            extracted_terms=extracted_terms,
            competitor_avg_rating=competitor_avg_rating,
            location_hint=location_hint,
        )

        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=_SEO_MAX_TOKENS,
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a local SEO expert for small businesses in India. "
                        "Respond ONLY with a valid JSON object — no preamble, "
                        "no explanation, no markdown code fences. "
                        "The JSON must have exactly two keys: "
                        '"keywords" (array of objects with "keyword" and "relevance" strings) and '
                        '"tips" (array of objects with "tip", "category", and "priority" strings). '
                        "keywords: 5 to 8 items. tips: 3 to 5 items. "
                        'relevance values: "high", "medium", or "low". '
                        'priority values: "high", "medium", or "low". '
                        'category values: "profile", "reviews", "content", or "keywords".'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS,
        )

        raw = response.choices[0].message.content or ""
        return _parse_seo_response(raw.strip())


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _rating_tier(avg_rating: float) -> str:
    """Classify rating into high / mid / low tier."""
    if avg_rating >= 4.5:
        return "high_rating"
    if avg_rating >= 3.5:
        return "mid_rating"
    return "low_rating"


def _extract_terms_from_reviews(review_texts: list[str]) -> list[str]:
    """
    Extract high-frequency meaningful terms from positive review texts.

    Uses simple token frequency — no ML. Useful as context for the
    OpenAI prompt and as fallback keyword seeds.

    Args:
        review_texts: List of positive review text strings.

    Returns:
        list[str]: Top 10 most frequent meaningful terms.
    """
    if not review_texts:
        return []

    all_words: list[str] = []
    texts_to_use = review_texts[:_MAX_REVIEW_TEXTS_FOR_SEO]

    for text in texts_to_use:
        # Lowercase and extract word tokens
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        for word in words:
            if (
                len(word) >= _MIN_KEYWORD_LENGTH
                and word not in _STOPWORDS
            ):
                all_words.append(word)

    counter = Counter(all_words)
    return [term for term, _ in counter.most_common(10)]


def _build_seo_prompt(
    business_name: str,
    business_type: str,
    avg_rating: float,
    rating_trend: str,
    rating_tier: str,
    extracted_terms: list[str],
    competitor_avg_rating: Optional[float],
    location_hint: Optional[str],
) -> str:
    """Build the user-role prompt for the SEO OpenAI call."""
    lines = [
        f"Business name: {business_name}",
        f"Business type: {business_type}",
        f"Current average Google rating: {avg_rating:.1f} / 5.0",
        f"Rating trend: {rating_trend}",
        f"Rating tier: {rating_tier.replace('_', ' ')}",
    ]

    if location_hint:
        lines.append(f"Location: {location_hint}")

    if extracted_terms:
        terms_str = ", ".join(extracted_terms[:10])
        lines.append(f"Terms customers frequently mention in reviews: {terms_str}")

    if competitor_avg_rating is not None:
        delta = avg_rating - competitor_avg_rating
        direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
        lines.append(
            f"Competitor average rating: {competitor_avg_rating:.1f} "
            f"(our rating is {abs(delta):.1f} stars {direction})"
        )

    lines.append(
        "\nGenerate local SEO keyword suggestions and actionable improvement tips "
        "for this business. Focus on Google Maps and 'near me' search visibility."
    )

    return "\n".join(lines)


def _parse_seo_response(
    raw: str,
) -> tuple[list[SeoKeyword], list[SeoTip]]:
    """
    Parse the OpenAI JSON response into keyword and tip lists.

    Handles:
      - Markdown code fences (model wraps despite instructions)
      - Missing keys (falls back to empty list for that section)
      - Malformed individual items (skipped, not crashed)

    Args:
        raw: Raw text from OpenAI.

    Returns:
        tuple[list[SeoKeyword], list[SeoTip]]

    Raises:
        ValueError: If the response cannot be parsed at all.
    """
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SEO response is not valid JSON: {raw!r}") from exc

    # Parse keywords
    keywords: list[SeoKeyword] = []
    for item in data.get("keywords", []):
        try:
            kw = str(item.get("keyword", "")).strip()
            if not kw:
                continue
            relevance = str(item.get("relevance", "medium")).lower()
            if relevance not in ("high", "medium", "low"):
                relevance = "medium"
            keywords.append(SeoKeyword(keyword=kw, relevance=relevance, source="ai"))
        except Exception:
            continue

    # Parse tips
    tips: list[SeoTip] = []
    for item in data.get("tips", []):
        try:
            tip_text = str(item.get("tip", "")).strip()
            if not tip_text:
                continue
            category = str(item.get("category", "reviews")).lower()
            if category not in ("profile", "reviews", "content", "keywords"):
                category = "reviews"
            priority = str(item.get("priority", "medium")).lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            tips.append(SeoTip(tip=tip_text, category=category, priority=priority, source="ai"))
        except Exception:
            continue

    if not keywords and not tips:
        raise ValueError(f"SEO response parsed but produced no usable content: {raw!r}")

    return keywords, tips


def _build_fallback_suggestions(
    business_type: str,
    rating_tier: str,
    extracted_terms: list[str],
    location_hint: Optional[str],
) -> tuple[list[SeoKeyword], list[SeoTip]]:
    """
    Build rule-based keyword and tip suggestions when OpenAI is unavailable.

    Uses business type → keyword lookup and rating tier → tip lookup.
    Injects extracted review terms and location hint if available.

    Args:
        business_type:    Business type string.
        rating_tier:      "high_rating", "mid_rating", or "low_rating".
        extracted_terms:  Terms extracted from positive reviews.
        location_hint:    Optional location string for geo-specific keywords.

    Returns:
        tuple[list[SeoKeyword], list[SeoTip]]
    """
    type_key = business_type.lower().strip()
    # Try exact match first, then check if type_key contains a known key
    if type_key not in _FALLBACK_KEYWORDS:
        for known_key in _FALLBACK_KEYWORDS:
            if known_key in type_key or type_key in known_key:
                type_key = known_key
                break
        else:
            type_key = "default"

    base_keywords = _FALLBACK_KEYWORDS[type_key]

    # Inject location into first two keywords if hint is available
    if location_hint:
        location_kw = [
            f"{base_keywords[0]} in {location_hint}",
            f"{base_keywords[1]} near {location_hint}" if len(base_keywords) > 1 else "",
        ]
        location_kw = [k for k in location_kw if k]
        combined = location_kw + base_keywords
    else:
        combined = list(base_keywords)

    # Add extracted review terms as low-relevance keyword candidates
    for term in extracted_terms[:3]:
        combined.append(f"{term} {type_key}")

    keywords = [
        SeoKeyword(keyword=kw, relevance="medium", source="fallback")
        for kw in combined[:8]
    ]

    # Get tips for the rating tier
    tip_texts = _FALLBACK_TIPS.get(rating_tier, _FALLBACK_TIPS["mid_rating"])
    tips = [
        SeoTip(
            tip=tip_text,
            category="reviews",
            priority="high" if i == 0 else "medium",
            source="fallback",
        )
        for i, tip_text in enumerate(tip_texts[:5])
    ]

    return keywords, tips