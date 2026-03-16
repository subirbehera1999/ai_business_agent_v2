# ==============================================================================
# File: app/services/content_generation_service.py
# Purpose: Generates weekly social media content suggestions for local
#          businesses — Instagram captions, WhatsApp status text, and
#          Google Business Profile post copy.
#
#          Value delivered to business owners:
#            Small local businesses rarely have time or marketing expertise
#            to create consistent social media content. This service produces
#            ready-to-use post copy each week, contextualised to their
#            business type, recent customer sentiment, and seasonal timing.
#
#          Outputs per weekly run:
#            - 3 content pieces, each targeting a different platform/goal:
#                1. Instagram caption   (engagement-focused, with hashtags)
#                2. WhatsApp status     (short, conversational, with CTA)
#                3. Google Post copy    (professional, keyword-rich, SEO-aware)
#
#          Context injected into prompts (guardrails §9 compliant):
#            - Business name and type
#            - Current average rating
#            - Dominant sentiment from recent reviews (positive/neutral/negative)
#            - Top themes customers praise (from extracted review terms)
#            - Current week / month label (for seasonal relevance)
#            - Any special context the business registered (e.g. "We do catering")
#            NO: reviewer names, phone numbers, emails, payment data
#
#          Prompt file:
#            app/prompts/content_generation_prompt.txt
#
#          Fallback:
#            If OpenAI fails, rule-based template content is generated
#            per business type. Never returns empty — every business
#            always gets at least basic post copy suggestions.
#
#          Rate limiting:
#            Content generation counts toward ai_replies_generated usage
#            (it is an OpenAI call, same cost category).
#            Check is performed before calling OpenAI.
# ==============================================================================

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_openai_retry
from app.utils.time_utils import get_current_week_label, get_current_month_label

logger = logging.getLogger(ServiceName.CONTENT)
settings = get_settings()

# Prompt file location
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CONTENT_PROMPT_FILE = "content_generation_prompt.txt"

# OpenAI settings
_CONTENT_MAX_TOKENS = 800
_CONTENT_TEMPERATURE = 0.75   # slightly more creative than insights

# Max characters per piece sent back from OpenAI before truncation
_MAX_PIECE_CHARS = 600

# Platform identifiers
PLATFORM_INSTAGRAM = "instagram"
PLATFORM_WHATSAPP  = "whatsapp_status"
PLATFORM_GOOGLE    = "google_post"

ALL_PLATFORMS = (PLATFORM_INSTAGRAM, PLATFORM_WHATSAPP, PLATFORM_GOOGLE)

# Max hashtags allowed in Instagram captions
_MAX_HASHTAGS = 10


# ==============================================================================
# Fallback template library — one set per business type
# ==============================================================================

_FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "restaurant": {
        PLATFORM_INSTAGRAM: (
            "Every dish tells a story. Come experience flavours crafted with love "
            "and the freshest ingredients. 🍽️ We're open today — your table is waiting!\n"
            "#restaurant #foodie #localfood #homestyle #freshfood "
            "#delicious #foodphotography #eatlocal"
        ),
        PLATFORM_WHATSAPP: (
            "🍽️ Fresh food, warm smiles — we're open today! "
            "Drop in or call to reserve your table. See you soon! 😊"
        ),
        PLATFORM_GOOGLE: (
            "Visit us this week for freshly prepared meals made with quality ingredients. "
            "Our team is committed to giving you a great dining experience every visit. "
            "Check our menu and book your table today."
        ),
    },
    "salon": {
        PLATFORM_INSTAGRAM: (
            "Your best look starts here. ✂️💆 Our skilled stylists are ready to "
            "transform your look this week. Book your appointment now!\n"
            "#salon #hairstyle #beauty #haircut #skincare "
            "#glamour #selfcare #hairgoals #beautysalon"
        ),
        PLATFORM_WHATSAPP: (
            "✂️ New week, new look! Book your appointment with us today "
            "and walk out feeling amazing. Call us to reserve your slot! 💆"
        ),
        PLATFORM_GOOGLE: (
            "Book a salon appointment this week for a fresh, professional look. "
            "Our experienced stylists specialise in haircuts, colouring, and skin treatments. "
            "Walk-ins welcome. Call us or visit to schedule your appointment."
        ),
    },
    "clinic": {
        PLATFORM_INSTAGRAM: (
            "Your health is your greatest wealth. 🏥💙 Our clinic is open today "
            "with experienced doctors ready to serve you. "
            "Book your consultation now.\n"
            "#health #clinic #doctor #wellness #healthcare "
            "#medicalcare #localclinic #staywell"
        ),
        PLATFORM_WHATSAPP: (
            "🏥 Your health matters. Our clinic is open today — "
            "call us to book your appointment with our experienced team. "
            "Take care of yourself! 💙"
        ),
        PLATFORM_GOOGLE: (
            "Book a consultation at our clinic this week. "
            "Our qualified medical team provides compassionate, professional care "
            "for patients of all ages. Same-day appointments available. Contact us today."
        ),
    },
    "gym": {
        PLATFORM_INSTAGRAM: (
            "Every rep brings you closer to your goal. 💪🔥 "
            "Join us this week and let's train together. "
            "Your fitness journey starts with one step!\n"
            "#gym #fitness #workout #training #fitlife "
            "#motivation #health #strengthtraining #gymlife"
        ),
        PLATFORM_WHATSAPP: (
            "💪 New week, new goals! Come train with us — "
            "our coaches are ready to help you push past your limits. "
            "See you at the gym! 🔥"
        ),
        PLATFORM_GOOGLE: (
            "Start your fitness journey with us this week. "
            "Our gym offers modern equipment, certified personal trainers, "
            "and flexible membership plans. Visit us for a free trial session today."
        ),
    },
    "spa": {
        PLATFORM_INSTAGRAM: (
            "You deserve to feel refreshed and renewed. 🧖✨ "
            "Book your spa session this week and let us take care of you. "
            "Relaxation is just one appointment away!\n"
            "#spa #wellness #massage #relax #selfcare "
            "#beauty #rejuvenate #spday #bodymassage"
        ),
        PLATFORM_WHATSAPP: (
            "🧖 Treat yourself this week! Book a relaxing spa session "
            "and recharge your body and mind. Call us to reserve your slot. ✨"
        ),
        PLATFORM_GOOGLE: (
            "Book a spa session this week for complete relaxation and rejuvenation. "
            "We offer full body massages, skin treatments, and wellness therapies "
            "by trained professionals. Gift vouchers available. Book online or call us."
        ),
    },
    "retail": {
        PLATFORM_INSTAGRAM: (
            "New week, new arrivals! 🛍️ Visit our store and discover "
            "quality products at the best prices in town. "
            "Your favourites are waiting for you!\n"
            "#shopping #retail #localbusiness #newarrivals "
            "#shoplocal #deals #quality #store"
        ),
        PLATFORM_WHATSAPP: (
            "🛍️ Fresh stock just arrived! Visit us this week "
            "for great products at unbeatable prices. "
            "We're open and ready for you! 😊"
        ),
        PLATFORM_GOOGLE: (
            "Visit our store this week to explore our latest collection "
            "and take advantage of our weekly specials. "
            "We stock quality products at competitive prices for local customers."
        ),
    },
    "default": {
        PLATFORM_INSTAGRAM: (
            "Another week, another opportunity to serve you better! 🙌 "
            "We're open and ready to make your experience exceptional. "
            "Visit us today!\n"
            "#localbusiness #service #quality #community "
            "#supportlocal #shoplocal #customerservice"
        ),
        PLATFORM_WHATSAPP: (
            "🙌 We're open this week and ready to serve you! "
            "Come visit us or get in touch — we'd love to see you. 😊"
        ),
        PLATFORM_GOOGLE: (
            "We are open this week and committed to providing excellent service "
            "to our valued customers. Visit us in person or contact us today "
            "to learn how we can help you."
        ),
    },
}


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass(frozen=True)
class ContentPiece:
    """
    A single generated content piece for one platform.

    Attributes:
        platform:       One of PLATFORM_INSTAGRAM, PLATFORM_WHATSAPP,
                        PLATFORM_GOOGLE.
        text:           The generated post copy.
        hashtags:       Extracted hashtag list (Instagram only, others empty).
        char_count:     Length of text in characters.
        source:         "ai" or "fallback".
        platform_label: Human-readable platform name for display.
    """
    platform: str
    text: str
    hashtags: list[str]
    char_count: int
    source: str

    @property
    def platform_label(self) -> str:
        labels = {
            PLATFORM_INSTAGRAM: "📸 Instagram",
            PLATFORM_WHATSAPP:  "💬 WhatsApp Status",
            PLATFORM_GOOGLE:    "🔍 Google Business Post",
        }
        return labels.get(self.platform, self.platform)

    @property
    def text_without_hashtags(self) -> str:
        """Caption text with hashtags stripped (for platforms that don't use them)."""
        return re.sub(r"#\w+", "", self.text).strip()

    def to_whatsapp_block(self) -> str:
        """Format this content piece as a WhatsApp message block."""
        lines = [
            f"*{self.platform_label}*",
            "",
            self.text,
        ]
        if self.source == "fallback":
            lines.append("")
            lines.append("_💡 Customise this with your specific offer or news._")
        return "\n".join(lines)


@dataclass
class ContentGenerationResult:
    """
    Complete weekly content package for a business.

    Attributes:
        business_id:    Business UUID.
        business_name:  Business name.
        week_label:     ISO week label e.g. "Week 42, 2024".
        generated_at:   Date generated.
        pieces:         List of ContentPiece — one per platform.
        used_fallback:  True if OpenAI was unavailable.
        context_used:   Summary of the context injected into the prompt.
    """
    business_id: str
    business_name: str
    week_label: str
    generated_at: date
    pieces: list[ContentPiece]
    used_fallback: bool
    context_used: dict = field(default_factory=dict)

    @property
    def instagram_piece(self) -> Optional[ContentPiece]:
        return next((p for p in self.pieces if p.platform == PLATFORM_INSTAGRAM), None)

    @property
    def whatsapp_piece(self) -> Optional[ContentPiece]:
        return next((p for p in self.pieces if p.platform == PLATFORM_WHATSAPP), None)

    @property
    def google_piece(self) -> Optional[ContentPiece]:
        return next((p for p in self.pieces if p.platform == PLATFORM_GOOGLE), None)

    def to_whatsapp_messages(self) -> list[str]:
        """
        Build WhatsApp delivery messages for the full content package.

        Returns one message per platform piece plus an intro header,
        keeping each message under WhatsApp's character limit.
        """
        intro = (
            f"📣 *Your Weekly Content Pack — {self.week_label}*\n"
            f"Here are 3 ready-to-post pieces for {self.business_name}.\n"
            f"Copy, personalise, and post! 🚀"
        )
        messages = [intro]
        for piece in self.pieces:
            messages.append(piece.to_whatsapp_block())
        return messages

    def __str__(self) -> str:
        return (
            f"ContentGenerationResult("
            f"business={self.business_name!r} "
            f"week={self.week_label} "
            f"pieces={len(self.pieces)} "
            f"fallback={self.used_fallback})"
        )


@dataclass(frozen=True)
class ContentGenerationError:
    """Returned when content generation fails entirely."""
    business_id: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return (
            f"ContentGenerationError("
            f"business={self.business_id} "
            f"reason={self.reason}: {self.detail})"
        )


# ==============================================================================
# Content Generation Service
# ==============================================================================

class ContentGenerationService:
    """
    Generates weekly social media content for local businesses.

    Stateless — safe to share a single instance.

    Usage:
        service = ContentGenerationService()

        result = await service.generate_weekly_content(
            business_id="uuid",
            business_name="Raj Restaurant",
            business_type="Restaurant",
            avg_rating=4.3,
            dominant_sentiment="positive",
            top_praised_themes=["biryani", "fast service", "fresh food"],
            special_context="We now offer home delivery",
        )

        if isinstance(result, ContentGenerationError):
            logger.warning(str(result))
        else:
            for msg in result.to_whatsapp_messages():
                await whatsapp_service.send(business_id, msg)
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._prompt_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_weekly_content(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        avg_rating: float,
        dominant_sentiment: str = "positive",
        top_praised_themes: Optional[list[str]] = None,
        special_context: Optional[str] = None,
        generation_date: Optional[date] = None,
    ) -> ContentGenerationResult | ContentGenerationError:
        """
        Generate one week's social media content for a business.

        Produces 3 content pieces:
          1. Instagram caption (with hashtags)
          2. WhatsApp status (short + CTA)
          3. Google Business post (professional + SEO-aware)

        Always returns a valid result — falls back to template library
        if OpenAI is unavailable.

        Args:
            business_id:          Business UUID.
            business_name:        Business name for display and prompt context.
            business_type:        Business type e.g. "Restaurant", "Salon".
            avg_rating:           Current average rating (1.0–5.0).
            dominant_sentiment:   "positive", "negative", or "neutral" —
                                  derived from recent review analysis.
            top_praised_themes:   Top 5 themes customers praise in reviews
                                  e.g. ["biryani", "fast delivery", "clean"].
            special_context:      Optional free-text context the business
                                  registered e.g. "We now do home delivery" or
                                  "We have a new chef this month".
                                  Max 200 chars used.
            generation_date:      Override today's date (for testing).

        Returns:
            ContentGenerationResult or ContentGenerationError. Never raises.
        """
        today = generation_date or date.today()
        week_label = get_current_week_label(today)
        month_label = get_current_month_label(today)

        log_extra = {
            "service": ServiceName.CONTENT,
            "business_id": business_id,
            "business_type": business_type,
            "avg_rating": avg_rating,
            "week": week_label,
        }

        context_used = {
            "business_type": business_type,
            "avg_rating": avg_rating,
            "dominant_sentiment": dominant_sentiment,
            "top_praised_themes": (top_praised_themes or [])[:5],
            "week_label": week_label,
            "month_label": month_label,
            "has_special_context": bool(special_context),
        }

        # ----------------------------------------------------------
        # Try OpenAI path
        # ----------------------------------------------------------
        try:
            pieces = await self._generate_with_openai(
                business_name=business_name,
                business_type=business_type,
                avg_rating=avg_rating,
                dominant_sentiment=dominant_sentiment,
                top_praised_themes=top_praised_themes or [],
                special_context=special_context,
                week_label=week_label,
                month_label=month_label,
                log_extra=log_extra,
            )

            logger.info(
                "Content generated via OpenAI",
                extra={**log_extra, "pieces": len(pieces)},
            )

            return ContentGenerationResult(
                business_id=business_id,
                business_name=business_name,
                week_label=week_label,
                generated_at=today,
                pieces=pieces,
                used_fallback=False,
                context_used=context_used,
            )

        except Exception as exc:
            logger.error(
                "Content generation OpenAI call failed — using template fallback",
                extra={
                    **log_extra,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

        # ----------------------------------------------------------
        # Fallback: template-based content
        # ----------------------------------------------------------
        pieces = _build_fallback_content(
            business_type=business_type,
            business_name=business_name,
        )

        logger.info(
            "Content generated via template fallback",
            extra={**log_extra, "pieces": len(pieces)},
        )

        return ContentGenerationResult(
            business_id=business_id,
            business_name=business_name,
            week_label=week_label,
            generated_at=today,
            pieces=pieces,
            used_fallback=True,
            context_used=context_used,
        )

    # ------------------------------------------------------------------
    # OpenAI generation
    # ------------------------------------------------------------------

    @with_openai_retry
    async def _generate_with_openai(
        self,
        business_name: str,
        business_type: str,
        avg_rating: float,
        dominant_sentiment: str,
        top_praised_themes: list[str],
        special_context: Optional[str],
        week_label: str,
        month_label: str,
        log_extra: dict,
    ) -> list[ContentPiece]:
        """
        Call OpenAI to generate content for all three platforms.

        Returns a JSON array of 3 objects, one per platform.
        Parses and validates the response before returning.

        Raises:
            Exception: Any OpenAI or parse error (caller triggers fallback).
        """
        system_prompt = self._load_prompt()
        user_prompt = _build_content_prompt(
            business_name=business_name,
            business_type=business_type,
            avg_rating=avg_rating,
            dominant_sentiment=dominant_sentiment,
            top_praised_themes=top_praised_themes,
            special_context=special_context,
            week_label=week_label,
            month_label=month_label,
        )

        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=_CONTENT_MAX_TOKENS,
            temperature=_CONTENT_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS,
        )

        raw = response.choices[0].message.content or ""
        pieces = _parse_content_response(raw.strip())

        logger.debug(
            "OpenAI content raw response received",
            extra={**log_extra, "raw_chars": len(raw), "pieces_parsed": len(pieces)},
        )

        return pieces

    def _load_prompt(self) -> str:
        """Load content_generation_prompt.txt from disk (cached after first read)."""
        if self._prompt_cache:
            return self._prompt_cache

        prompt_path = _PROMPTS_DIR / _CONTENT_PROMPT_FILE
        if not prompt_path.exists():
            logger.warning(
                "content_generation_prompt.txt not found — using inline system prompt",
                extra={"service": ServiceName.CONTENT},
            )
            self._prompt_cache = _INLINE_SYSTEM_PROMPT
            return self._prompt_cache

        self._prompt_cache = prompt_path.read_text(encoding="utf-8").strip()
        return self._prompt_cache


# ==============================================================================
# Inline system prompt — used when prompt file is absent
# ==============================================================================

_INLINE_SYSTEM_PROMPT = (
    "You are a social media content writer for small local businesses in India. "
    "Generate ready-to-post content for three platforms: Instagram, WhatsApp Status, "
    "and Google Business Post. "
    "Respond ONLY with a valid JSON array containing exactly 3 objects. "
    "No preamble, no explanation, no markdown code fences. "
    "Each object must have exactly these keys: "
    '"platform" (one of: "instagram", "whatsapp_status", "google_post"), '
    '"text" (the post copy as a string). '
    "Instagram: engaging caption with relevant hashtags at the end (max 10 hashtags), "
    "2–5 sentences, include an emoji. "
    "WhatsApp Status: short (max 2 sentences), conversational, include a CTA and emoji. "
    "Google Post: professional, no hashtags, 2–3 sentences, keyword-rich for local SEO. "
    "All content must be in English. Do not include placeholder text."
)


# ==============================================================================
# Prompt builder
# ==============================================================================

def _build_content_prompt(
    business_name: str,
    business_type: str,
    avg_rating: float,
    dominant_sentiment: str,
    top_praised_themes: list[str],
    special_context: Optional[str],
    week_label: str,
    month_label: str,
) -> str:
    """
    Build the user-role prompt for content generation.

    Prompt safety (guardrails §9):
    Only aggregated, non-sensitive business context is included.
    No reviewer names, contact details, or financial data.
    """
    lines = [
        f"Business name: {business_name}",
        f"Business type: {business_type}",
        f"Google rating: {avg_rating:.1f} / 5.0",
        f"Customer sentiment this week: {dominant_sentiment}",
        f"Current week: {week_label}",
        f"Current month: {month_label}",
    ]

    if top_praised_themes:
        themes_str = ", ".join(top_praised_themes[:5])
        lines.append(f"What customers praise: {themes_str}")

    if special_context:
        # Cap special context to 200 chars to control prompt size
        safe_context = special_context.strip()[:200]
        lines.append(f"Special note from business: {safe_context}")

    lines.append(
        "\nGenerate 3 ready-to-post pieces (Instagram, WhatsApp Status, "
        "Google Post) for this business this week. "
        "Make the content feel fresh, local, and specific to this business type. "
        "Respond with a JSON array only."
    )

    return "\n".join(lines)


# ==============================================================================
# Response parser
# ==============================================================================

def _parse_content_response(raw: str) -> list[ContentPiece]:
    """
    Parse OpenAI's JSON array response into ContentPiece objects.

    Handles:
      - Markdown code fences wrapping the JSON
      - Items in wrong order (re-ordered by platform)
      - Extra or missing platform keys (skipped gracefully)
      - Text too long (truncated to _MAX_PIECE_CHARS)

    Args:
        raw: Raw text from OpenAI.

    Returns:
        list[ContentPiece]: Up to 3 pieces, one per platform.

    Raises:
        ValueError: If JSON cannot be parsed at all.
    """
    cleaned = raw.strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Content response is not valid JSON: {raw[:200]!r}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Content response is not a JSON array: {type(data).__name__}"
        )

    pieces: list[ContentPiece] = []
    seen_platforms: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        platform = str(item.get("platform", "")).strip().lower()
        if platform not in ALL_PLATFORMS:
            continue

        if platform in seen_platforms:
            continue   # skip duplicate platforms
        seen_platforms.add(platform)

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        # Truncate if excessively long
        if len(text) > _MAX_PIECE_CHARS:
            text = text[:_MAX_PIECE_CHARS - 3].rstrip() + "..."

        hashtags = _extract_hashtags(text) if platform == PLATFORM_INSTAGRAM else []

        pieces.append(ContentPiece(
            platform=platform,
            text=text,
            hashtags=hashtags,
            char_count=len(text),
            source="ai",
        ))

    if not pieces:
        raise ValueError(
            f"Content response parsed but produced no usable pieces: {raw[:200]!r}"
        )

    # Ensure consistent ordering: Instagram → WhatsApp → Google
    order = {p: i for i, p in enumerate(ALL_PLATFORMS)}
    pieces.sort(key=lambda p: order.get(p.platform, 99))

    return pieces


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _extract_hashtags(text: str) -> list[str]:
    """
    Extract hashtag strings from post text.

    Args:
        text: Post copy that may contain #hashtags.

    Returns:
        list[str]: List of hashtag strings including the # prefix,
                   capped at _MAX_HASHTAGS.
    """
    return re.findall(r"#\w+", text)[:_MAX_HASHTAGS]


def _build_fallback_content(
    business_type: str,
    business_name: str,
) -> list[ContentPiece]:
    """
    Build template-based content pieces when OpenAI is unavailable.

    Looks up the template library by business type with a fuzzy key match.
    Injects the business name into each platform template where possible.

    Args:
        business_type: Business type string.
        business_name: Business name for personalisation.

    Returns:
        list[ContentPiece]: 3 content pieces, one per platform.
    """
    type_key = business_type.lower().strip()
    if type_key not in _FALLBACK_TEMPLATES:
        for known_key in _FALLBACK_TEMPLATES:
            if known_key in type_key or type_key in known_key:
                type_key = known_key
                break
        else:
            type_key = "default"

    templates = _FALLBACK_TEMPLATES[type_key]
    pieces: list[ContentPiece] = []

    for platform in ALL_PLATFORMS:
        text = templates.get(platform, _FALLBACK_TEMPLATES["default"][platform])

        # Personalise with business name if the template doesn't already have it
        if business_name.lower() not in text.lower() and platform == PLATFORM_GOOGLE:
            text = f"Welcome to {business_name}. " + text

        hashtags = _extract_hashtags(text) if platform == PLATFORM_INSTAGRAM else []

        pieces.append(ContentPiece(
            platform=platform,
            text=text,
            hashtags=hashtags,
            char_count=len(text),
            source="fallback",
        ))

    return pieces