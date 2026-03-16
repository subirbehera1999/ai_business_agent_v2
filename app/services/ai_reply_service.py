# ==============================================================================
# File: app/services/ai_reply_service.py
# Purpose: Generates AI-powered reply text for Google reviews using OpenAI.
#
#          This is the core value-delivery service of the platform. Every
#          new review that passes validation triggers this service to produce
#          a context-aware, sentiment-appropriate reply that the business
#          owner can post (or the system posts automatically) to Google.
#
#          Prompt selection logic (from product spec and review_validator.py):
#            POSITIVE  → app/prompts/positive_review_reply_prompt.txt
#            NEGATIVE  → app/prompts/negative_review_reply_prompt.txt
#            NEUTRAL   → app/prompts/neutral_review_reply_prompt.txt
#
#          Guardrails enforced in this file:
#            § 1  Idempotency    — skips re-generation if reply already exists
#            § 4  Rate Limiting  — checks ai_reply limit before calling OpenAI
#            § 5  Usage Tracking — increments ai_replies_generated on success
#            § 3  Retry Policy   — @with_openai_retry (3 attempts, backoff)
#            § 9  Prompt Safety  — no sensitive data in prompt
#            §10  Prompt Size    — review text capped before injection
#
#          Never raises to the scheduler caller — returns ReplyResult which
#          carries a success/failure flag so the scheduler can record outcome.
# ==============================================================================

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from app.config.constants import (
    PROMPT_NEGATIVE_REVIEW_REPLY,
    PROMPT_NEUTRAL_REVIEW_REPLY,
    PROMPT_POSITIVE_REVIEW_REPLY,
    ReviewSentiment,
    ServiceName,
)
from app.config.settings import get_settings
from app.repositories.review_repository import ReviewRepository
from app.repositories.usage_repository import UsageRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.services.sentiment_service import SentimentResult, SentimentService
from app.utils.idempotency_utils import make_review_reply_key
from app.utils.rate_limiter import enforce_rate_limit
from app.utils.retry_utils import with_openai_retry
from app.utils.usage_tracker import (
    track_ai_reply_failed,
    track_ai_reply_generated,
    track_openai_api_error,
)

logger = logging.getLogger(ServiceName.AI_REPLY)
settings = get_settings()

# Prompts directory — relative to this file's location
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Maximum characters of review text injected into the reply prompt.
# Tighter than the 800-char DB cap — keeps prompt tokens predictable.
_MAX_REVIEW_TEXT_IN_PROMPT = 500

# Maximum tokens for the generated reply — enough for a full paragraph,
# not so large that runaway completions inflate cost.
_REPLY_MAX_TOKENS = 250

# Temperature for reply generation — slight creativity, still professional
_REPLY_TEMPERATURE = 0.7


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass(frozen=True)
class ReplyResult:
    """
    Immutable outcome of a single AI reply generation attempt.

    Attributes:
        success:         True if a reply was generated and saved.
        reply_text:      The generated reply (None if failed or skipped).
        review_id:       The review this result corresponds to.
        business_id:     Business that owns the review.
        sentiment:       Sentiment used to select the prompt template.
        skipped:         True if reply was skipped (idempotency / rate limit).
        skip_reason:     Human-readable reason if skipped.
        error:           Error message if success=False and not skipped.
        used_fallback_sentiment: True if sentiment came from star-rating heuristic.
    """

    success: bool
    review_id: str
    business_id: str
    sentiment: str
    reply_text: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    used_fallback_sentiment: bool = False

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"ReplyResult(skipped=True reason={self.skip_reason!r} "
                f"review={self.review_id} business={self.business_id})"
            )
        status = "success" if self.success else "failed"
        return (
            f"ReplyResult(status={status} sentiment={self.sentiment} "
            f"review={self.review_id} business={self.business_id})"
        )


# ==============================================================================
# AI Reply Service
# ==============================================================================

class AIReplyService:
    """
    Generates sentiment-appropriate AI reply text for Google reviews.

    Orchestrates the full reply pipeline:
      1. Idempotency check     — skip if reply already generated
      2. Rate limit check      — skip if daily ai_reply limit exceeded
      3. Sentiment resolution  — use provided result or run analysis
      4. Prompt loading        — load template from app/prompts/
      5. OpenAI call           — generate reply with retry
      6. Reply persistence     — save reply to database via repository
      7. Usage tracking        — increment ai_replies_generated counter

    Usage:
        service = AIReplyService(
            review_repo=review_repo,
            usage_repo=usage_repo,
            subscription_repo=subscription_repo,
            sentiment_service=sentiment_service,
        )

        result = await service.generate_reply(
            db=db,
            review_id="uuid",
            business_id="uuid",
            review_text="Food was cold and service was slow.",
            star_rating=2,
            reviewer_name="Priya S.",
            business_name="Raj Restaurant",
            business_type="Restaurant",
            sentiment_result=sentiment_result,   # optional pre-computed
        )
    """

    def __init__(
        self,
        review_repo: ReviewRepository,
        usage_repo: UsageRepository,
        subscription_repo: SubscriptionRepository,
        sentiment_service: SentimentService,
    ) -> None:
        self._review_repo = review_repo
        self._usage_repo = usage_repo
        self._subscription_repo = subscription_repo
        self._sentiment_service = sentiment_service
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_reply(
        self,
        db,
        review_id: str,
        business_id: str,
        review_text: Optional[str],
        star_rating: int,
        reviewer_name: str,
        business_name: str,
        business_type: str = "Local Business",
        sentiment_result: Optional[SentimentResult] = None,
    ) -> ReplyResult:
        """
        Generate and persist an AI reply for a single review.

        This method is the single entry point for the reply pipeline.
        It is safe to call multiple times for the same review — idempotency
        is enforced at step 1.

        Args:
            db:                Database session (AsyncSession).
            review_id:         UUID of the review record.
            business_id:       UUID of the owning business.
            review_text:       Raw review text (may be None for rating-only).
            star_rating:       Star rating 1–5.
            reviewer_name:     Name of the reviewer (used in prompt context).
            business_name:     Business name (used in prompt context).
            business_type:     Type of business for prompt context.
            sentiment_result:  Pre-computed SentimentResult from sentiment_service.
                               If None, this service runs sentiment analysis.

        Returns:
            ReplyResult: Always returns — never raises.
        """
        log_extra = {
            "service": ServiceName.AI_REPLY,
            "business_id": business_id,
            "review_id": review_id,
            "star_rating": star_rating,
        }

        # ----------------------------------------------------------
        # Step 1: Idempotency — skip if reply already generated
        # ----------------------------------------------------------
        idempotency_key = make_review_reply_key(business_id, review_id)
        existing_review = await self._review_repo.get_by_id(db, review_id)

        if existing_review and existing_review.ai_reply:
            logger.info(
                "AI reply already exists — skipping (idempotent)",
                extra={**log_extra, "idempotency_key": idempotency_key},
            )
            return ReplyResult(
                success=True,
                review_id=review_id,
                business_id=business_id,
                sentiment=existing_review.sentiment or ReviewSentiment.NEUTRAL,
                reply_text=existing_review.ai_reply,
                skipped=True,
                skip_reason="reply_already_exists",
            )

        # ----------------------------------------------------------
        # Step 2: Rate limit check
        # ----------------------------------------------------------
        rate_result = await enforce_rate_limit(
            db=db,
            business_id=business_id,
            metric="ai_replies",
            usage_repo=self._usage_repo,
            subscription_repo=self._subscription_repo,
        )

        if not rate_result.allowed:
            logger.warning(
                "AI reply blocked — daily limit reached",
                extra={
                    **log_extra,
                    "limit": rate_result.limit,
                    "current": rate_result.current_count,
                    "remaining": rate_result.remaining,
                },
            )
            return ReplyResult(
                success=False,
                review_id=review_id,
                business_id=business_id,
                sentiment=ReviewSentiment.NEUTRAL,
                skipped=True,
                skip_reason=f"rate_limit_exceeded:{rate_result.limit}_per_day",
            )

        # ----------------------------------------------------------
        # Step 3: Sentiment resolution
        # ----------------------------------------------------------
        if sentiment_result is None:
            logger.debug(
                "No pre-computed sentiment — running analysis",
                extra=log_extra,
            )
            sentiment_result = await self._sentiment_service.analyze(
                review_text=review_text,
                star_rating=star_rating,
                business_id=business_id,
                business_type=business_type,
                review_id=review_id,
            )

        sentiment = sentiment_result.sentiment
        log_extra["sentiment"] = sentiment
        log_extra["sentiment_score"] = sentiment_result.score

        # Determine which prompt file is being used (for auditing and persistence)
        prompt_filename = _sentiment_to_prompt_filename(sentiment)

        # ----------------------------------------------------------
        # Step 4: Load prompt template
        # ----------------------------------------------------------
        try:
            prompt_template = self._load_prompt(sentiment)
        except FileNotFoundError as exc:
            logger.error(
                "Prompt template file not found",
                extra={**log_extra, "error": str(exc)},
            )
            await track_ai_reply_failed(db, business_id, self._usage_repo)
            return ReplyResult(
                success=False,
                review_id=review_id,
                business_id=business_id,
                sentiment=sentiment,
                error=f"prompt_not_found:{exc}",
                used_fallback_sentiment=sentiment_result.used_fallback,
            )

        # ----------------------------------------------------------
        # Step 5: Build prompt and call OpenAI
        # ----------------------------------------------------------
        filled_prompt = _fill_prompt_template(
            template=prompt_template,
            business_name=business_name,
            business_type=business_type,
            reviewer_name=reviewer_name,
            star_rating=star_rating,
            review_text=review_text or "",
            sentiment=sentiment,
        )

        try:
            reply_text = await self._call_openai(
                prompt=filled_prompt,
                log_extra=log_extra,
            )
        except Exception as exc:
            logger.error(
                "AI reply generation failed — OpenAI call exhausted retries",
                extra={**log_extra, "error": str(exc), "error_type": type(exc).__name__},
            )
            await track_openai_api_error(db, business_id, self._usage_repo)
            await track_ai_reply_failed(db, business_id, self._usage_repo)
            return ReplyResult(
                success=False,
                review_id=review_id,
                business_id=business_id,
                sentiment=sentiment,
                error=str(exc),
                used_fallback_sentiment=sentiment_result.used_fallback,
            )

        # ----------------------------------------------------------
        # Step 6: Persist reply to database
        # ----------------------------------------------------------
        try:
            await self._review_repo.save_ai_reply(
                db=db,
                review_id=review_id,
                ai_reply=reply_text,
                prompt_used=prompt_filename,
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            logger.error(
                "Failed to persist AI reply to database",
                extra={**log_extra, "error": str(exc)},
            )
            await track_ai_reply_failed(db, business_id, self._usage_repo)
            return ReplyResult(
                success=False,
                review_id=review_id,
                business_id=business_id,
                sentiment=sentiment,
                error=f"db_persist_failed:{exc}",
                used_fallback_sentiment=sentiment_result.used_fallback,
            )

        # ----------------------------------------------------------
        # Step 7: Usage tracking — non-fatal if this fails
        # ----------------------------------------------------------
        await track_ai_reply_generated(db, business_id, self._usage_repo)

        logger.info(
            "AI reply generated and saved successfully",
            extra={
                **log_extra,
                "reply_length": len(reply_text),
                "used_fallback_sentiment": sentiment_result.used_fallback,
            },
        )

        return ReplyResult(
            success=True,
            review_id=review_id,
            business_id=business_id,
            sentiment=sentiment,
            reply_text=reply_text,
            used_fallback_sentiment=sentiment_result.used_fallback,
        )

    # ------------------------------------------------------------------
    # Prompt loading with in-memory cache
    # ------------------------------------------------------------------

    def _load_prompt(self, sentiment: str) -> str:
        """
        Load the prompt template for the given sentiment.

        Templates are loaded from disk once and cached in memory for the
        lifetime of the service instance. This avoids repeated file I/O
        during batch review processing.

        Args:
            sentiment: ReviewSentiment constant.

        Returns:
            str: Raw prompt template text with placeholder variables.

        Raises:
            FileNotFoundError: If the prompt file does not exist on disk.
        """
        if sentiment in self._prompt_cache:
            return self._prompt_cache[sentiment]

        filename = _sentiment_to_prompt_filename(sentiment)
        prompt_path = _PROMPTS_DIR / filename

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Expected at app/prompts/{filename}"
            )

        template = prompt_path.read_text(encoding="utf-8").strip()

        if not template:
            raise FileNotFoundError(
                f"Prompt file is empty: {prompt_path}"
            )

        self._prompt_cache[sentiment] = template
        logger.debug(
            "Prompt template loaded from disk",
            extra={
                "service": ServiceName.AI_REPLY,
                "sentiment": sentiment,
                "filename": filename,
                "chars": len(template),
            },
        )
        return template

    def clear_prompt_cache(self) -> None:
        """
        Clear the in-memory prompt cache.

        Call this after updating prompt files on disk to force a reload
        on the next generate_reply() call. Useful in development or after
        a hot-reload prompt update.
        """
        self._prompt_cache.clear()
        logger.info(
            "Prompt cache cleared",
            extra={"service": ServiceName.AI_REPLY},
        )

    # ------------------------------------------------------------------
    # OpenAI call with retry
    # ------------------------------------------------------------------

    @with_openai_retry
    async def _call_openai(self, prompt: str, log_extra: dict) -> str:
        """
        Call OpenAI to generate the review reply.

        Decorated with @with_openai_retry (3 attempts, exponential backoff).
        Uses the chat completions endpoint with the system and user roles
        separated to keep the instruction layer distinct from the content.

        Args:
            prompt:     Fully-filled prompt string ready for OpenAI.
            log_extra:  Structured log context dict.

        Returns:
            str: Cleaned reply text.

        Raises:
            Exception: Propagated to caller after all retries exhausted.
        """
        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=_REPLY_MAX_TOKENS,
            temperature=_REPLY_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional business representative "
                        "writing a reply to a customer review. "
                        "Write a concise, genuine, and professional response. "
                        "Do not use placeholder text. "
                        "Do not include subject lines or email formatting. "
                        "Reply in plain text only — no bullet points, no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS,
        )

        raw_reply = response.choices[0].message.content or ""
        cleaned = _clean_reply_text(raw_reply)

        logger.debug(
            "OpenAI reply generated",
            extra={
                **log_extra,
                "reply_chars": len(cleaned),
                "finish_reason": response.choices[0].finish_reason,
            },
        )

        if not cleaned:
            raise ValueError("OpenAI returned an empty reply after cleaning")

        return cleaned


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _sentiment_to_prompt_filename(sentiment: str) -> str:
    """
    Map a sentiment constant to its prompt template filename.

    Args:
        sentiment: ReviewSentiment constant.

    Returns:
        str: Filename from app/config/constants.py prompt filename constants.
    """
    mapping = {
        ReviewSentiment.POSITIVE: PROMPT_POSITIVE_REVIEW_REPLY,
        ReviewSentiment.NEGATIVE: PROMPT_NEGATIVE_REVIEW_REPLY,
        ReviewSentiment.NEUTRAL:  PROMPT_NEUTRAL_REVIEW_REPLY,
    }
    return mapping.get(sentiment, PROMPT_NEUTRAL_REVIEW_REPLY)


def _fill_prompt_template(
    template: str,
    business_name: str,
    business_type: str,
    reviewer_name: str,
    star_rating: int,
    review_text: str,
    sentiment: str,
) -> str:
    """
    Inject context variables into a prompt template.

    Template placeholders use double-brace format: {{VARIABLE_NAME}}
    This avoids conflicts with Python's str.format() and f-string syntax
    when prompt files contain curly braces for other purposes.

    Placeholders supported in prompt files:
        {{BUSINESS_NAME}}
        {{BUSINESS_TYPE}}
        {{REVIEWER_NAME}}
        {{STAR_RATING}}
        {{REVIEW_TEXT}}
        {{SENTIMENT}}

    Args:
        template:      Raw template string from prompt file.
        business_name: Name of the business.
        business_type: Type of business (e.g. "Restaurant").
        reviewer_name: Name of the reviewer.
        star_rating:   Star rating integer 1–5.
        review_text:   Review text (will be truncated for prompt safety).
        sentiment:     Sentiment label string.

    Returns:
        str: Filled prompt ready for OpenAI.
    """
    # Truncate review text before injecting — prompt size control §10
    safe_review_text = (
        review_text[:_MAX_REVIEW_TEXT_IN_PROMPT - 3].rstrip() + "..."
        if len(review_text) > _MAX_REVIEW_TEXT_IN_PROMPT
        else review_text
    )

    replacements = {
        "{{BUSINESS_NAME}}":  business_name,
        "{{BUSINESS_TYPE}}":  business_type,
        "{{REVIEWER_NAME}}":  reviewer_name,
        "{{STAR_RATING}}":    str(star_rating),
        "{{REVIEW_TEXT}}":    safe_review_text,
        "{{SENTIMENT}}":      sentiment,
    }

    filled = template
    for placeholder, value in replacements.items():
        filled = filled.replace(placeholder, value)

    return filled.strip()


def _clean_reply_text(raw: str) -> str:
    """
    Clean and normalise the raw reply text from OpenAI.

    Removes:
      - Leading/trailing whitespace
      - Quotation marks wrapping the entire reply
      - Markdown bold/italic markers
      - Excessive blank lines (more than one consecutive blank line)

    Args:
        raw: Raw text from OpenAI completion.

    Returns:
        str: Cleaned reply text.
    """
    text = raw.strip()

    # Remove wrapping quotes (model sometimes wraps entire reply in "...")
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1].strip()

    # Remove markdown bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)

    # Collapse more than one consecutive blank line into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Alias used by review_monitor.py
AiReplyService = AIReplyService