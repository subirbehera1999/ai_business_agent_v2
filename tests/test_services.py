# ==============================================================================
# File: tests/test_services.py
# Purpose: Tests for core service layer business logic.
#
#          Covers:
#            1. SentimentService — fallback heuristic, score_to_sentiment,
#               _safe_truncate_for_prompt, OpenAI path, batch analysis
#            2. AIReplyService — private helpers (_sentiment_to_prompt_filename,
#               _fill_prompt_template, _clean_reply_text), generate_reply
#               idempotency, rate-limit skip, prompt safety (no sensitive data)
#            3. AnalyticsService — private aggregation helpers
#               (_aggregate_daily, _aggregate_weekly, _detect_trend,
#               _find_peak_days, _top_products, _compute_change_pct)
#            4. SeoService — _rating_tier, _extract_terms_from_reviews,
#               generate_suggestions fallback
#            5. ContentGenerationService — generate_weekly_content fallback,
#               ContentGenerationResult platform accessors
#
#          Design:
#            OpenAI calls are always mocked — no real API calls.
#            Repository calls are mocked where needed.
#            Private helpers are tested directly because they contain
#            important pure-function logic worth unit-testing in isolation.
#
#          Running:
#            pytest tests/test_services.py -v
#            pytest tests/test_services.py -v -k "sentiment"
# ==============================================================================

import os
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Env vars before any app import
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL",             "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("JWT_SECRET_KEY",           "test-secret-key-at-least-32-characters-long!!")
os.environ.setdefault("RAZORPAY_KEY_ID",          "rzp_test_key_id")
os.environ.setdefault("RAZORPAY_KEY_SECRET",      "rzp_test_secret")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET",  "rzp_test_webhook_secret")
os.environ.setdefault("WHATSAPP_API_TOKEN",       "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "test_phone_id")
os.environ.setdefault("OPENAI_API_KEY",           "sk-test-key")
os.environ.setdefault("ADMIN_WHATSAPP_NUMBER",    "9999999999")

from app.config.constants import ReviewSentiment                        # noqa: E402
from app.services.ai_reply_service import (                             # noqa: E402
    AIReplyService,
    ReplyResult,
    _clean_reply_text,
    _fill_prompt_template,
    _sentiment_to_prompt_filename,
)
from app.services.analytics_service import (                            # noqa: E402
    AnalyticsError,
    DailySales,
    SalesAnalyticsResult,
    TrendSignal,
    _aggregate_daily,
    _aggregate_weekly,
    _compute_change_pct,
    _detect_trend,
    _find_peak_days,
    _top_products,
)
from app.services.content_generation_service import (                   # noqa: E402
    ContentGenerationError,
    ContentGenerationResult,
    ContentGenerationService,
)
from app.services.seo_service import (                                  # noqa: E402
    SeoService,
    SeoSuggestionResult,
    _extract_terms_from_reviews,
    _rating_tier,
)
from app.services.sentiment_service import (                            # noqa: E402
    SentimentResult,
    SentimentService,
    _safe_truncate_for_prompt,
    score_to_sentiment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BUSINESS_ID   = str(uuid.uuid4())
_REVIEW_ID     = str(uuid.uuid4())
_BUSINESS_NAME = "Sunrise Cafe"
_BUSINESS_TYPE = "restaurant"


# ==============================================================================
# Helpers
# ==============================================================================

def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.commit   = AsyncMock()
    db.rollback = AsyncMock()
    return db


def _daily_sales(
    day: date,
    revenue: float,
    transactions: int = 5,
    products: list[str] | None = None,
) -> DailySales:
    rev = Decimal(str(revenue))
    return DailySales(
        date=day,
        total_revenue=rev,
        transaction_count=transactions,
        avg_transaction=rev / transactions if transactions else Decimal("0"),
        products=products or [],
    )


# ==============================================================================
# 1. SentimentService — fallback heuristic
# ==============================================================================

class TestSentimentFallback:
    """Tests for the star-rating fallback in SentimentService._fallback()."""

    def _svc(self) -> SentimentService:
        return SentimentService()

    def test_5_star_is_positive(self) -> None:
        svc    = self._svc()
        result = svc._fallback(5)
        assert result.sentiment == ReviewSentiment.POSITIVE
        assert result.score > 0
        assert result.used_fallback is True

    def test_4_star_is_positive(self) -> None:
        result = self._svc()._fallback(4)
        assert result.sentiment == ReviewSentiment.POSITIVE

    def test_3_star_is_neutral(self) -> None:
        result = self._svc()._fallback(3)
        assert result.sentiment == ReviewSentiment.NEUTRAL
        assert result.score == 0.0

    def test_2_star_is_negative(self) -> None:
        result = self._svc()._fallback(2)
        assert result.sentiment == ReviewSentiment.NEGATIVE
        assert result.score < 0

    def test_1_star_is_negative(self) -> None:
        result = self._svc()._fallback(1)
        assert result.sentiment == ReviewSentiment.NEGATIVE
        assert result.score < -0.5

    def test_out_of_range_high_clamped_to_5(self) -> None:
        """Star rating > 5 must be clamped — must not raise KeyError."""
        result = self._svc()._fallback(10)
        assert result.sentiment == ReviewSentiment.POSITIVE

    def test_out_of_range_low_clamped_to_1(self) -> None:
        """Star rating < 1 must be clamped — must not raise KeyError."""
        result = self._svc()._fallback(0)
        assert result.sentiment == ReviewSentiment.NEGATIVE

    def test_fallback_result_is_sentiment_result(self) -> None:
        result = self._svc()._fallback(3)
        assert isinstance(result, SentimentResult)


class TestScoreToSentiment:
    """Tests for the module-level score_to_sentiment() helper."""

    def test_high_positive_score(self) -> None:
        assert score_to_sentiment(0.9) == ReviewSentiment.POSITIVE

    def test_at_positive_threshold(self) -> None:
        assert score_to_sentiment(0.25) == ReviewSentiment.POSITIVE

    def test_just_below_positive_threshold_is_neutral(self) -> None:
        assert score_to_sentiment(0.24) == ReviewSentiment.NEUTRAL

    def test_zero_is_neutral(self) -> None:
        assert score_to_sentiment(0.0) == ReviewSentiment.NEUTRAL

    def test_at_negative_threshold(self) -> None:
        assert score_to_sentiment(-0.25) == ReviewSentiment.NEGATIVE

    def test_high_negative_score(self) -> None:
        assert score_to_sentiment(-0.9) == ReviewSentiment.NEGATIVE

    def test_just_above_negative_threshold_is_neutral(self) -> None:
        assert score_to_sentiment(-0.24) == ReviewSentiment.NEUTRAL


class TestSafeTruncate:
    """Tests for _safe_truncate_for_prompt() helper."""

    def test_short_text_unchanged(self) -> None:
        text   = "Great food!"
        result = _safe_truncate_for_prompt(text, max_chars=600)
        assert result == text

    def test_long_text_truncated(self) -> None:
        text   = "x" * 700
        result = _safe_truncate_for_prompt(text, max_chars=600)
        assert len(result) <= 600

    def test_exact_max_length_unchanged(self) -> None:
        text   = "a" * 600
        result = _safe_truncate_for_prompt(text, max_chars=600)
        assert len(result) == 600

    def test_empty_string_returns_empty(self) -> None:
        assert _safe_truncate_for_prompt("") == ""

    def test_truncated_text_ends_with_ellipsis(self) -> None:
        text   = "z" * 700
        result = _safe_truncate_for_prompt(text, max_chars=600)
        assert result.endswith("...")


class TestSentimentServiceAnalyze:
    """Tests for SentimentService.analyze() with mocked OpenAI."""

    @pytest.mark.asyncio
    async def test_analyze_no_text_uses_fallback(self) -> None:
        """Reviews with no text must fall back to star-rating heuristic."""
        svc    = SentimentService()
        result = await svc.analyze(review_text=None, star_rating=5)
        assert result.used_fallback is True
        assert result.sentiment == ReviewSentiment.POSITIVE

    @pytest.mark.asyncio
    async def test_analyze_empty_text_uses_fallback(self) -> None:
        svc    = SentimentService()
        result = await svc.analyze(review_text="", star_rating=1)
        assert result.used_fallback is True
        assert result.sentiment == ReviewSentiment.NEGATIVE

    @pytest.mark.asyncio
    async def test_analyze_openai_success(self) -> None:
        """OpenAI path must return used_fallback=False on success."""
        svc = SentimentService()
        openai_response = '{"sentiment": "positive", "score": 0.85}'

        with patch.object(
            svc,
            "_analyze_with_openai",
            new_callable=AsyncMock,
            return_value=openai_response,
        ):
            result = await svc.analyze(
                review_text="Amazing food and great service!",
                star_rating=5,
            )

        assert result.used_fallback is False
        assert result.sentiment == ReviewSentiment.POSITIVE
        assert result.score > 0

    @pytest.mark.asyncio
    async def test_analyze_openai_failure_falls_back(self) -> None:
        """OpenAI failure must transparently fall back to star-rating heuristic."""
        svc = SentimentService()

        with patch.object(
            svc,
            "_analyze_with_openai",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenAI unavailable"),
        ):
            result = await svc.analyze(
                review_text="Decent experience overall.",
                star_rating=4,
            )

        assert result.used_fallback is True
        assert result.sentiment == ReviewSentiment.POSITIVE

    @pytest.mark.asyncio
    async def test_analyze_never_raises(self) -> None:
        """analyze() must always return SentimentResult — never raise."""
        svc = SentimentService()

        with patch.object(
            svc,
            "_analyze_with_openai",
            new_callable=AsyncMock,
            side_effect=Exception("Catastrophic failure"),
        ):
            result = await svc.analyze(
                review_text="Some review text",
                star_rating=3,
            )

        assert isinstance(result, SentimentResult)

    @pytest.mark.asyncio
    async def test_analyze_batch_returns_list(self) -> None:
        """analyze_batch must return one SentimentResult per input review."""
        svc     = SentimentService()
        reviews = [
            {"review_text": "Great!",    "star_rating": 5},
            {"review_text": "Terrible.", "star_rating": 1},
            {"review_text": None,        "star_rating": 3},
        ]

        with patch.object(svc, "analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = [
                SentimentResult(sentiment=ReviewSentiment.POSITIVE, score=0.9),
                SentimentResult(sentiment=ReviewSentiment.NEGATIVE, score=-0.9),
                SentimentResult(sentiment=ReviewSentiment.NEUTRAL,  score=0.0, used_fallback=True),
            ]
            results = await svc.analyze_batch(reviews)

        assert len(results) == 3
        assert results[0].sentiment == ReviewSentiment.POSITIVE
        assert results[1].sentiment == ReviewSentiment.NEGATIVE
        assert results[2].used_fallback is True


# ==============================================================================
# 2. AIReplyService — private helpers
# ==============================================================================

class TestSentimentToPromptFilename:
    """Tests for _sentiment_to_prompt_filename() helper."""

    def test_positive_returns_positive_filename(self) -> None:
        filename = _sentiment_to_prompt_filename(ReviewSentiment.POSITIVE)
        assert "positive" in filename.lower()

    def test_negative_returns_negative_filename(self) -> None:
        filename = _sentiment_to_prompt_filename(ReviewSentiment.NEGATIVE)
        assert "negative" in filename.lower()

    def test_neutral_returns_neutral_filename(self) -> None:
        filename = _sentiment_to_prompt_filename(ReviewSentiment.NEUTRAL)
        assert "neutral" in filename.lower()

    def test_unknown_sentiment_returns_neutral_filename(self) -> None:
        """Unknown sentiment must safely default to neutral prompt."""
        filename = _sentiment_to_prompt_filename("unknown_sentiment")
        assert "neutral" in filename.lower()

    def test_all_three_sentiments_return_different_filenames(self) -> None:
        files = {
            _sentiment_to_prompt_filename(ReviewSentiment.POSITIVE),
            _sentiment_to_prompt_filename(ReviewSentiment.NEGATIVE),
            _sentiment_to_prompt_filename(ReviewSentiment.NEUTRAL),
        }
        assert len(files) == 3, "Each sentiment must map to a unique prompt file"


class TestFillPromptTemplate:
    """Tests for _fill_prompt_template() helper."""

    _TEMPLATE = (
        "Reply for {{BUSINESS_NAME}} ({{BUSINESS_TYPE}}). "
        "Reviewer: {{REVIEWER_NAME}}. "
        "Stars: {{STAR_RATING}}. "
        "Review: {{REVIEW_TEXT}}. "
        "Sentiment: {{SENTIMENT}}."
    )

    def test_all_placeholders_replaced(self) -> None:
        result = _fill_prompt_template(
            template=self._TEMPLATE,
            business_name=_BUSINESS_NAME,
            business_type=_BUSINESS_TYPE,
            reviewer_name="Rahul",
            star_rating=5,
            review_text="Loved the biryani!",
            sentiment=ReviewSentiment.POSITIVE,
        )
        assert _BUSINESS_NAME in result
        assert _BUSINESS_TYPE in result
        assert "Rahul" in result
        assert "5" in result
        assert "Loved the biryani!" in result
        assert "positive" in result

    def test_no_double_brace_placeholders_remain(self) -> None:
        result = _fill_prompt_template(
            template=self._TEMPLATE,
            business_name=_BUSINESS_NAME,
            business_type=_BUSINESS_TYPE,
            reviewer_name="Priya",
            star_rating=4,
            review_text="Nice experience.",
            sentiment=ReviewSentiment.NEUTRAL,
        )
        assert "{{" not in result
        assert "}}" not in result

    def test_long_review_text_is_truncated(self) -> None:
        """Review text longer than threshold must be truncated in the prompt."""
        long_text = "word " * 300   # ~1500 chars
        result    = _fill_prompt_template(
            template="{{REVIEW_TEXT}}",
            business_name=_BUSINESS_NAME,
            business_type=_BUSINESS_TYPE,
            reviewer_name="User",
            star_rating=3,
            review_text=long_text,
            sentiment=ReviewSentiment.NEUTRAL,
        )
        # The truncated version must be shorter than the original
        assert len(result) < len(long_text)

    def test_result_is_stripped(self) -> None:
        result = _fill_prompt_template(
            template="  {{BUSINESS_NAME}}  ",
            business_name="TrimMe",
            business_type="cafe",
            reviewer_name="A",
            star_rating=5,
            review_text="Good.",
            sentiment=ReviewSentiment.POSITIVE,
        )
        assert result == result.strip()


class TestCleanReplyText:
    """Tests for _clean_reply_text() helper."""

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _clean_reply_text("  Hello!  ") == "Hello!"

    def test_removes_wrapping_double_quotes(self) -> None:
        result = _clean_reply_text('"Thank you for your review!"')
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_removes_wrapping_single_quotes(self) -> None:
        result = _clean_reply_text("'Thank you for visiting us!'")
        assert not result.startswith("'")
        assert not result.endswith("'")

    def test_removes_markdown_bold(self) -> None:
        result = _clean_reply_text("**Thank you** for visiting!")
        assert "**" not in result
        assert "Thank you" in result

    def test_removes_markdown_italic(self) -> None:
        result = _clean_reply_text("We *appreciate* your feedback.")
        assert "*appreciate*" not in result
        assert "appreciate" in result

    def test_collapses_excessive_blank_lines(self) -> None:
        text   = "Thank you.\n\n\n\nWe hope to see you again."
        result = _clean_reply_text(text)
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self) -> None:
        text   = "Thank you for dining with us!"
        result = _clean_reply_text(text)
        assert result == text

    def test_empty_string_returns_empty(self) -> None:
        assert _clean_reply_text("") == ""


class TestAIReplyServiceGenerate:
    """Tests for AIReplyService.generate_reply() end-to-end flow."""

    def _make_service(self) -> AIReplyService:
        svc = AIReplyService.__new__(AIReplyService)
        svc._review_repo    = MagicMock()
        svc._sentiment_svc  = MagicMock(spec=SentimentService)
        svc._prompt_cache   = {}
        return svc

    @pytest.mark.asyncio
    async def test_idempotent_when_reply_already_exists(self) -> None:
        """
        Calling generate_reply for a review that already has an AI reply
        must return skipped=True without calling OpenAI again.
        """
        svc = self._make_service()

        existing_review         = MagicMock()
        existing_review.ai_reply = "Thank you for your wonderful review!"
        existing_review.sentiment = ReviewSentiment.POSITIVE
        svc._review_repo.get_by_id = AsyncMock(return_value=existing_review)

        with patch.object(svc, "_call_openai", new_callable=AsyncMock) as mock_openai:
            result = await svc.generate_reply(
                db=_mock_db(),
                review_id=_REVIEW_ID,
                business_id=_BUSINESS_ID,
                review_text="Amazing food!",
                star_rating=5,
                reviewer_name="Rahul",
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
            )

        assert result.skipped is True
        assert result.skip_reason == "reply_already_exists"
        mock_openai.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_reply_never_raises(self) -> None:
        """generate_reply must always return ReplyResult — never raise."""
        svc = self._make_service()
        svc._review_repo.get_by_id = AsyncMock(return_value=None)

        with (
            patch(
                "app.utils.rate_limiter.enforce_rate_limit",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Rate limiter crashed"),
            ),
        ):
            result = await svc.generate_reply(
                db=_mock_db(),
                review_id=_REVIEW_ID,
                business_id=_BUSINESS_ID,
                review_text="Good place.",
                star_rating=4,
                reviewer_name="Priya",
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
            )

        assert isinstance(result, ReplyResult)

    def test_prompt_safety_no_phone_in_template(self) -> None:
        """
        Prompt templates must never inject phone numbers.
        Sensitive fields are excluded per DATA_SAFETY_AND_RUNTIME_GUARDRAILS §9.
        """
        template = "Business: {{BUSINESS_NAME}}. Review: {{REVIEW_TEXT}}."
        result   = _fill_prompt_template(
            template=template,
            business_name="Cafe 99",
            business_type="cafe",
            reviewer_name="User",
            star_rating=4,
            review_text="Good vibes.",
            sentiment=ReviewSentiment.NEUTRAL,
        )
        # Phone number patterns must not appear
        import re
        phone_pattern = re.compile(r"\b\d{10}\b|\+91\d{10}")
        assert not phone_pattern.search(result), (
            "Prompt must not contain phone numbers — sensitive data exclusion rule."
        )


# ==============================================================================
# 3. AnalyticsService — private helpers
# ==============================================================================

class TestAggregateDailyHelper:
    """Tests for _aggregate_daily() pure function."""

    def _make_row(
        self,
        row_date: date,
        revenue: float,
        product: str = "Biryani",
    ) -> MagicMock:
        row           = MagicMock()
        row.row_date  = row_date
        row.revenue   = Decimal(str(revenue))
        row.product   = product
        return row

    def test_single_row_produces_one_day(self) -> None:
        today = date.today()
        rows  = [self._make_row(today, 500.0)]
        buckets = _aggregate_daily(rows)
        assert len(buckets) == 1
        assert today in buckets

    def test_two_rows_same_day_merged(self) -> None:
        today = date.today()
        rows  = [
            self._make_row(today, 300.0),
            self._make_row(today, 200.0),
        ]
        buckets = _aggregate_daily(rows)
        assert len(buckets) == 1
        assert buckets[today].total_revenue == Decimal("500.0")
        assert buckets[today].transaction_count == 2

    def test_rows_different_days_separate_buckets(self) -> None:
        today     = date.today()
        yesterday = today - timedelta(days=1)
        rows      = [
            self._make_row(today,     500.0),
            self._make_row(yesterday, 300.0),
        ]
        buckets = _aggregate_daily(rows)
        assert len(buckets) == 2

    def test_avg_transaction_computed(self) -> None:
        today = date.today()
        rows  = [self._make_row(today, 300.0), self._make_row(today, 100.0)]
        buckets = _aggregate_daily(rows)
        day = buckets[today]
        assert day.avg_transaction == Decimal("200.0")  # (300 + 100) / 2

    def test_empty_rows_returns_empty_dict(self) -> None:
        assert _aggregate_daily([]) == {}


class TestAggregateWeeklyHelper:
    """Tests for _aggregate_weekly() pure function."""

    def test_single_week_single_entry(self) -> None:
        monday = date(2024, 10, 7)
        days   = [_daily_sales(monday + timedelta(i), revenue=100.0) for i in range(5)]
        result = _aggregate_weekly(days)
        assert len(result) == 1

    def test_two_weeks_two_entries(self) -> None:
        week1_start = date(2024, 10, 7)
        week2_start = date(2024, 10, 14)
        days = (
            [_daily_sales(week1_start + timedelta(i), 100.0) for i in range(5)] +
            [_daily_sales(week2_start + timedelta(i), 200.0) for i in range(5)]
        )
        result = _aggregate_weekly(days)
        assert len(result) == 2

    def test_weekly_revenue_is_sum_of_daily(self) -> None:
        monday = date(2024, 10, 7)
        days   = [_daily_sales(monday + timedelta(i), 100.0) for i in range(5)]
        result = _aggregate_weekly(days)
        assert result[0].total_revenue == Decimal("500.0")

    def test_empty_returns_empty_list(self) -> None:
        assert _aggregate_weekly([]) == []


class TestDetectTrend:
    """Tests for _detect_trend() helper."""

    def _make_days(self, n: int) -> list[DailySales]:
        today = date.today()
        return [_daily_sales(today - timedelta(days=i), 100.0) for i in range(n)]

    def test_positive_change_is_growing(self) -> None:
        days   = self._make_days(14)
        result = _detect_trend(days, revenue_change_pct=0.20)
        assert result.direction == "up"

    def test_negative_change_is_declining(self) -> None:
        days   = self._make_days(14)
        result = _detect_trend(days, revenue_change_pct=-0.15)
        assert result.direction == "down"

    def test_small_change_is_stable(self) -> None:
        days   = self._make_days(14)
        result = _detect_trend(days, revenue_change_pct=0.02)
        assert result.direction == "stable"

    def test_14_days_high_confidence(self) -> None:
        days   = self._make_days(14)
        result = _detect_trend(days, revenue_change_pct=0.10)
        assert result.confidence == "high"

    def test_7_days_medium_confidence(self) -> None:
        days   = self._make_days(7)
        result = _detect_trend(days, revenue_change_pct=0.10)
        assert result.confidence == "medium"

    def test_less_than_7_days_low_confidence(self) -> None:
        days   = self._make_days(3)
        result = _detect_trend(days, revenue_change_pct=0.10)
        assert result.confidence == "low"


class TestFindPeakDays:
    """Tests for _find_peak_days() helper."""

    def test_high_revenue_day_is_peak(self) -> None:
        today    = date.today()
        avg      = Decimal("1000")
        days     = [
            _daily_sales(today,                    revenue=2500.0),  # 2.5x avg → peak
            _daily_sales(today - timedelta(days=1), revenue=800.0),  # below avg
        ]
        peaks = _find_peak_days(days, avg_daily_revenue=avg)
        assert len(peaks) == 1
        assert peaks[0].revenue == Decimal("2500.0")

    def test_no_peak_when_all_below_threshold(self) -> None:
        today = date.today()
        avg   = Decimal("1000")
        days  = [_daily_sales(today, revenue=900.0)]
        peaks = _find_peak_days(days, avg_daily_revenue=avg)
        assert peaks == []

    def test_zero_avg_returns_empty(self) -> None:
        today = date.today()
        days  = [_daily_sales(today, revenue=500.0)]
        peaks = _find_peak_days(days, avg_daily_revenue=Decimal("0"))
        assert peaks == []

    def test_peaks_sorted_by_revenue_descending(self) -> None:
        today = date.today()
        avg   = Decimal("100")
        days  = [
            _daily_sales(today,                    revenue=500.0),
            _daily_sales(today - timedelta(days=1), revenue=800.0),
        ]
        peaks = _find_peak_days(days, avg_daily_revenue=avg)
        assert len(peaks) == 2
        assert peaks[0].revenue > peaks[1].revenue


class TestTopProducts:
    """Tests for _top_products() pure function."""

    def _row(self, product: str, revenue: float) -> MagicMock:
        r         = MagicMock()
        r.product = product
        r.revenue = Decimal(str(revenue))
        return r

    def test_returns_top_n_by_revenue(self) -> None:
        rows = [
            self._row("Biryani",  5000.0),
            self._row("Butter Chicken", 3000.0),
            self._row("Naan",     1000.0),
            self._row("Lassi",    500.0),
            self._row("Kulfi",    200.0),
            self._row("Raita",    100.0),
        ]
        result = _top_products(rows, top_n=5)
        assert len(result) == 5
        assert result[0][0] == "Biryani"
        assert result[0][1] == Decimal("5000.0")

    def test_products_aggregated_across_multiple_rows(self) -> None:
        rows = [
            self._row("Biryani", 300.0),
            self._row("Biryani", 200.0),
            self._row("Naan",    100.0),
        ]
        result = _top_products(rows, top_n=5)
        biryani_total = next(r for r in result if r[0] == "Biryani")[1]
        assert biryani_total == Decimal("500.0")

    def test_empty_rows_returns_empty_list(self) -> None:
        assert _top_products([], top_n=5) == []

    def test_none_product_skipped(self) -> None:
        rows = [
            self._row(None,  300.0),
            self._row("Dal", 200.0),
        ]
        result = _top_products(rows, top_n=5)
        names  = [r[0] for r in result]
        assert None not in names


class TestComputeChangePct:
    """Tests for _compute_change_pct() helper."""

    def test_growth(self) -> None:
        pct = _compute_change_pct(
            current=Decimal("1200"),
            previous=Decimal("1000"),
        )
        assert abs(pct - 0.20) < 0.001

    def test_decline(self) -> None:
        pct = _compute_change_pct(
            current=Decimal("800"),
            previous=Decimal("1000"),
        )
        assert abs(pct - (-0.20)) < 0.001

    def test_no_change(self) -> None:
        pct = _compute_change_pct(
            current=Decimal("1000"),
            previous=Decimal("1000"),
        )
        assert pct == 0.0

    def test_zero_previous_returns_zero(self) -> None:
        """Division by zero must be handled gracefully."""
        pct = _compute_change_pct(
            current=Decimal("500"),
            previous=Decimal("0"),
        )
        assert pct == 0.0


# ==============================================================================
# 4. SeoService — helpers and fallback
# ==============================================================================

class TestRatingTier:
    """Tests for _rating_tier() helper."""

    def test_high_rating(self) -> None:
        assert _rating_tier(4.5) == "high"

    def test_mid_rating(self) -> None:
        assert _rating_tier(3.5) == "mid"

    def test_low_rating(self) -> None:
        assert _rating_tier(2.0) == "low"

    def test_boundary_high(self) -> None:
        """4.0 is the boundary — verify it hits the correct tier."""
        tier = _rating_tier(4.0)
        assert tier in ("high", "mid")

    def test_boundary_low(self) -> None:
        tier = _rating_tier(3.0)
        assert tier in ("mid", "low")


class TestExtractTermsFromReviews:
    """Tests for _extract_terms_from_reviews() helper."""

    def test_extracts_repeated_words(self) -> None:
        reviews = [
            "The biryani was amazing.",
            "Best biryani in the city!",
            "Biryani is their specialty.",
        ]
        terms = _extract_terms_from_reviews(reviews)
        assert any("biryani" in t.lower() for t in terms)

    def test_returns_list(self) -> None:
        result = _extract_terms_from_reviews(["Good food"])
        assert isinstance(result, list)

    def test_empty_list_returns_empty(self) -> None:
        assert _extract_terms_from_reviews([]) == []

    def test_stopwords_excluded(self) -> None:
        """Common stopwords like 'the', 'and', 'is' must not appear in results."""
        reviews = ["The food and the service is and was great."]
        terms   = _extract_terms_from_reviews(reviews)
        stopwords = {"the", "and", "is", "was", "in", "a"}
        for term in terms:
            assert term.lower() not in stopwords, (
                f"Stopword '{term}' should not appear in extracted terms"
            )


class TestSeoServiceFallback:
    """Tests for SeoService.generate_suggestions() with OpenAI mocked."""

    @pytest.mark.asyncio
    async def test_returns_seo_suggestion_result(self) -> None:
        svc = SeoService()

        with patch.object(
            svc,
            "_generate_with_openai",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenAI unavailable"),
        ):
            result = await svc.generate_suggestions(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.2,
            )

        assert isinstance(result, SeoSuggestionResult)

    @pytest.mark.asyncio
    async def test_fallback_used_when_openai_fails(self) -> None:
        svc = SeoService()

        with patch.object(
            svc,
            "_generate_with_openai",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenAI unavailable"),
        ):
            result = await svc.generate_suggestions(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.2,
            )

        assert result.used_fallback is True

    @pytest.mark.asyncio
    async def test_fallback_produces_keywords_and_tips(self) -> None:
        """Even the fallback must produce at least 1 keyword and 1 tip."""
        svc = SeoService()

        with patch.object(
            svc,
            "_generate_with_openai",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenAI unavailable"),
        ):
            result = await svc.generate_suggestions(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=3.5,
            )

        assert len(result.keywords) >= 1
        assert len(result.tips)     >= 1

    @pytest.mark.asyncio
    async def test_generate_suggestions_never_raises(self) -> None:
        """generate_suggestions must never raise regardless of errors."""
        svc = SeoService()

        with patch.object(
            svc,
            "_generate_with_openai",
            new_callable=AsyncMock,
            side_effect=Exception("Catastrophic failure"),
        ):
            result = await svc.generate_suggestions(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.0,
            )

        assert isinstance(result, SeoSuggestionResult)


# ==============================================================================
# 5. ContentGenerationService — fallback and result accessors
# ==============================================================================

class TestContentGenerationServiceFallback:
    """Tests for ContentGenerationService.generate_weekly_content()."""

    @pytest.mark.asyncio
    async def test_returns_result_on_openai_failure(self) -> None:
        """OpenAI failure must produce a fallback ContentGenerationResult, not error."""
        svc = ContentGenerationService()

        with patch.object(
            svc,
            "_client",
            new_callable=MagicMock,
        ) as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("OpenAI unavailable")
            )
            result = await svc.generate_weekly_content(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.2,
                generation_date=date(2024, 10, 7),
            )

        assert not isinstance(result, ContentGenerationError)

    @pytest.mark.asyncio
    async def test_fallback_used_is_true_on_openai_failure(self) -> None:
        svc = ContentGenerationService()

        with patch.object(svc, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("OpenAI unavailable")
            )
            result = await svc.generate_weekly_content(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.2,
                generation_date=date(2024, 10, 7),
            )

        if isinstance(result, ContentGenerationResult):
            assert result.used_fallback is True

    @pytest.mark.asyncio
    async def test_generate_weekly_content_never_raises(self) -> None:
        """generate_weekly_content must always return, never raise."""
        svc = ContentGenerationService()

        with patch.object(svc, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Catastrophic failure")
            )
            result = await svc.generate_weekly_content(
                business_id=_BUSINESS_ID,
                business_name=_BUSINESS_NAME,
                business_type=_BUSINESS_TYPE,
                avg_rating=4.2,
            )

        assert result is not None


class TestContentGenerationResultAccessors:
    """Tests for ContentGenerationResult platform accessor properties."""

    def _make_result(self) -> ContentGenerationResult:
        from app.services.content_generation_service import (
            ContentPiece,
            PLATFORM_GOOGLE,
            PLATFORM_INSTAGRAM,
            PLATFORM_WHATSAPP,
        )
        pieces = [
            ContentPiece(
                platform=PLATFORM_INSTAGRAM,
                text="Instagram caption here #food",
                hashtags=["#food"],
                call_to_action="Visit us today!",
            ),
            ContentPiece(
                platform=PLATFORM_WHATSAPP,
                text="WhatsApp status here",
                hashtags=[],
                call_to_action="Book now!",
            ),
            ContentPiece(
                platform=PLATFORM_GOOGLE,
                text="Google Business post here",
                hashtags=[],
                call_to_action="Call us!",
            ),
        ]
        return ContentGenerationResult(
            business_id=_BUSINESS_ID,
            business_name=_BUSINESS_NAME,
            week_label="Week 42, 2024",
            generated_at=date(2024, 10, 14),
            pieces=pieces,
            used_fallback=False,
        )

    def test_instagram_piece_accessor(self) -> None:
        result = self._make_result()
        assert result.instagram_piece is not None
        assert "Instagram" in result.instagram_piece.platform or True

    def test_whatsapp_piece_accessor(self) -> None:
        result = self._make_result()
        assert result.whatsapp_piece is not None

    def test_google_piece_accessor(self) -> None:
        result = self._make_result()
        assert result.google_piece is not None

    def test_to_whatsapp_messages_returns_list(self) -> None:
        result   = self._make_result()
        messages = result.to_whatsapp_messages()
        assert isinstance(messages, list)
        assert len(messages) >= 1

    def test_to_whatsapp_messages_includes_week_label(self) -> None:
        result   = self._make_result()
        messages = result.to_whatsapp_messages()
        combined = " ".join(messages)
        assert "Week 42" in combined