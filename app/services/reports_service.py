# ==============================================================================
# File: app/services/reports_service.py
# Purpose: Assembles complete WhatsApp report messages for weekly, monthly,
#          and quarterly delivery to business owners.
#
#          This is the final aggregation layer — it consumes structured
#          outputs from all other services and assembles them into
#          human-readable, actionable WhatsApp messages.
#
#          Report types:
#            WEEKLY   — Review summary + sentiment + competitor + suggestions
#            MONTHLY  — Business performance + reputation trends + AI insights
#            QUARTERLY — Strategic analysis + growth signals + 90-day plan
#
#          AI involvement:
#            - Weekly report: rule-based assembly, no OpenAI (fast, free)
#            - Monthly report: OpenAI generates strategic insights paragraph
#            - Quarterly report: OpenAI generates full 90-day strategy section
#            All AI calls use insight_generation_prompt.txt as the base template.
#
#          Output contract:
#            - ReportResult: one or more WhatsApp message strings (split if
#              longer than 4096 chars), ready for whatsapp_service.py delivery
#            - ReportError: returned (never raised) when assembly fails
#
#          Prompt safety (guardrails §9):
#            Only aggregated metrics, sentiment counts, and trend labels
#            are sent to OpenAI. No reviewer names, phone numbers, emails,
#            or payment data are included in prompts.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from app.config.constants import ReportType, ServiceName
from app.config.settings import get_settings
from app.services.analytics_service import AnalyticsError, SalesAnalyticsResult
from app.services.competitor_service import (
    CompetitorScanResult,
    build_competitor_summary_lines,
)
from app.services.seo_service import SeoSuggestionResult
from app.utils.formatting_utils import (
    build_whatsapp_section,
    format_currency_inr,
    format_number,
    format_percentage,
    format_report_footer,
    format_report_header,
    format_review_summary_section,
    format_star_rating,
    split_long_message,
    whatsapp_bold,
    whatsapp_divider,
    whatsapp_italic,
)
from app.utils.retry_utils import with_openai_retry
from app.utils.time_utils import get_date_range_label

logger = logging.getLogger(ServiceName.REPORTS)
settings = get_settings()

# Prompts directory
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_INSIGHT_PROMPT_FILE = "insight_generation_prompt.txt"

# OpenAI settings for insight generation
_INSIGHT_MAX_TOKENS = 500
_INSIGHT_TEMPERATURE = 0.6

# Minimum data required for a meaningful report
MIN_REVIEWS_FOR_REPORT = 1

# Placeholder shown when a section has no data
_NO_DATA_LINE = "ℹ️  Insufficient data for this period."


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class ReportResult:
    """
    Complete assembled report ready for WhatsApp delivery.

    Attributes:
        report_type:      ReportType constant (weekly/monthly/quarterly).
        business_id:      Business UUID.
        business_name:    Business name (for logging).
        messages:         List of WhatsApp message strings. May be more than
                          one if the report exceeds 4096 chars and was split.
        period_label:     Human-readable period e.g. "07–13 Oct 2024".
        generated_at:     Date the report was assembled.
        used_ai_insights: True if OpenAI was used for the insights section.
        sections_included: List of section names that were assembled.
    """
    report_type: str
    business_id: str
    business_name: str
    messages: list[str]
    period_label: str
    generated_at: date
    used_ai_insights: bool
    sections_included: list[str] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def total_chars(self) -> int:
        return sum(len(m) for m in self.messages)

    def __str__(self) -> str:
        return (
            f"ReportResult("
            f"type={self.report_type} "
            f"business={self.business_name!r} "
            f"messages={self.message_count} "
            f"chars={self.total_chars} "
            f"ai={self.used_ai_insights})"
        )


@dataclass(frozen=True)
class ReportError:
    """Returned when a report cannot be assembled."""
    report_type: str
    business_id: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return (
            f"ReportError("
            f"type={self.report_type} "
            f"business={self.business_id} "
            f"reason={self.reason}: {self.detail})"
        )


# ==============================================================================
# Reports Service
# ==============================================================================

class ReportsService:
    """
    Assembles WhatsApp report messages from structured service outputs.

    Stateless — safe to share a single instance across the application.

    Usage:
        service = ReportsService()

        result = await service.build_weekly_report(
            business_id="uuid",
            business_name="Raj Restaurant",
            business_type="Restaurant",
            period_start=date(2024, 10, 7),
            period_end=date(2024, 10, 13),
            total_reviews=12,
            positive_count=9,
            negative_count=2,
            neutral_count=1,
            avg_rating=4.3,
            analytics=analytics_result,       # SalesAnalyticsResult | None
            competitor_scan=competitor_result, # CompetitorScanResult | None
            seo=seo_result,                    # SeoSuggestionResult | None
        )
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._insight_prompt_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Weekly Report
    # ------------------------------------------------------------------

    async def build_weekly_report(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        period_start: date,
        period_end: date,
        total_reviews: int,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        avg_rating: float,
        previous_avg_rating: Optional[float] = None,
        analytics: Optional[SalesAnalyticsResult] = None,
        competitor_scan: Optional[CompetitorScanResult] = None,
        seo: Optional[SeoSuggestionResult] = None,
    ) -> ReportResult | ReportError:
        """
        Build the weekly WhatsApp report.

        Weekly reports are rule-based (no OpenAI) — fast, cheap, reliable.
        Structure:
          1. Header
          2. Review Summary
          3. Sentiment Breakdown
          4. Sales Snapshot (if analytics available)
          5. Competitor Snapshot (if competitor data available)
          6. SEO Tips (if SEO data available)
          7. Footer

        Returns:
            ReportResult or ReportError. Never raises.
        """
        log_extra = {
            "service": ServiceName.REPORTS,
            "business_id": business_id,
            "report_type": ReportType.WEEKLY,
        }

        period_label = get_date_range_label(period_start, period_end)
        sections_included: list[str] = []

        try:
            parts: list[str] = []

            # 1. Header
            parts.append(format_report_header(
                business_name=business_name,
                report_type="Weekly Report",
                period_label=period_label,
            ))
            sections_included.append("header")

            # 2. Review Summary
            review_section = format_review_summary_section(
                total_reviews=total_reviews,
                positive=positive_count,
                negative=negative_count,
                neutral=neutral_count,
                average_rating=avg_rating,
            )
            parts.append(review_section)
            sections_included.append("review_summary")

            # 3. Rating Trend
            if previous_avg_rating is not None:
                rating_section = _build_rating_trend_section(
                    current=avg_rating,
                    previous=previous_avg_rating,
                )
                parts.append(rating_section)
                sections_included.append("rating_trend")

            # 4. Sales Snapshot
            if analytics and not isinstance(analytics, AnalyticsError):
                sales_section = _build_sales_snapshot_section(analytics)
                parts.append(sales_section)
                sections_included.append("sales_snapshot")

            # 5. Competitor Snapshot
            if competitor_scan and not competitor_scan.rate_limited:
                comp_section = _build_competitor_section(competitor_scan)
                parts.append(comp_section)
                sections_included.append("competitor_snapshot")

            # 6. SEO Tips (top 2 tips only in weekly report)
            if seo and seo.tips:
                seo_section = _build_seo_tips_section(
                    seo=seo,
                    max_tips=2,
                    max_keywords=4,
                )
                parts.append(seo_section)
                sections_included.append("seo_tips")

            # 7. Footer
            parts.append(format_report_footer())

            full_report = "\n\n".join(p for p in parts if p)
            messages = split_long_message(full_report)

            logger.info(
                "Weekly report built",
                extra={
                    **log_extra,
                    "sections": sections_included,
                    "messages": len(messages),
                    "total_chars": sum(len(m) for m in messages),
                },
            )

            return ReportResult(
                report_type=ReportType.WEEKLY,
                business_id=business_id,
                business_name=business_name,
                messages=messages,
                period_label=period_label,
                generated_at=date.today(),
                used_ai_insights=False,
                sections_included=sections_included,
            )

        except Exception as exc:
            logger.error(
                "Weekly report assembly failed",
                extra={**log_extra, "error": str(exc), "error_type": type(exc).__name__},
            )
            return ReportError(
                report_type=ReportType.WEEKLY,
                business_id=business_id,
                reason="assembly_error",
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # Monthly Report
    # ------------------------------------------------------------------

    async def build_monthly_report(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        period_start: date,
        period_end: date,
        total_reviews: int,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        avg_rating: float,
        previous_avg_rating: Optional[float] = None,
        analytics: Optional[SalesAnalyticsResult] = None,
        competitor_scan: Optional[CompetitorScanResult] = None,
        seo: Optional[SeoSuggestionResult] = None,
        top_positive_themes: Optional[list[str]] = None,
        top_negative_themes: Optional[list[str]] = None,
    ) -> ReportResult | ReportError:
        """
        Build the monthly WhatsApp report with AI-generated strategic insights.

        Monthly reports include an OpenAI-generated insights paragraph that
        synthesises all the data into specific, actionable recommendations.
        Falls back to rule-based tips if OpenAI is unavailable.

        Structure:
          1. Header
          2. Monthly Performance Summary
          3. Reputation Trends
          4. Sales Analysis (if available)
          5. Competitor Overview (if available)
          6. AI Strategic Insights (OpenAI or rule-based fallback)
          7. SEO Recommendations
          8. Footer
        """
        log_extra = {
            "service": ServiceName.REPORTS,
            "business_id": business_id,
            "report_type": ReportType.MONTHLY,
        }

        period_label = get_date_range_label(period_start, period_end)
        sections_included: list[str] = []
        used_ai = False

        try:
            parts: list[str] = []

            # 1. Header
            parts.append(format_report_header(
                business_name=business_name,
                report_type="Monthly Report",
                period_label=period_label,
            ))
            sections_included.append("header")

            # 2. Monthly Performance Summary
            parts.append(_build_monthly_performance_section(
                total_reviews=total_reviews,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                avg_rating=avg_rating,
                previous_avg_rating=previous_avg_rating,
                analytics=analytics,
            ))
            sections_included.append("performance_summary")

            # 3. Reputation Trends
            if previous_avg_rating is not None or total_reviews > 0:
                parts.append(_build_reputation_trend_section(
                    avg_rating=avg_rating,
                    previous_avg_rating=previous_avg_rating,
                    total_reviews=total_reviews,
                    positive_count=positive_count,
                    negative_count=negative_count,
                    top_positive_themes=top_positive_themes or [],
                    top_negative_themes=top_negative_themes or [],
                ))
                sections_included.append("reputation_trends")

            # 4. Sales Analysis
            if analytics and not isinstance(analytics, AnalyticsError):
                parts.append(_build_sales_analysis_section(analytics))
                sections_included.append("sales_analysis")

            # 5. Competitor Overview
            if competitor_scan and not competitor_scan.rate_limited:
                parts.append(_build_competitor_section(competitor_scan))
                sections_included.append("competitor_overview")

            # 6. AI Strategic Insights
            insights_text, used_ai = await self._generate_monthly_insights(
                business_name=business_name,
                business_type=business_type,
                avg_rating=avg_rating,
                previous_avg_rating=previous_avg_rating,
                total_reviews=total_reviews,
                positive_count=positive_count,
                negative_count=negative_count,
                analytics=analytics,
                competitor_scan=competitor_scan,
                top_negative_themes=top_negative_themes or [],
                log_extra=log_extra,
            )
            parts.append(build_whatsapp_section(
                title="AI Strategic Insights",
                lines=[insights_text],
            ))
            sections_included.append("ai_insights")

            # 7. SEO Recommendations
            if seo:
                parts.append(_build_seo_tips_section(seo, max_tips=4, max_keywords=6))
                sections_included.append("seo_recommendations")

            # 8. Footer
            parts.append(format_report_footer())

            full_report = "\n\n".join(p for p in parts if p)
            messages = split_long_message(full_report)

            logger.info(
                "Monthly report built",
                extra={
                    **log_extra,
                    "sections": sections_included,
                    "messages": len(messages),
                    "used_ai": used_ai,
                },
            )

            return ReportResult(
                report_type=ReportType.MONTHLY,
                business_id=business_id,
                business_name=business_name,
                messages=messages,
                period_label=period_label,
                generated_at=date.today(),
                used_ai_insights=used_ai,
                sections_included=sections_included,
            )

        except Exception as exc:
            logger.error(
                "Monthly report assembly failed",
                extra={**log_extra, "error": str(exc)},
            )
            return ReportError(
                report_type=ReportType.MONTHLY,
                business_id=business_id,
                reason="assembly_error",
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # Quarterly Report
    # ------------------------------------------------------------------

    async def build_quarterly_report(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        period_start: date,
        period_end: date,
        quarter_label: str,
        total_reviews: int,
        avg_rating: float,
        positive_count: int,
        negative_count: int,
        neutral_count: int,
        previous_quarter_avg_rating: Optional[float] = None,
        analytics: Optional[SalesAnalyticsResult] = None,
        competitor_scan: Optional[CompetitorScanResult] = None,
        seo: Optional[SeoSuggestionResult] = None,
        top_positive_themes: Optional[list[str]] = None,
        top_negative_themes: Optional[list[str]] = None,
    ) -> ReportResult | ReportError:
        """
        Build the quarterly WhatsApp report with AI-generated 90-day strategy.

        Quarterly reports include a full AI-generated strategic section that
        analyses 90 days of performance and suggests the next 90-day focus.
        """
        log_extra = {
            "service": ServiceName.REPORTS,
            "business_id": business_id,
            "report_type": ReportType.QUARTERLY,
        }

        period_label = get_date_range_label(period_start, period_end)
        sections_included: list[str] = []
        used_ai = False

        try:
            parts: list[str] = []

            # 1. Header
            parts.append(format_report_header(
                business_name=business_name,
                report_type=f"Quarterly Report — {quarter_label}",
                period_label=period_label,
            ))
            sections_included.append("header")

            # 2. Quarter Performance Summary
            parts.append(_build_monthly_performance_section(
                total_reviews=total_reviews,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                avg_rating=avg_rating,
                previous_avg_rating=previous_quarter_avg_rating,
                analytics=analytics,
            ))
            sections_included.append("quarter_performance")

            # 3. Reputation Trends
            parts.append(_build_reputation_trend_section(
                avg_rating=avg_rating,
                previous_avg_rating=previous_quarter_avg_rating,
                total_reviews=total_reviews,
                positive_count=positive_count,
                negative_count=negative_count,
                top_positive_themes=top_positive_themes or [],
                top_negative_themes=top_negative_themes or [],
            ))
            sections_included.append("reputation_trends")

            # 4. Sales Analysis
            if analytics and not isinstance(analytics, AnalyticsError):
                parts.append(_build_sales_analysis_section(analytics))
                sections_included.append("sales_analysis")

            # 5. Competitor Overview
            if competitor_scan and not competitor_scan.rate_limited:
                parts.append(_build_competitor_section(competitor_scan))
                sections_included.append("competitor_overview")

            # 6. AI 90-Day Strategy
            strategy_text, used_ai = await self._generate_quarterly_strategy(
                business_name=business_name,
                business_type=business_type,
                quarter_label=quarter_label,
                avg_rating=avg_rating,
                previous_avg_rating=previous_quarter_avg_rating,
                total_reviews=total_reviews,
                positive_count=positive_count,
                negative_count=negative_count,
                analytics=analytics,
                competitor_scan=competitor_scan,
                top_positive_themes=top_positive_themes or [],
                top_negative_themes=top_negative_themes or [],
                log_extra=log_extra,
            )
            parts.append(build_whatsapp_section(
                title=f"90-Day Strategy — Next Quarter",
                lines=[strategy_text],
            ))
            sections_included.append("quarterly_strategy")

            # 7. SEO Recommendations
            if seo:
                parts.append(_build_seo_tips_section(seo, max_tips=5, max_keywords=8))
                sections_included.append("seo_recommendations")

            # 8. Footer
            parts.append(format_report_footer())

            full_report = "\n\n".join(p for p in parts if p)
            messages = split_long_message(full_report)

            logger.info(
                "Quarterly report built",
                extra={
                    **log_extra,
                    "sections": sections_included,
                    "messages": len(messages),
                    "used_ai": used_ai,
                },
            )

            return ReportResult(
                report_type=ReportType.QUARTERLY,
                business_id=business_id,
                business_name=business_name,
                messages=messages,
                period_label=period_label,
                generated_at=date.today(),
                used_ai_insights=used_ai,
                sections_included=sections_included,
            )

        except Exception as exc:
            logger.error(
                "Quarterly report assembly failed",
                extra={**log_extra, "error": str(exc)},
            )
            return ReportError(
                report_type=ReportType.QUARTERLY,
                business_id=business_id,
                reason="assembly_error",
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # AI Insight Generation — Monthly
    # ------------------------------------------------------------------

    async def _generate_monthly_insights(
        self,
        business_name: str,
        business_type: str,
        avg_rating: float,
        previous_avg_rating: Optional[float],
        total_reviews: int,
        positive_count: int,
        negative_count: int,
        analytics: Optional[SalesAnalyticsResult],
        competitor_scan: Optional[CompetitorScanResult],
        top_negative_themes: list[str],
        log_extra: dict,
    ) -> tuple[str, bool]:
        """
        Generate the AI strategic insights paragraph for monthly reports.

        Returns:
            tuple[str, bool]: (insights_text, used_ai)
        """
        try:
            text = await self._call_openai_for_insights(
                prompt=_build_monthly_insights_prompt(
                    business_name=business_name,
                    business_type=business_type,
                    avg_rating=avg_rating,
                    previous_avg_rating=previous_avg_rating,
                    total_reviews=total_reviews,
                    positive_count=positive_count,
                    negative_count=negative_count,
                    analytics=analytics,
                    competitor_scan=competitor_scan,
                    top_negative_themes=top_negative_themes,
                    period="monthly",
                ),
                log_extra=log_extra,
            )
            return text, True

        except Exception as exc:
            logger.warning(
                "Monthly insights OpenAI call failed — using fallback",
                extra={**log_extra, "error": str(exc)},
            )
            return _fallback_insights(
                avg_rating=avg_rating,
                total_reviews=total_reviews,
                negative_count=negative_count,
                period="month",
            ), False

    async def _generate_quarterly_strategy(
        self,
        business_name: str,
        business_type: str,
        quarter_label: str,
        avg_rating: float,
        previous_avg_rating: Optional[float],
        total_reviews: int,
        positive_count: int,
        negative_count: int,
        analytics: Optional[SalesAnalyticsResult],
        competitor_scan: Optional[CompetitorScanResult],
        top_positive_themes: list[str],
        top_negative_themes: list[str],
        log_extra: dict,
    ) -> tuple[str, bool]:
        """Generate the AI 90-day strategy section for quarterly reports."""
        try:
            text = await self._call_openai_for_insights(
                prompt=_build_monthly_insights_prompt(
                    business_name=business_name,
                    business_type=business_type,
                    avg_rating=avg_rating,
                    previous_avg_rating=previous_avg_rating,
                    total_reviews=total_reviews,
                    positive_count=positive_count,
                    negative_count=negative_count,
                    analytics=analytics,
                    competitor_scan=competitor_scan,
                    top_negative_themes=top_negative_themes,
                    period="quarterly",
                    quarter_label=quarter_label,
                    top_positive_themes=top_positive_themes,
                ),
                log_extra=log_extra,
            )
            return text, True

        except Exception as exc:
            logger.warning(
                "Quarterly strategy OpenAI call failed — using fallback",
                extra={**log_extra, "error": str(exc)},
            )
            return _fallback_insights(
                avg_rating=avg_rating,
                total_reviews=total_reviews,
                negative_count=negative_count,
                period="quarter",
            ), False

    @with_openai_retry
    async def _call_openai_for_insights(self, prompt: str, log_extra: dict) -> str:
        """
        Call OpenAI to generate the insights/strategy text.

        Uses insight_generation_prompt.txt as the system prompt base.
        The template is loaded from disk and cached for the process lifetime.
        """
        system_prompt = self._load_insight_prompt()

        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=_INSIGHT_MAX_TOKENS,
            temperature=_INSIGHT_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS,
        )

        raw = response.choices[0].message.content or ""
        text = raw.strip()

        if not text:
            raise ValueError("OpenAI returned empty insights text")

        logger.debug(
            "OpenAI insights generated",
            extra={**log_extra, "chars": len(text)},
        )
        return text

    def _load_insight_prompt(self) -> str:
        """Load insight_generation_prompt.txt from disk (cached after first load)."""
        if self._insight_prompt_cache:
            return self._insight_prompt_cache

        prompt_path = _PROMPTS_DIR / _INSIGHT_PROMPT_FILE
        if not prompt_path.exists():
            # Graceful fallback if prompt file is missing
            logger.warning(
                "insight_generation_prompt.txt not found — using inline fallback",
                extra={"service": ServiceName.REPORTS},
            )
            return (
                "You are a business advisor for small local businesses in India. "
                "Analyse the provided business metrics and write 3–5 concise, "
                "specific, and actionable insights in plain text. "
                "Focus on reputation, customer engagement, and revenue growth. "
                "Do not use bullet points or markdown. Write in a warm, professional tone."
            )

        self._insight_prompt_cache = prompt_path.read_text(encoding="utf-8").strip()
        return self._insight_prompt_cache


# ==============================================================================
# Section builders — module-level pure functions
# ==============================================================================

def _build_rating_trend_section(current: float, previous: float) -> str:
    delta = round(current - previous, 2)
    if delta > 0:
        line = f"⬆️  Rating improved by {delta:.1f} stars vs last period."
    elif delta < 0:
        line = f"⬇️  Rating decreased by {abs(delta):.1f} stars vs last period."
    else:
        line = "➡️  Rating remained stable vs last period."
    return build_whatsapp_section("Rating Trend", [line])


def _build_sales_snapshot_section(analytics: SalesAnalyticsResult) -> str:
    lines = [
        f"Revenue:         {analytics.revenue_display}",
        f"Transactions:    {format_number(analytics.total_transactions)}",
        f"Avg per Sale:    ₹{analytics.avg_transaction_value:,.2f}",
        f"Trend:           {analytics.revenue_trend.direction.upper()} "
        f"({analytics.growth_display})",
    ]
    if analytics.busiest_day_of_week:
        lines.append(f"Busiest Day:     {analytics.busiest_day_of_week}")
    if not analytics.has_sufficient_data:
        lines.append("ℹ️  Limited data — trends will improve with more history.")
    return build_whatsapp_section("Sales Snapshot", lines)


def _build_sales_analysis_section(analytics: SalesAnalyticsResult) -> str:
    lines = [
        f"Period Revenue:   {analytics.revenue_display}",
        f"vs Prior Period:  {analytics.growth_display}",
        f"Total Sales:      {format_number(analytics.total_transactions)}",
        f"Avg Sale Value:   ₹{analytics.avg_transaction_value:,.2f}",
        f"Daily Average:    {analytics.avg_daily_display}",
        f"Revenue Trend:    {analytics.revenue_trend.description}",
    ]
    if analytics.peak_days:
        peak = analytics.peak_days[0]
        lines.append(
            f"Peak Day:         {peak.date.strftime('%d %b')} "
            f"({peak.day_of_week}) — ₹{peak.revenue:,.2f}"
        )
    if analytics.busiest_day_of_week:
        lines.append(f"Busiest Day:      {analytics.busiest_day_of_week}")
    if analytics.top_products:
        top_name, top_rev = analytics.top_products[0]
        lines.append(f"Top Product:      {top_name} (₹{top_rev:,.2f})")
    return build_whatsapp_section("Sales Analysis", lines)


def _build_competitor_section(scan: CompetitorScanResult) -> str:
    summary_lines = build_competitor_summary_lines(scan)
    if not summary_lines:
        summary_lines = [_NO_DATA_LINE]
    return build_whatsapp_section("Competitor Overview", summary_lines)


def _build_seo_tips_section(
    seo: SeoSuggestionResult,
    max_tips: int = 3,
    max_keywords: int = 5,
) -> str:
    lines: list[str] = []

    if seo.keywords:
        kw_list = ", ".join(seo.keyword_list[:max_keywords])
        lines.append(whatsapp_bold("Suggested Keywords:"))
        lines.append(kw_list)
        lines.append("")

    if seo.tips:
        lines.append(whatsapp_bold("SEO Tips:"))
        for i, tip in enumerate(seo.tip_list[:max_tips], 1):
            lines.append(f"{i}. {tip}")

    if not lines:
        lines.append(_NO_DATA_LINE)

    return build_whatsapp_section("Local SEO", lines)


def _build_monthly_performance_section(
    total_reviews: int,
    positive_count: int,
    negative_count: int,
    neutral_count: int,
    avg_rating: float,
    previous_avg_rating: Optional[float],
    analytics: Optional[SalesAnalyticsResult],
) -> str:
    stars = format_star_rating(round(avg_rating))
    lines = [
        f"Total Reviews:    {format_number(total_reviews)}",
        f"Avg Rating:       {stars} ({avg_rating:.1f}/5.0)",
        f"😊 Positive:      {format_number(positive_count)} "
        f"({format_percentage(positive_count / total_reviews if total_reviews else 0)})",
        f"😞 Negative:      {format_number(negative_count)} "
        f"({format_percentage(negative_count / total_reviews if total_reviews else 0)})",
        f"😐 Neutral:       {format_number(neutral_count)}",
    ]
    if previous_avg_rating is not None:
        delta = round(avg_rating - previous_avg_rating, 2)
        sign = "+" if delta >= 0 else ""
        lines.append(f"Rating Change:    {sign}{delta:.2f} vs prior period")
    if analytics and not isinstance(analytics, AnalyticsError):
        lines.append(f"Revenue:          {analytics.revenue_display}")
        lines.append(f"Growth:           {analytics.growth_display}")

    return build_whatsapp_section("Performance Summary", lines)


def _build_reputation_trend_section(
    avg_rating: float,
    previous_avg_rating: Optional[float],
    total_reviews: int,
    positive_count: int,
    negative_count: int,
    top_positive_themes: list[str],
    top_negative_themes: list[str],
) -> str:
    lines: list[str] = []

    if previous_avg_rating is not None:
        delta = round(avg_rating - previous_avg_rating, 2)
        direction = "improved ⬆️" if delta > 0 else "declined ⬇️" if delta < 0 else "stable ➡️"
        lines.append(f"Your reputation has {direction} this period.")

    positive_rate = positive_count / total_reviews if total_reviews else 0
    lines.append(f"Positive review rate: {format_percentage(positive_rate)}")

    if top_positive_themes:
        themes = ", ".join(top_positive_themes[:4])
        lines.append(f"Customers love: {themes}")

    if top_negative_themes:
        themes = ", ".join(top_negative_themes[:3])
        lines.append(f"Areas to improve: {themes}")

    if not lines:
        lines.append(_NO_DATA_LINE)

    return build_whatsapp_section("Reputation Trends", lines)


# ==============================================================================
# Prompt builders — keep OpenAI prompts separate from assembly logic
# ==============================================================================

def _build_monthly_insights_prompt(
    business_name: str,
    business_type: str,
    avg_rating: float,
    previous_avg_rating: Optional[float],
    total_reviews: int,
    positive_count: int,
    negative_count: int,
    analytics: Optional[SalesAnalyticsResult],
    competitor_scan: Optional[CompetitorScanResult],
    top_negative_themes: list[str],
    period: str,
    quarter_label: str = "",
    top_positive_themes: Optional[list[str]] = None,
) -> str:
    """
    Build the user-role prompt for monthly or quarterly AI insights.

    Prompt safety (guardrails §9):
    Only aggregated metrics and labels are included.
    No reviewer names, phone numbers, emails, or payment data.
    """
    lines = [
        f"Business: {business_name} ({business_type})",
        f"Period: {period}" + (f" — {quarter_label}" if quarter_label else ""),
        f"Average Google Rating: {avg_rating:.1f}/5.0",
    ]

    if previous_avg_rating is not None:
        delta = round(avg_rating - previous_avg_rating, 2)
        sign = "+" if delta >= 0 else ""
        lines.append(f"Rating change vs prior period: {sign}{delta:.2f}")

    lines += [
        f"Total reviews: {total_reviews}",
        f"Positive: {positive_count}, Negative: {negative_count}, "
        f"Neutral: {total_reviews - positive_count - negative_count}",
    ]

    if top_negative_themes:
        lines.append(f"Common complaints: {', '.join(top_negative_themes[:4])}")

    if top_positive_themes:
        lines.append(f"What customers praise: {', '.join(top_positive_themes[:4])}")

    if analytics and not isinstance(analytics, AnalyticsError):
        lines += [
            f"Revenue: ₹{analytics.total_revenue:,.2f}",
            f"Revenue trend: {analytics.revenue_trend.direction} "
            f"({analytics.growth_display})",
            f"Total transactions: {analytics.total_transactions}",
        ]
        if analytics.busiest_day_of_week:
            lines.append(f"Busiest day: {analytics.busiest_day_of_week}")

    if competitor_scan and not competitor_scan.rate_limited and competitor_scan.snapshots:
        comp = competitor_scan
        lines.append(
            f"Competitors tracked: {comp.competitors_scanned}, "
            f"We are {'leading' if comp.we_are_leading else 'behind'} on ratings"
        )
        if comp.rating_vs_best is not None:
            sign = "+" if comp.rating_vs_best >= 0 else ""
            lines.append(
                f"Our rating vs best competitor: {sign}{comp.rating_vs_best:.1f}"
            )

    lines.append(
        f"\nBased on this data, generate {'3-5 strategic insights' if period == 'monthly' else 'a 90-day action plan'} "
        f"for this {business_type.lower()} to improve their reputation, "
        f"customer engagement, and revenue. Be specific and actionable. "
        f"Write in plain text, no bullet points, no markdown."
    )

    return "\n".join(lines)


def _fallback_insights(
    avg_rating: float,
    total_reviews: int,
    negative_count: int,
    period: str,
) -> str:
    """Rule-based insights paragraph used when OpenAI is unavailable."""
    lines: list[str] = []

    if avg_rating >= 4.5:
        lines.append(
            f"Your reputation this {period} is strong with a {avg_rating:.1f}-star rating. "
            "To maintain this momentum, continue responding to every review promptly — "
            "businesses that reply to all reviews rank higher in Google Maps searches."
        )
    elif avg_rating >= 4.0:
        lines.append(
            f"Your {avg_rating:.1f}-star rating this {period} is solid. "
            "Focus on increasing review volume — the more recent reviews you have, "
            "the more Google's algorithm favours your listing in local searches."
        )
    else:
        lines.append(
            f"Your current rating of {avg_rating:.1f} has room for improvement. "
            "Start by responding to all negative reviews with a professional apology "
            "and resolution offer — this shows future customers you care."
        )

    if negative_count > 0:
        lines.append(
            f"You received {negative_count} negative review{'s' if negative_count > 1 else ''} "
            f"this {period}. Review each one and look for common themes — "
            "recurring complaints represent the highest-impact improvement opportunities."
        )

    if total_reviews < 10:
        lines.append(
            "Your review volume is low. After each customer interaction, "
            "ask satisfied customers to leave a Google review — even a simple "
            "verbal request at checkout can increase review volume by 30%."
        )

    return " ".join(lines) if lines else "Keep engaging with your customers and responding to reviews to build your online reputation."