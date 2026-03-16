# ==============================================================================
# File: app/schedulers/sales_analysis_job.py
# Purpose: Daily scheduled job (01:00 UTC) that reads each business's
#          Google Sheets sales data, runs analytics, detects anomalies,
#          and delivers an insight summary via WhatsApp.
#
#          Schedule: Every day at 01:00 UTC (registered by scheduler_manager)
#
#          Every subscribed business receives sales analysis — there is
#          no plan gating. The job runs for all businesses with an active
#          subscription AND a configured Google Sheet.
#
#          Processing pipeline per business:
#            1. Verify active subscription
#            2. Verify Google Sheet is configured (skip if not)
#            3. Fetch sheet data via google_sheets_client
#            4. Map columns dynamically via column_mapper_service
#            5. Validate rows via sheet_validator
#            6. Run analytics via analytics_service
#            7. Dispatch sales anomaly alerts via alert_manager
#            8. Send insight summary via WhatsApp
#            9. Increment usage counters
#
#          Skip conditions (cheapest first):
#            business_not_found     → no record in DB
#            no_sheet_configured    → business has no Google Sheet URL
#            sheet_fetch_failed     → Google Sheets API returned error
#            column_mapping_failed  → could not identify required columns
#            no_valid_rows          → sheet has no usable data after validation
#
#          Performance contract:
#            - Businesses processed in batches of BUSINESS_BATCH_SIZE (10)
#            - Max rows fetched per sheet: MAX_SHEET_ROWS (500)
#            - Lookback window for trend analysis: LOOKBACK_DAYS (30)
#            - Each business isolated — failure never blocks others
#            - Scheduler-level lock managed by scheduler_manager.py
# ==============================================================================

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.alerts.alert_manager import AlertManager, SalesAnomalyInput
from app.config.constants import ServiceName
from app.notifications.admin_notification_service import AdminNotificationService
from app.notifications.whatsapp_service import WhatsAppService
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.services.analytics_service import AnalyticsService
from app.services.column_mapper_service import ColumnMapperService
from app.integrations.google_sheets_client import GoogleSheetsClient
from app.utils.batch_utils import process_in_batches
from app.utils.time_utils import today_local
from app.utils.usage_tracker import UsageTracker
from app.validators.sheet_validator import SheetValidator

logger = logging.getLogger(ServiceName.SALES_ANALYSIS)

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
BUSINESS_BATCH_SIZE: int = 10
MAX_SHEET_ROWS: int = 500
LOOKBACK_DAYS: int = 30


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class BusinessAnalysisResult:
    """Result of running sales analysis for a single business."""
    business_id: str
    rows_analysed: int = 0
    alerts_dispatched: int = 0
    insight_sent: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SalesAnalysisJobResult:
    """Aggregate result of a full sales analysis job run."""
    businesses_analysed: int = 0
    businesses_skipped: int = 0
    businesses_errored: int = 0
    total_rows_processed: int = 0
    total_alerts_dispatched: int = 0

    def merge(self, biz: BusinessAnalysisResult) -> None:
        if biz.skipped:
            self.businesses_skipped += 1
            return
        if biz.error and biz.rows_analysed == 0:
            self.businesses_errored += 1
            return
        self.businesses_analysed += 1
        self.total_rows_processed += biz.rows_analysed
        self.total_alerts_dispatched += biz.alerts_dispatched


# ==============================================================================
# Sales Analysis Job
# ==============================================================================

class SalesAnalysisJob:
    """
    Daily sales analytics job for all active subscribed businesses.

    Reads Google Sheets data, detects revenue anomalies, and delivers
    insight summaries via WhatsApp. Every subscribed business with a
    configured Google Sheet receives this analysis.
    """

    def __init__(
        self,
        business_repo: BusinessRepository,
        subscription_repo: SubscriptionRepository,
        sheets_client: GoogleSheetsClient,
        column_mapper: ColumnMapperService,
        analytics_service: AnalyticsService,
        sheet_validator: SheetValidator,
        alert_manager: AlertManager,
        whatsapp_service: WhatsAppService,
        usage_tracker: UsageTracker,
        admin_notification: AdminNotificationService,
    ) -> None:
        self._biz_repo = business_repo
        self._sub_repo = subscription_repo
        self._sheets = sheets_client
        self._column_mapper = column_mapper
        self._analytics = analytics_service
        self._validator = sheet_validator
        self._alerts = alert_manager
        self._whatsapp = whatsapp_service
        self._usage_tracker = usage_tracker
        self._admin = admin_notification

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, db: AsyncSession) -> SalesAnalysisJobResult:
        """
        Execute one full sales analysis cycle for all active businesses.

        Returns:
            SalesAnalysisJobResult. Never raises.
        """
        aggregate = SalesAnalysisJobResult()
        log_extra = {"service": ServiceName.SALES_ANALYSIS}

        logger.info("Sales analysis job started", extra=log_extra)

        try:
            active_ids = await self._sub_repo.get_active_business_ids(db=db)

            if not active_ids:
                logger.info(
                    "No active businesses — skipping sales analysis",
                    extra=log_extra,
                )
                return aggregate

            logger.info(
                "Running sales analysis for %d businesses",
                len(active_ids),
                extra=log_extra,
            )

            async def process_batch(batch: list[str]) -> None:
                for business_id in batch:
                    result = await self._process_business(
                        db=db,
                        business_id=business_id,
                    )
                    aggregate.merge(result)

            await process_in_batches(
                items=active_ids,
                batch_size=BUSINESS_BATCH_SIZE,
                processor=process_batch,
            )

        except Exception as exc:
            logger.error(
                "Sales analysis job failed at top level",
                extra={**log_extra, "error": str(exc)},
            )
            await self._admin.send_job_failure(
                job_name="sales_analysis_job",
                error=str(exc),
            )

        logger.info(
            "Sales analysis job complete",
            extra={
                **log_extra,
                "analysed": aggregate.businesses_analysed,
                "skipped": aggregate.businesses_skipped,
                "errored": aggregate.businesses_errored,
                "rows": aggregate.total_rows_processed,
                "alerts": aggregate.total_alerts_dispatched,
            },
        )
        return aggregate

    # ------------------------------------------------------------------
    # Per-business processing
    # ------------------------------------------------------------------

    async def _process_business(
        self,
        db: AsyncSession,
        business_id: str,
    ) -> BusinessAnalysisResult:
        """Run the full sales analysis pipeline for a single business."""
        result = BusinessAnalysisResult(business_id=business_id)
        log_extra = {
            "service": ServiceName.SALES_ANALYSIS,
            "business_id": business_id,
        }

        try:
            # ── Load business ─────────────────────────────────────────
            business = await self._biz_repo.get_by_id(
                db=db, business_id=business_id
            )
            if not business:
                result.skipped = True
                result.skip_reason = "business_not_found"
                return result

            # ── Google Sheet configured? ──────────────────────────────
            if not getattr(business, "google_sheet_url", None):
                result.skipped = True
                result.skip_reason = "no_sheet_configured"
                return result

            # ── Fetch sheet data ──────────────────────────────────────
            today = today_local()
            lookback_start = today - timedelta(days=LOOKBACK_DAYS)

            fetch_result = await self._sheets.fetch_rows(
                sheet_url=business.google_sheet_url,
                max_rows=MAX_SHEET_ROWS,
            )

            if not fetch_result.success:
                logger.warning(
                    "Sheet fetch failed",
                    extra={**log_extra, "error": fetch_result.error},
                )
                result.skipped = True
                result.skip_reason = f"sheet_fetch_failed: {fetch_result.error}"
                return result

            raw_rows = fetch_result.rows or []

            # ── Map columns ───────────────────────────────────────────
            mapping_result = await self._column_mapper.detect_columns(
                rows=raw_rows,
                business_id=business_id,
            )

            if not mapping_result.success:
                logger.warning(
                    "Column mapping failed",
                    extra={**log_extra, "error": mapping_result.error},
                )
                result.skipped = True
                result.skip_reason = f"column_mapping_failed: {mapping_result.error}"
                return result

            # ── Validate rows ─────────────────────────────────────────
            validation_result = self._validator.validate_rows(
                rows=raw_rows,
                column_map=mapping_result.column_map,
                start_date=lookback_start,
                end_date=today,
            )

            if not validation_result.valid_rows:
                result.skipped = True
                result.skip_reason = "no_valid_rows"
                return result

            valid_rows = validation_result.valid_rows
            result.rows_analysed = len(valid_rows)

            # ── Run analytics ─────────────────────────────────────────
            analytics_result = await self._analytics.run(
                business_id=business_id,
                rows=valid_rows,
                column_map=mapping_result.column_map,
                start_date=lookback_start,
                end_date=today,
            )

            if not analytics_result.success:
                logger.warning(
                    "Analytics computation failed",
                    extra={**log_extra, "error": analytics_result.error},
                )
                result.error = f"analytics_failed: {analytics_result.error}"
                return result

            # ── Dispatch anomaly alerts ───────────────────────────────
            alerts_dispatched = await self._dispatch_alerts(
                db=db,
                business_id=business_id,
                analytics=analytics_result,
                log_extra=log_extra,
            )
            result.alerts_dispatched = alerts_dispatched

            # ── Send insight summary via WhatsApp ─────────────────────
            if analytics_result.total_revenue and analytics_result.total_revenue > 0:
                try:
                    summary = _build_insight_summary(
                        business_name=business.business_name,
                        analytics=analytics_result,
                        lookback_days=LOOKBACK_DAYS,
                    )
                    await self._whatsapp.send_text_message(
                        db=db,
                        business_id=business_id,
                        text=summary,
                    )
                    result.insight_sent = True
                except Exception as exc:
                    logger.warning(
                        "Insight summary delivery failed",
                        extra={**log_extra, "error": str(exc)},
                    )

            # ── Usage counter ─────────────────────────────────────────
            try:
                await self._usage_tracker.increment_reports_generated(
                    db=db, business_id=business_id
                )
            except Exception as exc:
                logger.warning(
                    "Usage counter increment failed",
                    extra={**log_extra, "error": str(exc)},
                )

            logger.info(
                "Sales analysis complete",
                extra={
                    **log_extra,
                    "rows": result.rows_analysed,
                    "alerts": result.alerts_dispatched,
                    "insight_sent": result.insight_sent,
                },
            )

        except Exception as exc:
            logger.error(
                "Sales analysis failed for business",
                extra={**log_extra, "error": str(exc)},
            )
            result.error = str(exc)

        return result

    # ------------------------------------------------------------------
    # Alert dispatch helper
    # ------------------------------------------------------------------

    async def _dispatch_alerts(
        self,
        db: AsyncSession,
        business_id: str,
        analytics,
        log_extra: dict,
    ) -> int:
        """
        Dispatch sales anomaly alerts independently.

        Each alert signal dispatched in its own try/except — one
        failed alert does not block the others.

        Returns:
            Number of alerts successfully dispatched.
        """
        dispatched = 0
        signals = getattr(analytics, "anomaly_signals", []) or []

        for signal in signals:
            try:
                alert_input = SalesAnomalyInput(
                    business_id=business_id,
                    signal_type=signal.signal_type,
                    description=signal.description,
                    severity=signal.severity,
                    metric_value=signal.metric_value,
                    threshold_value=signal.threshold_value,
                )
                alert_result = await self._alerts.dispatch_sales_alert(
                    db=db,
                    input=alert_input,
                )
                if alert_result.dispatched:
                    dispatched += 1
            except Exception as exc:
                logger.warning(
                    "Sales alert dispatch failed",
                    extra={
                        **log_extra,
                        "signal": getattr(signal, "signal_type", "unknown"),
                        "error": str(exc),
                    },
                )

        return dispatched


# ==============================================================================
# Pure helpers
# ==============================================================================

def _build_insight_summary(
    business_name: str,
    analytics,
    lookback_days: int,
) -> str:
    """
    Build a WhatsApp-formatted insight summary from analytics results.

    Only sent when total_revenue > 0 to avoid empty noise messages.
    """
    total_revenue = getattr(analytics, "total_revenue", 0) or 0
    avg_daily_revenue = getattr(analytics, "avg_daily_revenue", 0) or 0
    peak_day = getattr(analytics, "peak_day", None)
    total_transactions = getattr(analytics, "total_transactions", 0) or 0
    revenue_trend = getattr(analytics, "revenue_trend", "stable")

    trend_emoji = {
        "rising": "📈",
        "declining": "📉",
        "stable": "➡️",
    }.get(revenue_trend, "📊")

    lines = [
        f"📊 *Daily Sales Insight — {business_name}*",
        f"_Last {lookback_days} days_",
        "",
        f"Total Revenue:      *₹{total_revenue:,.0f}*",
        f"Avg Daily Revenue:  *₹{avg_daily_revenue:,.0f}*",
        f"Total Transactions: *{total_transactions}*",
        f"Revenue Trend:      {trend_emoji} *{revenue_trend.title()}*",
    ]

    if peak_day:
        lines += [
            "",
            f"🏆 Peak Day: *{peak_day}*",
        ]

    return "\n".join(lines)


# ==============================================================================
# Functional entry point
# ==============================================================================

async def run_sales_analysis(
    db: AsyncSession,
    business_repo: BusinessRepository,
    subscription_repo: SubscriptionRepository,
    admin_notification: AdminNotificationService,
    **kwargs,
) -> SalesAnalysisJobResult:
    """Functional entry point called by scheduler_manager.py."""
    job = SalesAnalysisJob(
        business_repo=business_repo,
        subscription_repo=subscription_repo,
        sheets_client=kwargs["sheets_client"],
        column_mapper=kwargs["column_mapper"],
        analytics_service=kwargs["analytics_service"],
        sheet_validator=kwargs["sheet_validator"],
        alert_manager=kwargs["alert_manager"],
        whatsapp_service=kwargs["whatsapp_service"],
        usage_tracker=kwargs["usage_tracker"],
        admin_notification=admin_notification,
    )
    return await job.run(db=db)