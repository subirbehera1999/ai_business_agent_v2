# ==============================================================================
# File: app/config/constants.py
# Purpose: Defines all static, immutable constants used across the system.
#          No environment variables here — only fixed values that govern
#          system behaviour, naming conventions, limits, and identifiers.
#
#          SUBSCRIPTION MODEL — ONE TIER ONLY
#          ─────────────────────────────────────
#          There are no plan tiers (no BASIC / PRO / ENTERPRISE).
#          Pay = full access to every feature.
#          The only choice is billing_cycle: "monthly" or "annual".
#          DAILY_USAGE_LIMITS defines a single flat set of daily caps.
#          These caps exist solely to prevent abuse — not to gate features.
# ==============================================================================

from enum import Enum


# ==============================================================================
# APPLICATION
# ==============================================================================

API_VERSION_PREFIX = "/api/v1"
API_TITLE = "AI Business Agent API"
API_DESCRIPTION = (
    "SaaS backend for automated review management, "
    "AI insights, and business analytics."
)

# Health check route (excluded from auth middleware)
HEALTH_CHECK_PATH = "/api/v1/health"


# ==============================================================================
# SUBSCRIPTION
# ==============================================================================

class SubscriptionStatus(str, Enum):
    """Lifecycle states of a subscription."""
    ACTIVE    = "active"
    EXPIRED   = "expired"
    CANCELLED = "cancelled"
    PENDING   = "pending"
    TRIAL     = "trial"


# Daily usage caps — one flat set for all paying businesses.
# These are abuse-prevention guards only, NOT feature gates.
# plan_manager.py and rate_limiter.py enforce these limits.
DAILY_USAGE_LIMITS: dict[str, int] = {
    "max_reviews_per_day":          200,
    "max_ai_replies_per_day":       200,
    "max_competitor_scans_per_day":  10,
    "max_reports_per_day":            5,
}

# Trial period duration (days)
TRIAL_PERIOD_DAYS = 14

# Days before expiry to send renewal reminder
SUBSCRIPTION_EXPIRY_REMINDER_DAYS = 3


# ==============================================================================
# PAYMENT
# ==============================================================================

class PaymentStatus(str, Enum):
    """Razorpay payment lifecycle states."""
    INITIATED = "initiated"
    SUCCESS   = "success"
    FAILED    = "failed"
    REFUNDED  = "refunded"
    PENDING   = "pending"
    CAPTURED  = "captured"


class PaymentEventType(str, Enum):
    """Razorpay webhook event types handled by the system."""
    PAYMENT_CAPTURED       = "payment.captured"
    PAYMENT_FAILED         = "payment.failed"
    REFUND_PROCESSED       = "refund.processed"
    SUBSCRIPTION_ACTIVATED = "subscription.activated"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_COMPLETED = "subscription.completed"


# Billing cycle options — the only choice available to the business
BILLING_CYCLE_MONTHLY = "monthly"
BILLING_CYCLE_ANNUAL  = "annual"
VALID_BILLING_CYCLES  = (BILLING_CYCLE_MONTHLY, BILLING_CYCLE_ANNUAL)


# ==============================================================================
# REVIEW PROCESSING
# ==============================================================================

class ReviewSentiment(str, Enum):
    """Possible sentiment classifications for a review."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


class ReviewStatus(str, Enum):
    """Processing state of a review record."""
    NEW        = "new"
    PROCESSING = "processing"
    REPLIED    = "replied"
    SKIPPED    = "skipped"
    FAILED     = "failed"


# Google review star rating thresholds for sentiment classification
POSITIVE_RATING_THRESHOLD = 4   # 4–5 stars → positive
NEGATIVE_RATING_THRESHOLD = 2   # 1–2 stars → negative
                                  # 3 stars   → neutral

# Maximum number of reviews fetched per polling cycle per business
REVIEW_POLL_BATCH_SIZE = 50

# Minimum review text length to qualify for AI reply generation
REVIEW_MIN_TEXT_LENGTH = 5

# Maximum review text length sent to AI (truncated beyond this)
REVIEW_MAX_TEXT_LENGTH_FOR_AI = 800


# ==============================================================================
# AI REPLY GENERATION
# ==============================================================================

# Prompt file names (located in app/prompts/)
PROMPT_POSITIVE_REVIEW_REPLY  = "positive_review_reply_prompt.txt"
PROMPT_NEGATIVE_REVIEW_REPLY  = "negative_review_reply_prompt.txt"
PROMPT_NEUTRAL_REVIEW_REPLY   = "neutral_review_reply_prompt.txt"
PROMPT_CONTENT_GENERATE       = "content_generation_prompt.txt"
PROMPT_INSIGHT_GENERATE       = "insight_generation_prompt.txt"

# Keep old names as aliases so existing imports do not break
PROMPT_POSITIVE_REVIEW  = PROMPT_POSITIVE_REVIEW_REPLY
PROMPT_NEGATIVE_REVIEW  = PROMPT_NEGATIVE_REVIEW_REPLY
PROMPT_NEUTRAL_REVIEW   = PROMPT_NEUTRAL_REVIEW_REPLY

# Maximum characters in a generated AI reply
AI_REPLY_MAX_LENGTH = 500

# Maximum characters in a generated insight
AI_INSIGHT_MAX_LENGTH = 1200


# ==============================================================================
# ALERT SYSTEM
# ==============================================================================

class AlertType(str, Enum):
    """Business event alert categories delivered via WhatsApp."""
    NEGATIVE_REVIEW           = "negative_review"
    POSITIVE_REVIEW           = "positive_review"
    REVIEW_SPIKE              = "review_spike"
    RATING_DROP               = "rating_drop"
    COMPETITOR_RATING_CHANGE  = "competitor_rating_change"
    COMPETITOR_REVIEW_SPIKE   = "competitor_review_spike"
    SALES_TREND               = "sales_trend"
    OPPORTUNITY               = "opportunity"
    USAGE_LIMIT_REACHED       = "usage_limit_reached"
    SUBSCRIPTION_EXPIRING     = "subscription_expiring"
    SUBSCRIPTION_EXPIRED      = "subscription_expired"


class AlertSeverity(str, Enum):
    """Severity level of a business alert."""
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


# Minimum rating drop (in stars) to trigger a RATING_DROP alert
RATING_DROP_THRESHOLD = 0.3

# Number of new reviews within the polling window to trigger a REVIEW_SPIKE alert
REVIEW_SPIKE_THRESHOLD = 5


# ==============================================================================
# REPORT TYPES
# ==============================================================================

class ReportType(str, Enum):
    """Scheduled report categories generated by the system."""
    WEEKLY    = "weekly"
    MONTHLY   = "monthly"
    QUARTERLY = "quarterly"


# ==============================================================================
# SCHEDULER & JOB LOCKING
# ==============================================================================

class JobType(str, Enum):
    """Identifiers for all background jobs in the scheduler."""
    REVIEW_MONITOR   = "review_monitor"
    SALES_ANALYSIS   = "sales_analysis"
    WEEKLY_CONTENT   = "weekly_content"
    WEEKLY_REPORT    = "weekly_report"
    MONTHLY_REPORT   = "monthly_report"
    QUARTERLY_REPORT = "quarterly_report"
    EXPIRY_CHECK     = "expiry_check"
    HEALTH_REPORT    = "health_report"
    COMPETITOR_SCAN  = "competitor_scan"
    ALERT_DETECTION  = "alert_detection"


class JobStatus(str, Enum):
    """Execution state of a scheduled job record."""
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


# Job lock key format — used by scheduler_manager to prevent duplicate execution
# Format: JOB_LOCK_{JOB_TYPE}_BUSINESS_{BUSINESS_ID}
JOB_LOCK_KEY_FORMAT = "JOB_LOCK_{job_type}_BUSINESS_{business_id}"

# Maximum allowed job execution time before the lock is considered stale (seconds)
JOB_MAX_EXECUTION_SECONDS = 120

# Job log retention policy (days) — older records are purged automatically
JOB_LOG_RETENTION_DAYS = 90


# ==============================================================================
# IDEMPOTENCY
# ==============================================================================

# Idempotency key formats — used by idempotency_utils.py to deduplicate tasks.
# Each key must be deterministic and unique per operation instance.

IDEMPOTENCY_KEY_REVIEW_REPLY   = "BUSINESS_{business_id}_REVIEW_REPLY_REVIEWID_{review_id}"
IDEMPOTENCY_KEY_PAYMENT_VERIFY = "BUSINESS_{business_id}_PAYMENT_VERIFY_PAYMENTID_{payment_id}"
IDEMPOTENCY_KEY_REVIEW_PROCESS = "BUSINESS_{business_id}_REVIEW_PROCESS_REVIEWID_{review_id}"
IDEMPOTENCY_KEY_REPORT_GEN     = "BUSINESS_{business_id}_REPORT_{report_type}_DATE_{date}"
IDEMPOTENCY_KEY_CONTENT_GEN    = "BUSINESS_{business_id}_CONTENT_GEN_WEEK_{week}"
IDEMPOTENCY_KEY_ALERT_SEND     = "BUSINESS_{business_id}_ALERT_{alert_type}_DATE_{date}"
IDEMPOTENCY_KEY_PAYMENT_INIT   = "PAYMENT_INIT_{business_id}_{billing_cycle}"


# ==============================================================================
# USAGE TRACKING
# ==============================================================================

class UsageMetric(str, Enum):
    """Tracked usage counters per business per day."""
    REVIEWS_PROCESSED    = "reviews_processed"
    AI_REPLIES_GENERATED = "ai_replies_generated"
    COMPETITOR_SCANS     = "competitor_scans"
    REPORTS_GENERATED    = "reports_generated"


# ==============================================================================
# WHATSAPP MESSAGING
# ==============================================================================

class WhatsAppMessageType(str, Enum):
    """Types of WhatsApp messages sent by the system."""
    TEXT     = "text"
    TEMPLATE = "template"


# Maximum character length of a single WhatsApp message
WHATSAPP_MAX_MESSAGE_LENGTH = 4096

# Maximum number of WhatsApp send retries before marking as failed
WHATSAPP_MAX_SEND_RETRIES = 3


# ==============================================================================
# GOOGLE INTEGRATIONS
# ==============================================================================

# Google Places API — maximum results returned per competitor search
GOOGLE_PLACES_MAX_RESULTS = 5

# Google Sheets — maximum rows read per sync cycle
GOOGLE_SHEETS_MAX_ROWS = 1000

# Google Reviews — maximum reviews fetched per API call
GOOGLE_REVIEWS_MAX_PER_FETCH = 50


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

# Default batch sizes — can be overridden by BatchSettings from settings.py
DEFAULT_BUSINESS_BATCH_SIZE = 20
DEFAULT_RECORD_BATCH_SIZE   = 50

# Maximum records processed per single AI prompt call
AI_BATCH_PROMPT_MAX_RECORDS = 10


# ==============================================================================
# SECURITY
# ==============================================================================

class TokenType(str, Enum):
    """Token types stored and validated by token_manager.py."""
    ACCESS     = "access"
    REFRESH    = "refresh"
    ONBOARDING = "onboarding"


# Number of rounds for password hashing (bcrypt)
BCRYPT_ROUNDS = 12

# Webhook signature header sent by Razorpay
RAZORPAY_SIGNATURE_HEADER = "X-Razorpay-Signature"


# ==============================================================================
# FEEDBACK
# ==============================================================================

# Number of days after onboarding before sending the feedback request
FEEDBACK_SEND_AFTER_DAYS = 30


# ==============================================================================
# SYSTEM HEALTH
# ==============================================================================

# External connectivity check targets used by system_health.py
HEALTH_CHECK_URLS = [
    "https://api.openai.com",
    "https://graph.facebook.com",
    "https://api.razorpay.com",
]

# Database ping query used by system_health.py
DB_HEALTH_PING_QUERY = "SELECT 1"


# ==============================================================================
# LOGGING — SERVICE NAMES
# ==============================================================================

class ServiceName(str, Enum):
    """Service name tags used in structured log entries."""
    API            = "api"
    AI_REPLY       = "ai_reply_service"
    SENTIMENT      = "sentiment_service"
    ANALYTICS      = "analytics_service"
    SEO            = "seo_service"
    COMPETITOR     = "competitor_service"
    REPORTS        = "reports_service"
    CONTENT        = "content_generation_service"
    REVIEW_MONITOR = "review_monitor"
    SCHEDULER      = "scheduler_manager"
    PAYMENTS       = "payment_service"
    SUBSCRIPTION   = "subscription_service"
    WHATSAPP       = "whatsapp_service"
    ALERT          = "alert_manager"
    GOOGLE_REVIEWS = "google_reviews_client"
    GOOGLE_SHEETS  = "google_sheets_client"
    SYSTEM_HEALTH  = "system_health"
    FEEDBACK = "feedback_service"


# ==============================================================================
# DATE & TIME
# ==============================================================================

DATE_FORMAT         = "%Y-%m-%d"
DATETIME_FORMAT     = "%Y-%m-%dT%H:%M:%SZ"
DISPLAY_DATE_FORMAT = "%d %B %Y"


# ==============================================================================
# ERROR CODES
# ==============================================================================

class ErrorCode(str, Enum):
    """Machine-readable error codes returned in API error responses."""
    VALIDATION_ERROR      = "VALIDATION_ERROR"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    PERMISSION_DENIED     = "PERMISSION_DENIED"
    RESOURCE_NOT_FOUND    = "RESOURCE_NOT_FOUND"
    USAGE_LIMIT_EXCEEDED  = "USAGE_LIMIT_EXCEEDED"
    SUBSCRIPTION_INACTIVE = "SUBSCRIPTION_INACTIVE"
    PAYMENT_FAILED        = "PAYMENT_FAILED"
    EXTERNAL_API_ERROR    = "EXTERNAL_API_ERROR"
    DUPLICATE_REQUEST     = "DUPLICATE_REQUEST"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    RATE_LIMIT_EXCEEDED   = "RATE_LIMIT_EXCEEDED"
    JOB_ALREADY_RUNNING   = "JOB_ALREADY_RUNNING"


# ==============================================================================
# MISSING EXPORTS — added to satisfy all import contracts
# ==============================================================================

# Billing cycle enum used by subscription_service.py
class BillingCycle(str, Enum):
    """Billing cycle options. One tier only — billing_cycle is duration only."""
    MONTHLY = "monthly"
    ANNUAL  = "annual"

# Job name constants used by scheduler_manager.py and expiry_checker.py
class JobName(str, Enum):
    """Human-readable job name identifiers for lock keys and logging."""
    REVIEW_MONITOR       = "review_monitor"
    SALES_ANALYSIS       = "sales_analysis"
    WEEKLY_REPORT        = "weekly_report"
    MONTHLY_REPORT       = "monthly_report"
    QUARTERLY_REPORT     = "quarterly_report"
    WEEKLY_CONTENT       = "weekly_content"
    EXPIRY_CHECK         = "expiry_check"
    COMPETITOR_SCAN      = "competitor_scan"
    ALERT_DETECTION      = "alert_detection"
    FEEDBACK_REQUEST     = "feedback_request"
    HEALTH_REPORT        = "health_report"

# Flat daily limit constants used by plan_manager.py directly
MAX_REVIEWS_PER_DAY          = DAILY_USAGE_LIMITS["max_reviews_per_day"]
MAX_AI_REPLIES_PER_DAY       = DAILY_USAGE_LIMITS["max_ai_replies_per_day"]
MAX_COMPETITOR_SCANS_PER_DAY = DAILY_USAGE_LIMITS["max_competitor_scans_per_day"]

# Review text length constant used by formatting_utils.py
REVIEW_TEXT_MAX_LENGTH = REVIEW_MAX_TEXT_LENGTH_FOR_AI  # alias

# Sensitive field names used by error_tracker.py for log scrubbing
_SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset({
    "password", "token", "access_token", "refresh_token",
    "api_key", "secret", "razorpay_key_secret", "whatsapp_token",
    "openai_api_key", "phone", "email", "card_number",
})