# AI Business Agent

A production-grade SaaS backend that helps local businesses automate Google review replies, track sales insights, monitor competitors, and receive AI-generated reports — all delivered through WhatsApp.

---

## What It Does

- **Review Monitoring** — polls Google Reviews every 15 minutes and detects new reviews automatically
- **AI Review Replies** — generates sentiment-aware replies (positive / negative / neutral) using OpenAI GPT-4o and posts them to Google
- **Sales Analytics** — reads sales data from a connected Google Sheet and detects trends, peak days, and top products
- **Competitor Monitoring** — tracks nearby competitor ratings and alerts the business owner to any significant changes
- **SEO Suggestions** — generates local SEO keyword and improvement tips based on review signals
- **Social Media Content** — produces weekly ready-to-post content for Instagram, WhatsApp Status, and Google Business
- **Scheduled Reports** — delivers weekly, monthly, and quarterly performance reports via WhatsApp
- **Alert System** — real-time WhatsApp alerts for rating drops, review spikes, and sales anomalies
- **Subscription & Payments** — single-tier subscription processed via Razorpay (₹999/month or ₹9,999/year)

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115 |
| Database | PostgreSQL + SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| AI | OpenAI GPT-4o (via openai 2.x SDK) |
| Messaging | WhatsApp Cloud API (Meta Graph API) |
| Payments | Razorpay |
| Reviews | Google My Business API + Google Places API |
| Sales data | Google Sheets API v4 |
| Scheduler | APScheduler 3.x (async) |
| Auth | JWT (PyJWT) + AES-GCM encryption (cryptography) |
| Runtime | Python 3.11+ / Uvicorn |

---

## Project Structure

```
ai_business_agent/
├── app/
│   ├── main.py                     # FastAPI app entrypoint
│   ├── config/
│   │   ├── settings.py             # Environment variables & settings
│   │   └── constants.py            # Static constants
│   ├── api/
│   │   ├── router.py               # Master API router
│   │   └── routes/
│   │       ├── health_route.py     # GET /api/v1/health
│   │       ├── onboarding_route.py # Business registration & profile
│   │       ├── payment_route.py    # Payment initiation & status
│   │       └── webhook_route.py    # Razorpay webhook receiver
│   ├── database/
│   │   ├── db.py                   # Async engine & session factory
│   │   ├── base.py                 # Base ORM model
│   │   └── models/                 # SQLAlchemy ORM models
│   ├── repositories/               # All database queries
│   ├── services/                   # Core business logic
│   ├── integrations/               # Google, WhatsApp, external APIs
│   ├── schedulers/                 # Automated background jobs
│   ├── payments/                   # Razorpay payment processing
│   ├── subscriptions/              # Subscription lifecycle management
│   ├── notifications/              # WhatsApp notification delivery
│   ├── alerts/                     # Business event detection
│   ├── validators/                 # Input & data validation
│   ├── security/                   # JWT, encryption, auth middleware
│   ├── logging/                    # Structured logging & error tracking
│   ├── prompts/                    # AI prompt templates (plain text)
│   ├── jobs/                       # Background job helpers
│   ├── feedback/                   # Testimonial collection
│   └── utils/                      # Shared utilities
├── tests/
│   ├── test_api.py
│   ├── test_payments.py
│   ├── test_subscriptions.py
│   ├── test_services.py
│   └── test_integrations.py
├── scripts/
│   ├── create_admin.py             # Interactive admin setup
│   └── seed_demo_data.py           # Load demo businesses & reviews
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/ai-business-agent.git
cd ai-business-agent
```

### 2. Create and activate Conda environment

```bash
conda create -n ai_business_agent_env python=3.11
conda activate ai_business_agent_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in all required values. See the [Environment Variables](#environment-variables) section below for what each value means.

### 5. Set up the database

Create the PostgreSQL database, then run migrations:

```bash
alembic upgrade head
```

### 6. Configure the admin

```bash
python scripts/create_admin.py
```

This sets `ADMIN_WHATSAPP_NUMBER`, `ADMIN_EMAIL`, and `ADMIN_SECRET_TOKEN` in your `.env` file interactively.

### 7. (Optional) Load demo data

```bash
python scripts/seed_demo_data.py
```

This creates 3 demo businesses (restaurant, salon, clinic) with reviews, subscriptions, and usage records for development and testing.

### 8. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

Interactive API docs (development only — requires `APP_DEBUG=true`):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

All endpoints are versioned under `/api/v1/`.

### Health

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/health` | None | Basic liveness check |
| GET | `/api/v1/health/detailed` | None | DB + scheduler + API connectivity |

### Onboarding

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/onboarding/register` | None | Register a new business |
| GET | `/api/v1/onboarding/profile` | JWT | Get business profile |
| PATCH | `/api/v1/onboarding/profile` | JWT | Update business profile |
| POST | `/api/v1/onboarding/google` | JWT | Connect Google Business account |
| POST | `/api/v1/onboarding/complete` | JWT | Mark onboarding as complete |

### Payments

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/payments/initiate` | JWT | Create Razorpay order |
| GET | `/api/v1/payments/status/{order_id}` | JWT | Check payment status |
| GET | `/api/v1/payments/subscription` | JWT | Get current subscription |

### Webhooks

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/webhooks/razorpay` | HMAC signature | Receive Razorpay payment events |
| GET | `/api/v1/webhooks/razorpay/ping` | None | Webhook endpoint probe |

---

## Subscription & Pricing

Single tier — full access to all features on any active subscription.

| Billing cycle | Price |
|---|---|
| Monthly | ₹999 / month |
| Annual | ₹9,999 / year (~17% saving) |

Payments are processed via Razorpay. Subscription is activated automatically after Razorpay webhook confirmation.

---

## Scheduled Jobs

| Job | Schedule | Description |
|---|---|---|
| Review Monitor | Every 15 minutes | Polls Google Reviews for new reviews, triggers AI reply generation |
| Sales Analysis | Daily 7:00 AM | Reads Google Sheet, generates sales insights |
| Weekly Report | Monday 8:00 AM | Sends weekly review + sentiment + competitor summary via WhatsApp |
| Monthly Report | 1st of month 8:00 AM | Business performance summary via WhatsApp |
| Quarterly Report | 1st of Jan/Apr/Jul/Oct | Strategic business suggestions |
| Weekly Content | Monday 9:00 AM | Generates Instagram, WhatsApp, Google Business posts |
| Expiry Check | Daily 6:00 AM | Detects expiring subscriptions, sends renewal reminders |
| Health Report | Daily 8:00 AM | Sends system health summary to admin WhatsApp |

All times are UTC. Configure cron expressions in `.env` under the `SCHEDULER_*` keys.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below.

### Application

| Variable | Description |
|---|---|
| `APP_ENV` | `production` or `development` (use `development` locally) |
| `APP_SECRET_KEY` | Random secret key (minimum 32 characters) |
| `APP_DEBUG` | `true` enables Swagger UI. Always `false` in production |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins |

### Database

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL async URL — `postgresql+asyncpg://user:pass@host:5432/dbname` |
| `DATABASE_POOL_SIZE` | Connection pool size (default: 10) |

### OpenAI

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (`sk-...`) |
| `OPENAI_MODEL` | Model to use — recommended: `gpt-4o` |
| `EXTERNAL_API_TIMEOUT_SECONDS` | Timeout (in seconds) for outbound HTTP calls (OpenAI, Google, WhatsApp, Razorpay, etc.) |

### Razorpay

| Variable | Description |
|---|---|
| `RAZORPAY_KEY_ID` | Razorpay API key ID |
| `RAZORPAY_KEY_SECRET` | Razorpay API secret |
| `RAZORPAY_WEBHOOK_SECRET` | Webhook signing secret from Razorpay dashboard |

### WhatsApp

| Variable | Description |
|---|---|
| `WHATSAPP_API_TOKEN` | WhatsApp Cloud API access token |
| `WHATSAPP_PHONE_NUMBER_ID` | Phone number ID from Meta Business dashboard |

### Google

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google API key for Places API |
| `GOOGLE_CLIENT_ID` | OAuth2 client ID for Google login |
| `GOOGLE_CLIENT_SECRET` | OAuth2 client secret |
| `GOOGLE_PLACES_API_KEY` | Google Places API key for review fetching |
| `GOOGLE_MY_BUSINESS_ACCESS_TOKEN` | OAuth2 access token for Google My Business API (optional in development; required for live review fetching/posting) |

### Admin

| Variable | Description |
|---|---|
| `ADMIN_WHATSAPP_NUMBER` | WhatsApp number that receives system alerts (E.164 format) |
| `ADMIN_EMAIL` | Admin email address |
| `ADMIN_SECRET_TOKEN` | Secret token for internal admin API endpoints (min 16 chars) |

### Security

| Variable | Description |
|---|---|
| `JWT_SECRET_KEY` | JWT signing secret (minimum 32 characters) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiry (default: 60) |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiry (default: 30) |
| `ENCRYPTION_KEY` | AES-GCM key for encrypting sensitive DB fields |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_api.py -v
pytest tests/test_payments.py -v
pytest tests/test_subscriptions.py -v
pytest tests/test_services.py -v
pytest tests/test_integrations.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

Coverage report is saved to `htmlcov/index.html`.

---

## Key Engineering Rules

- **Multi-tenant isolation** — every database query filters by `business_id`. Cross-business data access is forbidden.
- **Single subscription tier** — all paying businesses get full feature access. There are no plan tiers or feature gates.
- **Idempotency** — all critical operations (review replies, payment processing, scheduled jobs) are protected against duplicate execution.
- **Failsafe jobs** — every background job runs inside a failsafe wrapper. A failure in one business never affects other businesses.
- **Retry policy** — all external API calls retry up to 3 times with exponential backoff (2s → 5s → 10s).
- **Prompt safety** — AI prompts never include phone numbers, payment data, or internal admin notes.
- **No raw SQL** — all database operations use SQLAlchemy ORM through the repository layer.

---

## Architecture

```
HTTP Request
    │
    ▼
API Layer (FastAPI routes)
    │  validates input, calls service
    ▼
Service Layer (business logic)
    │  orchestrates workflow
    ├──► Repository Layer (database queries only)
    │        │
    │        ▼
    │    PostgreSQL (SQLAlchemy async)
    │
    └──► Integration Layer (external APIs)
             ├── Google Reviews / Places API
             ├── Google Sheets API
             ├── WhatsApp Cloud API
             ├── OpenAI API
             └── Razorpay API

Background Scheduler (APScheduler)
    └── triggers services on cron schedule
        independently of HTTP request cycle
```

---

## Demo Data

After running `python scripts/seed_demo_data.py`, three demo businesses are available:

| Business | Type | City |
|---|---|---|
| Spice Garden Restaurant | Restaurant | Bangalore |
| Glamour Salon & Spa | Salon | Mumbai |
| Wellness First Clinic | Clinic | Hyderabad |

Each demo business has 10 reviews (mix of positive, neutral, negative), an active monthly subscription, and pre-generated AI replies for half the reviews.

To reset and re-seed:

```bash
python scripts/seed_demo_data.py --reset
```

---

## License

Private and proprietary. All rights reserved.