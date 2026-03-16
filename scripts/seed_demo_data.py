#!/usr/bin/env python3
# ==============================================================================
# File: scripts/seed_demo_data.py
# Purpose: Populate the database with realistic demo data for development,
#          testing, and manual QA of the AI Business Agent platform.
#
#          Creates:
#            - 3 demo businesses (restaurant, salon, clinic)
#            - 1 active monthly subscription per business
#            - 10 reviews per business (mix of positive, negative, neutral)
#              with AI replies already generated for half of them
#            - Seeded usage records for today
#
#          Design:
#            - Idempotent — safe to run multiple times.
#              Businesses are matched by owner_email; if the record already
#              exists the script skips creation and reuses the existing ID.
#            - All data is realistic for Indian local businesses.
#            - No external API calls — all data is pre-defined.
#            - Progress is printed to stdout with clear section headers.
#
#          Usage:
#            conda activate ai_business_agent_env
#            python scripts/seed_demo_data.py
#
#            To wipe existing demo data and re-seed:
#            python scripts/seed_demo_data.py --reset
#
#          Requirements:
#            - DATABASE_URL must be set in .env
#            - Tables must already exist (run: alembic upgrade head first)
# ==============================================================================

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path before any app imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load .env before importing settings
# ---------------------------------------------------------------------------
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Now safe to import app modules
# ---------------------------------------------------------------------------
from sqlalchemy import text                                     # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession                 # noqa: E402

from app.config.constants import (                             # noqa: E402
    BillingCycle,
    ReviewSentiment,
    ReviewStatus,
    SubscriptionStatus,
)
from app.database.db import AsyncSessionFactory                # noqa: E402
from app.database.models.business_model import BusinessModel   # noqa: E402
from app.database.models.review_model import ReviewModel       # noqa: E402
from app.database.models.subscription_model import SubscriptionModel  # noqa: E402
from app.database.models.usage_model import UsageModel         # noqa: E402
from app.repositories.business_repository import BusinessRepository    # noqa: E402
from app.repositories.review_repository import ReviewRepository        # noqa: E402
from app.repositories.subscription_repository import SubscriptionRepository  # noqa: E402
from app.repositories.usage_repository import UsageRepository          # noqa: E402

# ==============================================================================
# ANSI helpers
# ==============================================================================
_IS_TTY = sys.stdout.isatty()


def _green(t: str) -> str:  return f"\033[32m{t}\033[0m" if _IS_TTY else t
def _yellow(t: str) -> str: return f"\033[33m{t}\033[0m" if _IS_TTY else t
def _red(t: str) -> str:    return f"\033[31m{t}\033[0m" if _IS_TTY else t
def _bold(t: str) -> str:   return f"\033[1m{t}\033[0m"  if _IS_TTY else t
def _dim(t: str) -> str:    return f"\033[2m{t}\033[0m"  if _IS_TTY else t


def _ok(msg: str)   -> None: print(f"  {_green('✓')} {msg}")
def _skip(msg: str) -> None: print(f"  {_yellow('→')} {msg}")
def _info(msg: str) -> None: print(f"  {_dim('·')} {msg}")
def _err(msg: str)  -> None: print(f"  {_red('✗')} {msg}")


def _section(title: str) -> None:
    print()
    print(f"  {_bold('──')} {_bold(title)}")
    print()


# ==============================================================================
# Demo data definitions
# ==============================================================================

_NOW    = datetime.now(tz=timezone.utc)
_TODAY  = _NOW.date()


# ---------------------------------------------------------------------------
# Business definitions
# ---------------------------------------------------------------------------

DEMO_BUSINESSES: list[dict] = [
    {
        "business_name":          "Spice Garden Restaurant",
        "owner_name":             "Ramesh Nair",
        "owner_email":            "demo.spicegarden@aiagent.local",
        "owner_whatsapp_number":  "+919800000001",
        "business_type":          "restaurant",
        "business_description":   "Authentic South Indian cuisine in the heart of Bangalore",
        "city":                   "Bangalore",
        "state":                  "Karnataka",
        "google_place_id":        "ChIJDEMO_SPICE_GARDEN_001",
        "current_google_rating":  4.3,
        "total_google_reviews":   128,
        "is_onboarding_complete": True,
        "is_google_connected":    True,
    },
    {
        "business_name":          "Glamour Salon & Spa",
        "owner_name":             "Priya Sharma",
        "owner_email":            "demo.glamour@aiagent.local",
        "owner_whatsapp_number":  "+919800000002",
        "business_type":          "salon",
        "business_description":   "Premium unisex salon offering hair, skin and spa treatments",
        "city":                   "Mumbai",
        "state":                  "Maharashtra",
        "google_place_id":        "ChIJDEMO_GLAMOUR_SALON_002",
        "current_google_rating":  4.6,
        "total_google_reviews":   87,
        "is_onboarding_complete": True,
        "is_google_connected":    True,
    },
    {
        "business_name":          "Wellness First Clinic",
        "owner_name":             "Dr. Suresh Kumar",
        "owner_email":            "demo.wellness@aiagent.local",
        "owner_whatsapp_number":  "+919800000003",
        "business_type":          "clinic",
        "business_description":   "General practice and preventive healthcare clinic",
        "city":                   "Hyderabad",
        "state":                  "Telangana",
        "google_place_id":        "ChIJDEMO_WELLNESS_CLINIC_003",
        "current_google_rating":  4.1,
        "total_google_reviews":   54,
        "is_onboarding_complete": True,
        "is_google_connected":    True,
    },
]


# ---------------------------------------------------------------------------
# Review templates — keyed by business_type
# ---------------------------------------------------------------------------

_REVIEWS: dict[str, list[dict]] = {
    "restaurant": [
        # Positive
        {
            "google_review_id": "grev_sg_001",
            "rating": 5,
            "reviewer_name": "Anita Reddy",
            "review_text": "The biryani here is absolutely outstanding! Perfect spices and the service was prompt. Highly recommend the mutton curry as well. Will definitely come back with family.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you so much, Anita! We're thrilled to hear you loved our biryani and mutton curry. Our chef will be delighted by your kind words. We look forward to welcoming you and your family again soon!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 2,
        },
        {
            "google_review_id": "grev_sg_002",
            "rating": 5,
            "reviewer_name": "Karan Mehta",
            "review_text": "Best dosa in Bangalore! Crispy, perfectly fermented batter and the chutneys are homemade. The staff is very friendly and attentive.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you for your wonderful review, Karan! We take great pride in our homemade chutneys and traditional dosa preparation. Your feedback means the world to our team. See you again soon!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 5,
        },
        {
            "google_review_id": "grev_sg_003",
            "rating": 4,
            "reviewer_name": "Deepa Iyer",
            "review_text": "Great food and ambience. The thali was filling and delicious. Waiting time was a bit long on weekends but understandable given how popular this place is.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 7,
        },
        {
            "google_review_id": "grev_sg_004",
            "rating": 4,
            "reviewer_name": "Vikram Shetty",
            "review_text": "Solid South Indian food with authentic flavours. Rasam was excellent. Parking can be tricky but the food is worth it.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 9,
        },
        # Neutral
        {
            "google_review_id": "grev_sg_005",
            "rating": 3,
            "reviewer_name": "Sunita Rao",
            "review_text": "Average experience overall. Food was okay, nothing exceptional. Service was decent. Prices are fair for the portion size.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": "Thank you for sharing your feedback, Sunita. We appreciate your honest review and are always looking to improve. We hope to offer you a more memorable experience on your next visit.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 12,
        },
        {
            "google_review_id": "grev_sg_006",
            "rating": 3,
            "reviewer_name": "Rajat Gupta",
            "review_text": "Decent place. Food is consistent but nothing new on the menu in a while. Would love to see some seasonal specials.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 15,
        },
        # Negative
        {
            "google_review_id": "grev_sg_007",
            "rating": 2,
            "reviewer_name": "Pooja Nambiar",
            "review_text": "Very disappointing visit. The sambar was watery and the food was served cold. Waited 40 minutes for the order. Staff seemed indifferent.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": "We sincerely apologise for your disappointing experience, Pooja. Cold food and long waiting times are completely unacceptable and do not reflect our standards. Please allow us to make it right — contact us directly and we would love to invite you back for a complimentary meal.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 18,
        },
        {
            "google_review_id": "grev_sg_008",
            "rating": 1,
            "reviewer_name": "Arjun Pillai",
            "review_text": "Found a foreign object in my food. Very unhygienic. Raised the issue but staff was dismissive. Will not return.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 20,
        },
        {
            "google_review_id": "grev_sg_009",
            "rating": 2,
            "reviewer_name": "Meena Subramaniam",
            "review_text": "Quality has gone down compared to last year. Portions are smaller and prices have gone up. Not worth the money anymore.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 22,
        },
        {
            "google_review_id": "grev_sg_010",
            "rating": 4,
            "reviewer_name": "Nikhil Joshi",
            "review_text": "Really enjoyed the Sunday special. Curd rice and pickle combination was divine. Will visit again on weekends.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you, Nikhil! Our Sunday specials are crafted with extra love and we're so happy you enjoyed the curd rice. Looking forward to seeing you again this weekend!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 25,
        },
    ],

    "salon": [
        {
            "google_review_id": "grev_gl_001",
            "rating": 5,
            "reviewer_name": "Kavya Menon",
            "review_text": "Incredible experience! The balayage turned out exactly as I had envisioned. Stylist was professional and very patient explaining the process.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you so much, Kavya! We're so happy your balayage came out exactly how you imagined. Our stylists love bringing your vision to life. We can't wait to see you again!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 1,
        },
        {
            "google_review_id": "grev_gl_002",
            "rating": 5,
            "reviewer_name": "Rohini Desai",
            "review_text": "Best facial I've had in Mumbai! The therapist was gentle and the products used were premium quality. Skin is glowing. Will be a regular customer.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "We're overjoyed to hear that, Rohini! A glowing complexion is exactly what we aim for. Our therapists use only the best products and your kind words motivate us to keep up the great work. Welcome to the Glamour family!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 4,
        },
        {
            "google_review_id": "grev_gl_003",
            "rating": 4,
            "reviewer_name": "Shweta Kulkarni",
            "review_text": "Good salon with skilled staff. Hair cut and colour was nicely done. A bit pricey compared to nearby salons but the quality justifies it.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 6,
        },
        {
            "google_review_id": "grev_gl_004",
            "rating": 5,
            "reviewer_name": "Neha Patil",
            "review_text": "Loved the ambience and the staff was so welcoming. Got a keratin treatment and my hair feels amazing. Booking was easy through their app.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 8,
        },
        {
            "google_review_id": "grev_gl_005",
            "rating": 3,
            "reviewer_name": "Tanya Singh",
            "review_text": "Service was okay. The manicure was fine but not as detailed as I expected for the price. Staff were friendly though.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": "Thank you for your honest feedback, Tanya! We'd love to ensure your next manicure exceeds expectations. Please ask for our senior nail technician on your next visit — we're confident you'll notice the difference!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 11,
        },
        {
            "google_review_id": "grev_gl_006",
            "rating": 3,
            "reviewer_name": "Shruti Verma",
            "review_text": "Decent salon. The haircut was fine but the waiting time was nearly an hour even with an appointment. They need to manage bookings better.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 14,
        },
        {
            "google_review_id": "grev_gl_007",
            "rating": 2,
            "reviewer_name": "Divya Krishnan",
            "review_text": "Very unhappy with my hair colour. Asked for a warm brown and got something much darker. They tried to fix it but it's still not what I wanted.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": "We are truly sorry to hear about your colour experience, Divya. This is not the outcome we strive for. Please reach out to us directly and we will arrange a complimentary correction session with our colour specialist at your earliest convenience.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 17,
        },
        {
            "google_review_id": "grev_gl_008",
            "rating": 1,
            "reviewer_name": "Anjali Bose",
            "review_text": "Absolutely ruined my hair! Used cheap bleach that damaged my hair severely. Scalp is still irritated. Complete waste of money.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 19,
        },
        {
            "google_review_id": "grev_gl_009",
            "rating": 2,
            "reviewer_name": "Poornima Hegde",
            "review_text": "Staff was rude when I asked for a change. Expected much better customer service for the prices they charge.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 21,
        },
        {
            "google_review_id": "grev_gl_010",
            "rating": 5,
            "reviewer_name": "Madhuri Rao",
            "review_text": "Absolutely love this salon! Been coming here for 2 years and the quality is always consistent. Highly recommend the hair spa treatment.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Two years of loyalty means the world to us, Madhuri! Thank you for your continued trust and for recommending our hair spa. We look forward to many more years of keeping your hair looking gorgeous!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 24,
        },
    ],

    "clinic": [
        {
            "google_review_id": "grev_wc_001",
            "rating": 5,
            "reviewer_name": "Suresh Babu",
            "review_text": "Dr. Kumar is an excellent doctor who listens patiently. Diagnosis was accurate and the prescribed medicines worked quickly. Reception staff is very organised.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you for your kind words, Suresh! Patient care and attentive listening are at the heart of what we do. We're glad the treatment worked well for you. Please take care of yourself and do visit us whenever needed.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 2,
        },
        {
            "google_review_id": "grev_wc_002",
            "rating": 5,
            "reviewer_name": "Lalitha Prasad",
            "review_text": "Very clean clinic with modern equipment. The health check-up package was thorough and the report was easy to understand. Highly recommended.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Thank you so much, Lalitha! Hygiene and clarity of communication are our top priorities. We are thrilled the health check-up package was helpful for you. Wishing you continued good health!",
            "status": ReviewStatus.REPLIED,
            "days_ago": 5,
        },
        {
            "google_review_id": "grev_wc_003",
            "rating": 4,
            "reviewer_name": "Ravi Teja",
            "review_text": "Good clinic overall. Doctor is knowledgeable and approachable. Waiting time can be long on some days but the quality of care is worth it.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 8,
        },
        {
            "google_review_id": "grev_wc_004",
            "rating": 4,
            "reviewer_name": "Padma Venkat",
            "review_text": "Compassionate doctor and helpful staff. The follow-up call after my treatment was a nice touch and shows they genuinely care.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 10,
        },
        {
            "google_review_id": "grev_wc_005",
            "rating": 3,
            "reviewer_name": "Mohan Das",
            "review_text": "Average experience. Doctor was okay but consultation felt rushed. Did not get enough time to explain all my symptoms properly.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": "Thank you for sharing your experience, Mohan. We are sorry the consultation felt hurried — that is something we take seriously. Please request an extended consultation slot at your next visit so we can address all your concerns thoroughly.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 13,
        },
        {
            "google_review_id": "grev_wc_006",
            "rating": 3,
            "reviewer_name": "Usha Narayanan",
            "review_text": "Clinic is clean and staff are polite. Waited nearly 45 minutes past my appointment time which was frustrating. The consultation itself was good.",
            "sentiment": ReviewSentiment.NEUTRAL,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 16,
        },
        {
            "google_review_id": "grev_wc_007",
            "rating": 2,
            "reviewer_name": "Ganesh Iyer",
            "review_text": "Prescribed medicines that I was allergic to despite me mentioning it at the start. Careless approach. Had to visit another doctor to sort it out.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": "We are extremely sorry to hear about your experience, Ganesh. Patient safety is our absolute priority and what you have described is deeply concerning to us. Please contact us directly so we can address this matter seriously and ensure it never happens again.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 18,
        },
        {
            "google_review_id": "grev_wc_008",
            "rating": 1,
            "reviewer_name": "Sreekanth Pillai",
            "review_text": "Very poor experience. Billing was incorrect and when I raised it the receptionist was dismissive. No one followed up despite my complaint.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 20,
        },
        {
            "google_review_id": "grev_wc_009",
            "rating": 2,
            "reviewer_name": "Hemalatha Reddy",
            "review_text": "Parking is nearly impossible and the clinic is difficult to find. The doctor was decent but the overall experience was frustrating.",
            "sentiment": ReviewSentiment.NEGATIVE,
            "ai_reply": None,
            "status": ReviewStatus.NEW,
            "days_ago": 23,
        },
        {
            "google_review_id": "grev_wc_010",
            "rating": 5,
            "reviewer_name": "Balasubramaniam S.",
            "review_text": "Have been visiting this clinic for 3 years. Dr. Kumar is thorough, patient and genuinely cares about your wellbeing. Excellent clinic.",
            "sentiment": ReviewSentiment.POSITIVE,
            "ai_reply": "Three years of trust — thank you so much! Continuity of care and building a genuine doctor-patient relationship is what we cherish most. We look forward to supporting your health for many more years to come.",
            "status": ReviewStatus.REPLIED,
            "days_ago": 26,
        },
    ],
}


# ==============================================================================
# Core seeding functions
# ==============================================================================

async def _get_or_create_business(
    db: AsyncSession,
    repo: BusinessRepository,
    defn: dict,
) -> tuple[BusinessModel, bool]:
    """
    Return an existing business by email, or create a new one.

    Returns:
        (BusinessModel, created: bool)
    """
    existing = await repo.get_by_email(db, defn["owner_email"])
    if existing:
        return existing, False

    business = await repo.create(
        db,
        business_name         = defn["business_name"],
        owner_name            = defn["owner_name"],
        owner_email           = defn["owner_email"],
        owner_whatsapp_number = defn["owner_whatsapp_number"],
        business_type         = defn.get("business_type"),
        business_description  = defn.get("business_description"),
        city                  = defn.get("city"),
        state                 = defn.get("state"),
    )

    # Apply extra fields that create() doesn't accept directly
    extra_fields = [
        "google_place_id", "current_google_rating", "total_google_reviews",
        "is_onboarding_complete", "is_google_connected",
    ]
    for field in extra_fields:
        if field in defn:
            setattr(business, field, defn[field])

    await db.flush()
    return business, True


async def _get_or_create_subscription(
    db: AsyncSession,
    repo: SubscriptionRepository,
    business_id: uuid.UUID,
) -> tuple[SubscriptionModel, bool]:
    """
    Return an existing active subscription, or create a new monthly one.

    Returns:
        (SubscriptionModel, created: bool)
    """
    existing = await repo.get_active(db, business_id)
    if existing:
        return existing, False

    starts_at  = _NOW - timedelta(days=10)
    expires_at = starts_at + timedelta(days=30)

    sub = await repo.create(
        db,
        business_id          = business_id,
        billing_cycle        = BillingCycle.MONTHLY,
        billing_cycle_months = 1,
        amount               = 699.0,
        currency             = "INR",
        status               = SubscriptionStatus.ACTIVE,
        starts_at            = starts_at,
        expires_at           = expires_at,
        auto_renew           = True,
        notes                = "Demo subscription — seeded by seed_demo_data.py",
    )
    return sub, True


async def _seed_reviews(
    db: AsyncSession,
    repo: ReviewRepository,
    business: BusinessModel,
    reviews: list[dict],
) -> tuple[int, int]:
    """
    Upsert reviews for a business. Applies sentiment and AI reply where defined.

    Returns:
        (created_count, skipped_count)
    """
    created = skipped = 0

    for r in reviews:
        published_at = _NOW - timedelta(days=r["days_ago"])

        review, is_new = await repo.upsert(
            db,
            business_id      = business.id,
            google_review_id = r["google_review_id"],
            rating           = r["rating"],
            reviewer_name    = r.get("reviewer_name"),
            review_text      = r.get("review_text"),
            google_place_id  = business.google_place_id,
            published_at     = published_at,
        )

        if not is_new:
            skipped += 1
            continue

        # Apply sentiment
        if r.get("sentiment") and review:
            await repo.update_sentiment(
                db,
                review_id = review.id,
                sentiment = r["sentiment"],
                score     = _sentiment_to_score(r["sentiment"]),
            )

        # Apply AI reply if present
        if r.get("ai_reply") and r.get("status") == ReviewStatus.REPLIED and review:
            await repo.save_ai_reply(
                db,
                review_id       = review.id,
                ai_reply        = r["ai_reply"],
                prompt_used     = f"{r['sentiment']}_review_reply_prompt.txt",
                idempotency_key = f"SEED_{business.id}_{r['google_review_id']}",
            )

        created += 1

    return created, skipped


def _sentiment_to_score(sentiment: str) -> float:
    """Map sentiment label to a representative confidence score."""
    return {
        ReviewSentiment.POSITIVE: 0.85,
        ReviewSentiment.NEUTRAL:  0.00,
        ReviewSentiment.NEGATIVE: -0.85,
    }.get(sentiment, 0.0)


async def _seed_usage(
    db: AsyncSession,
    repo: UsageRepository,
    business_id: uuid.UUID,
) -> bool:
    """
    Ensure today's usage record exists for the business.

    Returns:
        True if created, False if already existed.
    """
    record = await repo.get_or_create_today(db, business_id)
    created = record.reviews_processed == 0  # heuristic — just created
    return created


# ==============================================================================
# Reset helper
# ==============================================================================

async def _reset_demo_data(db: AsyncSession) -> None:
    """
    Delete all demo records identified by the demo email domain.

    Cascades to subscriptions, reviews, and usage via FK constraints.
    """
    print()
    print(f"  {_yellow('!')} Resetting demo data (demo email domain: aiagent.local)...")

    # Collect business IDs to clean dependent tables
    result = await db.execute(
        text("SELECT id FROM businesses WHERE owner_email LIKE '%@aiagent.local'")
    )
    ids = [str(row[0]) for row in result.fetchall()]

    if not ids:
        _info("No demo businesses found — nothing to reset.")
        return

    id_list = ", ".join(f"'{bid}'" for bid in ids)

    await db.execute(text(f"DELETE FROM usage WHERE business_id IN ({id_list})"))
    await db.execute(text(f"DELETE FROM reviews WHERE business_id IN ({id_list})"))
    await db.execute(text(f"DELETE FROM subscriptions WHERE business_id IN ({id_list})"))
    await db.execute(text(f"DELETE FROM businesses WHERE id IN ({id_list})"))
    await db.commit()

    _ok(f"Deleted {len(ids)} demo business(es) and all related records.")


# ==============================================================================
# Main
# ==============================================================================

def _print_banner() -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  AI Business Agent — Demo Data Seeder"))
    print(_bold("=" * 60))
    print()
    print("  Populates the database with 3 demo businesses,")
    print("  subscriptions, reviews and usage records.")
    print()


async def main(reset: bool = False) -> None:
    _print_banner()

    # ------------------------------------------------------------------
    # Validate DATABASE_URL
    # ------------------------------------------------------------------
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        _err("DATABASE_URL is not set. Add it to .env and try again.")
        sys.exit(1)

    _info(f"Database: {database_url.split('@')[-1]}")  # host/db only — no credentials

    # ------------------------------------------------------------------
    # Connect and optionally reset
    # ------------------------------------------------------------------
    async with AsyncSessionFactory() as db:
        if reset:
            await _reset_demo_data(db)

        business_repo      = BusinessRepository()
        subscription_repo  = SubscriptionRepository()
        review_repo        = ReviewRepository()
        usage_repo         = UsageRepository()

        total_businesses_created     = 0
        total_subscriptions_created  = 0
        total_reviews_created        = 0
        total_reviews_skipped        = 0

        # ------------------------------------------------------------------
        # Seed each business
        # ------------------------------------------------------------------
        for defn in DEMO_BUSINESSES:
            _section(f"{defn['business_name']} ({defn['business_type'].title()}, {defn['city']})")

            # Business
            business, b_created = await _get_or_create_business(db, business_repo, defn)
            if b_created:
                _ok(f"Business created  — ID: {business.id}")
                total_businesses_created += 1
            else:
                _skip(f"Business already exists — ID: {business.id}")

            # Subscription
            sub, s_created = await _get_or_create_subscription(db, subscription_repo, business.id)
            if s_created:
                _ok(f"Subscription created (monthly ₹699) — ID: {sub.id}")
                total_subscriptions_created += 1
            else:
                _skip(f"Subscription already exists — ID: {sub.id}")

            # Reviews
            reviews_for_type = _REVIEWS.get(defn["business_type"], [])
            r_created, r_skipped = await _seed_reviews(db, review_repo, business, reviews_for_type)
            if r_created > 0:
                _ok(f"Reviews seeded — {r_created} created, {r_skipped} skipped")
            else:
                _skip(f"Reviews already exist — {r_skipped} skipped")
            total_reviews_created += r_created
            total_reviews_skipped += r_skipped

            # Usage
            await _seed_usage(db, usage_repo, business.id)
            _ok("Usage record ready for today")

        # ------------------------------------------------------------------
        # Commit everything
        # ------------------------------------------------------------------
        await db.commit()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _section("Seeding Complete")
    _ok(f"Businesses created  : {total_businesses_created}")
    _ok(f"Subscriptions created: {total_subscriptions_created}")
    _ok(f"Reviews created     : {total_reviews_created}  (skipped: {total_reviews_skipped})")
    print()
    print("  Demo login details:")
    print()
    for defn in DEMO_BUSINESSES:
        print(f"    {_bold(defn['business_name'])}")
        print(f"      Email    : {_yellow(defn['owner_email'])}")
        print(f"      WhatsApp : {defn['owner_whatsapp_number']}")
        print()

    print(f"  {_dim('Tip: run with --reset to wipe and re-seed all demo data.')}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed the AI Business Agent database with demo data."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing demo data before seeding.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(reset=args.reset))
    except KeyboardInterrupt:
        print()
        print(f"  {_red('✗')} Seeding interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print()
        print(f"  {_red('✗')} Seeding failed: {exc}")
        raise