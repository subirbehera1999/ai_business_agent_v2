# ==============================================================================
# File: app/repositories/business_repository.py
# Purpose: Repository class encapsulating all database operations for the
#          BusinessModel. This is the only layer permitted to query the
#          businesses table directly. Services must never issue raw database
#          queries — they call this repository instead.
#
#          All queries enforce:
#            - business_id isolation (multi-tenant safety)
#            - soft-delete exclusion (is_deleted = false)
#            - query limits / pagination (no unbounded SELECT *)
#            - structured error logging on failure
# ==============================================================================

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import ServiceName
from app.database.models.business_model import BusinessModel

logger = logging.getLogger(ServiceName.API)


# ==============================================================================
# Business Repository
# ==============================================================================

class BusinessRepository:
    """
    Handles all database operations for BusinessModel.

    Every public method accepts an AsyncSession injected from the service
    layer. The repository does not manage transactions — commit/rollback
    is the responsibility of the caller (service layer or FastAPI dependency).

    All read queries exclude soft-deleted records by default.
    Methods that intentionally include deleted records are explicitly named.

    Usage:
        repo = BusinessRepository()
        business = await repo.get_by_id(db, business_id)
    """

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        db: AsyncSession,
        *,
        business_name: str,
        owner_name: str,
        owner_email: str,
        owner_whatsapp_number: str,
        country: str = "India",
        timezone: str = "Asia/Kolkata",
        business_type: Optional[str] = None,
        business_description: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
    ) -> BusinessModel:
        """
        Create and persist a new business record.

        Args:
            db:                      Active async database session.
            business_name:           Registered name of the business.
            owner_name:              Full name of the business owner.
            owner_email:             Owner email — must be unique across businesses.
            owner_whatsapp_number:   WhatsApp number with country code.
            country:                 Country of operation (default: India).
            timezone:                IANA timezone string (default: Asia/Kolkata).
            business_type:           Optional business category.
            business_description:    Optional short description.
            city:                    Optional city.
            state:                   Optional state/province.

        Returns:
            BusinessModel: The newly created and persisted business instance.

        Raises:
            IntegrityError: If owner_email or owner_whatsapp_number already exists.
            SQLAlchemyError: On any other database error.
        """
        try:
            business = BusinessModel(
                business_name=business_name,
                owner_name=owner_name,
                owner_email=owner_email.lower().strip(),
                owner_whatsapp_number=owner_whatsapp_number.strip(),
                country=country,
                timezone=timezone,
                business_type=business_type,
                business_description=business_description,
                city=city,
                state=state,
            )
            db.add(business)
            await db.flush()  # Populate id and server defaults without committing

            logger.info(
                "Business record created",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business.id),
                    "owner_email": owner_email,
                },
            )
            return business

        except IntegrityError as exc:
            logger.error(
                "Business creation failed — duplicate email or WhatsApp number",
                extra={
                    "service": ServiceName.API,
                    "owner_email": owner_email,
                    "error": str(exc),
                },
            )
            raise

        except SQLAlchemyError as exc:
            logger.error(
                "Business creation failed — database error",
                extra={
                    "service": ServiceName.API,
                    "owner_email": owner_email,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Single Record ───────────────────────────────────────────────────

    async def get_by_id(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> Optional[BusinessModel]:
        """
        Fetch a single active business by its primary key.

        Args:
            db:           Active async database session.
            business_id:  UUID primary key of the business.

        Returns:
            BusinessModel if found and not soft-deleted, else None.
        """
        try:
            result = await db.execute(
                select(BusinessModel).where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch business by ID",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_by_email(
        self,
        db: AsyncSession,
        owner_email: str,
    ) -> Optional[BusinessModel]:
        """
        Fetch a single active business by owner email address.

        Email is normalised to lowercase before comparison.

        Args:
            db:           Active async database session.
            owner_email:  Owner email address to search.

        Returns:
            BusinessModel if found and not soft-deleted, else None.
        """
        try:
            result = await db.execute(
                select(BusinessModel).where(
                    BusinessModel.owner_email == owner_email.lower().strip(),
                    BusinessModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch business by email",
                extra={
                    "service": ServiceName.API,
                    "owner_email": owner_email,
                    "error": str(exc),
                },
            )
            raise

    async def get_by_whatsapp_number(
        self,
        db: AsyncSession,
        whatsapp_number: str,
    ) -> Optional[BusinessModel]:
        """
        Fetch a single active business by owner WhatsApp number.

        Args:
            db:               Active async database session.
            whatsapp_number:  WhatsApp number including country code.

        Returns:
            BusinessModel if found and not soft-deleted, else None.
        """
        try:
            result = await db.execute(
                select(BusinessModel).where(
                    BusinessModel.owner_whatsapp_number == whatsapp_number.strip(),
                    BusinessModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch business by WhatsApp number",
                extra={
                    "service": ServiceName.API,
                    "whatsapp_number": whatsapp_number,
                    "error": str(exc),
                },
            )
            raise

    async def get_by_google_place_id(
        self,
        db: AsyncSession,
        google_place_id: str,
    ) -> Optional[BusinessModel]:
        """
        Fetch a single active business by its Google Place ID.

        Args:
            db:               Active async database session.
            google_place_id:  Google Places identifier string.

        Returns:
            BusinessModel if found and not soft-deleted, else None.
        """
        try:
            result = await db.execute(
                select(BusinessModel).where(
                    BusinessModel.google_place_id == google_place_id.strip(),
                    BusinessModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch business by Google Place ID",
                extra={
                    "service": ServiceName.API,
                    "google_place_id": google_place_id,
                    "error": str(exc),
                },
            )
            raise

    # ── Read — Collections ─────────────────────────────────────────────────────

    async def get_all_active(
        self,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
    ) -> list[BusinessModel]:
        """
        Fetch a paginated batch of active, non-deleted businesses.

        Used by scheduler jobs to iterate through businesses in controlled
        batches. Never fetches all businesses at once.

        Args:
            db:      Active async database session.
            limit:   Maximum records to return (default 20, max enforced by caller).
            offset:  Pagination offset.

        Returns:
            list[BusinessModel]: Active businesses in the requested page.
        """
        try:
            result = await db.execute(
                select(BusinessModel)
                .where(
                    BusinessModel.is_active.is_(True),
                    BusinessModel.is_deleted.is_(False),
                )
                .order_by(BusinessModel.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch active businesses",
                extra={
                    "service": ServiceName.API,
                    "limit": limit,
                    "offset": offset,
                    "error": str(exc),
                },
            )
            raise

    async def get_all_google_connected(
        self,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
    ) -> list[BusinessModel]:
        """
        Fetch a paginated batch of active businesses with Google connected.

        Used by review_monitor.py to process only businesses that have a
        linked Google Business account.

        Args:
            db:      Active async database session.
            limit:   Maximum records to return.
            offset:  Pagination offset.

        Returns:
            list[BusinessModel]: Google-connected active businesses.
        """
        try:
            result = await db.execute(
                select(BusinessModel)
                .where(
                    BusinessModel.is_active.is_(True),
                    BusinessModel.is_deleted.is_(False),
                    BusinessModel.is_google_connected.is_(True),
                    BusinessModel.google_place_id.isnot(None),
                )
                .order_by(BusinessModel.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch Google-connected businesses",
                extra={
                    "service": ServiceName.API,
                    "limit": limit,
                    "offset": offset,
                    "error": str(exc),
                },
            )
            raise

    async def get_pending_feedback_request(
        self,
        db: AsyncSession,
        registered_before: datetime,
        limit: int = 20,
    ) -> list[BusinessModel]:
        """
        Fetch active businesses eligible for the 30-day feedback request.

        Returns businesses where:
        - Onboarding is complete
        - Feedback has not yet been requested
        - Business was created before the given cutoff timestamp

        Args:
            db:                 Active async database session.
            registered_before:  Only include businesses created before this timestamp.
            limit:              Maximum records to return.

        Returns:
            list[BusinessModel]: Businesses eligible for feedback request.
        """
        try:
            result = await db.execute(
                select(BusinessModel)
                .where(
                    BusinessModel.is_active.is_(True),
                    BusinessModel.is_deleted.is_(False),
                    BusinessModel.is_onboarding_complete.is_(True),
                    BusinessModel.feedback_requested.is_(False),
                    BusinessModel.created_at <= registered_before,
                )
                .order_by(BusinessModel.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch businesses for feedback request",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise

    async def count_active(self, db: AsyncSession) -> int:
        """
        Return the total count of active, non-deleted businesses.

        Used by admin health reports and system monitoring.

        Args:
            db: Active async database session.

        Returns:
            int: Total count of active businesses.
        """
        try:
            result = await db.execute(
                select(func.count(BusinessModel.id)).where(
                    BusinessModel.is_active.is_(True),
                    BusinessModel.is_deleted.is_(False),
                )
            )
            return result.scalar_one()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to count active businesses",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise

    # ── Update ─────────────────────────────────────────────────────────────────

    async def update_google_integration(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        *,
        google_place_id: Optional[str] = None,
        google_business_name: Optional[str] = None,
        google_access_token: Optional[str] = None,
        google_refresh_token: Optional[str] = None,
        is_google_connected: Optional[bool] = None,
    ) -> Optional[BusinessModel]:
        """
        Update Google integration fields for a business.

        Only non-None arguments are applied. Tokens must be pre-encrypted
        by the caller before passing to this method.

        Args:
            db:                    Active async database session.
            business_id:           UUID of the business to update.
            google_place_id:       New Google Place ID.
            google_business_name:  Business name from Google.
            google_access_token:   Encrypted OAuth access token.
            google_refresh_token:  Encrypted OAuth refresh token.
            is_google_connected:   Connection status flag.

        Returns:
            Updated BusinessModel if found, else None.
        """
        fields: dict = {}
        if google_place_id is not None:
            fields["google_place_id"] = google_place_id
        if google_business_name is not None:
            fields["google_business_name"] = google_business_name
        if google_access_token is not None:
            fields["google_access_token"] = google_access_token
        if google_refresh_token is not None:
            fields["google_refresh_token"] = google_refresh_token
        if is_google_connected is not None:
            fields["is_google_connected"] = is_google_connected

        if not fields:
            return await self.get_by_id(db, business_id)

        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(**fields)
            )
            await db.flush()
            return await self.get_by_id(db, business_id)

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update Google integration fields",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def update_google_rating(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        current_google_rating: float,
        total_google_reviews: int,
    ) -> None:
        """
        Update the cached Google rating and review count for a business.

        Called by review_monitor.py after each polling cycle.

        Args:
            db:                    Active async database session.
            business_id:           UUID of the business to update.
            current_google_rating: Latest average rating (1.0–5.0).
            total_google_reviews:  Total review count from Google.
        """
        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(
                    current_google_rating=current_google_rating,
                    total_google_reviews=total_google_reviews,
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update Google rating",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def update_sheets_connection(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        google_sheets_url: str,
        is_sheets_connected: bool,
    ) -> None:
        """
        Update the Google Sheets connection details for a business.

        Args:
            db:                 Active async database session.
            business_id:        UUID of the business to update.
            google_sheets_url:  URL of the connected Google Sheet.
            is_sheets_connected: Whether the connection is active.
        """
        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(
                    google_sheets_url=google_sheets_url,
                    is_sheets_connected=is_sheets_connected,
                )
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update Sheets connection",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_onboarding_complete(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> None:
        """
        Mark the onboarding process as complete for a business.

        Called by the onboarding service after all required setup steps
        have been confirmed.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business to update.
        """
        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(is_onboarding_complete=True)
            )
            await db.flush()

            logger.info(
                "Business onboarding marked complete",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark onboarding complete",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def mark_feedback_requested(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> None:
        """
        Set the feedback_requested flag to prevent duplicate feedback messages.

        Called by review_request_service.py after the feedback WhatsApp
        message is confirmed as sent.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business to update.
        """
        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(feedback_requested=True)
            )
            await db.flush()

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to mark feedback requested",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    async def set_active_status(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
        is_active: bool,
    ) -> None:
        """
        Activate or deactivate a business account.

        Deactivated businesses are excluded from all scheduler processing
        but remain in the database. This is not a soft-delete.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business to update.
            is_active:    True to activate, False to deactivate.
        """
        try:
            await db.execute(
                update(BusinessModel)
                .where(
                    BusinessModel.id == business_id,
                    BusinessModel.is_deleted.is_(False),
                )
                .values(is_active=is_active)
            )
            await db.flush()

            logger.info(
                "Business active status updated",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "is_active": is_active,
                },
            )

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to update business active status",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Soft Delete ────────────────────────────────────────────────────────────

    async def soft_delete(
        self,
        db: AsyncSession,
        business_id: uuid.UUID,
    ) -> bool:
        """
        Soft-delete a business account.

        Sets is_deleted=True and deleted_at to the current UTC timestamp.
        The record remains in the database for audit purposes. All child
        records (reviews, payments, jobs) remain linked via CASCADE.

        Args:
            db:           Active async database session.
            business_id:  UUID of the business to soft-delete.

        Returns:
            bool: True if the record was found and deleted, False if not found.
        """
        try:
            business = await self.get_by_id(db, business_id)
            if not business:
                return False

            business.soft_delete()
            business.is_active = False
            await db.flush()

            logger.info(
                "Business soft-deleted",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                },
            )
            return True

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to soft-delete business",
                extra={
                    "service": ServiceName.API,
                    "business_id": str(business_id),
                    "error": str(exc),
                },
            )
            raise

    # ── Existence Checks ───────────────────────────────────────────────────────

    async def email_exists(
        self,
        db: AsyncSession,
        owner_email: str,
    ) -> bool:
        """
        Check whether an email address is already registered.

        Includes soft-deleted records — a deleted business's email should
        not be reused without explicit admin intervention.

        Args:
            db:           Active async database session.
            owner_email:  Email address to check.

        Returns:
            bool: True if the email exists in any record (active or deleted).
        """
        try:
            result = await db.execute(
                select(func.count(BusinessModel.id)).where(
                    BusinessModel.owner_email == owner_email.lower().strip(),
                )
            )
            return result.scalar_one() > 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to check email existence",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise

    async def whatsapp_number_exists(
        self,
        db: AsyncSession,
        whatsapp_number: str,
    ) -> bool:
        """
        Check whether a WhatsApp number is already registered.

        Includes soft-deleted records for the same reason as email_exists.

        Args:
            db:               Active async database session.
            whatsapp_number:  WhatsApp number to check.

        Returns:
            bool: True if the number exists in any record.
        """
        try:
            result = await db.execute(
                select(func.count(BusinessModel.id)).where(
                    BusinessModel.owner_whatsapp_number == whatsapp_number.strip(),
                )
            )
            return result.scalar_one() > 0

        except SQLAlchemyError as exc:
            logger.error(
                "Failed to check WhatsApp number existence",
                extra={
                    "service": ServiceName.API,
                    "error": str(exc),
                },
            )
            raise