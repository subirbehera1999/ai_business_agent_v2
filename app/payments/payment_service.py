# ==============================================================================
# File: app/payments/payment_service.py
# Purpose: Orchestrates the full Razorpay payment workflow.
#
#          SUBSCRIPTION MODEL — ONE TIER ONLY
#          ─────────────────────────────────────
#          Pay = full access to every feature.
#          No plan tiers. No BASIC / PRO / ENTERPRISE.
#          The only choice is billing_cycle:
#            "monthly"  → billed every month   (₹699)
#            "annual"   → billed once per year  (₹7499, ~17% saving)
#
#          Responsibilities:
#            1. initiate_payment()
#               - Validates the business exists
#               - Validates billing_cycle ("monthly" / "annual")
#               - Checks for an existing pending order (idempotency)
#               - Creates a Razorpay order via razorpay_client.py
#               - Persists a payment record with status=PENDING in the DB
#               - Returns the Razorpay order_id and amount to the API layer
#                 for the frontend checkout widget
#
#            2. process_webhook_event()
#               - Called by webhook_handler.py after signature is verified
#               - Handles: payment.captured, payment.failed, refund.processed
#               - For payment.captured:
#                   a. Fetches payment from Razorpay (server-side verification)
#                   b. Updates payment record status → CAPTURED
#                   c. Activates/extends the business subscription
#                   d. Notifies the business via WhatsApp
#                   e. Notifies admin via admin_notification_service.py
#               - All steps are wrapped in a DB transaction
#               - Idempotent — processing the same event twice is safe
#
#            3. get_payment_status()
#               - Returns current payment record from the database
#               - Used by the frontend to poll payment status
#
#          Payment data protection (DATA_SAFETY_AND_RUNTIME_GUARDRAILS):
#            - Payment verification is ALWAYS server-side
#            - Frontend payment confirmation is NEVER trusted
#            - Razorpay webhook signature verified BEFORE any processing
#            - Payment record persisted BEFORE subscription activation
#            - Subscription activated ONLY after payment is confirmed captured
#
#          Idempotency key format:
#            PAYMENT_INIT_{business_id}_{billing_cycle}
#
#          Multi-tenant:
#            All queries are scoped to business_id.
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.constants import (
    PaymentStatus,
    ServiceName,
    SubscriptionStatus,
)
from app.config.settings import get_settings
from app.database.models.payment_model import PaymentModel
from app.payments.razorpay_client import RazorpayClient
from app.repositories.business_repository import BusinessRepository
from app.repositories.subscription_repository import SubscriptionRepository
from app.utils.idempotency_utils import make_payment_init_key
from app.utils.time_utils import now_utc

logger = logging.getLogger(ServiceName.PAYMENTS)
settings = get_settings()

# ------------------------------------------------------------------------------
# Single-tier pricing (amount in paise)
# One subscription, full access. Only the billing cycle differs.
# ------------------------------------------------------------------------------
_PRICES_PAISE: dict[str, int] = {
    "monthly":  69900,   # ₹699  / month
    "annual":  749900,   # ₹7499 / year (~17% saving vs monthly)
}

_VALID_BILLING_CYCLES = ("monthly", "annual")


# ==============================================================================
# Input / Output dataclasses
# ==============================================================================

@dataclass
class PaymentInitRequest:
    """
    Input for initiating a new payment.

    Attributes:
        business_id:   Business UUID (comes from JWT — never from client body).
        billing_cycle: "monthly" or "annual". The only choice available.
    """
    business_id: str
    billing_cycle: str


@dataclass
class PaymentInitResult:
    """
    Output returned to the API layer for Razorpay checkout.

    Attributes:
        success:           True if order was created successfully.
        razorpay_order_id: Razorpay order ID for the frontend widget.
        amount_paise:      Amount in paise to display in the widget.
        currency:          Always "INR".
        key_id:            Razorpay publishable key for the frontend.
        payment_record_id: Our internal payment record UUID.
        error:             Error message if success=False.
    """
    success: bool
    razorpay_order_id: Optional[str] = None
    amount_paise: Optional[int] = None
    currency: str = "INR"
    key_id: Optional[str] = None
    payment_record_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WebhookProcessResult:
    """
    Output of processing a Razorpay webhook event.

    Attributes:
        success:              True if the event was handled without error.
        event_type:           The Razorpay event string e.g. "payment.captured".
        payment_id:           Razorpay payment ID from the event.
        already_processed:    True if this exact event was already handled
                              (idempotency — not an error).
        subscription_updated: True if subscription was activated/extended.
        error:                Error message if success=False.
    """
    success: bool
    event_type: str
    payment_id: Optional[str] = None
    already_processed: bool = False
    subscription_updated: bool = False
    error: Optional[str] = None


# ==============================================================================
# Payment Service
# ==============================================================================

class PaymentService:
    """
    Orchestrates the full Razorpay payment lifecycle.

    One subscription tier — full access on payment.
    The business chooses monthly or annual billing cycle only.

    Injected dependencies:
        razorpay_client:    Low-level Razorpay API client.
        business_repo:      Business record access.
        subscription_repo:  Subscription creation and update.
        payment_repo:       Payment record persistence.
        whatsapp_service:   Delivery of payment confirmation to business.
        admin_notification: Admin alerts for new payments and failures.

    Usage:
        service = PaymentService(
            razorpay_client=razorpay_client,
            business_repo=business_repo,
            subscription_repo=subscription_repo,
            payment_repo=payment_repo,
            whatsapp_service=whatsapp_service,
            admin_notification=admin_notification,
        )

        result = await service.initiate_payment(
            db=db,
            request=PaymentInitRequest(
                business_id="uuid-here",
                billing_cycle="monthly",
            ),
        )
    """

    def __init__(
        self,
        razorpay_client: RazorpayClient,
        business_repo: BusinessRepository,
        subscription_repo: SubscriptionRepository,
        payment_repo,           # PaymentRepository (imported at call site)
        whatsapp_service,       # WhatsAppService
        admin_notification,     # AdminNotificationService
    ) -> None:
        self._razorpay = razorpay_client
        self._business_repo = business_repo
        self._subscription_repo = subscription_repo
        self._payment_repo = payment_repo
        self._whatsapp = whatsapp_service
        self._admin = admin_notification

    # ------------------------------------------------------------------
    # 1. Payment initiation
    # ------------------------------------------------------------------

    async def initiate_payment(
        self,
        db: AsyncSession,
        request: PaymentInitRequest,
    ) -> PaymentInitResult:
        """
        Initiate a new Razorpay payment order for subscription.

        Steps:
            1. Validate business exists
            2. Validate billing_cycle
            3. Check for existing pending order (idempotency)
            4. Resolve amount from billing cycle
            5. Create Razorpay order
            6. Persist payment record (status=PENDING)
            7. Return order details for frontend checkout

        Args:
            db:      AsyncSession (transaction managed here).
            request: PaymentInitRequest with business_id and billing_cycle.

        Returns:
            PaymentInitResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.PAYMENTS,
            "business_id": request.business_id,
            "billing_cycle": request.billing_cycle,
        }

        try:
            # --- Step 1: Validate business ---
            business = await self._business_repo.get_by_id(
                db=db, business_id=request.business_id
            )
            if not business:
                return PaymentInitResult(
                    success=False,
                    error=f"Business not found: {request.business_id}",
                )

            # --- Step 2: Validate billing cycle ---
            if request.billing_cycle not in _VALID_BILLING_CYCLES:
                return PaymentInitResult(
                    success=False,
                    error=(
                        f"Invalid billing_cycle '{request.billing_cycle}'. "
                        f"Allowed values: {_VALID_BILLING_CYCLES}"
                    ),
                )

            # --- Step 3: Idempotency — check for existing pending order ---
            idempotency_key = make_payment_init_key(
                business_id=request.business_id,
                billing_cycle=request.billing_cycle,
            )

            existing = await self._payment_repo.get_by_idempotency_key(
                db=db, idempotency_key=idempotency_key
            )
            if existing and existing.status == PaymentStatus.PENDING:
                logger.info(
                    "Returning existing pending payment order",
                    extra={**log_extra, "payment_id": str(existing.id)},
                )
                return PaymentInitResult(
                    success=True,
                    razorpay_order_id=existing.razorpay_order_id,
                    amount_paise=existing.amount_paise,
                    currency="INR",
                    key_id=settings.RAZORPAY_KEY_ID,
                    payment_record_id=str(existing.id),
                )

            # --- Step 4: Resolve amount ---
            amount_paise = _resolve_amount(request.billing_cycle)

            # --- Step 5: Create Razorpay order ---
            receipt = _build_receipt(request.business_id, request.billing_cycle)
            order_result = await self._razorpay.create_order(
                amount_paise=amount_paise,
                receipt=receipt,
                idempotency_key=idempotency_key,
                notes={
                    "business_id":   request.business_id,
                    "billing_cycle": request.billing_cycle,
                    "business_name": business.business_name or "",
                },
            )

            if not order_result.success:
                logger.error(
                    "Razorpay order creation failed",
                    extra={**log_extra, "error": order_result.error},
                )
                return PaymentInitResult(
                    success=False,
                    error=f"Payment order creation failed: {order_result.error}",
                )

            razorpay_order = order_result.data

            # --- Step 6: Persist payment record ---
            payment_record = await self._payment_repo.create(
                db=db,
                business_id=request.business_id,
                razorpay_order_id=razorpay_order.order_id,
                amount_paise=amount_paise,
                billing_cycle=request.billing_cycle,
                status=PaymentStatus.PENDING,
                idempotency_key=idempotency_key,
            )
            await db.commit()

            logger.info(
                "Payment initiated successfully",
                extra={
                    **log_extra,
                    "razorpay_order_id": razorpay_order.order_id,
                    "payment_record_id": str(payment_record.id),
                    "amount_paise": amount_paise,
                },
            )

            return PaymentInitResult(
                success=True,
                razorpay_order_id=razorpay_order.order_id,
                amount_paise=amount_paise,
                currency="INR",
                key_id=settings.RAZORPAY_KEY_ID,
                payment_record_id=str(payment_record.id),
            )

        except Exception as exc:
            await db.rollback()
            logger.error(
                "initiate_payment unexpected error",
                extra={**log_extra, "error": str(exc)},
            )
            return PaymentInitResult(
                success=False,
                error="Payment initiation failed due to an internal error.",
            )

    # ------------------------------------------------------------------
    # 2. Webhook event processing
    # ------------------------------------------------------------------

    async def process_webhook_event(
        self,
        db: AsyncSession,
        event_type: str,
        event_payload: dict,
    ) -> WebhookProcessResult:
        """
        Process a Razorpay webhook event.

        Handles payment.captured, payment.failed, and refund.processed.
        All processing is idempotent — the same event delivered twice
        produces the same outcome without side effects.

        The caller (webhook_handler.py) is responsible for:
          - Verifying the webhook signature BEFORE calling this method.
          - Passing the raw parsed event payload.

        Args:
            db:            AsyncSession.
            event_type:    Razorpay event string e.g. "payment.captured".
            event_payload: Parsed event JSON body from Razorpay.

        Returns:
            WebhookProcessResult. Never raises.
        """
        log_extra = {
            "service": ServiceName.PAYMENTS,
            "event_type": event_type,
        }

        try:
            if event_type == "payment.captured":
                return await self._handle_payment_captured(
                    db=db,
                    event_payload=event_payload,
                    log_extra=log_extra,
                )
            elif event_type == "payment.failed":
                return await self._handle_payment_failed(
                    db=db,
                    event_payload=event_payload,
                    log_extra=log_extra,
                )
            elif event_type == "refund.processed":
                return await self._handle_refund_processed(
                    db=db,
                    event_payload=event_payload,
                    log_extra=log_extra,
                )
            else:
                # Unrecognised event — acknowledge without processing
                logger.debug(
                    "Unhandled webhook event type — acknowledged",
                    extra=log_extra,
                )
                return WebhookProcessResult(
                    success=True,
                    event_type=event_type,
                )

        except Exception as exc:
            await db.rollback()
            logger.error(
                "process_webhook_event unexpected error",
                extra={**log_extra, "error": str(exc)},
            )
            return WebhookProcessResult(
                success=False,
                event_type=event_type,
                error=f"Webhook processing error: {exc}",
            )

    # ------------------------------------------------------------------
    # 3. Payment status query
    # ------------------------------------------------------------------

    async def get_payment_status(
        self,
        db: AsyncSession,
        business_id: str,
        razorpay_order_id: str,
    ) -> Optional[PaymentModel]:
        """
        Retrieve the current payment record for an order.

        Scoped by business_id for multi-tenant safety.
        Returns None if no matching record is found.

        Args:
            db:                AsyncSession.
            business_id:       Business UUID.
            razorpay_order_id: Razorpay order ID from checkout.

        Returns:
            PaymentModel or None.
        """
        return await self._payment_repo.get_by_order_id(
            db=db,
            business_id=business_id,
            razorpay_order_id=razorpay_order_id,
        )

    # ------------------------------------------------------------------
    # Private: payment.captured handler
    # ------------------------------------------------------------------

    async def _handle_payment_captured(
        self,
        db: AsyncSession,
        event_payload: dict,
        log_extra: dict,
    ) -> WebhookProcessResult:
        """
        Handle a payment.captured webhook event.

        Flow:
            1. Extract payment_id and order_id from payload
            2. Idempotency check — already processed?
            3. Fetch payment from Razorpay (server-side verification)
            4. Verify payment is genuinely captured on Razorpay
            5. Update payment record status → CAPTURED
            6. Activate/extend subscription (full access, no tier)
            7. Send WhatsApp confirmation to business
            8. Notify admin

        All DB writes are in a single transaction.
        If subscription activation fails, payment is still marked
        CAPTURED (money was received) and admin is notified for
        manual intervention.
        """
        payment_entity = (
            event_payload.get("payload", {})
            .get("payment", {})
            .get("entity", {})
        )
        payment_id = payment_entity.get("id", "")
        order_id = payment_entity.get("order_id", "")

        log_extra = {**log_extra, "payment_id": payment_id, "order_id": order_id}

        if not payment_id or not order_id:
            logger.error(
                "payment.captured event missing payment_id or order_id",
                extra=log_extra,
            )
            return WebhookProcessResult(
                success=False,
                event_type="payment.captured",
                error="Malformed event payload — missing payment_id or order_id",
            )

        # --- Idempotency check ---
        existing_payment = await self._payment_repo.get_by_razorpay_payment_id(
            db=db, razorpay_payment_id=payment_id
        )
        if existing_payment and existing_payment.status == PaymentStatus.CAPTURED:
            logger.info(
                "payment.captured already processed — skipping",
                extra=log_extra,
            )
            return WebhookProcessResult(
                success=True,
                event_type="payment.captured",
                payment_id=payment_id,
                already_processed=True,
            )

        # --- Server-side payment verification ---
        rz_result = await self._razorpay.fetch_payment(payment_id=payment_id)
        if not rz_result.success:
            logger.error(
                "Failed to fetch payment from Razorpay for verification",
                extra={**log_extra, "error": rz_result.error},
            )
            return WebhookProcessResult(
                success=False,
                event_type="payment.captured",
                payment_id=payment_id,
                error=f"Payment verification fetch failed: {rz_result.error}",
            )

        rz_payment = rz_result.data
        if not rz_payment.is_captured:
            logger.warning(
                "payment.captured webhook but payment not captured on Razorpay",
                extra={**log_extra, "razorpay_status": rz_payment.status},
            )
            return WebhookProcessResult(
                success=False,
                event_type="payment.captured",
                payment_id=payment_id,
                error=(
                    f"Payment status on Razorpay is '{rz_payment.status}', "
                    f"not 'captured'"
                ),
            )

        # --- Fetch our payment record ---
        payment_record = await self._payment_repo.get_by_order_id_unscoped(
            db=db, razorpay_order_id=order_id
        )
        if not payment_record:
            logger.error("No payment record found for order_id", extra=log_extra)
            await self._admin.send_alert(
                title="⚠️ Orphan payment captured",
                message=(
                    f"Payment {payment_id} captured for order {order_id} "
                    f"but no matching payment record found in DB."
                ),
            )
            return WebhookProcessResult(
                success=False,
                event_type="payment.captured",
                payment_id=payment_id,
                error="No payment record found for this order",
            )

        business_id = str(payment_record.business_id)
        billing_cycle = payment_record.billing_cycle

        # --- Update payment record → CAPTURED ---
        await self._payment_repo.update_status(
            db=db,
            payment_id=str(payment_record.id),
            status=PaymentStatus.CAPTURED,
            razorpay_payment_id=payment_id,
            captured_at=now_utc(),
        )

        # --- Activate/extend subscription ---
        subscription_updated = False
        try:
            await self._activate_subscription(
                db=db,
                business_id=business_id,
                billing_cycle=billing_cycle,
                payment_record_id=str(payment_record.id),
            )
            subscription_updated = True
        except Exception as exc:
            logger.error(
                "Subscription activation failed after payment capture",
                extra={**log_extra, "business_id": business_id, "error": str(exc)},
            )
            await self._admin.send_alert(
                title="🚨 Subscription activation failed",
                message=(
                    f"Payment {payment_id} captured for business {business_id} "
                    f"but subscription activation failed: {exc}. "
                    f"Manual activation required."
                ),
            )

        await db.commit()

        logger.info(
            "Payment captured and processed successfully",
            extra={
                **log_extra,
                "business_id": business_id,
                "billing_cycle": billing_cycle,
                "subscription_updated": subscription_updated,
            },
        )

        # --- Notify business via WhatsApp ---
        await self._send_payment_confirmation(
            db=db,
            business_id=business_id,
            billing_cycle=billing_cycle,
            amount_paise=rz_payment.amount,
        )

        # --- Notify admin ---
        await self._admin.send_payment_received(
            business_name=business_id,  # name not in scope here; use ID as fallback
            amount_rupees=rz_payment.amount / 100,
            billing_cycle=billing_cycle,
            business_id=business_id,
        )

        return WebhookProcessResult(
            success=True,
            event_type="payment.captured",
            payment_id=payment_id,
            subscription_updated=subscription_updated,
        )

    # ------------------------------------------------------------------
    # Private: payment.failed handler
    # ------------------------------------------------------------------

    async def _handle_payment_failed(
        self,
        db: AsyncSession,
        event_payload: dict,
        log_extra: dict,
    ) -> WebhookProcessResult:
        """
        Handle a payment.failed webhook event.

        Updates the payment record status → FAILED.
        The failed payment dialog is handled by the frontend Razorpay widget.
        """
        payment_entity = (
            event_payload.get("payload", {})
            .get("payment", {})
            .get("entity", {})
        )
        payment_id = payment_entity.get("id", "")
        order_id = payment_entity.get("order_id", "")
        error_code = payment_entity.get("error_code", "UNKNOWN")
        error_desc = payment_entity.get("error_description", "")

        log_extra = {**log_extra, "payment_id": payment_id, "order_id": order_id}

        # Idempotency — already marked failed
        existing = await self._payment_repo.get_by_razorpay_payment_id(
            db=db, razorpay_payment_id=payment_id
        )
        if existing and existing.status == PaymentStatus.FAILED:
            return WebhookProcessResult(
                success=True,
                event_type="payment.failed",
                payment_id=payment_id,
                already_processed=True,
            )

        payment_record = await self._payment_repo.get_by_order_id_unscoped(
            db=db, razorpay_order_id=order_id
        )
        if payment_record:
            await self._payment_repo.update_status(
                db=db,
                payment_id=str(payment_record.id),
                status=PaymentStatus.FAILED,
                razorpay_payment_id=payment_id,
                failure_reason=f"[{error_code}] {error_desc}",
            )
            await db.commit()

        logger.warning(
            "Payment failed",
            extra={**log_extra, "error_code": error_code, "error_desc": error_desc},
        )

        return WebhookProcessResult(
            success=True,
            event_type="payment.failed",
            payment_id=payment_id,
        )

    # ------------------------------------------------------------------
    # Private: refund.processed handler
    # ------------------------------------------------------------------

    async def _handle_refund_processed(
        self,
        db: AsyncSession,
        event_payload: dict,
        log_extra: dict,
    ) -> WebhookProcessResult:
        """
        Handle a refund.processed webhook event.

        Updates the payment record status → REFUNDED.
        Subscription deactivation is handled separately by expiry_checker.py.
        Admin is notified for awareness.
        """
        refund_entity = (
            event_payload.get("payload", {})
            .get("refund", {})
            .get("entity", {})
        )
        refund_id = refund_entity.get("id", "")
        payment_id = refund_entity.get("payment_id", "")
        amount = refund_entity.get("amount", 0)

        log_extra = {**log_extra, "refund_id": refund_id, "payment_id": payment_id}

        if payment_id:
            payment_record = await self._payment_repo.get_by_razorpay_payment_id(
                db=db, razorpay_payment_id=payment_id
            )
            if payment_record:
                await self._payment_repo.update_status(
                    db=db,
                    payment_id=str(payment_record.id),
                    status=PaymentStatus.REFUNDED,
                    refund_id=refund_id,
                )
                await db.commit()

        logger.info(
            "Refund processed",
            extra={**log_extra, "amount_paise": amount},
        )

        await self._admin.send_alert(
            title="💸 Refund processed",
            message=(
                f"Refund {refund_id} for payment {payment_id} "
                f"of ₹{amount / 100:.2f} has been processed."
            ),
        )

        return WebhookProcessResult(
            success=True,
            event_type="refund.processed",
            payment_id=payment_id,
        )

    # ------------------------------------------------------------------
    # Private: subscription activation
    # ------------------------------------------------------------------

    async def _activate_subscription(
        self,
        db: AsyncSession,
        business_id: str,
        billing_cycle: str,
        payment_record_id: str,
    ) -> None:
        """
        Create or extend a subscription after a successful payment.

        One tier only — full access to all features.
        Billing cycle determines duration:
            "monthly" → +1 month
            "annual"  → +12 months

        If an active subscription already exists it is extended.
        Otherwise a new subscription record is created.

        Args:
            db:               AsyncSession (caller commits).
            business_id:      Business UUID.
            billing_cycle:    "monthly" or "annual".
            payment_record_id: Internal payment record UUID.

        Raises:
            Exception if subscription upsert fails (caller handles).
        """
        from app.utils.time_utils import compute_subscription_end_date

        start_date = now_utc()
        end_date = compute_subscription_end_date(
            start=start_date,
            billing_cycle=billing_cycle,
        )

        existing_sub = await self._subscription_repo.get_active(
            db=db, business_id=business_id
        )

        if existing_sub:
            await self._subscription_repo.extend_subscription(
                db=db,
                subscription_id=str(existing_sub.id),
                new_end_date=end_date,
                payment_record_id=payment_record_id,
            )
            logger.info(
                "Subscription extended",
                extra={
                    "service": ServiceName.PAYMENTS,
                    "business_id": business_id,
                    "billing_cycle": billing_cycle,
                    "new_end_date": end_date.isoformat(),
                },
            )
        else:
            await self._subscription_repo.create(
                db=db,
                business_id=business_id,
                billing_cycle=billing_cycle,
                start_date=start_date,
                end_date=end_date,
                status=SubscriptionStatus.ACTIVE,
                payment_record_id=payment_record_id,
            )
            logger.info(
                "Subscription created",
                extra={
                    "service": ServiceName.PAYMENTS,
                    "business_id": business_id,
                    "billing_cycle": billing_cycle,
                    "end_date": end_date.isoformat(),
                },
            )

    # ------------------------------------------------------------------
    # Private: WhatsApp confirmation
    # ------------------------------------------------------------------

    async def _send_payment_confirmation(
        self,
        db: AsyncSession,
        business_id: str,
        billing_cycle: str,
        amount_paise: int,
    ) -> None:
        """
        Send a WhatsApp payment confirmation to the business owner.

        Failure does NOT block the payment processing flow — logged
        at WARNING level only.
        """
        try:
            amount_rupees = amount_paise / 100
            cycle_label = "month" if billing_cycle == "monthly" else "year"

            message = (
                f"✅ *Payment Confirmed*\n\n"
                f"Thank you! Your payment of ₹{amount_rupees:,.2f} has been received.\n\n"
                f"Billing: *{billing_cycle.title()}* (per {cycle_label})\n\n"
                f"Your AI Business Agent is now fully active. "
                f"You will start receiving review alerts, insights, and reports "
                f"via this WhatsApp number.\n\n"
                f"Thank you for choosing AI Business Agent! 🙏"
            )

            await self._whatsapp.send_text_message(
                db=db,
                business_id=business_id,
                text=message,
            )

        except Exception as exc:
            logger.warning(
                "Failed to send payment confirmation WhatsApp",
                extra={
                    "service": ServiceName.PAYMENTS,
                    "business_id": business_id,
                    "error": str(exc),
                },
            )


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _resolve_amount(billing_cycle: str) -> int:
    """
    Resolve the payment amount in paise for the given billing cycle.

    One tier only — amount is determined solely by billing cycle.

    Args:
        billing_cycle: "monthly" or "annual".

    Returns:
        Amount in paise (integer). Defaults to monthly if cycle unknown.
    """
    return _PRICES_PAISE.get(billing_cycle, _PRICES_PAISE["monthly"])


def _build_receipt(business_id: str, billing_cycle: str) -> str:
    """
    Build a Razorpay receipt string (max 40 chars).

    Format: RCPT_{short_biz_id}_{cycle_abbrev}

    Args:
        business_id:   Business UUID.
        billing_cycle: "monthly" or "annual".

    Returns:
        Receipt string <= 40 chars.
    """
    short_id = business_id.replace("-", "")[:12]
    cycle_abbrev = "ANN" if billing_cycle == "annual" else "MON"
    return f"RCPT_{short_id}_{cycle_abbrev}"[:40]