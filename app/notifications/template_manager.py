# ==============================================================================
# File: app/notifications/template_manager.py
# Purpose: Stores and builds WhatsApp template payloads for all
#          business-initiated messages that require Meta-approved templates.
#
#          WhatsApp Cloud API rules:
#            - Every message initiated by the business (not in response to
#              an inbound message within 24hrs) MUST use a pre-approved
#              template registered in Meta Business Suite.
#            - Templates have a name, language code, and optional variable
#              parameter slots {{1}}, {{2}}, etc.
#            - Template components: header, body, footer, buttons.
#            - This module owns the mapping: message_type → template_name
#              and builds the component parameter lists for each.
#
#          Template registry:
#            Templates are defined as dataclasses here and registered in
#            _TEMPLATE_REGISTRY at module load time. Each entry maps a
#            TemplateType to a TemplateDefinition.
#
#          Template types managed here:
#            PAYMENT_CONFIRMATION    → payment received + plan activated
#            RENEWAL_REMINDER        → N days before expiry
#            EXPIRY_NOTICE           → subscription expired
#            WEEKLY_REPORT_READY     → weekly report available
#            MONTHLY_REPORT_READY    → monthly report available
#            PLAN_UPGRADE            → plan upgraded confirmation
#            PLAN_DOWNGRADE          → plan downgrade scheduled
#            WELCOME                 → new business onboarding welcome
#            REVIEW_ALERT            → new review notification
#
#          Naming convention for Meta templates:
#            All template names use snake_case, lowercase, no spaces.
#            Names must match EXACTLY what is approved in Meta Business Suite.
#            Names are loaded from environment variables so they can be
#            changed without code changes when Meta re-approves templates.
#
#          Fallback:
#            If a template is not found for a message type, returns None.
#            Callers (whatsapp_service.py) fall back to free-form text
#            when possible, or skip delivery with a warning log.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.integrations.whatsapp_client import (
    WhatsAppTemplateComponent,
    WhatsAppTemplateParam,
)

logger = logging.getLogger(ServiceName.WHATSAPP)
settings = get_settings()


# ==============================================================================
# Template type constants
# ==============================================================================

class TemplateType:
    """Constants for all WhatsApp template message types."""
    PAYMENT_CONFIRMATION  = "payment_confirmation"
    RENEWAL_REMINDER      = "renewal_reminder"
    EXPIRY_NOTICE         = "expiry_notice"
    WEEKLY_REPORT_READY   = "weekly_report_ready"
    MONTHLY_REPORT_READY  = "monthly_report_ready"
    PLAN_UPGRADE          = "plan_upgrade"
    PLAN_DOWNGRADE        = "plan_downgrade"
    WELCOME               = "welcome"
    REVIEW_ALERT          = "review_alert"


# ==============================================================================
# Template dataclasses
# ==============================================================================

@dataclass
class BuiltTemplate:
    """
    A fully built template payload ready for WhatsApp Cloud API.

    Attributes:
        name:           Exact Meta-approved template name.
        language_code:  BCP-47 code e.g. "en", "en_US", "hi".
        components:     Parameter-substituted component list.
        template_type:  TemplateType constant for logging.
    """
    name: str
    language_code: str
    components: list[WhatsAppTemplateComponent]
    template_type: str

    def __str__(self) -> str:
        return f"BuiltTemplate(name={self.name} lang={self.language_code})"


@dataclass
class TemplateDefinition:
    """
    Internal definition of a registered WhatsApp template.

    Attributes:
        template_type:      TemplateType constant.
        name:               Meta-approved template name (from env var).
        language_code:      BCP-47 language code.
        body_param_count:   Number of {{N}} variables in body component.
        has_header:         Whether the template has a header component.
        header_param_count: Number of {{N}} variables in header.
        enabled:            False when env var not set — falls back to text.
    """
    template_type: str
    name: str
    language_code: str
    body_param_count: int = 0
    has_header: bool = False
    header_param_count: int = 0
    enabled: bool = True


# ==============================================================================
# Template Manager
# ==============================================================================

class TemplateManager:
    """
    Manages WhatsApp template definitions and builds payloads.

    Stateless — templates are registered at init time from settings.
    Instantiated once per application.

    Usage:
        manager = TemplateManager()

        template = manager.get_payment_confirmation_template(
            billing_cycle="monthly",
            amount_rupees=999.0,
        )

        if template:
            await whatsapp_client.send_template_message(
                to=number,
                template_name=template.name,
                language_code=template.language_code,
                components=template.components,
            )
    """

    def __init__(self) -> None:
        self._registry: dict[str, TemplateDefinition] = {}
        self._load_registry()

    # ------------------------------------------------------------------
    # Public builders — one per template type
    # ------------------------------------------------------------------

    def get_payment_confirmation_template(
        self,
        billing_cycle: str,
        amount_rupees: float,
    ) -> Optional[BuiltTemplate]:
        """
        Build a payment confirmation template payload.

        Template body variables (in order):
            {{1}} → Billing cycle  e.g. "Monthly"
            {{2}} → Amount         e.g. "₹999.00"
            {{3}} → Cycle          e.g. "monthly"

        Args:
            billing_cycle: "monthly" or "annual".
            amount_rupees: Amount charged in rupees.

        Returns:
            BuiltTemplate or None if template not registered/enabled.
        """
        defn = self._get(TemplateType.PAYMENT_CONFIRMATION)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(billing_cycle.title()),
                _param(f"₹{amount_rupees:,.2f}"),
                _param(billing_cycle),
            ],
        )

    def get_renewal_reminder_template(
        self,
        days_remaining: int,
    ) -> Optional[BuiltTemplate]:
        """
        Build a renewal reminder template payload.

        Template body variables:
            {{1}} → Days remaining  e.g. "3 days" or "today"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.RENEWAL_REMINDER)
        if not defn:
            return None

        day_label = (
            "today"
            if days_remaining == 0
            else f"{days_remaining} day{'s' if days_remaining != 1 else ''}"
        )
        return self._build(
            defn=defn,
            body_params=[
                _param(day_label),
            ],
        )

    def get_expiry_notice_template(
        self,
    ) -> Optional[BuiltTemplate]:
        """
        Build a subscription expiry notice template payload.

        One tier only — no plan name needed.

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.EXPIRY_NOTICE)
        if not defn:
            return None

        return self._build(defn=defn, body_params=[])

    def get_weekly_report_template(
        self,
        business_name: str,
        week_label: str,
    ) -> Optional[BuiltTemplate]:
        """
        Build a weekly report ready notification template.

        Template body variables:
            {{1}} → Business name  e.g. "Raj Restaurant"
            {{2}} → Week label     e.g. "10–16 Mar 2025"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.WEEKLY_REPORT_READY)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(business_name[:60]),
                _param(week_label),
            ],
        )

    def get_monthly_report_template(
        self,
        business_name: str,
        month_label: str,
    ) -> Optional[BuiltTemplate]:
        """
        Build a monthly report ready notification template.

        Template body variables:
            {{1}} → Business name  e.g. "Raj Restaurant"
            {{2}} → Month label    e.g. "March 2025"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.MONTHLY_REPORT_READY)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(business_name[:60]),
                _param(month_label),
            ],
        )

    def get_plan_upgrade_template(
        self,
        from_plan: str,
        to_plan: str,
    ) -> Optional[BuiltTemplate]:
        """
        Build a plan upgrade confirmation template.

        Template body variables:
            {{1}} → From plan  e.g. "Basic"
            {{2}} → To plan    e.g. "Pro"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.PLAN_UPGRADE)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(from_plan.title()),
                _param(to_plan.title()),
            ],
        )

    def get_plan_downgrade_template(
        self,
        from_plan: str,
        to_plan: str,
        effective_date: str,
    ) -> Optional[BuiltTemplate]:
        """
        Build a plan downgrade scheduled confirmation template.

        Template body variables:
            {{1}} → From plan       e.g. "Pro"
            {{2}} → To plan         e.g. "Basic"
            {{3}} → Effective date  e.g. "31 Mar 2025"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.PLAN_DOWNGRADE)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(from_plan.title()),
                _param(to_plan.title()),
                _param(effective_date),
            ],
        )

    def get_welcome_template(
        self,
        business_name: str,
    ) -> Optional[BuiltTemplate]:
        """
        Build an onboarding welcome template for new businesses.

        One tier only — no plan name. All businesses get full access.

        Template body variables:
            {{1}} → Business name  e.g. "Raj Restaurant"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.WELCOME)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(business_name[:60]),
            ],
        )

    def get_review_alert_template(
        self,
        reviewer_name: str,
        rating: int,
    ) -> Optional[BuiltTemplate]:
        """
        Build a new review alert template.

        Template body variables:
            {{1}} → Reviewer name  e.g. "Priya S."
            {{2}} → Star rating    e.g. "2"

        Returns:
            BuiltTemplate or None.
        """
        defn = self._get(TemplateType.REVIEW_ALERT)
        if not defn:
            return None

        return self._build(
            defn=defn,
            body_params=[
                _param(reviewer_name[:60]),
                _param(str(rating)),
            ],
        )

    # ------------------------------------------------------------------
    # Registry introspection
    # ------------------------------------------------------------------

    def is_registered(self, template_type: str) -> bool:
        """Return True if the template type is registered and enabled."""
        defn = self._registry.get(template_type)
        return defn is not None and defn.enabled

    def list_registered(self) -> list[str]:
        """Return a list of all registered and enabled template types."""
        return [t for t, d in self._registry.items() if d.enabled]

    def get_template_name(self, template_type: str) -> Optional[str]:
        """Return the Meta template name for a given type, or None."""
        defn = self._registry.get(template_type)
        return defn.name if defn and defn.enabled else None

    # ------------------------------------------------------------------
    # Internal registry management
    # ------------------------------------------------------------------

    def _load_registry(self) -> None:
        """
        Load template definitions from settings.

        Template names are configured via environment variables so they
        can be updated when Meta re-approves or renames templates without
        requiring a code deployment.

        A missing env var disables that template type — whatsapp_service.py
        falls back to free-form text delivery in that case.
        """
        lang = settings.WHATSAPP_TEMPLATE_LANGUAGE or "en"

        definitions: list[TemplateDefinition] = [
            TemplateDefinition(
                template_type=TemplateType.PAYMENT_CONFIRMATION,
                name=settings.WHATSAPP_TEMPLATE_PAYMENT_CONFIRMATION or "",
                language_code=lang,
                body_param_count=3,
                enabled=bool(settings.WHATSAPP_TEMPLATE_PAYMENT_CONFIRMATION),
            ),
            TemplateDefinition(
                template_type=TemplateType.RENEWAL_REMINDER,
                name=settings.WHATSAPP_TEMPLATE_RENEWAL_REMINDER or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_RENEWAL_REMINDER),
            ),
            TemplateDefinition(
                template_type=TemplateType.EXPIRY_NOTICE,
                name=settings.WHATSAPP_TEMPLATE_EXPIRY_NOTICE or "",
                language_code=lang,
                body_param_count=1,
                enabled=bool(settings.WHATSAPP_TEMPLATE_EXPIRY_NOTICE),
            ),
            TemplateDefinition(
                template_type=TemplateType.WEEKLY_REPORT_READY,
                name=settings.WHATSAPP_TEMPLATE_WEEKLY_REPORT or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_WEEKLY_REPORT),
            ),
            TemplateDefinition(
                template_type=TemplateType.MONTHLY_REPORT_READY,
                name=settings.WHATSAPP_TEMPLATE_MONTHLY_REPORT or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_MONTHLY_REPORT),
            ),
            TemplateDefinition(
                template_type=TemplateType.PLAN_UPGRADE,
                name=settings.WHATSAPP_TEMPLATE_PLAN_UPGRADE or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_PLAN_UPGRADE),
            ),
            TemplateDefinition(
                template_type=TemplateType.PLAN_DOWNGRADE,
                name=settings.WHATSAPP_TEMPLATE_PLAN_DOWNGRADE or "",
                language_code=lang,
                body_param_count=3,
                enabled=bool(settings.WHATSAPP_TEMPLATE_PLAN_DOWNGRADE),
            ),
            TemplateDefinition(
                template_type=TemplateType.WELCOME,
                name=settings.WHATSAPP_TEMPLATE_WELCOME or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_WELCOME),
            ),
            TemplateDefinition(
                template_type=TemplateType.REVIEW_ALERT,
                name=settings.WHATSAPP_TEMPLATE_REVIEW_ALERT or "",
                language_code=lang,
                body_param_count=2,
                enabled=bool(settings.WHATSAPP_TEMPLATE_REVIEW_ALERT),
            ),
        ]

        for defn in definitions:
            self._registry[defn.template_type] = defn

        enabled_count = sum(1 for d in self._registry.values() if d.enabled)
        logger.info(
            "Template registry loaded",
            extra={
                "service": ServiceName.WHATSAPP,
                "total": len(self._registry),
                "enabled": enabled_count,
                "disabled": len(self._registry) - enabled_count,
            },
        )

    def _get(self, template_type: str) -> Optional[TemplateDefinition]:
        """
        Return a registered, enabled TemplateDefinition or None.

        Logs at debug when a type is not registered or disabled so
        callers can transparently fall back to text delivery.
        """
        defn = self._registry.get(template_type)
        if not defn:
            logger.debug(
                "Template type not registered",
                extra={
                    "service": ServiceName.WHATSAPP,
                    "template_type": template_type,
                },
            )
            return None
        if not defn.enabled:
            logger.debug(
                "Template type disabled — env var not set",
                extra={
                    "service": ServiceName.WHATSAPP,
                    "template_type": template_type,
                },
            )
            return None
        return defn

    def _build(
        self,
        defn: TemplateDefinition,
        body_params: list[WhatsAppTemplateParam],
        header_params: Optional[list[WhatsAppTemplateParam]] = None,
    ) -> BuiltTemplate:
        """
        Assemble a BuiltTemplate from a definition and parameter lists.

        Builds the components array expected by WhatsApp Cloud API.
        Only includes components that have parameters — empty components
        are omitted (Meta rejects payloads with empty parameter arrays).

        Args:
            defn:          Template definition from registry.
            body_params:   Parameter values for the body component.
            header_params: Parameter values for header component (if any).

        Returns:
            BuiltTemplate ready for whatsapp_client.send_template_message().
        """
        components: list[WhatsAppTemplateComponent] = []

        # Header component (optional)
        if defn.has_header and header_params:
            components.append(
                WhatsAppTemplateComponent(
                    type="header",
                    parameters=header_params,
                )
            )

        # Body component — trim to declared param count to prevent Meta
        # validation errors if the caller passes more params than registered
        if body_params:
            trimmed = body_params[:defn.body_param_count]
            if trimmed:
                components.append(
                    WhatsAppTemplateComponent(
                        type="body",
                        parameters=trimmed,
                    )
                )

        return BuiltTemplate(
            name=defn.name,
            language_code=defn.language_code,
            components=components,
            template_type=defn.template_type,
        )


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _param(value: str) -> WhatsAppTemplateParam:
    """
    Create a text parameter for a WhatsApp template component.

    Meta requires all template parameters to have type="text".
    Values are truncated to 1024 chars (Meta's per-parameter limit).

    Args:
        value: Substitution value for a template {{N}} slot.

    Returns:
        WhatsAppTemplateParam with type="text".
    """
    return WhatsAppTemplateParam(type="text", text=str(value)[:1024])