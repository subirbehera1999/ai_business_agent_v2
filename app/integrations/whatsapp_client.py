# ==============================================================================
# File: app/integrations/whatsapp_client.py
# Purpose: Client for WhatsApp Cloud API (Meta Graph API v18.0+).
#          Handles all outbound WhatsApp message delivery for the system.
#
#          Message types supported:
#            - send_text_message()      → plain text, up to 4096 chars
#            - send_template_message()  → pre-approved WhatsApp templates
#                                         (required for business-initiated
#                                          conversations outside 24hr window)
#            - send_multi_part()        → splits long content across multiple
#                                         sequential text messages
#
#          Design:
#            - All methods use with_whatsapp_retry (3 attempts,
#              exponential backoff: 2s → 5s → 10s)
#            - Timeouts enforced on every HTTP call
#            - Rate limiting: WhatsApp Cloud API enforces per-phone-number
#              limits. This client serialises all sends (sequential, not
#              parallel) and never batches concurrent calls to the same number.
#            - Returns WhatsAppSendResult — structured success/failure wrapper.
#              Never raises to callers.
#            - Message IDs from successful sends are logged for traceability.
#            - Phone numbers are normalised to E.164 format before sending.
#
#          Authentication:
#            META_WHATSAPP_ACCESS_TOKEN  — permanent or temp token from
#                                          Meta Business Suite
#            META_PHONE_NUMBER_ID        — WhatsApp Business phone number ID
#                                          (not the phone number itself)
#
#          Compliance:
#            - Business-initiated messages outside the 24-hour customer
#              service window MUST use approved templates.
#            - send_text_message() is valid only within the 24hr reply window.
#            - For proactive alerts and reports (scheduled, not triggered by
#              user message), send_template_message() should be used.
#            - This module does not enforce the window rule — that is the
#              responsibility of whatsapp_service.py which calls this client.
# ==============================================================================

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_whatsapp_retry

logger = logging.getLogger(ServiceName.WHATSAPP)
settings = get_settings()

# ---------------------------------------------------------------------------
# WhatsApp Cloud API constants
# ---------------------------------------------------------------------------
_GRAPH_API_BASE = "https://graph.facebook.com/v18.0"
_MESSAGES_ENDPOINT = "{base}/{phone_number_id}/messages"

# Maximum characters per WhatsApp text message
MAX_TEXT_CHARS: int = 4096

# Delay in seconds between sequential multi-part messages
# Prevents out-of-order delivery due to Meta's routing
MULTI_PART_DELAY_SECONDS: float = 0.5

# Maximum parts in a multi-part message to prevent runaway sends
MAX_PARTS: int = 10


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class WhatsAppSendResult:
    """
    Result of a single WhatsApp message send attempt.

    Attributes:
        success:      True if Meta accepted the message.
        message_id:   Meta's wamid (WhatsApp message ID) on success.
        to:           Destination phone number (last 4 digits for logs).
        error:        Error description if success=False.
        error_code:   Meta error code (integer) if available.
        status_code:  HTTP status code from the Graph API response.
        parts_sent:   Number of message parts sent (1 for single messages,
                      N for multi-part sends).
    """
    success: bool
    message_id: Optional[str] = None
    to: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[int] = None
    status_code: Optional[int] = None
    parts_sent: int = 1

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429 or self.error_code == 130429

    @property
    def is_invalid_number(self) -> bool:
        # Meta error codes for unregistered / invalid WhatsApp numbers
        return self.error_code in (131026, 131047)

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403) or self.error_code == 190

    def __str__(self) -> str:
        if self.success:
            return (
                f"WhatsAppSendResult(success=True "
                f"to=...{(self.to or '')[-4:]} "
                f"mid={self.message_id} "
                f"parts={self.parts_sent})"
            )
        return (
            f"WhatsAppSendResult(success=False "
            f"error={self.error} "
            f"code={self.error_code})"
        )


@dataclass
class WhatsAppTemplateParam:
    """
    A single parameter value for a WhatsApp template component.

    Attributes:
        type:  Always "text" for text parameters.
        text:  The parameter value string.
    """
    type: str = "text"
    text: str = ""


@dataclass
class WhatsAppTemplateComponent:
    """
    A component of a WhatsApp template message.

    Attributes:
        type:        "header", "body", or "button".
        parameters:  List of WhatsAppTemplateParam for variable substitution.
        sub_type:    For button components: "quick_reply" or "url".
        index:       For button components: 0-based button index.
    """
    type: str
    parameters: list[WhatsAppTemplateParam] = field(default_factory=list)
    sub_type: Optional[str] = None
    index: Optional[int] = None


# ==============================================================================
# WhatsApp Client
# ==============================================================================

class WhatsAppClient:
    """
    Async client for WhatsApp Cloud API (Meta Graph API).

    Instantiated once per application and shared across notification
    services and schedulers. Uses a single shared httpx.AsyncClient
    for connection pooling.

    Usage:
        client = WhatsAppClient()
        await client.initialise()

        result = await client.send_text_message(
            to="+919876543210",
            text="Your weekly report is ready.",
        )

        if result.success:
            logger.info(f"Sent: {result.message_id}")
        else:
            logger.error(f"Failed: {result.error}")

        await client.close()
    """

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None
        self._access_token: str = settings.META_WHATSAPP_ACCESS_TOKEN
        self._phone_number_id: str = settings.META_PHONE_NUMBER_ID
        self._messages_url: str = _MESSAGES_ENDPOINT.format(
            base=_GRAPH_API_BASE,
            phone_number_id=settings.META_PHONE_NUMBER_ID,
        )

    async def initialise(self) -> None:
        """
        Initialise the shared HTTP client.
        Must be called before any send methods are used.
        Called from app lifespan or dependency injection.
        """
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.EXTERNAL_API_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers={
                "User-Agent": "AIBusinessAgent/1.0",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "WhatsAppClient initialised",
            extra={
                "service": ServiceName.WHATSAPP,
                "phone_number_id": self._phone_number_id,
            },
        )

    async def close(self) -> None:
        """Close the shared HTTP client. Called from app lifespan shutdown."""
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info(
            "WhatsAppClient closed",
            extra={"service": ServiceName.WHATSAPP},
        )

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    @with_whatsapp_retry
    async def send_text_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
    ) -> WhatsAppSendResult:
        """
        Send a plain text message to a WhatsApp number.

        Valid only within the 24-hour customer service window.
        For proactive messages (alerts, reports), use send_template_message().

        Text is truncated to MAX_TEXT_CHARS if longer — callers should use
        send_multi_part() for content that may exceed the limit.

        Args:
            to:          Recipient WhatsApp number in E.164 format.
                         e.g. "+919876543210"
            text:        Message body (max 4096 chars).
            preview_url: Whether to generate a link preview for URLs.

        Returns:
            WhatsAppSendResult. Never raises.
        """
        self._ensure_initialised()
        normalised_to = _normalise_phone_number(to)
        safe_text = text[:MAX_TEXT_CHARS]

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": normalised_to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": safe_text,
            },
        }

        log_extra = {
            "service": ServiceName.WHATSAPP,
            "to": f"...{normalised_to[-4:]}",
            "chars": len(safe_text),
        }

        return await self._post_message(payload=payload, log_extra=log_extra)

    @with_whatsapp_retry
    async def send_template_message(
        self,
        to: str,
        template_name: str,
        language_code: str = "en",
        components: Optional[list[WhatsAppTemplateComponent]] = None,
    ) -> WhatsAppSendResult:
        """
        Send a pre-approved WhatsApp template message.

        Required for business-initiated conversations outside the 24-hour
        customer service window — i.e. all proactive alerts and reports.

        Template must be approved in Meta Business Suite before use.
        Template names and language codes must match exactly as registered.

        Args:
            to:            Recipient phone number in E.164 format.
            template_name: Exact name of the approved template
                           e.g. "weekly_report_v1"
            language_code: BCP-47 language code. Default: "en"
                           Use "en_US" for English (US) templates.
            components:    Optional list of template components with variable
                           values for {{1}}, {{2}} etc. substitutions.

        Returns:
            WhatsAppSendResult. Never raises.
        """
        self._ensure_initialised()
        normalised_to = _normalise_phone_number(to)

        template_payload: dict[str, Any] = {
            "name": template_name,
            "language": {"code": language_code},
        }

        if components:
            template_payload["components"] = [
                _serialise_component(c) for c in components
            ]

        payload = {
            "messaging_product": "whatsapp",
            "to": normalised_to,
            "type": "template",
            "template": template_payload,
        }

        log_extra = {
            "service": ServiceName.WHATSAPP,
            "to": f"...{normalised_to[-4:]}",
            "template": template_name,
            "lang": language_code,
        }

        return await self._post_message(payload=payload, log_extra=log_extra)

    async def send_multi_part(
        self,
        to: str,
        parts: list[str],
    ) -> WhatsAppSendResult:
        """
        Send a list of message parts sequentially to a WhatsApp number.

        Used when content exceeds MAX_TEXT_CHARS or when reports/alerts
        are intentionally split into labelled sections for readability.

        Parts are sent in order with MULTI_PART_DELAY_SECONDS between each
        to preserve delivery ordering. WhatsApp does not guarantee ordering
        for messages sent in rapid succession.

        Capped at MAX_PARTS to prevent runaway sends.

        Args:
            to:     Recipient phone number in E.164 format.
            parts:  Ordered list of message strings. Each part is sent as
                    a separate WhatsApp message.

        Returns:
            WhatsAppSendResult reflecting the final part's outcome.
            parts_sent is set to the number of parts successfully sent.
            If any part fails, the result is marked failed at that point
            and remaining parts are not sent.
        """
        import asyncio

        self._ensure_initialised()
        normalised_to = _normalise_phone_number(to)

        if not parts:
            return WhatsAppSendResult(
                success=False,
                to=normalised_to,
                error="send_multi_part called with empty parts list",
            )

        capped_parts = parts[:MAX_PARTS]
        if len(parts) > MAX_PARTS:
            logger.warning(
                "send_multi_part: parts list capped at MAX_PARTS",
                extra={
                    "service": ServiceName.WHATSAPP,
                    "to": f"...{normalised_to[-4:]}",
                    "original_parts": len(parts),
                    "capped_at": MAX_PARTS,
                },
            )

        last_result: Optional[WhatsAppSendResult] = None
        parts_sent = 0

        for i, part_text in enumerate(capped_parts):
            if not part_text or not part_text.strip():
                continue   # skip blank parts silently

            result = await self.send_text_message(
                to=normalised_to,
                text=part_text,
            )

            if result.success:
                parts_sent += 1
                last_result = result
            else:
                logger.error(
                    "Multi-part send aborted after part failure",
                    extra={
                        "service": ServiceName.WHATSAPP,
                        "to": f"...{normalised_to[-4:]}",
                        "part_index": i,
                        "parts_sent_before_failure": parts_sent,
                        "error": result.error,
                    },
                )
                # Surface the failure — do not continue sending further parts
                return WhatsAppSendResult(
                    success=False,
                    to=normalised_to,
                    error=result.error,
                    error_code=result.error_code,
                    status_code=result.status_code,
                    parts_sent=parts_sent,
                )

            # Delay between parts to preserve ordering
            if i < len(capped_parts) - 1:
                await asyncio.sleep(MULTI_PART_DELAY_SECONDS)

        if last_result is None:
            return WhatsAppSendResult(
                success=False,
                to=normalised_to,
                error="No non-empty parts were sent",
            )

        return WhatsAppSendResult(
            success=True,
            message_id=last_result.message_id,
            to=normalised_to,
            parts_sent=parts_sent,
        )

    # ------------------------------------------------------------------
    # Core HTTP dispatch
    # ------------------------------------------------------------------

    async def _post_message(
        self,
        payload: dict,
        log_extra: dict,
    ) -> WhatsAppSendResult:
        """
        Execute a POST to the WhatsApp Cloud API messages endpoint.

        Handles all HTTP-level error parsing and maps Meta error codes
        to structured WhatsAppSendResult fields.

        Args:
            payload:   Fully constructed message payload dict.
            log_extra: Structured log context.

        Returns:
            WhatsAppSendResult. Raises httpx exceptions for retry wrapper.
        """
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._http.post(
                self._messages_url,
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                body = response.json()
                messages = body.get("messages", [])
                message_id = messages[0].get("id") if messages else None

                logger.info(
                    "WhatsApp message sent successfully",
                    extra={**log_extra, "message_id": message_id},
                )
                return WhatsAppSendResult(
                    success=True,
                    message_id=message_id,
                    to=log_extra.get("to"),
                    status_code=200,
                )

            # Parse Meta error structure
            return _parse_meta_error(response, log_extra)

        except httpx.TimeoutException as exc:
            logger.error(
                "WhatsApp API timeout",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

        except httpx.HTTPError as exc:
            logger.error(
                "WhatsApp API HTTP error",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Raise if the HTTP client has not been initialised."""
        if self._http is None:
            raise RuntimeError(
                "WhatsAppClient has not been initialised. "
                "Call await client.initialise() before use."
            )


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _normalise_phone_number(phone: str) -> str:
    """
    Normalise a phone number to E.164 format for WhatsApp API.

    Rules:
      - Strip all whitespace, dashes, parentheses, and dots.
      - If the number starts with "0" (local Indian format), replace
        with "+91" country code.
      - If the number has no leading "+", prepend "+91" (default India).
      - Preserve numbers that already start with "+".

    Examples:
        "+919876543210"  → "+919876543210"   (already E.164)
        "09876543210"    → "+919876543210"   (Indian local → E.164)
        "9876543210"     → "+919876543210"   (no country code → +91)
        "+447911123456"  → "+447911123456"   (UK number preserved)

    Args:
        phone: Phone number string in any common format.

    Returns:
        E.164 formatted string e.g. "+919876543210".
    """
    # Strip all non-digit characters except leading "+"
    stripped = re.sub(r"[\s\-().]+", "", phone)

    if stripped.startswith("+"):
        return stripped

    if stripped.startswith("0"):
        # Local format with leading zero — assume India (+91)
        return f"+91{stripped[1:]}"

    if len(stripped) == 10:
        # 10-digit number without country code — assume India (+91)
        return f"+91{stripped}"

    # For any other format, prepend "+" and hope it resolves
    return f"+{stripped}"


def _serialise_component(component: WhatsAppTemplateComponent) -> dict:
    """
    Serialise a WhatsAppTemplateComponent to the Meta API JSON format.

    Args:
        component: WhatsAppTemplateComponent to serialise.

    Returns:
        dict ready for inclusion in the "components" array of a
        template message payload.
    """
    result: dict[str, Any] = {
        "type": component.type,
        "parameters": [
            {"type": p.type, "text": p.text}
            for p in component.parameters
        ],
    }
    if component.sub_type:
        result["sub_type"] = component.sub_type
    if component.index is not None:
        result["index"] = component.index
    return result


def _parse_meta_error(
    response: httpx.Response,
    log_extra: dict,
) -> WhatsAppSendResult:
    """
    Parse a non-200 Meta Graph API response into a WhatsAppSendResult.

    Meta returns errors in a nested structure:
        {
            "error": {
                "message": "...",
                "type": "OAuthException",
                "code": 190,
                "fbtrace_id": "..."
            }
        }

    Maps known Meta error codes to descriptive messages.

    Args:
        response:  The failed httpx.Response.
        log_extra: Structured log context for the error log entry.

    Returns:
        WhatsAppSendResult with success=False and structured error info.
    """
    status_code = response.status_code

    try:
        body = response.json()
        meta_error = body.get("error", {})
        error_message = meta_error.get("message", "Unknown error")
        error_code = meta_error.get("code")
        error_type = meta_error.get("type", "")
        fbtrace_id = meta_error.get("fbtrace_id", "")
    except Exception:
        error_message = response.text[:200]
        error_code = None
        error_type = ""
        fbtrace_id = ""

    # Enrich with human-readable description for known codes
    friendly_msg = _meta_error_code_description(error_code, error_message)

    logger.error(
        "WhatsApp API send failed",
        extra={
            **log_extra,
            "status_code": status_code,
            "error_code": error_code,
            "error_type": error_type,
            "error_message": error_message,
            "fbtrace_id": fbtrace_id,
        },
    )

    return WhatsAppSendResult(
        success=False,
        to=log_extra.get("to"),
        error=friendly_msg,
        error_code=error_code,
        status_code=status_code,
    )


# Meta Graph API error code descriptions for common failures
_META_ERROR_DESCRIPTIONS: dict[int, str] = {
    190:    "Access token expired or invalid — refresh META_WHATSAPP_ACCESS_TOKEN",
    100:    "Invalid parameter in request payload",
    130429: "Rate limit hit — too many messages sent in short period",
    131026: "Recipient WhatsApp number is not registered",
    131047: "Message failed — recipient may have blocked this number",
    131051: "Unsupported message type for this conversation",
    131052: "Media upload failed",
    132000: "Template not found or not approved",
    132001: "Template message body parameters do not match approved template",
    133004: "Server temporarily unavailable — retry",
    133005: "WhatsApp service maintenance — retry later",
    368:    "Account temporarily restricted for policy violation",
}


def _meta_error_code_description(code: Optional[int], fallback: str) -> str:
    """
    Return a human-readable description for a Meta error code.

    Args:
        code:     Meta integer error code (may be None).
        fallback: Raw error message from Meta to use if code is unknown.

    Returns:
        Human-readable error string.
    """
    if code is None:
        return fallback
    description = _META_ERROR_DESCRIPTIONS.get(code)
    if description:
        return f"[{code}] {description}"
    return f"[{code}] {fallback}"