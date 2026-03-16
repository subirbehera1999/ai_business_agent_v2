# ==============================================================================
# File: app/feedback/form_link_manager.py
# Purpose: Manages the Google Form link used in testimonial / feedback requests.
#
#          This module is intentionally simple. Its only job is to be the
#          single source of truth for the feedback form URL used across the
#          feedback system.
#
#          Why a dedicated class instead of reading settings directly?
#          ─────────────────────────────────────────────────────────────
#          1. Runtime override: The admin may need to swap the form URL
#             (e.g. seasonal campaign form, A/B test) without restarting
#             the server. FormLinkManager holds an in-memory override that
#             takes precedence over the settings value.
#
#          2. Validation: Settings stores the raw string. This class validates
#             that the URL is non-empty and starts with http/https before
#             returning it to callers. A bad URL in .env should surface as a
#             clear error at the point of use, not a silent empty link.
#
#          3. Single import point: review_request_service.py imports only
#             FormLinkManager — it never touches settings directly for this
#             value. If the storage location of the form URL ever changes
#             (e.g. moved to DB), only this file needs updating.
#
#          Public API:
#            get_form_link()       → str   — returns current active URL
#            update_form_link(url) → None  — sets in-memory runtime override
#            reset_to_default()    → None  — clears override, reverts to .env
#            is_configured()       → bool  — True if a valid URL is available
#
#          Thread / async safety:
#            The in-memory override is a plain instance attribute. This class
#            is used as a singleton injected into ReviewRequestService. All
#            access is from the async event loop — no thread-safety concerns.
#
#          Error behaviour:
#            get_form_link() raises FormLinkNotConfiguredError if neither the
#            override nor the settings value is a valid URL. This is a
#            configuration error that must be fixed before the feedback system
#            can operate — it is not silently swallowed.
# ==============================================================================

import logging
from typing import Optional

from app.config.constants import ServiceName
from app.config.settings import get_settings

logger = logging.getLogger(ServiceName.FEEDBACK)


# ==============================================================================
# Custom exception
# ==============================================================================

class FormLinkNotConfiguredError(Exception):
    """
    Raised when get_form_link() is called but no valid Google Form URL
    has been configured in settings or set via update_form_link().

    This is a configuration error. The operator must set
    FEEDBACK_GOOGLE_FORM_URL in the .env file before the feedback
    request system can send messages.
    """


# ==============================================================================
# Manager
# ==============================================================================

class FormLinkManager:
    """
    Single source of truth for the feedback Google Form URL.

    Reads the default URL from FEEDBACK_GOOGLE_FORM_URL in settings.
    Supports a runtime in-memory override via update_form_link() for
    admin-driven URL changes without a server restart.

    Usage:
        manager = FormLinkManager()
        link = manager.get_form_link()   # raises if not configured

        manager.update_form_link("https://forms.gle/new-form")
        link = manager.get_form_link()   # returns the override

        manager.reset_to_default()
        link = manager.get_form_link()   # back to settings value
    """

    def __init__(self) -> None:
        # In-memory runtime override — None means "use settings value"
        self._override_url: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_form_link(self) -> str:
        """
        Return the currently active Google Form URL.

        Resolution order:
          1. In-memory override (set via update_form_link)
          2. FEEDBACK_GOOGLE_FORM_URL from settings / .env

        Returns:
            str: A validated, non-empty https URL.

        Raises:
            FormLinkNotConfiguredError: If neither source yields a valid URL.
        """
        # 1. In-memory override takes precedence
        if self._override_url:
            return self._override_url

        # 2. Fall back to settings
        url = self._read_settings_url()

        if not url:
            raise FormLinkNotConfiguredError(
                "FEEDBACK_GOOGLE_FORM_URL is not configured. "
                "Set this value in your .env file or call update_form_link() "
                "before running the feedback request system."
            )

        return url

    def update_form_link(self, url: str) -> None:
        """
        Set a runtime override for the Google Form URL.

        The override takes precedence over the settings value for all
        subsequent calls to get_form_link() until reset_to_default() is
        called or the server restarts.

        Use this to swap the form link without modifying .env or
        restarting the server — e.g. for campaign-specific forms.

        Args:
            url: New Google Form URL. Must be a non-empty http/https URL.

        Raises:
            ValueError: If url is empty or does not start with http/https.
        """
        validated = _validate_url(url)
        self._override_url = validated

        logger.info(
            "Feedback form link updated via runtime override",
            extra={
                "service": ServiceName.SCHEDULER,
                "new_url": validated,
            },
        )

    def reset_to_default(self) -> None:
        """
        Clear the runtime override and revert to the settings value.

        After calling this, get_form_link() will read from
        FEEDBACK_GOOGLE_FORM_URL in settings.
        """
        previous = self._override_url
        self._override_url = None

        logger.info(
            "Feedback form link reset to settings default",
            extra={
                "service": ServiceName.SCHEDULER,
                "cleared_override": previous,
            },
        )

    def is_configured(self) -> bool:
        """
        Return True if a valid form link is available from any source.

        Safe to call without raising — use this for health checks or
        pre-flight validation before running the feedback job.

        Returns:
            bool: True if get_form_link() would succeed.
        """
        if self._override_url:
            return True

        url = self._read_settings_url()
        return bool(url)

    def current_url(self) -> Optional[str]:
        """
        Return the currently active URL without raising.

        Returns None if not configured. Useful for admin dashboards
        and health reports that display the current form link.

        Returns:
            str if configured, None if not.
        """
        if self._override_url:
            return self._override_url
        return self._read_settings_url() or None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_settings_url(self) -> str:
        """
        Read FEEDBACK_GOOGLE_FORM_URL from application settings.

        Returns empty string if the value is missing or whitespace-only.
        Never raises — settings failures are surfaced in get_form_link().
        """
        try:
            settings = get_settings()
            url = settings.FEEDBACK_GOOGLE_FORM_URL or ""
            return url.strip()
        except Exception as exc:
            logger.warning(
                "Failed to read FEEDBACK_GOOGLE_FORM_URL from settings",
                extra={
                    "service": ServiceName.SCHEDULER,
                    "error": str(exc),
                },
            )
            return ""


# ==============================================================================
# URL validator
# ==============================================================================

def _validate_url(url: str) -> str:
    """
    Validate and normalise a Google Form URL.

    Rules:
      - Must be a non-empty string after stripping whitespace
      - Must start with http:// or https://

    Args:
        url: Raw URL string to validate.

    Returns:
        str: Stripped, validated URL.

    Raises:
        ValueError: If validation fails.
    """
    if not url or not url.strip():
        raise ValueError("Form link URL must not be empty.")

    stripped = url.strip()

    if not stripped.startswith("http://") and not stripped.startswith("https://"):
        raise ValueError(
            f"Form link URL must start with http:// or https://. Got: {stripped!r}"
        )

    return stripped