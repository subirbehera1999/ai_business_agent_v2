# ==============================================================================
# File: app/integrations/google_reviews_client.py
# Purpose: Client for Google Places API and Google My Business API.
#          Handles all outbound Google API calls for the system:
#
#          Google Places API (used for):
#            - get_place_details()    → competitor rating + review count
#            - search_nearby_places() → discover competitor businesses
#
#          Google My Business API (used for):
#            - get_reviews()          → fetch new reviews for a business
#            - post_reply()           → post AI-generated reply to a review
#            - get_account_info()     → verify Google Business connection
#
#          Design:
#            - All methods use retry_utils.with_google_retry (3 attempts,
#              exponential backoff: 2s → 5s → 10s)
#            - Timeouts enforced on every HTTP call (settings.EXTERNAL_API_TIMEOUT_SECONDS)
#            - Pagination handled internally — callers receive flat lists
#            - Rate limiting: Google Places enforces per-second and per-day quotas.
#              This client never parallelises calls; all requests are sequential.
#            - Never raises to callers — returns GoogleApiResult which carries
#              success/failure flag and structured data or error detail.
#            - Authentication via service account JSON (Google My Business)
#              and API key (Google Places) — both loaded from environment.
#
#          HTTP client: httpx.AsyncClient (async, production-grade)
#
#          Prompt safety compliance:
#            Reviewer names and review text are returned as-is from Google.
#            Callers (review_validator, sentiment_service) are responsible
#            for stripping sensitive data before further processing.
# ==============================================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from app.config.constants import ServiceName
from app.config.settings import get_settings
from app.utils.retry_utils import with_google_retry

logger = logging.getLogger(ServiceName.GOOGLE_REVIEWS)
settings = get_settings()

# ---------------------------------------------------------------------------
# Google API base URLs
# ---------------------------------------------------------------------------
_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
_PLACES_NEARBYSEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
_MY_BUSINESS_BASE_URL = "https://mybusinessaccountmanagement.googleapis.com/v1"
_MY_BUSINESS_REVIEWS_URL = (
    "https://mybusiness.googleapis.com/v4/accounts/{account_id}"
    "/locations/{location_id}/reviews"
)
_MY_BUSINESS_REPLY_URL = (
    "https://mybusiness.googleapis.com/v4/accounts/{account_id}"
    "/locations/{location_id}/reviews/{review_id}/reply"
)

# Default fields to request from Places Details API
_DEFAULT_PLACE_FIELDS = [
    "name",
    "rating",
    "user_ratings_total",
    "formatted_address",
    "opening_hours",
    "place_id",
    "types",
]

# Maximum reviews per page from My Business API
_REVIEWS_PAGE_SIZE = 50

# Maximum pages fetched per sync call (prevents unbounded pagination)
_MAX_PAGES = 10


# ==============================================================================
# Result dataclasses
# ==============================================================================

@dataclass
class GoogleReview:
    """
    A single review fetched from Google My Business API.

    Attributes:
        review_id:       Google's internal review identifier.
        reviewer_name:   Display name of the reviewer.
        star_rating:     Star rating string from Google:
                         "ONE", "TWO", "THREE", "FOUR", "FIVE".
        star_rating_int: Integer 1–5.
        comment:         Review text body (may be None for rating-only).
        create_time:     ISO 8601 timestamp of when the review was created.
        update_time:     ISO 8601 timestamp of the last update.
        reply_comment:   Existing reply text (None if no reply yet).
        reply_time:      ISO 8601 timestamp of the existing reply.
    """
    review_id: str
    reviewer_name: str
    star_rating: str
    star_rating_int: int
    comment: Optional[str]
    create_time: str
    update_time: str
    reply_comment: Optional[str] = None
    reply_time: Optional[str] = None

    @property
    def has_reply(self) -> bool:
        return self.reply_comment is not None

    @property
    def has_text(self) -> bool:
        return bool(self.comment and self.comment.strip())


@dataclass
class GoogleApiResult:
    """
    Wrapper for all Google API responses.

    Attributes:
        success:     True if the call succeeded.
        data:        Response data (type depends on the method called).
        error:       Error message if success=False.
        status_code: HTTP status code of the last response.
        raw:         Raw response dict (for debugging).
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    raw: Optional[dict] = field(default=None, repr=False)

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403)

    @property
    def is_not_found(self) -> bool:
        return self.status_code == 404


# ==============================================================================
# Google Reviews Client
# ==============================================================================

class GoogleReviewsClient:
    """
    Async client for Google Places API and Google My Business API.

    Instantiated once per application and shared across services
    and schedulers. Uses a single shared httpx.AsyncClient for
    connection pooling efficiency.

    Usage:
        client = GoogleReviewsClient()
        await client.initialise()

        result = await client.get_place_details(
            place_id="ChIJ...",
            fields=["name", "rating", "user_ratings_total"],
        )

        if result.success:
            rating = result.data.get("rating")
        else:
            logger.error(result.error)

        await client.close()
    """

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None
        self._places_api_key = settings.GOOGLE_PLACES_API_KEY
        self._access_token: Optional[str] = None   # OAuth2 token for My Business API

    async def initialise(self) -> None:
        """
        Initialise the shared HTTP client.
        Must be called before any API methods are used.
        Called from app lifespan or dependency injection.
        """
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.EXTERNAL_API_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers={"User-Agent": "AIBusinessAgent/1.0"},
        )
        logger.info(
            "GoogleReviewsClient initialised",
            extra={"service": ServiceName.GOOGLE_REVIEWS},
        )

    async def close(self) -> None:
        """Close the shared HTTP client. Called from app lifespan shutdown."""
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info(
            "GoogleReviewsClient closed",
            extra={"service": ServiceName.GOOGLE_REVIEWS},
        )

    # ------------------------------------------------------------------
    # Google Places API — Competitor data
    # ------------------------------------------------------------------

    @with_google_retry
    async def get_place_details(
        self,
        place_id: str,
        fields: Optional[list[str]] = None,
    ) -> GoogleApiResult:
        """
        Fetch business details from Google Places API for a given Place ID.

        Used by competitor_service.py to get competitor rating and review count.

        Args:
            place_id: Google Place ID (e.g. "ChIJN1t_tDeuEmsRUsoyG83frY4").
            fields:   List of fields to request. Defaults to _DEFAULT_PLACE_FIELDS.
                      Requesting only needed fields reduces billing costs.

        Returns:
            GoogleApiResult with data=dict of requested fields.
        """
        self._ensure_initialised()
        requested_fields = fields or _DEFAULT_PLACE_FIELDS
        fields_str = ",".join(requested_fields)

        params = {
            "place_id": place_id,
            "fields": fields_str,
            "key": self._places_api_key,
        }

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "place_id": place_id,
        }

        try:
            response = await self._http.get(
                _PLACES_DETAILS_URL,
                params=params,
            )
            data = response.json()
            status = data.get("status", "")

            if status == "OK":
                logger.debug(
                    "Place details fetched successfully",
                    extra={**log_extra, "name": data.get("result", {}).get("name")},
                )
                return GoogleApiResult(
                    success=True,
                    data=data.get("result", {}),
                    status_code=response.status_code,
                    raw=data,
                )

            # Google API error status codes
            error_msg = _google_status_to_message(status, place_id)
            logger.warning(
                "Places API returned error status",
                extra={**log_extra, "google_status": status, "error": error_msg},
            )
            return GoogleApiResult(
                success=False,
                error=error_msg,
                status_code=response.status_code,
                raw=data,
            )

        except httpx.TimeoutException as exc:
            logger.error(
                "Places API timeout",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

        except httpx.HTTPError as exc:
            logger.error(
                "Places API HTTP error",
                extra={**log_extra, "error": str(exc)},
            )
            raise   # re-raise for retry wrapper

    @with_google_retry
    async def search_nearby_places(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
        business_type: str,
        max_results: int = 5,
    ) -> GoogleApiResult:
        """
        Search for nearby businesses of a given type using Google Places API.

        Used during onboarding to help businesses discover competitors.

        Args:
            latitude:       Centre latitude for the search.
            longitude:      Centre longitude for the search.
            radius_meters:  Search radius in metres (max 50,000).
            business_type:  Google place type e.g. "restaurant", "salon".
            max_results:    Maximum number of results to return (default 5).

        Returns:
            GoogleApiResult with data=list of place dicts.
        """
        self._ensure_initialised()

        params = {
            "location": f"{latitude},{longitude}",
            "radius": min(radius_meters, 50_000),
            "type": business_type,
            "key": self._places_api_key,
        }

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "lat": latitude,
            "lng": longitude,
            "type": business_type,
        }

        try:
            response = await self._http.get(
                _PLACES_NEARBYSEARCH_URL,
                params=params,
            )
            data = response.json()
            status = data.get("status", "")

            if status in ("OK", "ZERO_RESULTS"):
                results = data.get("results", [])[:max_results]
                logger.debug(
                    "Nearby search completed",
                    extra={**log_extra, "result_count": len(results)},
                )
                return GoogleApiResult(
                    success=True,
                    data=results,
                    status_code=response.status_code,
                    raw=data,
                )

            error_msg = _google_status_to_message(status, f"{latitude},{longitude}")
            logger.warning(
                "Nearby search returned error status",
                extra={**log_extra, "google_status": status},
            )
            return GoogleApiResult(
                success=False,
                error=error_msg,
                status_code=response.status_code,
                raw=data,
            )

        except httpx.TimeoutException as exc:
            logger.error("Nearby search timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("Nearby search HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    # ------------------------------------------------------------------
    # Google My Business API — Review management
    # ------------------------------------------------------------------

    @with_google_retry
    async def get_reviews(
        self,
        account_id: str,
        location_id: str,
        page_token: Optional[str] = None,
        order_by: str = "updateTime desc",
    ) -> GoogleApiResult:
        """
        Fetch a single page of reviews for a Google My Business location.

        For full history, use get_all_reviews() which paginates automatically.

        Args:
            account_id:   Google My Business account ID.
            location_id:  Google My Business location ID.
            page_token:   Pagination token from a previous response.
            order_by:     Sort order. Default: newest first.

        Returns:
            GoogleApiResult with data={
                "reviews": list[dict],
                "nextPageToken": str | None,
                "totalReviewCount": int,
                "averageRating": float,
            }
        """
        self._ensure_initialised()
        token = await self._get_access_token()

        url = _MY_BUSINESS_REVIEWS_URL.format(
            account_id=account_id,
            location_id=location_id,
        )
        params: dict[str, Any] = {
            "pageSize": _REVIEWS_PAGE_SIZE,
            "orderBy": order_by,
        }
        if page_token:
            params["pageToken"] = page_token

        headers = {"Authorization": f"Bearer {token}"}

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "account_id": account_id,
            "location_id": location_id,
        }

        try:
            response = await self._http.get(url, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                reviews_raw = data.get("reviews", [])
                parsed_reviews = [_parse_review(r) for r in reviews_raw]
                logger.debug(
                    "Reviews page fetched",
                    extra={**log_extra, "count": len(parsed_reviews)},
                )
                return GoogleApiResult(
                    success=True,
                    data={
                        "reviews": parsed_reviews,
                        "nextPageToken": data.get("nextPageToken"),
                        "totalReviewCount": data.get("totalReviewCount", 0),
                        "averageRating": data.get("averageRating", 0.0),
                    },
                    status_code=200,
                    raw=data,
                )

            return _handle_http_error(response, log_extra, "get_reviews")

        except httpx.TimeoutException as exc:
            logger.error("get_reviews timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("get_reviews HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    async def get_all_reviews(
        self,
        account_id: str,
        location_id: str,
        max_reviews: int = 500,
    ) -> GoogleApiResult:
        """
        Fetch all reviews for a location, handling pagination automatically.

        Fetches up to max_reviews reviews across multiple pages.
        Stops when there are no more pages or max_reviews is reached.

        Args:
            account_id:   Google My Business account ID.
            location_id:  Google My Business location ID.
            max_reviews:  Maximum total reviews to fetch (safety cap).

        Returns:
            GoogleApiResult with data={
                "reviews": list[GoogleReview],
                "totalReviewCount": int,
                "averageRating": float,
                "pages_fetched": int,
            }
        """
        all_reviews: list[GoogleReview] = []
        page_token: Optional[str] = None
        pages_fetched = 0
        total_count = 0
        avg_rating = 0.0

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "account_id": account_id,
            "location_id": location_id,
        }

        for _ in range(_MAX_PAGES):
            result = await self.get_reviews(
                account_id=account_id,
                location_id=location_id,
                page_token=page_token,
            )

            if not result.success:
                if pages_fetched == 0:
                    # First page failed — surface the error
                    return result
                # Partial success — return what we have
                logger.warning(
                    "Pagination interrupted — returning partial results",
                    extra={**log_extra, "pages_fetched": pages_fetched},
                )
                break

            page_data = result.data
            all_reviews.extend(page_data["reviews"])
            pages_fetched += 1
            total_count = page_data.get("totalReviewCount", total_count)
            avg_rating = page_data.get("averageRating", avg_rating)
            page_token = page_data.get("nextPageToken")

            if len(all_reviews) >= max_reviews:
                all_reviews = all_reviews[:max_reviews]
                break

            if not page_token:
                break   # No more pages

        logger.info(
            "All reviews fetched",
            extra={
                **log_extra,
                "total_fetched": len(all_reviews),
                "pages_fetched": pages_fetched,
                "total_on_google": total_count,
            },
        )

        return GoogleApiResult(
            success=True,
            data={
                "reviews": all_reviews,
                "totalReviewCount": total_count,
                "averageRating": avg_rating,
                "pages_fetched": pages_fetched,
            },
            status_code=200,
        )

    @with_google_retry
    async def post_reply(
        self,
        account_id: str,
        location_id: str,
        review_id: str,
        reply_text: str,
    ) -> GoogleApiResult:
        """
        Post a reply to a Google review via My Business API.

        Called by the review pipeline after ai_reply_service.py generates
        a reply. If the review already has a reply, this will overwrite it.

        Args:
            account_id:   Google My Business account ID.
            location_id:  Google My Business location ID.
            review_id:    Google review identifier.
            reply_text:   The reply text to post (max 4096 chars).

        Returns:
            GoogleApiResult with data={"comment": reply_text, "updateTime": str}
        """
        self._ensure_initialised()
        token = await self._get_access_token()

        url = _MY_BUSINESS_REPLY_URL.format(
            account_id=account_id,
            location_id=location_id,
            review_id=review_id,
        )

        # Google My Business API expects a PUT with the reply body
        payload = {"comment": reply_text[:4096]}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "account_id": account_id,
            "location_id": location_id,
            "review_id": review_id,
            "reply_chars": len(reply_text),
        }

        try:
            response = await self._http.put(url, json=payload, headers=headers)

            if response.status_code in (200, 201):
                data = response.json()
                logger.info(
                    "Review reply posted successfully",
                    extra=log_extra,
                )
                return GoogleApiResult(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    raw=data,
                )

            return _handle_http_error(response, log_extra, "post_reply")

        except httpx.TimeoutException as exc:
            logger.error("post_reply timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("post_reply HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    @with_google_retry
    async def get_location_info(
        self,
        account_id: str,
        location_id: str,
    ) -> GoogleApiResult:
        """
        Fetch basic location information from Google My Business API.

        Used during onboarding to verify the Google Business connection
        and retrieve the current average rating and review count.

        Args:
            account_id:   Google My Business account ID.
            location_id:  Google My Business location ID.

        Returns:
            GoogleApiResult with data containing location metadata.
        """
        self._ensure_initialised()
        token = await self._get_access_token()

        url = (
            f"https://mybusiness.googleapis.com/v4/accounts/"
            f"{account_id}/locations/{location_id}"
        )
        headers = {"Authorization": f"Bearer {token}"}

        log_extra = {
            "service": ServiceName.GOOGLE_REVIEWS,
            "account_id": account_id,
            "location_id": location_id,
        }

        try:
            response = await self._http.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                logger.debug(
                    "Location info fetched",
                    extra={**log_extra, "name": data.get("locationName")},
                )
                return GoogleApiResult(
                    success=True,
                    data=data,
                    status_code=200,
                    raw=data,
                )

            return _handle_http_error(response, log_extra, "get_location_info")

        except httpx.TimeoutException as exc:
            logger.error("get_location_info timeout", extra={**log_extra, "error": str(exc)})
            raise
        except httpx.HTTPError as exc:
            logger.error("get_location_info HTTP error", extra={**log_extra, "error": str(exc)})
            raise

    # ------------------------------------------------------------------
    # OAuth2 token management
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """
        Return a valid OAuth2 access token for Google My Business API.

        Uses the service account credentials from settings.
        In production, tokens are short-lived (1 hour). This method
        returns the cached token or refreshes it if needed.

        For the current implementation, we use the pre-configured
        access token from settings (set via service account or OAuth flow).
        A full token refresh mechanism should be implemented when
        Google Cloud service account credentials are configured.

        Returns:
            str: Valid Bearer token.

        Raises:
            RuntimeError: If no token is available.
        """
        token = settings.GOOGLE_MY_BUSINESS_ACCESS_TOKEN
        if not token:
            raise RuntimeError(
                "GOOGLE_MY_BUSINESS_ACCESS_TOKEN is not configured. "
                "Complete Google Business OAuth flow first."
            )
        return token

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Raise if the HTTP client has not been initialised."""
        if self._http is None:
            raise RuntimeError(
                "GoogleReviewsClient has not been initialised. "
                "Call await client.initialise() before use."
            )


# ==============================================================================
# Module-level helpers
# ==============================================================================

_STAR_RATING_MAP: dict[str, int] = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
}


def _parse_review(raw: dict) -> GoogleReview:
    """
    Parse a raw Google My Business API review dict into a GoogleReview.

    Args:
        raw: Raw review dict from the API response.

    Returns:
        GoogleReview with typed fields.
    """
    star_str = raw.get("starRating", "THREE")
    star_int = _STAR_RATING_MAP.get(star_str, 3)

    reviewer = raw.get("reviewer", {})
    reviewer_name = reviewer.get("displayName", "Anonymous")

    reply = raw.get("reviewReply")
    reply_comment: Optional[str] = None
    reply_time: Optional[str] = None
    if reply:
        reply_comment = reply.get("comment")
        reply_time = reply.get("updateTime")

    return GoogleReview(
        review_id=raw.get("reviewId", ""),
        reviewer_name=reviewer_name,
        star_rating=star_str,
        star_rating_int=star_int,
        comment=raw.get("comment"),
        create_time=raw.get("createTime", ""),
        update_time=raw.get("updateTime", ""),
        reply_comment=reply_comment,
        reply_time=reply_time,
    )


def _google_status_to_message(status: str, context: str) -> str:
    """
    Convert a Google Places API status code to a human-readable error message.

    Args:
        status:  Google API status string e.g. "NOT_FOUND", "OVER_QUERY_LIMIT".
        context: Context string (place_id or coordinates) for the message.

    Returns:
        str: Human-readable error description.
    """
    messages = {
        "NOT_FOUND":            f"Place not found: {context}",
        "ZERO_RESULTS":         f"No results found for: {context}",
        "OVER_DAILY_LIMIT":     "Google Places API daily quota exceeded",
        "OVER_QUERY_LIMIT":     "Google Places API rate limit hit — retry after delay",
        "REQUEST_DENIED":       "Google Places API request denied — check API key",
        "INVALID_REQUEST":      f"Invalid request parameters for: {context}",
        "UNKNOWN_ERROR":        "Google Places API returned an unknown error",
    }
    return messages.get(status, f"Google Places API error: {status}")


def _handle_http_error(
    response: httpx.Response,
    log_extra: dict,
    method_name: str,
) -> GoogleApiResult:
    """
    Handle a non-200 HTTP response from the My Business API.

    Logs the failure and returns a GoogleApiResult with success=False.
    Does NOT raise — callers handle the failed result directly.

    Args:
        response:    The failed httpx.Response.
        log_extra:   Structured log context.
        method_name: Name of the calling method for log context.

    Returns:
        GoogleApiResult with success=False.
    """
    status_code = response.status_code
    try:
        body = response.json()
        error_detail = body.get("error", {}).get("message", response.text[:200])
    except Exception:
        error_detail = response.text[:200]

    error_msg = f"HTTP {status_code}: {error_detail}"

    logger.error(
        f"Google API {method_name} failed",
        extra={
            **log_extra,
            "status_code": status_code,
            "error": error_msg,
        },
    )

    return GoogleApiResult(
        success=False,
        error=error_msg,
        status_code=status_code,
    )