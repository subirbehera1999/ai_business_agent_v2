# ==============================================================================
# File: tests/test_integrations.py
# Purpose: Tests for all external platform integration clients.
#
#          Covers:
#            1. GoogleReviewsClient — _parse_review helper (all star ratings,
#               missing fields, replies), GoogleReview property methods,
#               GoogleApiResult properties, rate-limit / auth error detection
#            2. GoogleSheetsClient — _col_index_to_letter, _build_a1_range,
#               _normalise_rows, _filter_rows_since, _extract_iso_prefix,
#               SheetData property methods, SheetsApiResult error classification
#            3. WhatsAppClient — _normalise_phone_number (all Indian formats),
#               WhatsAppSendResult property methods, send_text_message with
#               mocked HTTP, send_multi_part sequencing and failure-stop,
#               empty-parts guard
#
#          Design:
#            No real HTTP calls — all network calls are mocked via
#            unittest.mock.AsyncMock or httpx mock responses.
#            Tests focus on:
#              - Pure helper functions (no mocking needed)
#              - Client method behaviour with controlled HTTP responses
#              - Error classification properties on result dataclasses
#
#          Running:
#            pytest tests/test_integrations.py -v
#            pytest tests/test_integrations.py -v -k "whatsapp"
# ==============================================================================

import os
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Env vars before any app import
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL",             "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("JWT_SECRET_KEY",           "test-secret-key-at-least-32-characters-long!!")
os.environ.setdefault("RAZORPAY_KEY_ID",          "rzp_test_key_id")
os.environ.setdefault("RAZORPAY_KEY_SECRET",      "rzp_test_secret")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET",  "rzp_test_webhook_secret")
os.environ.setdefault("WHATSAPP_API_TOKEN",       "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "test_phone_id")
os.environ.setdefault("OPENAI_API_KEY",           "sk-test-key")
os.environ.setdefault("ADMIN_WHATSAPP_NUMBER",    "9999999999")

from app.integrations.google_reviews_client import (       # noqa: E402
    GoogleApiResult,
    GoogleReview,
    GoogleReviewsClient,
    _parse_review,
)
from app.integrations.google_sheets_client import (        # noqa: E402
    SheetData,
    SheetsApiResult,
    _build_a1_range,
    _col_index_to_letter,
    _extract_iso_prefix,
    _filter_rows_since,
    _normalise_rows,
)
from app.integrations.whatsapp_client import (             # noqa: E402
    WhatsAppClient,
    WhatsAppSendResult,
    _normalise_phone_number,
)


# ==============================================================================
# 1. GoogleReviewsClient — _parse_review helper
# ==============================================================================

class TestParseReview:
    """Unit tests for the _parse_review() module-level helper."""

    def _raw(
        self,
        star_rating: str = "FIVE",
        comment: str | None = "Great food!",
        reviewer_name: str = "Rahul Sharma",
        review_id: str = "review_abc123",
        has_reply: bool = False,
    ) -> dict:
        raw: dict = {
            "reviewId":   review_id,
            "starRating": star_rating,
            "comment":    comment,
            "createTime": "2024-10-01T10:00:00Z",
            "updateTime": "2024-10-01T10:00:00Z",
            "reviewer": {
                "displayName": reviewer_name,
            },
        }
        if has_reply:
            raw["reviewReply"] = {
                "comment":    "Thank you!",
                "updateTime": "2024-10-01T11:00:00Z",
            }
        return raw

    def test_five_star_parsed_to_int_5(self) -> None:
        review = _parse_review(self._raw(star_rating="FIVE"))
        assert review.star_rating_int == 5

    def test_four_star_parsed_to_int_4(self) -> None:
        review = _parse_review(self._raw(star_rating="FOUR"))
        assert review.star_rating_int == 4

    def test_three_star_parsed_to_int_3(self) -> None:
        review = _parse_review(self._raw(star_rating="THREE"))
        assert review.star_rating_int == 3

    def test_two_star_parsed_to_int_2(self) -> None:
        review = _parse_review(self._raw(star_rating="TWO"))
        assert review.star_rating_int == 2

    def test_one_star_parsed_to_int_1(self) -> None:
        review = _parse_review(self._raw(star_rating="ONE"))
        assert review.star_rating_int == 1

    def test_unknown_star_rating_defaults_to_3(self) -> None:
        """Unrecognised star_rating string must default to 3 (neutral)."""
        review = _parse_review(self._raw(star_rating="UNKNOWN_RATING"))
        assert review.star_rating_int == 3

    def test_reviewer_name_populated(self) -> None:
        review = _parse_review(self._raw(reviewer_name="Priya Patel"))
        assert review.reviewer_name == "Priya Patel"

    def test_missing_reviewer_defaults_to_anonymous(self) -> None:
        raw    = self._raw()
        del raw["reviewer"]
        review = _parse_review(raw)
        assert review.reviewer_name == "Anonymous"

    def test_comment_text_preserved(self) -> None:
        review = _parse_review(self._raw(comment="Absolutely loved the biryani!"))
        assert review.comment == "Absolutely loved the biryani!"

    def test_none_comment_preserved(self) -> None:
        """Rating-only reviews (no comment) must produce comment=None."""
        review = _parse_review(self._raw(comment=None))
        assert review.comment is None

    def test_reply_parsed_when_present(self) -> None:
        review = _parse_review(self._raw(has_reply=True))
        assert review.reply_comment == "Thank you!"
        assert review.reply_time is not None

    def test_no_reply_fields_are_none(self) -> None:
        review = _parse_review(self._raw(has_reply=False))
        assert review.reply_comment is None
        assert review.reply_time is None

    def test_review_id_populated(self) -> None:
        review = _parse_review(self._raw(review_id="rev_xyz"))
        assert review.review_id == "rev_xyz"

    def test_returns_google_review_instance(self) -> None:
        result = _parse_review(self._raw())
        assert isinstance(result, GoogleReview)


class TestGoogleReviewProperties:
    """Tests for GoogleReview property methods."""

    def _review(self, comment: str | None = "Great!", reply: str | None = None) -> GoogleReview:
        return GoogleReview(
            review_id="rv1",
            reviewer_name="User",
            star_rating="FIVE",
            star_rating_int=5,
            comment=comment,
            create_time="2024-10-01T10:00:00Z",
            update_time="2024-10-01T10:00:00Z",
            reply_comment=reply,
        )

    def test_has_reply_true_when_reply_present(self) -> None:
        assert self._review(reply="Thank you!").has_reply is True

    def test_has_reply_false_when_no_reply(self) -> None:
        assert self._review(reply=None).has_reply is False

    def test_has_text_true_when_comment_present(self) -> None:
        assert self._review(comment="Nice place").has_text is True

    def test_has_text_false_when_comment_none(self) -> None:
        assert self._review(comment=None).has_text is False

    def test_has_text_false_when_comment_whitespace_only(self) -> None:
        assert self._review(comment="   ").has_text is False


class TestGoogleApiResultProperties:
    """Tests for GoogleApiResult error classification properties."""

    def test_is_rate_limited_on_429(self) -> None:
        result = GoogleApiResult(success=False, status_code=429)
        assert result.is_rate_limited is True

    def test_not_rate_limited_on_200(self) -> None:
        result = GoogleApiResult(success=True, status_code=200)
        assert result.is_rate_limited is False

    def test_is_auth_error_on_401(self) -> None:
        result = GoogleApiResult(success=False, status_code=401)
        assert result.is_auth_error is True

    def test_is_auth_error_on_403(self) -> None:
        result = GoogleApiResult(success=False, status_code=403)
        assert result.is_auth_error is True

    def test_is_not_found_on_404(self) -> None:
        result = GoogleApiResult(success=False, status_code=404)
        assert result.is_not_found is True

    def test_success_result_no_errors(self) -> None:
        result = GoogleApiResult(success=True, status_code=200)
        assert result.is_rate_limited is False
        assert result.is_auth_error   is False
        assert result.is_not_found    is False


# ==============================================================================
# 2. GoogleSheetsClient — private helpers
# ==============================================================================

class TestColIndexToLetter:
    """Tests for _col_index_to_letter() helper."""

    def test_col_1_is_A(self) -> None:
        assert _col_index_to_letter(1) == "A"

    def test_col_26_is_Z(self) -> None:
        assert _col_index_to_letter(26) == "Z"

    def test_col_27_is_AA(self) -> None:
        assert _col_index_to_letter(27) == "AA"

    def test_col_50_is_AX(self) -> None:
        assert _col_index_to_letter(50) == "AX"

    def test_col_52_is_AZ(self) -> None:
        assert _col_index_to_letter(52) == "AZ"

    def test_col_53_is_BA(self) -> None:
        assert _col_index_to_letter(53) == "BA"

    def test_all_single_letters_sequential(self) -> None:
        """Columns 1–26 must produce A through Z in order."""
        expected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for i, letter in enumerate(expected, start=1):
            assert _col_index_to_letter(i) == letter


class TestBuildA1Range:
    """Tests for _build_a1_range() helper."""

    def test_basic_range_no_spaces(self) -> None:
        result = _build_a1_range("Sheet1", row_start=1, row_end=100)
        assert "Sheet1" in result
        assert "A1" in result
        assert "100" in result

    def test_tab_with_spaces_is_quoted(self) -> None:
        result = _build_a1_range("Sales Data", row_start=1, row_end=500)
        assert "'Sales Data'" in result

    def test_tab_without_spaces_not_quoted(self) -> None:
        result = _build_a1_range("Sheet1", row_start=1, row_end=100)
        assert "'" not in result

    def test_result_contains_exclamation_separator(self) -> None:
        result = _build_a1_range("Sheet1", row_start=1, row_end=100)
        assert "!" in result

    def test_range_starts_at_given_row(self) -> None:
        result = _build_a1_range("Sheet1", row_start=5, row_end=100)
        assert "A5" in result


class TestNormaliseRows:
    """Tests for _normalise_rows() helper."""

    def test_short_row_padded_to_column_count(self) -> None:
        rows   = [["A", "B"]]
        result = _normalise_rows(rows, column_count=4, skip_empty=False)
        assert result[0] == ["A", "B", "", ""]

    def test_long_row_truncated_to_column_count(self) -> None:
        rows   = [["A", "B", "C", "D", "E"]]
        result = _normalise_rows(rows, column_count=3, skip_empty=False)
        assert result[0] == ["A", "B", "C"]

    def test_values_converted_to_strings(self) -> None:
        rows   = [[100, 200.5, None]]
        result = _normalise_rows(rows, column_count=3, skip_empty=False)
        assert result[0][0] == "100"
        assert result[0][1] == "200.5"

    def test_values_stripped_of_whitespace(self) -> None:
        rows   = [["  hello  ", "  world  "]]
        result = _normalise_rows(rows, column_count=2, skip_empty=False)
        assert result[0] == ["hello", "world"]

    def test_skip_empty_removes_all_empty_rows(self) -> None:
        rows   = [["", ""], ["A", "B"], ["", ""]]
        result = _normalise_rows(rows, column_count=2, skip_empty=True)
        assert len(result) == 1
        assert result[0] == ["A", "B"]

    def test_skip_empty_false_keeps_empty_rows(self) -> None:
        rows   = [["", ""], ["A", "B"]]
        result = _normalise_rows(rows, column_count=2, skip_empty=False)
        assert len(result) == 2

    def test_empty_input_returns_empty_list(self) -> None:
        assert _normalise_rows([], column_count=3, skip_empty=False) == []


class TestExtractIsoPrefix:
    """Tests for _extract_iso_prefix() helper."""

    def test_iso_format_returned_as_is(self) -> None:
        assert _extract_iso_prefix("2024-10-15") == "2024-10-15"

    def test_dd_mm_yyyy_reordered(self) -> None:
        assert _extract_iso_prefix("15-10-2024") == "2024-10-15"

    def test_dd_mm_yy_expanded(self) -> None:
        result = _extract_iso_prefix("15-10-24")
        assert result == "2024-10-15"

    def test_single_digit_day_padded(self) -> None:
        assert _extract_iso_prefix("5-3-2024") == "2024-03-05"

    def test_unrecognisable_format_returns_none(self) -> None:
        assert _extract_iso_prefix("not-a-date") is None

    def test_missing_parts_returns_none(self) -> None:
        assert _extract_iso_prefix("2024-10") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_iso_prefix("") is None


class TestFilterRowsSince:
    """Tests for _filter_rows_since() helper."""

    def _make_rows(self, dates: list[str]) -> list[list[str]]:
        return [[d, "some_data"] for d in dates]

    def test_rows_on_or_after_since_date_included(self) -> None:
        rows   = self._make_rows(["2024-10-10", "2024-10-15", "2024-10-20"])
        result = _filter_rows_since(rows, date_col_idx=0, since_date_str="2024-10-15")
        assert len(result) == 2
        assert result[0][0] == "2024-10-15"
        assert result[1][0] == "2024-10-20"

    def test_rows_before_since_date_excluded(self) -> None:
        rows   = self._make_rows(["2024-10-01", "2024-10-05", "2024-10-09"])
        result = _filter_rows_since(rows, date_col_idx=0, since_date_str="2024-10-10")
        assert len(result) == 0

    def test_empty_date_cell_excluded(self) -> None:
        rows   = [["", "data"], ["2024-10-15", "data"]]
        result = _filter_rows_since(rows, date_col_idx=0, since_date_str="2024-10-01")
        assert len(result) == 1

    def test_unparseable_date_excluded(self) -> None:
        rows   = [["not-a-date", "data"], ["2024-10-15", "data"]]
        result = _filter_rows_since(rows, date_col_idx=0, since_date_str="2024-10-01")
        assert len(result) == 1

    def test_indian_date_format_dd_mm_yyyy(self) -> None:
        """DD/MM/YYYY format (common in Indian spreadsheets) must be parsed."""
        rows   = [["15/10/2024", "data"]]
        result = _filter_rows_since(rows, date_col_idx=0, since_date_str="2024-10-01")
        assert len(result) == 1

    def test_date_col_out_of_range_excluded(self) -> None:
        rows   = [["only_one_col"]]
        result = _filter_rows_since(rows, date_col_idx=5, since_date_str="2024-10-01")
        assert len(result) == 0

    def test_empty_rows_returns_empty(self) -> None:
        assert _filter_rows_since([], date_col_idx=0, since_date_str="2024-10-01") == []


class TestSheetDataProperties:
    """Tests for SheetData property methods."""

    def _sheet(self, headers: list[str], rows: list[list[str]]) -> SheetData:
        return SheetData(
            spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
            sheet_tab="Sheet1",
            headers=headers,
            rows=rows,
            total_rows=len(rows),
        )

    def test_is_empty_true_when_no_rows(self) -> None:
        sheet = self._sheet(headers=["Date", "Revenue"], rows=[])
        assert sheet.is_empty is True

    def test_is_empty_false_when_has_rows(self) -> None:
        sheet = self._sheet(headers=["Date"], rows=[["2024-10-01"]])
        assert sheet.is_empty is False

    def test_column_count_matches_header_length(self) -> None:
        sheet = self._sheet(headers=["A", "B", "C", "D"], rows=[])
        assert sheet.column_count == 4

    def test_row_count_matches_data_rows(self) -> None:
        sheet = self._sheet(
            headers=["Date"],
            rows=[["2024-10-01"], ["2024-10-02"], ["2024-10-03"]],
        )
        assert sheet.row_count() == 3


class TestSheetsApiResultProperties:
    """Tests for SheetsApiResult error classification properties."""

    def test_is_auth_error_true(self) -> None:
        result = SheetsApiResult(success=False, error_type="auth")
        assert result.is_auth_error is True

    def test_is_not_found_true(self) -> None:
        result = SheetsApiResult(success=False, error_type="not_found")
        assert result.is_not_found is True

    def test_is_quota_error_true(self) -> None:
        result = SheetsApiResult(success=False, error_type="quota")
        assert result.is_quota_error is True

    def test_success_result_no_error_flags(self) -> None:
        result = SheetsApiResult(success=True)
        assert result.is_auth_error   is False
        assert result.is_not_found    is False
        assert result.is_quota_error  is False


# ==============================================================================
# 3. WhatsAppClient — _normalise_phone_number
# ==============================================================================

class TestNormalisePhoneNumber:
    """Tests for _normalise_phone_number() pure helper."""

    def test_e164_number_unchanged(self) -> None:
        assert _normalise_phone_number("+919876543210") == "+919876543210"

    def test_10_digit_number_gets_plus91(self) -> None:
        assert _normalise_phone_number("9876543210") == "+919876543210"

    def test_leading_zero_replaced_with_plus91(self) -> None:
        assert _normalise_phone_number("09876543210") == "+919876543210"

    def test_number_with_dashes_normalised(self) -> None:
        assert _normalise_phone_number("98765-43210") == "+919876543210"

    def test_number_with_spaces_normalised(self) -> None:
        assert _normalise_phone_number("98765 43210") == "+919876543210"

    def test_number_with_parentheses_normalised(self) -> None:
        result = _normalise_phone_number("(98765) 43210")
        assert "+" in result
        assert " " not in result
        assert "(" not in result

    def test_uk_number_preserved(self) -> None:
        """Non-Indian E.164 numbers must be preserved as-is."""
        assert _normalise_phone_number("+447911123456") == "+447911123456"

    def test_us_number_preserved(self) -> None:
        assert _normalise_phone_number("+12125551234") == "+12125551234"

    def test_result_starts_with_plus(self) -> None:
        result = _normalise_phone_number("9876543210")
        assert result.startswith("+")

    def test_result_contains_no_spaces(self) -> None:
        result = _normalise_phone_number("98765 43210")
        assert " " not in result


# ==============================================================================
# 4. WhatsAppClient — WhatsAppSendResult properties
# ==============================================================================

class TestWhatsAppSendResultProperties:
    """Tests for WhatsAppSendResult error classification properties."""

    def test_is_rate_limited_on_429_status(self) -> None:
        result = WhatsAppSendResult(success=False, status_code=429)
        assert result.is_rate_limited is True

    def test_is_rate_limited_on_meta_error_code_130429(self) -> None:
        result = WhatsAppSendResult(success=False, error_code=130429)
        assert result.is_rate_limited is True

    def test_is_invalid_number_on_131026(self) -> None:
        result = WhatsAppSendResult(success=False, error_code=131026)
        assert result.is_invalid_number is True

    def test_is_invalid_number_on_131047(self) -> None:
        result = WhatsAppSendResult(success=False, error_code=131047)
        assert result.is_invalid_number is True

    def test_is_auth_error_on_401(self) -> None:
        result = WhatsAppSendResult(success=False, status_code=401)
        assert result.is_auth_error is True

    def test_is_auth_error_on_403(self) -> None:
        result = WhatsAppSendResult(success=False, status_code=403)
        assert result.is_auth_error is True

    def test_is_auth_error_on_meta_error_190(self) -> None:
        result = WhatsAppSendResult(success=False, error_code=190)
        assert result.is_auth_error is True

    def test_successful_result_has_no_error_flags(self) -> None:
        result = WhatsAppSendResult(success=True, status_code=200)
        assert result.is_rate_limited  is False
        assert result.is_invalid_number is False
        assert result.is_auth_error    is False


# ==============================================================================
# 5. WhatsAppClient — send_text_message with mocked HTTP
# ==============================================================================

class TestWhatsAppClientSendText:
    """Tests for WhatsAppClient.send_text_message() with HTTP mocked."""

    def _make_client(self) -> WhatsAppClient:
        """Construct a WhatsAppClient bypassing __init__ HTTP setup."""
        client         = WhatsAppClient.__new__(WhatsAppClient)
        client._http   = AsyncMock()
        client._token  = "test_token"
        client._phone_number_id = "test_phone_id"
        return client

    def _meta_success_response(self, message_id: str = "wamid.test123") -> MagicMock:
        resp          = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "messages": [{"id": message_id}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    def _meta_error_response(self, status_code: int = 429) -> MagicMock:
        resp          = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "error": {
                "message": "Rate limit hit",
                "code": 130429,
            }
        }
        return resp

    @pytest.mark.asyncio
    async def test_send_text_returns_success_on_200(self) -> None:
        client = self._make_client()
        client._http.post = AsyncMock(
            return_value=self._meta_success_response()
        )

        with patch.object(client, "_ensure_initialised"):
            result = await client.send_text_message(
                to="+919876543210",
                text="Hello from AI Business Agent!",
            )

        assert result.success is True
        assert result.message_id == "wamid.test123"

    @pytest.mark.asyncio
    async def test_send_text_normalises_phone_number(self) -> None:
        """send_text_message must normalise bare 10-digit numbers before sending."""
        client      = self._make_client()
        posted_to   = {}

        async def _capture_post(url, **kwargs):
            posted_to["payload"] = kwargs.get("json", {})
            return self._meta_success_response()

        client._http.post = _capture_post

        with patch.object(client, "_ensure_initialised"):
            await client.send_text_message(
                to="9876543210",   # bare 10-digit — must become +919876543210
                text="Test message",
            )

        assert posted_to.get("payload", {}).get("to") == "+919876543210"

    @pytest.mark.asyncio
    async def test_send_text_returns_failure_on_error_response(self) -> None:
        client = self._make_client()
        client._http.post = AsyncMock(
            return_value=self._meta_error_response(status_code=429)
        )

        with patch.object(client, "_ensure_initialised"):
            result = await client.send_text_message(
                to="+919876543210",
                text="Hello!",
            )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_text_never_raises_on_http_exception(self) -> None:
        """send_text_message must catch exceptions and return failure result."""
        client = self._make_client()
        client._http.post = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with patch.object(client, "_ensure_initialised"):
            result = await client.send_text_message(
                to="+919876543210",
                text="Hello!",
            )

        assert isinstance(result, WhatsAppSendResult)
        assert result.success is False


# ==============================================================================
# 6. WhatsAppClient — send_multi_part
# ==============================================================================

class TestWhatsAppClientSendMultiPart:
    """Tests for WhatsAppClient.send_multi_part()."""

    def _make_client(self) -> WhatsAppClient:
        client          = WhatsAppClient.__new__(WhatsAppClient)
        client._http    = AsyncMock()
        client._token   = "test_token"
        client._phone_number_id = "test_phone_id"
        return client

    @pytest.mark.asyncio
    async def test_empty_parts_returns_failure(self) -> None:
        client = self._make_client()

        with patch.object(client, "_ensure_initialised"):
            result = await client.send_multi_part(
                to="+919876543210",
                parts=[],
            )

        assert result.success is False
        assert "empty" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_all_parts_sent_on_success(self) -> None:
        """All 3 parts must be sent if each succeeds."""
        client  = self._make_client()
        sent    = []

        async def _mock_send_text(to: str, text: str, **kwargs) -> WhatsAppSendResult:
            sent.append(text)
            return WhatsAppSendResult(success=True, message_id=f"mid_{len(sent)}", to=to)

        with (
            patch.object(client, "_ensure_initialised"),
            patch.object(client, "send_text_message", side_effect=_mock_send_text),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.send_multi_part(
                to="+919876543210",
                parts=["Part 1", "Part 2", "Part 3"],
            )

        assert result.success is True
        assert result.parts_sent == 3
        assert len(sent) == 3

    @pytest.mark.asyncio
    async def test_failure_on_part_2_stops_sending_part_3(self) -> None:
        """
        If part 2 fails, part 3 must NOT be sent.
        send_multi_part must abort at the first failure.
        """
        client  = self._make_client()
        sent    = []
        call_n  = 0

        async def _mock_send_text(to: str, text: str, **kwargs) -> WhatsAppSendResult:
            nonlocal call_n
            call_n += 1
            sent.append(text)
            if call_n == 2:
                return WhatsAppSendResult(success=False, error="Send failed")
            return WhatsAppSendResult(success=True, message_id=f"mid_{call_n}", to=to)

        with (
            patch.object(client, "_ensure_initialised"),
            patch.object(client, "send_text_message", side_effect=_mock_send_text),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.send_multi_part(
                to="+919876543210",
                parts=["Part 1", "Part 2", "Part 3"],
            )

        assert result.success is False
        assert result.parts_sent == 1      # part 1 succeeded before failure
        assert len(sent) == 2              # part 3 was never sent

    @pytest.mark.asyncio
    async def test_blank_parts_skipped_silently(self) -> None:
        """Blank or whitespace-only parts must be skipped without error."""
        client = self._make_client()
        sent   = []

        async def _mock_send_text(to: str, text: str, **kwargs) -> WhatsAppSendResult:
            sent.append(text)
            return WhatsAppSendResult(success=True, message_id="mid1", to=to)

        with (
            patch.object(client, "_ensure_initialised"),
            patch.object(client, "send_text_message", side_effect=_mock_send_text),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.send_multi_part(
                to="+919876543210",
                parts=["Real part", "   ", "", "Another real part"],
            )

        assert len(sent) == 2     # only non-blank parts sent
        assert result.parts_sent == 2

    @pytest.mark.asyncio
    async def test_parts_list_capped_at_max_parts(self) -> None:
        """Sending more than MAX_PARTS must be capped — no runaway sends."""
        from app.integrations.whatsapp_client import MAX_PARTS

        client = self._make_client()
        sent   = []

        async def _mock_send_text(to: str, text: str, **kwargs) -> WhatsAppSendResult:
            sent.append(text)
            return WhatsAppSendResult(success=True, message_id=f"m{len(sent)}", to=to)

        huge_parts = [f"Part {i}" for i in range(MAX_PARTS + 20)]

        with (
            patch.object(client, "_ensure_initialised"),
            patch.object(client, "send_text_message", side_effect=_mock_send_text),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await client.send_multi_part(
                to="+919876543210",
                parts=huge_parts,
            )

        assert len(sent) <= MAX_PARTS