# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Unit tests for custom email template tools and the _serialize helper.

These tests reproduce JIRA OKTA-1114952: EmailPreview serialization failure.

Root cause recap
----------------
``EmailPreview.to_dict()`` (the SDK-generated helper) explicitly excludes
``body`` and ``subject`` from its output because both are flagged as
server-readOnly in the OpenAPI spec.  Calling ``to_dict()`` therefore returns
``{}`` (or just ``{"_links": ...}``), silently dropping the preview content.

``model_dump(by_alias=True, exclude_none=True)`` does *not* apply this
exclusion and correctly returns ``body``, ``subject``, and ``_links``.

The ``_serialize`` helper in ``custom_templates.py`` must use ``model_dump``
and must never delegate to ``to_dict`` for preview objects.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from okta.models.email_preview import EmailPreview
from okta.models.email_preview_links import EmailPreviewLinks

from okta.models.email_template_response import EmailTemplateResponse

from okta_mcp_server.tools.customization.custom_templates.custom_templates import (
    _serialize,
    get_email_customization_preview,
    get_email_default_content_preview,
    list_email_customizations,
    list_email_templates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BRAND_ID = "brandABC123"
TEMPLATE_NAME = "UserActivation"
CUSTOMIZATION_ID = "custXYZ789"

_BODY = "<html><body>Activate: ${activationLink}</body></html>"
_SUBJECT = "Activate your Okta account"


def _make_preview(body=_BODY, subject=_SUBJECT, links=None) -> EmailPreview:
    """Return an ``EmailPreview`` as the SDK deserializes from the API response."""
    return EmailPreview(body=body, subject=subject, links=links)


def _make_preview_links() -> EmailPreviewLinks:
    return EmailPreviewLinks()


# ===========================================================================
# _serialize — unit tests (no I/O, no mocking)
# ===========================================================================

class TestSerializeEmailPreview:
    """Regression tests for the SDK v3 EmailPreview serialization bug."""

    def test_body_and_subject_are_included(self):
        """model_dump must return body and subject — to_dict() would silently drop them."""
        preview = _make_preview()
        result = _serialize(preview)
        assert result["body"] == _BODY
        assert result["subject"] == _SUBJECT

    def test_to_dict_exhibits_the_sdk_bug(self):
        """Document that to_dict() omits body and subject (SDK readOnly exclusion).

        This test exists to clearly demonstrate the SDK bug that motivated the
        fix.  If the SDK is ever patched to include these fields in to_dict(),
        this test will need to be updated.
        """
        preview = _make_preview()
        result = preview.to_dict()
        # to_dict() deliberately excludes readOnly fields — body and subject are absent
        assert "body" not in result
        assert "subject" not in result

    def test_serialize_does_not_lose_body_when_links_present(self):
        links = _make_preview_links()
        preview = _make_preview(links=links)
        result = _serialize(preview)
        assert result["body"] == _BODY
        assert result["subject"] == _SUBJECT

    def test_serialize_none_returns_none(self):
        assert _serialize(None) is None

    def test_serialize_none_with_fallback_returns_empty_dict(self):
        """The `or {}` in the tool ensures None serialization becomes {}."""
        assert (_serialize(None) or {}) == {}

    def test_serialize_list_of_previews(self):
        previews = [_make_preview(body=f"<b>{i}</b>", subject=f"Subject {i}") for i in range(3)]
        results = _serialize(previews)
        assert len(results) == 3
        for i, item in enumerate(results):
            assert item["body"] == f"<b>{i}</b>"
            assert item["subject"] == f"Subject {i}"

    def test_serialize_plain_pydantic_model(self):
        """model_dump is used for any Pydantic v2 model, not just EmailPreview."""
        links = _make_preview_links()
        result = _serialize(links)
        assert isinstance(result, dict)

    def test_serialize_non_pydantic_object_uses_dict_fallback(self):
        """Objects without model_dump fall back to __dict__ extraction."""
        class _FakeObj:
            def __init__(self):
                self.name = "test"
                self.value = 42
                self._private = "ignored"

        result = _serialize(_FakeObj())
        assert result == {"name": "test", "value": 42}

    def test_serialize_non_pydantic_object_excludes_none_values(self):
        class _FakeObj:
            def __init__(self):
                self.present = "yes"
                self.absent = None

        result = _serialize(_FakeObj())
        assert "present" in result
        assert "absent" not in result

    def test_serialize_scalar_passthrough(self):
        """Scalars (str, int) that have no model_dump and no __dict__ pass through."""
        assert _serialize("raw string") == "raw string"
        assert _serialize(42) == 42


# ===========================================================================
# get_email_default_content_preview — tool tests
# ===========================================================================

class TestGetEmailDefaultContentPreview:
    """Tests for the get_email_default_content_preview MCP tool."""

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_body_and_subject_from_sdk_preview(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Happy path: SDK returns an EmailPreview; tool must surface body and subject."""
        preview = _make_preview()
        client = AsyncMock()
        client.get_email_default_preview.return_value = (preview, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert result["body"] == _BODY
        assert result["subject"] == _SUBJECT

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_body_and_subject_with_language(
        self, mock_get_client, ctx_no_elicitation
    ):
        preview = _make_preview(subject="Activez votre compte", body="<b>FR</b>")
        client = AsyncMock()
        client.get_email_default_preview.return_value = (preview, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            language="fr",
        )

        assert result["body"] == "<b>FR</b>"
        assert result["subject"] == "Activez votre compte"

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_result_does_not_use_to_dict_serialization(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Verify the result is not the broken to_dict() output (which omits body/subject)."""
        preview = _make_preview()
        client = AsyncMock()
        client.get_email_default_preview.return_value = (preview, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        # to_dict() would produce {} (missing body/subject); we must not get that
        assert result != {}
        assert result != preview.to_dict()

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_empty_dict_when_sdk_returns_none_data(
        self, mock_get_client, ctx_no_elicitation
    ):
        """SDK may return (None, resp, None) for empty responses; tool returns {}."""
        client = AsyncMock()
        client.get_email_default_preview.return_value = (None, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert result == {}

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_on_api_error(
        self, mock_get_client, ctx_no_elicitation
    ):
        client = AsyncMock()
        client.get_email_default_preview.return_value = (None, MagicMock(), "403 Forbidden")
        mock_get_client.return_value = client

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert "error" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_on_exception(
        self, mock_get_client, ctx_no_elicitation
    ):
        mock_get_client.side_effect = Exception("connection refused")

        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_brand_id_returns_error_without_api_call(
        self, ctx_no_elicitation
    ):
        result = await get_email_default_content_preview(
            ctx=ctx_no_elicitation,
            brand_id="../../etc/passwd",
            template_name=TEMPLATE_NAME,
        )

        # @validate_ids returns a list with a single error string
        assert isinstance(result, list)
        assert any("brand_id" in str(item).lower() or "invalid" in str(item).lower() for item in result)


# ===========================================================================
# get_email_customization_preview — tool tests
# ===========================================================================

class TestGetEmailCustomizationPreview:
    """Tests for the get_email_customization_preview MCP tool."""

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_body_and_subject_from_sdk_preview(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Happy path: SDK returns an EmailPreview; tool must surface body and subject."""
        preview = _make_preview()
        client = AsyncMock()
        client.get_customization_preview.return_value = (preview, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        assert result["body"] == _BODY
        assert result["subject"] == _SUBJECT

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_result_is_not_to_dict_output(
        self, mock_get_client, ctx_no_elicitation
    ):
        """The result must not match the broken to_dict() output that omits body/subject."""
        preview = _make_preview()
        client = AsyncMock()
        client.get_customization_preview.return_value = (preview, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        assert result != {}
        assert "body" in result
        assert "subject" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_empty_dict_when_sdk_returns_none_data(
        self, mock_get_client, ctx_no_elicitation
    ):
        client = AsyncMock()
        client.get_customization_preview.return_value = (None, MagicMock(), None)
        mock_get_client.return_value = client

        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        assert result == {}

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_on_api_error(
        self, mock_get_client, ctx_no_elicitation
    ):
        client = AsyncMock()
        client.get_customization_preview.return_value = (None, MagicMock(), "404 Not Found")
        mock_get_client.return_value = client

        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        assert "error" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_on_exception(
        self, mock_get_client, ctx_no_elicitation
    ):
        mock_get_client.side_effect = Exception("timeout")

        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_brand_id_returns_error_without_api_call(
        self, ctx_no_elicitation
    ):
        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id="not valid!",
            template_name=TEMPLATE_NAME,
            customization_id=CUSTOMIZATION_ID,
        )

        # @validate_ids returns a list with a single error string
        assert isinstance(result, list)
        assert any("brand_id" in str(item).lower() or "invalid" in str(item).lower() for item in result)

    @pytest.mark.asyncio
    async def test_invalid_customization_id_returns_error_without_api_call(
        self, ctx_no_elicitation
    ):
        result = await get_email_customization_preview(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
            customization_id="../../etc/passwd",
        )

        # @validate_ids returns a list with a single error string
        assert isinstance(result, list)
        assert any("customization_id" in str(item).lower() or "invalid" in str(item).lower() for item in result)


# ===========================================================================
# list_email_templates — tool tests
# Reproduces: list_email_templates Returns Empty List for Valid Brand
#
# Root cause: when the Okta SDK returns (None, resp, None) — which it does
# whenever response_body == "" or response.status == 204 — the old code called
# len(templates) before the None guard, raising
# TypeError: object of type 'NoneType' has no len()
# That exception was caught and returned as {"error": "..."}, a dict.
# MCP's return-type validation then raised a second error because the declared
# return type is List[Dict[str, Any]], not Dict.
#
# Fix: compute the serialized result first, then log its length.
# ===========================================================================

def _make_template(name: str = "UserActivation") -> EmailTemplateResponse:
    """Return a minimal EmailTemplateResponse."""
    return EmailTemplateResponse(name=name)


class TestListEmailTemplates:
    """Tests for the list_email_templates MCP tool."""

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_list_of_templates(self, mock_get_client, ctx_no_elicitation):
        """Happy path: SDK returns a list of templates; tool returns serialized list."""
        templates = [_make_template("UserActivation"), _make_template("ForgotPassword")]
        client = AsyncMock()
        client.list_email_templates.return_value = (templates, MagicMock(), None)
        mock_get_client.return_value = client

        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "UserActivation"
        assert result[1]["name"] == "ForgotPassword"

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_sdk_returns_none_templates_yields_empty_list(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Regression: SDK may return (None, resp, None); tool must return [] not crash.

        Before the fix, len(None) raised TypeError which was caught and returned
        as {"error": "object of type 'NoneType' has no len()"}.  MCP then raised
        a validation error because the return type is List, not Dict.
        """
        client = AsyncMock()
        client.list_email_templates.return_value = (None, MagicMock(), None)
        mock_get_client.return_value = client

        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
        )

        assert result == []

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_dict_on_api_error(
        self, mock_get_client, ctx_no_elicitation
    ):
        """When the SDK surfaces an API error the tool returns {"error": ...}."""
        client = AsyncMock()
        client.list_email_templates.return_value = (None, MagicMock(), "403 Forbidden")
        mock_get_client.return_value = client

        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
        )

        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_dict_on_exception(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Unhandled exceptions are caught and returned as {"error": ...}."""
        mock_get_client.side_effect = Exception("connection refused")

        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
        )

        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_passes_expand_to_sdk(self, mock_get_client, ctx_no_elicitation):
        """The expand list is forwarded to the SDK call unchanged."""
        client = AsyncMock()
        client.list_email_templates.return_value = ([], MagicMock(), None)
        mock_get_client.return_value = client

        await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            expand=["settings", "customizationCount"],
        )

        client.list_email_templates.assert_called_once_with(
            BRAND_ID, expand=["settings", "customizationCount"]
        )

    @pytest.mark.asyncio
    async def test_invalid_brand_id_returns_error_without_api_call(
        self, ctx_no_elicitation
    ):
        """@validate_ids rejects path-traversal brand IDs before any SDK call."""
        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id="../../etc/passwd",
        )

        assert isinstance(result, list)
        assert any(
            "brand_id" in str(item).lower() or "invalid" in str(item).lower()
            for item in result
        )

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_empty_sdk_list_returns_empty_list(
        self, mock_get_client, ctx_no_elicitation
    ):
        """An empty list from the SDK is returned as-is (not converted to None or {})."""
        client = AsyncMock()
        client.list_email_templates.return_value = ([], MagicMock(), None)
        mock_get_client.return_value = client

        result = await list_email_templates(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
        )

        assert result == []


# ===========================================================================
# list_email_customizations — tool tests
# Same None-guard fix applied defensively.
# ===========================================================================

class TestListEmailCustomizations:
    """Tests for the list_email_customizations MCP tool."""

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_list_of_customizations(
        self, mock_get_client, ctx_no_elicitation
    ):
        """Happy path: SDK returns a list; tool returns serialized list."""
        from okta.models.email_customization import EmailCustomization

        cust = EmailCustomization(language="en", subject="Hello", body="<p>Hi</p>")
        client = AsyncMock()
        client.list_email_customizations.return_value = ([cust], MagicMock(), None)
        mock_get_client.return_value = client

        result = await list_email_customizations(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_sdk_returns_none_customizations_yields_empty_list(
        self, mock_get_client, ctx_no_elicitation
    ):
        """SDK returning (None, resp, None) must produce [] not a TypeError crash."""
        client = AsyncMock()
        client.list_email_customizations.return_value = (None, MagicMock(), None)
        mock_get_client.return_value = client

        result = await list_email_customizations(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert result == []

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_dict_on_api_error(
        self, mock_get_client, ctx_no_elicitation
    ):
        client = AsyncMock()
        client.list_email_customizations.return_value = (None, MagicMock(), "404 Not Found")
        mock_get_client.return_value = client

        result = await list_email_customizations(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    @patch(
        "okta_mcp_server.tools.customization.custom_templates.custom_templates.get_okta_client"
    )
    async def test_returns_error_dict_on_exception(
        self, mock_get_client, ctx_no_elicitation
    ):
        mock_get_client.side_effect = Exception("network failure")

        result = await list_email_customizations(
            ctx=ctx_no_elicitation,
            brand_id=BRAND_ID,
            template_name=TEMPLATE_NAME,
        )

        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_brand_id_returns_error_without_api_call(
        self, ctx_no_elicitation
    ):
        result = await list_email_customizations(
            ctx=ctx_no_elicitation,
            brand_id="not valid!",
            template_name=TEMPLATE_NAME,
        )

        assert isinstance(result, list)
        assert any(
            "brand_id" in str(item).lower() or "invalid" in str(item).lower()
            for item in result
        )
