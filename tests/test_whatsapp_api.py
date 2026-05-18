#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WhatsApp webhook payload parsing.

Meta's webhook payloads vary in shape — most notably the terminate-side
error sub-objects, which routinely omit ``href`` and ``error_data`` and use
``title`` instead of ``message``. These tests pin the schema-tolerance
contract so the parser keeps accepting the payloads Meta actually sends.
"""

import pytest
from pydantic import ValidationError

from pipecat.transports.whatsapp.api import (
    WhatsAppError,
    WhatsAppWebhookRequest,
)


def test_minimal_meta_terminate_error_parses():
    """Code + title only (Meta's ``no media for a long time`` shape) must parse."""
    err = WhatsAppError.model_validate({"code": 138021, "title": "No media for a long time."})
    assert err.code == 138021
    assert err.title == "No media for a long time."
    assert err.message is None
    assert err.href is None
    assert err.error_data is None


def test_full_payload_still_parses():
    """Backwards-compatible with payloads carrying every documented field."""
    err = WhatsAppError.model_validate(
        {
            "code": 131051,
            "message": "Unsupported message type",
            "href": "https://developers.facebook.com/docs/whatsapp/...",
            "error_data": {"details": "Audio codec not supported"},
        }
    )
    assert err.code == 131051
    assert err.message == "Unsupported message type"
    assert err.href is not None and err.href.startswith("https://")
    assert err.error_data is not None and err.error_data["details"]


def test_unknown_extra_fields_ignored():
    """Future Meta-added fields don't break parsing (extra='ignore')."""
    err = WhatsAppError.model_validate(
        {"code": 1, "title": "x", "future_field_added_by_meta": "ignored"}
    )
    assert err.code == 1
    assert err.title == "x"


def test_missing_code_still_rejected():
    """``code`` stays required even after the other fields are relaxed."""
    with pytest.raises(ValidationError):
        WhatsAppError.model_validate({"title": "no code provided"})


def test_real_world_terminate_webhook_with_no_media_error():
    """End-to-end webhook parse for the real Meta terminate-with-error shape."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "+15551234567",
                                "phone_number_id": "PHONE_NUMBER_ID",
                            },
                            "calls": [
                                {
                                    "id": "wacid.EXAMPLE",
                                    "from": "+15559876543",
                                    "to": "+15551234567",
                                    "event": "terminate",
                                    "timestamp": "2026-05-05T18:16:42Z",
                                    "direction": "inbound",
                                    "status": "FAILED",
                                }
                            ],
                            "errors": [{"code": 138021, "title": "No media for a long time."}],
                        },
                    }
                ],
            }
        ],
    }
    req = WhatsAppWebhookRequest.model_validate(payload)
    change = req.entry[0].changes[0]
    # Narrow the union type for the type-checker — terminate has `errors`
    value = change.value
    assert hasattr(value, "errors") and value.errors is not None
    assert value.errors[0].code == 138021
    assert value.errors[0].title == "No media for a long time."
    assert value.errors[0].href is None
