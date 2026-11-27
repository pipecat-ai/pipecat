#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `WhatsAppClient.handle_webhook_request`.

WhatsApp may batch several entries, changes and calls into a single webhook
delivery, and it also sends call events the client does not act on. These
tests check that every connect/terminate event in a delivery is processed,
that unsupported events are acknowledged instead of raising (which would make
WhatsApp retry the delivery), and that the HMAC signature check is enforced.

All phone numbers, ids and SDPs below are synthetic.
"""

import hashlib
import hmac
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aiortc")

from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest  # noqa: E402
from pipecat.transports.whatsapp.client import WhatsAppClient  # noqa: E402

APP_SECRET = "test-app-secret"
METADATA = {"display_phone_number": "15550000000", "phone_number_id": "100000000000000"}


def _connect_call(call_id: str, caller: str) -> dict:
    return {
        "id": call_id,
        "from": caller,
        "to": METADATA["display_phone_number"],
        "event": "connect",
        "timestamp": "1700000000",
        "direction": "USER_INITIATED",
        "session": {"sdp": f"v=0\r\no=- {call_id} offer\r\n", "sdp_type": "offer"},
    }


def _terminate_call(call_id: str, caller: str) -> dict:
    return {
        "id": call_id,
        "from": caller,
        "to": METADATA["display_phone_number"],
        "event": "terminate",
        "timestamp": "1700000060",
        "direction": "USER_INITIATED",
        "status": "COMPLETED",
        "duration": 60,
    }


def _status_call(call_id: str, caller: str) -> dict:
    return {
        "id": call_id,
        "from": caller,
        "to": METADATA["display_phone_number"],
        "event": "status",
        "timestamp": "1700000010",
        "direction": "BUSINESS_INITIATED",
        "status": "RINGING",
    }


def _connect_change(*calls: dict) -> dict:
    return {
        "field": "calls",
        "value": {
            "messaging_product": "whatsapp",
            "metadata": METADATA,
            "contacts": [
                {"profile": {"name": "Test User"}, "wa_id": call["from"]} for call in calls
            ],
            "calls": list(calls),
        },
    }


def _call_change(*calls: dict) -> dict:
    return {
        "field": "calls",
        "value": {"messaging_product": "whatsapp", "metadata": METADATA, "calls": list(calls)},
    }


def _body(*entries: list[dict]) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {"id": f"20000000000000{i}", "changes": changes} for i, changes in enumerate(entries)
        ],
    }


def _sign(raw_body: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def _fake_connection(*_args, **_kwargs) -> MagicMock:
    connection = MagicMock()
    connection.initialize = AsyncMock()
    connection.get_answer = MagicMock(return_value={"sdp": "v=0\r\n", "type": "answer"})
    connection.disconnect = AsyncMock()
    connection.event_handler = MagicMock(return_value=lambda handler: handler)
    return connection


class TestWhatsAppWebhook(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        api_patcher = patch("pipecat.transports.whatsapp.client.WhatsAppApi")
        connection_patcher = patch(
            "pipecat.transports.whatsapp.client.SmallWebRTCConnection",
            side_effect=_fake_connection,
        )
        self.api = api_patcher.start().return_value
        self.api.answer_call_to_whatsapp = AsyncMock(return_value={"success": True})
        self.connection_cls = connection_patcher.start()
        self.addCleanup(api_patcher.stop)
        self.addCleanup(connection_patcher.stop)

        self.callback = AsyncMock()

    def _client(self, secret: str | None = None) -> WhatsAppClient:
        return WhatsAppClient(
            whatsapp_token="test-token",
            phone_number_id=METADATA["phone_number_id"],
            session=MagicMock(),
            whatsapp_secret=secret,
        )

    async def _handle(self, client: WhatsAppClient, body: dict, **kwargs):
        request = WhatsAppWebhookRequest(**body)
        return await client.handle_webhook_request(request, self.callback, **kwargs)

    def _callback_call_ids(self) -> list[str]:
        return [c.args[1].id for c in self.callback.await_args_list]

    async def test_batch_connect_and_terminate_in_one_entry(self):
        """A connect followed by a terminate in the same delivery are both handled."""
        client = self._client()
        existing = _fake_connection()
        client._ongoing_calls_map["wacid.OLD"] = existing

        body = _body(
            [
                _connect_change(_connect_call("wacid.NEW", "15551110001")),
                _call_change(_terminate_call("wacid.OLD", "15551110002")),
            ]
        )
        result = await self._handle(client, body)

        self.assertTrue(result)
        self.assertEqual(self._callback_call_ids(), ["wacid.NEW"])
        existing.disconnect.assert_awaited_once()
        self.assertEqual(list(client._ongoing_calls_map), ["wacid.NEW"])

    async def test_multiple_entries_are_all_processed(self):
        """Connect events spread over several entries all get a connection."""
        client = self._client()
        body = _body(
            [_connect_change(_connect_call("wacid.A", "15551110001"))],
            [_connect_change(_connect_call("wacid.B", "15551110002"))],
        )
        result = await self._handle(client, body)

        self.assertTrue(result)
        self.assertEqual(self._callback_call_ids(), ["wacid.A", "wacid.B"])
        self.assertEqual(set(client._ongoing_calls_map), {"wacid.A", "wacid.B"})
        # pre_accept + accept for each call
        self.assertEqual(self.api.answer_call_to_whatsapp.await_count, 4)

    async def test_multiple_calls_in_one_change_are_all_processed(self):
        """Several terminate calls inside a single change are all handled."""
        client = self._client()
        first, second = _fake_connection(), _fake_connection()
        client._ongoing_calls_map.update({"wacid.A": first, "wacid.B": second})

        body = _body(
            [
                _call_change(
                    _terminate_call("wacid.A", "15551110001"),
                    _terminate_call("wacid.B", "15551110002"),
                )
            ]
        )
        result = await self._handle(client, body)

        self.assertTrue(result)
        first.disconnect.assert_awaited_once()
        second.disconnect.assert_awaited_once()
        self.assertEqual(client._ongoing_calls_map, {})

    async def test_unsupported_event_is_ignored(self):
        """An event the client does not handle is acknowledged, not raised."""
        client = self._client()
        body = _body([_call_change(_status_call("wacid.A", "15551110001"))])

        result = await self._handle(client, body)

        self.assertFalse(result)
        self.callback.assert_not_awaited()
        self.connection_cls.assert_not_called()

    async def test_unsupported_event_does_not_block_supported_ones(self):
        """An unsupported event before a terminate does not stop the terminate."""
        client = self._client()
        existing = _fake_connection()
        client._ongoing_calls_map["wacid.A"] = existing

        body = _body(
            [
                _call_change(_status_call("wacid.B", "15551110002")),
                _call_change(_terminate_call("wacid.A", "15551110001")),
            ]
        )
        result = await self._handle(client, body)

        self.assertTrue(result)
        existing.disconnect.assert_awaited_once()

    async def test_valid_signature_is_accepted(self):
        client = self._client(secret=APP_SECRET)
        body = _body([_connect_change(_connect_call("wacid.A", "15551110001"))])
        raw_body = json.dumps(body).encode()

        result = await self._handle(
            client, body, raw_body=raw_body, sha256_signature=_sign(raw_body)
        )

        self.assertTrue(result)
        self.assertEqual(self._callback_call_ids(), ["wacid.A"])

    async def test_invalid_signature_is_rejected(self):
        client = self._client(secret=APP_SECRET)
        body = _body([_connect_change(_connect_call("wacid.A", "15551110001"))])
        raw_body = json.dumps(body).encode()

        with self.assertRaisesRegex(Exception, "Invalid webhook signature"):
            await self._handle(
                client, body, raw_body=raw_body, sha256_signature=_sign(raw_body, "wrong")
            )

        self.connection_cls.assert_not_called()
        self.callback.assert_not_awaited()

    async def test_missing_signature_is_rejected(self):
        client = self._client(secret=APP_SECRET)
        body = _body([_connect_change(_connect_call("wacid.A", "15551110001"))])
        raw_body = json.dumps(body).encode()

        with self.assertRaisesRegex(Exception, "Missing X-Hub-Signature-256"):
            await self._handle(client, body, raw_body=raw_body, sha256_signature=None)

        self.connection_cls.assert_not_called()
        self.callback.assert_not_awaited()

    async def test_signature_not_checked_without_secret(self):
        client = self._client(secret=None)
        body = _body([_connect_change(_connect_call("wacid.A", "15551110001"))])

        result = await self._handle(client, body)

        self.assertTrue(result)
        self.assertEqual(self._callback_call_ids(), ["wacid.A"])


if __name__ == "__main__":
    unittest.main()
