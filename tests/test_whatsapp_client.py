#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WhatsApp outbound calling support.

Covers:
- Pydantic union discrimination for connect / status / terminate webhooks
- WhatsAppClient.initiate_call (happy path + API failure cases)
- Outbound connect webhook handling (SDP answer, ongoing map, stored callback)
- Call status webhook handling (RINGING / ACCEPTED / REJECTED)
- terminate_all_calls clears both pending and ongoing calls
"""

import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# aiortc is an optional extra not installed in the base dev environment.
# Stub it out before importing anything that transitively pulls in
# pipecat.transports.smallwebrtc.connection.
for _mod in ("aiortc", "aiortc.rtcrtpreceiver", "av", "av.frame"):
    sys.modules.setdefault(_mod, MagicMock())

import aiohttp  # noqa: E402 — must come after sys.modules patch

from pipecat.transports.whatsapp.api import (  # noqa: E402
    WhatsAppCallStatusValue,
    WhatsAppConnectCallValue,
    WhatsAppTerminateCallValue,
    WhatsAppWebhookRequest,
)
from pipecat.transports.whatsapp.client import WhatsAppClient  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALL_ID = "call_abc123"
FROM = "15550000001"
TO = "15550000002"


def _webhook(value: dict) -> dict:
    """Wrap a change value in a minimal valid WhatsApp webhook envelope."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "waba_id",
                "changes": [{"value": value, "field": "calls"}],
            }
        ],
    }


def _base_value(calls: list) -> dict:
    return {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "+1111111111", "phone_number_id": "pid_123"},
        "calls": calls,
    }


def _connect_call(direction: str = "inbound", sdp: str = "sdp", sdp_type: str = "offer") -> dict:
    return {
        "id": CALL_ID,
        "from": FROM,
        "to": TO,
        "event": "connect",
        "timestamp": "2024-01-01T00:00:00Z",
        "direction": direction,
        "session": {"sdp": sdp, "sdp_type": sdp_type},
    }


def _status_call(status: str) -> dict:
    return {
        "id": CALL_ID,
        "from": FROM,
        "to": TO,
        "event": "status",
        "timestamp": "2024-01-01T00:00:00Z",
        "direction": "outbound",
        "status": status,
    }


def _terminate_call(status: str = "COMPLETED") -> dict:
    return {
        "id": CALL_ID,
        "from": FROM,
        "to": TO,
        "event": "terminate",
        "timestamp": "2024-01-01T00:00:00Z",
        "direction": "outbound",
        "status": status,
        "duration": 30,
    }


def _parse(payload: dict) -> WhatsAppWebhookRequest:
    return WhatsAppWebhookRequest.model_validate(payload)


def _make_client(call_status_callback=None) -> WhatsAppClient:
    session = MagicMock(spec=aiohttp.ClientSession)
    return WhatsAppClient(
        whatsapp_token="tok",
        phone_number_id="pid",
        session=session,
        call_status_callback=call_status_callback,
    )


def _make_conn_mock() -> MagicMock:
    """A minimal SmallWebRTCConnection mock."""
    mock = MagicMock()
    mock.create_offer = AsyncMock(return_value="fake_offer_sdp")
    mock.set_remote_answer = AsyncMock()
    mock.disconnect = AsyncMock()
    # event_handler is used as a decorator: @conn.event_handler("closed")
    mock.event_handler = MagicMock(return_value=lambda fn: fn)
    return mock


# ---------------------------------------------------------------------------
# Model discrimination tests
# ---------------------------------------------------------------------------


class TestWhatsAppModelDiscrimination(unittest.TestCase):
    """Pydantic union in WhatsAppChange.value routes to the correct type."""

    def test_inbound_connect_parses_as_connect_value(self):
        payload = _webhook(
            {
                **_base_value([_connect_call("inbound", "offer_sdp", "offer")]),
                "contacts": [{"profile": {"name": "User"}, "wa_id": FROM}],
            }
        )
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppConnectCallValue)

    def test_outbound_connect_without_contacts_parses_as_connect_value(self):
        """Outbound webhooks may omit contacts — should still parse correctly."""
        payload = _webhook(_base_value([_connect_call("outbound", "answer_sdp", "answer")]))
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppConnectCallValue)

    def test_ringing_parses_as_status_value(self):
        payload = _webhook(_base_value([_status_call("RINGING")]))
        req = _parse(payload)
        value = req.entry[0].changes[0].value
        self.assertIsInstance(value, WhatsAppCallStatusValue)
        self.assertEqual(value.calls[0].status, "RINGING")

    def test_accepted_parses_as_status_value(self):
        payload = _webhook(_base_value([_status_call("ACCEPTED")]))
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppCallStatusValue)

    def test_rejected_parses_as_status_value(self):
        payload = _webhook(_base_value([_status_call("REJECTED")]))
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppCallStatusValue)

    def test_terminate_parses_as_terminate_value(self):
        payload = _webhook(_base_value([_terminate_call("COMPLETED")]))
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppTerminateCallValue)

    def test_terminate_without_duration_parses_as_terminate_value(self):
        call = {**_terminate_call(), "duration": None, "status": "FAILED"}
        payload = _webhook(_base_value([call]))
        req = _parse(payload)
        self.assertIsInstance(req.entry[0].changes[0].value, WhatsAppTerminateCallValue)


# ---------------------------------------------------------------------------
# initiate_call tests
# ---------------------------------------------------------------------------


class TestInitiateCall(unittest.IsolatedAsyncioTestCase):
    @patch("pipecat.transports.whatsapp.client.SmallWebRTCConnection")
    async def test_success_returns_call_id(self, MockConn):
        client = _make_client()
        mock_conn = _make_conn_mock()
        MockConn.return_value = mock_conn
        client._whatsapp_api.create_call_to_whatsapp = AsyncMock(
            return_value={"success": True, "call_id": "call_xyz"}
        )

        call_id = await client.initiate_call(to=TO)

        self.assertEqual(call_id, "call_xyz")

    @patch("pipecat.transports.whatsapp.client.SmallWebRTCConnection")
    async def test_success_stores_pending_connection(self, MockConn):
        client = _make_client()
        mock_conn = _make_conn_mock()
        MockConn.return_value = mock_conn
        client._whatsapp_api.create_call_to_whatsapp = AsyncMock(
            return_value={"success": True, "call_id": "call_xyz"}
        )

        await client.initiate_call(to=TO)

        self.assertIn("call_xyz", client._pending_outbound_calls)

    @patch("pipecat.transports.whatsapp.client.SmallWebRTCConnection")
    async def test_success_stores_callback(self, MockConn):
        client = _make_client()
        mock_conn = _make_conn_mock()
        MockConn.return_value = mock_conn
        client._whatsapp_api.create_call_to_whatsapp = AsyncMock(
            return_value={"success": True, "call_id": "call_xyz"}
        )
        cb = AsyncMock()

        await client.initiate_call(to=TO, connection_callback=cb)

        self.assertIn("call_xyz", client._pending_outbound_callbacks)
        self.assertIs(client._pending_outbound_callbacks["call_xyz"], cb)

    @patch("pipecat.transports.whatsapp.client.SmallWebRTCConnection")
    async def test_api_failure_raises_and_disconnects(self, MockConn):
        client = _make_client()
        mock_conn = _make_conn_mock()
        MockConn.return_value = mock_conn
        client._whatsapp_api.create_call_to_whatsapp = AsyncMock(
            return_value={"success": False, "error": "permission denied"}
        )

        with self.assertRaises(Exception):
            await client.initiate_call(to=TO)

        mock_conn.disconnect.assert_awaited_once()
        self.assertEqual(len(client._pending_outbound_calls), 0)

    @patch("pipecat.transports.whatsapp.client.SmallWebRTCConnection")
    async def test_missing_call_id_raises_and_disconnects(self, MockConn):
        client = _make_client()
        mock_conn = _make_conn_mock()
        MockConn.return_value = mock_conn
        client._whatsapp_api.create_call_to_whatsapp = AsyncMock(
            return_value={"success": True}  # call_id absent
        )

        with self.assertRaises(Exception):
            await client.initiate_call(to=TO)

        mock_conn.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# Outbound connect webhook tests
# ---------------------------------------------------------------------------


class TestOutboundConnectWebhook(unittest.IsolatedAsyncioTestCase):
    def _make_request(self, sdp: str = "remote_answer_sdp") -> WhatsAppWebhookRequest:
        payload = _webhook(_base_value([_connect_call("outbound", sdp, "answer")]))
        return WhatsAppWebhookRequest.model_validate(payload)

    async def test_sets_remote_answer(self):
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._pending_outbound_calls[CALL_ID] = mock_conn

        await client.handle_webhook_request(self._make_request("the_answer_sdp"))

        mock_conn.set_remote_answer.assert_awaited_once_with(
            sdp="the_answer_sdp", type="answer"
        )

    async def test_moves_call_to_ongoing_map(self):
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._pending_outbound_calls[CALL_ID] = mock_conn

        await client.handle_webhook_request(self._make_request())

        self.assertNotIn(CALL_ID, client._pending_outbound_calls)
        self.assertIn(CALL_ID, client._ongoing_calls_map)

    async def test_invokes_stored_callback(self):
        cb = AsyncMock()
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._pending_outbound_calls[CALL_ID] = mock_conn
        client._pending_outbound_callbacks[CALL_ID] = cb

        await client.handle_webhook_request(self._make_request())

        cb.assert_awaited_once_with(mock_conn)

    async def test_callback_removed_from_pending_after_invocation(self):
        cb = AsyncMock()
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._pending_outbound_calls[CALL_ID] = mock_conn
        client._pending_outbound_callbacks[CALL_ID] = cb

        await client.handle_webhook_request(self._make_request())

        self.assertNotIn(CALL_ID, client._pending_outbound_callbacks)

    async def test_unknown_call_id_raises(self):
        client = _make_client()
        # No pending call registered

        with self.assertRaises(Exception):
            await client.handle_webhook_request(self._make_request())


# ---------------------------------------------------------------------------
# Call status webhook tests
# ---------------------------------------------------------------------------


class TestCallStatusWebhook(unittest.IsolatedAsyncioTestCase):
    def _make_request(self, status: str) -> WhatsAppWebhookRequest:
        payload = _webhook(_base_value([_status_call(status)]))
        return WhatsAppWebhookRequest.model_validate(payload)

    async def test_callback_invoked_with_call_id_and_status(self):
        cb = AsyncMock()
        client = _make_client(call_status_callback=cb)

        await client.handle_webhook_request(self._make_request("RINGING"))

        cb.assert_awaited_once_with(CALL_ID, "RINGING")

    async def test_no_callback_does_not_raise(self):
        client = _make_client()

        await client.handle_webhook_request(self._make_request("ACCEPTED"))  # must not raise

    async def test_all_statuses_invoke_callback(self):
        for status in ("RINGING", "ACCEPTED", "REJECTED"):
            with self.subTest(status=status):
                cb = AsyncMock()
                client = _make_client(call_status_callback=cb)
                await client.handle_webhook_request(self._make_request(status))
                cb.assert_awaited_once_with(CALL_ID, status)


# ---------------------------------------------------------------------------
# terminate_all_calls tests
# ---------------------------------------------------------------------------


class TestTerminateAllCalls(unittest.IsolatedAsyncioTestCase):
    async def test_clears_pending_outbound_calls(self):
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._pending_outbound_calls["pending_call"] = mock_conn
        client._pending_outbound_callbacks["pending_call"] = AsyncMock()

        await client.terminate_all_calls()

        mock_conn.disconnect.assert_awaited_once()
        self.assertEqual(len(client._pending_outbound_calls), 0)
        self.assertEqual(len(client._pending_outbound_callbacks), 0)

    async def test_clears_ongoing_calls(self):
        client = _make_client()
        mock_conn = _make_conn_mock()
        client._ongoing_calls_map["ongoing_call"] = mock_conn
        client._whatsapp_api.terminate_call_to_whatsapp = AsyncMock(
            return_value={"success": True}
        )

        await client.terminate_all_calls()

        mock_conn.disconnect.assert_awaited_once()
        self.assertEqual(len(client._ongoing_calls_map), 0)

    async def test_pending_and_ongoing_both_cleared(self):
        client = _make_client()
        pending_conn = _make_conn_mock()
        ongoing_conn = _make_conn_mock()
        client._pending_outbound_calls["pending_call"] = pending_conn
        client._ongoing_calls_map["ongoing_call"] = ongoing_conn
        client._whatsapp_api.terminate_call_to_whatsapp = AsyncMock(
            return_value={"success": True}
        )

        await client.terminate_all_calls()

        pending_conn.disconnect.assert_awaited_once()
        ongoing_conn.disconnect.assert_awaited_once()

    async def test_no_calls_does_not_raise(self):
        client = _make_client()

        await client.terminate_all_calls()  # must not raise


if __name__ == "__main__":
    unittest.main()
