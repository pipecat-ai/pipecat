#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`pipecat.transports.websocket.rtvi_client.RTVIClientTransport`.

These cover the RTVI-specific bits added on top of the WebSocket client transport:
the ``client-ready`` handshake message and ``bot-ready`` detection. The full
connect/receive path is exercised end-to-end by the eval integration tests.
"""

import json
import unittest
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport


class TestRTVIClientTransportHandshake(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transport = RTVIClientTransport("ws://localhost:0")

    def test_defaults_to_rtvi_client_serializer(self):
        self.assertIsInstance(self.transport._params.serializer, RTVIClientSerializer)

    async def test_send_client_ready_message(self):
        self.transport._session.send = AsyncMock()
        await self.transport._send_client_ready()

        self.transport._session.send.assert_awaited_once()
        sent = json.loads(self.transport._session.send.call_args.args[0])
        self.assertEqual(sent["label"], RTVI.MESSAGE_LABEL)
        self.assertEqual(sent["type"], "client-ready")
        self.assertEqual(sent["data"]["version"], RTVI.PROTOCOL_VERSION)
        self.assertIn("library", sent["data"]["about"])

    def test_is_bot_ready(self):
        bot_ready = json.dumps(
            {
                "label": RTVI.MESSAGE_LABEL,
                "type": "bot-ready",
                "id": "1",
                "data": {"version": "2.0.0"},
            }
        )
        not_ready = json.dumps({"label": RTVI.MESSAGE_LABEL, "type": "bot-llm-text"})
        no_label = json.dumps({"type": "bot-ready"})

        self.assertTrue(RTVIClientTransport._is_bot_ready(bot_ready))
        self.assertFalse(RTVIClientTransport._is_bot_ready(not_ready))
        self.assertFalse(RTVIClientTransport._is_bot_ready(no_label))
        self.assertFalse(RTVIClientTransport._is_bot_ready("not json"))

    async def test_bot_ready_flag_set_on_message(self):
        self.assertFalse(self.transport.bot_ready)
        msg = json.dumps(
            {
                "label": RTVI.MESSAGE_LABEL,
                "type": "bot-ready",
                "id": "1",
                "data": {"version": "2.0.0"},
            }
        )
        await self.transport._on_message(None, msg)
        self.assertTrue(self.transport.bot_ready)


if __name__ == "__main__":
    unittest.main()
