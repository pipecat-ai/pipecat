#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from unittest.mock import MagicMock

from pipecat.runner.utils import parse_telephony_websocket


class MockAsyncIterator:
    """Mock async iterator for WebSocket messages."""

    def __init__(self, messages):
        self.messages = messages
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.messages):
            raise StopAsyncIteration
        message = self.messages[self.index]
        self.index += 1
        return message


class TestParseTelephonyWebSocket(unittest.IsolatedAsyncioTestCase):
    async def test_no_messages_raises_value_error(self):
        """Test that no messages raises ValueError."""
        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([])

        with self.assertRaises(ValueError) as context:
            await parse_telephony_websocket(mock_websocket)

        self.assertIn("WebSocket closed before receiving", str(context.exception))

    async def test_one_message_logs_warning_and_continues(self):
        """Test that one message logs warning but continues processing."""
        twilio_message = json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ123",
                    "callSid": "CA123",
                    "customParameters": {"user_id": "test_user"},
                },
            }
        )

        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([twilio_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "twilio")
        self.assertEqual(call_data["stream_id"], "MZ123")
        self.assertEqual(call_data["call_id"], "CA123")

    async def test_two_messages_normal_operation(self):
        """Test normal operation with two messages."""
        first_message = json.dumps({"event": "connected"})
        twilio_message = json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ456",
                    "callSid": "CA456",
                    "customParameters": {},
                },
            }
        )

        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([first_message, twilio_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "twilio")
        self.assertEqual(call_data["stream_id"], "MZ456")
        self.assertEqual(call_data["call_id"], "CA456")

    async def test_telnyx_detection(self):
        """Test Telnyx provider detection."""
        telnyx_message = json.dumps(
            {
                "stream_id": "stream_123",
                "start": {
                    "call_control_id": "cc_123",
                    "media_format": {"encoding": "PCMU"},
                    "from": "+15551234567",
                    "to": "+15559876543",
                },
            }
        )

        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([telnyx_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "telnyx")
        self.assertEqual(call_data["stream_id"], "stream_123")
        self.assertEqual(call_data["call_control_id"], "cc_123")

    async def test_plivo_detection(self):
        """Test Plivo provider detection."""
        plivo_message = json.dumps(
            {"start": {"streamId": "stream_plivo_123", "callId": "call_plivo_123"}}
        )

        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([plivo_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "plivo")
        self.assertEqual(call_data["stream_id"], "stream_plivo_123")
        self.assertEqual(call_data["call_id"], "call_plivo_123")

    async def test_exotel_detection(self):
        """Test Exotel provider detection."""
        exotel_message = json.dumps(
            {
                "event": "start",
                "start": {
                    "stream_sid": "stream_exo_123",
                    "call_sid": "call_exo_123",
                    "account_sid": "acc_123",
                    "from": "+15551111111",
                    "to": "+15552222222",
                },
            }
        )

        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([exotel_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "exotel")
        self.assertEqual(call_data["stream_id"], "stream_exo_123")
        self.assertEqual(call_data["call_id"], "call_exo_123")
        self.assertEqual(call_data["account_sid"], "acc_123")


if __name__ == "__main__":
    unittest.main()
