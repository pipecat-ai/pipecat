#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.runner.types import (
    CallData,
    ExotelCallData,
    TelnyxCallData,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import (
    _maybe_apply_daily_dialin,
    create_transport,
    parse_telephony_websocket,
)

try:
    import daily  # noqa: F401

    DAILY_AVAILABLE = True
except ImportError:
    DAILY_AVAILABLE = False


class TestCallData(unittest.TestCase):
    """CallData gives typed attribute access while staying dict-compatible."""

    def test_attribute_access_with_alias(self):
        cd = CallData.model_validate(
            {"stream_id": "MZ1", "call_id": "CA1", "from": "+1555", "to": "+1666"}
        )
        # Wire keys "from"/"to" map onto from_number/to_number (no keyword clash).
        self.assertEqual(cd.from_number, "+1555")
        self.assertEqual(cd.to_number, "+1666")
        self.assertEqual(cd.call_id, "CA1")

    def test_dict_compat_access(self):
        cd = CallData.model_validate({"stream_id": "MZ1", "from": "+1555"})
        # Subscript / get / in use the original wire keys.
        self.assertEqual(cd["from"], "+1555")
        self.assertEqual(cd["stream_id"], "MZ1")
        self.assertEqual(cd.get("from"), "+1555")
        self.assertEqual(cd.get("to", "n/a"), "n/a")  # unset -> default
        self.assertIn("from", cd)
        self.assertNotIn("to", cd)  # unset fields aren't "in"

    def test_unset_fields_are_none(self):
        cd = CallData.model_validate({"call_id": "CA1"})
        self.assertIsNone(cd.to_number)
        self.assertEqual(cd.body, {})

    def test_extra_provider_keys_preserved(self):
        cd = CallData.model_validate({"call_id": "CA1", "weird_provider_field": "x"})
        # extra="allow": unmodeled keys remain reachable via dict access.
        self.assertEqual(cd["weird_provider_field"], "x")

    def test_call_data_is_a_base_field(self):
        """call_data lives on the base, so any runner_args exposes it (defaults None),
        letting bots read runner_args.call_data uniformly without getattr guards."""
        from pipecat.runner.types import DailyRunnerArguments, RunnerArguments

        self.assertIsNone(RunnerArguments().call_data)
        self.assertIsNone(DailyRunnerArguments(room_url="https://example.daily.co/room").call_data)


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

    async def test_twilio_promotes_from_to_custom_params(self):
        """Twilio from/to TwiML stream params are promoted for a uniform typed API."""
        twilio_message = json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ789",
                    "callSid": "CA789",
                    "customParameters": {"from_number": "+1555", "to_number": "+1666"},
                },
            }
        )
        mock_websocket = MagicMock()
        mock_websocket.iter_text.return_value = MockAsyncIterator([twilio_message])

        transport_type, call_data = await parse_telephony_websocket(mock_websocket)

        self.assertEqual(transport_type, "twilio")
        self.assertEqual(call_data.from_number, "+1555")
        self.assertEqual(call_data.to_number, "+1666")
        # Raw custom params still available under body.
        self.assertEqual(call_data.body["to_number"], "+1666")

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
        self.assertIsInstance(call_data, TelnyxCallData)
        self.assertEqual(call_data["stream_id"], "stream_123")
        # Telnyx's call_control_id is normalized onto the common call_id field.
        self.assertEqual(call_data["call_id"], "cc_123")
        # Provider-specific field is typed on the subclass.
        self.assertEqual(call_data.outbound_encoding, "PCMU")

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
        self.assertIsInstance(call_data, ExotelCallData)
        self.assertEqual(call_data["stream_id"], "stream_exo_123")
        self.assertEqual(call_data["call_id"], "call_exo_123")
        self.assertEqual(call_data["account_sid"], "acc_123")


class TestParseTelephonyIdempotent(unittest.IsolatedAsyncioTestCase):
    async def test_second_call_returns_cached_parse(self):
        """The single-use stream is consumed once; a second call returns the cache."""
        twilio_message = json.dumps(
            {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}}
        )
        ws = MagicMock()
        # One iterator, exhausted after the first parse. A second *real* parse would
        # raise ValueError ("closed before receiving"); the cache must prevent that.
        ws.iter_text.return_value = MockAsyncIterator([twilio_message])

        first = await parse_telephony_websocket(ws)
        second = await parse_telephony_websocket(ws)

        self.assertEqual(first, second)
        self.assertEqual(first[0], "twilio")
        self.assertEqual(first[1]["call_id"], "CA1")


class TestCreateTransportExposesCallData(unittest.IsolatedAsyncioTestCase):
    async def test_telephony_call_data_set_on_runner_args(self):
        """create_transport exposes the parsed handshake on runner_args for the bot."""
        twilio_message = json.dumps(
            {"event": "start", "start": {"streamSid": "MZ9", "callSid": "CA9"}}
        )
        ws = MagicMock()
        ws.iter_text.return_value = MockAsyncIterator([twilio_message])
        args = WebSocketRunnerArguments(websocket=ws)  # transport_type=None -> telephony

        sentinel = object()
        with patch(
            "pipecat.runner.utils._create_telephony_transport",
            new=AsyncMock(return_value=sentinel),
        ):
            result = await create_transport(args, {"twilio": lambda: MagicMock()})

        self.assertIs(result, sentinel)
        self.assertEqual(args.transport_type, "twilio")
        self.assertIsNotNone(args.call_data)
        # Both styles work: typed attribute access and dict-style subscript.
        self.assertEqual(args.call_data.call_id, "CA9")
        self.assertEqual(args.call_data["call_id"], "CA9")


@unittest.skipUnless(DAILY_AVAILABLE, "requires the daily-python SDK")
class TestMaybeApplyDailyDialin(unittest.IsolatedAsyncioTestCase):
    def _params(self):
        from pipecat.transports.daily.transport import DailyParams

        return DailyParams()

    def test_dialin_body_populates_params(self):
        params = self._params()
        body = {
            "dialin_settings": {"call_id": "c1", "call_domain": "d1"},
            "daily_api_key": "key123",
            "daily_api_url": "https://example.test/v1",
        }
        _maybe_apply_daily_dialin(params, body)

        self.assertIsNotNone(params.dialin_settings)
        self.assertEqual(params.dialin_settings.call_id, "c1")
        self.assertEqual(params.dialin_settings.call_domain, "d1")
        self.assertEqual(params.api_key, "key123")
        self.assertEqual(params.api_url, "https://example.test/v1")

    def test_noop_for_non_dialin_body(self):
        for body in (None, {}, {"something": "else"}):
            params = self._params()
            _maybe_apply_daily_dialin(params, body)
            self.assertIsNone(params.dialin_settings)


if __name__ == "__main__":
    unittest.main()
