#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how frame serializers handle transport message payloads."""

import json
import unittest

from pipecat.frames.frames import (
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer


class TestTransportMessagePayloads(unittest.IsolatedAsyncioTestCase):
    def _serializers(self):
        return [
            TwilioFrameSerializer(
                stream_sid="s", params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
            ),
            PlivoFrameSerializer(
                stream_id="s", params=PlivoFrameSerializer.InputParams(auto_hang_up=False)
            ),
            TelnyxFrameSerializer(
                stream_id="s",
                outbound_encoding="PCMU",
                inbound_encoding="PCMU",
                params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),
            ),
            ExotelFrameSerializer(stream_sid="s"),
        ]

    async def test_non_dict_message_is_serialized(self):
        for serializer in self._serializers():
            for frame_class in (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame):
                for message in ("hello", [1, 2]):
                    with self.subTest(serializer=type(serializer).__name__, message=message):
                        result = await serializer.serialize(frame_class(message=message))
                        self.assertEqual(result, json.dumps(message))

    async def test_dict_message_is_serialized(self):
        for serializer in self._serializers():
            with self.subTest(serializer=type(serializer).__name__):
                message = {"label": "custom", "type": "ping"}
                result = await serializer.serialize(OutputTransportMessageFrame(message=message))
                self.assertEqual(result, json.dumps(message))

    async def test_rtvi_message_is_ignored(self):
        for serializer in self._serializers():
            with self.subTest(serializer=type(serializer).__name__):
                frame = OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "ping"})
                self.assertIsNone(await serializer.serialize(frame))
