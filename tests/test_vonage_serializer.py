#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VonageFrameSerializer.

deserialize() ignores malformed JSON and messages with unexpected shapes rather
than raising, since the transport's receive loop ends the call on any
exception. serialize() passes OutputTransportMessageFrame through to the wire
whatever the type of its message.
"""

import json
import unittest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import InputDTMFFrame, OutputTransportMessageFrame
from pipecat.serializers.vonage import VonageFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

SAMPLE_RATE = 16000


class TestVonageDeserializeRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = VonageFrameSerializer()
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_malformed_json_is_ignored_not_raised(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))

    async def test_non_object_json_is_ignored_not_raised(self):
        for name, data in (
            ("list", "[]"),
            ("string", '"str"'),
            ("number", "123"),
            ("null", "null"),
        ):
            with self.subTest(name=name):
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_dtmf_event_with_malformed_dtmf_is_ignored_not_raised(self):
        for name, dtmf in (
            ("null", None),
            ("string", "5"),
            ("missing_digit", {}),
            ("null_digit", {"digit": None}),
            ("unknown_digit", {"digit": "x"}),
        ):
            with self.subTest(name=name):
                data = json.dumps({"event": "websocket:dtmf", "dtmf": dtmf})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_valid_dtmf_events_are_still_deserialized(self):
        for name, message in (
            ("top_level_digit", {"event": "websocket:dtmf", "digit": "5"}),
            ("nested_digit", {"event": "websocket:dtmf", "dtmf": {"digit": "5"}}),
        ):
            with self.subTest(name=name):
                frame = await self.serializer.deserialize(json.dumps(message))
                self.assertIsInstance(frame, InputDTMFFrame)
                self.assertEqual(frame.button, KeypadEntry.FIVE)

    async def test_other_events_are_ignored(self):
        data = json.dumps({"event": "websocket:connected", "content-type": "audio/l16;rate=16000"})
        self.assertIsNone(await self.serializer.deserialize(data))


class TestVonageSerializeTransportMessages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = VonageFrameSerializer()
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_non_dict_transport_message_does_not_raise(self):
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=["a"]))
        self.assertEqual(json.loads(result), ["a"])

    async def test_rtvi_transport_message_is_ignored(self):
        message = {"label": "rtvi-ai", "type": "bot-ready"}
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=message))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
