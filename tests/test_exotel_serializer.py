#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ExotelFrameSerializer wire format."""

import base64
import json
import unittest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
)
from pipecat.serializers.exotel import ExotelFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_SID = "stream123"
SAMPLE_RATE = 8000
VALID_PAYLOAD = base64.b64encode(b"\x01\x02" * 160).decode()


class TestExotelSerializeWireFormat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = ExotelFrameSerializer(stream_sid=STREAM_SID, call_sid="call123")
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_media_event(self):
        frame = OutputAudioRawFrame(
            audio=b"\x00\x00" * 160, sample_rate=SAMPLE_RATE, num_channels=1
        )
        message = json.loads(await self.serializer.serialize(frame))
        self.assertEqual(message["event"], "media")
        self.assertEqual(message["stream_sid"], STREAM_SID)
        self.assertIn("payload", message["media"])

    async def test_clear_event(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"event": "clear", "stream_sid": STREAM_SID})

    async def test_non_dict_transport_message_does_not_raise(self):
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=["a"]))
        self.assertEqual(json.loads(result), ["a"])

    async def test_rtvi_transport_message_is_ignored(self):
        message = {"label": "rtvi-ai", "type": "bot-ready"}
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=message))
        self.assertIsNone(result)


class TestExotelDeserialize(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = ExotelFrameSerializer(stream_sid=STREAM_SID)
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_media_round_trip(self):
        audio = b"\x01\x02" * 160
        outbound = await self.serializer.serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)
        )
        frame = await self.serializer.deserialize(outbound)
        self.assertEqual(frame.audio, audio)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)

    async def test_malformed_json_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize("not json")
        self.assertIsNone(frame)

    async def test_media_event_missing_payload_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "media", "media": {}}))
        self.assertIsNone(frame)

    async def test_non_object_json_is_ignored_not_raised(self):
        for name, data in (
            ("list", "[]"),
            ("string", '"str"'),
            ("number", "123"),
            ("null", "null"),
        ):
            with self.subTest(name=name):
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_non_utf8_bytes_are_ignored_not_raised(self):
        self.assertIsNone(await self.serializer.deserialize(b"\xc3\x28"))

    async def test_media_event_with_malformed_media_is_ignored_not_raised(self):
        for name, media in (("null", None), ("string", "oops"), ("list", ["oops"])):
            with self.subTest(name=name):
                data = json.dumps({"event": "media", "media": media})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_media_event_with_undecodable_payload_is_ignored_not_raised(self):
        for name, payload in (
            ("invalid_base64", "!!!not-base64"),
            ("bad_padding", "AAAAA"),
            ("non_ascii", "\u00e9\u00e9\u00e9\u00e9"),
            ("number", 123),
            ("list", ["AAAA"]),
        ):
            with self.subTest(name=name):
                data = json.dumps({"event": "media", "media": {"payload": payload}})
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
                data = json.dumps({"event": "dtmf", "dtmf": dtmf})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_valid_media_event_is_still_deserialized(self):
        data = json.dumps({"event": "media", "media": {"payload": VALID_PAYLOAD}})
        frame = await self.serializer.deserialize(data)
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)

    async def test_valid_dtmf_event_is_still_deserialized(self):
        data = json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})
        frame = await self.serializer.deserialize(data)
        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.FIVE)


if __name__ == "__main__":
    unittest.main()
