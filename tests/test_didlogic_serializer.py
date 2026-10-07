#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DidlogicFrameSerializer wire format."""

import base64
import json
import unittest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.didlogic import DidlogicFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

SAMPLE_RATE = 24000
CALL_ID = "01K5ZQJ3M8N7P0R2T4V6W8X9Y0"


def start_message(**overrides) -> str:
    message = {
        "event": "start",
        "call_id": CALL_ID,
        "from": "442071234567",
        "to": "447700900000",
        "codec": "pcm16",
        "sample_rate": SAMPLE_RATE,
        "frame_bytes": 480,
        "ptime": 10,
    }
    message.update(overrides)
    return json.dumps(message)


class DidlogicSerializerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = DidlogicFrameSerializer()
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )


class TestSerialize(DidlogicSerializerTest):
    async def test_audio_becomes_a_base64_media_event(self):
        audio = b"\x01\x02" * 240
        message = json.loads(
            await self.serializer.serialize(
                OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)
            )
        )
        self.assertEqual(message["event"], "media")
        self.assertEqual(base64.b64decode(message["payload"]), audio)

    async def test_interruption_becomes_clear(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"event": "clear"})

    async def test_end_becomes_hangup(self):
        for frame in (EndFrame(), CancelFrame()):
            message = json.loads(await self.serializer.serialize(frame))
            self.assertEqual(message, {"event": "hangup"})

    async def test_auto_hang_up_off_sends_nothing(self):
        serializer = DidlogicFrameSerializer(
            DidlogicFrameSerializer.InputParams(auto_hang_up=False)
        )
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=SAMPLE_RATE))
        self.assertIsNone(await serializer.serialize(EndFrame()))

    async def test_a_transport_message_is_sent_as_given(self):
        frame = OutputTransportMessageUrgentFrame(message={"event": "clear"})
        self.assertEqual(json.loads(await self.serializer.serialize(frame)), {"event": "clear"})


class TestStartHandshake(DidlogicSerializerTest):
    async def test_start_is_recorded_rather_than_forwarded(self):
        self.assertIsNone(await self.serializer.deserialize(start_message()))
        self.assertEqual(self.serializer.call_id, CALL_ID)

    async def test_start_records_the_call(self):
        await self.serializer.deserialize(start_message(direction="outbound"))
        self.assertEqual(self.serializer.call_id, CALL_ID)
        self.assertEqual(self.serializer.from_number, "442071234567")
        self.assertEqual(self.serializer.to_number, "447700900000")
        self.assertTrue(self.serializer.is_outbound)

    async def test_an_inbound_call_is_not_outbound(self):
        await self.serializer.deserialize(start_message())
        self.assertFalse(self.serializer.is_outbound)

    async def test_the_call_may_be_supplied_when_start_was_read_elsewhere(self):
        serializer = DidlogicFrameSerializer(
            call_id=CALL_ID,
            from_number="442071234567",
            to_number="447700900000",
            direction="outbound",
        )
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=SAMPLE_RATE))

        self.assertEqual(serializer.call_id, CALL_ID)
        self.assertEqual(serializer.from_number, "442071234567")
        self.assertEqual(serializer.to_number, "447700900000")
        self.assertTrue(serializer.is_outbound)

    async def test_the_wire_rate_from_start_wins_over_the_default(self):
        await self.serializer.deserialize(start_message(sample_rate=8000))
        # 8000 Hz on the wire against a 24000 Hz pipeline, so the audio has to be
        # upsampled rather than passed through. The stream resampler buffers, so
        # this asserts that it grew, not by how much.
        audio = b"\x00\x00" * 1600
        frame = await self.serializer.deserialize(
            json.dumps({"event": "media", "payload": base64.b64encode(audio).decode()})
        )
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)
        self.assertGreater(len(frame.audio), len(audio))


class TestDeserialize(DidlogicSerializerTest):
    async def test_media_round_trips(self):
        audio = b"\x03\x04" * 240
        outbound = await self.serializer.serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)
        )
        frame = await self.serializer.deserialize(outbound)
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, audio)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)
        self.assertEqual(frame.num_channels, 1)

    async def test_answered_is_surfaced_and_recorded(self):
        await self.serializer.deserialize(start_message(direction="outbound"))
        self.assertFalse(self.serializer.answered)

        frame = await self.serializer.deserialize(
            json.dumps({"event": "answered", "call_id": CALL_ID})
        )

        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["event"], "answered")
        self.assertTrue(self.serializer.answered)

    async def test_stop_is_surfaced(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "stop", "call_id": CALL_ID}))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["event"], "stop")

    async def test_dtmf_becomes_a_keypad_entry(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "dtmf", "digit": "5"}))
        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.FIVE)

    async def test_an_unusable_dtmf_digit_is_dropped(self):
        self.assertIsNone(
            await self.serializer.deserialize(json.dumps({"event": "dtmf", "digit": "Z"}))
        )
        self.assertIsNone(await self.serializer.deserialize(json.dumps({"event": "dtmf"})))

    async def test_an_unknown_event_is_ignored(self):
        self.assertIsNone(
            await self.serializer.deserialize(json.dumps({"event": "something-new", "x": 1}))
        )

    async def test_malformed_input_is_dropped(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))
        self.assertIsNone(await self.serializer.deserialize(json.dumps(["not", "an", "object"])))
        self.assertIsNone(
            await self.serializer.deserialize(json.dumps({"event": "media", "payload": "!!!"}))
        )

    async def test_a_text_frame_delivered_as_bytes_still_parses(self):
        await self.serializer.deserialize(start_message().encode("utf-8"))
        self.assertEqual(self.serializer.call_id, CALL_ID)
