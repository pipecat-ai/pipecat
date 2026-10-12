#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for JambonzFrameSerializer wire format."""

import asyncio
import json
import unittest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    TTSStartedFrame,
)
from pipecat.serializers.jambonz import JambonzFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

# The preamble jambonz sends before any audio, with identifiers anonymized.
PREAMBLE = json.dumps(
    {
        "sampleRate": 16000,
        "mixType": "mono",
        "callSid": "3e5d869a-0000-0000-0000-000000000000",
        "direction": "inbound",
        "from": "alice",
        "to": "1234",
        "callId": "94682bcc-0000-0000-0000-000000000000",
        "sipStatus": 200,
        "sipReason": "OK",
        "callStatus": "in-progress",
        "accountSid": "00000000-0000-0000-0000-000000000000",
        "traceId": "00000000000000000000000000000000",
    }
)

JAMBONZ_RATE = 16000
PIPELINE_RATE = 24000  # deliberately different, so resampling has to happen
FRAME_BYTES = 640  # 20 ms of 16-bit mono at 16 kHz, the size jambonz sends

# The stream resampler buffers and flushes in bursts rather than returning one
# chunk per chunk, so these tests drive a run of frames and assert on the total.
PUMP = 20


def audio(num_bytes: int) -> bytes:
    return b"\x01\x02" * (num_bytes // 2)


class TestJambonzDeserialize(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = JambonzFrameSerializer()
        await self.serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await self.serializer.deserialize(PREAMBLE)

    async def test_sample_rate_is_read_from_the_preamble(self):
        self.assertEqual(self.serializer.jambonz_sample_rate, JAMBONZ_RATE)

    async def test_call_identifiers_are_kept(self):
        self.assertEqual(self.serializer.call_sid, "3e5d869a-0000-0000-0000-000000000000")
        self.assertEqual(self.serializer.trace_id, "00000000000000000000000000000000")

    async def test_audio_becomes_input_frames_at_the_pipeline_rate(self):
        frames = [await self.serializer.deserialize(audio(FRAME_BYTES)) for _ in range(PUMP)]
        received = [f for f in frames if f is not None]

        self.assertTrue(received)
        for frame in received:
            self.assertIsInstance(frame, InputAudioRawFrame)
            self.assertEqual(frame.sample_rate, PIPELINE_RATE)
            self.assertEqual(frame.num_channels, 1)

        ideal = PUMP * FRAME_BYTES * PIPELINE_RATE // JAMBONZ_RATE
        got = sum(len(f.audio) for f in received)
        self.assertGreaterEqual(got, 0.7 * ideal)
        self.assertLessEqual(got, 1.05 * ideal)

    async def test_an_undecodable_text_frame_is_ignored(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))
        self.assertEqual(self.serializer.jambonz_sample_rate, JAMBONZ_RATE)


class TestJambonzPreambleHandling(unittest.IsolatedAsyncioTestCase):
    async def test_a_narrowband_call_is_followed(self):
        """8 kHz is valid for narrowband SIP and must need no configuration."""
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await serializer.deserialize(json.dumps({**json.loads(PREAMBLE), "sampleRate": 8000}))

        self.assertEqual(serializer.jambonz_sample_rate, 8000)

    async def test_audio_before_the_preamble_is_dropped(self):
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))

        self.assertIsNone(await serializer.deserialize(audio(FRAME_BYTES)))

    async def test_a_configured_rate_is_not_overridden_by_the_preamble(self):
        serializer = JambonzFrameSerializer(
            params=JambonzFrameSerializer.InputParams(jambonz_sample_rate=8000)
        )
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await serializer.deserialize(PREAMBLE)

        self.assertEqual(serializer.jambonz_sample_rate, 8000)

    async def test_a_text_frame_that_is_not_an_object_is_ignored(self):
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))

        self.assertIsNone(await serializer.deserialize("[1, 2, 3]"))
        self.assertIsNone(serializer.jambonz_sample_rate)

    async def test_the_whole_preamble_is_available(self):
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await serializer.deserialize(PREAMBLE)

        self.assertEqual(serializer.call_meta["sipReason"], "OK")
        # A copy, so a caller cannot edit the serializer's own record of the call.
        serializer.call_meta["sipReason"] = "edited"
        self.assertEqual(serializer.call_meta["sipReason"], "OK")

    async def test_outgoing_audio_before_the_preamble_is_dropped(self):
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))

        frame = OutputAudioRawFrame(audio=audio(960), sample_rate=PIPELINE_RATE, num_channels=1)
        self.assertIsNone(await serializer.serialize(frame))


class TestJambonzSerializeWireFormat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = JambonzFrameSerializer()
        await self.serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await self.serializer.deserialize(PREAMBLE)

    def bot_audio(self) -> OutputAudioRawFrame:
        # 960 bytes is 20 ms at the pipeline rate.
        return OutputAudioRawFrame(audio=audio(960), sample_rate=PIPELINE_RATE, num_channels=1)

    async def test_audio_is_sent_as_raw_bytes(self):
        sent = [await self.serializer.serialize(self.bot_audio()) for _ in range(PUMP)]
        written = [s for s in sent if s is not None]

        self.assertTrue(written)
        for chunk in written:
            self.assertIsInstance(chunk, bytes)

        ideal = PUMP * 960 * JAMBONZ_RATE // PIPELINE_RATE
        got = sum(len(c) for c in written)
        self.assertGreaterEqual(got, 0.7 * ideal)
        self.assertLessEqual(got, 1.05 * ideal)

    async def test_interruption_becomes_kill_audio(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"type": "killAudio"})

    async def test_transport_messages_are_forwarded_as_json(self):
        frame = OutputTransportMessageFrame(message={"type": "custom", "value": 1})

        self.assertEqual(
            json.loads(await self.serializer.serialize(frame)),
            {"type": "custom", "value": 1},
        )

    async def test_rtvi_messages_are_not_forwarded(self):
        """RTVI is a browser protocol; jambonz would read it as a command."""
        frame = OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "bot-ready"})

        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_an_unhandled_frame_produces_nothing(self):
        self.assertIsNone(await self.serializer.serialize(TTSStartedFrame()))

    async def test_audio_is_withheld_briefly_after_kill_audio(self):
        """Message ordering on the socket is not guaranteed, so audio written
        immediately after killAudio can arrive after the flush and play over
        the user. Every frame in the window has to be withheld, not just the
        first, because the pipeline keeps producing the interrupted turn."""
        await self.serializer.serialize(InterruptionFrame())

        inside = [await self.serializer.serialize(self.bot_audio()) for _ in range(5)]
        self.assertTrue(all(s is None for s in inside))

        await asyncio.sleep(0.12)

        after = [await self.serializer.serialize(self.bot_audio()) for _ in range(PUMP)]
        self.assertTrue([s for s in after if s is not None])
