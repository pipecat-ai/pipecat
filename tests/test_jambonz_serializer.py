#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for JambonzFrameSerializer wire format."""

import json
import unittest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
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


def preamble_with(**overrides) -> str:
    """The preamble with fields replaced, for the cases jambonz can negotiate."""
    return json.dumps({**json.loads(PREAMBLE), **overrides})


class TestJambonzDisconnect(unittest.IsolatedAsyncioTestCase):
    """Ending the pipeline should end the call.

    `disconnect` ends the `listen` verb, after which jambonz runs the verb's
    actionHook or hangs up. This is the jambonz equivalent of Twilio's
    auto_hang_up, and it needs no API credentials.
    """

    async def _serializer(self, **params):
        serializer = JambonzFrameSerializer(JambonzFrameSerializer.InputParams(**params))
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await serializer.deserialize(PREAMBLE)
        return serializer

    async def test_the_pipeline_ending_ends_the_call(self):
        for frame in (EndFrame(), CancelFrame()):
            with self.subTest(frame=type(frame).__name__):
                serializer = await self._serializer()

                sent = await serializer.serialize(frame)

                self.assertEqual(json.loads(sent), {"type": "disconnect"})

    async def test_disconnect_is_sent_once(self):
        """Both frames can arrive on one shutdown, and the second would be
        written to a socket jambonz has already closed."""
        serializer = await self._serializer()

        self.assertIsNotNone(await serializer.serialize(EndFrame()))
        self.assertIsNone(await serializer.serialize(CancelFrame()))
        self.assertIsNone(await serializer.serialize(EndFrame()))

    async def test_disconnect_can_be_turned_off(self):
        """For an application that ends the verb itself, or wants the call to
        outlive the pipeline."""
        serializer = await self._serializer(disconnect_on_end=False)

        self.assertIsNone(await serializer.serialize(EndFrame()))
        self.assertIsNone(await serializer.serialize(CancelFrame()))


class TestJambonzTextFrames(unittest.IsolatedAsyncioTestCase):
    """The socket is not binary-only after the preamble.

    With `passDtmf` set, jambonz sends digits as text frames, and it answers
    `mark` and `killAudio` the same way. Routing all of those through the
    preamble handler replaced `call_meta` and set `call_sid` and `trace_id` to
    None, and lost the digit.
    """

    async def asyncSetUp(self):
        self.serializer = JambonzFrameSerializer()
        await self.serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await self.serializer.deserialize(PREAMBLE)
        self.call_sid = self.serializer.call_sid
        self.trace_id = self.serializer.trace_id

    def assert_call_survived(self):
        self.assertEqual(self.serializer.call_sid, self.call_sid)
        self.assertEqual(self.serializer.trace_id, self.trace_id)
        self.assertEqual(self.serializer.jambonz_sample_rate, JAMBONZ_RATE)

    async def test_a_dtmf_digit_becomes_a_frame_and_keeps_the_call(self):
        frame = await self.serializer.deserialize(
            json.dumps({"event": "dtmf", "dtmf": "5", "duration": "1600"})
        )

        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.FIVE)
        self.assert_call_survived()

    async def test_every_keypad_entry_is_recognised(self):
        for digit in "0123456789*#":
            with self.subTest(digit=digit):
                frame = await self.serializer.deserialize(
                    json.dumps({"event": "dtmf", "dtmf": digit})
                )
                self.assertIsInstance(frame, InputDTMFFrame)
                self.assertEqual(frame.button.value, digit)

    async def test_a_digit_that_is_not_on_a_keypad_is_ignored(self):
        """Dropping it beats raising inside the transport's read loop."""
        self.assertIsNone(
            await self.serializer.deserialize(json.dumps({"event": "dtmf", "dtmf": "A"}))
        )
        self.assert_call_survived()

    async def test_mark_and_play_done_replies_are_ignored(self):
        for event in (
            {"type": "mark", "data": {"name": "my-mark-1", "event": "playout"}},
            {"type": "mark", "data": {"name": "my-mark-1", "event": "cleared"}},
            {"type": "playDone"},
        ):
            with self.subTest(event=event):
                self.assertIsNone(await self.serializer.deserialize(json.dumps(event)))
                self.assert_call_survived()

    async def test_only_the_first_text_frame_is_the_preamble(self):
        """A second preamble-shaped frame is an event, not a renegotiation."""
        await self.serializer.deserialize(
            preamble_with(sampleRate=8000, callSid="other", traceId="other")
        )

        self.assert_call_survived()

    async def test_an_undecodable_event_is_ignored(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))
        self.assert_call_survived()

    async def test_audio_still_arrives_after_an_event(self):
        """The regression that mattered: a digit left the rate unusable, so
        every audio frame after it was dropped."""
        await self.serializer.deserialize(json.dumps({"event": "dtmf", "dtmf": "1"}))

        frames = [await self.serializer.deserialize(audio(FRAME_BYTES)) for _ in range(PUMP)]
        self.assertTrue([f for f in frames if f is not None])


class TestJambonzMixType(unittest.IsolatedAsyncioTestCase):
    """Only mono is interpretable; the other two layouts carry two channels."""

    async def _serializer(self, mix_type):
        serializer = JambonzFrameSerializer()
        await serializer.setup(frame_processor_setup(audio_in_sample_rate=PIPELINE_RATE))
        await serializer.deserialize(preamble_with(mixType=mix_type))
        return serializer

    async def test_a_mono_call_is_interpreted(self):
        serializer = await self._serializer("mono")
        self.assertEqual(serializer.mix_type, "mono")

        frames = [await serializer.deserialize(audio(FRAME_BYTES)) for _ in range(PUMP)]
        self.assertTrue([f for f in frames if f is not None])

    async def test_stereo_is_dropped_rather_than_misread(self):
        """Two interleaved channels read as mono play at double speed with the
        channels alternating, which is worse than no audio at all."""
        serializer = await self._serializer("stereo")
        self.assertEqual(serializer.mix_type, "stereo")

        frames = [await serializer.deserialize(audio(FRAME_BYTES)) for _ in range(PUMP)]
        self.assertEqual([f for f in frames if f is not None], [])

    async def test_mixed_is_dropped_even_though_it_would_decode(self):
        """`mixed` is a single channel, so it would be interpreted correctly.
        It is refused because it is both legs summed, so the bot would be fed
        its own audio -- a different reason from stereo, and the one that is
        easy to get wrong."""
        serializer = await self._serializer("mixed")
        self.assertEqual(serializer.mix_type, "mixed")

        frames = [await serializer.deserialize(audio(FRAME_BYTES)) for _ in range(PUMP)]
        self.assertEqual([f for f in frames if f is not None], [])


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

    async def test_audio_keeps_flowing_after_kill_audio(self):
        """jambonz reads the socket on one loop, so frames sent after killAudio
        are queued rather than racing the flush, and Pipecat's transport drops
        its own queued output on an interruption. Withholding audio here would
        add latency to every barge-in recovery for no reason."""
        await self.serializer.serialize(InterruptionFrame())

        after = [await self.serializer.serialize(self.bot_audio()) for _ in range(PUMP)]
        self.assertTrue([s for s in after if s is not None])
