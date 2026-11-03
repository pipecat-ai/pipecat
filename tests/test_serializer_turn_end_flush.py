#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A speech run that ends normally must flush the serializer's output resampler.

Without it the filter tail is either discarded (clipping the end of the turn) or
carried over and played at the start of the next turn, seconds later.
"""

import audioop
import base64
import json
import struct
import unittest

from pipecat.frames.frames import AudioRawFrame, TTSStoppedFrame
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vonage import VonageFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_SID = "stream123"
OUT_RATE = 24000


def tone(ms: int, rate: int = OUT_RATE, amp: int = 12000) -> bytes:
    n = int(rate * ms / 1000)
    return b"".join(struct.pack("<h", amp if (i // 24) % 2 else -amp) for i in range(n))


def silence(ms: int, rate: int = OUT_RATE) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def _serializers():
    return [
        TwilioFrameSerializer(
            stream_sid=STREAM_SID,
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        ),
        PlivoFrameSerializer(
            stream_id=STREAM_SID,
            params=PlivoFrameSerializer.InputParams(auto_hang_up=False),
        ),
        TelnyxFrameSerializer(
            stream_id=STREAM_SID,
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
            params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),
        ),
        VonageFrameSerializer(),
        ExotelFrameSerializer(stream_sid=STREAM_SID),
    ]


class TestTurnEndFlush(unittest.IsolatedAsyncioTestCase):
    async def _say(self, serializer, payload):
        for i in range(0, len(payload), 960):
            await serializer.serialize(
                AudioRawFrame(audio=payload[i : i + 960], sample_rate=OUT_RATE, num_channels=1)
            )

    async def test_turn_end_emits_the_tail(self):
        """Every telephony serializer emits held audio when the speech run ends."""
        for serializer in _serializers():
            with self.subTest(serializer=type(serializer).__name__):
                await serializer.setup(
                    frame_processor_setup(audio_in_sample_rate=16000, audio_out_sample_rate=OUT_RATE)
                )
                await self._say(serializer, tone(200))

                result = await serializer.serialize(TTSStoppedFrame())
                self.assertIsNotNone(result, "no tail emitted at the end of the turn")

                # Nothing is left behind, so the next turn starts from an empty filter.
                self.assertEqual(await serializer._output_resampler.flush(), b"")

    async def test_turn_end_without_audio_emits_nothing(self):
        """A turn that produced no audio has no tail to flush."""
        for serializer in _serializers():
            with self.subTest(serializer=type(serializer).__name__):
                await serializer.setup(
                    frame_processor_setup(audio_in_sample_rate=16000, audio_out_sample_rate=OUT_RATE)
                )
                self.assertIsNone(await serializer.serialize(TTSStoppedFrame()))

    async def test_tail_does_not_leak_into_the_next_turn(self):
        """With the filter never cleared on gaps, the previous turn must not bleed in."""
        serializer = TwilioFrameSerializer(
            stream_sid=STREAM_SID,
            params=TwilioFrameSerializer.InputParams(
                auto_hang_up=False, resampler_clear_after_secs=None
            ),
        )
        await serializer.setup(
            frame_processor_setup(audio_in_sample_rate=16000, audio_out_sample_rate=OUT_RATE)
        )

        await self._say(serializer, tone(200))
        tail = await serializer.serialize(TTSStoppedFrame())
        self.assertIsNotNone(tail)

        # The next turn is digital silence, so anything audible came from the last one.
        out = b""
        for i in range(0, len(silence(200)), 960):
            result = await serializer.serialize(
                AudioRawFrame(audio=silence(200)[i : i + 960], sample_rate=OUT_RATE, num_channels=1)
            )
            if result:
                out += base64.b64decode(json.loads(result)["media"]["payload"])

        peak = audioop.max(audioop.ulaw2lin(out, 2), 2)
        self.assertLess(peak, 500, f"previous turn leaked into this one (peak {peak})")


if __name__ == "__main__":
    unittest.main()
