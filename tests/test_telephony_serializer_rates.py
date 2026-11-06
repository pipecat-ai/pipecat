#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the wire sample rates of the telephony serializers."""

import audioop
import base64
import json
import unittest
import warnings

from pipecat.frames.frames import InterruptionFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.genesys import GenesysAudioHookSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vonage import VonageFrameSerializer
from pipecat.utils import deprecation
from tests.frame_processor_helpers import frame_processor_setup

# Each serializer, its deprecated wire rate field, and its default wire rate.
PROVIDERS = [
    (TwilioFrameSerializer, "twilio_sample_rate", 8000),
    (TelnyxFrameSerializer, "telnyx_sample_rate", 8000),
    (PlivoFrameSerializer, "plivo_sample_rate", 8000),
    (ExotelFrameSerializer, "exotel_sample_rate", 8000),
    (GenesysAudioHookSerializer, "genesys_sample_rate", 8000),
    (VonageFrameSerializer, "vonage_sample_rate", 16000),
]


def deprecations_while(build):
    """What `build` returns, and the deprecation warnings it raised."""
    deprecation._warned_sites.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build()
    return result, [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


class TestWireRates(unittest.TestCase):
    def test_each_serializer_states_its_wire_rate(self):
        for cls, _, rate in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                params = cls.InputParams()
                self.assertEqual(params.audio_in_sample_rate, rate)
                self.assertEqual(params.audio_out_sample_rate, rate)

    def test_a_serializer_without_a_fixed_rate_states_none(self):
        """A wire format that carries the rate leaves the transport's rate alone."""
        params = FrameSerializer.InputParams()
        self.assertIsNone(params.audio_in_sample_rate)
        self.assertIsNone(params.audio_out_sample_rate)


class TestDeprecatedRateFields(unittest.TestCase):
    def test_the_provider_rate_sets_both_wire_rates(self):
        for cls, field, _ in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                params, messages = deprecations_while(
                    lambda cls=cls, field=field: cls.InputParams(**{field: 16000})
                )

                self.assertEqual(params.audio_in_sample_rate, 16000)
                self.assertEqual(params.audio_out_sample_rate, 16000)
                self.assertTrue([m for m in messages if f".InputParams.{field}`" in m])

    def test_a_wire_rate_given_alongside_wins(self):
        params, _ = deprecations_while(
            lambda: TwilioFrameSerializer.InputParams(
                twilio_sample_rate=16000, audio_out_sample_rate=8000
            )
        )

        self.assertEqual(params.audio_in_sample_rate, 16000)
        self.assertEqual(params.audio_out_sample_rate, 8000)

    def test_sample_rate_warns(self):
        for cls, _, _ in PROVIDERS:
            with self.subTest(serializer=cls.__name__):
                _, messages = deprecations_while(lambda cls=cls: cls.InputParams(sample_rate=16000))

                self.assertTrue([m for m in messages if ".InputParams.sample_rate`" in m])

    def test_resampler_clear_after_secs_warns(self):
        for cls in (FrameSerializer, TwilioFrameSerializer):
            with self.subTest(params=cls.__name__):
                _, messages = deprecations_while(
                    lambda cls=cls: cls.InputParams(resampler_clear_after_secs=0.5)
                )

                self.assertTrue([m for m in messages if "resampler_clear_after_secs" in m])


class TestTwilioWireAudio(unittest.IsolatedAsyncioTestCase):
    """Twilio stands in for the serializers whose base class converts their audio."""

    CHUNKS = 60

    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            "stream-sid", params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(frame_processor_setup(audio_in_sample_rate=16000))

    async def _payload_bytes(self, frame: OutputAudioRawFrame) -> int:
        """μ-law bytes sent for `CHUNKS` copies of `frame`: one per 8 kHz sample."""
        total = 0
        for _ in range(self.CHUNKS):
            message = await self.serializer.serialize(frame)
            if message:
                total += len(base64.b64decode(json.loads(message)["media"]["payload"]))
        return total

    async def test_audio_at_the_wire_rate_is_encoded_as_is(self):
        frame = OutputAudioRawFrame(audio=b"\x00\x10" * 160, sample_rate=8000, num_channels=1)

        self.assertEqual(await self._payload_bytes(frame), self.CHUNKS * 160)

    async def test_audio_at_another_rate_is_converted_to_the_wire_rate(self):
        frame = OutputAudioRawFrame(audio=b"\x00\x10" * 480, sample_rate=24000, num_channels=1)

        sent = await self._payload_bytes(frame)

        # 480 samples at 24 kHz are 160 at 8 kHz, less what the resampler still holds.
        self.assertGreater(sent, 0.9 * self.CHUNKS * 160)
        self.assertLessEqual(sent, self.CHUNKS * 160)

    async def test_incoming_audio_is_converted_to_the_pipeline_rate(self):
        payload = base64.b64encode(b"\x7f" * 160).decode()
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        frames = [await self.serializer.deserialize(message) for _ in range(self.CHUNKS)]
        frames = [frame for frame in frames if frame is not None]

        self.assertTrue(frames)
        self.assertEqual({frame.sample_rate for frame in frames}, {16000})
        received = sum(len(frame.audio) // 2 for frame in frames)
        self.assertGreater(received, 0.9 * self.CHUNKS * 320)
        self.assertLessEqual(received, self.CHUNKS * 320)

    async def test_an_interruption_discards_the_cut_off_audio(self):
        loud = OutputAudioRawFrame(audio=b"\x00\x40" * 480, sample_rate=24000, num_channels=1)
        await self._payload_bytes(loud)

        await self.serializer.serialize(InterruptionFrame())

        silence = OutputAudioRawFrame(audio=b"\x00\x00" * 480, sample_rate=24000, num_channels=1)
        message = None
        while not message:
            message = await self.serializer.serialize(silence)
        decoded = audioop.ulaw2lin(base64.b64decode(json.loads(message)["media"]["payload"]), 2)
        self.assertLess(audioop.max(decoded, 2), 100)
