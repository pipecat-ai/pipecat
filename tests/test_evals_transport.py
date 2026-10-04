#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval transport: per-connection query flags and user audio rates."""

import asyncio
import types
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from pipecat.evals.client_transport import FRAME_S
from pipecat.evals.transport import (
    CAPTURE_AUDIO_QUERY_PARAM,
    CAPTURE_IMAGES_QUERY_PARAM,
    SKIP_TTS_QUERY_PARAM,
    EvalInputTransport,
    EvalTransport,
    EvalTransportParams,
    _query_flag,
)
from pipecat.frames.frames import InputAudioRawFrame, LLMConfigureOutputFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.server import SingleClientWebsocketServerInputTransport


def _ws(path=None, request_path=None):
    """A minimal stand-in for a websockets connection object."""
    request = types.SimpleNamespace(path=request_path) if request_path is not None else None
    return types.SimpleNamespace(path=path, request=request)


class TestQueryFlag(unittest.TestCase):
    def test_true_via_legacy_path(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=true"), SKIP_TTS_QUERY_PARAM))

    def test_true_via_request_path(self):
        self.assertTrue(
            _query_flag(_ws(path=None, request_path="/?skip_tts=1"), SKIP_TTS_QUERY_PARAM)
        )

    def test_accepts_yes_and_mixed_case(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=YES"), SKIP_TTS_QUERY_PARAM))

    def test_capture_audio_flag(self):
        self.assertTrue(
            _query_flag(_ws(path="/?capture_bot_audio=true"), CAPTURE_AUDIO_QUERY_PARAM)
        )
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=true"), CAPTURE_AUDIO_QUERY_PARAM))

    def test_capture_images_flag(self):
        self.assertTrue(
            _query_flag(_ws(path="/?capture_bot_images=true"), CAPTURE_IMAGES_QUERY_PARAM)
        )
        self.assertFalse(
            _query_flag(_ws(path="/?capture_bot_audio=true"), CAPTURE_IMAGES_QUERY_PARAM)
        )

    def test_false_when_absent(self):
        self.assertFalse(_query_flag(_ws(path="/"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_falsey_value(self):
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=false"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_no_path_at_all(self):
        self.assertFalse(_query_flag(_ws(), SKIP_TTS_QUERY_PARAM))


class TestConnectionOutputSettings(unittest.IsolatedAsyncioTestCase):
    async def test_audio_connection_reenables_tts_without_query_flag(self):
        await self._assert_tts_resets_between_connections("/")

    async def test_audio_connection_reenables_tts_with_false_query_flag(self):
        await self._assert_tts_resets_between_connections("/?skip_tts=false")

    async def _assert_tts_resets_between_connections(self, audio_path: str):
        transport = EvalTransport(params=EvalTransportParams())
        self.addAsyncCleanup(transport.cleanup)
        input_transport = transport.input()
        push_frame = AsyncMock()
        input_transport.push_frame = push_frame
        transport.output().set_client_connection = AsyncMock()
        greeting_settings = asyncio.Queue()

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, websocket):
            settings = [
                call.args[0].skip_tts
                for call in push_frame.await_args_list
                if isinstance(call.args[0], LLMConfigureOutputFrame)
            ]
            greeting_settings.put_nowait(settings[-1])

        observed = []
        for path in ("/?skip_tts=true", audio_path, "/?skip_tts=true"):
            await transport._on_client_connected(_ws(path=path))
            observed.append(await asyncio.wait_for(greeting_settings.get(), timeout=1))

        # Each greeting must see its own session's output setting.
        self.assertEqual(observed, [True, False, True])


class TestUserAudioRate(unittest.IsolatedAsyncioTestCase):
    """User audio sent at a scenario's rate must reach the pipeline at the bot's.

    The bug: 24 kHz scenario audio arrived labeled as the bot's 16 kHz input, so
    VAD, turn detection and STT saw it stretched by 24/16 -- a 0.42 s utterance
    arrived as 0.62 s with the pitch an octave and a half down.
    """

    BOT_RATE, SCENARIO_RATE, TONE_HZ = 16000, 24000, 440.0

    async def _input(self) -> EvalInputTransport:
        """An eval input transport set up as a 16 kHz bot's, with audio input on."""
        transport = EvalTransport(params=EvalTransportParams(audio_in_enabled=True))
        self.addAsyncCleanup(transport.cleanup)
        input_transport = transport.input()
        input_transport._sample_rate = self.BOT_RATE
        return input_transport

    async def _pushed(
        self, frames: list[InputAudioRawFrame], input_transport: EvalInputTransport | None = None
    ) -> list[InputAudioRawFrame]:
        """Feed frames in as the wire does, and return what the audio path got."""
        input_transport = input_transport or await self._input()
        with patch.object(
            SingleClientWebsocketServerInputTransport, "push_audio_frame", new_callable=AsyncMock
        ) as base:
            for frame in frames:
                await input_transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        return [call.args[0] for call in base.await_args_list]

    def _frames(self, rate: int, secs: float = 1.0, channels: int = 1) -> list[InputAudioRawFrame]:
        """The scenario's audio as the harness paces it to the bot: 40 ms frames."""
        samples = np.arange(int(rate * secs)) / rate
        pcm = (8000 * np.sin(2 * np.pi * self.TONE_HZ * samples)).astype(np.int16).tobytes()
        frame_bytes = int(rate * FRAME_S) * 2
        return [
            InputAudioRawFrame(
                audio=pcm[i : i + frame_bytes] * channels, sample_rate=rate, num_channels=channels
            )
            for i in range(0, len(pcm), frame_bytes)
        ]

    def _peak_hz(self, audio: bytes, skip_secs: float = 0.1) -> float:
        """The loudest frequency in the audio, past the resampler's start-up."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)[
            int(skip_secs * self.BOT_RATE) :
        ]
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        return float(np.argmax(spectrum)) * self.BOT_RATE / len(samples)

    async def test_audio_at_another_rate_is_resampled(self):
        pushed = await self._pushed(self._frames(self.SCENARIO_RATE))

        self.assertTrue(all(f.sample_rate == self.BOT_RATE for f in pushed))
        self.assertTrue(all(f.num_channels == 1 for f in pushed))
        # One second of audio, not 1.5 of it: the held filter tail accounts for
        # less than the frame the harness next sends.
        audio = b"".join(f.audio for f in pushed)
        self.assertAlmostEqual(len(audio) / 2 / self.BOT_RATE, 1.0, delta=FRAME_S)
        self.assertAlmostEqual(self._peak_hz(audio), self.TONE_HZ, delta=5)

    async def test_audio_at_the_transport_rate_is_untouched(self):
        frames = self._frames(self.BOT_RATE)

        pushed = await self._pushed(frames)

        self.assertEqual(pushed, frames)

    async def test_audio_still_arrives_after_the_scenario_rate_changes(self):
        """A kept-alive server serves scenarios speaking at different rates."""
        input_transport = await self._input()

        for rate in (self.SCENARIO_RATE, 8000, self.SCENARIO_RATE):
            with self.subTest(rate=rate):
                pushed = await self._pushed(self._frames(rate, secs=0.5), input_transport)
                self.assertTrue(all(f.sample_rate == self.BOT_RATE for f in pushed))
                self.assertAlmostEqual(
                    self._peak_hz(b"".join(f.audio for f in pushed)), self.TONE_HZ, delta=5
                )

    async def test_multichannel_audio_is_passed_through(self):
        """The resampler is mono-only; interleaved channels must not be fed to it."""
        frames = self._frames(self.SCENARIO_RATE, secs=0.2, channels=2)

        self.assertEqual(await self._pushed(frames), frames)


if __name__ == "__main__":
    unittest.main()
