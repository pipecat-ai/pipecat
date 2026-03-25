#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import struct
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


class _PassthroughResampler:
    async def resample(
        self, audio: bytes, in_rate: int, out_rate: int
    ) -> bytes:  # pragma: no cover - trivial
        return audio


class TestAudioBufferProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processor = AudioBufferProcessor(sample_rate=16000, num_channels=2, buffer_size=4)
        self.processor._input_resampler = _PassthroughResampler()
        self.processor._output_resampler = _PassthroughResampler()
        self.processor._update_sample_rate(StartFrame(audio_out_sample_rate=16000))
        await self.processor.start_recording()

    async def asyncTearDown(self):
        if getattr(self.processor, "_recording", False):
            await self.processor.stop_recording()
        await self.processor.cleanup()

    async def test_flush_user_audio_pads_bot_track(self):
        user_audio = struct.pack("<hh", 1000, -1000)
        audio_event = asyncio.Event()
        track_event = asyncio.Event()
        captured = {}

        async def on_audio_data(_, audio: bytes, sample_rate: int, num_channels: int):
            captured["merged"] = (audio, sample_rate, num_channels)
            audio_event.set()

        async def on_track_audio_data(
            _, user: bytes, bot: bytes, sample_rate: int, num_channels: int
        ):
            captured["tracks"] = (user, bot, sample_rate, num_channels)
            track_event.set()

        self.processor.add_event_handler("on_audio_data", on_audio_data)
        self.processor.add_event_handler("on_track_audio_data", on_track_audio_data)

        frame = InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1)
        await self.processor._process_recording(frame)

        await asyncio.wait_for(audio_event.wait(), timeout=1)
        await asyncio.wait_for(track_event.wait(), timeout=1)

        merged_audio, merged_sr, merged_channels = captured["merged"]
        user_track, bot_track, track_sr, track_channels = captured["tracks"]

        self.assertEqual(merged_sr, 16000)
        self.assertEqual(merged_channels, 2)
        self.assertEqual(track_sr, 16000)
        self.assertEqual(track_channels, 2)
        self.assertEqual(user_track, user_audio)
        self.assertEqual(bot_track, b"\x00" * len(user_audio))
        self.assertEqual(len(merged_audio), len(user_audio) * 2)
        self.assertEqual(merged_audio[0:2], user_audio[0:2])
        self.assertEqual(merged_audio[2:4], b"\x00\x00")
        self.assertEqual(merged_audio[4:6], user_audio[2:4])
        self.assertEqual(merged_audio[6:8], b"\x00\x00")
        self.assertEqual(len(self.processor._user_audio_buffer), 0)
        self.assertEqual(len(self.processor._bot_audio_buffer), 0)

    async def test_flush_bot_audio_pads_user_track(self):
        bot_audio = struct.pack("<hh", -800, 400)
        audio_event = asyncio.Event()
        track_event = asyncio.Event()
        captured = {}

        async def on_audio_data(_, audio: bytes, sample_rate: int, num_channels: int):
            captured["merged"] = (audio, sample_rate, num_channels)
            audio_event.set()

        async def on_track_audio_data(
            _, user: bytes, bot: bytes, sample_rate: int, num_channels: int
        ):
            captured["tracks"] = (user, bot, sample_rate, num_channels)
            track_event.set()

        self.processor.add_event_handler("on_audio_data", on_audio_data)
        self.processor.add_event_handler("on_track_audio_data", on_track_audio_data)

        frame = OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1)
        await self.processor._process_recording(frame)

        await asyncio.wait_for(audio_event.wait(), timeout=1)
        await asyncio.wait_for(track_event.wait(), timeout=1)

        merged_audio, merged_sr, merged_channels = captured["merged"]
        user_track, bot_track, track_sr, track_channels = captured["tracks"]

        self.assertEqual(merged_sr, 16000)
        self.assertEqual(merged_channels, 2)
        self.assertEqual(track_sr, 16000)
        self.assertEqual(track_channels, 2)
        self.assertEqual(user_track, b"\x00" * len(bot_audio))
        self.assertEqual(bot_track, bot_audio)
        self.assertEqual(len(merged_audio), len(bot_audio) * 2)
        self.assertEqual(merged_audio[0:2], b"\x00\x00")
        self.assertEqual(merged_audio[2:4], bot_audio[0:2])
        self.assertEqual(merged_audio[4:6], b"\x00\x00")
        self.assertEqual(merged_audio[6:8], bot_audio[2:4])
        self.assertEqual(len(self.processor._user_audio_buffer), 0)
        self.assertEqual(len(self.processor._bot_audio_buffer), 0)


class TestSilenceInjectionGuards(unittest.IsolatedAsyncioTestCase):
    """Tests that silence is not injected mid-utterance (fix for crackling artifacts)."""

    async def _make_processor(self) -> AudioBufferProcessor:
        """Return a processor with no auto-flush and a passthrough resampler."""
        processor = AudioBufferProcessor(sample_rate=16000, num_channels=2, buffer_size=0)
        processor._input_resampler = _PassthroughResampler()
        processor._output_resampler = _PassthroughResampler()
        processor._update_sample_rate(StartFrame(audio_out_sample_rate=16000))
        await processor.start_recording()
        return processor

    async def asyncTearDown(self):
        # Processors created inside each test are cleaned up there; nothing shared.
        pass

    async def test_no_silence_injected_into_bot_buffer_while_bot_speaking(self):
        """Bot buffer must not receive silence padding while the bot is actively speaking."""
        p = await self._make_processor()

        # Give user buffer a head-start so a sync *would* pad the bot buffer normally.
        p._user_audio_buffer = bytearray(b"\x01\x02\x03\x04")

        # Bot starts speaking.
        await p._process_recording(BotStartedSpeakingFrame())

        # User audio arrives — without the fix the bot buffer would be padded to 4 bytes.
        user_audio = b"\x05\x06\x07\x08"
        await p._process_recording(
            InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1)
        )

        self.assertEqual(len(p._bot_audio_buffer), 0, "Bot buffer must not be padded while bot is speaking")
        self.assertEqual(bytes(p._user_audio_buffer), b"\x01\x02\x03\x04" + user_audio)

        await p.stop_recording()
        await p.cleanup()

    async def test_no_silence_injected_into_user_buffer_while_user_speaking(self):
        """User buffer must not receive silence padding while the user is actively speaking."""
        p = await self._make_processor()

        # Give bot buffer a head-start so a sync *would* pad the user buffer normally.
        p._bot_audio_buffer = bytearray(b"\x01\x02\x03\x04")

        # User starts speaking.
        await p._process_recording(UserStartedSpeakingFrame())

        # Bot audio arrives — without the fix the user buffer would be padded to 4 bytes.
        bot_audio = b"\x05\x06\x07\x08"
        await p._process_recording(
            OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1)
        )

        self.assertEqual(len(p._user_audio_buffer), 0, "User buffer must not be padded while user is speaking")
        self.assertEqual(bytes(p._bot_audio_buffer), b"\x01\x02\x03\x04" + bot_audio)

        await p.stop_recording()
        await p.cleanup()

    async def test_silence_resumes_into_bot_buffer_after_bot_stops_speaking(self):
        """After bot stops speaking, silence injection into the bot buffer resumes."""
        p = await self._make_processor()

        # Give user buffer a head-start.
        p._user_audio_buffer = bytearray(b"\x01\x02\x03\x04")

        # Bot speaks and stops.
        await p._process_recording(BotStartedSpeakingFrame())
        await p._process_recording(BotStoppedSpeakingFrame())

        # User audio arrives now — bot is no longer speaking so sync should run.
        user_audio = b"\x05\x06\x07\x08"
        await p._process_recording(
            InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1)
        )

        # Bot buffer should have been padded with silence to match the 4-byte head-start.
        self.assertEqual(len(p._bot_audio_buffer), 4, "Bot buffer should be padded after bot stops speaking")
        self.assertEqual(bytes(p._bot_audio_buffer), b"\x00\x00\x00\x00")

        await p.stop_recording()
        await p.cleanup()

    async def test_silence_resumes_into_user_buffer_after_user_stops_speaking(self):
        """After user stops speaking, silence injection into the user buffer resumes."""
        p = await self._make_processor()

        # Give bot buffer a head-start.
        p._bot_audio_buffer = bytearray(b"\x01\x02\x03\x04")

        # User speaks and stops.
        await p._process_recording(UserStartedSpeakingFrame())
        await p._process_recording(UserStoppedSpeakingFrame())

        # Bot audio arrives now — user is no longer speaking so sync should run.
        bot_audio = b"\x05\x06\x07\x08"
        await p._process_recording(
            OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1)
        )

        # User buffer should have been padded with silence to match the 4-byte head-start.
        self.assertEqual(len(p._user_audio_buffer), 4, "User buffer should be padded after user stops speaking")
        self.assertEqual(bytes(p._user_audio_buffer), b"\x00\x00\x00\x00")

        await p.stop_recording()
        await p.cleanup()


if __name__ == "__main__":
    unittest.main()
