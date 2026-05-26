#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import struct
import unittest

from pipecat.clocks.system_clock import SystemClock
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class _PassthroughResampler:
    async def resample(
        self, audio: bytes, in_rate: int, out_rate: int
    ) -> bytes:  # pragma: no cover - trivial
        return audio


async def _setup_processor(*, buffer_size: int = 0) -> AudioBufferProcessor:
    """Create and initialise a processor without starting recording.

    Calls setup() and sends a StartFrame through the public process_frame path so that
    the processor is fully initialised (task manager set, sample rate configured,
    __started flag set) without needing a full pipeline. The clock is explicitly
    started so tests exercising pipeline-clock values (e.g. ``recording_start_time``)
    observe realistic non-zero timestamps.
    """
    processor = AudioBufferProcessor(sample_rate=16000, num_channels=2, buffer_size=buffer_size)
    processor._input_resampler = _PassthroughResampler()
    processor._output_resampler = _PassthroughResampler()

    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    clock = SystemClock()
    clock.start()
    await processor.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    await processor.process_frame(
        StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
    )
    # Yield once so the processor's input-frame task actually starts running before
    # any test tears it down; avoids spurious "coroutine was never awaited" warnings.
    await asyncio.sleep(0)
    return processor


async def _make_processor(*, buffer_size: int = 0) -> AudioBufferProcessor:
    """Create a processor and start recording, ready for audio frames."""
    processor = await _setup_processor(buffer_size=buffer_size)
    await processor.start_recording()
    return processor


async def _capture_track_audio(processor: AudioBufferProcessor) -> tuple[bytes, bytes]:
    """Flush the processor and return (user_track, bot_track) from on_track_audio_data."""
    captured = {}
    event = asyncio.Event()

    async def on_track_audio_data(_, user, bot, sample_rate, num_channels):
        captured["user"] = user
        captured["bot"] = bot
        event.set()

    processor.add_event_handler("on_track_audio_data", on_track_audio_data)
    await processor.stop_recording()
    await asyncio.wait_for(event.wait(), timeout=1)
    return captured["user"], captured["bot"]


class TestAudioBufferProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processor = await _make_processor(buffer_size=4)

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
        await self.processor.process_frame(frame, FrameDirection.DOWNSTREAM)

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
        await self.processor.process_frame(frame, FrameDirection.DOWNSTREAM)

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
    """Tests that silence is not injected mid-utterance (fix for crackling artifacts).

    Each test verifies the audio alignment in the flushed tracks to confirm that
    silence is only added by _align_track_buffers at flush time (end of the buffer),
    never injected mid-stream while the affected track is actively producing audio.
    """

    async def test_no_silence_injected_into_bot_buffer_while_bot_speaking(self):
        """Bot audio must appear at the start of the bot track, not after mid-stream silence.

        Timeline:
          1. User sends 4 bytes  (bot not speaking → normal sync, no-op since bot is at 0)
          2. Bot starts speaking
          3. User sends 4 more bytes  (bot speaking → sync skipped; bot stays at 0)
          4. Bot sends 4 bytes of known audio

        Expected final bot track (8 bytes total after _align_track_buffers at flush):
          [bot_audio][silence_padding]  ← audio first, silence only at the end

        With the bug the bot track would be:
          [silence_injected_mid_stream][bot_audio]  ← silence inserted before the audio
        """
        p = await _make_processor()

        bot_audio = b"\xaa\xbb\xcc\xdd"

        await p.process_frame(
            InputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(
            OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        _, bot_track = await _capture_track_audio(p)
        await p.cleanup()

        # Audio must appear at the beginning of the bot track (not after injected silence).
        self.assertEqual(bot_track[:4], bot_audio)
        self.assertEqual(bot_track[4:], b"\x00" * 4)

    async def test_no_silence_injected_into_user_buffer_while_user_speaking(self):
        """User audio must appear at the start of the user track, not after mid-stream silence.

        Timeline:
          1. Bot sends 4 bytes  (user not speaking → normal sync, no-op since user is at 0)
          2. User starts speaking
          3. Bot sends 4 more bytes  (user speaking → sync skipped; user stays at 0)
          4. User sends 4 bytes of known audio

        Expected final user track (8 bytes total after _align_track_buffers at flush):
          [user_audio][silence_padding]  ← audio first, silence only at the end

        With the bug the user track would be:
          [silence_injected_mid_stream][user_audio]
        """
        p = await _make_processor()

        user_audio = b"\xaa\xbb\xcc\xdd"

        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(
            InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        await p.cleanup()

        self.assertEqual(user_track[:4], user_audio)
        self.assertEqual(user_track[4:], b"\x00" * 4)

    async def test_silence_resumes_into_bot_buffer_after_bot_stops_speaking(self):
        """After bot stops speaking, the bot buffer is synced again on user audio arrival.

        Timeline:
          1. User sends 4 bytes  (user=4, bot=0)
          2. Bot starts speaking
          3. User sends 4 more bytes  (sync skipped; user=8, bot=0)
          4. Bot stops speaking
          5. User sends 4 more bytes  (sync resumes; bot gets 8 bytes silence, user=12)

        Expected final bot track (12 bytes): 8 bytes silence then no more audio (bot never
        sent audio, _align_track_buffers pads bot to 12).
        The key assertion: bot has 8 bytes of silence at positions 0-7, confirming that
        the sync at step 5 did inject 8 bytes (positions 0-7 of the bot buffer).
        """
        p = await _make_processor()

        await p.process_frame(
            InputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x09\x0a\x0b\x0c", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        _, bot_track = await _capture_track_audio(p)
        await p.cleanup()

        # The sync at step 5 targets len(user)=8, so bot must have 8 bytes of silence
        # written before user's third chunk was added.
        self.assertEqual(bot_track[:8], b"\x00" * 8)

    async def test_silence_resumes_into_user_buffer_after_user_stops_speaking(self):
        """After user stops speaking, the user buffer is synced again on bot audio arrival.

        Timeline:
          1. Bot sends 4 bytes  (user=0, bot=4)
          2. User starts speaking
          3. Bot sends 4 more bytes  (sync skipped; user=0, bot=8)
          4. User stops speaking
          5. Bot sends 4 more bytes  (sync resumes; user gets 8 bytes silence, bot=12)

        Expected: user track has 8 bytes of silence at positions 0-7.
        """
        p = await _make_processor()

        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x09\x0a\x0b\x0c", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        await p.cleanup()

        self.assertEqual(user_track[:8], b"\x00" * 8)


class TestRecordingStartTime(unittest.IsolatedAsyncioTestCase):
    """Tests for the ``recording_start_time`` property.

    The property exposes the pipeline-clock time (in nanoseconds) at which the
    current recording started, letting downstream consumers align pipeline-clock
    events to the recorded audio's t=0.
    """

    async def test_recording_start_time_is_none_before_start(self):
        """Before ``start_recording()`` is ever called, the property is ``None``."""
        p = await _setup_processor()
        try:
            self.assertIsNone(p.recording_start_time)
        finally:
            await p.cleanup()

    async def test_recording_start_time_captured_on_start(self):
        """``start_recording()`` captures the current pipeline-clock time."""
        p = await _setup_processor()
        try:
            self.assertIsNone(p.recording_start_time)
            await p.start_recording()
            anchor = p.recording_start_time
            self.assertIsNotNone(anchor)
            self.assertIsInstance(anchor, int)
            self.assertGreater(anchor, 0)
        finally:
            await p.stop_recording()
            await p.cleanup()

    async def test_recording_start_time_cleared_on_stop(self):
        """``stop_recording()`` clears the anchor back to ``None``."""
        p = await _setup_processor()
        try:
            await p.start_recording()
            self.assertIsNotNone(p.recording_start_time)
            await p.stop_recording()
            self.assertIsNone(p.recording_start_time)
        finally:
            await p.cleanup()

    async def test_recording_start_time_refreshed_on_restart(self):
        """A second ``start_recording()`` captures a fresh, later anchor."""
        p = await _setup_processor()
        try:
            await p.start_recording()
            first = p.recording_start_time
            self.assertIsNotNone(first)

            await p.stop_recording()
            # Allow the monotonic clock to advance so the second anchor is strictly
            # greater than the first; ~1 ms is plenty for nanosecond resolution.
            await asyncio.sleep(0.001)

            await p.start_recording()
            second = p.recording_start_time
            self.assertIsNotNone(second)
            self.assertGreater(second, first)
        finally:
            await p.stop_recording()
            await p.cleanup()


if __name__ == "__main__":
    unittest.main()
