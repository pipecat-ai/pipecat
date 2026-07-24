#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    AudioBufferStartRecordingFrame,
    AudioBufferStopRecordingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class _PassthroughResampler:
    async def resample(
        self, audio: bytes, in_rate: int, out_rate: int
    ) -> bytes:  # pragma: no cover - trivial
        return audio


async def _make_processor(
    *, buffer_size: int = 0, start: bool = True, auto_start: bool = False
) -> AudioBufferProcessor:
    """Create a processor ready to record.

    Calls setup() and sends a StartFrame through the public process_frame path so that
    the processor is fully initialised (task manager set, sample rate configured,
    __started flag set) without needing a full pipeline.

    When ``start`` is True the processor starts recording before returning; pass
    ``start=False`` to leave recording off (e.g. to test frame-driven start).
    """
    processor = AudioBufferProcessor(
        sample_rate=16000,
        num_channels=2,
        buffer_size=buffer_size,
        auto_start_recording=auto_start,
    )
    processor._input_resampler = _PassthroughResampler()
    processor._output_resampler = _PassthroughResampler()

    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )

    await processor.process_frame(
        StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
    )
    if start:
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


class TestMuteGapSilenceInsertion(unittest.IsolatedAsyncioTestCase):
    """Tests for _fill_buffer_silence_gap (user buffer path).

    When the microphone is muted, no InputAudioRawFrame arrives. Without gap
    detection the next utterance is appended directly after the previous one,
    making two utterances spoken seconds apart sound concatenated.

    These tests verify that silence proportional to the wall-clock gap is
    inserted into the user buffer so the recorded timeline stays accurate.
    """

    # 16-bit mono at 16 kHz → 2 bytes per sample
    _BYTES_PER_SECOND = 16000 * 2

    def _silence_for_gap(self, elapsed: float, frame_bytes: int) -> int:
        """Expected silence bytes for a given elapsed time and incoming frame size."""
        frame_duration = frame_bytes / self._BYTES_PER_SECOND
        gap = elapsed - frame_duration
        if gap <= 0.2:
            return 0
        n = int(gap * self._BYTES_PER_SECOND)
        return n - (n % 2)  # 16-bit alignment

    async def _send_user_frame(
        self, processor: AudioBufferProcessor, audio: bytes = b"\x01\x02\x03\x04"
    ):
        await processor.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def _send_bot_frame(
        self, processor: AudioBufferProcessor, audio: bytes = b"\x01\x02\x03\x04"
    ):
        await processor.process_frame(
            OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def test_no_silence_when_no_prior_timestamp(self):
        """First user frame must not trigger silence insertion when there is no prior timestamp."""
        p = await _make_processor()
        p._last_user_buffer_update_time = None

        audio = b"\x01\x02\x03\x04"
        await self._send_user_frame(p, audio)

        self.assertEqual(len(p._user_audio_buffer), 4)
        self.assertEqual(bytes(p._user_audio_buffer), audio)
        await p.cleanup()

    async def test_no_silence_for_gap_below_threshold(self):
        """A 100 ms gap (below the 200 ms threshold) must not insert any silence."""
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.1  # 100 ms later
            await self._send_user_frame(p, audio)

        self.assertEqual(len(p._user_audio_buffer), 4)
        self.assertEqual(bytes(p._user_audio_buffer), audio)
        await p.cleanup()

    async def test_silence_proportional_to_mute_gap(self):
        """A 1-second mute gap must insert ~1 second of silence before the new audio."""
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0  # 1-second gap
            await self._send_user_frame(p, audio)

        expected_silence = self._silence_for_gap(1.0, len(audio))

        self.assertEqual(len(p._user_audio_buffer), expected_silence + len(audio))
        # Silence prefix.
        self.assertEqual(bytes(p._user_audio_buffer[:expected_silence]), b"\x00" * expected_silence)
        # Audio at the end.
        self.assertEqual(bytes(p._user_audio_buffer[-len(audio) :]), audio)
        await p.cleanup()

    async def test_two_utterances_separated_by_mute_have_silence_gap(self):
        """Two utterances with a muted-mic gap between them must not be concatenated.

        This is the original bug report: without the fix the second utterance is
        appended directly after the first with no silence, making them sound like
        one continuous utterance.
        """
        p = await _make_processor()
        utterance = b"\x11\x22\x33\x44"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            # Utterance 1 is already in the buffer at t=0.
            mock_time.monotonic.return_value = 0.0
            p._last_user_buffer_update_time = 0.0
            p._user_audio_buffer.extend(utterance)

            # One second later the user unmutes and speaks utterance 2.
            mock_time.monotonic.return_value = 1.0
            await self._send_user_frame(p, utterance)

        # Must be longer than both utterances back-to-back (the bug).
        self.assertGreater(len(p._user_audio_buffer), len(utterance) * 2)
        # Utterance 1 at the start.
        self.assertEqual(bytes(p._user_audio_buffer[: len(utterance)]), utterance)
        # Utterance 2 at the end.
        self.assertEqual(bytes(p._user_audio_buffer[-len(utterance) :]), utterance)
        # Everything in between must be silence.
        silence_region = bytes(p._user_audio_buffer[len(utterance) : -len(utterance)])
        self.assertTrue(all(b == 0 for b in silence_region))
        await p.cleanup()

    async def test_bot_audio_during_mute_advances_timestamp_preventing_double_counting(self):
        """Bot audio that syncs the user buffer must advance the user-buffer timestamp.

        Timeline (all times mocked):
          t=0.0  prior user activity — timestamp is pinned here
          t=1.0  bot speaks; _last_user_buffer_update_time advances to 1.0
          t=2.0  user unmutes and speaks

        The gap fill must measure from t=1.0 (last sync by bot audio), not from
        t=0.0 (last real user audio). Without this guard the silence would be
        doubled (~2 s instead of ~1 s).
        """
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            # Pin timestamp at t=0 (simulates prior user audio with no frames sent).
            p._last_user_buffer_update_time = 0.0

            # Bot speaks at t=1 → must advance _last_user_buffer_update_time to 1.0.
            mock_time.monotonic.return_value = 1.0
            await self._send_bot_frame(p, audio)

            # User unmutes at t=2.
            mock_time.monotonic.return_value = 2.0
            await self._send_user_frame(p, audio)

        expected_silence_1s = self._silence_for_gap(2.0 - 1.0, len(audio))
        expected_silence_2s = self._silence_for_gap(2.0 - 0.0, len(audio))

        actual_silence = len(p._user_audio_buffer) - len(audio)
        # Gap should be ~1 s worth of silence, not ~2 s.
        self.assertAlmostEqual(actual_silence, expected_silence_1s, delta=4)
        self.assertNotAlmostEqual(actual_silence, expected_silence_2s, delta=4)
        await p.cleanup()

    async def test_bot_buffer_synced_to_user_position_after_gap_fill(self):
        """After gap silence is inserted in the user buffer, the bot buffer is synced.

        The sync targets the user buffer length *after* the silence is inserted
        but *before* the new user audio is appended, so both buffers share the
        same temporal reference point.
        """
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0
            await self._send_user_frame(p, audio)

        expected_silence = self._silence_for_gap(1.0, len(audio))

        # Bot buffer should equal the silence that was inserted (user position
        # after silence, before the new audio frame was appended).
        self.assertEqual(len(p._bot_audio_buffer), expected_silence)
        self.assertEqual(bytes(p._bot_audio_buffer), b"\x00" * expected_silence)
        await p.cleanup()

    async def test_reset_recording_clears_timestamp(self):
        """stop_recording must reset _last_user_buffer_update_time to None."""
        p = await _make_processor()
        p._last_user_buffer_update_time = 999.0

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            await p.stop_recording()

        self.assertIsNone(p._last_user_buffer_update_time)
        await p.cleanup()

    async def test_buffer_flush_resets_timestamp_to_flush_time(self):
        """After a buffer flush, the timestamp is set to the flush time (not None).

        This ensures that when the next user frame arrives after a flush the gap
        is measured from the flush point, not from a stale earlier timestamp.
        """
        audio = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # 8 bytes == buffer_size
        p = await _make_processor(buffer_size=len(audio))

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            mock_time.monotonic.return_value = 5.0
            p._last_user_buffer_update_time = 0.0

            flushed = asyncio.Event()

            async def on_audio_data(*_):
                flushed.set()

            p.add_event_handler("on_audio_data", on_audio_data)
            await self._send_user_frame(p, audio)
            # _reset_primary_audio_buffers is called synchronously inside
            # process_frame once buffer_size is hit, so no wait is needed —
            # but we yield to let any background tasks settle.
            await asyncio.sleep(0)

        # Timestamp should be set to the mocked flush time, not None.
        self.assertIsNotNone(p._last_user_buffer_update_time)
        self.assertEqual(p._last_user_buffer_update_time, 5.0)
        await p.cleanup()


class TestBotSilenceGapInsertion(unittest.IsolatedAsyncioTestCase):
    """Tests for _fill_buffer_silence_gap (bot buffer path).

    Mirror of TestMuteGapSilenceInsertion for the bot-audio path. When the
    bot is briefly idle between utterances (e.g. progressive hold messages
    spoken while a slow function call runs), no OutputAudioRawFrame arrives.
    Without gap detection on the bot side the next utterance is appended
    directly after the previous one, making two utterances spoken seconds
    apart sound concatenated in the recording.

    These tests verify that silence proportional to the wall-clock gap is
    inserted into the bot buffer so the recorded timeline stays accurate.
    """

    # 16-bit mono at 16 kHz → 2 bytes per sample
    _BYTES_PER_SECOND = 16000 * 2

    def _silence_for_gap(self, elapsed: float, frame_bytes: int) -> int:
        """Expected silence bytes for a given elapsed time and incoming frame size."""
        frame_duration = frame_bytes / self._BYTES_PER_SECOND
        gap = elapsed - frame_duration
        if gap <= 0.2:
            return 0
        n = int(gap * self._BYTES_PER_SECOND)
        return n - (n % 2)  # 16-bit alignment

    async def _send_user_frame(
        self, processor: AudioBufferProcessor, audio: bytes = b"\x01\x02\x03\x04"
    ):
        await processor.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def _send_bot_frame(
        self, processor: AudioBufferProcessor, audio: bytes = b"\x01\x02\x03\x04"
    ):
        await processor.process_frame(
            OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def test_no_silence_when_no_prior_timestamp(self):
        """First bot frame must not trigger silence insertion when there is no prior timestamp."""
        p = await _make_processor()
        p._last_bot_buffer_update_time = None

        audio = b"\x01\x02\x03\x04"
        await self._send_bot_frame(p, audio)

        self.assertEqual(len(p._bot_audio_buffer), 4)
        self.assertEqual(bytes(p._bot_audio_buffer), audio)
        await p.cleanup()

    async def test_no_silence_for_gap_below_threshold(self):
        """A 100 ms gap (below the 200 ms threshold) must not insert any silence."""
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_bot_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.1  # 100 ms later
            await self._send_bot_frame(p, audio)

        self.assertEqual(len(p._bot_audio_buffer), 4)
        self.assertEqual(bytes(p._bot_audio_buffer), audio)
        await p.cleanup()

    async def test_silence_proportional_to_idle_gap(self):
        """A 1-second idle gap must insert ~1 second of silence before the new audio."""
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_bot_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0  # 1-second gap
            await self._send_bot_frame(p, audio)

        expected_silence = self._silence_for_gap(1.0, len(audio))

        self.assertEqual(len(p._bot_audio_buffer), expected_silence + len(audio))
        # Silence prefix.
        self.assertEqual(bytes(p._bot_audio_buffer[:expected_silence]), b"\x00" * expected_silence)
        # Audio at the end.
        self.assertEqual(bytes(p._bot_audio_buffer[-len(audio) :]), audio)
        await p.cleanup()

    async def test_two_utterances_separated_by_pause_have_silence_gap(self):
        """Two bot utterances spoken seconds apart must not be concatenated.

        This is the bug report for the progressive hold messages: without
        the fix the second hold line is appended directly after the first
        with no silence, making them sound like one continuous utterance
        in the recording.
        """
        p = await _make_processor()
        utterance = b"\x11\x22\x33\x44"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            # Utterance 1 is already in the buffer at t=0.
            mock_time.monotonic.return_value = 0.0
            p._last_bot_buffer_update_time = 0.0
            p._bot_audio_buffer.extend(utterance)

            # One second later the bot speaks utterance 2.
            mock_time.monotonic.return_value = 1.0
            await self._send_bot_frame(p, utterance)

        # Must be longer than both utterances back-to-back (the bug).
        self.assertGreater(len(p._bot_audio_buffer), len(utterance) * 2)
        # Utterance 1 at the start.
        self.assertEqual(bytes(p._bot_audio_buffer[: len(utterance)]), utterance)
        # Utterance 2 at the end.
        self.assertEqual(bytes(p._bot_audio_buffer[-len(utterance) :]), utterance)
        # Everything in between must be silence.
        silence_region = bytes(p._bot_audio_buffer[len(utterance) : -len(utterance)])
        self.assertTrue(all(b == 0 for b in silence_region))
        await p.cleanup()

    async def test_user_audio_during_pause_advances_timestamp_preventing_double_counting(self):
        """User audio that syncs the bot buffer must advance the bot-buffer timestamp.

        Timeline (all times mocked):
          t=0.0  prior bot activity — timestamp is pinned here
          t=1.0  user speaks; _last_bot_buffer_update_time advances to 1.0
          t=2.0  bot resumes speaking

        The gap fill must measure from t=1.0 (last sync by user audio), not from
        t=0.0 (last real bot audio). Without this guard the silence would be
        doubled (~2 s instead of ~1 s).
        """
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            # Pin bot timestamp at t=0 (simulates prior bot audio with no frames sent).
            p._last_bot_buffer_update_time = 0.0

            # User speaks at t=1 → must advance _last_bot_buffer_update_time to 1.0.
            mock_time.monotonic.return_value = 1.0
            await self._send_user_frame(p, audio)

            # Snapshot the bot buffer size right after the user frame; everything
            # added after this is from the bot frame (gap fill + new audio).
            bot_len_after_user = len(p._bot_audio_buffer)

            # Bot resumes at t=2.
            mock_time.monotonic.return_value = 2.0
            await self._send_bot_frame(p, audio)

        # New bytes added by the bot frame = gap fill + audio frame.
        bot_frame_contribution = len(p._bot_audio_buffer) - bot_len_after_user
        gap_fill_bytes = bot_frame_contribution - len(audio)

        expected_silence_1s = self._silence_for_gap(2.0 - 1.0, len(audio))
        expected_silence_2s = self._silence_for_gap(2.0 - 0.0, len(audio))

        # Gap fill should reflect a 1 s gap (since user frame advanced the
        # bot timestamp), not a 2 s gap (which would happen without the
        # advance, double-counting the silence the user-sync already added).
        self.assertAlmostEqual(gap_fill_bytes, expected_silence_1s, delta=4)
        self.assertNotAlmostEqual(gap_fill_bytes, expected_silence_2s, delta=4)
        await p.cleanup()

    async def test_user_buffer_synced_to_bot_position_after_gap_fill(self):
        """After gap silence is inserted in the bot buffer, the user buffer is synced.

        The sync targets the bot buffer length *after* the silence is inserted
        but *before* the new bot audio is appended, so both buffers share the
        same temporal reference point.
        """
        p = await _make_processor()
        audio = b"\x01\x02\x03\x04"

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_bot_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0
            await self._send_bot_frame(p, audio)

        expected_silence = self._silence_for_gap(1.0, len(audio))

        # User buffer should equal the silence that was inserted (bot position
        # after silence, before the new audio frame was appended).
        self.assertEqual(len(p._user_audio_buffer), expected_silence)
        self.assertEqual(bytes(p._user_audio_buffer), b"\x00" * expected_silence)
        await p.cleanup()

    async def test_reset_recording_clears_bot_timestamp(self):
        """stop_recording must reset _last_bot_buffer_update_time to None."""
        p = await _make_processor()
        p._last_bot_buffer_update_time = 999.0

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            await p.stop_recording()

        self.assertIsNone(p._last_bot_buffer_update_time)
        await p.cleanup()

    async def test_buffer_flush_resets_bot_timestamp_to_flush_time(self):
        """After a buffer flush, the bot timestamp is set to the flush time (not None)."""
        audio = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # 8 bytes == buffer_size
        p = await _make_processor(buffer_size=len(audio))

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            mock_time.monotonic.return_value = 5.0
            p._last_bot_buffer_update_time = 0.0

            flushed = asyncio.Event()

            async def on_audio_data(*_):
                flushed.set()

            p.add_event_handler("on_audio_data", on_audio_data)
            await self._send_bot_frame(p, audio)
            await asyncio.sleep(0)

        # Timestamp should be set to the mocked flush time, not None.
        self.assertIsNotNone(p._last_bot_buffer_update_time)
        self.assertEqual(p._last_bot_buffer_update_time, 5.0)
        await p.cleanup()


class TestRecordingControlFrames(unittest.IsolatedAsyncioTestCase):
    """Tests for frame-driven recording control.

    AudioBufferStartRecordingFrame / AudioBufferStopRecordingFrame let any
    upstream processor start and stop recording from within the frame flow,
    triggering the same start_recording() / stop_recording() methods as the
    direct API.
    """

    async def test_start_recording_frame_enables_recording(self):
        """A start frame turns recording on so subsequent audio is buffered."""
        p = await _make_processor(start=False)
        self.assertFalse(p._recording)

        await p.process_frame(AudioBufferStartRecordingFrame(), FrameDirection.DOWNSTREAM)
        self.assertTrue(p._recording)

        audio = struct.pack("<hh", 1000, -1000)
        await p.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        self.assertEqual(user_track, audio)
        await p.cleanup()

    async def test_audio_ignored_before_start_recording_frame(self):
        """Audio arriving before a start frame is not buffered."""
        p = await _make_processor(start=False)

        await p.process_frame(
            InputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        self.assertFalse(p.has_audio())
        await p.cleanup()

    async def test_stop_recording_frame_flushes_and_disables(self):
        """A stop frame flushes buffered audio and turns recording off."""
        p = await _make_processor()

        audio = struct.pack("<hh", 1000, -1000)
        await p.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        captured = {}
        event = asyncio.Event()

        async def on_track_audio_data(_, user, bot, sample_rate, num_channels):
            captured["user"] = user
            event.set()

        p.add_event_handler("on_track_audio_data", on_track_audio_data)
        await p.process_frame(AudioBufferStopRecordingFrame(), FrameDirection.DOWNSTREAM)

        await asyncio.wait_for(event.wait(), timeout=1)
        self.assertEqual(captured["user"], audio)
        self.assertFalse(p._recording)
        self.assertFalse(p.has_audio())
        await p.cleanup()

    async def test_recording_control_frames_passed_downstream(self):
        """Control frames are re-pushed so other processors also react."""
        processor = AudioBufferProcessor(sample_rate=16000, num_channels=2)

        await run_test(
            processor,
            frames_to_send=[
                AudioBufferStartRecordingFrame(),
                AudioBufferStopRecordingFrame(),
            ],
            expected_down_frames=[
                AudioBufferStartRecordingFrame,
                AudioBufferStopRecordingFrame,
            ],
        )


class TestAutoStartRecording(unittest.IsolatedAsyncioTestCase):
    """Tests for the auto_start_recording constructor option.

    With auto_start_recording=True the processor starts recording as soon as
    it handles the StartFrame, without an explicit start_recording() call or
    an AudioBufferStartRecordingFrame.
    """

    async def test_auto_start_begins_recording_on_start_frame(self):
        """Recording is active right after pipeline start; audio is buffered."""
        p = await _make_processor(start=False, auto_start=True)
        self.assertTrue(p._recording)

        audio = struct.pack("<hh", 1000, -1000)
        await p.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        self.assertEqual(user_track, audio)
        await p.cleanup()

    async def test_auto_start_disabled_by_default(self):
        """Without the option, the StartFrame does not begin recording."""
        p = await _make_processor(start=False)
        self.assertFalse(p._recording)
        await p.cleanup()

    async def test_auto_start_fires_started_event(self):
        """on_recording_started fires when auto-start kicks in."""
        p = AudioBufferProcessor(sample_rate=16000, auto_start_recording=True)

        fired = asyncio.Event()
        p.add_event_handler("on_recording_started", lambda _: fired.set())

        await run_test(p, frames_to_send=[], expected_down_frames=[])
        await asyncio.wait_for(fired.wait(), timeout=1)


class TestRecordingLifecycleEvents(unittest.IsolatedAsyncioTestCase):
    """Tests for the on_recording_started / on_recording_stopped events.

    These fire only on actual recording state transitions, decoupling the
    code that triggers recording (a direct call or a control frame) from
    code that wants to react to it (UI indicators, logging, etc.).
    """

    async def test_started_event_fires_on_start(self):
        """on_recording_started fires when recording transitions to active."""
        p = await _make_processor(start=False)

        fired = asyncio.Event()
        p.add_event_handler("on_recording_started", lambda _: fired.set())

        await p.start_recording()
        await asyncio.wait_for(fired.wait(), timeout=1)
        await p.cleanup()

    async def test_started_event_fires_via_frame(self):
        """on_recording_started fires when started by a control frame."""
        p = await _make_processor(start=False)

        fired = asyncio.Event()
        p.add_event_handler("on_recording_started", lambda _: fired.set())

        await p.process_frame(AudioBufferStartRecordingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(fired.wait(), timeout=1)
        await p.cleanup()

    async def test_started_event_not_refired_when_already_recording(self):
        """A redundant start (already recording) does not re-fire the event."""
        p = await _make_processor()  # already recording

        started = []
        p.add_event_handler("on_recording_started", lambda _: started.append(True))

        await p.start_recording()
        await asyncio.sleep(0.05)  # give any erroneously-scheduled handler a chance
        self.assertEqual(len(started), 0)
        await p.cleanup()

    async def test_redundant_start_preserves_buffered_audio(self):
        """A redundant start (already recording) does not reset the buffers."""
        p = await _make_processor()  # already recording

        await p.process_frame(
            InputAudioRawFrame(
                audio=struct.pack("<hh", 1000, -1000), sample_rate=16000, num_channels=1
            ),
            FrameDirection.DOWNSTREAM,
        )
        self.assertTrue(p.has_audio())

        await p.start_recording()
        self.assertTrue(p.has_audio())
        await p.cleanup()

    async def test_stopped_event_fires_with_final_audio_already_emitted(self):
        """on_recording_stopped fires; the final audio handler runs as part of stop."""
        p = await _make_processor()

        await p.process_frame(
            InputAudioRawFrame(
                audio=struct.pack("<hh", 1000, -1000), sample_rate=16000, num_channels=1
            ),
            FrameDirection.DOWNSTREAM,
        )

        audio_fired = asyncio.Event()
        stopped_fired = asyncio.Event()
        p.add_event_handler("on_audio_data", lambda *_: audio_fired.set())
        p.add_event_handler("on_recording_stopped", lambda _: stopped_fired.set())

        await p.stop_recording()

        await asyncio.wait_for(audio_fired.wait(), timeout=1)
        await asyncio.wait_for(stopped_fired.wait(), timeout=1)
        # By the time stop signals, recording is off and buffers are cleared.
        self.assertFalse(p._recording)
        self.assertFalse(p.has_audio())
        await p.cleanup()

    async def test_stopped_event_not_fired_when_not_recording(self):
        """Stopping when not recording (e.g. EndFrame before start) does not fire."""
        p = await _make_processor(start=False)

        stopped = []
        p.add_event_handler("on_recording_stopped", lambda _: stopped.append(True))

        await p.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)
        self.assertEqual(len(stopped), 0)
        await p.cleanup()


class TestStallBurstReconciliation(unittest.IsolatedAsyncioTestCase):
    """Tests for catch-up burst trimming in _fill_buffer_silence_gap.

    When the event loop stalls (slow sync callback, GC pause...) the transport
    keeps queuing real audio. On resume, the first frame arrives with a large
    wall-clock gap (silence is injected for the stall window) and the queued
    backlog then arrives in a back-to-back burst. Without reconciliation the
    stall window is counted twice: once as injected silence and once as the
    late real audio, so the recording grows past the real call length.

    These tests verify that the overshoot is trimmed out of the previously
    injected silence (and only out of that silence, never real audio), while
    genuine gaps (muted mic, idle bot) keep their injected silence unchanged.
    """

    # 16-bit mono at 16 kHz → 2 bytes per sample
    _BYTES_PER_SECOND = 16000 * 2
    # One 20 ms frame of non-zero audio (distinguishable from injected silence).
    _FRAME = b"\x11\x22" * 320  # 640 bytes == 20 ms

    async def _send_user_frame(self, processor: AudioBufferProcessor, audio: bytes):
        await processor.process_frame(
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def _send_bot_frame(self, processor: AudioBufferProcessor, audio: bytes):
        await processor.process_frame(
            OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def test_user_stall_burst_does_not_double_count(self):
        """A 500 ms stall followed by a catch-up burst must not grow the recording.

        Timeline (mocked): the buffer was last written at t=0. The event loop
        stalls; at t=0.5 the backlog of 25 x 20 ms frames drains back-to-back.
        Real elapsed time is 0.5 s and real audio is 0.5 s, so the recorded
        track must be 0.5 s — not 0.5 s audio + ~0.48 s injected silence.
        """
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.5
            for _ in range(25):
                await self._send_user_frame(p, self._FRAME)

        # 0.5 s of wall clock → exactly 0.5 s of recorded audio.
        self.assertEqual(len(p._user_audio_buffer), int(0.5 * self._BYTES_PER_SECOND))
        # All injected silence was trimmed; only the real audio remains.
        self.assertEqual(bytes(p._user_audio_buffer), self._FRAME * 25)
        await p.cleanup()

    async def test_bot_stall_burst_does_not_double_count(self):
        """Mirror of the stall+burst case on the bot buffer path."""
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_bot_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.5
            for _ in range(25):
                await self._send_bot_frame(p, self._FRAME)

        self.assertEqual(len(p._bot_audio_buffer), int(0.5 * self._BYTES_PER_SECOND))
        self.assertEqual(bytes(p._bot_audio_buffer), self._FRAME * 25)
        # The user buffer was synced to the bot timeline (all silence) and must
        # not exceed the wall-clock duration either.
        self.assertLessEqual(len(p._user_audio_buffer), int(0.5 * self._BYTES_PER_SECOND))
        await p.cleanup()

    async def test_burst_longer_than_stall_never_trims_real_audio(self):
        """Trimming is capped at the injected silence; real audio is untouched.

        The burst delivers 0.6 s of real audio after a 0.5 s stall. All of the
        injected silence gets trimmed, and every byte of real audio survives.
        """
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.5
            for _ in range(30):  # 0.6 s of audio
                await self._send_user_frame(p, self._FRAME)

        self.assertEqual(bytes(p._user_audio_buffer), self._FRAME * 30)
        await p.cleanup()

    async def test_burst_without_prior_stall_trims_nothing(self):
        """A burst with no injected silence must keep all real audio."""
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 0.02
            for _ in range(6):
                await self._send_user_frame(p, self._FRAME)

        self.assertEqual(bytes(p._user_audio_buffer), self._FRAME * 6)
        await p.cleanup()

    async def test_genuine_mute_gap_silence_is_preserved(self):
        """The original #4561 case: a genuine gap keeps its injected silence.

        A 1 s muted-mic gap injects ~1 s of silence. Frames that then resume
        at real-time pace (no catch-up burst) must not trim any of it.
        """
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            # User unmutes 1 s later.
            mock_time.monotonic.return_value = 1.0
            await self._send_user_frame(p, self._FRAME)
            # Frames continue at real-time pace.
            for i in range(1, 4):
                mock_time.monotonic.return_value = 1.0 + i * 0.02
                await self._send_user_frame(p, self._FRAME)

        # Same silence amount the pre-fix gap fill produced: (1.0 - 0.02) s.
        frame_duration = len(self._FRAME) / self._BYTES_PER_SECOND
        expected_silence = int((1.0 - frame_duration) * self._BYTES_PER_SECOND)
        expected_silence -= expected_silence % 2

        self.assertEqual(len(p._user_audio_buffer), expected_silence + 4 * len(self._FRAME))
        self.assertEqual(bytes(p._user_audio_buffer[:expected_silence]), b"\x00" * expected_silence)
        self.assertEqual(bytes(p._user_audio_buffer[expected_silence:]), self._FRAME * 4)
        await p.cleanup()

    async def test_mute_gap_after_stall_burst_still_injects_silence(self):
        """A genuine gap occurring after a trimmed burst still gets silence."""
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            # Stall + burst (fully reconciled to 0.5 s).
            mock_time.monotonic.return_value = 0.5
            for _ in range(25):
                await self._send_user_frame(p, self._FRAME)
            len_after_burst = len(p._user_audio_buffer)

            # User then mutes for 1 s and speaks again.
            mock_time.monotonic.return_value = 1.5
            await self._send_user_frame(p, self._FRAME)

        frame_duration = len(self._FRAME) / self._BYTES_PER_SECOND
        expected_silence = int((1.0 - frame_duration) * self._BYTES_PER_SECOND)
        expected_silence -= expected_silence % 2

        added = len(p._user_audio_buffer) - len_after_burst
        self.assertEqual(added, expected_silence + len(self._FRAME))
        await p.cleanup()

    async def test_buffer_flush_resets_trim_state(self):
        """A buffer flush must clear the trimmable-silence region.

        After a flush the buffers are emptied, so a stale silence region from
        before the flush would point into (new) real audio. The tracker must
        be reset so no post-flush trim can touch real audio.
        """
        audio = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # 8 bytes == buffer_size
        p = await _make_processor(buffer_size=len(audio))

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            # Simulate leftover trimmable silence from before the flush.
            p._user_gap_tracker.silence_start = 2
            p._user_gap_tracker.silence_len = 100
            mock_time.monotonic.return_value = 0.01  # below gap threshold
            await self._send_user_frame(p, audio)  # hits buffer_size → flush

        self.assertEqual(p._user_gap_tracker.silence_len, 0)
        self.assertEqual(p._user_gap_tracker.expected_len, 0.0)
        self.assertEqual(p._bot_gap_tracker.silence_len, 0)
        await p.cleanup()

    async def test_stop_recording_resets_trim_state(self):
        """stop_recording must clear the wall-clock tracking state."""
        p = await _make_processor()
        p._user_gap_tracker.expected_len = 123.0
        p._user_gap_tracker.silence_start = 10
        p._user_gap_tracker.silence_len = 50

        await p.stop_recording()

        self.assertIsNone(p._user_gap_tracker.expected_len)
        self.assertEqual(p._user_gap_tracker.silence_len, 0)
        self.assertIsNone(p._bot_gap_tracker.expected_len)
        await p.cleanup()


class TestCaptureTimePositioning(unittest.IsolatedAsyncioTestCase):
    """Tests for capture-timestamp positioning.

    Frames that carry a capture timestamp in
    ``frame.metadata["audio_capture_time_ns"]`` (e.g. recorded by
    ``TwilioFrameSerializer`` from Twilio's per-message ``timestamp``) are
    placed in the buffer by capture time instead of arrival pacing. This
    removes the muted-mic vs stalled-audio ambiguity over WebSocket
    transports: a mute shows up as a jump in capture time (silence is padded
    in), while late audio delivered in a burst keeps contiguous capture times
    (nothing to pad, and never anything to trim). The generic ``frame.pts``
    field is intentionally NOT the trigger, since transports set it in varying
    units; see ``test_raw_pts_without_metadata_uses_arrival_pacing``.
    """

    # 16-bit mono at 16 kHz → 2 bytes per sample
    _BYTES_PER_SECOND = 16000 * 2
    # One 20 ms frame of non-zero audio (distinguishable from injected silence).
    _FRAME = b"\x11\x22" * 320  # 640 bytes == 20 ms
    _FRAME_NS = 20 * 1_000_000  # 20 ms in nanoseconds

    async def _send_user_frame(
        self, processor: AudioBufferProcessor, audio: bytes, capture_time_ns: int
    ):
        frame = InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        frame.metadata["audio_capture_time_ns"] = capture_time_ns
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _send_bot_frame(
        self, processor: AudioBufferProcessor, audio: bytes, capture_time_ns: int
    ):
        frame = OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        frame.metadata["audio_capture_time_ns"] = capture_time_ns
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_pts_gap_pads_silence(self):
        """A 1 s jump in capture time pads ~1 s of silence before the audio."""
        p = await _make_processor()

        await self._send_user_frame(p, self._FRAME, 0)
        await self._send_user_frame(p, self._FRAME, 1_000_000_000)

        # Frame at pts=0 anchors at position 0; frame at pts=1 s belongs at
        # exactly 1 s worth of bytes.
        one_second = self._BYTES_PER_SECOND
        self.assertEqual(len(p._user_audio_buffer), one_second + len(self._FRAME))
        self.assertEqual(bytes(p._user_audio_buffer[: len(self._FRAME)]), self._FRAME)
        silence_region = bytes(p._user_audio_buffer[len(self._FRAME) : one_second])
        self.assertTrue(all(b == 0 for b in silence_region))
        self.assertEqual(bytes(p._user_audio_buffer[one_second:]), self._FRAME)
        await p.cleanup()

    async def test_pts_contiguous_burst_adds_no_silence(self):
        """Contiguous capture timestamps delivered in a burst add no silence.

        This is the stalled-delivery case: the audio existed on the wire and
        arrived late, so pts values are contiguous and the recording must
        contain only the real audio, with no phantom silence.
        """
        p = await _make_processor()

        for i in range(25):
            await self._send_user_frame(p, self._FRAME, i * self._FRAME_NS)

        self.assertEqual(bytes(p._user_audio_buffer), self._FRAME * 25)
        await p.cleanup()

    async def test_pts_mute_gap_preserved_despite_burst_arrival(self):
        """The #4561 regression case: a genuine mute gap survives a burst.

        The user speaks, mutes for 1 s, then the resumed audio arrives in a
        back-to-back burst. Arrival pacing reads this as a stall catch-up and
        trims the gap silence; capture timestamps say the gap is real, so
        every byte of it must be preserved.
        """
        p = await _make_processor()

        await self._send_user_frame(p, self._FRAME, 0)
        # Resumed audio: pts jumps 1 s, then continues contiguously. All of
        # it is delivered back-to-back (a WebSocket burst).
        for i in range(10):
            await self._send_user_frame(p, self._FRAME, 1_000_000_000 + i * self._FRAME_NS)

        one_second = self._BYTES_PER_SECOND
        self.assertEqual(len(p._user_audio_buffer), one_second + 10 * len(self._FRAME))
        silence_region = bytes(p._user_audio_buffer[len(self._FRAME) : one_second])
        self.assertTrue(all(b == 0 for b in silence_region))
        self.assertEqual(bytes(p._user_audio_buffer[one_second:]), self._FRAME * 10)
        await p.cleanup()

    async def test_pts_never_trims_when_buffer_ahead(self):
        """A duplicate/late capture timestamp appends without trimming."""
        p = await _make_processor()

        await self._send_user_frame(p, self._FRAME, 0)
        await self._send_user_frame(p, self._FRAME, 0)  # duplicate pts

        self.assertEqual(bytes(p._user_audio_buffer), self._FRAME * 2)
        await p.cleanup()

    async def test_pts_padding_is_not_trimmable(self):
        """Silence padded by capture time must not be trimmable by the gap fill."""
        p = await _make_processor()

        await self._send_user_frame(p, self._FRAME, 0)
        await self._send_user_frame(p, self._FRAME, 1_000_000_000)

        self.assertEqual(p._user_gap_tracker.silence_len, 0)
        await p.cleanup()

    async def test_gap_tracker_anchored_at_post_append_length(self):
        """The gap tracker anchor must include the frame appended after the reset.

        If it anchored at the pre-append length and capture-time metadata later
        disappeared mid-stream, the wall-clock reconciliation would start one
        frame behind the real buffer length and could over-trim injected
        silence on the next stall+burst.
        """
        p = await _make_processor()

        await self._send_user_frame(p, self._FRAME, 0)
        self.assertEqual(p._user_gap_tracker.expected_len, float(len(p._user_audio_buffer)))

        bot_frame = OutputAudioRawFrame(audio=self._FRAME, sample_rate=16000, num_channels=1)
        bot_frame.metadata["audio_capture_time_ns"] = 0
        await p.process_frame(bot_frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(p._bot_gap_tracker.expected_len, float(len(p._bot_audio_buffer)))
        await p.cleanup()

    async def test_buffer_flush_resets_capture_anchor(self):
        """After a flush the capture anchor re-anchors on the next frame.

        Without the reset, the stale anchor would place post-flush audio at
        the absolute capture offset, padding the emptied buffer with the
        whole recording's worth of silence again.
        """
        audio = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # 8 bytes == buffer_size
        p = await _make_processor(buffer_size=len(audio))

        frame = InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        frame.metadata["audio_capture_time_ns"] = 10_000_000_000  # 10 s into the stream
        await p.process_frame(frame, FrameDirection.DOWNSTREAM)  # hits buffer_size → flush
        self.assertIsNone(p._user_capture_tracker.base_capture_ns)

        # The next frame re-anchors: appended at position 0, no 10 s pad.
        # Keep it below buffer_size so it does not trigger another flush.
        small = b"\x11\x22"
        await self._send_user_frame(p, small, 10_020_000_000)
        self.assertEqual(bytes(p._user_audio_buffer), small)
        await p.cleanup()

    async def test_stop_recording_resets_capture_anchor(self):
        """stop_recording must clear the capture-timestamp anchors."""
        p = await _make_processor()
        await self._send_user_frame(p, self._FRAME, 0)
        self.assertIsNotNone(p._user_capture_tracker.base_capture_ns)

        await p.stop_recording()

        self.assertIsNone(p._user_capture_tracker.base_capture_ns)
        self.assertIsNone(p._bot_capture_tracker.base_capture_ns)
        await p.cleanup()

    async def test_bot_pts_gap_pads_silence(self):
        """Mirror of the capture-time gap padding on the bot buffer path."""
        p = await _make_processor()

        await self._send_bot_frame(p, self._FRAME, 0)
        await self._send_bot_frame(p, self._FRAME, 1_000_000_000)

        one_second = self._BYTES_PER_SECOND
        self.assertEqual(len(p._bot_audio_buffer), one_second + len(self._FRAME))
        silence_region = bytes(p._bot_audio_buffer[len(self._FRAME) : one_second])
        self.assertTrue(all(b == 0 for b in silence_region))
        self.assertEqual(bytes(p._bot_audio_buffer[one_second:]), self._FRAME)
        await p.cleanup()

    async def test_pts_none_still_uses_arrival_pacing(self):
        """Frames without pts keep the arrival-pacing gap fill unchanged."""
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0  # 1-second gap, no pts
            await p.process_frame(
                InputAudioRawFrame(audio=self._FRAME, sample_rate=16000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

        # Same result the pacing path has always produced for a 1 s gap.
        frame_duration = len(self._FRAME) / self._BYTES_PER_SECOND
        expected_silence = int((1.0 - frame_duration) * self._BYTES_PER_SECOND)
        expected_silence -= expected_silence % 2
        self.assertEqual(len(p._user_audio_buffer), expected_silence + len(self._FRAME))
        await p.cleanup()

    async def test_raw_pts_without_metadata_uses_arrival_pacing(self):
        """A frame with frame.pts set but no capture-time metadata must NOT
        take the capture-time path.

        Transports like SmallWebRTC stamp frame.pts on input audio in their
        own units (samples, not nanoseconds). Keying capture positioning off
        raw pts would (mis)route that audio through _position_buffer_by_capture_time
        and, with the wrong units, silently stop padding gaps: the #4561
        too-short regression. Only the explicit metadata key may trigger it.
        """
        p = await _make_processor()

        with patch("pipecat.processors.audio.audio_buffer_processor.time") as mock_time:
            p._last_user_buffer_update_time = 0.0
            mock_time.monotonic.return_value = 1.0  # 1-second arrival gap
            frame = InputAudioRawFrame(audio=self._FRAME, sample_rate=16000, num_channels=1)
            frame.pts = 12345  # sample-unit pts, as a WebRTC transport would set
            # No frame.metadata["audio_capture_time_ns"].
            await p.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Arrival pacing ran (silence padded for the 1 s gap) and the capture
        # tracker never anchored.
        frame_duration = len(self._FRAME) / self._BYTES_PER_SECOND
        expected_silence = int((1.0 - frame_duration) * self._BYTES_PER_SECOND)
        expected_silence -= expected_silence % 2
        self.assertEqual(len(p._user_audio_buffer), expected_silence + len(self._FRAME))
        self.assertIsNone(p._user_capture_tracker.base_capture_ns)
        await p.cleanup()

    async def test_capture_time_reset_reanchors_on_backwards_jump(self):
        """A backwards capture time (e.g. a Twilio stream restart resetting the
        timestamp to 0) re-anchors instead of stopping all future gap padding.
        """
        p = await _make_processor()

        # Anchor deep into the stream, then a later frame contiguous with it.
        await self._send_user_frame(p, self._FRAME, 10_000_000_000)
        await self._send_user_frame(p, self._FRAME, 10_000_000_000 + self._FRAME_NS)
        len_before = len(p._user_audio_buffer)

        # Stream restarts: capture time drops back to 0. This re-anchors
        # (base_len = the buffer length now, base_capture_ns = 0), so the frame
        # is simply appended: no negative offset, no wedged state.
        await self._send_user_frame(p, self._FRAME, 0)
        self.assertEqual(p._user_capture_tracker.base_capture_ns, 0)
        self.assertEqual(len(p._user_audio_buffer), len_before + len(self._FRAME))

        # A real 1 s gap after the restart still pads. The target is absolute
        # from the re-anchor (base_len == len_before), so the restart frame's
        # own audio sits inside that 1 s window: final length is base_len + 1 s
        # + this frame, not an extra frame on top.
        await self._send_user_frame(p, self._FRAME, 1_000_000_000)
        self.assertEqual(
            len(p._user_audio_buffer),
            len_before + self._BYTES_PER_SECOND + len(self._FRAME),
        )
        await p.cleanup()


if __name__ == "__main__":
    unittest.main()
