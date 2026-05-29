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


async def _make_processor(*, buffer_size: int = 0) -> AudioBufferProcessor:
    """Create and start a processor ready to record.

    Calls setup() and sends a StartFrame through the public process_frame path so that
    the processor is fully initialised (task manager set, sample rate configured,
    __started flag set) without needing a full pipeline.
    """
    processor = AudioBufferProcessor(sample_rate=16000, num_channels=2, buffer_size=buffer_size)
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


if __name__ == "__main__":
    unittest.main()
