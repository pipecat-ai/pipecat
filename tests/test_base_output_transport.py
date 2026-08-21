#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for interruption handling in :class:`BaseOutputTransport`."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    InterruptionFrame,
    MixerControlFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager


class _PassthroughMixer(BaseAudioMixer):
    """Minimal mixer that returns the input audio unchanged."""

    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame: MixerControlFrame):
        pass

    async def mix(self, audio: bytes) -> bytes:
        return audio


async def _make_transport(
    mixer: BaseAudioMixer | None = None, audio_out_sample_rate: int | None = None
) -> BaseOutputTransport:
    params = TransportParams(
        audio_out_enabled=True,
        audio_out_mixer=mixer,
        audio_out_sample_rate=audio_out_sample_rate,
    )
    transport = BaseOutputTransport(params)
    transport.push_frame = AsyncMock()
    transport.write_audio_frame = AsyncMock(return_value=True)

    task_manager = TaskManager()
    await transport.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    start_frame = StartFrame(audio_out_sample_rate=16000)
    await transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await transport.set_transport_ready(start_frame)
    return transport


class TestBaseOutputTransportInterruptions(unittest.IsolatedAsyncioTestCase):
    async def _make_transport(self, mixer: BaseAudioMixer | None = None) -> BaseOutputTransport:
        return await _make_transport(mixer)

    async def test_interruption_with_mixer_keeps_audio_task_and_mixer_output(self):
        transport = await self._make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]
            task_before = sender._audio_task
            self.assertIsNotNone(task_before)

            # Mixer-only frames flow while the queue is empty.
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, 0)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # Same task object: not cancelled and not recreated.
            self.assertIs(sender._audio_task, task_before)
            self.assertFalse(task_before.cancelled())

            # Mixer frames keep flowing across the interruption.
            count_after_interruption = transport.write_audio_frame.call_count
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, count_after_interruption)
        finally:
            await transport.cancel(CancelFrame())

    async def test_interruption_without_mixer_recreates_audio_task(self):
        transport = await self._make_transport(mixer=None)
        try:
            sender = transport._media_senders[None]
            task_before = sender._audio_task
            self.assertIsNotNone(task_before)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            self.assertIsNot(sender._audio_task, task_before)
            self.assertIsNotNone(sender._audio_task)
        finally:
            await transport.cancel(CancelFrame())

    async def test_interruption_with_mixer_still_discards_queued_bot_audio(self):
        transport = await self._make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]

            # Pause the audio task by patching write_audio_frame with a slow
            # write, so the queued bot audio can't be consumed before the
            # interruption arrives.
            write_started = asyncio.Event()
            release_write = asyncio.Event()

            async def slow_write(frame):
                write_started.set()
                await release_write.wait()
                return True

            transport.write_audio_frame = AsyncMock(side_effect=slow_write)
            await write_started.wait()

            # Queue bot audio (one full chunk) behind the in-flight write.
            bot_audio = OutputAudioRawFrame(
                audio=b"\x01\x02" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(bot_audio, FrameDirection.DOWNSTREAM)
            self.assertFalse(sender._audio_queue.empty())

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # The queued bot audio was dropped by the reset.
            self.assertTrue(sender._audio_queue.empty())
            release_write.set()
        finally:
            await transport.cancel(CancelFrame())


class TestBaseOutputTransportAudioBuffering(unittest.IsolatedAsyncioTestCase):
    """Test for the trailing-partial-chunk audio buffer.

    ``MediaSender._audio_buffer`` only enqueues complete ``audio_chunk_size``
    chunks (see ``handle_audio_frame``); whatever hasn't reached a full chunk
    stays buffered. When ``TTSStoppedFrame`` arrives, that leftover audio is
    padded with silence to a full chunk and queued for playback (see
    ``handle_tts_stopped``), instead of being silently discarded.
    """

    async def test_tts_stopped_frame_flushes_partial_chunk_padded_with_silence(self):
        transport = await _make_transport(mixer=None)
        try:
            sender = transport._media_senders[None]
            chunk_size = sender.audio_chunk_size

            # A full chunk: gets queued and played immediately, and marks the
            # bot as speaking so `_bot_stopped_speaking` won't just no-op.
            full_audio = b"\x01\x02" * (chunk_size // 2)
            full_chunk = TTSAudioRawFrame(
                audio=full_audio,
                sample_rate=sender.sample_rate,
                num_channels=1,
                context_id="ctx1",
            )
            await transport.process_frame(full_chunk, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)
            self.assertTrue(sender._bot_speaking)

            written = b"".join(
                call.args[0].audio for call in transport.write_audio_frame.call_args_list
            )
            self.assertEqual(written, full_audio)

            # Trailing audio that doesn't fill up a whole chunk, like the last
            # bit of a TTS turn typically would.
            partial_len = chunk_size // 2
            partial_audio = b"\x03\x04" * (partial_len // 2)
            partial_chunk = TTSAudioRawFrame(
                audio=partial_audio,
                sample_rate=sender.sample_rate,
                num_channels=1,
                context_id="ctx1",
            )
            await transport.process_frame(partial_chunk, FrameDirection.DOWNSTREAM)

            # It's sitting in the buffer, not yet queued for playback.
            self.assertEqual(len(sender._audio_buffer), partial_len)

            # TTSStoppedFrame marks the end of the turn: the leftover audio
            # should be padded with silence and flushed, not discarded.
            await transport.process_frame(
                TTSStoppedFrame(context_id="ctx1"), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0.1)

            # Everything written to the transport across both calls: the
            # first full chunk, followed by the flushed partial chunk once
            # it's been padded out to `chunk_size` with silence.
            written = b"".join(
                call.args[0].audio for call in transport.write_audio_frame.call_args_list
            )
            silence_padding = b"\x00" * (chunk_size - partial_len)
            expected = full_audio + partial_audio + silence_padding
            self.assertEqual(written, expected)
            # The buffer should be drained by the flush, not just cleared.
            self.assertEqual(sender._audio_buffer, bytearray())
        finally:
            await transport.cancel(CancelFrame())

    async def test_tts_stopped_frame_for_short_turn_signals_bot_speaking(self):
        """A turn whose entire audio never fills one chunk must still flush
        as the frame type it was buffered from (e.g. `TTSAudioRawFrame`), so
        that bot-speaking tracking (which dispatches on frame type) still
        fires for it, instead of silently skipping start/stop speaking
        events.
        """
        transport = await _make_transport(mixer=None)
        try:
            sender = transport._media_senders[None]
            chunk_size = sender.audio_chunk_size

            # Audio shorter than a single chunk: never queued by
            # `handle_audio_frame`, only ever sitting in `_audio_buffer`.
            partial_audio = b"\x03\x04" * (chunk_size // 4)
            partial_chunk = TTSAudioRawFrame(
                audio=partial_audio,
                sample_rate=sender.sample_rate,
                num_channels=1,
                context_id="ctx1",
            )
            await transport.process_frame(partial_chunk, FrameDirection.DOWNSTREAM)
            self.assertEqual(len(sender._audio_buffer), len(partial_audio))
            self.assertFalse(sender._bot_speaking)

            await transport.process_frame(
                TTSStoppedFrame(context_id="ctx1"), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0.1)

            # The flushed frame must be written as the buffered frame's
            # original type, not a generic `OutputAudioRawFrame`, so that bot
            # speaking tracking (which dispatches on frame type) recognizes it.
            written_frame = transport.write_audio_frame.call_args_list[0].args[0]
            self.assertIsInstance(written_frame, TTSAudioRawFrame)
            silence_padding = b"\x00" * (chunk_size - len(partial_audio))
            self.assertEqual(written_frame.audio, partial_audio + silence_padding)

            # Bot started and stopped speaking events must have fired even
            # though no full chunk was ever queued for this turn.
            pushed_types = [call.args[0].__class__ for call in transport.push_frame.call_args_list]
            self.assertIn(BotStartedSpeakingFrame, pushed_types)
            self.assertIn(BotStoppedSpeakingFrame, pushed_types)
        finally:
            await transport.cancel(CancelFrame())


class TestBaseOutputTransportResampling(unittest.IsolatedAsyncioTestCase):
    """Tests for audio that has to be resampled to the transport's output rate.

    ``MediaSender`` resamples on arrival, ahead of the paced ``_audio_queue``,
    so it sits idle whenever TTS delivers audio in bursts. Its resampler must
    therefore keep the audio held in its filter across those pauses, and give
    it up only when the speech run actually ends.
    """

    TTS_RATE = 16000
    OUT_RATE = 8000

    def _tts_frames(self, sample_count: int, chunk_samples: int = 640):
        """Split a ramp of TTS-rate audio into frames, as a TTS service would."""
        audio = bytes(
            b for i in range(sample_count) for b in (i % 251).to_bytes(2, "little", signed=False)
        )
        chunk = chunk_samples * 2
        return [
            TTSAudioRawFrame(
                audio=audio[offset : offset + chunk],
                sample_rate=self.TTS_RATE,
                num_channels=1,
                context_id="ctx1",
            )
            for offset in range(0, len(audio), chunk)
        ]

    def _constant_frames(self, value: int, sample_count: int, chunk_samples: int = 640):
        """TTS frames holding a constant level, so leftovers are easy to spot."""
        audio = int(value).to_bytes(2, "little", signed=True) * sample_count
        chunk = chunk_samples * 2
        return [
            TTSAudioRawFrame(
                audio=audio[offset : offset + chunk],
                sample_rate=self.TTS_RATE,
                num_channels=1,
                context_id="ctx1",
            )
            for offset in range(0, len(audio), chunk)
        ]

    def _written(self, transport) -> bytes:
        return b"".join(call.args[0].audio for call in transport.write_audio_frame.call_args_list)

    async def test_delivery_pause_does_not_drop_audio(self):
        """A pause between TTS chunks must not cost any audio.

        The resampler runs ahead of playback, so a gap between two chunks says
        nothing about the stream: the audio either side of it is one utterance,
        and all of it has to reach the transport.
        """
        transport = await _make_transport(audio_out_sample_rate=self.OUT_RATE)
        try:
            sender = transport._media_senders[None]
            in_samples = 16000

            for index, frame in enumerate(self._tts_frames(sample_count=in_samples)):
                if index == 12:
                    # A delivery pause longer than the resampler's inactivity
                    # timeout, e.g. a TTS round trip between two sentences.
                    await asyncio.sleep(0.3)
                await transport.process_frame(frame, FrameDirection.DOWNSTREAM)

            await transport.process_frame(
                TTSStoppedFrame(context_id="ctx1"), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0.2)

            # Every input sample reaches the transport, give or take the
            # silence the last partial chunk is padded with.
            expected = in_samples * self.OUT_RATE // self.TTS_RATE
            written = len(self._written(transport)) // 2
            self.assertGreaterEqual(written, expected)
            self.assertLess(written, expected + sender.audio_chunk_size // 2)
        finally:
            await transport.cancel(CancelFrame())

    async def test_interruption_does_not_replay_aborted_audio(self):
        """Audio cut short is dropped, not carried into what the bot says next."""
        transport = await _make_transport(audio_out_sample_rate=self.OUT_RATE)
        try:
            sender = transport._media_senders[None]

            for frame in self._constant_frames(8000, 8000):
                await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)
            self.assertTrue(sender._bot_speaking)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)
            already_written = len(transport.write_audio_frame.call_args_list)

            # The next turn is entirely negative, so anything positive in it is
            # left over from the turn that was cut short.
            for frame in self._constant_frames(-8000, 8000):
                await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
            await transport.process_frame(
                TTSStoppedFrame(context_id="ctx1"), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0.2)

            after = b"".join(
                call.args[0].audio
                for call in transport.write_audio_frame.call_args_list[already_written:]
            )
            samples = np.frombuffer(after, dtype=np.int16)
            self.assertGreater(len(samples), 0)
            self.assertLessEqual(int(samples.max()), 0)
        finally:
            await transport.cancel(CancelFrame())
