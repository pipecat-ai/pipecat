#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`BotRecordingSilencePadder`.

These tests verify that when no user input audio is flowing, inserting
``BotRecordingSilencePadder`` upstream of ``AudioBufferProcessor`` keeps the
bot recording buffer growing in sync with wall-clock time, so consecutive bot
utterances are not concatenated in the recording.
"""

import asyncio
import unittest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.audio.bot_recording_silence_padder import BotRecordingSilencePadder
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


SAMPLE_RATE = 16000
UTTERANCE_BYTES = SAMPLE_RATE * 2  # 1s of 16-bit mono PCM


class _PassthroughResampler:
    async def resample(
        self, audio: bytes, in_rate: int, out_rate: int
    ) -> bytes:  # pragma: no cover - trivial
        return audio


async def _setup_chain(*, chunk_ms: int = 100):
    """Wire BotRecordingSilencePadder -> AudioBufferProcessor.

    The padder's downstream pushes are routed into the buffer processor via the
    padder's ``_next`` attribute so the test does not require a full pipeline.
    """
    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=task_manager)

    buffer = AudioBufferProcessor(sample_rate=SAMPLE_RATE)
    buffer._input_resampler = _PassthroughResampler()
    buffer._output_resampler = _PassthroughResampler()
    await buffer.setup(setup)

    padder = BotRecordingSilencePadder(chunk_ms=chunk_ms)
    await padder.setup(setup)
    padder.link(buffer)

    start = StartFrame(audio_out_sample_rate=SAMPLE_RATE)
    await padder.process_frame(start, FrameDirection.DOWNSTREAM)
    await buffer.start_recording()

    return padder, buffer


async def _speak(padder: BotRecordingSilencePadder, audio: bytes):
    """Drive a bot utterance through the padder."""
    await padder.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await padder.process_frame(
        OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await padder.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


class TestBotRecordingSilencePadder(unittest.IsolatedAsyncioTestCase):
    async def test_pads_silence_between_bot_utterances(self):
        """Two utterances separated by a wall-clock gap should preserve the gap.

        Without the padder, AudioBufferProcessor's bot buffer would only contain
        the two utterances back-to-back (2s). With the padder, the buffer should
        also contain ~3s of silence between them.
        """
        padder, buffer = await _setup_chain(chunk_ms=50)
        try:
            utt = b"\x10\x00" * SAMPLE_RATE  # 1s of low-amplitude tone-ish PCM
            await _speak(padder, utt)

            # Wall-clock gap. No InputAudioRawFrame and no OutputAudioRawFrame
            # flow through the pipeline during this window. The padder should
            # emit silent OutputAudioRawFrame chunks so the bot buffer keeps
            # advancing.
            await asyncio.sleep(0.6)

            await _speak(padder, utt)
            # Let the loop emit a final couple of chunks after the second
            # utterance ends so the test isn't sensitive to scheduling.
            await asyncio.sleep(0.1)

            bot_bytes = len(buffer._bot_audio_buffer)
            # 2 utterances = 2s. Gap >= 0.6s would have been silently dropped
            # without the padder. Allow generous tolerance for scheduling.
            self.assertGreaterEqual(
                bot_bytes,
                int(2.4 * UTTERANCE_BYTES),
                f"expected >= 2.4s of bot audio, got {bot_bytes / UTTERANCE_BYTES:.2f}s",
            )
        finally:
            if getattr(buffer, "_recording", False):
                await buffer.stop_recording()
            await padder.cleanup()
            await buffer.cleanup()

    async def test_silence_stops_during_real_bot_utterance(self):
        """Padder should pause its silence loop while the bot is speaking.

        It only resumes once BotStoppedSpeakingFrame is observed. We check that
        an utterance is recorded contiguously by inspecting the buffer length
        immediately after the utterance is pushed.
        """
        padder, buffer = await _setup_chain(chunk_ms=50)
        try:
            # Let the loop pad for a moment.
            await asyncio.sleep(0.1)
            pre_len = len(buffer._bot_audio_buffer)

            utt = b"\x10\x00" * SAMPLE_RATE
            await padder.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            # Once we declare the bot is speaking, the padder should stop
            # emitting silence even if some wall-clock passes.
            await asyncio.sleep(0.1)
            mid_len = len(buffer._bot_audio_buffer)
            await padder.process_frame(
                OutputAudioRawFrame(audio=utt, sample_rate=SAMPLE_RATE, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            await padder.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

            self.assertGreater(pre_len, 0, "padder should have emitted silence before utterance")
            self.assertEqual(
                mid_len,
                pre_len,
                "padder should NOT emit silence while bot is speaking",
            )
        finally:
            if getattr(buffer, "_recording", False):
                await buffer.stop_recording()
            await padder.cleanup()
            await buffer.cleanup()


if __name__ == "__main__":
    unittest.main()
