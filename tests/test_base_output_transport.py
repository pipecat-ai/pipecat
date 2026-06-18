#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for BaseOutputTransport behavior."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    CancelTaskFrame,
    Frame,
    InterruptionFrame,
    MixerControlFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.tests.mock_transport import MockOutputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class CapturingOutputTransport(MockOutputTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed_frames: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.pushed_frames.append((frame, direction))


class TestBaseOutputTransportFailures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))

    async def test_non_audio_frame_does_not_reset_consecutive_audio_write_failures(self):
        transport = CapturingOutputTransport(
            TransportParams(
                audio_out_enabled=True,
                audio_out_10ms_chunks=1,
                audio_out_end_silence_secs=0,
                audio_out_max_consecutive_failures=2,
                audio_out_sleep_between_failures=0,
            ),
            audio_write_succeeds=False,
        )
        await transport.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=self.task_manager,
                pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
            )
        )

        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

        try:
            audio = b"\x01\x02" * 240
            await transport.process_frame(
                OutputAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            await transport.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            await transport.process_frame(
                OutputAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

            for _ in range(20):
                if any(isinstance(frame, CancelTaskFrame) for frame, _ in transport.pushed_frames):
                    break
                await asyncio.sleep(0.01)

            assert any(
                isinstance(frame, CancelTaskFrame) and direction == FrameDirection.UPSTREAM
                for frame, direction in transport.pushed_frames
            )
        finally:
            await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
            await transport.cleanup()


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


class TestBaseOutputTransportInterruptions(unittest.IsolatedAsyncioTestCase):
    async def _make_transport(self, mixer: BaseAudioMixer | None = None) -> BaseOutputTransport:
        params = TransportParams(audio_out_enabled=True, audio_out_mixer=mixer)
        transport = BaseOutputTransport(params)
        transport.push_frame = AsyncMock()
        transport.write_audio_frame = AsyncMock(return_value=True)

        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
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
