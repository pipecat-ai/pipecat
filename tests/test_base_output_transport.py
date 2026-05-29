#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from types import SimpleNamespace

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    CancelTaskFrame,
    Frame,
    OutputAudioRawFrame,
    StartFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.tests.mock_transport import MockOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class CapturingOutputTransport(MockOutputTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed_frames: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.pushed_frames.append((frame, direction))


class TestBaseOutputTransport(unittest.IsolatedAsyncioTestCase):
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
