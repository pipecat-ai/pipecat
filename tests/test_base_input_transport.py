#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame-based audio handling in :class:`BaseInputTransport`."""

import asyncio
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import AUDIO_INPUT_TIMEOUT_SECS, BaseInputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


async def _make_ready_input_transport() -> BaseInputTransport:
    """Start an input transport with its audio task running."""
    params = TransportParams(audio_in_enabled=True, audio_in_passthrough=True)
    transport = BaseInputTransport(params)
    transport.push_frame = AsyncMock()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await transport.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    start_frame = StartFrame(audio_in_sample_rate=16000)
    await transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await transport.set_transport_ready(start_frame)
    return transport


class TestBaseInputTransportFrameAudio(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> BaseInputTransport:
        return BaseInputTransport(TransportParams(audio_in_enabled=True))

    async def test_incoming_audio_frame_routed_to_push_audio_frame(self):
        transport = self._transport()
        transport.push_audio_frame = AsyncMock()
        transport.push_frame = AsyncMock()
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Fed into the VAD path, not forwarded as a plain frame.
        transport.push_audio_frame.assert_called_once_with(frame)

    async def test_start_audio_streaming_frame_triggers_streaming(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        await transport.process_frame(
            InputTransportStartAudioStreamingFrame(), FrameDirection.DOWNSTREAM
        )
        transport._start_audio_in_streaming.assert_called_once()

    async def test_start_audio_in_streaming_method_is_deprecated(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await transport.start_audio_in_streaming()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        transport._start_audio_in_streaming.assert_called_once()


class TestBaseInputTransportAudioStallSignal(unittest.IsolatedAsyncioTestCase):
    async def test_no_stall_warning_before_first_audio(self):
        transport = await _make_ready_input_transport()
        try:
            with patch("pipecat.transports.base_input.logger") as mock_logger:
                await asyncio.sleep(AUDIO_INPUT_TIMEOUT_SECS * 2.5)
                mock_logger.warning.assert_not_called()
        finally:
            await transport.cancel(CancelFrame())

    async def test_stall_warns_once_then_recovery(self):
        transport = await _make_ready_input_transport()
        try:
            frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
            await transport.push_audio_frame(frame)
            # Let the audio task consume the first frame.
            await asyncio.sleep(0.05)

            with patch("pipecat.transports.base_input.logger") as mock_logger:
                await asyncio.sleep(AUDIO_INPUT_TIMEOUT_SECS * 3.5)

                stall_msgs = [
                    call.args[0]
                    for call in mock_logger.warning.call_args_list
                    if "may have stalled" in call.args[0]
                ]
                self.assertEqual(len(stall_msgs), 1)

                await transport.push_audio_frame(frame)
                await asyncio.sleep(0.1)

                recovery_msgs = [
                    call.args[0]
                    for call in mock_logger.warning.call_args_list
                    if "recovered after" in call.args[0]
                ]
                self.assertEqual(len(recovery_msgs), 1)
        finally:
            await transport.cancel(CancelFrame())


if __name__ == "__main__":
    unittest.main()
