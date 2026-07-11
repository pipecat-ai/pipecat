#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame-based audio handling in :class:`BaseInputTransport`."""

import unittest
import warnings
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


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


if __name__ == "__main__":
    unittest.main()
