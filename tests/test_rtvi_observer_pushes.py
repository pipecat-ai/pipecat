#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for which pushes RTVIObserver handles."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


class TestRTVIObserverPushes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams())
        self.observer.send_rtvi_message = AsyncMock()
        self.source = FrameProcessor()

    async def _push(self, frame, *, first_push=True, source=None):
        source = source or self.source
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    async def test_unhandled_frame_types_are_skipped_from_then_on(self):
        for frame_type in (InputAudioRawFrame, TTSAudioRawFrame):
            for _ in range(100):
                await self._push(frame_type(audio=b"\0" * 320, sample_rate=16000, num_channels=1))

        self.assertEqual(
            self.observer._unhandled_frame_types, {InputAudioRawFrame, TTSAudioRawFrame}
        )
        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_a_frame_is_handled_on_its_first_push_only(self):
        frame = UserStartedSpeakingFrame()

        await self._push(frame)
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_awaited_once()

    async def test_a_frame_disabled_on_its_first_push_is_never_handled(self):
        frame = VADUserStartedSpeakingFrame()

        await self._push(frame)
        self.observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_aggregated_text_is_handled_once_it_has_gone_through_the_transport(self):
        frame = AggregatedTextFrame(text="hello", aggregated_by="sentence")
        transport = BaseOutputTransport(TransportParams())

        await self._push(frame)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [])

        await self._push(frame, first_push=False, source=transport)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [frame])


if __name__ == "__main__":
    unittest.main()
