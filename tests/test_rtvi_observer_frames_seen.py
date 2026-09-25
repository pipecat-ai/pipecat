#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how RTVIObserver remembers the frames it has handled."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams


class TestRTVIObserverFramesSeen(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams())
        self.observer.send_rtvi_message = AsyncMock()
        self.source = FrameProcessor()

    async def _push(self, frame):
        await self.observer.on_push_frame(
            FramePushed(
                source=self.source,
                destination=self.source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def test_unhandled_frames_are_not_remembered(self):
        for frame_type in (InputAudioRawFrame, TTSAudioRawFrame):
            for _ in range(100):
                await self._push(frame_type(audio=b"\0" * 320, sample_rate=16000, num_channels=1))

        self.assertEqual(self.observer._frames_seen, set())
        self.assertEqual(
            self.observer._unhandled_frame_types, {InputAudioRawFrame, TTSAudioRawFrame}
        )
        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_handled_frames_are_handled_once(self):
        frame = UserStartedSpeakingFrame()

        await self._push(frame)
        await self._push(frame)

        self.observer.send_rtvi_message.assert_awaited_once()
        self.assertEqual(self.observer._frames_seen, {frame.id})

    async def test_frame_seen_while_disabled_stays_skipped(self):
        frame = VADUserStartedSpeakingFrame()

        await self._push(frame)
        self.observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        await self._push(frame)

        self.observer.send_rtvi_message.assert_not_awaited()
        self.assertEqual(self.observer._frames_seen, {frame.id})


if __name__ == "__main__":
    unittest.main()
