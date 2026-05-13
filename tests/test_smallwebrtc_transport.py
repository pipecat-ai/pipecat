#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SmallWebRTC transport client iterators."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from aiortc.mediastreams import MediaStreamError

from pipecat.transports.smallwebrtc.transport import (
    CAM_VIDEO_SOURCE,
    SmallWebRTCClient,
)


class TestReadAudioFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_exits_on_media_stream_error(self):
        """`read_audio_frame` must exit when the track is permanently dead.

        Regression test: previously the generator caught `MediaStreamError`,
        slept 10 ms, and re-entered the loop — but the next `recv()` raised
        the same error, so the iterator never terminated and the caller's
        receive task stayed wedged forever.
        """
        fake_self = MagicMock()
        fake_self._audio_input_track = MagicMock()
        fake_self._audio_input_track.recv = AsyncMock(
            side_effect=MediaStreamError("track ended")
        )

        iterator = SmallWebRTCClient.read_audio_frame(fake_self)

        async def consume():
            async for _ in iterator:
                pass

        # Should return promptly; before the fix this would hit the timeout.
        await asyncio.wait_for(consume(), timeout=1.0)


class TestReadVideoFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_exits_on_media_stream_error(self):
        """`read_video_frame` must exit when the track is permanently dead.

        Same regression as the audio case.
        """
        fake_self = MagicMock()
        fake_self._video_input_track = MagicMock()
        fake_self._video_input_track.recv = AsyncMock(
            side_effect=MediaStreamError("track ended")
        )

        iterator = SmallWebRTCClient.read_video_frame(fake_self, CAM_VIDEO_SOURCE)

        async def consume():
            async for _ in iterator:
                pass

        await asyncio.wait_for(consume(), timeout=1.0)


if __name__ == "__main__":
    unittest.main()
