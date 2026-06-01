#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SmallWebRTC transport client iterators.

Covers the `MediaStreamError` handling in
`SmallWebRTCClient.read_audio_frame` and `read_video_frame`:

1. **Park on dead track** — when the underlying aiortc track is permanently
   raising `MediaStreamError`, the iterator must stop calling `recv()` on it
   (clear the track reference) so we don't busy-loop a CPU core. Without the
   fix, the loop hits `recv()` ~100 times per second indefinitely.

2. **Renegotiation resumes** — after the dead track is replaced by a fresh
   one (the same mechanism `_handle_client_connected` uses), the iterator
   must pick up frames from the new track. A plain `break` on
   `MediaStreamError` would terminate the iterator and regress this path.
"""

import asyncio
import fractions
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# The `webrtc` extra is optional; skip the whole module when it (and its
# transitive `av` dependency) is unavailable, matching the default CI unit
# test environment which does not install extras.
pytest.importorskip("aiortc")
pytest.importorskip("av")

from aiortc.mediastreams import MediaStreamError  # noqa: E402
from av import AudioFrame, VideoFrame  # noqa: E402

from pipecat.transports.smallwebrtc.transport import (  # noqa: E402
    CAM_VIDEO_SOURCE,
    SCREEN_VIDEO_SOURCE,
    SmallWebRTCClient,
)


def _make_audio_self(track):
    fake = MagicMock()
    fake._audio_input_track = track
    fake._webrtc_connection = MagicMock()
    fake._webrtc_connection.is_connected.return_value = True
    fake._in_sample_rate = 16_000
    fake._audio_in_channels = 1
    # Passthrough resampler.
    fake._audio_in_resampler.resample.side_effect = lambda f: [f]
    return fake


def _make_video_self(video_track=None, screen_track=None):
    fake = MagicMock()
    fake._video_input_track = video_track
    fake._screen_video_track = screen_track
    fake._webrtc_connection = MagicMock()
    fake._webrtc_connection.is_connected.return_value = True
    fake._webrtc_connection.pc_id = "test-pc"
    fake._convert_frame.side_effect = lambda arr, fmt: arr
    return fake


def _good_audio_frame():
    samples = 320  # 20 ms @ 16 kHz
    arr = np.zeros((1, samples), dtype=np.int16)
    f = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    f.sample_rate = 16_000
    f.pts = 0
    f.time_base = fractions.Fraction(1, 16_000)
    return f


def _good_video_frame():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    f = VideoFrame.from_ndarray(arr, format="rgb24")
    f.pts = 0
    return f


class TestReadAudioFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_parks_on_dead_track(self):
        """Dead track: iterator must null the track ref and stop calling recv().

        Without the fix this loop calls `track.recv()` ~100Hz forever, pinning
        a CPU core. With the fix, `_audio_input_track` is set to None on the
        first `MediaStreamError` and the loop parks on the `is None` gate.
        """
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_audio_self(track)

        async def consume():
            async for _ in SmallWebRTCClient.read_audio_frame(fake):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        # Exactly one recv() call: after MediaStreamError, the track ref is
        # cleared and the loop sleeps on `is None` instead of re-calling recv.
        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._audio_input_track)

    async def test_renegotiation_resumes(self):
        """After the dead track is replaced, the iterator must yield frames.

        This is the renegotiation path: a plain `break` on `MediaStreamError`
        would terminate the generator. The track-nulling fix lets the
        existing `is None: sleep; continue` gate wait for a fresh track from
        `_handle_client_connected`.
        """
        dead = MagicMock()
        dead.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fresh = MagicMock()
        fresh.recv = AsyncMock(return_value=_good_audio_frame())
        fake = _make_audio_self(dead)

        yielded = 0

        async def consume():
            nonlocal yielded
            async for _ in SmallWebRTCClient.read_audio_frame(fake):
                yielded += 1
                if yielded >= 3:
                    break

        task = asyncio.create_task(consume())
        # Let the dead track raise + the loop park on `is None`.
        await asyncio.sleep(0.05)
        # Simulate _handle_client_connected reassigning a fresh track.
        fake._audio_input_track = fresh
        await asyncio.wait_for(task, timeout=1.0)

        self.assertEqual(dead.recv.await_count, 1)
        self.assertGreaterEqual(yielded, 3)


class TestReadVideoFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_camera_parks_on_dead_track(self):
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_video_self(video_track=track)

        async def consume():
            async for _ in SmallWebRTCClient.read_video_frame(fake, CAM_VIDEO_SOURCE):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._video_input_track)

    async def test_screen_parks_on_dead_track(self):
        """Screen-share uses a separate track reference."""
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_video_self(screen_track=track)

        async def consume():
            async for _ in SmallWebRTCClient.read_video_frame(fake, SCREEN_VIDEO_SOURCE):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._screen_video_track)

    async def test_camera_renegotiation_resumes(self):
        dead = MagicMock()
        dead.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fresh = MagicMock()
        fresh.recv = AsyncMock(return_value=_good_video_frame())
        fake = _make_video_self(video_track=dead)

        yielded = 0

        async def consume():
            nonlocal yielded
            async for _ in SmallWebRTCClient.read_video_frame(fake, CAM_VIDEO_SOURCE):
                yielded += 1
                if yielded >= 2:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        fake._video_input_track = fresh
        await asyncio.wait_for(task, timeout=1.0)

        self.assertEqual(dead.recv.await_count, 1)
        self.assertGreaterEqual(yielded, 2)


if __name__ == "__main__":
    unittest.main()
