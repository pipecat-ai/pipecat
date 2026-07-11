#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LiveKit transport video stream handling.

Regression tests for issue #3116: Memory leak when video_in_enabled=False
but video tracks are subscribed. The fix ensures video stream processing
only starts when there is a consumer for the frames.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from livekit import rtc

    from pipecat.transports.livekit.transport import (
        LiveKitCallbacks,
        LiveKitParams,
        LiveKitTransportClient,
    )

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitVideoStreamMemoryLeak(unittest.IsolatedAsyncioTestCase):
    """Regression tests for video queue memory leak (#3116).

    The bug: When video_in_enabled=False, subscribing to a video track would
    start a producer that fills _video_queue, but no consumer would drain it,
    causing unbounded memory growth (~3GB/min).

    The fix: Only start video stream processing when video_in_enabled=True.
    """

    def _create_client(self, video_in_enabled: bool) -> LiveKitTransportClient:
        """Create a client with the specified video input setting."""
        params = LiveKitParams(video_in_enabled=video_in_enabled)
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        return client

    def _create_mock_video_track(self):
        """Create a mock video track subscription event."""
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_VIDEO
        track.sid = "video-track-123"
        publication = MagicMock()
        participant = MagicMock()
        participant.sid = "participant-456"
        return track, publication, participant

    async def test_disabled_video_input_does_not_start_queue_producer(self):
        """When video input is disabled, no producer should fill the queue.

        This prevents the memory leak where frames accumulate with no consumer.
        """
        client = self._create_client(video_in_enabled=False)
        track, publication, participant = self._create_mock_video_track()

        await client._async_on_track_subscribed(track, publication, participant)

        # Verify no video processing task was started
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [name for name in task_names if "video" in name.lower()]
        self.assertEqual(video_tasks, [], "No video processing task should be started")

        # Queue should remain empty
        self.assertEqual(client._video_queue.qsize(), 0)

        # Track metadata should still be recorded
        self.assertIn(participant.sid, client._video_tracks)

        # Callback should still fire for user code
        client._callbacks.on_video_track_subscribed.assert_called_once()

    async def test_enabled_video_input_starts_queue_producer(self):
        """When video input is enabled, the producer should start."""
        client = self._create_client(video_in_enabled=True)
        track, publication, participant = self._create_mock_video_track()

        with patch.object(rtc, "VideoStream"):
            await client._async_on_track_subscribed(track, publication, participant)

        # Verify video processing task was started
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [name for name in task_names if "video" in name.lower()]
        self.assertEqual(len(video_tasks), 1, "Video processing task should be started")

        # Track metadata should be recorded
        self.assertIn(participant.sid, client._video_tracks)

        # Callback should fire
        client._callbacks.on_video_track_subscribed.assert_called_once()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAudioStreamLeakOnUnsubscribe(unittest.IsolatedAsyncioTestCase):
    """Regression tests for AudioStream leak on track unsubscribe.

    The bug: ``_async_on_track_subscribed`` creates an owned ``rtc.AudioStream``
    plus a ``_process_audio_stream`` task feeding the shared ``_audio_queue``,
    but only saves the track. ``_async_on_track_unsubscribed`` never closes the
    stream or cancels the task. Per livekit-rtc ``audio_stream.py``, an owned
    ``AudioStream._run`` loops over the FFI queue and exits only on a native
    ``eos`` event emitted by ``aclose()`` → ``_ffi_handle.dispose()``. So when a
    participant republishes their mic (e.g. mute/unmute), the previous stream
    keeps pushing frames forever; N republishes → N concurrent producers
    interleave audio into the shared queue and downstream STT receives garbage.

    The fix: store ``(stream, task)`` per ``participant.sid`` in
    ``_audio_streams`` on subscribe, then ``aclose()`` + cancel on unsubscribe
    and again on a re-subscribe for the same sid (to handle missed unsubscribe).
    Symmetric for video.
    """

    def _create_client(self, video_in_enabled: bool = False) -> LiveKitTransportClient:
        params = LiveKitParams(video_in_enabled=video_in_enabled)
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()

        # Return a real (mockable) Task so the cleanup path can call ``.done()``
        # and ``.cancel()`` on it without blowing up.
        def _make_task(coro, name):
            coro.close()  # we never run the producer in the unit test
            t = MagicMock()
            t.done.return_value = False
            t.cancel = MagicMock()
            return t

        client._task_manager.create_task.side_effect = _make_task
        return client

    def _audio_track(self, sid: str = "audio-track-1", participant_sid: str = "p-1"):
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_AUDIO
        track.sid = sid
        publication = MagicMock()
        publication.sid = sid
        participant = MagicMock()
        participant.sid = participant_sid
        participant.identity = "user"
        return track, publication, participant

    def _video_track(self, sid: str = "video-track-1", participant_sid: str = "p-1"):
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_VIDEO
        track.sid = sid
        publication = MagicMock()
        publication.sid = sid
        participant = MagicMock()
        participant.sid = participant_sid
        participant.identity = "user"
        return track, publication, participant

    async def test_audio_stream_registered_on_subscribe(self):
        """Subscribing an audio track registers ``(stream, task)`` for the sid."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "AudioStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)

        self.assertIn(participant.sid, client._audio_streams)
        stream, task = client._audio_streams[participant.sid]
        self.assertIs(stream, mock_stream)
        self.assertIsNotNone(task)

    async def test_audio_stream_closed_and_task_cancelled_on_unsubscribe(self):
        """Unsubscribing closes the stream, cancels the task, clears the registry."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "AudioStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)
        _, task = client._audio_streams[participant.sid]

        await client._async_on_track_unsubscribed(track, pub, participant)

        mock_stream.aclose.assert_awaited_once()
        task.cancel.assert_called_once()
        self.assertNotIn(participant.sid, client._audio_streams)
        client._callbacks.on_audio_track_unsubscribed.assert_called_once()

    async def test_resubscribe_closes_previous_audio_stream(self):
        """Re-subscribing the same sid (mic republish) closes the prior stream."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        first = MagicMock()
        first.aclose = AsyncMock()
        second = MagicMock()
        second.aclose = AsyncMock()

        with patch.object(rtc, "AudioStream", return_value=first):
            await client._async_on_track_subscribed(track, pub, participant)
        first_task = client._audio_streams[participant.sid][1]

        # Republish without an explicit unsubscribe in between.
        with patch.object(rtc, "AudioStream", return_value=second):
            await client._async_on_track_subscribed(track, pub, participant)

        first.aclose.assert_awaited_once()
        first_task.cancel.assert_called_once()
        self.assertIs(client._audio_streams[participant.sid][0], second)

    async def test_unsubscribe_without_subscribe_is_noop(self):
        """Unsubscribe for an unknown sid does not raise."""
        client = self._create_client()
        track, pub, participant = self._audio_track()
        # No subscribe before this call.
        await client._async_on_track_unsubscribed(track, pub, participant)
        client._callbacks.on_audio_track_unsubscribed.assert_called_once()

    async def test_video_stream_closed_on_unsubscribe(self):
        """Symmetric behaviour for video when ``video_in_enabled=True``."""
        client = self._create_client(video_in_enabled=True)
        track, pub, participant = self._video_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "VideoStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)
        self.assertIn(participant.sid, client._video_streams)

        await client._async_on_track_unsubscribed(track, pub, participant)
        mock_stream.aclose.assert_awaited_once()
        self.assertNotIn(participant.sid, client._video_streams)


if __name__ == "__main__":
    unittest.main()
