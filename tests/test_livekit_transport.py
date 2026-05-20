#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LiveKit transport behavior.

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
class TestLiveKitTransportClient(unittest.IsolatedAsyncioTestCase):
    """Tests for the LiveKit transport client.

    Includes regression tests for the video queue memory leak (#3116). When
    video_in_enabled=False, subscribing to a video track would start a producer
    that fills _video_queue, but no consumer would drain it, causing unbounded
    memory growth.

    The fix was to only start video stream processing when video_in_enabled=True.
    """

    def _create_client(
        self, video_in_enabled: bool, single_peer_connection: bool | None = None
    ) -> "LiveKitTransportClient":
        """Create a client with the specified video input setting."""
        params = LiveKitParams(
            video_in_enabled=video_in_enabled,
            single_peer_connection=single_peer_connection,
        )
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
            on_track_subscription_failed=AsyncMock(),
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
        client._task_manager.create_task.side_effect = lambda coro, _name: coro.close()
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

    async def test_single_peer_connection_is_passed_to_room_options(self):
        """When configured, single_peer_connection should be forwarded to LiveKit."""
        out_sample_rate = 16000
        local_participant_id = "local-participant"

        client = self._create_client(video_in_enabled=False, single_peer_connection=True)
        client._out_sample_rate = out_sample_rate

        room = MagicMock()
        room.connect = AsyncMock()
        room.local_participant.sid = local_participant_id
        room.local_participant.publish_track = AsyncMock()
        room.remote_participants = {}
        client._room = room

        with (
            patch.object(rtc, "AudioSource"),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
            patch.object(rtc, "TrackPublishOptions", return_value=MagicMock()),
        ):
            await client.connect()

        room_options = room.connect.call_args.kwargs["options"]
        self.assertTrue(room_options.auto_subscribe)
        self.assertTrue(room_options.single_peer_connection)

    async def test_track_subscription_failed_callback_fires(self):
        """Track subscription failures should be exposed through transport callbacks."""
        participant_id = "participant-456"
        track_sid = "track-123"
        error = "track not bound"

        client = self._create_client(video_in_enabled=False)
        participant = MagicMock()
        participant.sid = participant_id

        await client._async_on_track_subscription_failed(participant, track_sid, error)

        client._callbacks.on_track_subscription_failed.assert_awaited_once_with(
            participant_id, track_sid, error
        )


if __name__ == "__main__":
    unittest.main()
