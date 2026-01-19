#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LiveKit transport video stream handling.

Regression tests for issue #3116: Memory leak when video_in_enabled=False
but video tracks are subscribed.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from livekit import rtc

from pipecat.transports.livekit.transport import (
    LiveKitCallbacks,
    LiveKitParams,
    LiveKitTransportClient,
)


class TestLiveKitTransportClient(unittest.IsolatedAsyncioTestCase):
    """Tests for LiveKitTransportClient video stream handling."""

    def _create_client(self, video_in_enabled: bool) -> LiveKitTransportClient:
        """Create a LiveKitTransportClient with the specified video_in_enabled setting."""
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

        # Mock the task manager
        client._task_manager = MagicMock()
        client._task_manager.create_task = MagicMock()

        return client

    def _create_mock_video_track(self) -> tuple:
        """Create mock video track, publication, and participant."""
        mock_track = MagicMock()
        mock_track.kind = rtc.TrackKind.KIND_VIDEO
        mock_track.sid = "test-track-sid"

        mock_publication = MagicMock()

        mock_participant = MagicMock()
        mock_participant.sid = "test-participant-sid"

        return mock_track, mock_publication, mock_participant

    def _was_video_stream_task_created(self, client: LiveKitTransportClient) -> bool:
        """Check if _process_video_stream task was created."""
        for call in client._task_manager.create_task.call_args_list:
            task_name = call[0][1] if len(call[0]) > 1 else call[1].get("name", "")
            if "_process_video_stream" in task_name:
                return True
        return False

    async def test_video_stream_not_started_when_video_in_disabled(self):
        """Test that _process_video_stream is NOT started when video_in_enabled=False.

        This prevents unbounded queue growth when there is no consumer for video frames.
        Regression test for issue #3116.
        """
        client = self._create_client(video_in_enabled=False)
        mock_track, mock_publication, mock_participant = self._create_mock_video_track()

        # Call the track subscribed handler
        await client._async_on_track_subscribed(mock_track, mock_publication, mock_participant)

        # Verify that create_task was NOT called for video stream processing
        self.assertFalse(
            self._was_video_stream_task_created(client),
            "Video stream processing should NOT be started when video_in_enabled=False",
        )

        # Verify that the callback was still called
        client._callbacks.on_video_track_subscribed.assert_called_once_with(mock_participant.sid)

        # Verify that the track was still added to _video_tracks
        self.assertIn(mock_participant.sid, client._video_tracks)

    async def test_video_stream_started_when_video_in_enabled(self):
        """Test that _process_video_stream IS started when video_in_enabled=True."""
        from unittest.mock import patch

        client = self._create_client(video_in_enabled=True)
        mock_track, mock_publication, mock_participant = self._create_mock_video_track()

        # Mock rtc.VideoStream to avoid needing a real LiveKit connection
        with patch("pipecat.transports.livekit.transport.rtc.VideoStream"):
            # Call the track subscribed handler
            await client._async_on_track_subscribed(mock_track, mock_publication, mock_participant)

        # Verify that create_task WAS called for video stream processing
        self.assertTrue(
            self._was_video_stream_task_created(client),
            "Video stream processing SHOULD be started when video_in_enabled=True",
        )

        # Verify that the callback was called
        client._callbacks.on_video_track_subscribed.assert_called_once_with(mock_participant.sid)

        # Verify that the track was added to _video_tracks
        self.assertIn(mock_participant.sid, client._video_tracks)


if __name__ == "__main__":
    unittest.main()
