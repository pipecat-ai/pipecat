#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDailyTransportRaceCondition(unittest.IsolatedAsyncioTestCase):
    """Tests for the race condition fix in DailyTransport.send_message()"""

    async def test_send_message_waits_for_join(self):
        """Test that send_message() waits for join to complete instead of rejecting immediately."""
        from pipecat.frames.frames import OutputTransportMessageFrame
        from pipecat.transports.daily.transport import DailyTransportClient

        # Create a mock transport object with just the attributes we need
        transport = MagicMock(spec=DailyTransportClient)
        transport._joined = False
        transport._joined_event = asyncio.Event()
        transport._client = MagicMock()

        # Mock the send_app_message to succeed via completion callback
        def mock_send(msg, pid, completion):
            completion(None)

        transport._client.send_app_message = mock_send
        transport._get_event_loop = MagicMock(return_value=asyncio.get_event_loop())

        # Set up the joined event to fire after a short delay
        async def set_joined_after_delay():
            await asyncio.sleep(0.05)
            transport._joined = True
            transport._joined_event.set()

        # Bind the real send_message method to our mock
        from pipecat.transports.daily.transport import DailyTransportClient

        send_message = DailyTransportClient.send_message

        # Schedule the event setter
        task = asyncio.create_task(set_joined_after_delay())

        # Call the real send_message with our mock object
        frame = OutputTransportMessageFrame(message="test message")
        result = await send_message(transport, frame)

        await task

        # Should succeed (no error)
        self.assertIsNone(result)

    async def test_send_message_timeout_if_join_slow(self):
        """Test that send_message() times out if join takes longer than 10 seconds."""
        from pipecat.frames.frames import OutputTransportMessageFrame
        from pipecat.transports.daily.transport import DailyTransportClient

        # Create a mock transport that never joins
        transport = MagicMock(spec=DailyTransportClient)
        transport._joined = False
        transport._joined_event = asyncio.Event()  # Event that never gets set
        transport._client = MagicMock()
        transport._get_event_loop = MagicMock(return_value=asyncio.get_event_loop())

        # Bind the real send_message method
        from pipecat.transports.daily.transport import DailyTransportClient

        send_message = DailyTransportClient.send_message

        frame = OutputTransportMessageFrame(message="test message")

        # Call send_message - it should timeout after ~10 seconds
        # For testing, we'll wrap it with a shorter timeout to fail fast
        start = asyncio.get_event_loop().time()
        result = await asyncio.wait_for(send_message(transport, frame), timeout=11.0)
        elapsed = asyncio.get_event_loop().time() - start

        # Should fail with timeout error (took at least 10 seconds)
        self.assertGreaterEqual(elapsed, 9.5)
        self.assertIn("timed out", result.lower() if result else "")

    async def test_send_message_already_joined(self):
        """Test that send_message() returns immediately if already joined."""
        from pipecat.frames.frames import OutputTransportMessageFrame
        from pipecat.transports.daily.transport import DailyTransportClient

        # Create a mock transport that's already joined
        transport = MagicMock(spec=DailyTransportClient)
        transport._joined = True
        transport._joined_event = asyncio.Event()
        transport._joined_event.set()
        transport._client = MagicMock()
        transport._get_event_loop = MagicMock(return_value=asyncio.get_event_loop())

        # Mock the send_app_message to succeed
        def mock_send(msg, pid, completion):
            completion(None)

        transport._client.send_app_message = mock_send

        # Bind the real send_message method
        from pipecat.transports.daily.transport import DailyTransportClient

        send_message = DailyTransportClient.send_message

        frame = OutputTransportMessageFrame(message="test message")

        start_time = asyncio.get_event_loop().time()
        result = await send_message(transport, frame)
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should succeed immediately
        self.assertIsNone(result)
        # Should not take significant time
        self.assertLess(elapsed, 0.1)

    async def test_send_message_disconnects_during_wait(self):
        """Test that send_message() handles disconnect during wait."""
        from pipecat.frames.frames import OutputTransportMessageFrame
        from pipecat.transports.daily.transport import DailyTransportClient

        transport = MagicMock(spec=DailyTransportClient)
        transport._joined = False
        transport._joined_event = asyncio.Event()
        transport._client = MagicMock()
        transport._get_event_loop = MagicMock(return_value=asyncio.get_event_loop())

        # Simulate transport being left while waiting
        async def clear_joined_during_wait():
            await asyncio.sleep(0.05)
            transport._joined = False
            transport._joined_event.set()

        # Bind the real method
        from pipecat.transports.daily.transport import DailyTransportClient

        send_message = DailyTransportClient.send_message

        frame = OutputTransportMessageFrame(message="test message")

        # Schedule disconnect
        task = asyncio.create_task(clear_joined_during_wait())

        result = await send_message(transport, frame)

        await task

        # Should fail because transport disconnected
        self.assertIn("disconnected", result.lower() if result else "")


class TestDailyTransport(unittest.IsolatedAsyncioTestCase):
    @unittest.skip("FIXME: This test is failing")
    async def test_event_handler(self):
        from pipecat.transports.daily_transport import DailyTransport

        transport = DailyTransport("mock.daily.co/mock", "token", "bot")

        was_called = False

        @transport.event_handler("on_first_other_participant_joined")
        def test_event_handler(transport, participant):
            nonlocal was_called
            was_called = True

        transport.on_first_other_participant_joined({"id": "user-id"})

        self.assertTrue(was_called)

    """
    TODO: fix this test, it broke when I added the `.result` call in the patch.
    async def test_event_handler_async(self):
        from pipecat.services.daily_transport_service import DailyTransportService

        transport = DailyTransportService("mock.daily.co/mock", "token", "bot")

        event = asyncio.Event()

        @transport.event_handler("on_first_other_participant_joined")
        async def test_event_handler(transport, participant):
            nonlocal event
            print("sleeping")
            await asyncio.sleep(0.1)
            print("setting")
            event.set()
            print("returning")

        thread = threading.Thread(target=transport.on_first_other_participant_joined)
        thread.start()
        thread.join()

        await asyncio.wait_for(event.wait(), timeout=1)
        self.assertTrue(event.is_set())
    """

    """
    @patch("pipecat.services.daily_transport_service.CallClient")
    @patch("pipecat.services.daily_transport_service.Daily")
    async def test_run_with_camera_and_mic(self, daily_mock, callclient_mock):
        from pipecat.services.daily_transport_service import DailyTransportService
        transport = DailyTransportService(
            "https://mock.daily.co/mock",
            "token",
            "bot",
            mic_enabled=True,
            camera_enabled=True,
            duration_minutes=0.01,
        )

        mic = MagicMock()
        camera = MagicMock()
        daily_mock.create_microphone_device.return_value = mic
        daily_mock.create_camera_device.return_value = camera

        async def send_audio_frame():
            await transport.send_queue.put(AudioFrame(bytes([0] * 3300)))

        async def send_video_frame():
            await transport.send_queue.put(ImageFrame(b"test", (0, 0)))

        await asyncio.gather(transport.run(), send_audio_frame(), send_video_frame())

        daily_mock.init.assert_called_once_with()
        daily_mock.create_microphone_device.assert_called_once()
        daily_mock.create_camera_device.assert_called_once()

        callclient_mock.return_value.set_user_name.assert_called_once_with("bot")
        callclient_mock.return_value.join.assert_called_once_with(
            "https://mock.daily.co/mock", "token", completion=transport.call_joined
        )

        camera.write_frame.assert_called_with(b"test")
        mic.write_frames.assert_called()
    """
