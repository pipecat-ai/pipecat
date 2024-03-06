import asyncio
import unittest

from unittest.mock import MagicMock, patch

from dailyai.pipeline.frames import AudioFrame, ImageFrame


class TestDailyTransport(unittest.IsolatedAsyncioTestCase):

    async def test_event_handler(self):
        from dailyai.services.daily_transport_service import DailyTransportService

        transport = DailyTransportService("mock.daily.co/mock", "token", "bot")

        was_called = False

        @transport.event_handler("on_first_other_participant_joined")
        def test_event_handler(transport):
            nonlocal was_called
            was_called = True

        transport.on_first_other_participant_joined()

        self.assertTrue(was_called)

    async def test_event_handler_async(self):
        from dailyai.services.daily_transport_service import DailyTransportService

        transport = DailyTransportService("mock.daily.co/mock", "token", "bot")

        event = asyncio.Event()

        @transport.event_handler("on_first_other_participant_joined")
        async def test_event_handler(transport):
            nonlocal event
            await asyncio.sleep(0.1)
            event.set()

        transport.on_first_other_participant_joined()

        await asyncio.wait_for(event.wait(), timeout=1)
        self.assertTrue(event.is_set())

    """
    @patch("dailyai.services.daily_transport_service.CallClient")
    @patch("dailyai.services.daily_transport_service.Daily")
    async def test_run_with_camera_and_mic(self, daily_mock, callclient_mock):
        from dailyai.services.daily_transport_service import DailyTransportService
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
            await transport.send_queue.put(AudioQueueFrame(bytes([0] * 3300)))

        async def send_video_frame():
            await transport.send_queue.put(ImageQueueFrame(None, b"test"))

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
