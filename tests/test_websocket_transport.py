#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# import asyncio
# import unittest
# from unittest.mock import AsyncMock, patch, Mock

# from pipecat.pipeline.frames import AudioFrame, EndFrame, TextFrame, TTSEndFrame, TTSStartFrame
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.transports.websocket_transport import WebSocketFrameProcessor, WebsocketTransport


# class TestWebSocketTransportService(unittest.IsolatedAsyncioTestCase):
#     def setUp(self):
#         self.transport = WebsocketTransport(host="localhost", port=8765)
#         self.pipeline = Pipeline([])
#         self.sample_frame = TextFrame("Hello there!")
#         self.serialized_sample_frame = self.transport._serializer.serialize(
#             self.sample_frame)

#     async def queue_frame(self):
#         await asyncio.sleep(0.1)
#         await self.pipeline.queue_frames([self.sample_frame, EndFrame()])

#     async def test_websocket_handler(self):
#         mock_websocket = AsyncMock()

#         with patch("websockets.serve", return_value=AsyncMock()) as mock_serve:
#             mock_serve.return_value.__anext__.return_value = (
#                 mock_websocket, "/")

#             await self.transport._websocket_handler(mock_websocket, "/")

#             await asyncio.gather(self.transport.run(self.pipeline), self.queue_frame())
#             self.assertEqual(mock_websocket.send.call_count, 1)

#             self.assertEqual(
#                 mock_websocket.send.call_args[0][0], self.serialized_sample_frame)

#     async def test_on_connection_decorator(self):
#         mock_websocket = AsyncMock()

#         connection_handler_called = asyncio.Event()

#         @self.transport.on_connection
#         async def connection_handler():
#             connection_handler_called.set()

#         with patch("websockets.serve", return_value=AsyncMock()):
#             await self.transport._websocket_handler(mock_websocket, "/")

#         self.assertTrue(connection_handler_called.is_set())

#     async def test_frame_processor(self):
#         processor = WebSocketFrameProcessor(audio_frame_size=4)

#         source_frames = [
#             TTSStartFrame(),
#             AudioFrame(b"1234"),
#             AudioFrame(b"5678"),
#             TTSEndFrame(),
#             TextFrame("hello world")
#         ]

#         frames = []
#         for frame in source_frames:
#             async for output_frame in processor.process_frame(frame):
#                 frames.append(output_frame)

#         self.assertEqual(len(frames), 3)
#         self.assertIsInstance(frames[0], AudioFrame)
#         self.assertEqual(frames[0].data, b"1234")
#         self.assertIsInstance(frames[1], AudioFrame)
#         self.assertEqual(frames[1].data, b"5678")
#         self.assertIsInstance(frames[2], TextFrame)
#         self.assertEqual(frames[2].text, "hello world")

#     async def test_serializer_parameter(self):
#         mock_websocket = AsyncMock()

#         # Test with ProtobufFrameSerializer (default)
#         with patch("websockets.serve", return_value=AsyncMock()) as mock_serve:
#             mock_serve.return_value.__anext__.return_value = (
#                 mock_websocket, "/")

#             await self.transport._websocket_handler(mock_websocket, "/")

#             await asyncio.gather(self.transport.run(self.pipeline), self.queue_frame())
#             self.assertEqual(mock_websocket.send.call_count, 1)
#             self.assertEqual(
#                 mock_websocket.send.call_args[0][0],
#                 self.serialized_sample_frame,
#             )

#         # Test with a mock serializer
#         mock_serializer = Mock()
#         mock_serializer.serialize.return_value = b"mock_serialized_data"
#         self.transport = WebsocketTransport(
#             host="localhost", port=8765, serializer=mock_serializer
#         )
#         mock_websocket.reset_mock()
#         with patch("websockets.serve", return_value=AsyncMock()) as mock_serve:
#             mock_serve.return_value.__anext__.return_value = (
#                 mock_websocket, "/")

#             await self.transport._websocket_handler(mock_websocket, "/")
#             await asyncio.gather(self.transport.run(self.pipeline), self.queue_frame())
#             self.assertEqual(mock_websocket.send.call_count, 1)
#             self.assertEqual(
#                 mock_websocket.send.call_args[0][0], b"mock_serialized_data")
#             mock_serializer.serialize.assert_called_once_with(
#                 TextFrame("Hello there!"))


# if __name__ == "__main__":
#     unittest.main()
