import asyncio
import time
from typing import AsyncGenerator, List

from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.frames import AudioFrame, ControlFrame, EndFrame, Frame, TTSEndFrame, TTSStartFrame, TextFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.serializers.protobuf_serializer import ProtobufFrameSerializer
from dailyai.transports.abstract_transport import AbstractTransport
from dailyai.transports.threaded_transport import ThreadedTransport

try:
    import websockets
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use the websocket transport, you need to `pip install dailyai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebSocketFrameProcessor(FrameProcessor):
    """This FrameProcessor filters and mutates frames before they're sent over the websocket.
    This is necessary to aggregate audio frames into sizes that are cleanly playable by the client"""

    def __init__(
            self,
            audio_frame_size: int | None = None,
            sendable_frames: List[Frame] | None = None):
        super().__init__()
        if not audio_frame_size:
            raise ValueError("audio_frame_size must be provided")

        self._audio_frame_size = audio_frame_size
        self._sendable_frames = sendable_frames or [TextFrame, AudioFrame]
        self._audio_buffer = bytes()
        self._in_tts_audio = False

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TTSStartFrame):
            self._in_tts_audio = True
        elif isinstance(frame, AudioFrame):
            if self._in_tts_audio:
                self._audio_buffer += frame.data
                while len(self._audio_buffer) >= self._audio_frame_size:
                    yield AudioFrame(self._audio_buffer[:self._audio_frame_size])
                    self._audio_buffer = self._audio_buffer[self._audio_frame_size:]
        elif isinstance(frame, TTSEndFrame):
            self._in_tts_audio = False
            if self._audio_buffer:
                yield AudioFrame(self._audio_buffer)
                self._audio_buffer = bytes()
        elif type(frame) in self._sendable_frames or isinstance(frame, ControlFrame):
            yield frame


class WebsocketTransport(AbstractTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sample_width = kwargs.get("sample_width", 2)
        self._n_channels = kwargs.get("n_channels", 1)
        self._port = kwargs.get("port", 8765)
        self._host = kwargs.get("host", "localhost")
        self._audio_frame_size = kwargs.get("audio_frame_size", 16000)
        self._sendable_frames = kwargs.get(
            "sendable_frames", [
                TextFrame, AudioFrame, TTSEndFrame, TTSStartFrame])
        self._serializer = kwargs.get("serializer", ProtobufFrameSerializer())

        self._server: websockets.WebSocketServer | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._connection_handlers = []

    async def run(self, pipeline: Pipeline, override_pipeline_source_queue=True):
        self._stop_server_event = asyncio.Event()
        pipeline.set_sink(self.send_queue)
        if override_pipeline_source_queue:
            pipeline.set_source(self.receive_queue)

        pipeline.add_processor(WebSocketFrameProcessor(
            audio_frame_size=self._audio_frame_size,
            sendable_frames=self._sendable_frames))

        async def timeout():
            sleep_time = self._expiration - time.time()
            await asyncio.sleep(sleep_time)
            self._stop_server_event.set()

        async def send_task():
            while not self._stop_server_event.is_set():
                frame = await self.send_queue.get()
                if isinstance(frame, EndFrame):
                    self._stop_server_event.set()
                    break
                if self._websocket and frame:
                    proto = self._serializer.serialize(frame)
                    await self._websocket.send(proto)

        async def start_server():
            async with websockets.serve(self._websocket_handler, self._host, self._port) as server:
                self._logger.debug("Websocket server started.")
                await self._stop_server_event.wait()
                self._logger.debug("Websocket server stopped.")
            await self.receive_queue.put(EndFrame())

        timeout_task = asyncio.create_task(timeout())
        await asyncio.gather(start_server(), send_task(), pipeline.run_pipeline())
        timeout_task.cancel()

    def on_connection(self, handler):
        self._connection_handlers.append(handler)

    async def _websocket_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        if self._websocket:
            await self._websocket.close()
            self._logger.warning(
                "Got another websocket connection; closing first.")

        for handler in self._connection_handlers:
            await handler()

        self._websocket = websocket
        async for message in websocket:
            frame = self._serializer.deserialize(message)
            await self.receive_queue.put(frame)
