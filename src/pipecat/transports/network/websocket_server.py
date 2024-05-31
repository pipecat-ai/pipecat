#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import io
import queue
import wave
import websockets

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_connection: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]


class WebsocketServerInputTransport(BaseInputTransport):

    def __init__(
            self,
            host: str,
            port: int,
            params: WebsocketServerParams,
            callbacks: WebsocketServerCallbacks):
        super().__init__(params)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._client_audio_queue = queue.Queue()
        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self):
        self._stop_server_event.set()
        await self._server_task
        await super().stop()

    def read_next_audio_frame(self) -> AudioRawFrame | None:
        try:
            return self._client_audio_queue.get(timeout=1)
        except queue.Empty:
            return None

    async def _server_task_handler(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port) as server:
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_connection(websocket)

        # Handle incoming messages
        async for message in websocket:
            frame = self._params.serializer.deserialize(message)
            if isinstance(frame, AudioRawFrame) and self._params.audio_in_enabled:
                self._client_audio_queue.put_nowait(frame)
            else:
                await self._internal_push_frame(frame)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")


class WebsocketServerOutputTransport(BaseOutputTransport):

    def __init__(self, params: WebsocketServerParams):
        super().__init__(params)

        self._params = params

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._audio_buffer = bytes()

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol):
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    def write_raw_audio_frames(self, frames: bytes):
        self._audio_buffer += frames
        while len(self._audio_buffer) >= self._params.audio_frame_size:
            frame = AudioRawFrame(
                audio=self._audio_buffer[:self._params.audio_frame_size],
                sample_rate=self._params.audio_out_sample_rate,
                num_channels=self._params.audio_out_channels
            )

            if self._params.add_wav_header:
                content = io.BytesIO()
                ww = wave.open(content, "wb")
                ww.setsampwidth(2)
                ww.setnchannels(frame.num_channels)
                ww.setframerate(frame.sample_rate)
                ww.writeframes(frame.audio)
                ww.close()
                content.seek(0)
                wav_frame = AudioRawFrame(
                    content.read(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels)
                frame = wav_frame

            proto = self._params.serializer.serialize(frame)

            future = asyncio.run_coroutine_threadsafe(
                self._websocket.send(proto), self.get_event_loop())
            future.result()

            self._audio_buffer = self._audio_buffer[self._params.audio_frame_size:]


class WebsocketServerTransport(BaseTransport):

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8765,
            params: WebsocketServerParams = WebsocketServerParams(),
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(loop)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_connection=self._on_connection
        )
        self._input: WebsocketServerInputTransport | None = None
        self._output: WebsocketServerOutputTransport | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self._host, self._port, self._params, self._callbacks)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params)
        return self._output

    async def _on_connection(self, websocket):
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")
