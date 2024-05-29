#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import websockets

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TTSStartedFrame,
    TTSStoppedFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger


class WebsocketServerParams(TransportParams):
    audio_frame_size: int = 16000
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_connection: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]


class WebsocketServerInputTransport(FrameProcessor):

    def __init__(
            self,
            host: str,
            port: int,
            params: WebsocketServerParams,
            callbacks: WebsocketServerCallbacks):
        super().__init__()

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._stop_server_event = asyncio.Event()

        # Create push frame task. This is the task that will push frames in
        # order. We also guarantee that all frames are pushed in the same task.
        self._create_push_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, CancelFrame):
            await self._stop()
            # We don't queue a CancelFrame since we want to stop ASAP.
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
            await self._start()
            await self._internal_push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._stop()
            await self._internal_push_frame(frame, direction)
        else:
            await self._internal_push_frame(frame, direction)

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
            await self._internal_push_frame(frame)

    async def _start(self):
        loop = self.get_event_loop()
        self._server_task = loop.create_task(self._server_task_handler())

    async def _stop(self):
        self._stop_server_event.set()
        self._push_frame_task.cancel()
        await self._server_task

    #
    # Push frames task
    #

    def _create_push_task(self):
        loop = self.get_event_loop()
        self._push_frame_task = loop.create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

    async def _internal_push_frame(
            self,
            frame: Frame | None,
            direction: FrameDirection | None = FrameDirection.DOWNSTREAM):
        await self._push_queue.put((frame, direction))

    async def _push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
                running = not isinstance(frame, EndFrame)
            except asyncio.CancelledError:
                break


class WebsocketServerOutputTransport(FrameProcessor):

    def __init__(self, params: WebsocketServerParams):
        super().__init__()

        self._params = params

        self._websocket = None
        self._audio_buffer = bytes()

        self._websocket: websockets.WebSocketServerProtocol | None = None

        loop = self.get_event_loop()
        self._send_queue_task = loop.create_task(self._send_queue_task_handler())
        self._send_queue = asyncio.Queue()

        self._audio_buffer = bytes()
        self._in_tts_audio = False

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol):
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def _send_queue_task_handler(self):
        running = True
        while running:
            frame = await self._send_queue.get()
            if self._websocket and frame:
                proto = self._params.serializer.serialize(frame)
                await self._websocket.send(proto)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, CancelFrame):
            # await self.stop()
            # We don't queue a CancelFrame since we want to stop ASAP.
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            self._in_tts_audio = True
        elif isinstance(frame, AudioRawFrame):
            if self._in_tts_audio:
                self._audio_buffer += frame.audio
                while len(self._audio_buffer) >= self._params.audio_frame_size:
                    frame = AudioRawFrame(
                        audio=self._audio_buffer[:self._params.audio_frame_size],
                        sample_rate=self._params.audio_out_sample_rate,
                        num_channels=self._params.audio_out_channels
                    )
                    await self._send_queue.put(frame)
                    self._audio_buffer = self._audio_buffer[self._params.audio_frame_size:]
        elif isinstance(frame, TTSStoppedFrame):
            self._in_tts_audio = False
            if self._audio_buffer:
                frame = AudioRawFrame(
                    audio=self._audio_buffer,
                    sample_rate=self._params.audio_out_sample_rate,
                    num_channels=self._params.audio_out_channels
                )
                await self._send_queue.put(frame)
                self._audio_buffer = bytes()
        else:
            await self.push_frame(frame, direction)


class WebsocketServerTransport(BaseTransport):

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8765,
            params: WebsocketServerParams = WebsocketServerParams(),
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()):
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
