#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import time
import wave

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]


class WebsocketServerInputTransport(BaseInputTransport):
    def __init__(
        self,
        host: str,
        port: int,
        params: WebsocketServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._stop_server_event = asyncio.Event()

    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._stop_server_event.set()
        await self._server_task

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._stop_server_event.set()
        await self._server_task

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
        await self._callbacks.on_client_connected(websocket)

        # Handle incoming messages
        async for message in websocket:
            frame = self._params.serializer.deserialize(message)

            if not frame:
                continue

            if isinstance(frame, AudioRawFrame):
                await self.push_audio_frame(
                    InputAudioRawFrame(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    )
                )
            else:
                await self.push_frame(frame)

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")


class WebsocketServerOutputTransport(BaseOutputTransport):
    def __init__(self, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._websocket_audio_buffer = bytes()

        self._send_interval = (self._audio_chunk_size / self._params.audio_out_sample_rate) / 2
        self._next_send_time = 0

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol | None):
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._next_send_time = 0

    async def write_raw_audio_frames(self, frames: bytes):
        if not self._websocket:
            return

        frame = AudioRawFrame(
            audio=frames,
            sample_rate=self._params.audio_out_sample_rate,
            num_channels=self._params.audio_out_channels,
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
                content.read(), sample_rate=frame.sample_rate, num_channels=frame.num_channels
            )
            frame = wav_frame

        proto = self._params.serializer.serialize(frame)
        if proto:
            await self._websocket.send(proto)

        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval

        self._websocket_audio_buffer = bytes()


class WebsocketServerTransport(BaseTransport):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        params: WebsocketServerParams = WebsocketServerParams(),
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )
        self._input: WebsocketServerInputTransport | None = None
        self._output: WebsocketServerOutputTransport | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> WebsocketServerInputTransport:
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")
