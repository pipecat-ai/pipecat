#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import io
import time
import typing
import wave
from typing import Awaitable, Callable

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use FastAPI websockets, you need to `pip install pipecat-ai[websocket]`."
    )
    raise Exception(f"Missing module: {e}")


class FastAPIWebsocketParams(TransportParams):
    add_wav_header: bool = False
    serializer: FrameSerializer
    session_timeout: int | None = None


class FastAPIWebsocketCallbacks(BaseModel):
    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
    on_session_timeout: Callable[[WebSocket], Awaitable[None]]


class FastAPIWebsocketInputTransport(BaseInputTransport):
    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        callbacks: FastAPIWebsocketCallbacks,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params
        self._callbacks = callbacks

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._params.session_timeout:
            self._monitor_websocket_task = self.get_event_loop().create_task(
                self._monitor_websocket()
            )
        await self._callbacks.on_client_connected(self._websocket)
        self._receive_task = self.get_event_loop().create_task(self._receive_messages())

    def _iter_data(self) -> typing.AsyncIterator[bytes | str]:
        if self._params.serializer.type == FrameSerializerType.BINARY:
            return self._websocket.iter_bytes()
        else:
            return self._websocket.iter_text()

    async def _receive_messages(self):
        async for message in self._iter_data():
            frame = self._params.serializer.deserialize(message)

            if not frame:
                continue

            if isinstance(frame, InputAudioRawFrame):
                await self.push_audio_frame(frame)
            else:
                await self.push_frame(frame)

        await self._callbacks.on_client_disconnected(self._websocket)

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        try:
            await asyncio.sleep(self._params.session_timeout)
            await self._callbacks.on_session_timeout(self._websocket)
        except asyncio.CancelledError:
            logger.info(f"Monitoring task cancelled for: {self._websocket}")


class FastAPIWebsocketOutputTransport(BaseOutputTransport):
    def __init__(self, websocket: WebSocket, params: FastAPIWebsocketParams, **kwargs):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params

        self._send_interval = (self._audio_chunk_size / self._params.audio_out_sample_rate) / 2
        self._next_send_time = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def write_raw_audio_frames(self, frames: bytes):
        if self._websocket.client_state != WebSocketState.CONNECTED:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self._params.audio_out_sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        self._websocket_audio_buffer = bytes()

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        payload = self._params.serializer.serialize(frame)
        if payload and self._websocket.client_state == WebSocketState.CONNECTED:
            await self._send_data(payload)

    def _send_data(self, data: str | bytes):
        if self._params.serializer.type == FrameSerializerType.BINARY:
            return self._websocket.send_bytes(data)
        else:
            return self._websocket.send_text(data)

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class FastAPIWebsocketTransport(BaseTransport):
    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._input = FastAPIWebsocketInputTransport(
            websocket, self._params, self._callbacks, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            websocket, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> FastAPIWebsocketInputTransport:
        return self._input

    def output(self) -> FastAPIWebsocketOutputTransport:
        return self._output

    async def _on_client_connected(self, websocket):
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        await self._call_event_handler("on_session_timeout", websocket)
