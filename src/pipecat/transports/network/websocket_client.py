#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import io
import time
import wave
from typing import Awaitable, Callable, Optional

import websockets
from loguru import logger
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio import TaskManager


class WebsocketClientParams(TransportParams):
    add_wav_header: bool = True
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketClientCallbacks(BaseModel):
    on_connected: Callable[[websockets.WebSocketClientProtocol], Awaitable[None]]
    on_disconnected: Callable[[websockets.WebSocketClientProtocol], Awaitable[None]]
    on_message: Callable[[websockets.WebSocketClientProtocol, websockets.Data], Awaitable[None]]


class WebsocketClientSession:
    def __init__(
        self,
        uri: str,
        params: WebsocketClientParams,
        callbacks: WebsocketClientCallbacks,
        transport_name: str,
    ):
        self._uri = uri
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name

        self._task_manager: Optional[TaskManager] = None

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None

    @property
    def task_manager(self) -> TaskManager:
        if not self._task_manager:
            raise Exception(
                f"{self._transport_name}::WebsocketClientSession: TaskManager not initialized (pipeline not started?)"
            )
        return self._task_manager

    async def setup(self, frame: StartFrame):
        if not self._task_manager:
            self._task_manager = frame.task_manager

    async def connect(self):
        if self._websocket:
            return

        try:
            self._websocket = await websockets.connect(uri=self._uri, open_timeout=10)
            self._client_task = self.task_manager.create_task(
                self._client_task_handler(),
                f"{self._transport_name}::WebsocketClientSession::_client_task_handler",
            )
            await self._callbacks.on_connected(self._websocket)
        except TimeoutError:
            logger.error(f"Timeout connecting to {self._uri}")

    async def disconnect(self):
        if not self._websocket:
            return

        await self.task_manager.cancel_task(self._client_task)

        await self._websocket.close()
        self._websocket = None

    async def send(self, message: websockets.Data):
        try:
            if self._websocket:
                await self._websocket.send(message)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _client_task_handler(self):
        try:
            # Handle incoming messages
            async for message in self._websocket:
                await self._callbacks.on_message(self._websocket, message)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        await self._callbacks.on_disconnected(self._websocket)

    def __str__(self):
        return f"{self._transport_name}::WebsocketClientSession"


class WebsocketClientInputTransport(BaseInputTransport):
    def __init__(self, session: WebsocketClientSession, params: WebsocketClientParams):
        super().__init__(params)

        self._session = session
        self._params = params

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._params.serializer.setup(frame)
        await self._session.setup(frame)
        await self._session.connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._session.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._session.disconnect()

    async def on_message(self, websocket, message):
        frame = await self._params.serializer.deserialize(message)
        if not frame:
            return
        if isinstance(frame, InputAudioRawFrame) and self._params.audio_in_enabled:
            await self.push_audio_frame(frame)
        else:
            await self.push_frame(frame)


class WebsocketClientOutputTransport(BaseOutputTransport):
    def __init__(self, session: WebsocketClientSession, params: WebsocketClientParams):
        super().__init__(params)

        self._session = session
        self._params = params

        # write_raw_audio_frames() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._send_interval = (self._audio_chunk_size / self.sample_rate) / 2
        await self._params.serializer.setup(frame)
        await self._session.setup(frame)
        await self._session.connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._session.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._session.disconnect()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._write_frame(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
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

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        payload = await self._params.serializer.serialize(frame)
        if payload:
            await self._session.send(payload)

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class WebsocketClientTransport(BaseTransport):
    def __init__(
        self,
        uri: str,
        params: WebsocketClientParams = WebsocketClientParams(),
    ):
        super().__init__()

        self._params = params

        callbacks = WebsocketClientCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_message=self._on_message,
        )

        self._session = WebsocketClientSession(uri, params, callbacks, self.name)
        self._input: Optional[WebsocketClientInputTransport] = None
        self._output: Optional[WebsocketClientOutputTransport] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")

    def input(self) -> WebsocketClientInputTransport:
        if not self._input:
            self._input = WebsocketClientInputTransport(self._session, self._params)
        return self._input

    def output(self) -> WebsocketClientOutputTransport:
        if not self._output:
            self._output = WebsocketClientOutputTransport(self._session, self._params)
        return self._output

    async def _on_connected(self, websocket):
        await self._call_event_handler("on_connected", websocket)

    async def _on_disconnected(self, websocket):
        await self._call_event_handler("on_disconnected", websocket)

    async def _on_message(self, websocket, message):
        if self._input:
            await self._input.on_message(websocket, message)
