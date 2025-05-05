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
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
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
    session_timeout: Optional[int] = None


class FastAPIWebsocketCallbacks(BaseModel):
    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
    on_session_timeout: Callable[[WebSocket], Awaitable[None]]


class FastAPIWebsocketClient:
    def __init__(self, websocket: WebSocket, is_binary: bool, callbacks: FastAPIWebsocketCallbacks):
        self._websocket = websocket
        self._closing = False
        self._is_binary = is_binary
        self._callbacks = callbacks
        self._leave_counter = 0

    async def setup(self, _: StartFrame):
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        return self._websocket.iter_bytes() if self._is_binary else self._websocket.iter_text()

    async def send(self, data: str | bytes):
        if self._can_send():
            if self._is_binary:
                await self._websocket.send_bytes(data)
            else:
                await self._websocket.send_text(data)

    async def disconnect(self):
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            await self._websocket.close()
            await self.trigger_client_disconnected()

    async def trigger_client_disconnected(self):
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timout(self):
        await self._callbacks.on_session_timeout(self._websocket)

    def _can_send(self):
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        return self._websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_closing(self) -> bool:
        return self._closing


class FastAPIWebsocketInputTransport(BaseInputTransport):
    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._monitor_websocket_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(frame)
        await self._params.serializer.setup(frame)
        if not self._monitor_websocket_task and self._params.session_timeout:
            self._monitor_websocket_task = self.create_task(self._monitor_websocket())
        await self._client.trigger_client_connected()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_messages())
        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        if self._monitor_websocket_task:
            await self.cancel_task(self._monitor_websocket_task)
            self._monitor_websocket_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def _receive_messages(self):
        try:
            async for message in self._client.receive():
                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
        await self._client.trigger_client_timout()


class FastAPIWebsocketOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
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
        await self._client.setup(frame)
        await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._write_frame(frame)

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        if self._client.is_closing:
            return

        if not self._client.is_connected:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

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
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                await self._client.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

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
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        is_binary = self._params.serializer.type == FrameSerializerType.BINARY
        self._client = FastAPIWebsocketClient(websocket, is_binary, self._callbacks)

        self._input = FastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
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
