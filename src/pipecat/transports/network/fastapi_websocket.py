import asyncio
import io
import wave
from fastapi import WebSocket

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.serializers.TwilioFrameSerializer import TwilioFrameSerializer
from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger


class FastAPIWebsocketParams(TransportParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms
    serializer: FrameSerializer = TwilioFrameSerializer()


class FastAPIWebsocketCallbacks(BaseModel):
    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]


class FastAPIWebsocketInputTransport(BaseInputTransport):

    def __init__(self, websocket: WebSocket, params: FastAPIWebsocketParams, callbacks: FastAPIWebsocketCallbacks, **kwargs):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params
        self._callbacks = callbacks

    async def start(self, frame: StartFrame):
        await self._callbacks.on_client_connected(self._websocket)
        await super().start(frame)
        self._receive_task = self.get_event_loop().create_task(self._receive_messages())

    async def stop(self):
        await self._websocket.close()
        await super().stop()

    async def _receive_messages(self):
        async for message in self._websocket.iter_text():
            frame = self._params.serializer.deserialize(message)

            if not frame:
                continue

            if isinstance(frame, AudioRawFrame):
                self.push_audio_frame(frame)

        await self._callbacks.on_client_disconnected(self._websocket)

class FastAPIWebsocketOutputTransport(BaseOutputTransport):

    def __init__(self, websocket: WebSocket, params: FastAPIWebsocketParams, **kwargs):
        super().__init__(params, **kwargs)

        self._websocket = websocket
        self._params = params
        self._audio_buffer = bytes()

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

            payload = self._params.serializer.serialize(frame)

            future = asyncio.run_coroutine_threadsafe(
                self._websocket.send_json(payload), self.get_event_loop())
            future.result()

            self._audio_buffer = self._audio_buffer[self._params.audio_frame_size:]


class FastAPIWebsocketTransport(BaseTransport):

    def __init__(self, websocket: WebSocket, params: FastAPIWebsocketParams = FastAPIWebsocketParams(), input_name: str | None = None, output_name: str | None = None, loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected
        )

        self._input = FastAPIWebsocketInputTransport(websocket, self._params, self._callbacks, name=self._input_name)
        self._output = FastAPIWebsocketOutputTransport(websocket, self._params, name=self._output_name)

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    async def _on_client_connected(self, websocket):
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        await self._call_event_handler("on_client_disconnected", websocket)
