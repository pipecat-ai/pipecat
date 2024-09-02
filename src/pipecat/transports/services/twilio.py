import asyncio

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    MarkFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketInputTransport,
    FastAPIWebsocketCallbacks,
)
from pipecat.transports.base_transport import BaseTransport
from starlette.websockets import WebSocket, WebSocketState
from loguru import logger


class TwilioOutputTransport(FastAPIWebsocketOutputTransport):
    def __init__(self, websocket: WebSocket, params: FastAPIWebsocketParams, **kwargs):
        super().__init__(websocket, params, **kwargs)
        self.current_count = 0
        self.received_count = 0

    async def _bot_started_speaking(self):
        logger.debug(
            f"Bot started speaking, Bot already speaking: {self._bot_speaking}"
        )
        if not self._bot_speaking:
            self._bot_speaking = True
            await self._internal_push_frame(
                BotStartedSpeakingFrame(), FrameDirection.UPSTREAM
            )

    async def _bot_stopped_speaking(self):
        logger.info("Pushing the Marker at the end of stream")
        self.current_count += 1
        mark_frame = MarkFrame(
            passed_name=str(self.current_count),
            type="request",
            seq_number=self.current_count,
        )
        payload = self._params.serializer.serialize(mark_frame)
        if payload and self._websocket.client_state == WebSocketState.CONNECTED:
            await self._websocket.send_text(payload)
            logger.debug(f"Pushed the mark frame {payload}")

    async def _send_bot_stopped_speaking(self):
        logger.info("Bot Stopped")
        await self._internal_push_frame(
            BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        logger.trace(f"Received the frame: {frame}, {direction}")
        if (
            self._bot_speaking
            and isinstance(frame, MarkFrame)
            and frame.type == "response"
        ):
            logger.info(
                "Received the mark frame and sending the Signal for Bot Being stopped from speaking."
            )
            self.received_count += 1
            if self.received_count == self.current_count:
                logger.info("Bot Stopped Speaking")
                await self._send_bot_stopped_speaking()
                self._bot_speaking = False
        else:
            await super().process_frame(frame, direction)


class TwilioInputTransport(FastAPIWebsocketInputTransport):
    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        callbacks: FastAPIWebsocketCallbacks,
        **kwargs,
    ):
        super().__init__(websocket, params, callbacks, **kwargs)

    async def _receive_messages(self):
        async for message in self._websocket.iter_text():
            frame = self._params.serializer.deserialize(message)

            if not frame:
                continue

            if isinstance(frame, MarkFrame):
                logger.info(
                    f"Pushing the {frame} downstream from CustomTwilioInputProcessor"
                )
                await self._internal_push_frame(frame, FrameDirection.DOWNSTREAM)

            if isinstance(frame, AudioRawFrame):
                # logger.info(f"Pushing the audio frame {frame}")
                await self.push_audio_frame(frame)

        await self._callbacks.on_client_disconnected(self._websocket)


class TwilioTransport(BaseTransport):
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
        )

        self._input = TwilioInputTransport(
            websocket, self._params, self._callbacks, name=self._input_name
        )

        self._output = TwilioOutputTransport(
            websocket, self._params, name=self._output_name
        )

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
