import asyncio
import base64
import json
import websockets

from typing import List, Optional
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    Frame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from loguru import logger

# temp: websocket logger
# import logging
# logging.basicConfig(
#     format="%(message)s",
#     level=logging.DEBUG,
# )


class OpenAIInputTranscription(BaseModel):
    # enabled: bool = Field(description="Whether to enable input audio transcription.", default=True)
    model: str = Field(
        description="The model to use for transcription (e.g., 'whisper-1').", default="whisper-1"
    )


class OpenAITurnDetection(BaseModel):
    type: str = Field(
        default="server_vad",
        description="Type of turn detection, only 'server_vad' is currently supported.",
    )
    threshold: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Activation threshold for VAD (0.0 to 1.0)."
    )
    prefix_padding_ms: int = Field(
        default=300,
        description="Amount of audio to include before speech starts (in milliseconds).",
    )
    silence_duration_ms: int = Field(
        default=200, description="Duration of silence to detect speech stop (in milliseconds)."
    )


class RealtimeSessionProperties(BaseModel):
    modalities: List[str] = Field(default=["text", "audio"])
    instructions: str = Field(default="")
    voice: str = Field(default="alloy")
    input_audio_format: str = Field(default="pcm16")
    output_audio_format: str = Field(default="pcm16")
    input_audio_transcription: Optional[OpenAIInputTranscription] = Field(
        default=OpenAIInputTranscription()
    )
    turn_detection: Optional[OpenAITurnDetection] = Field(default=None)
    tools: List[str] = Field(default=[])
    tool_choice: str = Field(default="auto")
    temperature: float = Field(default=0.8)
    max_response_output_tokens: int = Field(default=4096)


class OpenAILLMServiceRealtimeBeta(LLMService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        session_properties: RealtimeSessionProperties = RealtimeSessionProperties(),
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._websocket = None
        self._receive_task = None

        self._session_properties = session_properties
        self._responses_in_flight = {}

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def update_session_properties(self):
        logger.debug(f"Updating session properties: {self._session_properties.dict()}")
        await self._websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": self._session_properties.dict(),
                }
            )
        )

    async def _connect(self):
        try:
            self._websocket = await websockets.connect(
                uri=self.base_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            )
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None

            self._context_id = None
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_task_handler(self):
        try:
            async for message in self._get_websocket():
                msg = json.loads(message)
                # logger.debug(f"Received message: {msg}")
                if not msg:
                    continue
                if msg["type"] == "session.created":
                    logger.debug(f"Received session.created: {msg}")
                    await self.update_session_properties()
                elif msg["type"] == "session.updated":
                    logger.debug(f"Received session configuration: {msg}")
                    self._session_properties = msg["session"]
                elif msg["type"] == "response.created":
                    pass
                elif msg["type"] == "response.output_item.added":
                    pass
                elif msg["type"] == "response.audio.delta":
                    frame = TTSAudioRawFrame(
                        audio=base64.b64decode(msg["delta"]),
                        sample_rate=24000,
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif msg["type"] == "response.text.delta":
                    logger.debug(f"!!! {msg['delta']}")
                    pass
                elif msg["type"] == "response.output_item.done":
                    if msg["item"]["type"] == "message":
                        for item in msg["item"]["content"]:
                            if item["type"] == "text":
                                await self.push_frame(TextFrame(item["text"]))
                elif msg["type"] == "response.done":
                    await self.stop_processing_metrics()
                    await self.push_frame(LLMFullResponseEndFrame())
                elif msg["type"] == "error":
                    raise Exception(f"Error: {msg}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _create_response(self, context: OpenAILLMContext, messages: list):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "status": "completed",
                            "role": "user",
                            "content": [{"type": "input_text", "text": messages[0]["content"]}],
                        },
                    }
                )
            )
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                        },
                    },
                )
            )
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _send_user_audio(self, frame):
        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self._websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": payload,
                },
            )
        )
        # await self._websocket.send(json.dumps(({"type": "input_audio_buffer.commit"})))

    async def _handle_interruption(self, frame):
        logger.debug(f"Handling interruption: {frame}")
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        await self._websocket.send(
            json.dumps(
                {
                    "type": "response.cancel",
                },
            )
        )
        await self._websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.clear",
                },
            )
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            messages = [{"role": "user", "content": frame.text}]
            context = OpenAILLMContext(messages)
            # await self._create_response(context, messages)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame)

        await self.push_frame(frame, direction)

    # async def get_chat_completions(
    #     self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    # ) -> AsyncStream[ChatCompletionChunk]:
    #     async def _empty_async_generator() -> AsyncGenerator[str, None]:
    #         try:
    #             if False:
    #                 yield ""
    #         except asyncio.CancelledError:
    #             return
    #         except Exception as e:
    #             logger.error(f"{self} exception: {e}")

    #     return _empty_async_generator()
