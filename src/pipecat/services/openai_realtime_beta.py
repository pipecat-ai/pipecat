import asyncio
import base64
import random
import json
import websockets

from copy import deepcopy
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
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
    OpenAIContextAggregatorPair,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

from loguru import logger

# temp: websocket logger
# import logging
# logging.basicConfig(
#     format="%(message)s",
#     level=logging.DEBUG,
# )


class OpenAIUnhandledFunctionException(Exception):
    pass


class OpenAIRealtimeLLMContext(OpenAILLMContext):
    @staticmethod
    def upgrade_to_realtime(obj: OpenAILLMContext) -> "OpenAIRealtimeLLMContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, OpenAIRealtimeLLMContext):
            obj.__class__ = OpenAIRealtimeLLMContext
            obj._unsent_messages = deepcopy(obj._messages)
            obj._marker = random.randint(1, 1000)
        return obj

    def add_message(self, message):
        super().add_message(message)
        if message.get("role") == "tool":
            self._unsent_messages.append(message)

    def set_messages(self, messages):
        super().set_messages(messages)
        self._unsent_messages = deepcopy(self._messages)

    def get_unsent_messages(self):
        return self._unsent_messages

    def update_all_messages_sent(self):
        logger.debug("!!! Updating all messages sent")
        self._unsent_messages = []


class OpenAIRealtimeUserContextAggregator(OpenAIUserContextAggregator):
    async def _push_aggregation(self):
        pass
        # idx = len(self._context.messages)
        # logger.debug(f"!!! 1 {idx}")

        # await super()._push_aggregation()
        # self._context._unsent_messages.extend(self._context.messages[idx:])
        # logger.debug(f"!!! 2 {self._context._unsent_messages}")


class OpenAIRealtimeAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def _push_aggregation(self):
        await super()._push_aggregation()


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
    tools: List[dict] = Field(default=[])
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
        self._context = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _ws_send(self, realtime_message):
        try:
            if realtime_message.get("type") != "input_audio_buffer.append":
                logger.debug(f"!!! Sending message to websocket: {realtime_message}")

            await self._websocket.send(json.dumps(realtime_message))
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")

    async def update_session_properties(self):
        logger.debug(f"Updating session properties: {self._session_properties.dict()}")
        await self._ws_send(
            {
                "type": "session.update",
                "session": self._session_properties.dict(),
            }
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
                    await self.update_session_properties()
                elif msg["type"] == "session.updated":
                    logger.debug(f"Received session configuration: {msg}")
                    self._session_properties = msg["session"]
                elif msg["type"] == "input_audio_buffer.speech_started":
                    # user started speaking
                    pass
                elif msg["type"] == "input_audio_buffer.speech_stopped":
                    # user stopped speaking
                    pass
                elif msg["type"] == "conversation.item.created":
                    # for input, this will get sent from the server whether the
                    # conversation item is created by audio transcription or by
                    # sending a client conversation.item.create message.
                    # for function calls
                    logger.debug(f"Received {msg}")
                    pass
                elif msg["type"] == "response.created":
                    # could use for processing metrics
                    pass
                elif msg["type"] == "conversation.item.input_audio_transcription.completed":
                    # or here maybe (possible send upstream to user context aggregator)
                    logger.debug(f"Received {msg}")
                    if msg.get("transcript"):
                        self._context.add_message({"role": "user", "content": msg["transcript"]})
                elif msg["type"] == "response.output_item.added":
                    # maybe ignore for now but could be useful for UI updates
                    pass
                elif msg["type"] == "response.content_part.added":
                    # same thing, ignore for now until we think more about UI updates
                    pass
                elif msg["type"] == "response.audio_transcript.delta":
                    # openai playground app uses this, not "text"
                    pass
                elif msg["type"] == "response.audio.delta":
                    frame = TTSAudioRawFrame(
                        audio=base64.b64decode(msg["delta"]),
                        sample_rate=24000,
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif msg["type"] == "response.audio.done":
                    # bot stopped speaking - or do that at the end of the response?
                    pass
                elif msg["type"] == "response.audio_transcript.done":
                    # probably ignore for now
                    pass
                elif msg["type"] == "response.content_part.done":
                    pass
                elif msg["type"] == "response.output_item.done":
                    logger.debug(f"Received response item done: {msg}")
                    item = msg["item"]
                    if item["type"] == "message" and item["status"] == "completed":
                        for item in item["content"]:
                            # output text
                            if item["type"] == "audio" and item["transcript"] is not None:
                                logger.debug(f"!!! >{item['transcript']}")
                                await self.push_frame(TextFrame(item["transcript"]))
                elif msg["type"] == "response.done":
                    logger.debug(f"Received response done: {msg}")
                    await self.stop_processing_metrics()
                    # send usage metrics from here
                    # ...
                    # function calls - do all calls here to support parallel function calls
                    items = msg["response"]["output"]
                    function_calls = [item for item in items if item.get("type") == "function_call"]
                    if function_calls:
                        await self._handle_function_call_items(function_calls)
                    await self.push_frame(LLMFullResponseEndFrame())
                elif msg["type"] == "rate_limits.updated":
                    pass
                elif msg["type"] == "error":
                    raise Exception(f"Error: {msg}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _handle_function_call_items(self, items):
        logger.debug(f"Handling function call items: {items}")
        total_items = len(items)
        logger.debug("!!!!!")
        for index, item in enumerate(items):
            logger.debug(f"!!! function call item: {item}")
            function_name = item["name"]
            tool_id = item["call_id"]
            arguments = json.loads(item["arguments"])
            if self.has_function(function_name):
                run_llm = index == total_items - 1
                if function_name in self._callbacks.keys():
                    f = self._callbacks[function_name]
                elif None in self._callbacks.keys():
                    await self.call_function(
                        context=self._context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                        run_llm=run_llm,
                    )
            else:
                raise OpenAIUnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )

    async def _reset_conversation(self):
        # need to think about how to implement this, and how to think about interop with messages lists
        # used with the HTTP API
        pass

    async def _create_response(self, context: OpenAIRealtimeLLMContext):
        try:
            messages = context.get_unsent_messages()
            context.update_all_messages_sent()
            logger.debug(
                f"Creating response: {context._marker} {context.get_messages_for_logging()}"
            )

            items = []
            for m in messages:
                if m and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        items.append(
                            {
                                "type": "message",
                                "status": "completed",
                                "role": "user",
                                "content": [{"type": "input_text", "text": content}],
                            }
                        )
                    else:
                        raise Exception(f"Invalid message content {m}")
                elif m and m.get("role") == "tool":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": m.get("tool_call_id"),
                            "output": m["content"],
                        }
                    )

            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            for item in items:
                logger.debug(f"||| {item}")
                await self._ws_send({"type": "conversation.item.create", "item": item})
            logger.debug("||| RESPONSE.CREATE")
            await self._ws_send(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                    },
                },
            )
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _send_user_audio(self, frame):
        payload = base64.b64encode(frame.audio).decode("utf-8")
        await self._ws_send(
            {
                "type": "input_audio_buffer.append",
                "audio": payload,
            },
        )

    async def _handle_interruption(self, frame):
        logger.debug(f"Handling interruption: {frame}")
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())
        # await self._ws_send(
        #     {
        #         "type": "response.cancel",
        #     },
        # )
        # await self._ws_send(
        #     {
        #         "type": "input_audio_buffer.clear",
        #     },
        # )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            pass
        elif isinstance(frame, OpenAILLMContextFrame):
            context: OpenAIRealtimeLLMContext = OpenAIRealtimeLLMContext.upgrade_to_realtime(
                frame.context
            )
            self._context = context
            await self._create_response(context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame)

        await self.push_frame(frame, direction)

    def create_context_aggregator(
        self, context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = False
    ) -> OpenAIContextAggregatorPair:
        OpenAIRealtimeLLMContext.upgrade_to_realtime(context)
        user = OpenAIRealtimeUserContextAggregator(context)
        assistant = OpenAIRealtimeAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)
