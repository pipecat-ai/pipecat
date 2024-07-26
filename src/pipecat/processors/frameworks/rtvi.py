#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import dataclasses

from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Type
from pydantic import PrivateAttr, BaseModel, ValidationError

from pipecat.frames.frames import (
    BotInterruptionFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMModelUpdateFrame,
    MetricsFrame,
    StartFrame,
    SystemFrame,
    TTSSpeakFrame,
    TTSVoiceUpdateFrame,
    TextFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext
from pipecat.transports.base_transport import BaseTransport

from loguru import logger


class RTVIServiceOption(BaseModel):
    name: str
    handler: Optional[Callable[['RTVIProcessor',
                                'RTVIServiceOptionConfig'],
                               Awaitable[None]]] = None


class RTVIService(BaseModel):
    name: str
    cls: Type[FrameProcessor]
    options: List[RTVIServiceOption]
    _options_dict: Dict[str, RTVIServiceOption] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._options_dict = {}
        for option in self.options:
            self._options_dict[option.name] = option
        return super().model_post_init(__context)

#
# Client -> Pipecat messages.
#


class RTVIServiceOptionConfig(BaseModel):
    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    config: List[RTVIServiceConfig]
    _config_dict: Dict[str, RTVIServiceConfig] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._config_dict = {}
        for c in self.config:
            self._config_dict[c.service] = c
        return super().model_post_init(__context)


class RTVILLMContextData(BaseModel):
    messages: List[dict]


class RTVITTSSpeakData(BaseModel):
    text: str
    interrupt: Optional[bool] = False


class RTVIMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None

#
# Pipecat -> Client responses and messages.
#


class RTVIResponseData(BaseModel):
    success: bool
    error: Optional[str] = None


class RTVIResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["response"] = "response"
    id: str
    data: RTVIResponseData


class RTVIErrorData(BaseModel):
    message: str


class RTVIError(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error"] = "error"
    data: RTVIErrorData


class RTVILLMContextMessageData(BaseModel):
    messages: List[dict]


class RTVILLMContextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-context"] = "llm-context"
    data: RTVILLMContextMessageData


class RTVITTSTextMessageData(BaseModel):
    text: str


class RTVITTSTextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["tts-text"] = "tts-text"
    data: RTVITTSTextMessageData


class RTVIBotReady(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-ready"] = "bot-ready"


class RTVITranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class RTVITranscriptionMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVITranscriptionMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIJSONCompletion(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["json-completion"] = "json-completion"
    data: str


class FunctionCaller(FrameProcessor):

    def __init__(self, context):
        super().__init__()
        self._checking = False
        self._aggregating = False
        self._emitted_start = False
        self._aggregation = ""
        self._context = context

        self._callbacks = {}
        self._start_callbacks = {}

    def register_function(self, function_name: str, callback, start_callback=None):
        self._callbacks[function_name] = callback
        if start_callback:
            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: str):
        del self._callbacks[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        return function_name in self._callbacks.keys()

    async def call_function(self, function_name: str, args):
        if function_name in self._callbacks.keys():
            return await self._callbacks[function_name](self, args)
        return None

    async def call_start_function(self, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](self)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._checking = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and self._checking:
            # TODO-CB: should we expand this to any non-text character to start the completion?
            if frame.text.strip().startswith("{") or frame.text.strip().startswith("```"):
                self._emitted_start = False
                self._checking = False
                self._aggregation = frame.text
                self._aggregating = True
            else:
                self._checking = False
                self._aggregating = False
                self._aggregation = ""
                self._emitted_start = False
                await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and self._aggregating:
            self._aggregation += frame.text
            # TODO-CB: We can probably ignore function start I think
            # if not self._emitted_start:
            #     fn = re.search(r'{"function_name":\s*"(.*)",', self._aggregation)
            #     if fn and fn.group(1):
            #         await self.call_start_function(fn.group(1))
            #         self._emitted_start = True
        elif isinstance(frame, LLMFullResponseEndFrame) and self._aggregating:
            try:
                self._aggregation = self._aggregation.replace("```json", "").replace("```", "")
                self._context.add_message({"role": "assistant", "content": self._aggregation})
                message = RTVIJSONCompletion(data=self._aggregation)
                msg = message.model_dump(exclude_none=True)
                await self.push_frame(TransportMessageFrame(message=msg))

            except Exception as e:
                print(f"Error parsing function call json: {e}")
                print(f"aggregation was: {self._aggregation}")

            self._aggregating = False
            self._aggregation = ""
            self._emitted_start = False
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class RTVITTSTextProcessor(FrameProcessor):

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, TextFrame):
            message = RTVITTSTextMessage(data=RTVITTSTextMessageData(text=frame.text))
            await self.push_frame(TransportMessageFrame(message=message.model_dump(exclude_none=True)))


async def handle_llm_model_update(rtvi: 'RTVIProcessor', option: RTVIServiceOptionConfig):
    frame = LLMModelUpdateFrame(option.value)
    await rtvi.push_frame(frame)


async def handle_llm_messages_update(rtvi: 'RTVIProcessor', option: RTVIServiceOptionConfig):
    frame = LLMMessagesUpdateFrame(option.value)
    await rtvi.push_frame(frame)


async def handle_tts_voice_update(rtvi: 'RTVIProcessor', option: RTVIServiceOptionConfig):
    frame = TTSVoiceUpdateFrame(option.value)
    await rtvi.push_frame(frame)

DEFAULT_LLM_SERVICE = RTVIService(
    name="llm",
    cls=OpenAILLMService,
    options=[
        RTVIServiceOption(name="model", handler=handle_llm_model_update),
        RTVIServiceOption(name="messages", handler=handle_llm_messages_update)
    ])

DEFAULT_TTS_SERVICE = RTVIService(
    name="tts",
    cls=CartesiaTTSService,
    options=[
        RTVIServiceOption(name="voice_id", handler=handle_tts_voice_update),
    ])


class RTVIProcessor(FrameProcessor):

    def __init__(self, *, transport: BaseTransport):
        super().__init__()
        self._transport = transport
        self._config: RTVIConfig | None = None
        self._ctor_args: Dict[str, Any] = {}

        self._start_frame: Frame | None = None
        self._pipeline: FrameProcessor | None = None
        self._first_participant_joined: bool = False

        # Register transport event so we can send a `bot-ready` event (and maybe
        # others) when the participant joins.
        transport.add_event_handler(
            "on_first_participant_joined",
            self._on_first_participant_joined)

        # Register default services.
        self._registered_services: Dict[str, RTVIService] = {}
        self.register_service(DEFAULT_LLM_SERVICE)
        self.register_service(DEFAULT_TTS_SERVICE)

        self._frame_handler_task = self.get_event_loop().create_task(self._frame_handler())
        self._frame_queue = asyncio.Queue()

    def register_service(self, service: RTVIService):
        self._registered_services[service.name] = service

    def setup_on_start(self, config: RTVIConfig | None, ctor_args: Dict[str, Any]):
        self._config = config
        self._ctor_args = ctor_args

    async def update_config(self, config: RTVIConfig):
        await self._handle_config_update(config)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            await self._frame_queue.put((frame, direction))

        if isinstance(frame, StartFrame):
            try:
                await self._handle_pipeline_setup(frame, self._config)
            except Exception as e:
                await self._send_error(f"unable to setup RTVI pipeline: {e}")

    async def cleanup(self):
        self._frame_handler_task.cancel()
        await self._frame_handler_task

    async def _frame_handler(self):
        while True:
            try:
                (frame, direction) = await self._frame_queue.get()
                await self._handle_frame(frame, direction)
                self._frame_queue.task_done()
            except asyncio.CancelledError:
                break

    async def _handle_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TransportMessageFrame):
            await self._handle_message(frame)
        else:
            await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) or isinstance(frame, InterimTranscriptionFrame):
            await self._handle_transcriptions(frame)
        elif isinstance(frame, UserStartedSpeakingFrame) or isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_interruptions(frame)

    async def _handle_transcriptions(self, frame: Frame):
        # TODO(aleix): Once we add support for using custom piplines, the STTs will
        # be in the pipeline after this processor. This means the STT will have to
        # push transcriptions upstream as well.

        message = None
        if isinstance(frame, TranscriptionFrame):
            message = RTVITranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                    final=True))
        elif isinstance(frame, InterimTranscriptionFrame):
            message = RTVITranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                    final=False))

        if message:
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)

    async def _handle_interruptions(self, frame: Frame):
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVIUserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVIUserStoppedSpeakingMessage()

        if message:
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)

    async def _handle_message(self, frame: TransportMessageFrame):
        try:
            message = RTVIMessage.model_validate(frame.message)
        except ValidationError as e:
            await self._send_error(f"Invalid incoming message: {e}")
            logger.warning(f"Invalid incoming  message: {e}")
            return

        try:
            success = True
            error = None
            match message.type:
                case "config-update":
                    await self._handle_config_update(RTVIConfig.model_validate(message.data))
                case "llm-get-context":
                    await self._handle_llm_get_context()
                case "llm-append-context":
                    await self._handle_llm_append_context(RTVILLMContextData.model_validate(message.data))
                case "llm-update-context":
                    await self._handle_llm_update_context(RTVILLMContextData.model_validate(message.data))
                case "tts-speak":
                    await self._handle_tts_speak(RTVITTSSpeakData.model_validate(message.data))
                case "tts-interrupt":
                    await self._handle_tts_interrupt()
                case _:
                    success = False
                    error = f"Unsupported type {message.type}"

            await self._send_response(message.id, success, error)
        except ValidationError as e:
            await self._send_response(message.id, False, f"Invalid incoming message: {e}")
            logger.warning(f"Invalid incoming  message: {e}")
        except Exception as e:
            await self._send_response(message.id, False, f"Exception processing message: {e}")
            logger.warning(f"Exception processing message: {e}")

    async def _handle_pipeline_setup(self, start_frame: StartFrame, config: RTVIConfig | None):
        # TODO(aleix): We shouldn't need to save this in `self._tma_in`.
        self._tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        llm_cls = self._registered_services["llm"].cls
        llm_args = self._ctor_args["llm"]
        llm = llm_cls(**llm_args)

        tts_cls = self._registered_services["tts"].cls
        tts_args = self._ctor_args["tts"]
        tts = tts_cls(**tts_args)

        # TODO-CB: Eventually we'll need to switch the context aggregators to use the
        # OpenAI context frames instead of message frames
        context = OpenAILLMContext()
        fc = FunctionCaller(context)

        tts_text = RTVITTSTextProcessor()

        pipeline = Pipeline([
            self._tma_in,
            llm,
            fc,
            tts,
            tts_text,
            tma_out,
            self._transport.output(),
        ])

        parent = self.get_parent()
        if parent:
            parent.link(pipeline)

            # We need to initialize the new pipeline with the same settings
            # as the initial one.
            start_frame = dataclasses.replace(start_frame)
            await self.push_frame(start_frame)

            # Configure the pipeline
            if config:
                await self._handle_config_update(config)

            # Send new initial metrics with the new processors
            processors = parent.processors_with_metrics()
            processors.extend(pipeline.processors_with_metrics())
            ttfb = [{"processor": p.name, "value": 0.0} for p in processors]
            processing = [{"processor": p.name, "value": 0.0} for p in processors]
            await self.push_frame(MetricsFrame(ttfb=ttfb, processing=processing))

        self._pipeline = pipeline

        await self._maybe_send_bot_ready()

    async def _handle_config_service(self, config: RTVIServiceConfig):
        service = self._registered_services[config.service]
        for option in config.options:
            handler = service._options_dict[option.name].handler
            if handler:
                await handler(self, option)

    async def _handle_config_update(self, data: RTVIConfig):
        for config in data.config:
            await self._handle_config_service(config)

    async def _handle_llm_get_context(self):
        data = RTVILLMContextMessageData(messages=self._tma_in.messages)
        message = RTVILLMContextMessage(data=data)
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _handle_llm_append_context(self, data: RTVILLMContextData):
        if data and data.messages:
            frame = LLMMessagesAppendFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_llm_update_context(self, data: RTVILLMContextData):
        if data and data.messages:
            frame = LLMMessagesUpdateFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_tts_speak(self, data: RTVITTSSpeakData):
        if data and data.text:
            if data.interrupt:
                await self._handle_tts_interrupt()
            frame = TTSSpeakFrame(text=data.text)
            await self.push_frame(frame)

    async def _handle_tts_interrupt(self):
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def _on_first_participant_joined(self, transport, participant):
        self._first_participant_joined = True
        await self._maybe_send_bot_ready()

    async def _maybe_send_bot_ready(self):
        if self._pipeline and self._first_participant_joined:
            message = RTVIBotReady()
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)

    async def _send_error(self, error: str):
        message = RTVIError(data=RTVIErrorData(message=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _send_response(self, id: str, success: bool, error: str | None = None):
        # TODO(aleix): This is a bit hacky, but we might get invalid
        # configuration or something might going wrong during setup and we would
        # like to send the error to the client. However, if the pipeline is not
        # setup yet we don't have an output transport and therefore we can't
        # send any messages. So, we setup a super basic pipeline with just the
        # output transport so we can send messages.
        if not self._pipeline:
            pipeline = Pipeline([self._transport.output()])
            self._pipeline = pipeline

            parent = self.get_parent()
            if parent:
                parent.link(pipeline)

        message = RTVIResponse(id=id, data=RTVIResponseData(success=success, error=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)
