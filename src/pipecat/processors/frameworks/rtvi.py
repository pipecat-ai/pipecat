#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import dataclasses

from typing import List, Literal, Optional, Type
from pydantic import BaseModel, ValidationError

from pipecat.frames.frames import (
    BotInterruptionFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMModelUpdateFrame,
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
from pipecat.services.ai_services import AIService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext
from pipecat.transports.base_transport import BaseTransport

DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
    }
]

DEFAULT_MODEL = "llama3-70b-8192"

DEFAULT_VOICE = "79a125e8-cd45-4c13-8a67-188112f4dd22"


class RTVILLMConfig(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[dict]] = None


class RTVITTSConfig(BaseModel):
    voice: Optional[str] = None


class RTVIConfig(BaseModel):
    llm: Optional[RTVILLMConfig] = None
    tts: Optional[RTVITTSConfig] = None


class RTVISetup(BaseModel):
    config: Optional[RTVIConfig] = None


class RTVILLMMessageData(BaseModel):
    messages: List[dict]


class RTVITTSMessageData(BaseModel):
    text: str
    interrupt: Optional[bool] = False


class RTVIMessageData(BaseModel):
    setup: Optional[RTVISetup] = None
    config: Optional[RTVIConfig] = None
    llm: Optional[RTVILLMMessageData] = None
    tts: Optional[RTVITTSMessageData] = None


class RTVIMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: str
    id: str
    data: Optional[RTVIMessageData] = None


class RTVIResponseData(BaseModel):
    success: bool
    error: Optional[str] = None


class RTVIResponse(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["response"] = "response"
    id: str
    data: RTVIResponseData


class RTVIErrorData(BaseModel):
    message: str


class RTVIError(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["error"] = "error"
    data: RTVIErrorData


class RTVILLMContextMessageData(BaseModel):
    messages: List[dict]


class RTVILLMContextMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["llm-context"] = "llm-context"
    data: RTVILLMContextMessageData


class RTVITTSTextMessageData(BaseModel):
    text: str


class RTVITTSTextMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["tts-text"] = "tts-text"
    data: RTVITTSTextMessageData


class RTVIBotReady(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["bot-ready"] = "bot-ready"


class RTVITranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class RTVITranscriptionMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVITranscriptionMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi"] = "rtvi"
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIJSONCompletion(BaseModel):
    label: Literal["rtvi"] = "rtvi"
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


class RTVIProcessor(FrameProcessor):

    def __init__(
            self,
            *,
            transport: BaseTransport,
            setup: RTVISetup | None = None,
            llm_api_key: str = "",
            llm_base_url: str = "https://api.groq.com/openai/v1",
            tts_api_key: str = "",
            llm_cls: Type[AIService] = OpenAILLMService,
            tts_cls: Type[AIService] = CartesiaTTSService):
        super().__init__()
        self._transport = transport
        self._setup = setup
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._tts_api_key = tts_api_key
        self._llm_cls = llm_cls
        self._tts_cls = tts_cls
        self._start_frame: Frame | None = None
        self._llm: FrameProcessor | None = None
        self._tts: FrameProcessor | None = None
        self._pipeline: FrameProcessor | None = None

        self._frame_handler_task = self.get_event_loop().create_task(self._frame_handler())
        self._frame_queue = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            await self._frame_queue.put((frame, direction))

        if isinstance(frame, StartFrame):
            self._start_frame = frame
            try:
                await self._handle_setup(self._setup)
            except Exception as e:
                await self._send_error(f"unable to setup RTVI: {e}")

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
            await self._send_error(f"invalid message: {e}")
            return

        try:
            success = True
            error = None
            match message.type:
                case "setup":
                    setup = None
                    if message.data:
                        setup = message.data.setup
                    await self._handle_setup(message.id, setup)
                case "config-update":
                    await self._handle_config_update(message.data.config)
                case "llm-get-context":
                    await self._handle_llm_get_context()
                case "llm-append-context":
                    await self._handle_llm_append_context(message.data.llm)
                case "llm-update-context":
                    await self._handle_llm_update_context(message.data.llm)
                case "tts-speak":
                    await self._handle_tts_speak(message.data.tts)
                case "tts-interrupt":
                    await self._handle_tts_interrupt()
                case _:
                    success = False
                    error = f"unsupported type {message.type}"

            await self._send_response(message.id, success, error)
        except ValidationError as e:
            await self._send_response(message.id, False, f"invalid message: {e}")
        except Exception as e:
            await self._send_response(message.id, False, f"{e}")

    async def _handle_setup(self, setup: RTVISetup | None):
        model = DEFAULT_MODEL
        if setup and setup.config and setup.config.llm and setup.config.llm.model:
            model = setup.config.llm.model

        messages = DEFAULT_MESSAGES
        if setup and setup.config and setup.config.llm and setup.config.llm.messages:
            messages = setup.config.llm.messages

        voice = DEFAULT_VOICE
        if setup and setup.config and setup.config.tts and setup.config.tts.voice:
            voice = setup.config.tts.voice

        self._tma_in = LLMUserResponseAggregator(messages)
        self._tma_out = LLMAssistantResponseAggregator(messages)

        self._llm = self._llm_cls(
            name="LLM",
            base_url=self._llm_base_url,
            api_key=self._llm_api_key,
            model=model)

        self._tts = self._tts_cls(name="TTS", api_key=self._tts_api_key, voice_id=voice)

        # TODO-CB: Eventually we'll need to switch the context aggregators to use the
        # OpenAI context frames instead of message frames
        context = OpenAILLMContext(messages=messages)
        self._fc = FunctionCaller(context)

        self._tts_text = RTVITTSTextProcessor()

        pipeline = Pipeline([
            self._tma_in,
            self._llm,
            self._fc,
            self._tts,
            self._tts_text,
            self._tma_out,
            self._transport.output(),
        ])
        self._pipeline = pipeline

        parent = self.get_parent()
        if parent and self._start_frame:
            parent.link(pipeline)

            # We need to initialize the new pipeline with the same settings
            # as the initial one.
            start_frame = dataclasses.replace(self._start_frame)
            await self.push_frame(start_frame)

        message = RTVIBotReady()
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _handle_config_update(self, config: RTVIConfig):
        # Change voice before LLM updates, so we can hear the new vocie.
        if config.tts and config.tts.voice:
            frame = TTSVoiceUpdateFrame(config.tts.voice)
            await self.push_frame(frame)
        if config.llm and config.llm.model:
            frame = LLMModelUpdateFrame(config.llm.model)
            await self.push_frame(frame)
        if config.llm and config.llm.messages:
            frame = LLMMessagesUpdateFrame(config.llm.messages)
            await self.push_frame(frame)

    async def _handle_llm_get_context(self):
        data = RTVILLMContextMessageData(messages=self._tma_in.messages)
        message = RTVILLMContextMessage(data=data)
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _handle_llm_append_context(self, data: RTVILLMMessageData):
        if data and data.messages:
            frame = LLMMessagesAppendFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_llm_update_context(self, data: RTVILLMMessageData):
        if data and data.messages:
            frame = LLMMessagesUpdateFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_tts_speak(self, data: RTVITTSMessageData):
        if data and data.text:
            if data.interrupt:
                await self._handle_tts_interrupt()
            frame = TTSSpeakFrame(text=data.text)
            await self.push_frame(frame)

    async def _handle_tts_interrupt(self):
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

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
            if parent and self._start_frame:
                parent.link(pipeline)

        message = RTVIResponse(id=id, data=RTVIResponseData(success=success, error=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)
