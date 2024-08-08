#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from loguru import logger

RTVI_PROTOCOL_VERSION = "0.1"


class RTVIServiceOption(BaseModel):
    name: str
    type: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, "RTVIServiceOptionConfig"],
                      Awaitable[None]] = Field(exclude=True)


class RTVIService(BaseModel):
    name: str
    options: List[RTVIServiceOption]
    _options_dict: Dict[str, RTVIServiceOption] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._options_dict = {}
        for option in self.options:
            self._options_dict[option.name] = option
        return super().model_post_init(__context)


class RTVIActionArgumentData(BaseModel):
    name: str
    value: Any


class RTVIActionArgument(BaseModel):
    name: str
    type: Literal["bool", "number", "string", "array", "object"]


class RTVIAction(BaseModel):
    service: str
    action: str
    arguments: List[RTVIActionArgument] = []
    handler: Callable[["RTVIProcessor", str, Dict[str, Any]], Awaitable[None]] = Field(exclude=True)
    _arguments_dict: Dict[str, RTVIActionArgument] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._arguments_dict = {}
        for arg in self.arguments:
            self._arguments_dict[arg.name] = arg
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


class RTVIActionRunArgument(BaseModel):
    name: str
    value: Any


class RTVIActionRun(BaseModel):
    service: str
    action: str
    arguments: Optional[List[RTVIActionRunArgument]] = None


class RTVIMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None

#
# Pipecat -> Client responses and messages.
#


class RTVIErrorResponseData(BaseModel):
    error: Optional[str] = None


class RTVIErrorResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error-response"] = "error-response"
    id: str
    data: RTVIErrorResponseData


class RTVIErrorData(BaseModel):
    message: str


class RTVIError(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error"] = "error"
    data: RTVIErrorData


class RTVIDescribeConfigData(BaseModel):
    config: List[RTVIService]


class RTVIDescribeConfig(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["config-available"] = "config-available"
    id: str
    data: RTVIDescribeConfigData


class RTVIConfigResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["config"] = "config"
    id: str
    data: RTVIConfig


class RTVILLMContextMessageData(BaseModel):
    messages: List[dict]


class RTVILLMContextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-context"] = "llm-context"
    data: RTVILLMContextMessageData


class RTVIBotReadyData(BaseModel):
    version: str
    config: List[RTVIServiceConfig]


class RTVIBotReady(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-ready"] = "bot-ready"
    data: RTVIBotReadyData


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


class RTVIProcessor(FrameProcessor):

    def __init__(self, default_config: RTVIConfig):
        super().__init__()
        self._config = default_config

        self._start_config: RTVIConfig | None = None
        self._pipeline: FrameProcessor | None = None

        self._registered_actions: Dict[str, RTVIAction] = {}
        self._registered_services: Dict[str, RTVIService] = {}

        self._frame_handler_task = self.get_event_loop().create_task(self._frame_handler())
        self._frame_queue = asyncio.Queue()

    def register_action(self, action: RTVIAction):
        id = self._action_id(action.service, action.action)
        self._registered_actions[id] = action

    def register_service(self, service: RTVIService):
        self._registered_services[service.name] = service

    def configure_on_start(self, config: RTVIConfig):
        self._start_config = config

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, StartFrame):
            await self._start(frame)
            await self._internal_push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self._internal_push_frame(frame, direction)
            await self._stop(frame)
        # Other frames
        else:
            await self._internal_push_frame(frame, direction)

    async def cleanup(self):
        if self._pipeline:
            await self._pipeline.cleanup()

    async def _start(self, frame: StartFrame):
        await self._update_config(self._config)
        if self._start_config:
            await self._update_config(self._start_config)
        await self._send_bot_ready()

    async def _stop(self, frame: EndFrame):
        await self._frame_handler_task

    async def _cancel(self, frame: CancelFrame):
        self._frame_handler_task.cancel()
        await self._frame_handler_task

    async def _internal_push_frame(
            self,
            frame: Frame | None,
            direction: FrameDirection | None = FrameDirection.DOWNSTREAM):
        await self._frame_queue.put((frame, direction))

    async def _frame_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._frame_queue.get()
                await self._handle_frame(frame, direction)
                self._frame_queue.task_done()
                running = not isinstance(frame, EndFrame)
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
        # TODO(aleix): Once we add support for using custom pipelines, the STTs will
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
            match message.type:
                case "describe-config":
                    await self._handle_describe_config(message.id)
                case "get-config":
                    await self._handle_get_config(message.id)
                case "update-config":
                    config = RTVIConfig.model_validate(message.data)
                    await self._handle_update_config(message.id, config)
                case "action":
                    action = RTVIActionRun.model_validate(message.data)
                    await self._handle_action(message.id, action)
                # case "llm-get-context":
                #     await self._handle_llm_get_context()
                case _:
                    await self._send_error_response(message.id, f"Unsupported type {message.type}")

        except ValidationError as e:
            await self._send_error_response(message.id, f"Invalid incoming message: {e}")
            logger.warning(f"Invalid incoming  message: {e}")
        except Exception as e:
            await self._send_error_response(message.id, f"Exception processing message: {e}")
            logger.warning(f"Exception processing message: {e}")

    async def _handle_describe_config(self, request_id: str):
        services = list(self._registered_services.values())
        message = RTVIDescribeConfig(id=request_id, data=RTVIDescribeConfigData(config=services))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _handle_get_config(self, request_id: str):
        message = RTVIConfigResponse(id=request_id, data=self._config)
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    def _update_config_option(self, service: str, config: RTVIServiceOptionConfig):
        for service_config in self._config.config:
            if service_config.service == service:
                for option_config in service_config.options:
                    if option_config.name == config.name:
                        option_config.value = config.value
                        return

    async def _update_service_config(self, config: RTVIServiceConfig):
        service = self._registered_services[config.service]
        for option in config.options:
            handler = service._options_dict[option.name].handler
            await handler(self, service.name, option)
            self._update_config_option(service.name, option)

    async def _update_config(self, data: RTVIConfig):
        for service_config in data.config:
            await self._update_service_config(service_config)

    async def _handle_update_config(self, request_id: str, data: RTVIConfig):
        await self._update_config(data)
        await self._handle_get_config(request_id)

    async def _handle_action(self, request_id: str, data: RTVIActionRun):
        action_id = self._action_id(data.service, data.action)
        if action_id not in self._registered_actions:
            await self._send_error_response(request_id, f"Action {action_id} not registered")
            return
        action = self._registered_actions[action_id]
        arguments = {}
        if data.arguments:
            for arg in data.arguments:
                arguments[arg.name] = arg.value
        await action.handler(self, action.service, arguments)

    # async def _handle_llm_get_context(self):
    #     data = RTVILLMContextMessageData(messages=self._tma_in.messages)
    #     message = RTVILLMContextMessage(data=data)
    #     frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
    #     await self.push_frame(frame)

    async def _send_bot_ready(self):
        message = RTVIBotReady(
            data=RTVIBotReadyData(
                version=RTVI_PROTOCOL_VERSION,
                config=self._config.config))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _send_error(self, error: str):
        message = RTVIError(data=RTVIErrorData(message=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _send_error_response(self, id: str, error: str):
        message = RTVIErrorResponse(id=id, data=RTVIErrorResponseData(error=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    def _action_id(self, service: str, action: str) -> str:
        return f"{service}/{action}"
