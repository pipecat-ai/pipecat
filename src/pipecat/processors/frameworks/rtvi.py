#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    DataFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence

RTVI_PROTOCOL_VERSION = "0.2"

ActionResult = Union[bool, int, float, str, list, dict]


class RTVIServiceOption(BaseModel):
    name: str
    type: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, "RTVIServiceOptionConfig"], Awaitable[None]] = Field(
        exclude=True
    )


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
    result: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, Dict[str, Any]], Awaitable[ActionResult]] = Field(
        exclude=True
    )
    _arguments_dict: Dict[str, RTVIActionArgument] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._arguments_dict = {}
        for arg in self.arguments:
            self._arguments_dict[arg.name] = arg
        return super().model_post_init(__context)


class RTVIServiceOptionConfig(BaseModel):
    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    config: List[RTVIServiceConfig]


#
# Client -> Pipecat messages.
#


class RTVIUpdateConfig(BaseModel):
    config: List[RTVIServiceConfig]
    interrupt: bool = False


class RTVIActionRunArgument(BaseModel):
    name: str
    value: Any


class RTVIActionRun(BaseModel):
    service: str
    action: str
    arguments: Optional[List[RTVIActionRunArgument]] = None


@dataclass
class RTVIActionFrame(DataFrame):
    rtvi_action_run: RTVIActionRun
    message_id: Optional[str] = None


class RTVIMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None


#
# Pipecat -> Client responses and messages.
#


class RTVIErrorResponseData(BaseModel):
    error: str


class RTVIErrorResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error-response"] = "error-response"
    id: str
    data: RTVIErrorResponseData


class RTVIErrorData(BaseModel):
    error: str
    fatal: bool


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


class RTVIDescribeActionsData(BaseModel):
    actions: List[RTVIAction]


class RTVIDescribeActions(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["actions-available"] = "actions-available"
    id: str
    data: RTVIDescribeActionsData


class RTVIConfigResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["config"] = "config"
    id: str
    data: RTVIConfig


class RTVIActionResponseData(BaseModel):
    result: ActionResult


class RTVIActionResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["action-response"] = "action-response"
    id: str
    data: RTVIActionResponseData


class RTVIBotReadyData(BaseModel):
    version: str
    config: List[RTVIServiceConfig]


class RTVIBotReady(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-ready"] = "bot-ready"
    id: str
    data: RTVIBotReadyData


class RTVILLMFunctionCallMessageData(BaseModel):
    function_name: str
    tool_call_id: str
    args: dict


class RTVILLMFunctionCallMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-function-call"] = "llm-function-call"
    data: RTVILLMFunctionCallMessageData


class RTVILLMFunctionCallStartMessageData(BaseModel):
    function_name: str


class RTVILLMFunctionCallStartMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-function-call-start"] = "llm-function-call-start"
    data: RTVILLMFunctionCallStartMessageData


class RTVILLMFunctionCallResultData(BaseModel):
    function_name: str
    tool_call_id: str
    arguments: dict
    result: dict | str


class RTVIBotLLMStartedMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-llm-started"] = "bot-llm-started"


class RTVIBotLLMStoppedMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-llm-stopped"] = "bot-llm-stopped"


class RTVIBotTTSStartedMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-tts-started"] = "bot-tts-started"


class RTVIBotTTSStoppedMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-tts-stopped"] = "bot-tts-stopped"


class RTVITextMessageData(BaseModel):
    text: str


class RTVIBotTranscriptionMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-transcription"] = "bot-transcription"
    data: RTVITextMessageData


class RTVIBotLLMTextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-llm-text"] = "bot-llm-text"
    data: RTVITextMessageData


class RTVIBotTTSTextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-tts-text"] = "bot-tts-text"
    data: RTVITextMessageData


class RTVIAudioMessageData(BaseModel):
    audio: str
    sample_rate: int
    num_channels: int


class RTVIBotTTSAudioMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-tts-audio"] = "bot-tts-audio"
    data: RTVIAudioMessageData


class RTVIUserTranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class RTVIUserTranscriptionMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVIUserTranscriptionMessageData


class RTVIUserLLMTextMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-llm-text"] = "user-llm-text"
    data: RTVITextMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIBotStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class RTVIBotStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class RTVIMetricsMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["metrics"] = "metrics"
    data: Mapping[str, Any]


class RTVIFrameProcessor(FrameProcessor):
    def __init__(self, direction: FrameDirection = FrameDirection.DOWNSTREAM, **kwargs):
        super().__init__(**kwargs)
        self._direction = direction

    async def _push_transport_message_urgent(self, model: BaseModel, exclude_none: bool = True):
        frame = TransportMessageUrgentFrame(message=model.model_dump(exclude_none=exclude_none))
        await self.push_frame(frame, self._direction)


class RTVISpeakingProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            await self._handle_interruptions(frame)
        elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            await self._handle_bot_speaking(frame)

    async def _handle_interruptions(self, frame: Frame):
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVIUserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVIUserStoppedSpeakingMessage()

        if message:
            await self._push_transport_message_urgent(message)

    async def _handle_bot_speaking(self, frame: Frame):
        message = None
        if isinstance(frame, BotStartedSpeakingFrame):
            message = RTVIBotStartedSpeakingMessage()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = RTVIBotStoppedSpeakingMessage()

        if message:
            await self._push_transport_message_urgent(message)


class RTVIUserTranscriptionProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_user_transcriptions(frame)

    async def _handle_user_transcriptions(self, frame: Frame):
        message = None
        if isinstance(frame, TranscriptionFrame):
            message = RTVIUserTranscriptionMessage(
                data=RTVIUserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=True
                )
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            message = RTVIUserTranscriptionMessage(
                data=RTVIUserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=False
                )
            )

        if message:
            await self._push_transport_message_urgent(message)


class RTVIUserLLMTextProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            await self._handle_context(frame)

    async def _handle_context(self, frame: OpenAILLMContextFrame):
        messages = frame.context.messages
        if len(messages) > 0:
            message = messages[-1]
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    text = " ".join(item["text"] for item in content if "text" in item)
                else:
                    text = content
                rtvi_message = RTVIUserLLMTextMessage(data=RTVITextMessageData(text=text))
                await self._push_transport_message_urgent(rtvi_message)


class RTVIBotTranscriptionProcessor(RTVIFrameProcessor):
    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self._push_aggregation()
        elif isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self._push_aggregation()

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            message = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text=self._aggregation))
            await self._push_transport_message_urgent(message)
            self._aggregation = ""


class RTVIBotLLMProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._push_transport_message_urgent(RTVIBotLLMStartedMessage())
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._push_transport_message_urgent(RTVIBotLLMStoppedMessage())
        elif type(frame) is TextFrame:
            message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=frame.text))
            await self._push_transport_message_urgent(message)


class RTVIBotTTSProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            await self._push_transport_message_urgent(RTVIBotTTSStartedMessage())
        elif isinstance(frame, TTSStoppedFrame):
            await self._push_transport_message_urgent(RTVIBotTTSStoppedMessage())
        elif type(frame) is TextFrame:
            message = RTVIBotTTSTextMessage(data=RTVITextMessageData(text=frame.text))
            await self._push_transport_message_urgent(message)


class RTVIMetricsProcessor(RTVIFrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            await self._handle_metrics(frame)

    async def _handle_metrics(self, frame: MetricsFrame):
        metrics = {}
        for d in frame.data:
            if isinstance(d, TTFBMetricsData):
                if "ttfb" not in metrics:
                    metrics["ttfb"] = []
                metrics["ttfb"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, ProcessingMetricsData):
                if "processing" not in metrics:
                    metrics["processing"] = []
                metrics["processing"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, LLMUsageMetricsData):
                if "tokens" not in metrics:
                    metrics["tokens"] = []
                metrics["tokens"].append(d.value.model_dump(exclude_none=True))
            elif isinstance(d, TTSUsageMetricsData):
                if "characters" not in metrics:
                    metrics["characters"] = []
                metrics["characters"].append(d.model_dump(exclude_none=True))

        message = RTVIMetricsMessage(data=metrics)
        await self._push_transport_message_urgent(message)


class RTVIProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        config: RTVIConfig = RTVIConfig(config=[]),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._config = config

        self._pipeline: FrameProcessor | None = None

        self._bot_ready = False
        self._client_ready = False
        self._client_ready_id = ""

        self._registered_actions: Dict[str, RTVIAction] = {}
        self._registered_services: Dict[str, RTVIService] = {}

        # A task to process incoming action frames.
        self._action_queue = asyncio.Queue()
        self._action_task = self.get_event_loop().create_task(self._action_task_handler())

        # A task to process incoming transport messages.
        self._message_queue = asyncio.Queue()
        self._message_task = self.get_event_loop().create_task(self._message_task_handler())

        self._register_event_handler("on_client_ready")

    def register_action(self, action: RTVIAction):
        id = self._action_id(action.service, action.action)
        self._registered_actions[id] = action

    def register_service(self, service: RTVIService):
        self._registered_services[service.name] = service

    async def set_client_ready(self):
        self._client_ready = True
        await self._call_event_handler("on_client_ready")

    async def set_bot_ready(self):
        self._bot_ready = True
        await self._update_config(self._config, False)
        await self._send_bot_ready()

    async def interrupt_bot(self):
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def send_error(self, error: str):
        message = RTVIError(data=RTVIErrorData(error=error, fatal=False))
        await self._push_transport_message(message)

    async def handle_message(self, message: RTVIMessage):
        await self._message_queue.put(message)

    async def handle_function_call(
        self,
        function_name: str,
        tool_call_id: str,
        arguments: dict,
        llm: FrameProcessor,
        context: OpenAILLMContext,
        result_callback,
    ):
        fn = RTVILLMFunctionCallMessageData(
            function_name=function_name, tool_call_id=tool_call_id, args=arguments
        )
        message = RTVILLMFunctionCallMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def handle_function_call_start(
        self, function_name: str, llm: FrameProcessor, context: OpenAILLMContext
    ):
        fn = RTVILLMFunctionCallStartMessageData(function_name=function_name)
        message = RTVILLMFunctionCallStartMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, ErrorFrame):
            await self._send_error_frame(frame)
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        # Data frames
        elif isinstance(frame, TransportMessageFrame):
            await self._handle_transport_message(frame)
        elif isinstance(frame, RTVIActionFrame):
            await self._action_queue.put(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._pipeline:
            await self._pipeline.cleanup()

    async def _start(self, frame: StartFrame):
        pass

    async def _stop(self, frame: EndFrame):
        await self._cancel_tasks()

    async def _cancel(self, frame: CancelFrame):
        await self._cancel_tasks()

    async def _cancel_tasks(self):
        if self._action_task:
            self._action_task.cancel()
            await self._action_task
            self._action_task = None

        if self._message_task:
            self._message_task.cancel()
            await self._message_task
            self._message_task = None

    async def _push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        frame = TransportMessageUrgentFrame(message=model.model_dump(exclude_none=exclude_none))
        await self.push_frame(frame)

    async def _action_task_handler(self):
        while True:
            try:
                frame = await self._action_queue.get()
                await self._handle_action(frame.message_id, frame.rtvi_action_run)
                self._action_queue.task_done()
            except asyncio.CancelledError:
                break

    async def _message_task_handler(self):
        while True:
            try:
                message = await self._message_queue.get()
                await self._handle_message(message)
                self._message_queue.task_done()
            except asyncio.CancelledError:
                break

    async def _handle_transport_message(self, frame: TransportMessageFrame):
        try:
            message = RTVIMessage.model_validate(frame.message)
            await self._message_queue.put(message)
        except ValidationError as e:
            await self.send_error(f"Invalid RTVI transport message: {e}")
            logger.warning(f"Invalid RTVI transport message: {e}")

    async def _handle_message(self, message: RTVIMessage):
        try:
            match message.type:
                case "client-ready":
                    await self._handle_client_ready(message.id)
                case "describe-actions":
                    await self._handle_describe_actions(message.id)
                case "describe-config":
                    await self._handle_describe_config(message.id)
                case "get-config":
                    await self._handle_get_config(message.id)
                case "update-config":
                    update_config = RTVIUpdateConfig.model_validate(message.data)
                    await self._handle_update_config(message.id, update_config)
                case "action":
                    action = RTVIActionRun.model_validate(message.data)
                    action_frame = RTVIActionFrame(message_id=message.id, rtvi_action_run=action)
                    await self._action_queue.put(action_frame)
                case "llm-function-call-result":
                    data = RTVILLMFunctionCallResultData.model_validate(message.data)
                    await self._handle_function_call_result(data)

                case _:
                    await self._send_error_response(message.id, f"Unsupported type {message.type}")

        except ValidationError as e:
            await self._send_error_response(message.id, f"Invalid message: {e}")
            logger.warning(f"Invalid message: {e}")
        except Exception as e:
            await self._send_error_response(message.id, f"Exception processing message: {e}")
            logger.warning(f"Exception processing message: {e}")

    async def _handle_client_ready(self, request_id: str):
        self._client_ready_id = request_id
        await self.set_client_ready()

    async def _handle_describe_config(self, request_id: str):
        services = list(self._registered_services.values())
        message = RTVIDescribeConfig(id=request_id, data=RTVIDescribeConfigData(config=services))
        await self._push_transport_message(message)

    async def _handle_describe_actions(self, request_id: str):
        actions = list(self._registered_actions.values())
        message = RTVIDescribeActions(id=request_id, data=RTVIDescribeActionsData(actions=actions))
        await self._push_transport_message(message)

    async def _handle_get_config(self, request_id: str):
        message = RTVIConfigResponse(id=request_id, data=self._config)
        await self._push_transport_message(message)

    def _update_config_option(self, service: str, config: RTVIServiceOptionConfig):
        for service_config in self._config.config:
            if service_config.service == service:
                for option_config in service_config.options:
                    if option_config.name == config.name:
                        option_config.value = config.value
                        return
                # If we couldn't find a value for this config, we simply need to
                # add it.
                service_config.options.append(config)

    async def _update_service_config(self, config: RTVIServiceConfig):
        service = self._registered_services[config.service]
        for option in config.options:
            handler = service._options_dict[option.name].handler
            await handler(self, service.name, option)
            self._update_config_option(service.name, option)

    async def _update_config(self, data: RTVIConfig, interrupt: bool):
        if interrupt:
            await self.interrupt_bot()
        for service_config in data.config:
            await self._update_service_config(service_config)

    async def _handle_update_config(self, request_id: str, data: RTVIUpdateConfig):
        await self._update_config(RTVIConfig(config=data.config), data.interrupt)
        await self._handle_get_config(request_id)

    async def _handle_function_call_result(self, data):
        frame = FunctionCallResultFrame(
            function_name=data.function_name,
            tool_call_id=data.tool_call_id,
            arguments=data.arguments,
            result=data.result,
        )
        await self.push_frame(frame)

    async def _handle_action(self, request_id: str | None, data: RTVIActionRun):
        action_id = self._action_id(data.service, data.action)
        if action_id not in self._registered_actions:
            await self._send_error_response(request_id, f"Action {action_id} not registered")
            return
        action = self._registered_actions[action_id]
        arguments = {}
        if data.arguments:
            for arg in data.arguments:
                arguments[arg.name] = arg.value
        result = await action.handler(self, action.service, arguments)
        # Only send a response if request_id is present. Things that don't care about
        # action responses (such as webhooks) don't set a request_id
        if request_id:
            message = RTVIActionResponse(id=request_id, data=RTVIActionResponseData(result=result))
            await self._push_transport_message(message)

    async def _send_bot_ready(self):
        message = RTVIBotReady(
            id=self._client_ready_id,
            data=RTVIBotReadyData(version=RTVI_PROTOCOL_VERSION, config=self._config.config),
        )
        await self._push_transport_message(message)

    async def _send_error_frame(self, frame: ErrorFrame):
        message = RTVIError(data=RTVIErrorData(error=frame.error, fatal=frame.fatal))
        await self._push_transport_message(message)

    async def _send_error_response(self, id: str, error: str):
        message = RTVIErrorResponse(id=id, data=RTVIErrorResponseData(error=error))
        await self._push_transport_message(message)

    def _action_id(self, service: str, action: str) -> str:
        return f"{service}:{action}"
