#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI (Real-Time Voice Interface) protocol implementation for Pipecat.

This module provides the RTVI protocol implementation for real-time voice interactions
between clients and AI agents. It includes message handling, action processing,
and frame observation for the RTVI protocol.
"""

import asyncio
import base64
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
    EndTaskFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    TransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import (
    FunctionCallParams,  # TODO(aleix): we shouldn't import `services` from `processors`
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.utils.string import match_endofsentence

RTVI_PROTOCOL_VERSION = "1.0.0"

RTVI_MESSAGE_LABEL = "rtvi-ai"
RTVIMessageLiteral = Literal["rtvi-ai"]

ActionResult = Union[bool, int, float, str, list, dict]


class RTVIServiceOption(BaseModel):
    """Configuration option for an RTVI service.

    Defines a configurable option that can be set for an RTVI service,
    including its name, type, and handler function.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    type: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, "RTVIServiceOptionConfig"], Awaitable[None]] = Field(
        exclude=True
    )


class RTVIService(BaseModel):
    """An RTVI service definition.

    Represents a service that can be configured and used within the RTVI protocol,
    containing a name and list of configurable options.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    name: str
    options: List[RTVIServiceOption]
    _options_dict: Dict[str, RTVIServiceOption] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        """Initialize the options dictionary after model creation."""
        self._options_dict = {}
        for option in self.options:
            self._options_dict[option.name] = option
        return super().model_post_init(__context)


class RTVIActionArgumentData(BaseModel):
    """Data for an RTVI action argument.

    Contains the name and value of an argument passed to an RTVI action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIActionArgument(BaseModel):
    """Definition of an RTVI action argument.

    Specifies the name and expected type of an argument for an RTVI action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    type: Literal["bool", "number", "string", "array", "object"]


class RTVIAction(BaseModel):
    """An RTVI action definition.

    Represents an action that can be executed within the RTVI protocol,
    including its service, name, arguments, and handler function.

    .. deprecated:: 0.0.75
       Actions have been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    action: str
    arguments: List[RTVIActionArgument] = Field(default_factory=list)
    result: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, Dict[str, Any]], Awaitable[ActionResult]] = Field(
        exclude=True
    )
    _arguments_dict: Dict[str, RTVIActionArgument] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        """Initialize the arguments dictionary after model creation."""
        self._arguments_dict = {}
        for arg in self.arguments:
            self._arguments_dict[arg.name] = arg
        return super().model_post_init(__context)


class RTVIServiceOptionConfig(BaseModel):
    """Configuration value for an RTVI service option.

    Contains the name and value to set for a specific service option.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    """Configuration for an RTVI service.

    Contains the service name and list of option configurations to apply.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    """Complete RTVI configuration.

    Contains the full configuration for all RTVI services.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    config: List[RTVIServiceConfig]


#
# Client -> Pipecat messages.
#


# deprecated
class RTVIUpdateConfig(BaseModel):
    """Request to update RTVI configuration.

    Contains new configuration settings and whether to interrupt the bot.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    config: List[RTVIServiceConfig]
    interrupt: bool = False


class RTVIActionRunArgument(BaseModel):
    """Argument for running an RTVI action.

    Contains the name and value of an argument to pass to an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIActionRun(BaseModel):
    """Request to run an RTVI action.

    Contains the service, action name, and optional arguments.

    .. deprecated:: 0.0.75
       Actions have been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    action: str
    arguments: Optional[List[RTVIActionRunArgument]] = None


@dataclass
class RTVIActionFrame(DataFrame):
    """Frame containing an RTVI action to execute.

    Parameters:
        rtvi_action_run: The action to execute.
        message_id: Optional message ID for response correlation.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    rtvi_action_run: RTVIActionRun
    message_id: Optional[str] = None


class RTVIRawClientMessageData(BaseModel):
    """Data structure expected from client messages sent to the RTVI server."""

    t: str
    d: Optional[Any] = None


class RTVIClientMessage(BaseModel):
    """Cleansed data structure for client messages for handling."""

    msg_id: str
    type: str
    data: Optional[Any] = None


@dataclass
class RTVIClientMessageFrame(SystemFrame):
    """A frame for sending messages from the client to the RTVI server.

    This frame is meant for custom messaging from the client to the server
    and expects a server-response message.
    """

    msg_id: str
    type: str
    data: Optional[Any] = None


@dataclass
class RTVIServerResponseFrame(SystemFrame):
    """A frame for responding to a client RTVI message.

    This frame should be sent in response to an RTVIClientMessageFrame
    and include the original RTVIClientMessageFrame to ensure the response
    is properly attributed to the original request. To respond with an error,
    set the `error` field to a string describing the error. This will result
    in the client receiving a `response-error` message instead of a
    `server-response` message.
    """

    client_msg: RTVIClientMessageFrame
    data: Optional[Any] = None
    error: Optional[str] = None


class RTVIRawServerResponseData(BaseModel):
    """Data structure for server responses to client messages."""

    t: str
    d: Optional[Any] = None


class RTVIServerResponse(BaseModel):
    """The RTVI-formatted message response from the server to the client.

    This message is used to respond to custom messages sent by the client.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["server-response"] = "server-response"
    id: str
    data: RTVIRawServerResponseData


class RTVIMessage(BaseModel):
    """Base RTVI message structure.

    Represents the standard format for RTVI protocol messages.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None


#
# Pipecat -> Client responses and messages.
#


class RTVIErrorResponseData(BaseModel):
    """Data for an RTVI error response.

    Contains the error message to send back to the client.
    """

    error: str


class RTVIErrorResponse(BaseModel):
    """RTVI error response message.

    RTVI Formatted error response message for relaying failed client requests.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["error-response"] = "error-response"
    id: str
    data: RTVIErrorResponseData


class RTVIErrorData(BaseModel):
    """Data for an RTVI error event.

    Contains error information including whether it's fatal.
    """

    error: str
    fatal: bool  # Indicates the pipeline has stopped due to this error


class RTVIError(BaseModel):
    """RTVI error event message.

    RTVI Formatted error message for relaying errors in the pipeline.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["error"] = "error"
    data: RTVIErrorData


class RTVIDescribeConfigData(BaseModel):
    """Data for describing available RTVI configuration.

    Contains the list of available services and their options.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    config: List[RTVIService]


class RTVIDescribeConfig(BaseModel):
    """Message describing available RTVI configuration.

    Sent in response to a describe-config request.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["config-available"] = "config-available"
    id: str
    data: RTVIDescribeConfigData


class RTVIDescribeActionsData(BaseModel):
    """Data for describing available RTVI actions.

    Contains the list of available actions that can be executed.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    actions: List[RTVIAction]


class RTVIDescribeActions(BaseModel):
    """Message describing available RTVI actions.

    Sent in response to a describe-actions request.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["actions-available"] = "actions-available"
    id: str
    data: RTVIDescribeActionsData


class RTVIConfigResponse(BaseModel):
    """Response containing current RTVI configuration.

    Sent in response to a get-config request.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["config"] = "config"
    id: str
    data: RTVIConfig


class RTVIActionResponseData(BaseModel):
    """Data for an RTVI action response.

    Contains the result of executing an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    result: ActionResult


class RTVIActionResponse(BaseModel):
    """Response to an RTVI action execution.

    Sent after successfully executing an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["action-response"] = "action-response"
    id: str
    data: RTVIActionResponseData


class AboutClientData(BaseModel):
    """Data about the RTVI client.

    Contains information about the client, including which RTVI library it
    is using, what platform it is on and any additional details, if available.
    """

    library: str
    library_version: Optional[str] = None
    platform: Optional[str] = None
    platform_version: Optional[str] = None
    platform_details: Optional[Any] = None


class RTVIClientReadyData(BaseModel):
    """Data format of client ready messages.

    Contains the RTVIprotocol version and client information.
    """

    version: str
    about: AboutClientData


class RTVIBotReadyData(BaseModel):
    """Data for bot ready notification.

    Contains protocol version and initial configuration.
    """

    version: str
    # The config field is deprecated and will not be included if
    # the client's rtvi version is 1.0.0 or higher.
    config: Optional[List[RTVIServiceConfig]] = None
    about: Optional[Mapping[str, Any]] = None


class RTVIBotReady(BaseModel):
    """Message indicating bot is ready for interaction.

    Sent after bot initialization is complete.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-ready"] = "bot-ready"
    id: str
    data: RTVIBotReadyData


class RTVILLMFunctionCallMessageData(BaseModel):
    """Data for LLM function call notification.

    Contains function call details including name, ID, and arguments.
    """

    function_name: str
    tool_call_id: str
    args: Mapping[str, Any]


class RTVILLMFunctionCallMessage(BaseModel):
    """Message notifying of an LLM function call.

    Sent when the LLM makes a function call.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["llm-function-call"] = "llm-function-call"
    data: RTVILLMFunctionCallMessageData


class RTVIAppendToContextData(BaseModel):
    """Data format for appending messages to the context.

    Contains the role, content, and whether to run the message immediately.
    """

    role: Literal["user", "assistant"] | str
    content: Any
    run_immediately: bool = False


class RTVIAppendToContext(BaseModel):
    """RTVI Message format to append content to the LLM context."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["append-to-context"] = "append-to-context"
    data: RTVIAppendToContextData


class RTVILLMFunctionCallStartMessageData(BaseModel):
    """Data for LLM function call start notification.

    Contains the function name being called.
    """

    function_name: str


class RTVILLMFunctionCallStartMessage(BaseModel):
    """Message notifying that an LLM function call has started.

    Sent when the LLM begins a function call.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["llm-function-call-start"] = "llm-function-call-start"
    data: RTVILLMFunctionCallStartMessageData


class RTVILLMFunctionCallResultData(BaseModel):
    """Data for LLM function call result.

    Contains function call details and result.
    """

    function_name: str
    tool_call_id: str
    arguments: dict
    result: dict | str


class RTVIBotLLMStartedMessage(BaseModel):
    """Message indicating bot LLM processing has started."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-llm-started"] = "bot-llm-started"


class RTVIBotLLMStoppedMessage(BaseModel):
    """Message indicating bot LLM processing has stopped."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-llm-stopped"] = "bot-llm-stopped"


class RTVIBotTTSStartedMessage(BaseModel):
    """Message indicating bot TTS processing has started."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-tts-started"] = "bot-tts-started"


class RTVIBotTTSStoppedMessage(BaseModel):
    """Message indicating bot TTS processing has stopped."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-tts-stopped"] = "bot-tts-stopped"


class RTVITextMessageData(BaseModel):
    """Data for text-based RTVI messages.

    Contains text content.
    """

    text: str


class RTVIBotTranscriptionMessage(BaseModel):
    """Message containing bot transcription text.

    Sent when the bot's speech is transcribed.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-transcription"] = "bot-transcription"
    data: RTVITextMessageData


class RTVIBotLLMTextMessage(BaseModel):
    """Message containing bot LLM text output.

    Sent when the bot's LLM generates text.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-llm-text"] = "bot-llm-text"
    data: RTVITextMessageData


class RTVIBotTTSTextMessage(BaseModel):
    """Message containing bot TTS text output.

    Sent when text is being processed by TTS.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-tts-text"] = "bot-tts-text"
    data: RTVITextMessageData


class RTVIAudioMessageData(BaseModel):
    """Data for audio-based RTVI messages.

    Contains audio data and metadata.
    """

    audio: str
    sample_rate: int
    num_channels: int


class RTVIBotTTSAudioMessage(BaseModel):
    """Message containing bot TTS audio output.

    Sent when the bot's TTS generates audio.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-tts-audio"] = "bot-tts-audio"
    data: RTVIAudioMessageData


class RTVIUserTranscriptionMessageData(BaseModel):
    """Data for user transcription messages.

    Contains transcription text and metadata.
    """

    text: str
    user_id: str
    timestamp: str
    final: bool


class RTVIUserTranscriptionMessage(BaseModel):
    """Message containing user transcription.

    Sent when user speech is transcribed.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVIUserTranscriptionMessageData


class RTVIUserLLMTextMessage(BaseModel):
    """Message containing user text input for LLM.

    Sent when user text is processed by the LLM.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["user-llm-text"] = "user-llm-text"
    data: RTVITextMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    """Message indicating user has started speaking."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    """Message indicating user has stopped speaking."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIBotStartedSpeakingMessage(BaseModel):
    """Message indicating bot has started speaking."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class RTVIBotStoppedSpeakingMessage(BaseModel):
    """Message indicating bot has stopped speaking."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class RTVIMetricsMessage(BaseModel):
    """Message containing performance metrics.

    Sent to provide performance and usage metrics.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["metrics"] = "metrics"
    data: Mapping[str, Any]


class RTVIServerMessage(BaseModel):
    """Generic server message.

    Used for custom server-to-client messages.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["server-message"] = "server-message"
    data: Any


@dataclass
class RTVIServerMessageFrame(SystemFrame):
    """A frame for sending server messages to the client.

    Parameters:
        data: The message data to send to the client.
    """

    data: Any

    def __str__(self):
        """String representation of the RTVI server message frame."""
        return f"{self.name}(data: {self.data})"


@dataclass
class RTVIObserverParams:
    """Parameters for configuring RTVI Observer behavior.

    Parameters:
        bot_llm_enabled: Indicates if the bot's LLM messages should be sent.
        bot_tts_enabled: Indicates if the bot's TTS messages should be sent.
        bot_speaking_enabled: Indicates if the bot's started/stopped speaking messages should be sent.
        user_llm_enabled: Indicates if the user's LLM input messages should be sent.
        user_speaking_enabled: Indicates if the user's started/stopped speaking messages should be sent.
        user_transcription_enabled: Indicates if user's transcription messages should be sent.
        metrics_enabled: Indicates if metrics messages should be sent.
        errors_enabled: Indicates if errors messages should be sent.
    """

    bot_llm_enabled: bool = True
    bot_tts_enabled: bool = True
    bot_speaking_enabled: bool = True
    user_llm_enabled: bool = True
    user_speaking_enabled: bool = True
    user_transcription_enabled: bool = True
    metrics_enabled: bool = True
    errors_enabled: bool = True


class RTVIObserver(BaseObserver):
    """Pipeline frame observer for RTVI server message handling.

    This observer monitors pipeline frames and converts them into appropriate RTVI messages
    for client communication. It handles various frame types including speech events,
    transcriptions, LLM responses, and TTS events.

    Note:
        This observer only handles outgoing messages. Incoming RTVI client messages
        are handled by the RTVIProcessor.
    """

    def __init__(
        self, rtvi: "RTVIProcessor", *, params: Optional[RTVIObserverParams] = None, **kwargs
    ):
        """Initialize the RTVI observer.

        Args:
            rtvi: The RTVI processor to push frames to.
            params: Settings to enable/disable specific messages.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._rtvi = rtvi
        self._params = params or RTVIObserverParams()
        self._bot_transcription = ""
        self._frames_seen = set()
        rtvi.set_errors_enabled(self._params.errors_enabled)

    async def on_push_frame(self, data: FramePushed):
        """Process a frame being pushed through the pipeline.

        Args:
            data: Frame push event data containing source, frame, direction, and timestamp.
        """
        src = data.source
        frame = data.frame
        direction = data.direction

        # If we have already seen this frame, let's skip it.
        if frame.id in self._frames_seen:
            return

        # This tells whether the frame is already processed. If false, we will try
        # again the next time we see the frame.
        mark_as_seen = True

        if (
            isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame))
            and self._params.user_speaking_enabled
        ):
            await self._handle_interruptions(frame)
        elif (
            isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
            and (direction == FrameDirection.UPSTREAM)
            and self._params.bot_speaking_enabled
        ):
            await self._handle_bot_speaking(frame)
        elif (
            isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and self._params.user_transcription_enabled
        ):
            await self._handle_user_transcriptions(frame)
        elif isinstance(frame, OpenAILLMContextFrame) and self._params.user_llm_enabled:
            await self._handle_context(frame)
        elif isinstance(frame, LLMFullResponseStartFrame) and self._params.bot_llm_enabled:
            await self.push_transport_message_urgent(RTVIBotLLMStartedMessage())
        elif isinstance(frame, LLMFullResponseEndFrame) and self._params.bot_llm_enabled:
            await self.push_transport_message_urgent(RTVIBotLLMStoppedMessage())
        elif isinstance(frame, LLMTextFrame) and self._params.bot_llm_enabled:
            await self._handle_llm_text_frame(frame)
        elif isinstance(frame, TTSStartedFrame) and self._params.bot_tts_enabled:
            await self.push_transport_message_urgent(RTVIBotTTSStartedMessage())
        elif isinstance(frame, TTSStoppedFrame) and self._params.bot_tts_enabled:
            await self.push_transport_message_urgent(RTVIBotTTSStoppedMessage())
        elif isinstance(frame, TTSTextFrame) and self._params.bot_tts_enabled:
            if isinstance(src, BaseOutputTransport):
                message = RTVIBotTTSTextMessage(data=RTVITextMessageData(text=frame.text))
                await self.push_transport_message_urgent(message)
            else:
                mark_as_seen = False
        elif isinstance(frame, MetricsFrame) and self._params.metrics_enabled:
            await self._handle_metrics(frame)
        elif isinstance(frame, RTVIServerMessageFrame):
            message = RTVIServerMessage(data=frame.data)
            await self.push_transport_message_urgent(message)
        elif isinstance(frame, RTVIServerResponseFrame):
            if frame.error is not None:
                await self._send_error_response(frame)
            else:
                await self._send_server_response(frame)

        if mark_as_seen:
            self._frames_seen.add(frame.id)

    async def push_transport_message_urgent(self, model: BaseModel, exclude_none: bool = True):
        """Push an urgent transport message to the RTVI processor.

        Args:
            model: The message model to send.
            exclude_none: Whether to exclude None values from the model dump.
        """
        frame = TransportMessageUrgentFrame(message=model.model_dump(exclude_none=exclude_none))
        await self._rtvi.push_frame(frame)

    async def _push_bot_transcription(self):
        """Push accumulated bot transcription as a message."""
        if len(self._bot_transcription) > 0:
            message = RTVIBotTranscriptionMessage(
                data=RTVITextMessageData(text=self._bot_transcription)
            )
            await self.push_transport_message_urgent(message)
            self._bot_transcription = ""

    async def _handle_interruptions(self, frame: Frame):
        """Handle user speaking interruption frames."""
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVIUserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVIUserStoppedSpeakingMessage()

        if message:
            await self.push_transport_message_urgent(message)

    async def _handle_bot_speaking(self, frame: Frame):
        """Handle bot speaking event frames."""
        message = None
        if isinstance(frame, BotStartedSpeakingFrame):
            message = RTVIBotStartedSpeakingMessage()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = RTVIBotStoppedSpeakingMessage()

        if message:
            await self.push_transport_message_urgent(message)

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        """Handle LLM text output frames."""
        message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=frame.text))
        await self.push_transport_message_urgent(message)

        self._bot_transcription += frame.text
        if match_endofsentence(self._bot_transcription):
            await self._push_bot_transcription()

    async def _handle_user_transcriptions(self, frame: Frame):
        """Handle user transcription frames."""
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
            await self.push_transport_message_urgent(message)

    async def _handle_context(self, frame: OpenAILLMContextFrame):
        """Process LLM context frames to extract user messages for the RTVI client."""
        try:
            messages = frame.context.messages
            if not messages:
                return

            message = messages[-1]

            # Handle Google LLM format (protobuf objects with attributes)
            if hasattr(message, "role") and message.role == "user" and hasattr(message, "parts"):
                text = "".join(part.text for part in message.parts if hasattr(part, "text"))
                if text:
                    rtvi_message = RTVIUserLLMTextMessage(data=RTVITextMessageData(text=text))
                    await self.push_transport_message_urgent(rtvi_message)

            # Handle OpenAI format (original implementation)
            elif isinstance(message, dict):
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, list):
                        text = " ".join(item["text"] for item in content if "text" in item)
                    else:
                        text = content
                    rtvi_message = RTVIUserLLMTextMessage(data=RTVITextMessageData(text=text))
                    await self.push_transport_message_urgent(rtvi_message)

        except Exception as e:
            logger.warning(f"Caught an error while trying to handle context: {e}")

    async def _handle_metrics(self, frame: MetricsFrame):
        """Handle metrics frames and convert to RTVI metrics messages."""
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
        await self.push_transport_message_urgent(message)

    async def _send_server_response(self, frame: RTVIServerResponseFrame):
        """Send a response to the client for a specific request."""
        message = RTVIServerResponse(
            id=str(frame.client_msg.msg_id),
            data=RTVIRawServerResponseData(t=frame.client_msg.type, d=frame.data),
        )
        await self.push_transport_message_urgent(message)

    async def _send_error_response(self, frame: RTVIServerResponseFrame):
        """Send a response to the client for a specific request."""
        if self._params.errors_enabled:
            message = RTVIErrorResponse(
                id=str(frame.client_msg.msg_id), data=RTVIErrorResponseData(error=frame.error)
            )
            await self.push_transport_message_urgent(message)


class RTVIProcessor(FrameProcessor):
    """Main processor for handling RTVI protocol messages and actions.

    This processor manages the RTVI protocol communication including client-server
    handshaking, configuration management, action execution, and message routing.
    It serves as the central hub for RTVI protocol operations.
    """

    def __init__(
        self,
        *,
        config: Optional[RTVIConfig] = None,
        transport: Optional[BaseTransport] = None,
        **kwargs,
    ):
        """Initialize the RTVI processor.

        Args:
            config: Initial RTVI configuration.
            transport: Transport layer for communication.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = config or RTVIConfig(config=[])

        self._bot_ready = False
        self._client_ready = False
        self._client_ready_id = ""
        self._client_version = []
        self._errors_enabled = True

        self._registered_actions: Dict[str, RTVIAction] = {}
        self._registered_services: Dict[str, RTVIService] = {}

        # A task to process incoming action frames.
        self._action_task: Optional[asyncio.Task] = None

        # A task to process incoming transport messages.
        self._message_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_bot_started")
        self._register_event_handler("on_client_ready")
        self._register_event_handler("on_client_message")

        self._input_transport = None
        self._transport = transport
        if self._transport:
            input_transport = self._transport.input()
            if isinstance(input_transport, BaseInputTransport):
                self._input_transport = input_transport
                self._input_transport.enable_audio_in_stream_on_start(False)

    def register_action(self, action: RTVIAction):
        """Register an action that can be executed via RTVI.

        Args:
            action: The action to register.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The actions API is deprecated, use server and client messages instead.",
                DeprecationWarning,
            )

        id = self._action_id(action.service, action.action)
        self._registered_actions[id] = action

    def register_service(self, service: RTVIService):
        """Register a service that can be configured via RTVI.

        Args:
            service: The service to register.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The actions API is deprecated, use server and client messages instead.",
                DeprecationWarning,
            )

        self._registered_services[service.name] = service

    async def set_client_ready(self):
        """Mark the client as ready and trigger the ready event."""
        self._client_ready = True
        await self._call_event_handler("on_client_ready")

    async def set_bot_ready(self):
        """Mark the bot as ready and send the bot-ready message."""
        self._bot_ready = True
        await self._update_config(self._config, False)
        await self._send_bot_ready()

    def set_errors_enabled(self, enabled: bool):
        """Enable or disable error message sending.

        Args:
            enabled: Whether to send error messages.
        """
        self._errors_enabled = enabled

    async def interrupt_bot(self):
        """Send a bot interruption frame upstream."""
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def send_server_message(self, data: Any):
        """Send a server message to the client."""
        message = RTVIServerMessage(data=data)
        await self._send_server_message(message)

    async def send_server_response(self, client_msg: RTVIClientMessage, data: Any):
        """Send a server response for a given client message."""
        message = RTVIServerResponse(
            id=client_msg.msg_id, data=RTVIRawServerResponseData(t=client_msg.type, d=data)
        )
        await self._send_server_message(message)

    async def send_error_response(self, client_msg: RTVIClientMessage, error: str):
        """Send an error response for a given client message."""
        await self._send_error_response(id=client_msg.msg_id, error=error)

    async def send_error(self, error: str):
        """Send an error message to the client.

        Args:
            error: The error message to send.
        """
        await self._send_error_frame(ErrorFrame(error=error))

    async def handle_message(self, message: RTVIMessage):
        """Handle an incoming RTVI message.

        Args:
            message: The RTVI message to handle.
        """
        await self._message_queue.put(message)

    async def handle_function_call(self, params: FunctionCallParams):
        """Handle a function call from the LLM.

        Args:
            params: The function call parameters.
        """
        fn = RTVILLMFunctionCallMessageData(
            function_name=params.function_name,
            tool_call_id=params.tool_call_id,
            args=params.arguments,
        )
        message = RTVILLMFunctionCallMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def handle_function_call_start(
        self, function_name: str, llm: FrameProcessor, context: OpenAILLMContext
    ):
        """Handle the start of a function call from the LLM.

        .. deprecated:: 0.0.66
            This method is deprecated and will be removed in a future version.
            Use `RTVIProcessor.handle_function_call()` instead.

        Args:
            function_name: Name of the function being called.
            llm: The LLM processor making the call.
            context: The LLM context.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Function `RTVIProcessor.handle_function_call_start()` is deprecated, use `RTVIProcessor.handle_function_call()` instead.",
                DeprecationWarning,
            )

        fn = RTVILLMFunctionCallStartMessageData(function_name=function_name)
        message = RTVILLMFunctionCallStartMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames through the RTVI processor.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
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
        elif isinstance(frame, TransportMessageUrgentFrame):
            await self._handle_transport_message(frame)
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
        elif isinstance(frame, RTVIActionFrame):
            await self._action_queue.put(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: StartFrame):
        """Start the RTVI processor tasks."""
        if not self._action_task:
            self._action_queue = asyncio.Queue()
            self._action_task = self.create_task(self._action_task_handler())
        if not self._message_task:
            self._message_queue = asyncio.Queue()
            self._message_task = self.create_task(self._message_task_handler())
        await self._call_event_handler("on_bot_started")

    async def _stop(self, frame: EndFrame):
        """Stop the RTVI processor tasks."""
        await self._cancel_tasks()

    async def _cancel(self, frame: CancelFrame):
        """Cancel the RTVI processor tasks."""
        await self._cancel_tasks()

    async def _cancel_tasks(self):
        """Cancel all running tasks."""
        if self._action_task:
            await self.cancel_task(self._action_task)
            self._action_task = None

        if self._message_task:
            await self.cancel_task(self._message_task)
            self._message_task = None

    async def _push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        """Push a transport message frame."""
        frame = TransportMessageUrgentFrame(message=model.model_dump(exclude_none=exclude_none))
        await self.push_frame(frame)

    async def _action_task_handler(self):
        """Handle incoming action frames."""
        while True:
            frame = await self._action_queue.get()
            await self._handle_action(frame.message_id, frame.rtvi_action_run)
            self._action_queue.task_done()

    async def _message_task_handler(self):
        """Handle incoming transport messages."""
        while True:
            message = await self._message_queue.get()
            await self._handle_message(message)
            self._message_queue.task_done()

    async def _handle_transport_message(self, frame: TransportMessageUrgentFrame):
        """Handle an incoming transport message frame."""
        try:
            transport_message = frame.message
            if transport_message.get("label") != RTVI_MESSAGE_LABEL:
                logger.warning(f"Ignoring not RTVI message: {transport_message}")
                return
            message = RTVIMessage.model_validate(transport_message)
            await self._message_queue.put(message)
        except ValidationError as e:
            await self.send_error(f"Invalid RTVI transport message: {e}")
            logger.warning(f"Invalid RTVI transport message: {e}")

    async def _handle_message(self, message: RTVIMessage):
        """Handle a parsed RTVI message."""
        try:
            match message.type:
                case "client-ready":
                    data = None
                    try:
                        data = RTVIClientReadyData.model_validate(message.data)
                    except ValidationError:
                        # Not all clients have been updated to RTVI 1.0.0.
                        # For now, that's okay, we just log their info as unknown.
                        data = None
                        pass
                    await self._handle_client_ready(message.id, data)
                case "describe-actions":
                    await self._handle_describe_actions(message.id)
                case "describe-config":
                    await self._handle_describe_config(message.id)
                case "get-config":
                    await self._handle_get_config(message.id)
                case "update-config":
                    update_config = RTVIUpdateConfig.model_validate(message.data)
                    await self._handle_update_config(message.id, update_config)
                case "disconnect-bot":
                    await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                case "client-message":
                    data = RTVIRawClientMessageData.model_validate(message.data)
                    await self._handle_client_message(message.id, data)
                case "action":
                    action = RTVIActionRun.model_validate(message.data)
                    action_frame = RTVIActionFrame(message_id=message.id, rtvi_action_run=action)
                    await self._action_queue.put(action_frame)
                case "llm-function-call-result":
                    data = RTVILLMFunctionCallResultData.model_validate(message.data)
                    await self._handle_function_call_result(data)
                case "append-to-context":
                    data = RTVIAppendToContextData.model_validate(message.data)
                    await self._handle_update_context(data)
                case "raw-audio" | "raw-audio-batch":
                    await self._handle_audio_buffer(message.data)

                case _:
                    await self._send_error_response(message.id, f"Unsupported type {message.type}")

        except ValidationError as e:
            await self._send_error_response(message.id, f"Invalid message: {e}")
            logger.warning(f"Invalid message: {e}")
        except Exception as e:
            await self._send_error_response(message.id, f"Exception processing message: {e}")
            logger.warning(f"Exception processing message: {e}")

    async def _handle_client_ready(self, request_id: str, data: RTVIClientReadyData | None):
        """Handle the client-ready message from the client."""
        version = data.version if data else "unknown"
        logger.debug(f"Received client-ready: version {version}")
        if version == "unknown":
            self._client_version = [0, 3, 0]  # Default to 0.3.0 if unknown
        else:
            try:
                self._client_version = [int(v) for v in version.split(".")]
            except ValueError:
                logger.warning(f"Invalid client version format: {version}")
                self._client_version = [0, 3, 0]
        about = data.about if data else {"library": "unknown"}
        logger.debug(f"Client Details: {about}")
        if self._input_transport:
            await self._input_transport.start_audio_in_streaming()

        self._client_ready_id = request_id
        await self.set_client_ready()

    async def _handle_audio_buffer(self, data):
        """Handle incoming audio buffer data."""
        if not self._input_transport:
            return

        # Extract audio batch ensuring it's a list
        audio_list = data.get("base64AudioBatch") or [data.get("base64Audio")]

        try:
            for base64_audio in filter(None, audio_list):  # Filter out None values
                pcm_bytes = base64.b64decode(base64_audio)
                frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=data["sampleRate"],
                    num_channels=data["numChannels"],
                )
                await self._input_transport.push_audio_frame(frame)

        except (KeyError, TypeError, ValueError) as e:
            # Handle missing keys, decoding errors, and invalid types
            logger.error(f"Error processing audio buffer: {e}")

    async def _handle_describe_config(self, request_id: str):
        """Handle a describe-config request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        services = list(self._registered_services.values())
        message = RTVIDescribeConfig(id=request_id, data=RTVIDescribeConfigData(config=services))
        await self._push_transport_message(message)

    async def _handle_describe_actions(self, request_id: str):
        """Handle a describe-actions request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The Actions API is deprecated, use custom server and client messages instead.",
                DeprecationWarning,
            )

        actions = list(self._registered_actions.values())
        message = RTVIDescribeActions(id=request_id, data=RTVIDescribeActionsData(actions=actions))
        await self._push_transport_message(message)

    async def _handle_get_config(self, request_id: str):
        """Handle a get-config request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        message = RTVIConfigResponse(id=request_id, data=self._config)
        await self._push_transport_message(message)

    def _update_config_option(self, service: str, config: RTVIServiceOptionConfig):
        """Update a specific configuration option."""
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
        """Update configuration for a specific service."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        service = self._registered_services[config.service]
        for option in config.options:
            handler = service._options_dict[option.name].handler
            await handler(self, service.name, option)
            self._update_config_option(service.name, option)

    async def _update_config(self, data: RTVIConfig, interrupt: bool):
        """Update the RTVI configuration."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        if interrupt:
            await self.interrupt_bot()
        for service_config in data.config:
            await self._update_service_config(service_config)

    async def _handle_update_config(self, request_id: str, data: RTVIUpdateConfig):
        """Handle an update-config request."""
        await self._update_config(RTVIConfig(config=data.config), data.interrupt)
        await self._handle_get_config(request_id)

    async def _handle_update_context(self, data: RTVIAppendToContextData):
        if data.run_immediately:
            await self.interrupt_bot()
        frame = LLMMessagesAppendFrame(
            messages=[{"role": data.role, "content": data.content}],
            run_llm=data.run_immediately,
        )
        await self.push_frame(frame)

    async def _handle_client_message(self, msg_id: str, data: RTVIRawClientMessageData):
        """Handle a client message frame."""
        if not data:
            await self._send_error_response(msg_id, "Malformed client message")
            return

        # Create a RTVIClientMessageFrame to push the message
        frame = RTVIClientMessageFrame(msg_id=msg_id, type=data.t, data=data.d)
        await self.push_frame(frame)
        await self._call_event_handler(
            "on_client_message",
            RTVIClientMessage(
                msg_id=msg_id,
                type=data.t,
                data=data.d,
            ),
        )

    async def _handle_function_call_result(self, data):
        """Handle a function call result from the client."""
        frame = FunctionCallResultFrame(
            function_name=data.function_name,
            tool_call_id=data.tool_call_id,
            arguments=data.arguments,
            result=data.result,
        )
        await self.push_frame(frame)

    async def _handle_action(self, request_id: Optional[str], data: RTVIActionRun):
        """Handle an action execution request."""
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
        """Send the bot-ready message to the client."""
        config = None
        if self._client_version[0] < 1:
            config = self._config.config
        message = RTVIBotReady(
            id=self._client_ready_id,
            data=RTVIBotReadyData(version=RTVI_PROTOCOL_VERSION, config=config),
        )
        await self._push_transport_message(message)

    async def _send_server_message(self, message: RTVIServerMessage | RTVIServerResponse):
        """Send a message or response to the client."""
        await self._push_transport_message(message)

    async def _send_error_frame(self, frame: ErrorFrame):
        """Send an error frame as an RTVI error message."""
        if self._errors_enabled:
            message = RTVIError(data=RTVIErrorData(error=frame.error, fatal=frame.fatal))
            await self._push_transport_message(message)

    async def _send_error_response(self, id: str, error: str):
        """Send an error response message."""
        if self._errors_enabled:
            message = RTVIErrorResponse(id=id, data=RTVIErrorResponseData(error=error))
            await self._push_transport_message(message)

    def _action_id(self, service: str, action: str) -> str:
        """Generate an action ID from service and action names."""
        return f"{service}:{action}"
