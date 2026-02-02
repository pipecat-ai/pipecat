#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI (Real-Time Voice Interface) protocol implementation for Pipecat.

This file includes all the RTVI protocol message definitions and data structures.
"""

from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
)

from pydantic import BaseModel

from pipecat.frames.frames import (
    AggregationType,
    FileSourceType,
)

# -- Constants --
RTVI_PROTOCOL_VERSION = "1.2.0"

RTVI_MESSAGE_LABEL = "rtvi-ai"
RTVIMessageLiteral = Literal["rtvi-ai"]

# -- Base Message Structure --


class RTVIMessage(BaseModel):
    """Base RTVI message structure.

    Represents the standard format for RTVI protocol messages.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None


# -- Client -> Pipecat messages.


class RTVIRawClientMessageData(BaseModel):
    """Data structure expected from client messages sent to the RTVI server."""

    t: str
    d: Optional[Any] = None


class RTVIClientMessage(BaseModel):
    """Cleansed data structure for client messages for handling."""

    msg_id: str
    type: str
    data: Optional[Any] = None


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


class RTVIAboutClientData(BaseModel):
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
    about: RTVIAboutClientData


# -- Pipecat -> Client errors


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


# -- Pipecat -> Client responses and messages.


class RTVIBotReadyData(BaseModel):
    """Data for bot ready notification.

    Contains protocol version and initial configuration.
    """

    version: str
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


class RTVISendTextOptions(BaseModel):
    """Options for sending text input to the LLM.

    Contains options for how the pipeline should process the text input.
    """

    run_immediately: bool = True
    audio_response: bool = True


class RTVISendTextData(BaseModel):
    """Data format for sending text input to the LLM.

    Contains the text content to send and any options for how the pipeline should process it.

    """

    content: str
    options: Optional[RTVISendTextOptions] = None


class RTVIFileSource(BaseModel):
    """Base class for RTVI file sources."""

    type: FileSourceType


class RTVIFileBytes(RTVIFileSource):
    """File source as base64-encoded bytes."""

    type: FileSourceType = "bytes"
    bytes: str  # base64-encoded string
    width: Optional[int] = None
    height: Optional[int] = None


class RTVIFileUrl(RTVIFileSource):
    """File source as a URL."""

    type: FileSourceType = "url"
    url: str


class RTVIFile(BaseModel):
    """File data structure for RTVI file sending."""

    format: str  # Mime format of the file, e.g., 'application/pdf'
    name: Optional[str] = None
    source: RTVIFileBytes | RTVIFileUrl
    customOpts: Optional[dict] = None  # ex. 'detail' in openAI or 'citations' in Bedrock


class RTVISendFileData(BaseModel):
    """Data format for sending a file to the LLM.

    Contains the information of the file to send and any options for how the pipeline should process it.
    """

    content: str  # Text to accompany the file
    file: RTVIFile
    options: Optional[RTVISendTextOptions] = None


class RTVIAppendToContextData(BaseModel):
    """Data format for appending messages to the context.

    Contains the role, content, and whether to run the message immediately.

    .. deprecated:: 0.0.85
        The RTVI message, append-to-context, has been deprecated. Use send-text
        or custom client and server messages instead.
    """

    role: Literal["user", "assistant"] | str
    content: Any
    run_immediately: bool = False


class RTVIAppendToContext(BaseModel):
    """RTVI Message format to append content to the LLM context.

    .. deprecated:: 0.0.85
        The RTVI message, append-to-context, has been deprecated. Use send-text
        or custom client and server messages instead.
    """

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


class RTVIBotOutputMessageData(RTVITextMessageData):
    """Data for bot output RTVI messages.

    Extends RTVITextMessageData to include metadata about the output.
    """

    spoken: bool = False  # Indicates if the text has been spoken by TTS
    aggregated_by: AggregationType | str
    # Indicates what form the text is in (e.g., by word, sentence, etc.)


class RTVIBotOutputMessage(BaseModel):
    """Message containing bot output text.

    An event meant to holistically represent what the bot is outputting,
    along with metadata about the output and if it has been spoken.
    """

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-output"] = "bot-output"
    data: RTVIBotOutputMessageData


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


class RTVIAudioLevelMessageData(BaseModel):
    """Data format for sending audio levels."""

    value: float


class RTVIUserAudioLevelMessage(BaseModel):
    """Message indicating user audio level."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["user-audio-level"] = "user-audio-level"
    data: RTVIAudioLevelMessageData


class RTVIBotAudioLevelMessage(BaseModel):
    """Message indicating bot audio level."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["bot-audio-level"] = "bot-audio-level"
    data: RTVIAudioLevelMessageData


class RTVISystemLogMessage(BaseModel):
    """Message including a system log."""

    label: RTVIMessageLiteral = RTVI_MESSAGE_LABEL
    type: Literal["system-log"] = "system-log"
    data: RTVITextMessageData


class RTVI:
    """A namespace for all RTVI protocol v1 message definitions."""

    def __init__(self):
        """Disallow instantiation of this class."""
        raise RuntimeError("The RTVI class is a namespace and should not be instantiated.")


_all_definitions = [
    # Constants
    ("PROTOCOL_VERSION", RTVI_PROTOCOL_VERSION),
    ("MESSAGE_LABEL", RTVI_MESSAGE_LABEL),
    ("MessageLiteral", RTVIMessageLiteral),
    # Classes (the name mapping is automatic if you strip the leading '_')
    ("Message", RTVIMessage),
    ("RawClientMessageData", RTVIRawClientMessageData),
    ("ClientMessage", RTVIClientMessage),
    ("RawServerResponseData", RTVIRawServerResponseData),
    ("ServerResponse", RTVIServerResponse),
    ("AboutClientData", RTVIAboutClientData),
    ("ClientReadyData", RTVIClientReadyData),
    ("ErrorResponseData", RTVIErrorResponseData),
    ("ErrorResponse", RTVIErrorResponse),
    ("ErrorData", RTVIErrorData),
    ("Error", RTVIError),
    ("BotReadyData", RTVIBotReadyData),
    ("BotReady", RTVIBotReady),
    ("LLMFunctionCallMessageData", RTVILLMFunctionCallMessageData),
    ("LLMFunctionCallMessage", RTVILLMFunctionCallMessage),
    ("SendTextOptions", RTVISendTextOptions),
    ("SendTextData", RTVISendTextData),
    ("FileBytes", RTVIFileBytes),
    ("FileUrl", RTVIFileUrl),
    ("File", RTVIFile),
    ("SendFileData", RTVISendFileData),
    ("AppendToContextData", RTVIAppendToContextData),
    ("AppendToContext", RTVIAppendToContext),
    ("LLMFunctionCallStartMessageData", RTVILLMFunctionCallStartMessageData),
    ("LLMFunctionCallStartMessage", RTVILLMFunctionCallStartMessage),
    ("LLMFunctionCallResultData", RTVILLMFunctionCallResultData),
    ("BotLLMStartedMessage", RTVIBotLLMStartedMessage),
    ("BotLLMStoppedMessage", RTVIBotLLMStoppedMessage),
    ("BotTTSStartedMessage", RTVIBotTTSStartedMessage),
    ("BotTTSStoppedMessage", RTVIBotTTSStoppedMessage),
    ("TextMessageData", RTVITextMessageData),
    ("BotOutputMessageData", RTVIBotOutputMessageData),
    ("BotOutputMessage", RTVIBotOutputMessage),
    ("BotTranscriptionMessage", RTVIBotTranscriptionMessage),
    ("BotLLMTextMessage", RTVIBotLLMTextMessage),
    ("BotTTSTextMessage", RTVIBotTTSTextMessage),
    ("AudioMessageData", RTVIAudioMessageData),
    ("BotTTSAudioMessage", RTVIBotTTSAudioMessage),
    ("UserTranscriptionMessageData", RTVIUserTranscriptionMessageData),
    ("UserTranscriptionMessage", RTVIUserTranscriptionMessage),
    ("UserLLMTextMessage", RTVIUserLLMTextMessage),
    ("UserStartedSpeakingMessage", RTVIUserStartedSpeakingMessage),
    ("UserStoppedSpeakingMessage", RTVIUserStoppedSpeakingMessage),
    ("BotStartedSpeakingMessage", RTVIBotStartedSpeakingMessage),
    ("BotStoppedSpeakingMessage", RTVIBotStoppedSpeakingMessage),
    ("MetricsMessage", RTVIMetricsMessage),
    ("ServerMessage", RTVIServerMessage),
    ("AudioLevelMessageData", RTVIAudioLevelMessageData),
    ("UserAudioLevelMessage", RTVIUserAudioLevelMessage),
    ("BotAudioLevelMessage", RTVIBotAudioLevelMessage),
    ("SystemLogMessage", RTVISystemLogMessage),
]

for item in _all_definitions:
    if isinstance(item, tuple):
        # It's a constant (name, value)
        name, value = item
        setattr(RTVI, name, value)
    else:
        # It's a class
        # Get the class name and remove the leading underscore
        name = item.__name__.lstrip("_")
        setattr(RTVI, name, item)

# Clean up the module's global namespace to avoid polluting imports
del _all_definitions
