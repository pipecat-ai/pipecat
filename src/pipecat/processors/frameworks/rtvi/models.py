#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI protocol v1 message models.

Contains all RTVI protocol v1 message definitions and data structures.
Import this module under the ``RTVI`` alias to use as a namespace::

    import pipecat.processors.frameworks.rtvi.models as RTVI

    msg = RTVI.BotReady(id="1", data=RTVI.BotReadyData(version=RTVI.PROTOCOL_VERSION))
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
    FileFormat,
    FileSourceType,
)

# -- Constants --
RTVI_PROTOCOL_VERSION = "1.3.0"

MESSAGE_LABEL = "rtvi-ai"
MessageLiteral = Literal["rtvi-ai"]

# -- Base Message Structure --


class Message(BaseModel):
    """Base RTVI message structure.

    Represents the standard format for RTVI protocol messages.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None


# -- Client -> Pipecat messages.


class RawClientMessageData(BaseModel):
    """Data structure expected from client messages sent to the RTVI server."""

    t: str
    d: Optional[Any] = None


class ClientMessage(BaseModel):
    """Cleansed data structure for client messages for handling."""

    msg_id: str
    type: str
    data: Optional[Any] = None


class RawServerResponseData(BaseModel):
    """Data structure for server responses to client messages."""

    t: str
    d: Optional[Any] = None


class ServerResponse(BaseModel):
    """The RTVI-formatted message response from the server to the client.

    This message is used to respond to custom messages sent by the client.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["server-response"] = "server-response"
    id: str
    data: RawServerResponseData


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


class ClientReadyData(BaseModel):
    """Data format of client ready messages.

    Contains the RTVI protocol version and client information.
    """

    version: str
    about: AboutClientData


# -- Pipecat -> Client errors


class ErrorResponseData(BaseModel):
    """Data for an RTVI error response.

    Contains the error message to send back to the client.
    """

    error: str


class ErrorResponse(BaseModel):
    """RTVI error response message.

    RTVI formatted error response message for relaying failed client requests.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["error-response"] = "error-response"
    id: str
    data: ErrorResponseData


class ErrorData(BaseModel):
    """Data for an RTVI error event.

    Contains error information including whether it's fatal.
    """

    error: str
    fatal: bool  # Indicates the pipeline has stopped due to this error


class Error(BaseModel):
    """RTVI error event message.

    RTVI formatted error message for relaying errors in the pipeline.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["error"] = "error"
    data: ErrorData


# -- Pipecat -> Client responses and messages.


class BotReadyData(BaseModel):
    """Data for bot ready notification.

    Contains protocol version and initial configuration.
    """

    version: str
    about: Optional[Mapping[str, Any]] = None


class BotReady(BaseModel):
    """Message indicating bot is ready for interaction.

    Sent after bot initialization is complete.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-ready"] = "bot-ready"
    id: str
    data: BotReadyData


class LLMFunctionCallMessageData(BaseModel):
    """Data for LLM function call notification.

    Contains function call details including name, ID, and arguments.

    .. deprecated:: 0.0.102
        Use ``LLMFunctionCallInProgressMessageData`` instead.
    """

    function_name: str
    tool_call_id: str
    args: Mapping[str, Any]


class LLMFunctionCallMessage(BaseModel):
    """Message notifying of an LLM function call.

    Sent when the LLM makes a function call.

    .. deprecated:: 0.0.102
        Use ``LLMFunctionCallInProgressMessage`` with the
        ``llm-function-call-in-progress`` event type instead.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call"] = "llm-function-call"
    data: LLMFunctionCallMessageData


class SendTextOptions(BaseModel):
    """Options for sending text input to the LLM.

    Contains options for how the pipeline should process the text input.
    """

    run_immediately: bool = True
    audio_response: bool = True


class SendTextData(BaseModel):
    """Data format for sending text input to the LLM.

    Contains the text content to send and any options for how the pipeline should process it.
    """

    content: str
    options: Optional[SendTextOptions] = None


class FileSource(BaseModel):
    """Base class for RTVI file sources."""

    type: FileSourceType


class FileBytes(FileSource):
    """File source as base64-encoded bytes."""

    type: FileSourceType = "bytes"
    bytes: str  # base64-encoded string
    width: Optional[int] = None
    height: Optional[int] = None


class FileUrl(FileSource):
    """File source as a URL."""

    type: FileSourceType = "url"
    url: str


class File(BaseModel):
    """File data structure for RTVI file sending."""

    format: FileFormat
    name: Optional[str] = None
    source: FileBytes | FileUrl
    customOpts: Optional[dict] = None  # ex. 'detail' in openAI or 'citations' in Bedrock


class SendFileData(BaseModel):
    """Data format for sending a file to the LLM.

    Contains the information of the file to send and any options for how the pipeline should process it.
    """

    content: str  # Text to accompany the file
    file: File
    options: Optional[SendTextOptions] = None


class AppendToContextData(BaseModel):
    """Data format for appending messages to the context.

    Contains the role, content, and whether to run the message immediately.

    .. deprecated:: 0.0.85
        The RTVI message, append-to-context, has been deprecated. Use send-text
        or custom client and server messages instead.
    """

    role: Literal["user", "assistant"] | str
    content: Any
    run_immediately: bool = False


class AppendToContext(BaseModel):
    """RTVI message format to append content to the LLM context.

    .. deprecated:: 0.0.85
        The RTVI message, append-to-context, has been deprecated. Use send-text
        or custom client and server messages instead.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["append-to-context"] = "append-to-context"
    data: AppendToContextData


class LLMFunctionCallStartMessageData(BaseModel):
    """Data for LLM function call start notification.

    Contains the function name being called. Fields may be omitted based on
    the configured function_call_report_level for security.
    """

    function_name: Optional[str] = None


class LLMFunctionCallStartMessage(BaseModel):
    """Message notifying that an LLM function call has started.

    Sent when the LLM begins a function call.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-started"] = "llm-function-call-started"
    data: LLMFunctionCallStartMessageData


class LLMFunctionCallResultData(BaseModel):
    """Data for LLM function call result.

    Contains function call details and result.
    """

    function_name: str
    tool_call_id: str
    arguments: dict
    result: dict | str


class LLMFunctionCallInProgressMessageData(BaseModel):
    """Data for LLM function call in-progress notification.

    Contains function call details including name, ID, and arguments.
    Fields may be omitted based on the configured function_call_report_level for security.
    """

    tool_call_id: str
    function_name: Optional[str] = None
    arguments: Optional[Mapping[str, Any]] = None


class LLMFunctionCallInProgressMessage(BaseModel):
    """Message notifying that an LLM function call is in progress.

    Sent when the LLM function call execution begins.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-in-progress"] = "llm-function-call-in-progress"
    data: LLMFunctionCallInProgressMessageData


class LLMFunctionCallStoppedMessageData(BaseModel):
    """Data for LLM function call stopped notification.

    Contains details about the function call that stopped, including
    whether it was cancelled or completed with a result.
    Fields may be omitted based on the configured function_call_report_level for security.
    """

    tool_call_id: str
    cancelled: bool
    function_name: Optional[str] = None
    result: Optional[Any] = None


class LLMFunctionCallStoppedMessage(BaseModel):
    """Message notifying that an LLM function call has stopped.

    Sent when a function call completes (with result) or is cancelled.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-stopped"] = "llm-function-call-stopped"
    data: LLMFunctionCallStoppedMessageData


class BotLLMStartedMessage(BaseModel):
    """Message indicating bot LLM processing has started."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-started"] = "bot-llm-started"


class BotLLMStoppedMessage(BaseModel):
    """Message indicating bot LLM processing has stopped."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-stopped"] = "bot-llm-stopped"


class BotTTSStartedMessage(BaseModel):
    """Message indicating bot TTS processing has started."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-started"] = "bot-tts-started"


class BotTTSStoppedMessage(BaseModel):
    """Message indicating bot TTS processing has stopped."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-stopped"] = "bot-tts-stopped"


class TextMessageData(BaseModel):
    """Data for text-based RTVI messages.

    Contains text content.
    """

    text: str


class BotOutputMessageData(TextMessageData):
    """Data for bot output RTVI messages.

    Extends TextMessageData to include metadata about the output.
    """

    spoken: bool = False  # Indicates if the text has been spoken by TTS
    aggregated_by: AggregationType | str
    # Indicates what form the text is in (e.g., by word, sentence, etc.)


class BotOutputMessage(BaseModel):
    """Message containing bot output text.

    An event meant to holistically represent what the bot is outputting,
    along with metadata about the output and if it has been spoken.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-output"] = "bot-output"
    data: BotOutputMessageData


class BotTranscriptionMessage(BaseModel):
    """Message containing bot transcription text.

    Sent when the bot's speech is transcribed.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-transcription"] = "bot-transcription"
    data: TextMessageData


class BotLLMTextMessage(BaseModel):
    """Message containing bot LLM text output.

    Sent when the bot's LLM generates text.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-text"] = "bot-llm-text"
    data: TextMessageData


class BotTTSTextMessage(BaseModel):
    """Message containing bot TTS text output.

    Sent when text is being processed by TTS.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-text"] = "bot-tts-text"
    data: TextMessageData


class AudioMessageData(BaseModel):
    """Data for audio-based RTVI messages.

    Contains audio data and metadata.
    """

    audio: str
    sample_rate: int
    num_channels: int


class BotTTSAudioMessage(BaseModel):
    """Message containing bot TTS audio output.

    Sent when the bot's TTS generates audio.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-audio"] = "bot-tts-audio"
    data: AudioMessageData


class UserTranscriptionMessageData(BaseModel):
    """Data for user transcription messages.

    Contains transcription text and metadata.
    """

    text: str
    user_id: str
    timestamp: str
    final: bool


class UserTranscriptionMessage(BaseModel):
    """Message containing user transcription.

    Sent when user speech is transcribed.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-transcription"] = "user-transcription"
    data: UserTranscriptionMessageData


class UserLLMTextMessage(BaseModel):
    """Message containing user text input for LLM.

    Sent when user text is processed by the LLM.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-llm-text"] = "user-llm-text"
    data: TextMessageData


class UserStartedSpeakingMessage(BaseModel):
    """Message indicating user has started speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-started-speaking"] = "user-started-speaking"


class UserStoppedSpeakingMessage(BaseModel):
    """Message indicating user has stopped speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class UserMuteStartedMessage(BaseModel):
    """Message indicating user has been muted."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-mute-started"] = "user-mute-started"


class UserMuteStoppedMessage(BaseModel):
    """Message indicating user has been unmuted."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-mute-stopped"] = "user-mute-stopped"


class BotStartedSpeakingMessage(BaseModel):
    """Message indicating bot has started speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class BotStoppedSpeakingMessage(BaseModel):
    """Message indicating bot has stopped speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class MetricsMessage(BaseModel):
    """Message containing performance metrics.

    Sent to provide performance and usage metrics.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["metrics"] = "metrics"
    data: Mapping[str, Any]


class ServerMessage(BaseModel):
    """Generic server message.

    Used for custom server-to-client messages.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["server-message"] = "server-message"
    data: Any


class AudioLevelMessageData(BaseModel):
    """Data format for sending audio levels."""

    value: float


class UserAudioLevelMessage(BaseModel):
    """Message indicating user audio level."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-audio-level"] = "user-audio-level"
    data: AudioLevelMessageData


class BotAudioLevelMessage(BaseModel):
    """Message indicating bot audio level."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-audio-level"] = "bot-audio-level"
    data: AudioLevelMessageData


class SystemLogMessage(BaseModel):
    """Message including a system log."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["system-log"] = "system-log"
    data: TextMessageData
