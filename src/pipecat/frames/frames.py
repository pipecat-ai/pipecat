#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core frame definitions for the Pipecat AI framework.

This module contains all frame types used throughout the Pipecat pipeline system,
including data frames, system frames, and control frames for audio, video, text,
and LLM processing.
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.dtmf.types import KeypadEntry as NewKeypadEntry
from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.metrics.metrics import MetricsData
from pipecat.transcriptions.language import Language
from pipecat.utils.time import nanoseconds_to_str
from pipecat.utils.utils import obj_count, obj_id

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext, NotGiven
    from pipecat.processors.frame_processor import FrameProcessor


class DeprecatedKeypadEntry:
    """DTMF keypad entries for phone system integration.

    .. deprecated:: 0.0.82
        This class is deprecated and will be removed in a future version.
        Instead, use `audio.dtmf.types.KeypadEntry`.

    Parameters:
        ONE: Number key 1.
        TWO: Number key 2.
        THREE: Number key 3.
        FOUR: Number key 4.
        FIVE: Number key 5.
        SIX: Number key 6.
        SEVEN: Number key 7.
        EIGHT: Number key 8.
        NINE: Number key 9.
        ZERO: Number key 0.
        POUND: Pound/hash key (#).
        STAR: Star/asterisk key (*).
    """

    _enum = NewKeypadEntry

    @classmethod
    def _warn(cls):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`pipecat.frames.frames.KeypadEntry` is deprecated and will be removed in a future version. "
                "Use `pipecat.audio.dtmf.types.KeypadEntry` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the instance to be called as a function."""
        self._warn()
        return self._enum(*args, **kwargs)

    def __getattr__(self, name):
        """Retrieve an attribute from the underlying enum."""
        self._warn()
        return getattr(self._enum, name)

    def __getitem__(self, name):
        """Retrieve an item from the underlying enum."""
        self._warn()
        return self._enum[name]


KeypadEntry = DeprecatedKeypadEntry()


def format_pts(pts: Optional[int]):
    """Format presentation timestamp (PTS) in nanoseconds to a human-readable string.

    Converts a PTS value in nanoseconds to a string representation.

    Args:
        pts: Presentation timestamp in nanoseconds, or None if not set.
    """
    return nanoseconds_to_str(pts) if pts else None


@dataclass
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.

    All frames inherit from this base class and automatically receive
    unique identifiers, names, and metadata support.

    Parameters:
        id: Unique identifier for the frame instance.
        name: Human-readable name combining class name and instance count.
        pts: Presentation timestamp in nanoseconds.
        metadata: Dictionary for arbitrary frame metadata.
        transport_source: Name of the transport source that created this frame.
        transport_destination: Name of the transport destination for this frame.
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: Optional[int] = field(init=False)
    metadata: Dict[str, Any] = field(init=False)
    transport_source: Optional[str] = field(init=False)
    transport_destination: Optional[str] = field(init=False)

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"
        self.pts: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        self.transport_source: Optional[str] = None
        self.transport_destination: Optional[str] = None

    def __str__(self):
        return self.name


@dataclass
class SystemFrame(Frame):
    """System frame class for immediate processing.

    A frame that takes higher priority than other frames. System frames are
    handled in order and are not affected by user interruptions.
    """

    pass


@dataclass
class DataFrame(Frame):
    """Data frame class for processing data in order.

    A frame that is processed in order and usually contains data such as LLM
    context, text, audio or images. Data frames are cancelled by user
    interruptions.
    """

    pass


@dataclass
class ControlFrame(Frame):
    """Control frame class for processing control information in order.

    A frame that, similar to data frames, is processed in order and usually
    contains control information such as update settings or to end the pipeline
    after everything is flushed. Control frames are cancelled by user
    interruptions.

    """

    pass


#
# Mixins
#


@dataclass
class AudioRawFrame:
    """A frame containing a chunk of raw audio.

    Parameters:
        audio: Raw audio bytes in PCM format.
        sample_rate: Audio sample rate in Hz.
        num_channels: Number of audio channels.
        num_frames: Number of audio frames (calculated automatically).
    """

    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))


@dataclass
class ImageRawFrame:
    """A frame containing a raw image.

    Parameters:
        image: Raw image bytes.
        size: Image dimensions as (width, height) tuple.
        format: Image format (e.g., 'JPEG', 'PNG').
    """

    image: bytes
    size: Tuple[int, int]
    format: Optional[str]


#
# Data frames.
#


@dataclass
class OutputAudioRawFrame(DataFrame, AudioRawFrame):
    """Audio data frame for output to transport.

    A chunk of raw audio that will be played by the output transport. If the
    transport supports multiple audio destinations (e.g. multiple audio tracks)
    the destination name can be specified in transport_destination.
    """

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, destination: {self.transport_destination}, size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class OutputImageRawFrame(DataFrame, ImageRawFrame):
    """Image data frame for output to transport.

    An image that will be shown by the transport. If the transport supports
    multiple video destinations (e.g. multiple video tracks) the destination
    name can be specified in transport_destination.
    """

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, destination: {self.transport_destination}, size: {self.size}, format: {self.format})"


@dataclass
class TTSAudioRawFrame(OutputAudioRawFrame):
    """Audio data frame generated by Text-to-Speech services.

    A chunk of output audio generated by a TTS service, ready for playback.
    """

    pass


@dataclass
class SpeechOutputAudioRawFrame(OutputAudioRawFrame):
    """An audio frame part of a speech audio stream.

    This frame is part of a continuous stream of audio frames containing speech.
    The audio stream might also contain silence frames, so a process to distinguish
    between speech and silence might be needed.
    """

    pass


@dataclass
class URLImageRawFrame(OutputImageRawFrame):
    """Image frame with an associated URL.

    An output image with an associated URL. These images are usually
    generated by third-party services that provide a URL to download the image.

    Parameters:
        url: URL where the image can be downloaded from.
    """

    url: Optional[str] = None

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, url: {self.url}, size: {self.size}, format: {self.format})"


@dataclass
class SpriteFrame(DataFrame):
    """Animated sprite frame containing multiple images.

    An animated sprite that will be shown by the transport if the transport's
    camera is enabled. Will play at the framerate specified in the transport's
    `camera_out_framerate` constructor parameter.

    Parameters:
        images: List of image frames that make up the sprite animation.
    """

    images: List[OutputImageRawFrame]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {len(self.images)})"


@dataclass
class TextFrame(DataFrame):
    """Text data frame for passing text through the pipeline.

    A chunk of text. Emitted by LLM services, consumed by context
    aggregators, TTS services and more. Can be used to send text
    through processors.

    Parameters:
        text: The text content.
    """

    text: str
    skip_tts: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = False

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, text: [{self.text}])"


@dataclass
class LLMTextFrame(TextFrame):
    """Text frame generated by LLM services."""

    pass


@dataclass
class TTSTextFrame(TextFrame):
    """Text frame generated by Text-to-Speech services."""

    pass


@dataclass
class TranscriptionFrame(TextFrame):
    """Text frame containing speech transcription data.

    A text frame with transcription-specific data. The `result` field
    contains the result from the STT service if available.

    Parameters:
        user_id: Identifier for the user who spoke.
        timestamp: When the transcription occurred.
        language: Detected or specified language of the speech.
        result: Raw result from the STT service.
    """

    user_id: str
    timestamp: str
    language: Optional[Language] = None
    result: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}], language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class InterimTranscriptionFrame(TextFrame):
    """Text frame containing partial/interim transcription data.

    A text frame with interim transcription-specific data that represents
    partial results before final transcription. The `result` field
    contains the result from the STT service if available.

    Parameters:
        user_id: Identifier for the user who spoke.
        timestamp: When the interim transcription occurred.
        language: Detected or specified language of the speech.
        result: Raw result from the STT service.
    """

    text: str
    user_id: str
    timestamp: str
    language: Optional[Language] = None
    result: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}], language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class TranslationFrame(TextFrame):
    """Text frame containing translated transcription data.

    A text frame with translated transcription data that will be placed
    in the transport's receive queue when a participant speaks.

    Parameters:
        user_id: Identifier for the user who spoke.
        timestamp: When the translation occurred.
        language: Target language of the translation.
    """

    user_id: str
    timestamp: str
    language: Optional[Language] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}], language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class OpenAILLMContextAssistantTimestampFrame(DataFrame):
    """Timestamp information for assistant messages in LLM context.

    Parameters:
        timestamp: Timestamp when the assistant message was created.
    """

    timestamp: str


# A more universal (LLM-agnostic) name for
# OpenAILLMContextAssistantTimestampFrame, matching LLMContext
LLMContextAssistantTimestampFrame = OpenAILLMContextAssistantTimestampFrame


@dataclass
class TranscriptionMessage:
    """A message in a conversation transcript.

    A message in a conversation transcript containing the role and content.
    Messages are in standard format with roles normalized to user/assistant.

    Parameters:
        role: The role of the message sender (user or assistant).
        content: The message content/text.
        user_id: Optional identifier for the user.
        timestamp: Optional timestamp when the message was created.
    """

    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class TranscriptionUpdateFrame(DataFrame):
    """Frame containing new messages added to conversation transcript.

    A frame containing new messages added to the conversation transcript.
    This frame is emitted when new messages are added to the conversation history,
    containing only the newly added messages rather than the full transcript.
    Messages have normalized roles (user/assistant) regardless of the LLM service used.
    Messages are always in the OpenAI standard message format, which supports both:

    Examples:
        Simple format::

            [
                {
                    "role": "user",
                    "content": "Hi, how are you?"
                },
                {
                    "role": "assistant",
                    "content": "Great! And you?"
                }
            ]

        Content list format::

            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi, how are you?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Great! And you?"}]
                }
            ]

    OpenAI supports both formats. Anthropic and Google messages are converted to the
    content list format.

    Parameters:
        messages: List of new transcript messages that were added.
    """

    messages: List[TranscriptionMessage]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, messages: {len(self.messages)})"


@dataclass
class LLMContextFrame(Frame):
    """Frame containing a universal LLM context.

    Used as a signal to LLM services to ingest the provided context and
    generate a response based on it.

    Parameters:
        context: The LLM context containing messages, tools, and configuration.
    """

    context: "LLMContext"


@dataclass
class LLMMessagesFrame(DataFrame):
    """Frame containing LLM messages for chat completion.

    .. deprecated:: 0.0.79
        This class is deprecated and will be removed in a future version.
        Instead, use either:
        - `LLMMessagesUpdateFrame` with `run_llm=True`
        - `OpenAILLMContextFrame` with desired messages in a new context

    A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLMFullResponseStartFrame,
    TextFrames and an LLMFullResponseEndFrame. Note that the `messages`
    property in this class is mutable, and will be updated by various
    aggregators.

    Parameters:
        messages: List of message dictionaries in LLM format.
    """

    messages: List[dict]

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "LLMMessagesFrame is deprecated and will be removed in a future version. "
                "Instead, use either "
                "`LLMMessagesUpdateFrame` with `run_llm=True`, or "
                "`OpenAILLMContextFrame` with desired messages in a new context",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class LLMRunFrame(DataFrame):
    """Frame to trigger LLM processing with current context.

    A frame that instructs the LLM service to process the current context and
    generate a response.
    """

    pass


@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """Frame containing LLM messages to append to current context.

    A frame containing a list of LLM messages that need to be added to the
    current context.

    Parameters:
        messages: List of message dictionaries to append.
        run_llm: Whether the context update should be sent to the LLM.
    """

    messages: List[dict]
    run_llm: Optional[bool] = None


@dataclass
class LLMMessagesUpdateFrame(DataFrame):
    """Frame containing LLM messages to replace current context.

    A frame containing a list of new LLM messages to replace the current
    context LLM messages.

    Parameters:
        messages: List of message dictionaries to replace current context.
        run_llm: Whether the context update should be sent to the LLM.
    """

    messages: List[dict]
    run_llm: Optional[bool] = None


@dataclass
class LLMSetToolsFrame(DataFrame):
    """Frame containing tools for LLM function calling.

    A frame containing a list of tools for an LLM to use for function calling.
    The specific format depends on the LLM being used, but it should typically
    contain JSON Schema objects.

    Parameters:
        tools: List of tool/function definitions for the LLM.
    """

    tools: List[dict] | ToolsSchema | "NotGiven"


@dataclass
class LLMSetToolChoiceFrame(DataFrame):
    """Frame containing tool choice configuration for LLM function calling.

    Parameters:
        tool_choice: Tool choice setting - 'none', 'auto', 'required', or specific tool dict.
    """

    tool_choice: Literal["none", "auto", "required"] | dict


@dataclass
class LLMEnablePromptCachingFrame(DataFrame):
    """Frame to enable/disable prompt caching in LLMs.

    Parameters:
        enable: Whether to enable prompt caching.
    """

    enable: bool


@dataclass
class LLMConfigureOutputFrame(DataFrame):
    """Frame to configure LLM output.

    This frame is used to configure how the LLM produces output. For example, it
    can tell the LLM to generate tokens that should be added to the context but
    not spoken by the TTS service (if one is present in the pipeline).

    Parameters:
        skip_tts: Whether LLM tokens should skip the TTS service (if any).
    """

    skip_tts: bool


@dataclass
class TTSSpeakFrame(DataFrame):
    """Frame containing text that should be spoken by TTS.

    A frame that contains text that should be spoken by the TTS service
    in the pipeline (if any).

    Parameters:
        text: The text to be spoken.
    """

    text: str


@dataclass
class OutputTransportMessageFrame(DataFrame):
    """Frame containing transport-specific message data.

    Parameters:
        message: The transport message payload.
    """

    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"


@dataclass
class TransportMessageFrame(OutputTransportMessageFrame):
    """Frame containing transport-specific message data.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `OutputTransportMessageFrame`.

    Parameters:
        message: The transport message payload.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "TransportMessageFrame is deprecated and will be removed in a future version. "
                "Instead, use OutputTransportMessageFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class DTMFFrame:
    """Base class for DTMF (Dual-Tone Multi-Frequency) keypad frames.

    Parameters:
        button: The DTMF keypad entry that was pressed.
    """

    button: NewKeypadEntry


@dataclass
class OutputDTMFFrame(DTMFFrame, DataFrame):
    """DTMF keypress output frame for transport queuing.

    A DTMF keypress output that will be queued. If your transport supports
    multiple dial-out destinations, use the `transport_destination` field to
    specify where the DTMF keypress should be sent.
    """

    pass


#
# System frames
#


@dataclass
class StartFrame(SystemFrame):
    """Initial frame to start pipeline processing.

    This is the first frame that should be pushed down a pipeline to
    initialize all processors with their configuration parameters.

    Parameters:
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        allow_interruptions: Whether to allow user interruptions.
        enable_metrics: Whether to enable performance metrics collection.
        enable_tracing: Whether to enable OpenTelemetry tracing.
        enable_usage_metrics: Whether to enable usage metrics collection.
        interruption_strategies: List of interruption handling strategies.
        report_only_initial_ttfb: Whether to report only initial time-to-first-byte.
    """

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    allow_interruptions: bool = False
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    interruption_strategies: List[BaseInterruptionStrategy] = field(default_factory=list)
    report_only_initial_ttfb: bool = False


@dataclass
class CancelFrame(SystemFrame):
    """Frame indicating pipeline should stop immediately.

    Indicates that a pipeline needs to stop right away without
    processing remaining queued frames.
    """

    pass


@dataclass
class ErrorFrame(SystemFrame):
    """Frame notifying of errors in the pipeline.

    This is used to notify upstream that an error has occurred downstream in
    the pipeline. A fatal error indicates the error is unrecoverable and that the
    bot should exit.

    Parameters:
        error: Description of the error that occurred.
        fatal: Whether the error is fatal and requires bot shutdown.
        processor: The frame processor that generated the error.
    """

    error: str
    fatal: bool = False
    processor: Optional["FrameProcessor"] = None

    def __str__(self):
        return f"{self.name}(error: {self.error}, fatal: {self.fatal})"


@dataclass
class FatalErrorFrame(ErrorFrame):
    """Frame notifying of unrecoverable errors requiring bot shutdown.

    This is used to notify upstream that an unrecoverable error has occurred and
    that the bot should exit immediately.

    Parameters:
        fatal: Always True for fatal errors.
    """

    fatal: bool = field(default=True, init=False)


@dataclass
class FrameProcessorPauseUrgentFrame(SystemFrame):
    """Frame to pause frame processing immediately.

    This frame is used to pause frame processing for the given processor as
    fast as possible. Pausing frame processing will keep frames in the internal
    queue which will then be processed when frame processing is resumed with
    `FrameProcessorResumeFrame`.

    Parameters:
        processor: The frame processor to pause.
    """

    processor: "FrameProcessor"


@dataclass
class FrameProcessorResumeUrgentFrame(SystemFrame):
    """Frame to resume frame processing immediately.

    This frame is used to resume frame processing for the given processor
    if it was previously paused as fast as possible. After resuming frame
    processing all queued frames will be processed in the order received.

    Parameters:
        processor: The frame processor to resume.
    """

    processor: "FrameProcessor"


@dataclass
class InterruptionFrame(SystemFrame):
    """Frame indicating user started speaking (interruption detected).

    Emitted by the BaseInputTransport to indicate that a user has started
    speaking (i.e. is interrupting). This is similar to
    UserStartedSpeakingFrame except that it should be pushed concurrently
    with other frames (so the order is not guaranteed).
    """

    pass


@dataclass
class StartInterruptionFrame(InterruptionFrame):
    """Frame indicating user started speaking (interruption detected).

    .. deprecated:: 0.0.85
        This frame is deprecated and will be removed in a future version.
        Instead, use `InterruptionFrame`.

    Emitted by the BaseInputTransport to indicate that a user has started
    speaking (i.e. is interrupting). This is similar to
    UserStartedSpeakingFrame except that it should be pushed concurrently
    with other frames (so the order is not guaranteed).
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "StartInterruptionFrame is deprecated and will be removed in a future version. "
                "Instead, use InterruptionFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class UserStartedSpeakingFrame(SystemFrame):
    """Frame indicating user has started speaking.

    Emitted by VAD to indicate that a user has started speaking. This can be
    used for interruptions or other times when detecting that someone is
    speaking is more important than knowing what they're saying (as you will
    get with a TranscriptionFrame).

    Parameters:
        emulated: Whether this event was emulated rather than detected by VAD.
    """

    emulated: bool = False


@dataclass
class UserStoppedSpeakingFrame(SystemFrame):
    """Frame indicating user has stopped speaking.

    Emitted by the VAD to indicate that a user stopped speaking.

    Parameters:
        emulated: Whether this event was emulated rather than detected by VAD.
    """

    emulated: bool = False


@dataclass
class UserSpeakingFrame(SystemFrame):
    """Frame indicating the user is speaking.

    Emitted by VAD to indicate the user is speaking.
    """

    pass


@dataclass
class EmulateUserStartedSpeakingFrame(SystemFrame):
    """Frame to emulate user started speaking behavior.

    Emitted by internal processors upstream to emulate VAD behavior when a
    user starts speaking.
    """

    pass


@dataclass
class EmulateUserStoppedSpeakingFrame(SystemFrame):
    """Frame to emulate user stopped speaking behavior.

    Emitted by internal processors upstream to emulate VAD behavior when a
    user stops speaking.
    """

    pass


@dataclass
class VADUserStartedSpeakingFrame(SystemFrame):
    """Frame emitted when VAD definitively detects user started speaking."""

    pass


@dataclass
class VADUserStoppedSpeakingFrame(SystemFrame):
    """Frame emitted when VAD definitively detects user stopped speaking."""

    pass


@dataclass
class BotStartedSpeakingFrame(SystemFrame):
    """Frame indicating the bot started speaking.

    Emitted upstream and downstream by the BaseTransportOutput to indicate the
    bot started speaking.
    """

    pass


@dataclass
class BotStoppedSpeakingFrame(SystemFrame):
    """Frame indicating the bot stopped speaking.

    Emitted upstream and downstream by the BaseTransportOutput to indicate the
    bot stopped speaking.
    """

    pass


@dataclass
class BotSpeakingFrame(SystemFrame):
    """Frame indicating the bot is currently speaking.

    Emitted upstream and downstream by the BaseOutputTransport while the bot is
    still speaking. This can be used, for example, to detect when a user is
    idle. That is, while the bot is speaking we don't want to trigger any user
    idle timeout since the user might be listening.
    """

    pass


@dataclass
class MetricsFrame(SystemFrame):
    """Frame containing performance metrics data.

    Emitted by processors that can compute metrics like latencies.

    Parameters:
        data: List of metrics data collected by the processor.
    """

    data: List[MetricsData]


@dataclass
class FunctionCallFromLLM:
    """Represents a function call returned by the LLM.

    Represents a function call returned by the LLM to be registered for execution.

    Parameters:
        function_name: The name of the function to call.
        tool_call_id: A unique identifier for the function call.
        arguments: The arguments to pass to the function.
        context: The LLM context when the function call was made.
    """

    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    context: Any


@dataclass
class FunctionCallsStartedFrame(SystemFrame):
    """Frame signaling that function call execution is starting.

    A frame signaling that one or more function call execution is going to
    start.

    Parameters:
        function_calls: Sequence of function calls that will be executed.
    """

    function_calls: Sequence[FunctionCallFromLLM]


@dataclass
class FunctionCallInProgressFrame(SystemFrame):
    """Frame signaling that a function call is currently executing.

    Parameters:
        function_name: Name of the function being executed.
        tool_call_id: Unique identifier for this function call.
        arguments: Arguments passed to the function.
        cancel_on_interruption: Whether to cancel this call if interrupted.
    """

    function_name: str
    tool_call_id: str
    arguments: Any
    cancel_on_interruption: bool = False


@dataclass
class FunctionCallCancelFrame(SystemFrame):
    """Frame signaling that a function call has been cancelled.

    Parameters:
        function_name: Name of the function that was cancelled.
        tool_call_id: Unique identifier for the cancelled function call.
    """

    function_name: str
    tool_call_id: str


@dataclass
class FunctionCallResultProperties:
    """Properties for configuring function call result behavior.

    Parameters:
        run_llm: Whether to run the LLM after receiving this result.
        on_context_updated: Callback to execute when context is updated.
    """

    run_llm: Optional[bool] = None
    on_context_updated: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class FunctionCallResultFrame(SystemFrame):
    """Frame containing the result of an LLM function call.

    Parameters:
        function_name: Name of the function that was executed.
        tool_call_id: Unique identifier for the function call.
        arguments: Arguments that were passed to the function.
        result: The result returned by the function.
        run_llm: Whether to run the LLM after this result.
        properties: Additional properties for result handling.
    """

    function_name: str
    tool_call_id: str
    arguments: Any
    result: Any
    run_llm: Optional[bool] = None
    properties: Optional[FunctionCallResultProperties] = None


@dataclass
class STTMuteFrame(SystemFrame):
    """Frame to mute/unmute the Speech-to-Text service.

    Parameters:
        mute: Whether to mute (True) or unmute (False) the STT service.
    """

    mute: bool


@dataclass
class InputTransportMessageFrame(SystemFrame):
    """Frame for transport messages received from external sources.

    Parameters:
        message: The urgent transport message payload.
    """

    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"


@dataclass
class InputTransportMessageUrgentFrame(InputTransportMessageFrame):
    """Frame for transport messages received from external sources.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `InputTransportMessageFrame`.

    Parameters:
        message: The urgent transport message payload.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "InputTransportMessageUrgentFrame is deprecated and will be removed in a future version. "
                "Instead, use InputTransportMessageFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class OutputTransportMessageUrgentFrame(SystemFrame):
    """Frame for urgent transport messages that need to be sent immediately.

    Parameters:
        message: The urgent transport message payload.
    """

    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"


@dataclass
class TransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages that need to be sent immediately.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `OutputTransportMessageUrgentFrame`.

    Parameters:
        message: The urgent transport message payload.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "TransportMessageUrgentFrame is deprecated and will be removed in a future version. "
                "Instead, use OutputTransportMessageFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class UserImageRequestFrame(SystemFrame):
    """Frame requesting an image from a specific user.

    A frame to request an image from the given user. The frame might be
    generated by a function call in which case the corresponding fields will be
    properly set.

    Parameters:
        user_id: Identifier of the user to request image from.
        context: Optional context for the image request.
        function_name: Name of function that generated this request (if any).
        tool_call_id: Tool call ID if generated by function call.
        video_source: Specific video source to capture from.
    """

    user_id: str
    context: Optional[Any] = None
    function_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    video_source: Optional[str] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, video_source: {self.video_source}, function: {self.function_name}, request: {self.tool_call_id})"


@dataclass
class InputAudioRawFrame(SystemFrame, AudioRawFrame):
    """Raw audio input frame from transport.

    A chunk of audio usually coming from an input transport. If the transport
    supports multiple audio sources (e.g. multiple audio tracks) the source name
    will be specified in transport_source.
    """

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, source: {self.transport_source}, size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class InputImageRawFrame(SystemFrame, ImageRawFrame):
    """Raw image input frame from transport.

    An image usually coming from an input transport. If the transport
    supports multiple video sources (e.g. multiple video tracks) the source name
    will be specified in transport_source.
    """

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, source: {self.transport_source}, size: {self.size}, format: {self.format})"


@dataclass
class InputTextRawFrame(SystemFrame, TextFrame):
    """Raw text input frame from transport.

    Text input usually coming from user typing or programmatic text injection
    that should be sent to LLM services as input, similar to how InputAudioRawFrame
    and InputImageRawFrame represent user audio and video input.
    """

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, source: {self.transport_source}, text: [{self.text}])"


@dataclass
class UserAudioRawFrame(InputAudioRawFrame):
    """Raw audio input frame associated with a specific user.

    A chunk of audio, usually coming from an input transport, associated to a user.

    Parameters:
        user_id: Identifier of the user who provided this audio.
    """

    user_id: str = ""

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, user: {self.user_id}, source: {self.transport_source}, size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class UserImageRawFrame(InputImageRawFrame):
    """Raw image input frame associated with a specific user.

    An image associated to a user, potentially in response to an image request.

    Parameters:
        user_id: Identifier of the user who provided this image.
        request: The original image request frame if this is a response.
    """

    user_id: str = ""
    request: Optional[UserImageRequestFrame] = None

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, user: {self.user_id}, source: {self.transport_source}, size: {self.size}, format: {self.format}, request: {self.request})"


@dataclass
class InputDTMFFrame(DTMFFrame, SystemFrame):
    """DTMF keypress input frame from transport."""

    pass


@dataclass
class OutputDTMFUrgentFrame(DTMFFrame, SystemFrame):
    """DTMF keypress output frame for immediate sending.

    A DTMF keypress output that will be sent right away. If your transport
    supports multiple dial-out destinations, use the `transport_destination`
    field to specify where the DTMF keypress should be sent.
    """

    pass


@dataclass
class SpeechControlParamsFrame(SystemFrame):
    """Frame for notifying processors of speech control parameter changes.

    This includes parameters for both VAD (Voice Activity Detection) and
    turn-taking analysis. It allows downstream processors to adjust their
    behavior based on updated interaction control settings.

    Parameters:
        vad_params: Current VAD parameters.
        turn_params: Current turn-taking analysis parameters.
    """

    vad_params: Optional[VADParams] = None
    turn_params: Optional[SmartTurnParams] = None


#
# Task frames
#


@dataclass
class TaskFrame(SystemFrame):
    """Base frame for task frames.

    This is a base class for frames that are meant to be sent and handled
    upstream by the pipeline task. This might result in a corresponding frame
    sent downstream (e.g. `InterruptionTaskFrame` / `InterruptionFrame` or
    `EndTaskFrame` / `EndFrame`).

    """

    pass


@dataclass
class EndTaskFrame(TaskFrame):
    """Frame to request graceful pipeline task closure.

    This is used to notify the pipeline task that the pipeline should be
    closed nicely (flushing all the queued frames) by pushing an EndFrame
    downstream. This frame should be pushed upstream.
    """

    pass


@dataclass
class CancelTaskFrame(TaskFrame):
    """Frame to request immediate pipeline task cancellation.

    This is used to notify the pipeline task that the pipeline should be
    stopped immediately by pushing a CancelFrame downstream. This frame
    should be pushed upstream.
    """

    pass


@dataclass
class StopTaskFrame(TaskFrame):
    """Frame to request pipeline task stop while keeping processors running.

    This is used to notify the pipeline task that it should be stopped as
    soon as possible (flushing all the queued frames) but that the pipeline
    processors should be kept in a running state. This frame should be pushed
    upstream.
    """

    pass


@dataclass
class InterruptionTaskFrame(TaskFrame):
    """Frame indicating the bot should be interrupted.

    Emitted when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.
    This frame should be pushed upstream.
    """

    pass


@dataclass
class BotInterruptionFrame(InterruptionTaskFrame):
    """Frame indicating the bot should be interrupted.

    .. deprecated:: 0.0.85
        This frame is deprecated and will be removed in a future version.
        Instead, use `InterruptionTaskFrame`.

    Emitted when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.
    This frame should be pushed upstream.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "BotInterruptionFrame is deprecated and will be removed in a future version. "
                "Instead, use InterruptionTaskFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


#
# Control frames
#


@dataclass
class EndFrame(ControlFrame):
    """Frame indicating pipeline has ended and should shut down.

    Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will be received in the order it
    was sent.
    """

    pass


@dataclass
class StopFrame(ControlFrame):
    """Frame indicating pipeline should stop but keep processors running.

    Indicates that a pipeline should be stopped but that the pipeline
    processors should be kept in a running state. This is normally queued from
    the pipeline task.
    """

    pass


@dataclass
class OutputTransportReadyFrame(ControlFrame):
    """Frame indicating that the output transport is ready.

    Indicates that the output transport is ready and able to receive frames.
    """

    pass


@dataclass
class HeartbeatFrame(ControlFrame):
    """Frame used by pipeline task to monitor pipeline health.

    This frame is used by the pipeline task as a mechanism to know if the
    pipeline is running properly.

    Parameters:
        timestamp: Timestamp when the heartbeat was generated.
    """

    timestamp: int


@dataclass
class FrameProcessorPauseFrame(ControlFrame):
    """Frame to pause frame processing for a specific processor.

    This frame is used to pause frame processing for the given
    processor. Pausing frame processing will keep frames in the internal queue
    which will then be processed when frame processing is resumed with
    `FrameProcessorResumeFrame`.

    Parameters:
        processor: The frame processor to pause.
    """

    processor: "FrameProcessor"


@dataclass
class FrameProcessorResumeFrame(ControlFrame):
    """Frame to resume frame processing for a specific processor.

    This frame is used to resume frame processing for the given processor if
    it was previously paused. After resuming frame processing all queued frames
    will be processed in the order received.

    Parameters:
        processor: The frame processor to resume.
    """

    processor: "FrameProcessor"


@dataclass
class LLMFullResponseStartFrame(ControlFrame):
    """Frame indicating the beginning of an LLM response.

    Used to indicate the beginning of an LLM response. Followed by one or
    more TextFrames and a final LLMFullResponseEndFrame.
    """

    skip_tts: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = False


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    """Frame indicating the end of an LLM response."""

    skip_tts: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = False


@dataclass
class TTSStartedFrame(ControlFrame):
    """Frame indicating the beginning of a TTS response.

    Used to indicate the beginning of a TTS response. Following
    TTSAudioRawFrames are part of the TTS response until a
    TTSStoppedFrame. These frames can be used for aggregating audio frames in a
    transport to optimize the size of frames sent to the session, without
    needing to control this in the TTS service.
    """

    pass


@dataclass
class TTSStoppedFrame(ControlFrame):
    """Frame indicating the end of a TTS response."""

    pass


@dataclass
class ServiceUpdateSettingsFrame(ControlFrame):
    """Base frame for updating service settings.

    A control frame containing a request to update service settings.

    Parameters:
        settings: Dictionary of setting name to value mappings.
    """

    settings: Mapping[str, Any]


@dataclass
class LLMUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    """Frame for updating LLM service settings."""

    pass


@dataclass
class TTSUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    """Frame for updating TTS service settings."""

    pass


@dataclass
class STTUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    """Frame for updating STT service settings."""

    pass


@dataclass
class VADParamsUpdateFrame(ControlFrame):
    """Frame for updating VAD parameters.

    A control frame containing a request to update VAD params. Intended
    to be pushed upstream from RTVI processor.

    Parameters:
        params: New VAD parameters to apply.
    """

    params: VADParams


@dataclass
class FilterControlFrame(ControlFrame):
    """Base control frame for audio filter operations."""

    pass


@dataclass
class FilterUpdateSettingsFrame(FilterControlFrame):
    """Frame for updating audio filter settings.

    Parameters:
        settings: Dictionary of filter setting name to value mappings.
    """

    settings: Mapping[str, Any]


@dataclass
class FilterEnableFrame(FilterControlFrame):
    """Frame for enabling/disabling audio filters at runtime.

    Parameters:
        enable: Whether to enable (True) or disable (False) the filter.
    """

    enable: bool


@dataclass
class MixerControlFrame(ControlFrame):
    """Base control frame for audio mixer operations."""

    pass


@dataclass
class MixerUpdateSettingsFrame(MixerControlFrame):
    """Frame for updating audio mixer settings.

    Parameters:
        settings: Dictionary of mixer setting name to value mappings.
    """

    settings: Mapping[str, Any]


@dataclass
class MixerEnableFrame(MixerControlFrame):
    """Frame for enabling/disabling audio mixer at runtime.

    Parameters:
        enable: Whether to enable (True) or disable (False) the mixer.
    """

    enable: bool


@dataclass
class ServiceSwitcherFrame(ControlFrame):
    """A base class for frames that affect ServiceSwitcher behavior."""

    pass


@dataclass
class ManuallySwitchServiceFrame(ServiceSwitcherFrame):
    """A frame to request a manual switch in the active service in a ServiceSwitcher.

    Handled by ServiceSwitcherStrategyManual to switch the active service.
    """

    service: "FrameProcessor"
