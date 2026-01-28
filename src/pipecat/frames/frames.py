#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core frame definitions for the Pipecat AI framework.

This module contains all frame types used throughout the Pipecat pipeline system,
including data frames, system frames, and control frames for audio, video, text,
and LLM processing.
"""

from dataclasses import dataclass, field
from enum import Enum
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
from pipecat.audio.turn.base_turn_analyzer import BaseTurnParams
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
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass


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
        format: Image format (e.g., 'RGB', 'RGBA').
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
    skip_tts: Optional[bool] = field(init=False)
    # Whether any necessary inter-frame (leading/trailing) spaces are already
    # included in the text.
    # NOTE: Ideally this would be available at init time with a default value,
    # but that would impact how subclasses can be initialized (it would require
    # mandatory fields of theirs to have defaults to preserve
    # non-default-before-default argument order)
    includes_inter_frame_spaces: bool = field(init=False)
    # Whether this text frame should be appended to the LLM context.
    append_to_context: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = None
        self.includes_inter_frame_spaces = False
        self.append_to_context = True

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, text: [{self.text}])"


@dataclass
class LLMTextFrame(TextFrame):
    """Text frame generated by LLM services."""

    def __post_init__(self):
        super().__post_init__()
        # LLM services send text frames with all necessary spaces included
        self.includes_inter_frame_spaces = True


class AggregationType(str, Enum):
    """Built-in aggregation strings."""

    SENTENCE = "sentence"
    WORD = "word"

    def __str__(self):
        return self.value


@dataclass
class AggregatedTextFrame(TextFrame):
    """Text frame representing an aggregation of TextFrames.

    This frame contains multiple TextFrames aggregated together for processing
    or output along with a field to indicate how they are aggregated.

    Parameters:
        aggregated_by: Method used to aggregate the text frames.
    """

    aggregated_by: AggregationType | str


@dataclass
class VisionTextFrame(LLMTextFrame):
    """Text frame generated by vision services."""

    pass


@dataclass
class TTSTextFrame(AggregatedTextFrame):
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
        finalized: Whether this is the final transcription for an utterance.
            Set by STT services that support commit/finalize signals.
    """

    user_id: str
    timestamp: str
    language: Optional[Language] = None
    result: Optional[Any] = None
    finalized: bool = False

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

    .. deprecated:: 0.0.99
        `OpenAILLMContextAssistantTimestampFrame` is deprecated and will be removed in a future version.
        Use `LLMContextAssistantTimestampFrame` with the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.

    Parameters:
        timestamp: Timestamp when the assistant message was created.
    """

    timestamp: str

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "OpenAILLMContextAssistantTimestampFrame is deprecated and will be removed in a future version. "
                "Use LLMContextAssistantTimestampFrame with the universal LLMContext and LLMContextAggregatorPair instead. "
                "See OpenAILLMContext docstring for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class LLMContextAssistantTimestampFrame(DataFrame):
    """Timestamp information for assistant messages in LLM context.

    Parameters:
        timestamp: Timestamp when the assistant message was created.
    """

    timestamp: str


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

    .. deprecated:: 0.0.99
        `TranscriptionMessage` is deprecated and will be removed in a future version.
        Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.
    """

    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "TranscriptionMessage is deprecated and will be removed in a future version. "
                "Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class ThoughtTranscriptionMessage:
    """An LLM thought message in a conversation transcript.

    .. deprecated:: 0.0.99
        `ThoughtTranscriptionMessage` is deprecated and will be removed in a future version.
        Use `LLMAssistantAggregator`'s new events instead.
    """

    role: Literal["assistant"] = field(default="assistant", init=False)
    content: str
    timestamp: Optional[str] = None

    def __post_init__(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "ThoughtTranscriptionMessage is deprecated and will be removed in a future version. "
                "Use `LLMAssistantAggregator`'s new events instead.",
                DeprecationWarning,
                stacklevel=2,
            )


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

    .. deprecated:: 0.0.99
        `TranscriptionUpdateFrame` is deprecated and will be removed in a future version.
        Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.
    """

    messages: List[TranscriptionMessage | ThoughtTranscriptionMessage]

    def __post_init__(self):
        super().__post_init__()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "TranscriptionUpdateFrame is deprecated and will be removed in a future version. "
                "Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.",
                DeprecationWarning,
                stacklevel=2,
            )

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
class LLMThoughtStartFrame(ControlFrame):
    """Frame indicating the start of an LLM thought.

    Parameters:
        append_to_context: Whether the thought should be appended to the LLM context.
            If it is appended, the `llm` field is required, since it will be
            appended as an `LLMSpecificMessage`.
        llm: Optional identifier of the LLM provider for LLM-specific handling.
            Only required if `append_to_context` is True, as the thought is
            appended to context as an `LLMSpecificMessage`.
    """

    append_to_context: bool = False
    llm: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.append_to_context and self.llm is None:
            raise ValueError("When append_to_context is True, llm must be set")

    def __str__(self):
        pts = format_pts(self.pts)
        return (
            f"{self.name}(pts: {pts}, append_to_context: {self.append_to_context}, llm: {self.llm})"
        )


@dataclass
class LLMThoughtTextFrame(DataFrame):
    """Frame containing the text (or text chunk) of an LLM thought.

    Note that despite this containing text, it is a DataFrame and not a
    TextFrame, to avoid most typical text processing, such as TTS.

    Parameters:
        text: The text (or text chunk) of the thought.
    """

    text: str
    includes_inter_frame_spaces: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # Assume that thought text chunks include all necessary spaces
        self.includes_inter_frame_spaces = True

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, thought text: {self.text})"


@dataclass
class LLMThoughtEndFrame(ControlFrame):
    """Frame indicating the end of an LLM thought.

    Parameters:
        signature: Optional signature associated with the thought.
            This is used by Anthropic, which includes a signature at the end of
            each thought.
    """

    signature: Any = None

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, signature: {self.signature})"


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
class FunctionCallResultProperties:
    """Properties for configuring function call result behavior.

    Parameters:
        run_llm: Whether to run the LLM after receiving this result.
        on_context_updated: Callback to execute when context is updated.
    """

    run_llm: Optional[bool] = None
    on_context_updated: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class FunctionCallResultFrame(DataFrame, UninterruptibleFrame):
    """Frame containing the result of an LLM function call.

    This is an uninterruptible frame because once a result is generated we
    always want to update the context.

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

            .. deprecated:: 0.0.99
                Use  `LLMUserAggregator`'s new `user_mute_strategies` parameter instead.

        enable_metrics: Whether to enable performance metrics collection.
        enable_tracing: Whether to enable OpenTelemetry tracing.
        enable_usage_metrics: Whether to enable usage metrics collection.
        interruption_strategies: List of interruption handling strategies.

            .. deprecated:: 0.0.99
                Use  `LLMUserAggregator`'s new `user_turn_strategies` parameter instead.

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

    Parameters:
        reason: Optional reason for pushing a cancel frame.
    """

    reason: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(reason: {self.reason})"


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
        exception: The exception that occurred.
    """

    error: str
    fatal: bool = False
    processor: Optional["FrameProcessor"] = None
    exception: Optional[Exception] = None

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
    """Frame indicating that the user turn has started.

    Emitted when the user turn starts, which usually means that some
    transcriptions are already available.

    Parameters:
        emulated: Whether this event was emulated rather than detected by VAD.

            .. deprecated:: 0.0.99
                This field is deprecated and will be removed in a future version.

    """

    emulated: bool = False


@dataclass
class UserStoppedSpeakingFrame(SystemFrame):
    """Frame indicating that the user turn has ended.

    Emitted when the user turn ends. This usually coincides with the start of
    the bot turn.

    Parameters:
        emulated: Whether this event was emulated rather than detected by VAD.

            .. deprecated:: 0.0.99
                This field is deprecated and will be removed in a future version.

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

    .. deprecated:: 0.0.99
        This frame is deprecated and will be removed in a future version.
    """

    def __post_init__(self):
        super().__post_init__()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "EmulateUserStartedSpeakingFrame is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class EmulateUserStoppedSpeakingFrame(SystemFrame):
    """Frame to emulate user stopped speaking behavior.

    Emitted by internal processors upstream to emulate VAD behavior when a
    user stops speaking.

    .. deprecated:: 0.0.99
        This frame is deprecated and will be removed in a future version.
    """

    def __post_init__(self):
        super().__post_init__()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "EmulateUserStoppedSpeakingFrame is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )


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
class FunctionCallCancelFrame(SystemFrame):
    """Frame signaling that a function call has been cancelled.

    Parameters:
        function_name: Name of the function that was cancelled.
        tool_call_id: Unique identifier for the cancelled function call.
    """

    function_name: str
    tool_call_id: str


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

    A frame to request an image from the given user. The request might come with
    a text that can be later used to describe the requested image.

    Parameters:
        user_id: Identifier of the user to request image from.
        text: An optional text associated to the image request.
        append_to_context: Whether the requested image should be appended to the LLM context.
        video_source: Specific video source to capture from.
        function_name: Name of function that generated this request (if any).
        tool_call_id: Tool call ID if generated by function call (if any).
        result_callback: Optional callback to invoke when the image is retrieved.
        context: [DEPRECATED] Optional context for the image request.
    """

    user_id: str
    text: Optional[str] = None
    append_to_context: Optional[bool] = None
    video_source: Optional[str] = None
    function_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    result_callback: Optional[Any] = None
    context: Optional[Any] = None

    def __post_init__(self):
        super().__post_init__()

        if self.context:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`UserImageRequestFrame` field `context` is deprecated.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, append_to_context: {self.append_to_context}, {self.video_source})"


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
        text: An optional text associated to this image.
        append_to_context: Whether the requested image should be appended to the LLM context.
        request: The original image request frame if this is a response.
    """

    user_id: str = ""
    text: Optional[str] = None
    append_to_context: Optional[bool] = None
    request: Optional[UserImageRequestFrame] = None

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, user: {self.user_id}, source: {self.transport_source}, size: {self.size}, format: {self.format}, text: {self.text}, append_to_context: {self.append_to_context})"


@dataclass
class AssistantImageRawFrame(OutputImageRawFrame):
    """Frame containing an image generated by the assistant.

    Contains both the raw frame for display (superclass functionality) as well
    as the original image, which can get used directly in LLM contexts.

    Parameters:
        original_data: The original image data, which can get used directly in
            an LLM context message without further encoding.
        original_mime_type: The MIME type of the original image data.
    """

    original_data: Optional[bytes] = None
    original_mime_type: Optional[str] = None


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
    turn_params: Optional[BaseTurnParams] = None


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

    Parameters:
        reason: Optional reason for pushing an end frame.
    """

    reason: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(reason: {self.reason})"


@dataclass
class CancelTaskFrame(TaskFrame):
    """Frame to request immediate pipeline task cancellation.

    This is used to notify the pipeline task that the pipeline should be
    stopped immediately by pushing a CancelFrame downstream. This frame
    should be pushed upstream.

    Parameters:
        reason: Optional reason for pushing a cancel frame.
    """

    reason: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(reason: {self.reason})"


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

    Parameters:
        reason: Optional reason for pushing an end frame.
    """

    reason: Optional[Any] = None

    def __str__(self):
        return f"{self.name}(reason: {self.reason})"


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

    skip_tts: Optional[bool] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = None


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    """Frame indicating the end of an LLM response."""

    skip_tts: Optional[bool] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.skip_tts = None


@dataclass
class FunctionCallInProgressFrame(ControlFrame, UninterruptibleFrame):
    """Frame signaling that a function call is currently executing.

    This is an uninterruptible frame because we always want to update the
    context.

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
class VisionFullResponseStartFrame(LLMFullResponseStartFrame):
    """Frame indicating the beginning of a vision model response.

    Used to indicate the beginning of a vision model response. Followed by one
    or more VisionTextFrames and a final VisionFullResponseEndFrame.

    """

    pass


@dataclass
class VisionFullResponseEndFrame(LLMFullResponseEndFrame):
    """Frame indicating the end of a Vision model response."""

    pass


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
