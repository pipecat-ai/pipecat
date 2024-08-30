#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, List, Mapping, Optional, Tuple

from dataclasses import dataclass, field

from pipecat.transcriptions.language import Language
from pipecat.utils.utils import obj_count, obj_id
from pipecat.vad.vad_analyzer import VADParams


@dataclass
class Frame:
    id: int = field(init=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

    def __str__(self):
        return self.name


@dataclass
class DataFrame(Frame):
    pass


@dataclass
class AudioRawFrame(DataFrame):
    """A chunk of audio. Will be played by the transport if the transport's
    microphone has been enabled.

    """
    audio: bytes
    sample_rate: int
    num_channels: int

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        return f"{self.name}(size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class ImageRawFrame(DataFrame):
    """An image. Will be shown by the transport if the transport's camera is
    enabled.

    """
    image: bytes
    size: Tuple[int, int]
    format: str | None

    def __str__(self):
        return f"{self.name}(size: {self.size}, format: {self.format})"


@dataclass
class URLImageRawFrame(ImageRawFrame):
    """An image with an associated URL. Will be shown by the transport if the
    transport's camera is enabled.

    """
    url: str | None

    def __str__(self):
        return f"{self.name}(url: {self.url}, size: {self.size}, format: {self.format})"


@dataclass
class VisionImageRawFrame(ImageRawFrame):
    """An image with an associated text to ask for a description of it. Will be
    shown by the transport if the transport's camera is enabled.

    """
    text: str | None

    def __str__(self):
        return f"{self.name}(text: {self.text}, size: {self.size}, format: {self.format})"


@dataclass
class UserImageRawFrame(ImageRawFrame):
    """An image associated to a user. Will be shown by the transport if the
    transport's camera is enabled.

    """
    user_id: str

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, size: {self.size}, format: {self.format})"


@dataclass
class SpriteFrame(Frame):
    """An animated sprite. Will be shown by the transport if the transport's
    camera is enabled. Will play at the framerate specified in the transport's
    `camera_out_framerate` constructor parameter.

    """
    images: List[ImageRawFrame]

    def __str__(self):
        return f"{self.name}(size: {len(self.images)})"


@dataclass
class TextFrame(DataFrame):
    """A chunk of text. Emitted by LLM services, consumed by TTS services, can
    be used to send text through pipelines.

    """
    text: str

    def __str__(self):
        return f"{self.name}(text: {self.text})"


@dataclass
class TranscriptionFrame(TextFrame):
    """A text frame with transcription-specific data. Will be placed in the
    transport's receive queue when a participant speaks.

    """
    user_id: str
    timestamp: str
    language: Language | None = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class InterimTranscriptionFrame(TextFrame):
    """A text frame with interim transcription-specific data. Will be placed in
    the transport's receive queue when a participant speaks."""
    user_id: str
    timestamp: str
    language: Language | None = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: {self.text}, language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class LLMMessagesFrame(DataFrame):
    """A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLMStartFrames, TextFrames
    and an LLMEndFrame. Note that the messages property on this class is
    mutable, and will be be updated by various ResponseAggregator frame
    processors.

    """
    messages: List[dict]


@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """A frame containing a list of LLM messages that neeed to be added to the
    current context.

    """
    messages: List[dict]


@dataclass
class LLMMessagesUpdateFrame(DataFrame):
    """A frame containing a list of new LLM messages. These messages will
    replace the current context LLM messages and should generate a new
    LLMMessagesFrame.

    """
    messages: List[dict]


@dataclass
class LLMSetToolsFrame(DataFrame):
    """A frame containing a list of tools for an LLM to use for function calling.
    The specific format depends on the LLM being used, but it should typically
    contain JSON Schema objects.
    """
    tools: List[dict]


@dataclass
class LLMEnablePromptCachingFrame(DataFrame):
    """A frame to enable/disable prompt caching in certain LLMs.
    """
    enable: bool


@dataclass
class TTSSpeakFrame(DataFrame):
    """A frame that contains a text that should be spoken by the TTS in the
    pipeline (if any).

    """
    text: str


@dataclass
class TransportMessageFrame(DataFrame):
    message: Any
    urgent: bool = False

    def __str__(self):
        return f"{self.name}(message: {self.message})"

#
# App frames. Application user-defined frames.
#


@dataclass
class AppFrame(Frame):
    pass

#
# System frames
#


@dataclass
class SystemFrame(Frame):
    pass


@dataclass
class CancelFrame(SystemFrame):
    """Indicates that a pipeline needs to stop right away."""
    pass


@dataclass
class ErrorFrame(SystemFrame):
    """This is used notify upstream that an error has occurred downstream the
    pipeline. A fatal error indicates the error is unrecoverable and that the
    bot should exit.

    """
    error: str
    fatal: bool = False

    def __str__(self):
        return f"{self.name}(error: {self.error}, fatal: {self.fatal})"


@dataclass
class FatalErrorFrame(ErrorFrame):
    """This is used notify upstream that an unrecoverable error has occurred and
    that the bot should exit.

    """
    fatal: bool = field(default=True, init=False)


@dataclass
class StopTaskFrame(SystemFrame):
    """Indicates that a pipeline task should be stopped but that the pipeline
    processors should be kept in a running state. This is normally queued from
    the pipeline task.

    """
    pass


@dataclass
class StartInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has started speaking (i.e. is
    interruption). This is similar to UserStartedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """
    pass


@dataclass
class StopInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has stopped speaking (i.e. no more
    interruptions). This is similar to UserStoppedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """
    pass


@dataclass
class BotInterruptionFrame(SystemFrame):
    """Emitted by when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.

    """
    pass


@dataclass
class MetricsFrame(SystemFrame):
    """Emitted by processor that can compute metrics like latencies.
    """
    ttfb: List[Mapping[str, Any]] | None = None
    processing: List[Mapping[str, Any]] | None = None
    tokens: List[Mapping[str, Any]] | None = None
    characters: List[Mapping[str, Any]] | None = None

#
# Control frames
#


@dataclass
class ControlFrame(Frame):
    pass


@dataclass
class StartFrame(ControlFrame):
    """This is the first frame that should be pushed down a pipeline."""
    allow_interruptions: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    report_only_initial_ttfb: bool = False


@dataclass
class EndFrame(ControlFrame):
    """Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will received in the order it
    was sent (unline system frames).

    """
    pass


@dataclass
class LLMFullResponseStartFrame(ControlFrame):
    """Used to indicate the beginning of an LLM response. Following by one or
    more TextFrame and a final LLMFullResponseEndFrame."""
    pass


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    """Indicates the end of an LLM response."""
    pass


@dataclass
class UserStartedSpeakingFrame(ControlFrame):
    """Emitted by VAD to indicate that a user has started speaking. This can be
    used for interruptions or other times when detecting that someone is
    speaking is more important than knowing what they're saying (as you will
    with a TranscriptionFrame)

    """
    pass


@dataclass
class UserStoppedSpeakingFrame(ControlFrame):
    """Emitted by the VAD to indicate that a user stopped speaking."""
    pass


@dataclass
class BotStartedSpeakingFrame(ControlFrame):
    """Emitted upstream by transport outputs to indicate the bot started speaking.

    """
    pass


@dataclass
class BotStoppedSpeakingFrame(ControlFrame):
    """Emitted upstream by transport outputs to indicate the bot stopped speaking.

    """
    pass


@dataclass
class BotSpeakingFrame(ControlFrame):
    """Emitted upstream by transport outputs while the bot is still
    speaking. This can be used, for example, to detect when a user is idle. That
    is, while the bot is speaking we don't want to trigger any user idle timeout
    since the user might be listening.

    """
    pass


@dataclass
class TTSStartedFrame(ControlFrame):
    """Used to indicate the beginning of a TTS response. Following
    AudioRawFrames are part of the TTS response until an TTSEndFrame. These
    frames can be used for aggregating audio frames in a transport to optimize
    the size of frames sent to the session, without needing to control this in
    the TTS service.

    """
    pass


@dataclass
class TTSStoppedFrame(ControlFrame):
    """Indicates the end of a TTS response."""
    pass


@dataclass
class UserImageRequestFrame(ControlFrame):
    """A frame user to request an image from the given user."""
    user_id: str
    context: Optional[Any] = None

    def __str__(self):
        return f"{self.name}, user: {self.user_id}"


@dataclass
class LLMModelUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new LLM model.
    """
    model: str


@dataclass
class TTSModelUpdateFrame(ControlFrame):
    """A control frame containing a request to update the TTS model.
    """
    model: str


@dataclass
class TTSVoiceUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new TTS voice.
    """
    voice: str


@dataclass
class TTSLanguageUpdateFrame(ControlFrame):
    """A control frame containing a request to update to a new TTS language and
    optional voice.

    """
    language: Language


@dataclass
class STTModelUpdateFrame(ControlFrame):
    """A control frame containing a request to update the STT model and optional
    language.

    """
    model: str


@dataclass
class STTLanguageUpdateFrame(ControlFrame):
    """A control frame containing a request to update to STT language.
    """
    language: Language


@dataclass
class FunctionCallInProgressFrame(SystemFrame):
    """A frame signaling that a function call is in progress.
    """
    function_name: str
    tool_call_id: str
    arguments: str


@dataclass
class FunctionCallResultFrame(DataFrame):
    """A frame containing the result of an LLM function (tool) call.
    """
    function_name: str
    tool_call_id: str
    arguments: str
    result: Any


@dataclass
class VADParamsUpdateFrame(ControlFrame):
    """A control frame containing a request to update VAD params. Intended
    to be pushed upstream from RTVI processor.
    """
    params: VADParams
