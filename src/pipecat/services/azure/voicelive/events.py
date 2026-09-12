#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event models and data structures for Azure Voice Live API communication.

Based on Azure's Voice Live API documentation:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-api-reference-2026-07-15

Voice Live keeps the original Realtime event names (``response.audio.delta``,
``conversation.item.created``) and a flat session object, rather than the
``response.output_audio.delta`` names and nested ``session.audio`` object the
current OpenAI Realtime API uses.
"""

import json
import uuid
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

Modality: TypeAlias = Literal["text", "audio"]
"""A modality the model can respond with."""

InputAudioFormat: TypeAlias = Literal["pcm16", "g711_ulaw", "g711_alaw"]
"""Encoding of the audio sent to the service."""

OutputAudioFormat: TypeAlias = Literal[
    "pcm16", "pcm16_8000hz", "pcm16_16000hz", "g711_ulaw", "g711_alaw"
]
"""Encoding of the audio returned by the service."""


#
# Voice configuration
#


class OpenAIVoice(BaseModel):
    """A voice belonging to the generative model itself.

    Parameters:
        type: Voice type, always "openai".
        name: Voice name, e.g. "alloy" or "marin".
    """

    type: Literal["openai"] = "openai"
    name: str


class AzureStandardVoice(BaseModel):
    """A prebuilt Azure text-to-speech voice.

    Parameters:
        type: Voice type, always "azure-standard".
        name: Full voice name, e.g. "en-US-Ava:DragonHDLatestNeural".
        temperature: Variation across renditions of the same text, for HD
            voices. Higher values sound less uniform.
        style: Speaking style, for voices that offer styles.
        pitch: Pitch adjustment, e.g. "+5%".
        rate: Speaking rate, e.g. "1.2".
        volume: Volume adjustment, e.g. "+10%".
        custom_lexicon_url: Public URL of a custom lexicon to apply.
        custom_text_normalization_url: Public URL of custom text normalization
            rules to apply.
        prefer_locales: Locales to prefer when the voice is multilingual.
        locale: Locale to speak, for multilingual voices.
    """

    type: Literal["azure-standard"] = "azure-standard"
    name: str
    temperature: float | None = None
    style: str | None = None
    pitch: str | None = None
    rate: str | None = None
    volume: str | None = None
    custom_lexicon_url: str | None = None
    custom_text_normalization_url: str | None = None
    prefer_locales: list[str] | None = None
    locale: str | None = None


class AzureCustomVoice(BaseModel):
    """A custom neural voice trained on the caller's own recordings.

    Parameters:
        type: Voice type, always "azure-custom".
        name: Deployed custom voice name.
        endpoint_id: Deployment (endpoint) ID of the custom voice.
        temperature: Variation across renditions of the same text.
        rate: Speaking rate, e.g. "1.2".
        custom_lexicon_url: Public URL of a custom lexicon to apply.
    """

    type: Literal["azure-custom"] = "azure-custom"
    name: str
    endpoint_id: str
    temperature: float | None = None
    rate: str | None = None
    custom_lexicon_url: str | None = None


class AzurePersonalVoice(BaseModel):
    """A personal voice built from a short speaker sample.

    Parameters:
        type: Voice type, always "azure-personal".
        name: Personal voice name.
        model: Base model rendering the voice, e.g. "DragonLatestNeural".
        temperature: Variation across renditions of the same text.
        rate: Speaking rate, e.g. "1.2".
    """

    type: Literal["azure-personal"] = "azure-personal"
    name: str
    model: Literal["DragonLatestNeural", "DragonHDLatestNeural", "DragonHDOmniLatestNeural"]
    temperature: float | None = None
    rate: str | None = None


class AzureRealtimeNativeVoice(BaseModel):
    """A voice native to Azure's own realtime model.

    Parameters:
        type: Voice type, always "azure-realtime-native".
        name: Voice name.
        temperature: Variation across renditions of the same text.
    """

    type: Literal["azure-realtime-native"] = "azure-realtime-native"
    name: str
    temperature: float | None = None


Voice: TypeAlias = (
    OpenAIVoice
    | AzureStandardVoice
    | AzureCustomVoice
    | AzurePersonalVoice
    | AzureRealtimeNativeVoice
    | dict[str, Any]
)
"""Any voice configuration Voice Live accepts."""


#
# Turn detection
#


class EndOfUtteranceDetection(BaseModel):
    """Semantic end-of-turn detection layered on top of a VAD.

    Parameters:
        model: Detection model to use, e.g. "semantic_detection_v1".
        threshold: Confidence above which the turn is considered finished.
        timeout_ms: How long to keep waiting for the turn to finish.
    """

    model: str | None = None
    threshold: float | None = None
    timeout_ms: int | None = None


class TurnDetection(BaseModel):
    """Server-side turn detection configuration.

    The ``azure_semantic_vad`` types weigh what was said as well as whether
    anyone is speaking, so a mid-sentence pause doesn't end the turn.

    Parameters:
        type: Detection strategy to use.
        threshold: Speech probability above which audio counts as speech.
        prefix_padding_ms: Audio to keep from before speech was detected.
        silence_duration_ms: Silence that ends a turn.
        speech_duration_ms: Speech required before a turn starts.
        create_response: Whether the service responds when a turn ends.
        interrupt_response: Whether caller speech interrupts the response.
        auto_truncate: Whether an interrupted response is truncated in the
            conversation history.
        remove_filler_words: Whether fillers ("um", "uh") are dropped from
            transcription.
        eagerness: How readily a semantic VAD decides the turn is over.
        end_of_utterance_detection: Semantic end-of-turn detection settings.
        languages: Languages to detect, for multilingual semantic VAD.
    """

    type: (
        Literal[
            "server_vad",
            "semantic_vad",
            "azure_semantic_vad",
            "azure_semantic_vad_multilingual",
        ]
        | None
    ) = "azure_semantic_vad"
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    speech_duration_ms: int | None = None
    create_response: bool | None = None
    interrupt_response: bool | None = None
    auto_truncate: bool | None = None
    remove_filler_words: bool | None = None
    eagerness: Literal["auto", "low", "high"] | None = None
    end_of_utterance_detection: EndOfUtteranceDetection | None = None
    languages: list[str] | None = None


#
# Audio input processing
#


class InputAudioNoiseReduction(BaseModel):
    """Noise suppression applied to incoming audio.

    Parameters:
        type: Suppression model to apply.
    """

    type: Literal["near_field", "far_field", "azure_deep_noise_suppression"] = (
        "azure_deep_noise_suppression"
    )


class InputAudioEchoCancellation(BaseModel):
    """Echo cancellation applied to incoming audio.

    Keeps the agent from hearing its own playback. With ``reference_source``
    set to "client", the caller sends its own playback as a second channel and
    ``channels`` must be 2.

    Parameters:
        type: Cancellation model, always "server_echo_cancellation".
        reference_source: Where the reference signal comes from.
        channels: Channels in the incoming audio: 1 for microphone only, 2 when
            playback is supplied alongside it.
    """

    type: Literal["server_echo_cancellation"] = "server_echo_cancellation"
    reference_source: Literal["server", "client"] | None = None
    channels: int | None = None


class InputAudioTranscription(BaseModel):
    """Transcription of the caller's audio.

    Parameters:
        model: Transcription model to use.
        language: Language to transcribe, as BCP-47 or ISO 639-1.
        phrase_list: Words and phrases to bias recognition toward.
        prompt: Prompt steering transcription, for models that accept one.
        custom_speech: Custom speech model deployments, keyed by locale.
    """

    model: (
        Literal[
            "whisper-1",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe-diarize",
            "azure-speech",
            "mai-transcribe",
        ]
        | None
    ) = None
    language: str | None = None
    phrase_list: list[str] | None = None
    prompt: str | None = None
    custom_speech: dict[str, Any] | None = None


#
# Animation and avatar
#


class Animation(BaseModel):
    """Animation data emitted alongside audio.

    Parameters:
        model_name: Animation model to use.
        outputs: Animation streams to emit.
    """

    model_name: str | None = None
    outputs: list[Literal["blendshapes", "viseme_id"]] | None = None


class AvatarVideoResolution(BaseModel):
    """Pixel dimensions of the avatar video.

    Parameters:
        width: Width in pixels.
        height: Height in pixels.
    """

    width: int
    height: int


class AvatarVideo(BaseModel):
    """Encoding of the avatar video stream.

    Parameters:
        codec: Video codec, e.g. "h264".
        bitrate: Target bitrate in bits per second.
        resolution: Pixel dimensions of the video.
        crop: Crop region applied to the rendered frame.
        background_color: Background color behind the avatar, e.g. "#FFFFFFFF".
        background_image_url: Public URL of a background image.
    """

    codec: str | None = None
    bitrate: int | None = None
    resolution: AvatarVideoResolution | None = None
    crop: dict[str, Any] | None = None
    background_color: str | None = None
    background_image_url: str | None = None


class Avatar(BaseModel):
    """Avatar rendered alongside the audio response.

    Parameters:
        character: Avatar character to render, e.g. "lisa".
        style: Character style, e.g. "casual-sitting".
        customized: Whether ``character`` names a custom avatar.
        video: Encoding of the avatar video stream.
        ice_servers: ICE servers for the avatar's WebRTC connection.
    """

    character: str
    style: str | None = None
    customized: bool | None = None
    video: AvatarVideo | None = None
    ice_servers: list[dict[str, Any]] | None = None


#
# Tool definitions
#


class FunctionTool(BaseModel):
    """Custom function tool configuration.

    Parameters:
        type: Tool type, always "function".
        name: Name of the function.
        description: Description of what the function does.
        parameters: JSON schema for function parameters.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict[str, Any]


VoiceLiveTool: TypeAlias = FunctionTool | dict[str, Any]
"""A tool definition Voice Live accepts."""


#
# Session properties
#


class SessionProperties(BaseModel):
    """Configuration properties for a Voice Live session.

    Voice Live carries audio settings directly on the session rather than
    nesting them under an ``audio`` object.

    Parameters:
        model: Model backing the session, e.g. "gpt-4o-mini". Usually named in
            the connection URL instead.
        instructions: System instructions for the assistant.
        modalities: Modalities the assistant responds with.
        voice: Voice used for audio responses.
        input_audio_format: Encoding of the audio sent to the service.
        output_audio_format: Encoding of the audio returned by the service.
        input_audio_sampling_rate: Sample rate of the audio sent to the service.
        output_audio_timestamp_types: Timestamp streams to emit alongside audio.
        turn_detection: Turn detection settings, or None/False to disable the
            service's own turn detection.
        input_audio_transcription: Transcription settings for caller audio.
        input_audio_noise_reduction: Noise suppression settings.
        input_audio_echo_cancellation: Echo cancellation settings.
        temperature: Sampling temperature for responses.
        max_response_output_tokens: Output token ceiling per response, or "inf".
        tools: Tools the assistant may call.
        tool_choice: How the assistant picks among tools.
        animation: Animation streams to emit alongside audio.
        avatar: Avatar rendered alongside the audio response.
    """

    # Needed to support ToolsSchema in tools field.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | None = None
    instructions: str | None = None
    modalities: list[Modality] | None = None
    voice: Voice | None = None
    input_audio_format: InputAudioFormat | None = None
    output_audio_format: OutputAudioFormat | None = None
    input_audio_sampling_rate: int | None = None
    output_audio_timestamp_types: list[Literal["word"]] | None = None
    turn_detection: TurnDetection | bool | None = None
    input_audio_transcription: InputAudioTranscription | None = None
    input_audio_noise_reduction: InputAudioNoiseReduction | None = None
    input_audio_echo_cancellation: InputAudioEchoCancellation | None = None
    temperature: float | None = None
    max_response_output_tokens: int | Literal["inf"] | None = None
    # Tools provided by the user may be a ToolsSchema or a plain list of standard
    # tools (the validator below normalizes that to a ToolsSchema); a list of
    # provider-native VoiceLiveTool objects passes through.
    tools: ToolsSchema | list[FunctionSchema | DirectFunction] | list[VoiceLiveTool] | None = None
    tool_choice: str | None = None
    animation: Animation | None = None
    avatar: Avatar | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, v):
        """Wrap a plain list of standard tools in a ``ToolsSchema``.

        Provider-native tool lists pass through unchanged.
        """
        if isinstance(v, list):
            normalized = LLMContext._normalize_and_validate_tools(v, allow_provider_tools=True)
            return normalized if isinstance(normalized, (ToolsSchema, list)) else None
        return v


#
# Conversation items
#


class ItemContent(BaseModel):
    """A single piece of content within a conversation item.

    Parameters:
        type: Content type.
        text: Text content, for text parts.
        audio: Base64-encoded audio, for audio parts.
        transcript: Transcript of the audio, for audio parts.
    """

    type: Literal["input_text", "input_audio", "text", "audio"]
    text: str | None = None
    audio: str | None = None
    transcript: str | None = None


class ConversationItem(BaseModel):
    """An item in the conversation history.

    Parameters:
        id: Identifier of the item, assigned by the service when absent.
        object: Object type, always "realtime.item".
        type: Kind of item this is.
        status: Whether the item is finished.
        role: Who the item belongs to, for message items.
        content: Content parts, for message items.
        call_id: Identifier tying a call to its output.
        name: Function name, for function call items.
        arguments: JSON-encoded arguments, for function call items.
        output: JSON-encoded result, for function call output items.
    """

    id: str | None = None
    object: Literal["realtime.item"] | None = None
    type: Literal["message", "function_call", "function_call_output"]
    status: Literal["completed", "in_progress", "incomplete"] | None = None
    role: Literal["user", "assistant", "system"] | None = None
    content: list[ItemContent] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None


class ResponseProperties(BaseModel):
    """Overrides applied to a single response.

    Parameters:
        modalities: Modalities this response uses.
        instructions: Instructions for this response only.
        voice: Voice for this response only.
        temperature: Sampling temperature for this response.
        max_output_tokens: Output token ceiling for this response, or "inf".
        tools: Tools available to this response.
        tool_choice: How this response picks among tools.
    """

    modalities: list[Modality] | None = None
    instructions: str | None = None
    voice: Voice | None = None
    temperature: float | None = None
    max_output_tokens: int | Literal["inf"] | None = None
    tools: list[VoiceLiveTool] | None = None
    tool_choice: str | None = None


class TokenUsageDetails(BaseModel):
    """Per-modality breakdown of token usage.

    Parameters:
        cached_tokens: Tokens served from cache.
        text_tokens: Tokens attributable to text.
        audio_tokens: Tokens attributable to audio.
    """

    cached_tokens: int | None = None
    text_tokens: int | None = None
    audio_tokens: int | None = None


class TokenUsage(BaseModel):
    """Token usage reported for a response.

    Parameters:
        total_tokens: Tokens across input and output.
        input_tokens: Tokens in the input.
        output_tokens: Tokens in the output.
        input_token_details: Per-modality breakdown of the input.
        output_token_details: Per-modality breakdown of the output.
    """

    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_token_details: TokenUsageDetails | None = None
    output_token_details: TokenUsageDetails | None = None


class RealtimeError(BaseModel):
    """An error reported by the service.

    Parameters:
        type: Error category.
        code: Machine-readable error code.
        message: Human-readable description.
        param: Parameter the error relates to.
        event_id: Client event that caused the error.
    """

    type: str | None = None
    code: str | None = None
    message: str | None = None
    param: str | None = None
    event_id: str | None = None


#
# Client events (sent to Voice Live)
#


class ClientEvent(BaseModel):
    """Base class for client events sent to the Voice Live API.

    Parameters:
        event_id: Unique identifier for the event, auto-generated if not provided.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SessionUpdateEvent(ClientEvent):
    """Event to update session properties.

    Parameters:
        type: Event type, always "session.update".
        session: Updated session properties.
    """

    type: Literal["session.update"] = "session.update"
    session: SessionProperties

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Serialize the event to a dictionary.

        Turn detection is disabled by an explicit null. Leaving the field out
        instead keeps the service's own server VAD running, so a session that
        set it to None or False has the null restored after ``exclude_none``
        drops it.

        Args:
            *args: Positional arguments passed to parent model_dump.
            **kwargs: Keyword arguments passed to parent model_dump.

        Returns:
            Dictionary representation of the event.
        """
        dump = super().model_dump(*args, **kwargs)

        if "turn_detection" in self.session.model_fields_set and not self.session.turn_detection:
            dump["session"]["turn_detection"] = None

        return dump


class InputAudioBufferAppendEvent(ClientEvent):
    """Event to append audio to the input buffer.

    Parameters:
        type: Event type, always "input_audio_buffer.append".
        audio: Base64-encoded audio to append.
    """

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str


class InputAudioBufferCommitEvent(ClientEvent):
    """Event to commit the input buffer as a conversation item.

    Parameters:
        type: Event type, always "input_audio_buffer.commit".
    """

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(ClientEvent):
    """Event to discard the buffered input audio.

    Parameters:
        type: Event type, always "input_audio_buffer.clear".
    """

    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreateEvent(ClientEvent):
    """Event to add an item to the conversation.

    Parameters:
        type: Event type, always "conversation.item.create".
        item: The item to add.
        previous_item_id: Item to insert after, or None to append.
    """

    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: ConversationItem
    previous_item_id: str | None = None


class ConversationItemTruncateEvent(ClientEvent):
    """Event to truncate an assistant audio item the caller interrupted.

    Parameters:
        type: Event type, always "conversation.item.truncate".
        item_id: Item to truncate.
        content_index: Content part to truncate.
        audio_end_ms: Point, in milliseconds, to truncate the audio at.
    """

    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int = 0
    audio_end_ms: int = 0


class ConversationItemDeleteEvent(ClientEvent):
    """Event to remove an item from the conversation.

    Parameters:
        type: Event type, always "conversation.item.delete".
        item_id: Item to remove.
    """

    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ResponseCreateEvent(ClientEvent):
    """Event asking the model to respond.

    Parameters:
        type: Event type, always "response.create".
        response: Overrides applied to this response.
    """

    type: Literal["response.create"] = "response.create"
    response: ResponseProperties | None = None


class ResponseCancelEvent(ClientEvent):
    """Event to cancel the response in progress.

    Parameters:
        type: Event type, always "response.cancel".
    """

    type: Literal["response.cancel"] = "response.cancel"


#
# Server events (received from Voice Live)
#


class ServerEvent(BaseModel):
    """Base class for server events received from the Voice Live API.

    Parameters:
        event_id: Unique identifier assigned by the service.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str | None = None


class SessionCreatedEvent(ServerEvent):
    """Event reporting that the session is open.

    Parameters:
        type: Event type, always "session.created".
        session: The session as created.
    """

    type: Literal["session.created"]
    session: dict[str, Any]


class SessionUpdatedEvent(ServerEvent):
    """Event reporting that the session configuration was applied.

    Parameters:
        type: Event type, always "session.updated".
        session: The session as updated.
    """

    type: Literal["session.updated"]
    session: dict[str, Any]


class ConversationItemCreated(ServerEvent):
    """Event reporting that an item was added to the conversation.

    Parameters:
        type: Event type, always "conversation.item.created".
        item: The item that was added.
        previous_item_id: Item it was inserted after.
    """

    type: Literal["conversation.item.created"]
    item: ConversationItem
    previous_item_id: str | None = None


class ConversationItemTruncated(ServerEvent):
    """Event reporting that an assistant audio item was truncated.

    Parameters:
        type: Event type, always "conversation.item.truncated".
        item_id: Item that was truncated.
        content_index: Content part that was truncated.
        audio_end_ms: Point, in milliseconds, the audio was truncated at.
    """

    type: Literal["conversation.item.truncated"]
    item_id: str
    content_index: int = 0
    audio_end_ms: int = 0


class ConversationItemDeleted(ServerEvent):
    """Event reporting that an item was removed from the conversation.

    Parameters:
        type: Event type, always "conversation.item.deleted".
        item_id: Item that was removed.
    """

    type: Literal["conversation.item.deleted"]
    item_id: str


class ConversationItemInputAudioTranscriptionDelta(ServerEvent):
    """Event carrying part of a caller transcript.

    Parameters:
        type: Event type, always
            "conversation.item.input_audio_transcription.delta".
        item_id: Item being transcribed.
        content_index: Content part being transcribed.
        delta: Transcript text for this update.
    """

    type: Literal["conversation.item.input_audio_transcription.delta"]
    item_id: str
    content_index: int = 0
    delta: str | None = None


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    """Event carrying a finished caller transcript.

    Parameters:
        type: Event type, always
            "conversation.item.input_audio_transcription.completed".
        item_id: Item that was transcribed.
        content_index: Content part that was transcribed.
        transcript: The finished transcript.
    """

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
    content_index: int = 0
    transcript: str = ""


class ConversationItemInputAudioTranscriptionFailed(ServerEvent):
    """Event reporting that a caller transcript could not be produced.

    Parameters:
        type: Event type, always
            "conversation.item.input_audio_transcription.failed".
        item_id: Item that failed to transcribe.
        content_index: Content part that failed to transcribe.
        error: Why transcription failed.
    """

    type: Literal["conversation.item.input_audio_transcription.failed"]
    item_id: str
    content_index: int = 0
    error: RealtimeError | None = None


class InputAudioBufferCommitted(ServerEvent):
    """Event reporting that the input buffer became a conversation item.

    Parameters:
        type: Event type, always "input_audio_buffer.committed".
        item_id: Item the buffer became.
        previous_item_id: Item it was inserted after.
    """

    type: Literal["input_audio_buffer.committed"]
    item_id: str | None = None
    previous_item_id: str | None = None


class InputAudioBufferCleared(ServerEvent):
    """Event reporting that the input buffer was discarded.

    Parameters:
        type: Event type, always "input_audio_buffer.cleared".
    """

    type: Literal["input_audio_buffer.cleared"]


class InputAudioBufferSpeechStarted(ServerEvent):
    """Event reporting that the caller started speaking.

    Parameters:
        type: Event type, always "input_audio_buffer.speech_started".
        audio_start_ms: Where in the buffered audio speech began.
        item_id: Item the speech belongs to.
    """

    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: int | None = None
    item_id: str | None = None


class InputAudioBufferSpeechStopped(ServerEvent):
    """Event reporting that the caller stopped speaking.

    Parameters:
        type: Event type, always "input_audio_buffer.speech_stopped".
        audio_end_ms: Where in the buffered audio speech ended.
        item_id: Item the speech belongs to.
    """

    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: int | None = None
    item_id: str | None = None


class ResponseCreated(ServerEvent):
    """Event reporting that a response has begun.

    Parameters:
        type: Event type, always "response.created".
        response: The response as created.
    """

    type: Literal["response.created"]
    response: dict[str, Any]


class ResponseDone(ServerEvent):
    """Event reporting that a response has finished.

    Parameters:
        type: Event type, always "response.done".
        response: The finished response, including status and token usage.
    """

    type: Literal["response.done"]
    response: dict[str, Any]

    @property
    def status(self) -> str | None:
        """Status the response finished with."""
        return self.response.get("status")

    @property
    def usage(self) -> TokenUsage | None:
        """Token usage reported for the response, when present."""
        usage = self.response.get("usage")
        return TokenUsage.model_validate(usage) if usage else None


class ResponseOutputItemAdded(ServerEvent):
    """Event reporting that an output item was added to a response.

    Parameters:
        type: Event type, always "response.output_item.added".
        response_id: Response the item belongs to.
        output_index: Position of the item in the output.
        item: The item that was added.
    """

    type: Literal["response.output_item.added"]
    response_id: str | None = None
    output_index: int = 0
    item: ConversationItem


class ResponseOutputItemDone(ServerEvent):
    """Event reporting that an output item is finished.

    Parameters:
        type: Event type, always "response.output_item.done".
        response_id: Response the item belongs to.
        output_index: Position of the item in the output.
        item: The finished item.
    """

    type: Literal["response.output_item.done"]
    response_id: str | None = None
    output_index: int = 0
    item: ConversationItem


class ResponseContentPartAdded(ServerEvent):
    """Event reporting that a content part was added to an output item.

    Parameters:
        type: Event type, always "response.content_part.added".
        response_id: Response the part belongs to.
        item_id: Item the part belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        part: The part that was added.
    """

    type: Literal["response.content_part.added"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    part: dict[str, Any] | None = None


class ResponseContentPartDone(ServerEvent):
    """Event reporting that a content part is finished.

    Parameters:
        type: Event type, always "response.content_part.done".
        response_id: Response the part belongs to.
        item_id: Item the part belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        part: The finished part.
    """

    type: Literal["response.content_part.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    part: dict[str, Any] | None = None


class ResponseTextDelta(ServerEvent):
    """Event carrying part of a text response.

    Parameters:
        type: Event type, always "response.text.delta".
        response_id: Response the text belongs to.
        item_id: Item the text belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        delta: Text for this update.
    """

    type: Literal["response.text.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseTextDone(ServerEvent):
    """Event carrying a finished text response.

    Parameters:
        type: Event type, always "response.text.done".
        response_id: Response the text belongs to.
        item_id: Item the text belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        text: The finished text.
    """

    type: Literal["response.text.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    text: str = ""


class ResponseAudioDelta(ServerEvent):
    """Event carrying part of an audio response.

    Parameters:
        type: Event type, always "response.audio.delta".
        response_id: Response the audio belongs to.
        item_id: Item the audio belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        delta: Base64-encoded audio for this update.
    """

    type: Literal["response.audio.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseAudioDone(ServerEvent):
    """Event reporting that an audio response is finished.

    Parameters:
        type: Event type, always "response.audio.done".
        response_id: Response the audio belongs to.
        item_id: Item the audio belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
    """

    type: Literal["response.audio.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0


class ResponseAudioTranscriptDelta(ServerEvent):
    """Event carrying part of the transcript of an audio response.

    Parameters:
        type: Event type, always "response.audio_transcript.delta".
        response_id: Response the transcript belongs to.
        item_id: Item the transcript belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        delta: Transcript text for this update.
    """

    type: Literal["response.audio_transcript.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseAudioTranscriptDone(ServerEvent):
    """Event carrying the finished transcript of an audio response.

    Parameters:
        type: Event type, always "response.audio_transcript.done".
        response_id: Response the transcript belongs to.
        item_id: Item the transcript belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        transcript: The finished transcript.
    """

    type: Literal["response.audio_transcript.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    transcript: str = ""


class ResponseAudioTimestampDelta(ServerEvent):
    """Event carrying the timing of one spoken word.

    Emitted when ``output_audio_timestamp_types`` includes "word".

    Parameters:
        type: Event type, always "response.audio_timestamp.delta".
        response_id: Response the word belongs to.
        item_id: Item the word belongs to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
        audio_offset_ms: Where in the response audio the word begins.
        audio_duration_ms: How long the word takes to speak.
        text: The word itself.
        timestamp_type: Granularity of the timestamp, always "word".
    """

    type: Literal["response.audio_timestamp.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    audio_offset_ms: int = 0
    audio_duration_ms: int = 0
    text: str = ""
    timestamp_type: Literal["word"] | None = None


class ResponseAudioTimestampDone(ServerEvent):
    """Event reporting that word timings for a response are finished.

    Parameters:
        type: Event type, always "response.audio_timestamp.done".
        response_id: Response the timings belong to.
        item_id: Item the timings belong to.
        output_index: Position of the item in the output.
        content_index: Position of the part within the item.
    """

    type: Literal["response.audio_timestamp.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0


class ResponseFunctionCallArgumentsDelta(ServerEvent):
    """Event carrying part of a function call's arguments.

    Parameters:
        type: Event type, always "response.function_call_arguments.delta".
        response_id: Response the call belongs to.
        item_id: Item the call belongs to.
        output_index: Position of the item in the output.
        call_id: Identifier tying the call to its output.
        delta: Argument text for this update.
    """

    type: Literal["response.function_call_arguments.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    call_id: str | None = None
    delta: str = ""


class ResponseFunctionCallArgumentsDone(ServerEvent):
    """Event carrying a function call's finished arguments.

    Parameters:
        type: Event type, always "response.function_call_arguments.done".
        response_id: Response the call belongs to.
        item_id: Item the call belongs to.
        output_index: Position of the item in the output.
        call_id: Identifier tying the call to its output.
        name: Function being called.
        arguments: JSON-encoded arguments.
    """

    type: Literal["response.function_call_arguments.done"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


class RateLimitsUpdated(ServerEvent):
    """Event reporting the caller's remaining rate limit allowance.

    Parameters:
        type: Event type, always "rate_limits.updated".
        rate_limits: Remaining allowance per limit.
    """

    type: Literal["rate_limits.updated"]
    rate_limits: list[dict[str, Any]] = Field(default_factory=list)


class ErrorEvent(ServerEvent):
    """Event reporting that an error occurred.

    Parameters:
        type: Event type, always "error".
        error: Error details.
    """

    type: Literal["error"]
    error: RealtimeError


#
# Event parsing
#

_server_event_types = {
    "error": ErrorEvent,
    "rate_limits.updated": RateLimitsUpdated,
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "conversation.item.created": ConversationItemCreated,
    "conversation.item.truncated": ConversationItemTruncated,
    "conversation.item.deleted": ConversationItemDeleted,
    "conversation.item.input_audio_transcription.delta": ConversationItemInputAudioTranscriptionDelta,
    "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompleted,
    "conversation.item.input_audio_transcription.failed": ConversationItemInputAudioTranscriptionFailed,
    "input_audio_buffer.committed": InputAudioBufferCommitted,
    "input_audio_buffer.cleared": InputAudioBufferCleared,
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
    "response.created": ResponseCreated,
    "response.done": ResponseDone,
    "response.output_item.added": ResponseOutputItemAdded,
    "response.output_item.done": ResponseOutputItemDone,
    "response.content_part.added": ResponseContentPartAdded,
    "response.content_part.done": ResponseContentPartDone,
    "response.text.delta": ResponseTextDelta,
    "response.text.done": ResponseTextDone,
    "response.audio.delta": ResponseAudioDelta,
    "response.audio.done": ResponseAudioDone,
    "response.audio_transcript.delta": ResponseAudioTranscriptDelta,
    "response.audio_transcript.done": ResponseAudioTranscriptDone,
    "response.audio_timestamp.delta": ResponseAudioTimestampDelta,
    "response.audio_timestamp.done": ResponseAudioTimestampDone,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDelta,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDone,
}


def parse_server_event(data: str | bytes):
    """Parse a server event from JSON.

    Args:
        data: JSON text containing the server event, as delivered by the
            websocket.

    Returns:
        Parsed server event object of the appropriate type, or ``None`` if the
        event type is not recognized (e.g. an avatar or animation event, or a
        newer server event without a model).

    Raises:
        Exception: If a recognized event type fails to parse.
    """
    event = json.loads(data)
    event_type = event["type"]
    if event_type not in _server_event_types:
        return None
    try:
        return _server_event_types[event_type].model_validate(event)
    except Exception as e:
        raise Exception(f"{e} \n\n{data}")
