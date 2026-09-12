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

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator

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
        turn_detection: Turn detection settings, or None to disable.
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
    turn_detection: TurnDetection | None = None
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
