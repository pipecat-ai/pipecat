#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration for the Gladia STT service."""

from typing import Any

from pydantic import BaseModel


class LanguageConfig(BaseModel):
    """Configuration for language detection and handling.

    Parameters:
        languages: List of language codes to use for transcription
        code_switching: Whether to auto-detect language changes during transcription
    """

    languages: list[str] | None = None
    code_switching: bool | None = None


class PreProcessingConfig(BaseModel):
    """Configuration for audio pre-processing options.

    Parameters:
        audio_enhancer: Apply pre-processing to the audio stream to enhance quality
        speech_threshold: Sensitivity for speech detection (0-1)
    """

    audio_enhancer: bool | None = None
    speech_threshold: float | None = None


class CustomVocabularyItem(BaseModel):
    """Represents a custom vocabulary item with an intensity value.

    Parameters:
        value: The vocabulary word or phrase
        intensity: The bias intensity for this vocabulary item (0-1)
        pronunciations: The pronunciations used in the transcription.
        language: Specify the language in which it will be pronounced when sound comparison occurs. Default to transcription language.
    """

    value: str
    intensity: float
    pronunciations: list[str] | None = None
    language: str | None = None


class CustomVocabularyConfig(BaseModel):
    """Configuration for custom vocabulary.

    Parameters:
        vocabulary: List of words/phrases or CustomVocabularyItem objects
        default_intensity: Default intensity for simple string vocabulary items
    """

    vocabulary: list[str | CustomVocabularyItem] | None = None
    default_intensity: float | None = None


class CustomSpellingConfig(BaseModel):
    """Configuration for custom spelling rules.

    Parameters:
        spelling_dictionary: Mapping of correct spellings to phonetic variations
    """

    spelling_dictionary: dict[str, list[str]] | None = None


class TranslationConfig(BaseModel):
    """Configuration for real-time translation.

    Parameters:
        target_languages: List of target language codes for translation
        model: Translation model to use ("base" or "enhanced")
        match_original_utterances: Whether to align translations with original utterances
        lipsync: Whether to enable lip-sync optimization for translations
        context_adaptation: Whether to enable context-aware translation adaptation
        context: Additional context to help with translation accuracy
        informal: Force informal language forms when available
    """

    target_languages: list[str] | None = None
    model: str | None = None
    match_original_utterances: bool | None = None
    lipsync: bool | None = None
    context_adaptation: bool | None = None
    context: str | None = None
    informal: bool | None = None


class RealtimeProcessingConfig(BaseModel):
    """Configuration for real-time processing features.

    Parameters:
        words_accurate_timestamps: Whether to provide per-word timestamps
        custom_vocabulary: Whether to enable custom vocabulary
        custom_vocabulary_config: Custom vocabulary configuration
        custom_spelling: Whether to enable custom spelling
        custom_spelling_config: Custom spelling configuration
        translation: Whether to enable translation
        translation_config: Translation configuration
        named_entity_recognition: Whether to enable named entity recognition
        sentiment_analysis: Whether to enable sentiment analysis
    """

    words_accurate_timestamps: bool | None = None
    custom_vocabulary: bool | None = None
    custom_vocabulary_config: CustomVocabularyConfig | None = None
    custom_spelling: bool | None = None
    custom_spelling_config: CustomSpellingConfig | None = None
    translation: bool | None = None
    translation_config: TranslationConfig | None = None
    named_entity_recognition: bool | None = None
    sentiment_analysis: bool | None = None


class MessagesConfig(BaseModel):
    """Configuration for controlling which message types are sent via WebSocket.

    Parameters:
        receive_partial_transcripts: Whether to receive intermediate transcription results
        receive_final_transcripts: Whether to receive final transcription results
        receive_speech_events: Whether to receive speech begin/end events
        receive_pre_processing_events: Whether to receive pre-processing events
        receive_realtime_processing_events: Whether to receive real-time processing events
        receive_post_processing_events: Whether to receive post-processing events
        receive_acknowledgments: Whether to receive acknowledgment messages
        receive_errors: Whether to receive error messages
        receive_lifecycle_events: Whether to receive lifecycle events
    """

    receive_partial_transcripts: bool | None = None
    receive_final_transcripts: bool | None = None
    receive_speech_events: bool | None = None
    receive_pre_processing_events: bool | None = None
    receive_realtime_processing_events: bool | None = None
    receive_post_processing_events: bool | None = None
    receive_acknowledgments: bool | None = None
    receive_errors: bool | None = None
    receive_lifecycle_events: bool | None = None


class GladiaInputParams(BaseModel):
    """Configuration parameters for the Gladia STT service.

    .. deprecated:: 0.0.105
        Use ``settings=GladiaSTTService.Settings(...)`` for runtime-updatable
        fields and direct init parameters for encoding/bit_depth/channels.

    Parameters:
        encoding: Audio encoding format
        bit_depth: Audio bit depth
        channels: Number of audio channels
        custom_metadata: Additional metadata to include with requests
        endpointing: Silence duration in seconds to mark end of speech
        maximum_duration_without_endpointing: Maximum utterance duration without silence
        language_config: Detailed language configuration
        pre_processing: Audio pre-processing options
        realtime_processing: Real-time processing features
        messages_config: WebSocket message filtering options
        enable_vad: Enable VAD to trigger end of utterance detection. This should be used
            without any other VAD enabled in the agent and will emit the speaker started
            and stopped frames. Defaults to False.
    """

    encoding: str | None = "wav/pcm"
    bit_depth: int | None = 16
    channels: int | None = 1
    custom_metadata: dict[str, Any] | None = None
    endpointing: float | None = None
    maximum_duration_without_endpointing: int | None = 5
    language_config: LanguageConfig | None = None
    pre_processing: PreProcessingConfig | None = None
    realtime_processing: RealtimeProcessingConfig | None = None
    messages_config: MessagesConfig | None = None
    enable_vad: bool = False
