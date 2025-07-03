#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration for the Gladia STT service."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from pipecat.transcriptions.language import Language


class LanguageConfig(BaseModel):
    """Configuration for language detection and handling.

    Parameters:
        languages: List of language codes to use for transcription
        code_switching: Whether to auto-detect language changes during transcription
    """

    languages: Optional[List[str]] = None
    code_switching: Optional[bool] = None


class PreProcessingConfig(BaseModel):
    """Configuration for audio pre-processing options.

    Parameters:
        speech_threshold: Sensitivity for speech detection (0-1)
    """

    speech_threshold: Optional[float] = None


class CustomVocabularyItem(BaseModel):
    """Represents a custom vocabulary item with an intensity value.

    Parameters:
        value: The vocabulary word or phrase
        intensity: The bias intensity for this vocabulary item (0-1)
    """

    value: str
    intensity: float


class CustomVocabularyConfig(BaseModel):
    """Configuration for custom vocabulary.

    Parameters:
        vocabulary: List of words/phrases or CustomVocabularyItem objects
        default_intensity: Default intensity for simple string vocabulary items
    """

    vocabulary: Optional[List[Union[str, CustomVocabularyItem]]] = None
    default_intensity: Optional[float] = None


class CustomSpellingConfig(BaseModel):
    """Configuration for custom spelling rules.

    Parameters:
        spelling_dictionary: Mapping of correct spellings to phonetic variations
    """

    spelling_dictionary: Optional[Dict[str, List[str]]] = None


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

    target_languages: Optional[List[str]] = None
    model: Optional[str] = None
    match_original_utterances: Optional[bool] = None
    lipsync: Optional[bool] = None
    context_adaptation: Optional[bool] = None
    context: Optional[str] = None
    informal: Optional[bool] = None


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

    words_accurate_timestamps: Optional[bool] = None
    custom_vocabulary: Optional[bool] = None
    custom_vocabulary_config: Optional[CustomVocabularyConfig] = None
    custom_spelling: Optional[bool] = None
    custom_spelling_config: Optional[CustomSpellingConfig] = None
    translation: Optional[bool] = None
    translation_config: Optional[TranslationConfig] = None
    named_entity_recognition: Optional[bool] = None
    sentiment_analysis: Optional[bool] = None


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

    receive_partial_transcripts: Optional[bool] = None
    receive_final_transcripts: Optional[bool] = None
    receive_speech_events: Optional[bool] = None
    receive_pre_processing_events: Optional[bool] = None
    receive_realtime_processing_events: Optional[bool] = None
    receive_post_processing_events: Optional[bool] = None
    receive_acknowledgments: Optional[bool] = None
    receive_errors: Optional[bool] = None
    receive_lifecycle_events: Optional[bool] = None


class GladiaInputParams(BaseModel):
    """Configuration parameters for the Gladia STT service.

    Parameters:
        encoding: Audio encoding format
        bit_depth: Audio bit depth
        channels: Number of audio channels
        custom_metadata: Additional metadata to include with requests
        endpointing: Silence duration in seconds to mark end of speech
        maximum_duration_without_endpointing: Maximum utterance duration without silence
        language: Language code for transcription

            .. deprecated:: 0.0.62
                The 'language' parameter is deprecated and will be removed in a future version.
                Use 'language_config' instead.

        language_config: Detailed language configuration
        pre_processing: Audio pre-processing options
        realtime_processing: Real-time processing features
        messages_config: WebSocket message filtering options
    """

    encoding: Optional[str] = "wav/pcm"
    bit_depth: Optional[int] = 16
    channels: Optional[int] = 1
    custom_metadata: Optional[Dict[str, Any]] = None
    endpointing: Optional[float] = None
    maximum_duration_without_endpointing: Optional[int] = 10
    language: Optional[Language] = None  # Deprecated
    language_config: Optional[LanguageConfig] = None
    pre_processing: Optional[PreProcessingConfig] = None
    realtime_processing: Optional[RealtimeProcessingConfig] = None
    messages_config: Optional[MessagesConfig] = None
