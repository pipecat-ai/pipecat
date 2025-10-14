#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration for the AWS Transcribe STT service."""

from typing import List, Optional

from pydantic import BaseModel

from pipecat.transcriptions.language import Language


class AWSInputParams(BaseModel):
    """Configuration parameters for the AWS Transcribe STT service.

    Parameters:
        sample_rate: Audio sample rate in Hz. Must be 8000 or 16000. Defaults to 16000.
        language_code: Language for transcription. Cannot be used with identify_multiple_languages.
        language_options: List of languages for multi-language identification. Required when identify_multiple_languages is True.
        identify_multiple_languages: Enable multiple language identification. Defaults to False.
        identify_language: Enable language identification. Defaults to False.
        preferred_language: Preferred language from language_options to speed up identification.
        enable_partial_results_stabilization: Enable stabilization of partial results. Defaults to True.
        partial_results_stability: Stability level: "low" (faster, less stable) or "high" (slower, more stable). Defaults to "low".
        media_encoding: Audio encoding format. Defaults to "linear16".
        number_of_channels: Number of audio channels. Defaults to 1.
        show_speaker_label: Enable speaker identification in transcription results. Defaults to False.
        enable_channel_identification: Enable channel identification for multi-channel audio. Defaults to False.
        vocabulary_name: Name of custom vocabulary to use for improved transcription accuracy.
        vocabulary_filter_name: Name of vocabulary filter to apply for content filtering.

    Note:
        For real-time conversations, use partial_results_stability="low" for faster responses.
        Multi-language identification may have higher latency than single language mode.
    """

    sample_rate: int = 16000
    language_code: Optional[Language] = None
    language_options: Optional[List[Language]] = None
    identify_multiple_languages: bool = False
    identify_language: bool = False
    preferred_language: Optional[Language] = None
    enable_partial_results_stabilization: bool = True
    partial_results_stability: str = "high"
    media_encoding: str = "linear16"
    number_of_channels: int = 1
    show_speaker_label: bool = False
    enable_channel_identification: bool = False
    vocabulary_name: Optional[str] = None
    vocabulary_filter_name: Optional[str] = None
