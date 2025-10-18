#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Metrics data models for Pipecat framework.

This module defines Pydantic models for various types of metrics data
collected throughout the pipeline, including timing, token usage, and
processing statistics.
"""

from typing import Optional

from pydantic import BaseModel


class MetricsData(BaseModel):
    """Base class for all metrics data.

    Parameters:
        processor: Name of the processor generating the metrics.
        model: Optional model name associated with the metrics.
    """

    processor: str
    model: Optional[str] = None


class TTFBMetricsData(MetricsData):
    """Time To First Byte (TTFB) metrics data.

    Parameters:
        value: TTFB measurement in seconds.
    """

    value: float


class ProcessingMetricsData(MetricsData):
    """General processing time metrics data.

    Parameters:
        value: Processing time measurement in seconds.
    """

    value: float


class LLMTokenUsage(BaseModel):
    """Token usage statistics for LLM operations.

    Parameters:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Total number of tokens used (prompt + completion).
        cache_read_input_tokens: Number of tokens read from cache, if applicable.
        cache_creation_input_tokens: Number of tokens used to create cache entries, if applicable.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class LLMUsageMetricsData(MetricsData):
    """LLM token usage metrics data.

    Parameters:
        value: Token usage statistics for the LLM operation.
    """

    value: LLMTokenUsage


class TTSUsageMetricsData(MetricsData):
    """Text-to-Speech usage metrics data.

    Parameters:
        value: Number of characters processed by TTS.
    """

    value: int


class STTUsage(BaseModel):
    """Audio usage statistics for STT operations.

    Parameters:
        audio_duration_seconds: Duration of audio processed in seconds.
        requests: Number of STT requests made.

        # Content metrics (similar to TTS character counting)
        word_count: Number of words transcribed.
        character_count: Number of characters transcribed.

        # Performance metrics
        processing_time_seconds: Total processing time in seconds.
        real_time_factor: Processing time / audio duration (< 1.0 is faster than real-time).
        words_per_second: Words transcribed per second (throughput).
        time_to_first_transcript: Time from audio start to first transcription (like TTFT in LLMs).
        time_to_final_transcript: Time from audio start to final transcription.

        # Quality metrics
        average_confidence: Average confidence score (0.0 to 1.0).
        word_error_rate: Word Error Rate percentage (if ground truth available).
        proper_noun_accuracy: Proper noun transcription accuracy percentage.

        # Audio metadata
        sample_rate: Audio sample rate in Hz (e.g., 16000).
        channels: Number of audio channels (1 for mono, 2 for stereo).
        encoding: Audio encoding format (e.g., "LINEAR16", "OPUS").

        # Cost tracking
        cost_per_word: Cost per word transcribed.
        estimated_cost: Estimated total cost for this transcription.

    Calculation Examples:
        # Words Per Second (WPS)
        words_per_second = word_count / processing_time_seconds

        # Real-Time Factor (RTF)
        real_time_factor = processing_time_seconds / audio_duration_seconds
        # RTF < 1.0 means faster than real-time (good!)
        # RTF = 0.5 means processing took half the audio duration

        # Word Error Rate (WER) - requires ground truth
        wer = (substitutions + insertions + deletions) / total_reference_words * 100

        # Cost Per Word
        cost_per_word = estimated_cost / word_count

        # Time to First Transcript (TTFT)
        ttft = timestamp_first_transcript - audio_start_time
    """

    audio_duration_seconds: float
    requests: int = 1

    # Content metrics
    word_count: Optional[int] = None
    character_count: Optional[int] = None

    # Performance metrics
    processing_time_seconds: Optional[float] = None
    real_time_factor: Optional[float] = None  # processing_time / audio_duration
    words_per_second: Optional[float] = None  # word_count / processing_time
    time_to_first_transcript: Optional[float] = None  # TTFT in seconds
    time_to_final_transcript: Optional[float] = None  # Total latency

    # Quality metrics
    average_confidence: Optional[float] = None  # 0.0 to 1.0
    word_error_rate: Optional[float] = None  # WER percentage
    proper_noun_accuracy: Optional[float] = None  # Proper noun accuracy percentage

    # Audio metadata
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    encoding: Optional[str] = None

    # Cost tracking
    cost_per_word: Optional[float] = None
    estimated_cost: Optional[float] = None


class STTUsageMetricsData(MetricsData):
    """Speech-to-Text usage metrics data.

    Parameters:
        value: Audio duration and request statistics for the STT operation.
    """

    value: STTUsage


class SmartTurnMetricsData(MetricsData):
    """Metrics data for smart turn predictions.

    Parameters:
        is_complete: Whether the turn is predicted to be complete.
        probability: Confidence probability of the turn completion prediction.
        inference_time_ms: Time taken for inference in milliseconds.
        server_total_time_ms: Total server processing time in milliseconds.
        e2e_processing_time_ms: End-to-end processing time in milliseconds.
    """

    is_complete: bool
    probability: float
    inference_time_ms: float
    server_total_time_ms: float
    e2e_processing_time_ms: float
