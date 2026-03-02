#
# Copyright (c) 2024-2026, Daily
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

    .. deprecated:: 0.0.104
        Processing metrics are deprecated and will be removed in a future version.
        Use TTFB metrics instead.

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


class TextAggregationMetricsData(MetricsData):
    """Text aggregation time metrics data.

    Measures the time from the first LLM token to the first complete sentence,
    representing the latency cost of sentence aggregation in the TTS pipeline.

    Parameters:
        value: Aggregation time in seconds.
    """

    value: float


class TurnMetricsData(MetricsData):
    """Metrics data for turn detection predictions.

    Parameters:
        is_complete: Whether the turn is predicted to be complete.
        probability: Confidence probability of the turn completion prediction.
        e2e_processing_time_ms: End-to-end processing time in milliseconds,
            measured from VAD speech-to-silence transition to turn completion.
    """

    is_complete: bool
    probability: float
    e2e_processing_time_ms: float


class SmartTurnMetricsData(TurnMetricsData):
    """Metrics data for smart turn predictions.

    .. deprecated:: 0.0.104
        Use :class:`TurnMetricsData` instead. This class will be removed in a future version.

    Parameters:
        inference_time_ms: Time taken for inference in milliseconds.
        server_total_time_ms: Total server processing time in milliseconds.
    """

    inference_time_ms: float = 0.0
    server_total_time_ms: float = 0.0
