#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer reporting what each service spent and consumed.

Services report metrics as they finish a piece of work, and this observer
turns each one into a record: what was measured, which processor and model
reported it, and when. Nothing is summed, so a consumer groups the records by
turn, session or model as it needs, and a session that ends abruptly still
leaves behind everything that happened before it did.
"""

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    MetricsData,
    STTUsageMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class ServiceLatencyKind(StrEnum):
    """Which measurement of a service's own time a record carries."""

    TTFB = "ttfb"
    TTFA = "ttfa"
    TTFAT = "ttfat"


class ServiceUsageKind(StrEnum):
    """Which kind of service consumed something."""

    STT = "stt"
    LLM = "llm"
    TTS = "tts"


class ServiceLatencyRecord(BaseModel):
    """One measurement of how long a service took.

    Parameters:
        kind: Which wait was measured.
        processor: Name of the processor that reported it.
        model: Model the processor was using, where it names one.
        timestamp: Unix timestamp when the measurement was observed.
        seconds: The measurement itself.
        ttfb_secs: The time to first byte the measurement builds on, for the
            kinds that report one.
        leading_silence_secs: Silence at the head of the first audio, for
            time to first audio.
        thinking_time_secs: Time between a model's first output and its first
            answer token, for time to first answer token.
    """

    kind: ServiceLatencyKind
    processor: str
    model: str | None = None
    timestamp: float
    seconds: float
    ttfb_secs: float | None = None
    leading_silence_secs: float | None = None
    thinking_time_secs: float | None = None


class ServiceUsageRecord(BaseModel):
    """What one service consumed doing a piece of work.

    A field is set only where the kind of service reports it, so an LLM record
    carries token counts and a text-to-speech record carries characters.

    Parameters:
        kind: Which kind of service reported.
        processor: Name of the processor that reported it.
        model: Model the processor was using, where it names one.
        timestamp: Unix timestamp when the usage was observed.
        audio_seconds: Audio transcribed, for speech-to-text.
        characters: Characters synthesised, for text-to-speech.
        prompt_tokens: Tokens in the prompt, for an LLM.
        completion_tokens: Tokens generated, for an LLM.
        total_tokens: Tokens in the prompt and the completion together.
        cache_read_input_tokens: Prompt tokens served from cache.
        cache_creation_input_tokens: Prompt tokens written to cache.
        reasoning_tokens: Tokens spent reasoning before answering.
        input_audio_tokens: Audio tokens in the prompt.
        output_audio_tokens: Audio tokens generated.
        cache_read_input_audio_tokens: Audio prompt tokens served from cache.
    """

    kind: ServiceUsageKind
    processor: str
    model: str | None = None
    timestamp: float

    audio_seconds: float | None = None
    characters: int | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    cache_read_input_audio_tokens: int | None = None


class ServiceMetricsObserver(BaseObserver):
    """Reports each metric a service publishes as its own record.

    A record arrives per piece of work rather than per turn or per session: a
    turn that runs two inferences reports two, and a consumer that wants a
    total groups them itself. Summing here would lose the grain, and a total
    held in memory is lost with the process holding it.

    What a service made someone wait for is here; what it did with its own
    time is not. Processing time, text aggregation and smart-turn predictions
    are all deliberately absent: aggregation already appears as a span in
    :class:`~pipecat.observers.user_bot_latency_observer.LatencyBreakdown`, and
    the other two describe how work was done rather than what it cost the
    person waiting.

    Events:
        on_service_latency(observer, record): Emitted for each measurement of
            a service's own time, as a :class:`ServiceLatencyRecord`.
        on_service_usage(observer, record): Emitted for each report of what a
            service consumed, as a :class:`ServiceUsageRecord`.

    Example::

        observer = ServiceMetricsObserver()

        @observer.event_handler("on_service_usage")
        async def on_service_usage(observer, record):
            logger.info(record.model_dump_json())
    """

    def __init__(self, *, time_source: Callable[[], float] = time.time, **kwargs):
        """Initialize the service metrics observer.

        Args:
            time_source: Reads the current time in seconds. Supplying one lets
                a test place records without waiting.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._now = time_source
        # Every processor that passes a frame along reports it, so a metric is
        # remembered once it has been read. Only metrics frames are kept, and a
        # call produces few enough of them for the set to stay small.
        self._reported: set[int] = set()

        self._register_event_handler("on_service_latency")
        self._register_event_handler("on_service_usage")

    async def on_push_frame(self, data: FramePushed):
        """Report the metrics carried by a frame, the first time it is seen.

        Metrics travel in one direction, so a frame is identified by its ID
        alone. A frame broadcast both ways would arrive as two frames with two
        IDs, and would be reported twice.

        Args:
            data: Frame push event containing the frame and direction.
        """
        if not isinstance(data.frame, MetricsFrame) or data.frame.id in self._reported:
            return

        self._reported.add(data.frame.id)

        for metrics in data.frame.data:
            latency = self._as_latency(metrics)
            if latency:
                await self._call_event_handler("on_service_latency", latency)
                continue
            usage = self._as_usage(metrics)
            if usage:
                await self._call_event_handler("on_service_usage", usage)

    def _as_latency(self, metrics: MetricsData) -> ServiceLatencyRecord | None:
        """Build a latency record, for the metrics that measure time spent.

        Args:
            metrics: One metric a processor reported.

        Returns:
            The record, or None if this metric measures something else.
        """
        common = {
            "processor": metrics.processor,
            "model": metrics.model,
            "timestamp": self._now(),
        }
        if isinstance(metrics, TTFAMetricsData):
            return ServiceLatencyRecord(
                kind=ServiceLatencyKind.TTFA,
                seconds=metrics.ttfa,
                ttfb_secs=metrics.ttfb,
                leading_silence_secs=metrics.leading_silence,
                **common,
            )
        elif isinstance(metrics, TTFATMetricsData):
            return ServiceLatencyRecord(
                kind=ServiceLatencyKind.TTFAT,
                seconds=metrics.ttfat,
                ttfb_secs=metrics.ttfb,
                thinking_time_secs=metrics.thinking_time,
                **common,
            )
        elif isinstance(metrics, TTFBMetricsData):
            return ServiceLatencyRecord(
                kind=ServiceLatencyKind.TTFB, seconds=metrics.value, **common
            )
        return None

    def _as_usage(self, metrics: MetricsData) -> ServiceUsageRecord | None:
        """Build a usage record, for the metrics that measure what was consumed.

        Args:
            metrics: One metric a processor reported.

        Returns:
            The record, or None if this metric measures something else.
        """
        common = {
            "processor": metrics.processor,
            "model": metrics.model,
            "timestamp": self._now(),
        }
        if isinstance(metrics, LLMUsageMetricsData):
            tokens = metrics.value
            return ServiceUsageRecord(
                kind=ServiceUsageKind.LLM,
                prompt_tokens=tokens.prompt_tokens,
                completion_tokens=tokens.completion_tokens,
                total_tokens=tokens.total_tokens,
                cache_read_input_tokens=tokens.cache_read_input_tokens,
                cache_creation_input_tokens=tokens.cache_creation_input_tokens,
                reasoning_tokens=tokens.reasoning_tokens,
                input_audio_tokens=tokens.input_audio_tokens,
                output_audio_tokens=tokens.output_audio_tokens,
                cache_read_input_audio_tokens=tokens.cache_read_input_audio_tokens,
                **common,
            )
        elif isinstance(metrics, STTUsageMetricsData):
            return ServiceUsageRecord(
                kind=ServiceUsageKind.STT, audio_seconds=metrics.value.audio_seconds, **common
            )
        elif isinstance(metrics, TTSUsageMetricsData):
            return ServiceUsageRecord(kind=ServiceUsageKind.TTS, characters=metrics.value, **common)
        return None
