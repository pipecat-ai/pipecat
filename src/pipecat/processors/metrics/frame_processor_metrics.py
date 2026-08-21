#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor metrics collection and reporting."""

import time

from loguru import logger

from pipecat.audio.utils import detect_speech_onset
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.utils.base_object import BaseObject


class FrameProcessorMetrics(BaseObject):
    """Metrics collection and reporting for frame processors.

    Provides comprehensive metrics tracking for frame processing operations,
    including timing measurements, resource usage, and performance analytics.
    Supports TTFB tracking, TTFA tracking, processing duration metrics, and
    usage statistics for LLM, STT, and TTS operations.
    """

    # Cap on buffered audio while waiting for a TTFA speech onset, so a response
    # that is all silence (or never becomes audible) can't grow the buffer without
    # bound.
    _TTFA_MAX_BUFFER_SECONDS = 3.0

    def __init__(self):
        """Initialize the frame processor metrics collector.

        Sets up internal state for tracking various metrics including TTFB,
        processing times, and usage statistics.
        """
        super().__init__()
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._start_text_aggregation_time = 0
        self._last_ttfb_time = 0
        self._should_report_ttfb = True
        # TTFA (time-to-first-audio) state. TTFA extends TTFB: scanning for the
        # first audible sample begins when TTFB stops (see stop_ttfb_metrics).
        # Audio is buffered across chunks until a sustained onset is confirmed.
        self._ttfa_active = False
        self._ttfa_buffer = b""
        # TTFAT (time-to-first-answer-token) state. TTFAT extends TTFB the same
        # way, so it keeps the request start that stop_ttfb_metrics clears.
        self._ttfat_active = False
        self._start_ttfat_time = 0

    @property
    def ttfb(self) -> float | None:
        """Get the current TTFB value in seconds.

        Returns:
            The TTFB value in seconds, or None if not measured.
        """
        if self._last_ttfb_time > 0:
            return self._last_ttfb_time

        # If TTFB is in progress, calculate current value
        if self._start_ttfb_time > 0:
            return time.time() - self._start_ttfb_time

        return None

    def _processor_name(self):
        """Get the processor name from core metrics data."""
        return self._core_metrics_data.processor

    def _model_name(self):
        """Get the model name from core metrics data."""
        return self._core_metrics_data.model

    def set_core_metrics_data(self, data: MetricsData):
        """Set the core metrics data for this collector.

        Args:
            data: The core metrics data containing processor and model information.
        """
        self._core_metrics_data = data

    def set_processor_name(self, name: str):
        """Set the processor name for metrics reporting.

        Args:
            name: The name of the processor to use in metrics.
        """
        self._core_metrics_data = MetricsData(processor=name)

    async def start_ttfb_metrics(
        self, *, start_time: float | None = None, report_only_initial_ttfb: bool
    ):
        """Start measuring time-to-first-byte (TTFB).

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
            report_only_initial_ttfb: Whether to report only the first TTFB measurement.
        """
        # A response that was interrupted before its first answer token leaves
        # TTFAT armed; drop it so this request measures its own. This is TTFAT
        # state rather than TTFB reporting policy, so it happens regardless of
        # whether this request's TTFB will be reported.
        self._ttfat_active = False
        self._start_ttfat_time = 0
        if self._should_report_ttfb:
            self._start_ttfb_time = start_time or time.time()
            self._last_ttfb_time = 0
            self._should_report_ttfb = not report_only_initial_ttfb

    async def cancel_ttfb_metrics(self):
        """Abandon the current TTFB measurement without reporting it.

        For a request whose response never came: there is no interval to
        report, and leaving the measurement open would let the next unrelated
        output be measured against it.
        """
        self._start_ttfb_time = 0

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        """Stop TTFB measurement and generate metrics frame.

        TTFB ends at the first output the service produces, of any kind —
        including content a caller never sees, such as an LLM's reasoning. Events
        that merely acknowledge the request (an HTTP response head, a stream-open
        or keepalive event) carry no output and must not stop it, or TTFB
        measures connection setup rather than the service's response.

        Only the first call per measurement takes effect, so services can call
        this from every branch that handles output rather than tracking which
        arrived first.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.

        Returns:
            MetricsFrame containing TTFB data, or None if not measuring.
        """
        if self._start_ttfb_time == 0:
            return None

        end_time = end_time or time.time()

        if end_time < self._start_ttfb_time:
            # The output being measured predates the request it is measured
            # from, so it cannot be a response to it -- a caller measuring to a
            # stale timestamp, or a wall clock that stepped backwards mid
            # measurement. Either way there is no interval to report.
            logger.warning(
                f"{self._processor_name()} TTFB not reported: output predates the "
                f"start of the measurement by {self._start_ttfb_time - end_time:.3f}s"
            )
            self._start_ttfb_time = 0
            return None

        self._last_ttfb_time = end_time - self._start_ttfb_time
        logger.debug(f"{self._processor_name()} TTFB: {self._last_ttfb_time:.3f}s")
        ttfb = TTFBMetricsData(
            processor=self._processor_name(), value=self._last_ttfb_time, model=self._model_name()
        )
        # The first byte has arrived; begin scanning leading silence so TTFA can
        # be reported as TTFB plus the silence duration (see process_ttfa_metrics).
        self._ttfa_active = True
        self._ttfa_buffer = b""
        # Likewise for TTFAT, which needs the request start that is cleared below
        # (see stop_ttfat_metrics).
        self._ttfat_active = True
        self._start_ttfat_time = self._start_ttfb_time
        self._start_ttfb_time = 0
        return MetricsFrame(data=[ttfb])

    async def process_ttfa_metrics(self, *, audio: bytes, sample_rate: int, num_channels: int):
        """Scan buffered audio for speech onset and report TTFA.

        TTFA extends TTFB: once :meth:`stop_ttfb_metrics` records the first byte,
        audio is buffered across chunks (the leading silence may span several,
        and onset confirmation needs a little audio past it) and scanned for a
        sustained speech onset. When found, TTFB plus the leading-silence
        duration is emitted and measuring stops until the next response.

        Args:
            audio: Raw PCM audio bytes (16-bit signed integers).
            sample_rate: Sample rate of the audio in Hz.
            num_channels: Number of interleaved audio channels.

        Returns:
            MetricsFrame containing TTFA data once onset is confirmed, otherwise
            None.
        """
        if not self._ttfa_active or sample_rate <= 0:
            return None

        self._ttfa_buffer += audio
        onset = detect_speech_onset(self._ttfa_buffer, sample_rate, num_channels)
        if onset is None:
            # No confirmed onset yet. Bound memory against pathologically long
            # (or never-arriving) silence by giving up past a sane limit.
            max_bytes = int(self._TTFA_MAX_BUFFER_SECONDS * sample_rate * max(num_channels, 1) * 2)
            if len(self._ttfa_buffer) >= max_bytes:
                logger.debug(
                    f"{self._processor_name()} TTFA: no onset within "
                    f"{self._TTFA_MAX_BUFFER_SECONDS:.0f}s of audio; not reporting"
                )
                self._ttfa_active = False
                self._ttfa_buffer = b""
            return None

        silence_duration = onset / sample_rate
        value = self._last_ttfb_time + silence_duration
        logger.debug(
            f"{self._processor_name()} TTFA: {value:.3f}s ({silence_duration:.3f}s leading silence)"
        )
        ttfa = TTFAMetricsData(
            processor=self._processor_name(),
            ttfa=value,
            leading_silence=silence_duration,
            ttfb=self._last_ttfb_time,
            model=self._model_name(),
        )
        self._ttfa_active = False
        self._ttfa_buffer = b""
        return MetricsFrame(data=[ttfa])

    async def stop_ttfat_metrics(self, *, end_time: float | None = None):
        """Stop TTFAT measurement and generate metrics frame.

        TTFAT extends TTFB: it runs from the same request start but ends at the
        first token of the answer the caller sees, so anything the model streams
        first — reasoning, most often — falls between the two. Measuring it at
        the point a service produces the token, rather than where the pipeline
        releases it, keeps buffering downstream of the service out of the number.

        Only the first call per measurement takes effect, so services can call
        this from every branch that produces an answer — streamed text, and the
        start of a tool call on a turn that answers with one instead of text.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.

        Returns:
            MetricsFrame containing TTFAT data, or None if not measuring.
        """
        if not self._ttfat_active or self._start_ttfat_time == 0:
            return None

        end_time = end_time or time.time()

        value = end_time - self._start_ttfat_time
        thinking_time = value - self._last_ttfb_time
        logger.debug(
            f"{self._processor_name()} TTFAT: {value:.3f}s ({thinking_time:.3f}s thinking)"
        )
        ttfat = TTFATMetricsData(
            processor=self._processor_name(),
            ttfat=value,
            ttfb=self._last_ttfb_time,
            thinking_time=thinking_time,
            model=self._model_name(),
        )
        self._ttfat_active = False
        self._start_ttfat_time = 0
        return MetricsFrame(data=[ttfat])

    async def start_processing_metrics(self, *, start_time: float | None = None):
        """Start measuring processing time.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        self._start_processing_time = start_time or time.time()

    async def stop_processing_metrics(self, *, end_time: float | None = None):
        """Stop processing time measurement and generate metrics frame.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.

        Returns:
            MetricsFrame containing processing duration data, or None if not measuring.
        """
        if self._start_processing_time == 0:
            return None

        end_time = end_time or time.time()

        value = end_time - self._start_processing_time
        logger.debug(f"{self._processor_name()} processing time: {value:.3f}s")
        processing = ProcessingMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_processing_time = 0
        return MetricsFrame(data=[processing])

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Record LLM token usage metrics.

        Args:
            tokens: Token usage information including prompt and completion tokens.

        Returns:
            MetricsFrame containing LLM usage data.
        """
        logstr = f"{self._processor_name()} prompt tokens: {tokens.prompt_tokens}, completion tokens: {tokens.completion_tokens}"
        if tokens.cache_read_input_tokens:
            logstr += f", cache read input tokens: {tokens.cache_read_input_tokens}"
        if tokens.reasoning_tokens:
            logstr += f", reasoning tokens: {tokens.reasoning_tokens}"
        if tokens.input_audio_tokens:
            logstr += f", input audio tokens: {tokens.input_audio_tokens}"
        if tokens.output_audio_tokens:
            logstr += f", output audio tokens: {tokens.output_audio_tokens}"
        if tokens.cache_read_input_audio_tokens:
            logstr += f", cache read input audio tokens: {tokens.cache_read_input_audio_tokens}"
        logger.debug(logstr)
        value = LLMUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=tokens
        )
        return MetricsFrame(data=[value])

    async def start_stt_usage_metrics(self, usage: STTUsage):
        """Record STT usage metrics.

        Args:
            usage: Usage information for the STT operation.

        Returns:
            MetricsFrame containing STT usage data.
        """
        logger.debug(f"{self._processor_name()} usage audio seconds: {usage.audio_seconds:.3f}")
        value = STTUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=usage
        )
        return MetricsFrame(data=[value])

    async def start_tts_usage_metrics(self, text: str):
        """Record TTS character usage metrics.

        Args:
            text: The text being processed by TTS.

        Returns:
            MetricsFrame containing TTS usage data.
        """
        characters = TTSUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=len(text)
        )
        logger.debug(f"{self._processor_name()} usage characters: {characters.value}")
        return MetricsFrame(data=[characters])

    async def start_text_aggregation_metrics(self):
        """Start measuring text aggregation time (first token to first sentence)."""
        self._start_text_aggregation_time = time.time()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation measurement and generate metrics frame.

        Returns:
            MetricsFrame containing text aggregation time, or None if not measuring.
        """
        if self._start_text_aggregation_time == 0:
            return None

        value = time.time() - self._start_text_aggregation_time
        logger.debug(f"{self._processor_name()} text aggregation time: {value}")
        aggregation = TextAggregationMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_text_aggregation_time = 0
        return MetricsFrame(data=[aggregation])
