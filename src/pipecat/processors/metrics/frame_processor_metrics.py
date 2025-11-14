#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor metrics collection and reporting."""

import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class FrameProcessorMetrics(BaseObject):
    """Metrics collection and reporting for frame processors.

    Provides comprehensive metrics tracking for frame processing operations,
    including timing measurements, resource usage, and performance analytics.
    Supports TTFB tracking, processing duration metrics, and usage statistics
    for LLM and TTS operations.
    """

    def __init__(self):
        """Initialize the frame processor metrics collector.

        Sets up internal state for tracking various metrics including TTFB,
        processing times, and usage statistics.
        """
        super().__init__()
        self._task_manager = None
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._last_ttfb_time = 0
        self._last_processing_time = 0
        self._should_report_ttfb = True

    async def setup(self, task_manager: BaseTaskManager):
        """Set up the metrics collector with a task manager.

        Args:
            task_manager: The task manager for handling async operations.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Clean up metrics collection resources."""
        await super().cleanup()

    @property
    def task_manager(self) -> BaseTaskManager:
        """Get the associated task manager.

        Returns:
            The task manager instance for async operations.
        """
        return self._task_manager

    @property
    def ttfb(self) -> Optional[float]:
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

    @property
    def processing_time(self) -> Optional[float]:
        """Get the current processing time value in seconds.

        Returns:
            The processing time value in seconds, or None if not measured.
        """
        if self._last_processing_time > 0:
            return self._last_processing_time

        # If processing is in progress, calculate current value
        if self._start_processing_time > 0:
            return time.time() - self._start_processing_time

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

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        """Start measuring time-to-first-byte (TTFB).

        Args:
            report_only_initial_ttfb: Whether to report only the first TTFB measurement.
        """
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            self._last_ttfb_time = 0
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
        """Stop TTFB measurement and generate metrics frame.

        Returns:
            MetricsFrame containing TTFB data, or None if not measuring.
        """
        if self._start_ttfb_time == 0:
            return None

        self._last_ttfb_time = time.time() - self._start_ttfb_time
        logger.debug(f"{self._processor_name()} TTFB: {self._last_ttfb_time}")
        ttfb = TTFBMetricsData(
            processor=self._processor_name(), value=self._last_ttfb_time, model=self._model_name()
        )
        self._start_ttfb_time = 0
        return MetricsFrame(data=[ttfb])

    async def start_processing_metrics(self):
        """Start measuring processing time."""
        self._start_processing_time = time.time()

    async def stop_processing_metrics(self):
        """Stop processing time measurement and generate metrics frame.

        Returns:
            MetricsFrame containing processing duration data, or None if not measuring.
        """
        if self._start_processing_time == 0:
            return None

        value = time.time() - self._start_processing_time
        self._last_processing_time = value
        logger.debug(f"{self._processor_name()} processing time: {value}")
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
        logger.debug(logstr)
        value = LLMUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=tokens
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

    async def start_stt_usage_metrics(
        self,
        audio_duration: float,
        transcript: Optional[str] = None,
        processing_time: Optional[float] = None,
        confidence: Optional[float] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        encoding: Optional[str] = None,
        cost_per_minute: Optional[float] = None,
        ttft: Optional[float] = None,
        ground_truth: Optional[str] = None,
    ):
        """Record enhanced STT usage metrics with automatic calculations.

        Args:
            audio_duration: Duration of audio processed in seconds (required).
            transcript: The transcribed text (used to calculate word/character counts).
            processing_time: Time taken to process the audio in seconds.
            confidence: Average confidence score from 0.0 to 1.0.
            sample_rate: Audio sample rate in Hz (e.g., 16000).
            channels: Number of audio channels (1 for mono, 2 for stereo).
            encoding: Audio encoding format (e.g., "LINEAR16", "OPUS").
            cost_per_minute: Cost per minute of audio (for cost estimation).
            ttft: Time to first transcript in seconds.
            ground_truth: Reference transcript for WER calculation (optional, for testing).

        Returns:
            MetricsFrame containing comprehensive STT usage data.

        Example:
            # Basic usage (backward compatible)
            await self.start_stt_usage_metrics(audio_duration=5.5)

            # Enhanced usage with all metrics
            await self.start_stt_usage_metrics(
                audio_duration=5.5,
                transcript="Hello world this is a test",
                processing_time=2.3,
                confidence=0.95,
                sample_rate=16000,
                cost_per_minute=0.006,
                ttft=0.5
            )
        """
        # Calculate content metrics from transcript
        word_count = None
        character_count = None
        if transcript:
            word_count = len(transcript.split())
            character_count = len(transcript)

        # Calculate performance metrics
        real_time_factor = None
        words_per_second = None
        if processing_time and audio_duration > 0:
            # RTF = processing_time / audio_duration
            # RTF < 1.0 means faster than real-time (good!)
            real_time_factor = processing_time / audio_duration

        if word_count and processing_time and processing_time > 0:
            # WPS = total_words / processing_time
            words_per_second = word_count / processing_time

        # Calculate cost metrics
        estimated_cost = None
        cost_per_word = None
        if cost_per_minute and audio_duration > 0:
            # Convert audio duration to minutes and calculate cost
            audio_minutes = audio_duration / 60.0
            estimated_cost = audio_minutes * cost_per_minute

            if word_count and word_count > 0:
                cost_per_word = estimated_cost / word_count

        # Calculate WER if ground truth is provided
        word_error_rate = None
        if ground_truth and transcript:
            word_error_rate = self._calculate_wer(transcript, ground_truth)

        # Build usage metrics
        usage = STTUsage(
            audio_duration_seconds=audio_duration,
            requests=1,
            word_count=word_count,
            character_count=character_count,
            processing_time_seconds=processing_time,
            real_time_factor=real_time_factor,
            words_per_second=words_per_second,
            time_to_first_transcript=ttft,
            time_to_final_transcript=processing_time,
            average_confidence=confidence,
            word_error_rate=word_error_rate,
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
            cost_per_word=cost_per_word,
            estimated_cost=estimated_cost,
        )

        value = STTUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=usage
        )

        # Build comprehensive log message
        log_parts = [f"{self._processor_name()} STT usage:"]
        log_parts.append(f"{audio_duration:.3f}s audio")
        if word_count:
            log_parts.append(f"{word_count} words")
        if words_per_second:
            log_parts.append(f"{words_per_second:.1f} WPS")
        if real_time_factor:
            log_parts.append(f"RTF={real_time_factor:.2f}")
        if estimated_cost:
            log_parts.append(f"${estimated_cost:.4f}")

        logger.debug(", ".join(log_parts))

        return MetricsFrame(data=[value])

    def _calculate_wer(self, hypothesis: str, reference: str) -> float:
        """Calculate Word Error Rate (WER) between hypothesis and reference.

        Args:
            hypothesis: The transcribed text.
            reference: The ground truth text.

        Returns:
            WER as a percentage (0-100).

        Formula:
            WER = (Substitutions + Insertions + Deletions) / Total_Reference_Words * 100
        """
        # Split into words
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        # Create matrix for dynamic programming
        d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]

        # Initialize first row and column
        for i in range(len(hyp_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j

        # Calculate edit distance
        for i in range(1, len(hyp_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if hyp_words[i - 1] == ref_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # Calculate WER percentage
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 100.0

        wer = (d[len(hyp_words)][len(ref_words)] / len(ref_words)) * 100
        return round(wer, 2)
