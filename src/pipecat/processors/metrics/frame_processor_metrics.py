#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)


class FrameProcessorMetrics:
    def __init__(self):
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._last_ttfb_time = 0
        self._should_report_ttfb = True

    @property
    def ttfb(self) -> Optional[float]:
        """Get the current TTFB value in seconds.

        Returns:
            Optional[float]: The TTFB value in seconds, or None if not measured
        """
        if self._last_ttfb_time > 0:
            return self._last_ttfb_time

        # If TTFB is in progress, calculate current value
        if self._start_ttfb_time > 0:
            return time.time() - self._start_ttfb_time

        return None

    def _processor_name(self):
        return self._core_metrics_data.processor

    def _model_name(self):
        return self._core_metrics_data.model

    def set_core_metrics_data(self, data: MetricsData):
        self._core_metrics_data = data

    def set_processor_name(self, name: str):
        self._core_metrics_data = MetricsData(processor=name)

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            self._last_ttfb_time = 0
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
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
        self._start_processing_time = time.time()

    async def stop_processing_metrics(self):
        if self._start_processing_time == 0:
            return None

        value = time.time() - self._start_processing_time
        logger.debug(f"{self._processor_name()} processing time: {value}")
        processing = ProcessingMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_processing_time = 0
        return MetricsFrame(data=[processing])

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        logger.debug(
            f"{self._processor_name()} prompt tokens: {tokens.prompt_tokens}, completion tokens: {tokens.completion_tokens}"
        )
        value = LLMUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=tokens
        )
        return MetricsFrame(data=[value])

    async def start_tts_usage_metrics(self, text: str):
        characters = TTSUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=len(text)
        )
        logger.debug(f"{self._processor_name()} usage characters: {characters.value}")
        return MetricsFrame(data=[characters])
