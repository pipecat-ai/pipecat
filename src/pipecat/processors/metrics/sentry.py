#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time

from loguru import logger

try:
    import sentry_sdk

    sentry_available = sentry_sdk.is_initialized()
    if not sentry_available:
        logger.warning("Sentry SDK not initialized. Sentry features will be disabled.")
except ImportError:
    sentry_available = False
    logger.warning("Sentry SDK not installed. Sentry features will be disabled.")

from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class SentryMetrics(FrameProcessorMetrics):
    def __init__(self):
        super().__init__()
        self._ttfb_metrics_span = None
        self._processing_metrics_span = None

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            if sentry_available:
                self._ttfb_metrics_span = sentry_sdk.start_span(
                    op="ttfb",
                    description=f"TTFB for {self._processor_name()}",
                    start_timestamp=self._start_ttfb_time,
                )
                logger.debug(
                    f"Sentry Span ID: {self._ttfb_metrics_span.span_id} Description: {self._ttfb_metrics_span.description} started."
                )
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
        stop_time = time.time()
        if sentry_available:
            self._ttfb_metrics_span.finish(end_timestamp=stop_time)

    async def start_processing_metrics(self):
        self._start_processing_time = time.time()
        if sentry_available:
            self._processing_metrics_span = sentry_sdk.start_span(
                op="processing",
                description=f"Processing for {self._processor_name()}",
                start_timestamp=self._start_processing_time,
            )
            logger.debug(
                f"Sentry Span ID: {self._processing_metrics_span.span_id} Description: {self._processing_metrics_span.description} started."
            )

    async def stop_processing_metrics(self):
        stop_time = time.time()
        if sentry_available:
            self._processing_metrics_span.finish(end_timestamp=stop_time)
