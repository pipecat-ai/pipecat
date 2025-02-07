#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

try:
    import sentry_sdk
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sentry, you need to `pip install pipecat-ai[sentry]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class SentryMetrics(FrameProcessorMetrics):
    def __init__(self):
        super().__init__()
        self._ttfb_metrics_tx = None
        self._processing_metrics_tx = None
        self._sentry_available = sentry_sdk.is_initialized()
        if not self._sentry_available:
            logger.warning("Sentry SDK not initialized. Sentry features will be disabled.")

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        await super().start_ttfb_metrics(report_only_initial_ttfb)

        if self._should_report_ttfb and self._sentry_available:
            self._ttfb_metrics_tx = sentry_sdk.start_transaction(
                op="ttfb",
                name=f"TTFB for {self._processor_name()}",
            )
            logger.debug(
                f"Sentry transaction started (ID: {self._ttfb_metrics_tx.span_id} Name: {self._ttfb_metrics_tx.name})"
            )

    async def stop_ttfb_metrics(self):
        await super().stop_ttfb_metrics()

        if self._sentry_available and self._ttfb_metrics_tx:
            self._ttfb_metrics_tx.finish()

    async def start_processing_metrics(self):
        await super().start_processing_metrics()

        if self._sentry_available:
            self._processing_metrics_tx = sentry_sdk.start_transaction(
                op="processing",
                name=f"Processing for {self._processor_name()}",
            )
            logger.debug(
                f"Sentry transaction started (ID: {self._processing_metrics_tx.span_id} Name: {self._processing_metrics_tx.name})"
            )

    async def stop_processing_metrics(self):
        await super().stop_processing_metrics()

        if self._sentry_available and self._processing_metrics_tx:
            self._processing_metrics_tx.finish()
