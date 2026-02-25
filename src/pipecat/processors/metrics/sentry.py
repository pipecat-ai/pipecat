#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sentry integration for frame processor metrics."""

import asyncio
from typing import Optional

from loguru import logger

from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    import sentry_sdk
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sentry, you need to `pip install pipecat-ai[sentry]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class SentryMetrics(FrameProcessorMetrics):
    """Frame processor metrics integration with Sentry monitoring.

    Extends FrameProcessorMetrics to send time-to-first-byte (TTFB) and
    processing metrics as Sentry transactions for performance monitoring
    and debugging.
    """

    def __init__(self):
        """Initialize the Sentry metrics collector.

        Sets up internal state for tracking transactions and verifies
        Sentry SDK initialization status.
        """
        super().__init__()
        self._ttfb_metrics_tx = None
        self._processing_metrics_tx = None
        self._sentry_available = sentry_sdk.is_initialized()
        if not self._sentry_available:
            logger.warning("Sentry SDK not initialized. Sentry features will be disabled.")
        self._sentry_task = None

    async def setup(self, task_manager: BaseTaskManager):
        """Setup the Sentry metrics system.

        Args:
            task_manager: The task manager to use for background operations.
        """
        await super().setup(task_manager)
        if self._sentry_available:
            self._sentry_queue = asyncio.Queue()
            self._sentry_task = self.task_manager.create_task(
                self._sentry_task_handler(), name=f"{self}::_sentry_task_handler"
            )

    async def cleanup(self):
        """Clean up Sentry resources and flush pending transactions.

        Ensures all pending transactions are sent to Sentry before shutdown.
        """
        await super().cleanup()
        if self._sentry_task:
            await self._sentry_queue.put(None)
            await self._sentry_task
            self._sentry_task = None
            logger.trace(f"{self} Flushing Sentry metrics")
            sentry_sdk.flush(timeout=5.0)

    async def start_ttfb_metrics(
        self, *, start_time: Optional[float] = None, report_only_initial_ttfb: bool
    ):
        """Start tracking time-to-first-byte metrics.

        Args:
            start_time: Optional start timestamp override.
            report_only_initial_ttfb: Whether to report only the initial TTFB measurement.
        """
        await super().start_ttfb_metrics(
            start_time=start_time, report_only_initial_ttfb=report_only_initial_ttfb
        )

        if self._should_report_ttfb and self._sentry_available:
            self._ttfb_metrics_tx = sentry_sdk.start_transaction(
                op="ttfb",
                name=f"TTFB for {self._processor_name()}",
            )
            logger.debug(
                f"{self} Sentry transaction started (ID: {self._ttfb_metrics_tx.span_id} Name: {self._ttfb_metrics_tx.name})"
            )

    async def stop_ttfb_metrics(self, *, end_time: Optional[float] = None):
        """Stop tracking time-to-first-byte metrics.

        Args:
            end_time: Optional end timestamp override.
        """
        await super().stop_ttfb_metrics(end_time=end_time)

        if self._sentry_available and self._ttfb_metrics_tx:
            await self._sentry_queue.put(self._ttfb_metrics_tx)
            self._ttfb_metrics_tx = None

    async def start_processing_metrics(self, *, start_time: Optional[float] = None):
        """Start tracking frame processing metrics.

        Args:
            start_time: Optional start timestamp override.
        """
        await super().start_processing_metrics(start_time=start_time)

        if self._sentry_available:
            self._processing_metrics_tx = sentry_sdk.start_transaction(
                op="processing",
                name=f"Processing for {self._processor_name()}",
            )
            logger.debug(
                f"{self} Sentry transaction started (ID: {self._processing_metrics_tx.span_id} Name: {self._processing_metrics_tx.name})"
            )

    async def stop_processing_metrics(self, *, end_time: Optional[float] = None):
        """Stop tracking frame processing metrics.

        Args:
            end_time: Optional end timestamp override.
        """
        await super().stop_processing_metrics(end_time=end_time)

        if self._sentry_available and self._processing_metrics_tx:
            await self._sentry_queue.put(self._processing_metrics_tx)
            self._processing_metrics_tx = None

    async def _sentry_task_handler(self):
        """Background task handler for completing Sentry transactions."""
        running = True
        while running:
            tx = await self._sentry_queue.get()
            if tx:
                await self.task_manager.get_event_loop().run_in_executor(None, tx.finish)
            running = tx is not None
            self._sentry_queue.task_done()
