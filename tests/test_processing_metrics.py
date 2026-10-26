#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class TestProcessingMetrics(unittest.IsolatedAsyncioTestCase):
    """Processing duration intervals, same bounds as the other metrics."""

    async def asyncSetUp(self):
        self.metrics = FrameProcessorMetrics()
        self.metrics.set_processor_name("TestService")

    async def test_a_normal_interval_is_reported(self):
        await self.metrics.start_processing_metrics(start_time=1000.0)
        frame = await self.metrics.stop_processing_metrics(end_time=1003.5)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.data[0].value, 3.5)

    async def test_an_interval_that_ends_before_it_started_is_not_reported(self):
        """Mirrors stop_ttfb_metrics: a stale end_time must not surface as a
        negative duration."""
        await self.metrics.start_processing_metrics(start_time=1000.0)
        frame = await self.metrics.stop_processing_metrics(end_time=500.0)

        self.assertIsNone(frame)

    async def test_a_rejected_interval_does_not_block_the_next_one(self):
        """The rejected measurement resets, so a second stop before any new
        start reports nothing, and the next interval reports."""
        await self.metrics.start_processing_metrics(start_time=1000.0)
        self.assertIsNone(await self.metrics.stop_processing_metrics(end_time=500.0))

        # Must already be ended: another stop without restart must not surface
        # the stale start_time as a positive interval.
        self.assertIsNone(await self.metrics.stop_processing_metrics(end_time=1001.0))

        await self.metrics.start_processing_metrics(start_time=2000.0)
        frame = await self.metrics.stop_processing_metrics(end_time=2002.0)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.data[0].value, 2.0)
