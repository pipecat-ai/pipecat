#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest import mock

from pipecat.metrics.metrics import ProcessingMetricsData, TextAggregationMetricsData
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class TestProcessingMetrics(unittest.IsolatedAsyncioTestCase):
    """Processing duration intervals, same bounds as the other metrics."""

    async def asyncSetUp(self):
        self.metrics = FrameProcessorMetrics()
        self.metrics.set_processor_name("TestService")

    async def test_a_normal_interval_is_reported(self):
        await self.metrics.start_processing_metrics(start_time=1000.0)
        frame = await self.metrics.stop_processing_metrics(end_time=1003.5)

        assert frame is not None
        (data,) = frame.data
        assert isinstance(data, ProcessingMetricsData)
        self.assertEqual(data.value, 3.5)

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

        assert frame is not None
        (data,) = frame.data
        assert isinstance(data, ProcessingMetricsData)
        self.assertEqual(data.value, 2.0)

    async def test_a_normal_text_aggregation_interval_is_reported(self):
        with mock.patch(
            "pipecat.processors.metrics.frame_processor_metrics.time.time",
            side_effect=[1000.0, 1000.4, 1001.0],
        ):
            await self.metrics.start_text_aggregation_metrics()
            frame = await self.metrics.stop_text_aggregation_metrics()

            # A second stop without restart reports nothing: the normal path
            # also resets after emitting.
            self.assertIsNone(await self.metrics.stop_text_aggregation_metrics())

        assert frame is not None
        (data,) = frame.data
        assert isinstance(data, TextAggregationMetricsData)
        self.assertAlmostEqual(data.value, 0.4)

    async def test_a_negative_text_aggregation_interval_is_not_reported(self):
        with mock.patch(
            "pipecat.processors.metrics.frame_processor_metrics.time.time",
            side_effect=[1000.0, 999.5],
        ):
            await self.metrics.start_text_aggregation_metrics()
            frame = await self.metrics.stop_text_aggregation_metrics()

        self.assertIsNone(frame)

    async def test_a_rejected_text_aggregation_resets(self):
        with mock.patch(
            "pipecat.processors.metrics.frame_processor_metrics.time.time",
            side_effect=[1000.0, 999.0, 2000.0, 2002.0],
        ):
            await self.metrics.start_text_aggregation_metrics()
            self.assertIsNone(await self.metrics.stop_text_aggregation_metrics())

            # Must already be ended: another stop without restart must not
            # surface the stale start as a positive interval.
            self.assertIsNone(await self.metrics.stop_text_aggregation_metrics())

            await self.metrics.start_text_aggregation_metrics()
            frame = await self.metrics.stop_text_aggregation_metrics()

        assert frame is not None
        (data,) = frame.data
        assert isinstance(data, TextAggregationMetricsData)
        self.assertAlmostEqual(data.value, 2.0)
