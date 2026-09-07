#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import MetricsFrame, TextFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    SmartTurnMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.service_metrics_observer import (
    ServiceLatencyKind,
    ServiceMetricsObserver,
    ServiceUsageKind,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager


class TestServiceMetricsObserver(unittest.IsolatedAsyncioTestCase):
    """Each metric a service reports becomes one record."""

    async def asyncSetUp(self):
        self.clock = 1_000_000.0
        self.observer = ServiceMetricsObserver(time_source=lambda: self.clock)
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.latency = []
        self.usage = []

        @self.observer.event_handler("on_service_latency")
        async def on_latency(observer, record):
            self.latency.append(record)

        @self.observer.event_handler("on_service_usage")
        async def on_usage(observer, record):
            self.usage.append(record)

    async def _push(self, frame, source="source"):
        """Feed one frame to the observer, as a pipeline push would."""
        await self.observer.on_push_frame(
            FramePushed(
                source=IdentityFilter(name=source),
                destination=IdentityFilter(name="destination"),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        # Event handlers run as tasks, so give them a chance to deliver.
        await self._settle()

    async def _settle(self):
        import asyncio

        await asyncio.sleep(0.01)

    async def test_time_to_first_byte(self):
        """The simplest measurement, carried as seconds."""
        await self._push(
            MetricsFrame(
                data=[TTFBMetricsData(processor="OpenAILLMService#0", model="gpt-4.1", value=0.757)]
            )
        )
        record = self.latency[0]
        self.assertEqual(record.kind, ServiceLatencyKind.TTFB)
        self.assertEqual(record.processor, "OpenAILLMService#0")
        self.assertEqual(record.model, "gpt-4.1")
        self.assertEqual(record.seconds, 0.757)
        self.assertEqual(record.timestamp, self.clock)
        self.assertIsNone(record.ttfb_secs)

    async def test_time_to_first_audio_keeps_what_it_builds_on(self):
        """A measurement that decomposes reports its parts."""
        await self._push(
            MetricsFrame(
                data=[
                    TTFAMetricsData(
                        processor="CartesiaTTSService#0", ttfa=0.31, ttfb=0.13, leading_silence=0.18
                    )
                ]
            )
        )
        record = self.latency[0]
        self.assertEqual(record.kind, ServiceLatencyKind.TTFA)
        self.assertEqual(record.seconds, 0.31)
        self.assertEqual(record.ttfb_secs, 0.13)
        self.assertEqual(record.leading_silence_secs, 0.18)

    async def test_time_to_first_answer_token_keeps_the_thinking(self):
        """Thinking time is what separates the answer from the first byte."""
        await self._push(
            MetricsFrame(
                data=[TTFATMetricsData(processor="LLM#0", ttfat=1.4, ttfb=0.4, thinking_time=1.0)]
            )
        )
        record = self.latency[0]
        self.assertEqual(record.kind, ServiceLatencyKind.TTFAT)
        self.assertEqual(record.seconds, 1.4)
        self.assertEqual(record.ttfb_secs, 0.4)
        self.assertEqual(record.thinking_time_secs, 1.0)

    async def test_llm_tokens_including_the_optional_ones(self):
        """Every token count a model reports survives into the record."""
        await self._push(
            MetricsFrame(
                data=[
                    LLMUsageMetricsData(
                        processor="OpenAILLMService#0",
                        model="gpt-4.1",
                        value=LLMTokenUsage(
                            prompt_tokens=298,
                            completion_tokens=58,
                            total_tokens=356,
                            cache_read_input_tokens=128,
                            reasoning_tokens=12,
                        ),
                    )
                ]
            )
        )
        record = self.usage[0]
        self.assertEqual(record.kind, ServiceUsageKind.LLM)
        self.assertEqual(record.model, "gpt-4.1")
        self.assertEqual(record.prompt_tokens, 298)
        self.assertEqual(record.completion_tokens, 58)
        self.assertEqual(record.total_tokens, 356)
        self.assertEqual(record.cache_read_input_tokens, 128)
        self.assertEqual(record.reasoning_tokens, 12)
        # Fields for other kinds of service stay empty.
        self.assertIsNone(record.characters)
        self.assertIsNone(record.audio_seconds)

    async def test_speech_to_text_and_text_to_speech_usage(self):
        """Audio in, characters out."""
        await self._push(
            MetricsFrame(
                data=[
                    STTUsageMetricsData(
                        processor="DeepgramSTTService#0", value=STTUsage(audio_seconds=42.24)
                    )
                ]
            )
        )
        await self._push(
            MetricsFrame(data=[TTSUsageMetricsData(processor="CartesiaTTSService#0", value=87)])
        )
        self.assertEqual(self.usage[0].kind, ServiceUsageKind.STT)
        self.assertEqual(self.usage[0].audio_seconds, 42.24)
        self.assertEqual(self.usage[1].kind, ServiceUsageKind.TTS)
        self.assertEqual(self.usage[1].characters, 87)

    async def test_nothing_is_summed(self):
        """Two inferences report twice, and the records stay apart."""
        for tokens in (10, 20):
            await self._push(
                MetricsFrame(
                    data=[
                        LLMUsageMetricsData(
                            processor="LLM#0",
                            value=LLMTokenUsage(
                                prompt_tokens=tokens, completion_tokens=1, total_tokens=tokens + 1
                            ),
                        )
                    ]
                )
            )
        self.assertEqual([r.prompt_tokens for r in self.usage], [10, 20])

    async def test_a_relayed_metric_is_reported_once(self):
        """A frame passed along the pipeline is one metric, not one per hop."""
        frame = MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.2)])
        for processor in ("LLM#0", "TTS#0", "Transport#0"):
            await self._push(frame, source=processor)
        self.assertEqual(len(self.latency), 1)

    async def test_metrics_measuring_something_else_are_left_alone(self):
        """Only what a service made someone wait for is a record."""
        await self._push(
            MetricsFrame(
                data=[
                    ProcessingMetricsData(processor="LLM#0", value=1.11),
                    TextAggregationMetricsData(processor="TTS#0", value=0.226),
                    SmartTurnMetricsData(
                        processor="SmartTurn#0",
                        is_complete=True,
                        probability=0.9,
                        e2e_processing_time_ms=18.0,
                    ),
                ]
            )
        )
        self.assertEqual(self.latency, [])
        self.assertEqual(self.usage, [])

    async def test_other_frames_are_ignored(self):
        """Only metrics frames carry metrics."""
        await self._push(TextFrame("hello"))
        self.assertEqual(self.latency, [])
        self.assertEqual(self.usage, [])

    async def test_one_frame_can_carry_several_metrics(self):
        """Each becomes its own record."""
        await self._push(
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="LLM#0", value=0.3),
                    LLMUsageMetricsData(
                        processor="LLM#0",
                        value=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                    ),
                ]
            )
        )
        self.assertEqual(len(self.latency), 1)
        self.assertEqual(len(self.usage), 1)
