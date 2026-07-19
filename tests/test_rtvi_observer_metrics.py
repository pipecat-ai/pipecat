#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RTVIObserver metrics handling."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams


class TestRTVIObserverMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_ttfb_and_ttfa_forwarded_to_clients(self):
        # TTFB and TTFA are emitted in separate frames during a real TTS turn;
        # both must reach RTVI clients.
        observer = RTVIObserver(params=RTVIObserverParams(metrics_enabled=True))
        sent = []
        observer.send_rtvi_message = AsyncMock(side_effect=lambda m: sent.append(m))

        await observer._handle_metrics(
            MetricsFrame(
                data=[TTFBMetricsData(processor="cartesia_tts", value=0.2, model="sonic-english")]
            )
        )
        await observer._handle_metrics(
            MetricsFrame(
                data=[
                    TTFAMetricsData(
                        processor="cartesia_tts",
                        ttfa=0.6,
                        ttfb=0.2,
                        leading_silence=0.4,
                        model="sonic-english",
                    )
                ]
            )
        )

        self.assertEqual(len(sent), 2)
        self.assertIn("ttfb", sent[0].data)
        self.assertEqual(sent[0].data["ttfb"][0]["value"], 0.2)
        self.assertIn("ttfa", sent[1].data)
        self.assertEqual(sent[1].data["ttfa"][0]["leading_silence"], 0.4)

    async def test_all_metric_types_accumulate_into_one_message(self):
        # A single frame can carry several metric types; each must land under its
        # own key in one outgoing message. LLM usage is special-cased: the message
        # carries the nested token-usage value, not the wrapper.
        observer = RTVIObserver(params=RTVIObserverParams(metrics_enabled=True))
        sent = []
        observer.send_rtvi_message = AsyncMock(side_effect=lambda m: sent.append(m))

        await observer._handle_metrics(
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="cartesia_tts", value=0.2),
                    TTFAMetricsData(
                        processor="cartesia_tts", ttfa=0.6, ttfb=0.2, leading_silence=0.4
                    ),
                    ProcessingMetricsData(processor="llm", value=0.05),
                    LLMUsageMetricsData(
                        processor="llm",
                        value=LLMTokenUsage(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            input_audio_tokens=7,
                            output_audio_tokens=13,
                        ),
                    ),
                    TTSUsageMetricsData(processor="cartesia_tts", value=42),
                ]
            )
        )

        self.assertEqual(len(sent), 1)
        data = sent[0].data
        self.assertEqual(data["ttfb"][0]["value"], 0.2)
        self.assertEqual(data["ttfa"][0]["leading_silence"], 0.4)
        self.assertEqual(data["processing"][0]["value"], 0.05)
        # LLM usage unwraps to the token-usage payload.
        self.assertEqual(data["tokens"][0]["total_tokens"], 30)
        # Populated optional fields are included; unset ones are dropped
        # entirely (exclude_none) rather than sent as null.
        self.assertEqual(data["tokens"][0]["input_audio_tokens"], 7)
        self.assertEqual(data["tokens"][0]["output_audio_tokens"], 13)
        self.assertNotIn("cache_read_input_audio_tokens", data["tokens"][0])
        self.assertEqual(data["characters"][0]["value"], 42)


if __name__ == "__main__":
    unittest.main()
