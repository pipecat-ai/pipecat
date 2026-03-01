import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    MetricsFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    TextAggregationMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.tests.utils import SleepFrame, run_test


class TestUserBotLatencyObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for UserBotLatencyObserver."""

    async def test_normal_latency_measurement(self):
        """Test basic latency measurement from user stop to bot start."""
        # Create observer
        observer = UserBotLatencyObserver()

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Capture latency events
        latencies = []

        @observer.event_handler("on_latency_measured")
        async def on_latency(obs, latency_seconds):
            latencies.append(latency_seconds)

        # Define frame sequence
        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        # Run test
        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        # Verify latency was measured
        self.assertEqual(len(latencies), 1)
        self.assertGreater(latencies[0], 0)
        self.assertLess(latencies[0], 1.0)  # Should be very quick

    async def test_multiple_latency_measurements(self):
        """Test that multiple user-bot exchanges produce separate latency events."""
        # Create observer
        observer = UserBotLatencyObserver()

        # Create identity filter
        processor = IdentityFilter()

        # Capture latency events
        latencies = []

        @observer.event_handler("on_latency_measured")
        async def on_latency(obs, latency_seconds):
            latencies.append(latency_seconds)

        # Define frame sequence with two complete cycles
        frames_to_send = [
            # First cycle
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            # Second cycle
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        # Run test
        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        # Verify two separate latencies were measured
        self.assertEqual(len(latencies), 2)
        self.assertGreater(latencies[0], 0)
        self.assertGreater(latencies[1], 0)

    async def test_breakdown_with_metrics(self):
        """Test that metrics collected between VADUserStopped and BotStarted appear in breakdown."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        stt_ttfb = TTFBMetricsData(processor="DeepgramSTTService#0", value=0.080)
        llm_ttfb = TTFBMetricsData(processor="OpenAILLMService#0", model="gpt-4o", value=0.250)
        tts_ttfb = TTFBMetricsData(processor="CartesiaTTSService#0", value=0.070)
        text_agg = TextAggregationMetricsData(processor="CartesiaTTSService#0", value=0.030)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            MetricsFrame(data=[stt_ttfb]),
            MetricsFrame(data=[llm_ttfb, text_agg]),
            MetricsFrame(data=[tts_ttfb]),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            MetricsFrame,
            MetricsFrame,
            MetricsFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        bd = breakdowns[0]
        self.assertEqual(len(bd.ttfb), 3)
        self.assertEqual(bd.ttfb[0].processor, "DeepgramSTTService#0")
        self.assertEqual(bd.ttfb[1].processor, "OpenAILLMService#0")
        self.assertEqual(bd.ttfb[2].processor, "CartesiaTTSService#0")
        self.assertIsNotNone(bd.text_aggregation)
        self.assertEqual(bd.text_aggregation.value, 0.030)

    async def test_interruption_resets_accumulators(self):
        """Test that InterruptionFrame clears stale metrics from earlier cycles."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        # First cycle metrics (will be interrupted)
        stale_llm = TTFBMetricsData(processor="OpenAILLMService#0", value=0.245)
        # Second cycle metrics (the ones that matter)
        final_llm = TTFBMetricsData(processor="OpenAILLMService#0", value=0.224)
        final_tts = TTFBMetricsData(processor="CartesiaTTSService#0", value=0.142)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            MetricsFrame(data=[stale_llm]),
            InterruptionFrame(),
            MetricsFrame(data=[final_llm]),
            MetricsFrame(data=[final_tts]),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            MetricsFrame,
            InterruptionFrame,
            MetricsFrame,
            MetricsFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        bd = breakdowns[0]
        # Only the post-interruption metrics should be present
        self.assertEqual(len(bd.ttfb), 2)
        self.assertEqual(bd.ttfb[0].processor, "OpenAILLMService#0")
        self.assertEqual(bd.ttfb[0].value, 0.224)
        self.assertEqual(bd.ttfb[1].processor, "CartesiaTTSService#0")
        self.assertEqual(bd.ttfb[1].value, 0.142)

    async def test_only_first_text_aggregation_kept(self):
        """Test that only the first text aggregation metric is kept per cycle."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        text_agg_1 = TextAggregationMetricsData(processor="CartesiaTTSService#0", value=0.030)
        text_agg_2 = TextAggregationMetricsData(processor="CartesiaTTSService#0", value=0.080)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            MetricsFrame(data=[text_agg_1]),
            MetricsFrame(data=[text_agg_2]),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            MetricsFrame,
            MetricsFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        self.assertIsNotNone(breakdowns[0].text_aggregation)
        self.assertEqual(breakdowns[0].text_aggregation.value, 0.030)

    async def test_user_turn_measured(self):
        """Test that pre-LLM wait from user silence to UserStopped is captured."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),  # Simulate turn analyzer wait
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        self.assertIsNotNone(breakdowns[0].user_turn_secs)
        self.assertGreaterEqual(breakdowns[0].user_turn_secs, 0.1)

    async def test_user_turn_none_without_user_stopped(self):
        """Test that user_turn is None when no UserStoppedSpeakingFrame arrives."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        self.assertIsNone(breakdowns[0].user_turn_secs)

    async def test_no_measurement_without_user_stop(self):
        """Test that BotStartedSpeaking without prior user stop emits nothing."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        latencies = []
        breakdowns = []

        @observer.event_handler("on_latency_measured")
        async def on_latency(obs, latency_seconds):
            latencies.append(latency_seconds)

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        frames_to_send = [
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(latencies), 0)
        self.assertEqual(len(breakdowns), 0)


if __name__ == "__main__":
    unittest.main()
