import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ClientConnectedFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
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
from pipecat.observers.user_bot_latency_observer import (
    FunctionCallMetrics,
    LatencyBreakdown,
    TextAggregationBreakdownMetrics,
    TTFBBreakdownMetrics,
    UserBotLatencyObserver,
)
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
        self.assertEqual(bd.text_aggregation.duration_secs, 0.030)

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
        self.assertEqual(bd.ttfb[0].duration_secs, 0.224)
        self.assertEqual(bd.ttfb[1].processor, "CartesiaTTSService#0")
        self.assertEqual(bd.ttfb[1].duration_secs, 0.142)

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
        self.assertEqual(breakdowns[0].text_aggregation.duration_secs, 0.030)

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

    async def test_first_bot_speech_latency(self):
        """Test first bot speech latency and breakdown from ClientConnected to BotStartedSpeaking."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        first_speech_latencies = []
        breakdowns = []

        @observer.event_handler("on_first_bot_speech_latency")
        async def on_first_bot_speech(obs, latency_seconds):
            first_speech_latencies.append(latency_seconds)

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        llm_ttfb = TTFBMetricsData(processor="OpenAILLMService#0", value=0.250)
        tts_ttfb = TTFBMetricsData(processor="CartesiaTTSService#0", value=0.070)

        frames_to_send = [
            ClientConnectedFrame(),
            MetricsFrame(data=[llm_ttfb]),
            MetricsFrame(data=[tts_ttfb]),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            ClientConnectedFrame,
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

        self.assertEqual(len(first_speech_latencies), 1)
        self.assertGreater(first_speech_latencies[0], 0)
        self.assertLess(first_speech_latencies[0], 1.0)

        # Breakdown should also be emitted with the accumulated metrics
        self.assertEqual(len(breakdowns), 1)
        self.assertEqual(len(breakdowns[0].ttfb), 2)
        self.assertEqual(breakdowns[0].ttfb[0].processor, "OpenAILLMService#0")
        self.assertEqual(breakdowns[0].ttfb[1].processor, "CartesiaTTSService#0")

    async def test_first_bot_speech_only_once(self):
        """Test that first bot speech latency is only emitted once."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        first_speech_latencies = []

        @observer.event_handler("on_first_bot_speech_latency")
        async def on_first_bot_speech(obs, latency_seconds):
            first_speech_latencies.append(latency_seconds)

        frames_to_send = [
            ClientConnectedFrame(),
            BotStartedSpeakingFrame(),
            # Second bot speech should not trigger the event again
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            ClientConnectedFrame,
            BotStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(first_speech_latencies), 1)

    async def test_first_bot_speech_skipped_when_user_speaks_first(self):
        """Test that first bot speech event is not emitted when user speaks before the bot."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        first_speech_latencies = []

        @observer.event_handler("on_first_bot_speech_latency")
        async def on_first_bot_speech(obs, latency_seconds):
            first_speech_latencies.append(latency_seconds)

        frames_to_send = [
            ClientConnectedFrame(),
            # User speaks before bot has a chance to greet
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            ClientConnectedFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        self.assertEqual(len(first_speech_latencies), 0)

    async def test_function_call_latency_in_breakdown(self):
        """Test that function call duration appears in the latency breakdown."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        tool_call_id = "call_abc123"

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id=tool_call_id,
                arguments={"location": "Atlanta"},
            ),
            SleepFrame(sleep=0.1),
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id=tool_call_id,
                arguments={"location": "Atlanta"},
                result={"temperature": "75"},
            ),
            BotStartedSpeakingFrame(),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        self.assertEqual(len(breakdowns[0].function_calls), 1)
        fc = breakdowns[0].function_calls[0]
        self.assertEqual(fc.function_name, "get_weather")
        self.assertGreaterEqual(fc.duration_secs, 0.1)

    async def test_function_call_reset_on_interruption(self):
        """Test that function call metrics are cleared on interruption."""
        observer = UserBotLatencyObserver()
        processor = IdentityFilter()

        breakdowns = []

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(breakdown)

        frames_to_send = [
            VADUserStoppedSpeakingFrame(),
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={},
            ),
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={},
                result={},
            ),
            InterruptionFrame(),
            BotStartedSpeakingFrame(),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            observers=[observer],
        )

        self.assertEqual(len(breakdowns), 1)
        self.assertEqual(len(breakdowns[0].function_calls), 0)


class TestLatencyBreakdownChronologicalEvents(unittest.TestCase):
    """Tests for LatencyBreakdown.chronological_events()."""

    def test_events_sorted_by_start_time(self):
        """Test that events are returned in chronological order."""
        breakdown = LatencyBreakdown(
            user_turn_start_time=100.0,
            user_turn_secs=0.150,
            ttfb=[
                TTFBBreakdownMetrics(
                    processor="OpenAILLMService#0",
                    model="gpt-4o",
                    start_time=100.200,
                    duration_secs=0.250,
                ),
                TTFBBreakdownMetrics(
                    processor="DeepgramSTTService#0",
                    start_time=100.050,
                    duration_secs=0.080,
                ),
                TTFBBreakdownMetrics(
                    processor="CartesiaTTSService#0",
                    start_time=100.500,
                    duration_secs=0.070,
                ),
            ],
            function_calls=[
                FunctionCallMetrics(
                    function_name="get_weather",
                    start_time=100.450,
                    duration_secs=0.120,
                ),
            ],
            text_aggregation=TextAggregationBreakdownMetrics(
                processor="CartesiaTTSService#0",
                start_time=100.480,
                duration_secs=0.030,
            ),
        )

        events = breakdown.chronological_events()

        self.assertEqual(len(events), 6)
        self.assertEqual(events[0], "User turn: 0.150s")
        self.assertEqual(events[1], "DeepgramSTTService#0: TTFB 0.080s")
        self.assertEqual(events[2], "OpenAILLMService#0: TTFB 0.250s")
        self.assertEqual(events[3], "get_weather: 0.120s")
        self.assertEqual(events[4], "CartesiaTTSService#0: text aggregation 0.030s")
        self.assertEqual(events[5], "CartesiaTTSService#0: TTFB 0.070s")

    def test_empty_breakdown(self):
        """Test that an empty breakdown returns no events."""
        breakdown = LatencyBreakdown()
        self.assertEqual(breakdown.chronological_events(), [])

    def test_user_turn_requires_both_fields(self):
        """Test that user turn is only included when both start_time and secs are set."""
        # Only start_time, no duration
        breakdown = LatencyBreakdown(user_turn_start_time=100.0)
        self.assertEqual(breakdown.chronological_events(), [])

        # Only duration, no start_time
        breakdown = LatencyBreakdown(user_turn_secs=0.150)
        self.assertEqual(breakdown.chronological_events(), [])

    def test_ttfb_only(self):
        """Test breakdown with only TTFB metrics."""
        breakdown = LatencyBreakdown(
            ttfb=[
                TTFBBreakdownMetrics(processor="LLM#0", start_time=100.0, duration_secs=0.200),
            ],
        )
        events = breakdown.chronological_events()
        self.assertEqual(events, ["LLM#0: TTFB 0.200s"])


if __name__ == "__main__":
    unittest.main()
