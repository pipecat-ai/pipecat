import asyncio
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ClientConnectedFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    TextAggregationMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.user_bot_latency_observer import (
    PRINTS_AS_ZERO_SECS,
    FunctionCallMetrics,
    LatencyBreakdown,
    LatencyContribution,
    TextAggregationBreakdownMetrics,
    TTFBBreakdownMetrics,
    UserBotLatencyObserver,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    UserTurnCompletionLLMServiceMixin,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _MarkerProcessor(UserTurnCompletionLLMServiceMixin, FrameProcessor):
    """The smallest processor that runs the turn completion protocol."""

    pass


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
            SleepFrame(sleep=0.2),  # Simulate turn analyzer wait
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
            SleepFrame(sleep=0.2),
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


class _CycleDriver:
    """Drives an observer through a cycle on a clock the test controls."""

    async def asyncSetUp(self):
        # A clock the test advances by hand, so a cycle can describe intervals
        # of any length without waiting them out.
        self.clock = 1_000_000.0
        await self._observe()

    async def _observe(self, **kwargs):
        """Start a fresh observer reading this test's clock."""
        self.observer = UserBotLatencyObserver(time_source=lambda: self.clock, **kwargs)
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.breakdowns = []

        @self.observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            self.breakdowns.append(breakdown)

    def _wait(self, seconds: float):
        """Advance the clock without sleeping."""
        self.clock += seconds

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

    async def _settle(self):
        """Let the event handler task deliver the breakdown."""
        await asyncio.sleep(0.01)


class TestLatencyContributions(_CycleDriver, unittest.IsolatedAsyncioTestCase):
    """Contributions name each part of a cycle and sum to its latency."""

    async def _complete_turn(
        self,
        *,
        marker=True,
        hold=False,
        hold_secs=0.06,
        detect=0.0,
        aggregation=0.0,
        buffered=0.0,
    ):
        # The cycle is measured from the silence the VAD waited out.
        self.silence_at = self.clock - 0.02
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        self._wait(0.02)
        await self._push(TranscriptionFrame(user_id="u", text="hi", timestamp=""), source="STT#0")
        if detect:
            self._wait(detect)
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        if hold:
            self._wait(0.01)
            await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.01)]))
            await self._push(LLMMarkerFrame("◐"))
            self._wait(hold_secs)
            await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        if marker:
            await self._push(LLMMarkerFrame("●", append_to_context_immediately=False))
            self._wait(0.02)
        if buffered:
            self._wait(buffered)
        await self._push(LLMTextFrame("Hi!"), source="LLM#0")
        if aggregation:
            self._wait(aggregation)
            await self._push(
                MetricsFrame(
                    data=[TextAggregationMetricsData(processor="TTS#0", value=aggregation)]
                )
            )
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        self.spoke_at = self.clock
        await asyncio.sleep(0.01)  # let the event handler task run
        return self.breakdowns[-1]

    async def test_contributions_sum_to_the_measured_latency(self):
        """Nothing in the interval goes unnamed."""
        breakdown = await self._complete_turn()
        total = sum(c.duration_secs for c in breakdown.contributions)
        self.assertAlmostEqual(total, self.spoke_at - self.silence_at, places=6)

    async def test_a_complete_turn_names_each_stage(self):
        """A turn the LLM completed has no wait, and lists the rest in order."""
        breakdown = await self._complete_turn()
        labels = [c.label for c in breakdown.contributions]
        self.assertEqual(
            labels,
            [
                "endpointing wait",
                "transcription",
                "LLM inference",
                "turn completion",
                "speech synthesis",
            ],
        )
        endpointing = breakdown.contributions[0]
        self.assertEqual(endpointing.owner, "config: VAD stop_secs")
        self.assertAlmostEqual(endpointing.duration_secs, 0.02, delta=0.01)

    async def test_a_held_turn_names_the_wait(self):
        """An incomplete marker shows up as a wait, not as slow inference."""
        breakdown = await self._complete_turn(hold=True)
        wait = next(c for c in breakdown.contributions if c.label == "waiting for user")
        self.assertEqual(wait.owner, "config: filter_incomplete_user_turns")
        self.assertGreater(wait.duration_secs, 0.05)
        # Both inferences are listed, so neither absorbs the wait.
        self.assertEqual(len([c for c in breakdown.contributions if c.label == "LLM inference"]), 2)

    async def test_a_bot_without_turn_completion_has_no_marker_entry(self):
        """Parts that didn't happen aren't listed."""
        breakdown = await self._complete_turn(marker=False)
        labels = [c.label for c in breakdown.contributions]
        self.assertNotIn("turn completion", labels)
        self.assertNotIn("waiting for user", labels)
        # The wait for a speakable token is the LLM streaming, not a diagnosis.
        self.assertNotIn("awaiting speakable text", labels)

    async def test_a_missing_marker_is_reported_once_markers_are_in_use(self):
        """A bot that emits markers, on a response that carried none."""
        await self._complete_turn()  # establishes that this bot uses markers
        breakdown = await self._complete_turn(marker=False, buffered=0.05)
        buffered = next(c for c in breakdown.contributions if c.label == "awaiting speakable text")
        self.assertGreater(buffered.duration_secs, 0.04)

    def test_contribution_lines_order_and_total(self):
        """Chronological by default, largest first by cost, always with a total."""
        breakdown = LatencyBreakdown(
            contributions=[
                LatencyContribution(
                    label="endpointing wait", owner="config", start_time=0.0, duration_secs=0.2
                ),
                LatencyContribution(
                    label="speech synthesis", owner="TTS#0", start_time=0.2, duration_secs=0.4
                ),
            ]
        )
        self.assertIn("endpointing wait", breakdown.contribution_lines()[0])
        self.assertIn("speech synthesis", breakdown.contribution_lines(by_cost=True)[0])
        self.assertIn("0.600s  TOTAL", breakdown.contribution_lines()[-1])

    async def test_turn_completion_covers_the_gate_and_the_marker_token(self):
        """The span runs from the LLM's first chunk to the first speakable token."""
        breakdown = await self._complete_turn()
        completion = next(c for c in breakdown.contributions if c.label == "turn completion")
        self.assertEqual(completion.owner, "config: filter_incomplete_user_turns")
        self.assertGreater(completion.duration_secs, 0.015)

    async def test_zero_threshold_lists_every_contribution(self):
        """Nothing is folded away when the threshold is off."""
        self.observer = UserBotLatencyObserver(min_contribution_secs=0)
        await self.observer.setup(TaskManager())
        self.breakdowns = []

        @self.observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            self.breakdowns.append(breakdown)

        breakdown = await self._complete_turn()
        # Every span is listed, so the residual is gone or negligible.
        pipeline = next((c for c in breakdown.contributions if c.label == "pipeline"), None)
        self.assertTrue(pipeline is None or pipeline.duration_secs < 0.01)
        self.assertIn("output transport", [c.label for c in breakdown.contributions])

    async def test_core_stages_are_listed_when_brief_but_measurable(self):
        """A stage of a few milliseconds is listed; one that rounds to zero is not."""
        breakdown = await self._complete_turn(detect=0.002)
        detection = next(c for c in breakdown.contributions if c.label == "turn detection")
        self.assertLess(detection.duration_secs, 0.005)
        self.assertEqual(detection.owner, "config: user turn strategies")

        breakdown = await self._complete_turn()
        self.assertNotIn("turn detection", [c.label for c in breakdown.contributions])

    async def test_sentence_aggregation_is_listed_only_when_it_waits(self):
        """Sentence mode reports the wait; token mode reports nothing."""
        breakdown = await self._complete_turn(aggregation=0.05)
        aggregation = next(c for c in breakdown.contributions if c.label == "sentence aggregation")
        self.assertAlmostEqual(aggregation.duration_secs, 0.05, delta=0.01)
        self.assertEqual(aggregation.owner, "config: text_aggregation_mode")
        synthesis = next(c for c in breakdown.contributions if c.label == "speech synthesis")
        self.assertGreater(synthesis.start_time, aggregation.start_time)

        breakdown = await self._complete_turn(aggregation=0)
        self.assertNotIn("sentence aggregation", [c.label for c in breakdown.contributions])

    async def test_function_handlers_are_listed_with_the_wait_that_precedes_them(self):
        """A tool turn names both writing the call and running the handler."""
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        await self._push(
            TranscriptionFrame(user_id="u", text="weather?", timestamp=""), source="STT#0"
        )
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        self._wait(0.02)
        await self._push(
            FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="1",
                arguments={},
                cancel_on_interruption=False,
            )
        )
        self._wait(0.03)
        await self._push(
            FunctionCallResultFrame(
                function_name="get_weather", tool_call_id="1", arguments={}, result={"ok": True}
            )
        )
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        await self._push(LLMTextFrame("Nice out."), source="LLM#0")
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await asyncio.sleep(0.01)  # let the event handler task run

        contributions = self.breakdowns[-1].contributions
        call = next(c for c in contributions if c.label == "function handler")
        self.assertEqual(call.owner, "get_weather")
        self.assertAlmostEqual(call.duration_secs, 0.03, delta=0.02)

        writing = next(c for c in contributions if c.label == "LLM tool call")
        self.assertEqual(writing.owner, "LLM#0")
        self.assertLess(writing.start_time, call.start_time)

        # Nothing left over once the tool spans are named.
        residual = next((c for c in contributions if c.label == "pipeline"), None)
        self.assertIsNone(residual)

    async def test_concurrent_handlers_are_one_span(self):
        """Handlers dispatched together count their wall clock once."""
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        await self._push(
            TranscriptionFrame(user_id="u", text="weather and food?", timestamp=""), source="STT#0"
        )
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        for name, call_id in [("get_weather", "1"), ("get_restaurant", "2")]:
            await self._push(
                FunctionCallInProgressFrame(
                    function_name=name,
                    tool_call_id=call_id,
                    arguments={},
                    cancel_on_interruption=False,
                )
            )
        self._wait(0.03)
        await self._push(
            FunctionCallResultFrame(
                function_name="get_restaurant", tool_call_id="2", arguments={}, result={}
            )
        )
        self._wait(0.05)
        await self._push(
            FunctionCallResultFrame(
                function_name="get_weather", tool_call_id="1", arguments={}, result={}
            )
        )
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        await self._push(LLMTextFrame("Here you go."), source="LLM#0")
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await asyncio.sleep(0.01)  # let the event handler task run

        breakdown = self.breakdowns[-1]
        handlers = [c for c in breakdown.contributions if c.label == "function handler"]
        self.assertEqual(len(handlers), 1)
        # Named for the handler the wait actually depended on.
        self.assertEqual(handlers[0].owner, "get_weather +1")
        self.assertAlmostEqual(handlers[0].duration_secs, 0.08, delta=0.02)

        # Overlapping spans would push the sum past the latency they describe.
        total = sum(c.duration_secs for c in breakdown.contributions)
        self.assertAlmostEqual(total, breakdown.user_turn_secs or total, delta=0.02)

    async def test_no_two_contributions_overlap(self):
        """Spans tile the interval rather than covering the same time twice."""
        breakdown = await self._complete_turn(hold=True, aggregation=0.03)
        spans = sorted(breakdown.contributions, key=lambda c: c.start_time)
        for previous, following in zip(spans, spans[1:]):
            if previous.label == "pipeline" or following.label == "pipeline":
                continue
            self.assertLessEqual(
                previous.start_time + previous.duration_secs, following.start_time + 1e-6
            )

    async def test_a_long_hold_is_reported_without_waiting_it_out(self):
        """A ten second hold costs the test nothing, because the clock is ours."""
        breakdown = await self._complete_turn(hold=True, hold_secs=10.0)
        wait = next(c for c in breakdown.contributions if c.label == "waiting for user")
        self.assertAlmostEqual(wait.duration_secs, 10.0, places=6)

    async def test_durations_are_exact(self):
        """Every span is the interval the cycle described, to the microsecond."""
        breakdown = await self._complete_turn(detect=0.4, aggregation=0.25)
        by_label = {c.label: c.duration_secs for c in breakdown.contributions}
        # Microseconds apart at most, where a wall clock left tens of them.
        self.assertAlmostEqual(by_label["endpointing wait"], 0.02, places=6)
        self.assertAlmostEqual(by_label["turn detection"], 0.4, places=6)
        self.assertAlmostEqual(by_label["sentence aggregation"], 0.25, places=6)
        self.assertNotIn("pipeline", by_label)

    async def test_a_turn_held_twice_reports_both_waits(self):
        """A turn can be told to wait more than once before it completes."""
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        self._wait(0.02)
        await self._push(TranscriptionFrame(user_id="u", text="hi", timestamp=""), source="STT#0")
        for _ in range(2):
            await self._push(LLMFullResponseStartFrame(), source="LLM#0")
            self._wait(0.02)
            await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
            await self._push(LLMMarkerFrame("◐"))
            self._wait(5.0)
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        await self._push(LLMMarkerFrame("●", append_to_context_immediately=False))
        self._wait(0.02)
        await self._push(LLMTextFrame("Go on."), source="LLM#0")
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await asyncio.sleep(0.01)  # let the event handler task run

        contributions = self.breakdowns[-1].contributions
        waits = [c for c in contributions if c.label == "waiting for user"]
        self.assertEqual([round(w.duration_secs, 3) for w in waits], [5.0, 5.0])
        # Each attempt is its own inference: none absorbs a wait.
        inferences = [c for c in contributions if c.label == "LLM inference"]
        self.assertEqual(len(inferences), 3)
        self.assertTrue(all(c.duration_secs < 1.0 for c in inferences))

    async def test_a_high_threshold_still_accounts_for_the_whole_interval(self):
        """Spans the threshold hides roll into pipeline rather than vanishing."""
        await self._observe(min_contribution_secs=0.1)
        breakdown = await self._complete_turn(aggregation=0.06)

        total = sum(c.duration_secs for c in breakdown.contributions)
        self.assertAlmostEqual(total, self.spoke_at - self.silence_at, places=6)
        hidden = next(c for c in breakdown.contributions if c.label == "pipeline")
        self.assertGreater(hidden.duration_secs, 0.06)


# Every label and setting the timeline is allowed to name, so a typo or a
# renamed stage shows up as a failure rather than as odd-looking output.
KNOWN_LABELS = {
    "endpointing wait",
    "transcription",
    "turn detection",
    "LLM inference",
    "turn completion",
    "waiting for user",
    "awaiting speakable text",
    "LLM tool call",
    "function handler",
    "sentence aggregation",
    "speech synthesis",
    "output transport",
    "pipeline",
}
KNOWN_SETTINGS = {
    "config: VAD stop_secs",
    "config: user turn strategies",
    "config: filter_incomplete_user_turns",
    "config: text_aggregation_mode",
}


class TestLatencyContributionInvariants(_CycleDriver, unittest.IsolatedAsyncioTestCase):
    """Properties that hold for any shape of turn.

    Where the other tests pin what a particular turn should say, these say what
    must be true of every turn, so a stage that stops matching its pair of
    moments fails here rather than quietly becoming pipeline time.
    """

    async def _cycle(self, *, filtered=True, holds=0, tools="none", aggregation=0.0):
        """Drive one turn of the given shape and return its breakdown."""
        self.silence_at = self.clock - 0.2
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=self.clock))
        self._wait(0.1)
        await self._push(
            TranscriptionFrame(user_id="u", text="hello", timestamp=""), source="STT#0"
        )
        self._wait(0.05)

        async def inference():
            await self._push(LLMFullResponseStartFrame(), source="LLM#0")
            self._wait(0.3)
            await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.3)]))

        for _ in range(holds):
            await inference()
            self._wait(0.02)
            await self._push(LLMMarkerFrame("◐"))
            self._wait(1.0)

        if tools != "none":
            await inference()
            self._wait(0.05)
            names = ["get_weather"] if tools == "one" else ["get_weather", "get_food"]
            for index, name in enumerate(names):
                await self._push(
                    FunctionCallInProgressFrame(
                        function_name=name,
                        tool_call_id=str(index),
                        arguments={},
                        cancel_on_interruption=False,
                    )
                )
            self._wait(0.2)
            for index, name in enumerate(names):
                await self._push(
                    FunctionCallResultFrame(
                        function_name=name, tool_call_id=str(index), arguments={}, result={}
                    )
                )

        await inference()
        if filtered:
            await self._push(LLMMarkerFrame("●", append_to_context_immediately=False))
            self._wait(0.03)
        await self._push(LLMTextFrame("Hello there."), source="LLM#0")
        if aggregation:
            self._wait(aggregation)
            await self._push(
                MetricsFrame(
                    data=[TextAggregationMetricsData(processor="TTS#0", value=aggregation)]
                )
            )
        self._wait(0.15)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        self._wait(0.01)
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        self.spoke_at = self.clock
        await self._settle()
        return self.breakdowns[-1]

    def _assert_invariants(self, breakdown):
        """Check what must hold of any timeline."""
        contributions = breakdown.contributions
        self.assertTrue(contributions)

        # The parts account for the interval they describe.
        total = sum(c.duration_secs for c in contributions)
        self.assertAlmostEqual(total, self.spoke_at - self.silence_at, places=6)

        # The pipeline entry is a residual spread across the cycle, so it is
        # left out of the checks that treat contributions as a timeline.
        timeline = [c for c in contributions if c.label != "pipeline"]
        for previous, following in zip(timeline, timeline[1:]):
            self.assertLessEqual(
                previous.start_time + previous.duration_secs, following.start_time + 1e-9
            )
        for contribution in contributions:
            self.assertGreaterEqual(contribution.duration_secs, PRINTS_AS_ZERO_SECS)
            self.assertGreaterEqual(contribution.start_time, self.silence_at - 1e-9)
            self.assertLessEqual(
                contribution.start_time + contribution.duration_secs, self.spoke_at + 1e-9
            )
            self.assertIn(contribution.label, KNOWN_LABELS)
            if contribution.owner.startswith("config:"):
                self.assertIn(contribution.owner, KNOWN_SETTINGS)

        # Nothing of substance went unnamed: a stretch the table does not cover
        # means a stage stopped matching its pair of moments.
        residual = next((c for c in contributions if c.label == "pipeline"), None)
        self.assertLess(residual.duration_secs if residual else 0.0, 0.01)

    async def test_invariants_hold_across_turn_shapes(self):
        """Filtering, holds, tools and aggregation, in combination."""
        for filtered in (True, False):
            for holds in (0, 1, 2) if filtered else (0,):
                for tools in ("none", "one", "concurrent"):
                    for aggregation in (0.0, 0.06):
                        with self.subTest(
                            filtered=filtered, holds=holds, tools=tools, aggregation=aggregation
                        ):
                            await self._observe()
                            breakdown = await self._cycle(
                                filtered=filtered,
                                holds=holds,
                                tools=tools,
                                aggregation=aggregation,
                            )
                            self._assert_invariants(breakdown)


class TestObserverEdges(_CycleDriver, unittest.IsolatedAsyncioTestCase):
    """Cycles that are interrupted, absent, or follow one another."""

    async def test_a_second_complete_marker_is_ignored(self):
        """The first completion is the one the turn waited for."""
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        await self._push(LLMMarkerFrame("●", append_to_context_immediately=False))
        first_marker_at = self.clock
        self._wait(0.05)
        await self._push(LLMMarkerFrame("●", append_to_context_immediately=False))
        self._wait(0.02)
        await self._push(LLMTextFrame("Hi."), source="LLM#0")
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await self._settle()

        completion = next(
            c for c in self.breakdowns[-1].contributions if c.label == "turn completion"
        )
        # The span ends at the first speakable token, so a second marker
        # arriving later neither restarts it nor adds one of its own.
        self.assertLessEqual(completion.start_time, first_marker_at)
        self.assertEqual(
            len([c for c in self.breakdowns[-1].contributions if c.label == "turn completion"]), 1
        )

    def test_an_empty_breakdown_has_nothing_to_print(self):
        """A breakdown with no contributions formats as nothing at all."""
        self.assertEqual(LatencyBreakdown().contribution_lines(), [])

    async def test_upstream_frames_are_ignored(self):
        """Only what flows towards the user shapes the timeline."""
        await self.observer.on_push_frame(
            FramePushed(
                source=IdentityFilter(name="STT#0"),
                destination=IdentityFilter(name="destination"),
                frame=VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock),
                direction=FrameDirection.UPSTREAM,
                timestamp=0,
            )
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await self._settle()
        self.assertEqual(self.breakdowns, [])

    async def test_metrics_outside_a_cycle_are_ignored(self):
        """A metric arriving before any user turn records nothing."""
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.5)]))
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await self._settle()
        self.assertEqual(self.breakdowns, [])

    async def test_a_greeting_has_no_timeline(self):
        """The bot speaking first is not a user-to-bot cycle."""
        await self._push(ClientConnectedFrame())
        self._wait(1.0)
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        await self._settle()
        self.assertEqual(self.breakdowns[-1].contributions, [])

    async def test_an_interruption_discards_the_cycle(self):
        """Frames from a cancelled cycle do not reach the next one."""
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        await self._push(TranscriptionFrame(user_id="u", text="wait", timestamp=""), source="STT#0")
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.3)
        await self._push(InterruptionFrame())

        self.silence_at = self.clock - 0.02
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.02, timestamp=self.clock))
        await self._push(TranscriptionFrame(user_id="u", text="go", timestamp=""), source="STT#0")
        await self._push(LLMFullResponseStartFrame(), source="LLM#0")
        self._wait(0.02)
        await self._push(MetricsFrame(data=[TTFBMetricsData(processor="LLM#0", value=0.02)]))
        await self._push(LLMTextFrame("Hi."), source="LLM#0")
        self._wait(0.02)
        await self._push(
            TTSAudioRawFrame(audio=b"", sample_rate=24000, num_channels=1), source="TTS#0"
        )
        await self._push(BotStartedSpeakingFrame(), source="Transport#0")
        self.spoke_at = self.clock
        await self._settle()

        contributions = self.breakdowns[-1].contributions
        total = sum(c.duration_secs for c in contributions)
        self.assertAlmostEqual(total, self.spoke_at - self.silence_at, places=6)
        # The abandoned inference is not in the timeline it preceded.
        self.assertEqual(len([c for c in contributions if c.label == "LLM inference"]), 1)

    async def test_one_cycle_does_not_reach_the_next(self):
        """Back to back turns are measured independently."""
        driver = TestLatencyContributionInvariants._cycle
        first = await driver(self, holds=1)
        first_total = sum(c.duration_secs for c in first.contributions)
        second = await driver(self)
        second_total = sum(c.duration_secs for c in second.contributions)

        self.assertAlmostEqual(second_total, self.spoke_at - self.silence_at, places=6)
        self.assertLess(second_total, first_total)
        self.assertNotIn("waiting for user", [c.label for c in second.contributions])


class TestFrameContracts(unittest.IsolatedAsyncioTestCase):
    """What the observer assumes about the frames other components push.

    These read the real classes rather than the frames a test makes up, so a
    change to the conventions the timeline is built on fails here.
    """

    def test_a_marker_is_stand_alone_unless_it_prefixes_a_response(self):
        """Which the observer reads as an incomplete turn."""
        self.assertTrue(LLMMarkerFrame("◐").append_to_context_immediately)
        self.assertFalse(
            LLMMarkerFrame("●", append_to_context_immediately=False).append_to_context_immediately
        )

    async def _markers_pushed_for(self, text: str) -> list[LLMMarkerFrame]:
        """Run one response through the turn completion protocol."""
        processor = _MarkerProcessor()
        # An incomplete verdict arms a timeout, which needs somewhere to run.
        await processor.setup(frame_processor_setup(TaskManager()))
        pushed = []
        processor.push_frame = AsyncMock(side_effect=lambda f, *a, **k: pushed.append(f))

        await processor._push_turn_text(text)
        return [f for f in pushed if isinstance(f, LLMMarkerFrame)]

    async def test_turn_completion_pushes_markers_that_way_round(self):
        """An incomplete verdict stands alone; a complete one prefixes the reply."""
        incomplete = await self._markers_pushed_for(USER_TURN_INCOMPLETE_SHORT_MARKER)
        self.assertEqual([m.marker for m in incomplete], [USER_TURN_INCOMPLETE_SHORT_MARKER])
        self.assertTrue(incomplete[0].append_to_context_immediately)

        complete = await self._markers_pushed_for(f"{USER_TURN_COMPLETE_MARKER} Hello!")
        self.assertEqual([m.marker for m in complete], [USER_TURN_COMPLETE_MARKER])
        self.assertFalse(complete[0].append_to_context_immediately)

    def test_a_metric_names_the_processor_that_reported_it(self):
        """Which is how an inference is matched to the request that opened it."""
        processor = IdentityFilter(name="LLMService#7")
        self.assertEqual(processor._metrics._processor_name(), processor.name)
