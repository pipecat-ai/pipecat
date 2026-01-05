import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.tests.utils import run_test


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

    async def test_no_measurement_without_user_stop(self):
        """Test that latency is not measured if bot starts without user stopping first."""
        # Create observer
        observer = UserBotLatencyObserver()

        # Create identity filter
        processor = IdentityFilter()

        # Capture latency events
        latencies = []

        @observer.event_handler("on_latency_measured")
        async def on_latency(obs, latency_seconds):
            latencies.append(latency_seconds)

        # Define frame sequence - bot starts without user stop
        frames_to_send = [
            BotStartedSpeakingFrame(),
        ]

        expected_down_frames = [
            BotStartedSpeakingFrame,
        ]

        # Run test
        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[observer],
        )

        # Verify no latency was measured
        self.assertEqual(len(latencies), 0)


if __name__ == "__main__":
    unittest.main()
