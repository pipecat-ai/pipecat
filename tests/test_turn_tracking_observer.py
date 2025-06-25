#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.tests.utils import SleepFrame, run_test


class TestTurnTrackingObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for TurnTrackingObserver."""

    async def test_normal_conversation_flow(self):
        """Test a normal conversation with two complete turns."""
        # Create observer with a short timeout
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=0.2)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Record start/end events with turn numbers
        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Turn 1
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),  # < 0.2 seconds turn_end_timeout
            # Turn 2
            UserStartedSpeakingFrame(),  # New turn starts
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Add a sleep frame to allow turn timeout to occur
            SleepFrame(sleep=0.4),  # > 0.2 seconds turn_end_timeout
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
        )

        # Verify turn events
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: False)",
            "Turn 2 started",
            "Turn 2 ended (interrupted: False)",
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 2)

    async def test_user_speaks_twice_before_bot(self):
        """Test when user speaks twice before bot responds, should be same turn."""
        # Create observer with a short timeout
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=0.2)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Record start/end events with turn numbers
        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Turn 1 - User speaks twice before bot responds
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),  # Second user speaking event should not start a new turn
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Turn 2
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Add a sleep frame to allow turn timeout to occur
            SleepFrame(sleep=0.4),  # > 0.2 seconds turn_end_timeout
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
        )

        # Verify turn events - should only see two turns despite user speaking twice
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: False)",
            "Turn 2 started",
            "Turn 2 ended (interrupted: False)",
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 2)

    async def test_user_interrupts_bot(self):
        """Test when user interrupts bot speaking, should end current turn and start new one.

        Note: This test also verifies that the EndFrame ends the turn correctly.
        """
        # Create observer with a short timeout
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=0.2)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Record start/end events with turn numbers
        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Turn 1
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            # Interruption here - user starts speaking while bot is still speaking
            UserStartedSpeakingFrame(),  # This should end Turn 1 and start Turn 2
            SleepFrame(sleep=0.4),  # > 0.2 seconds turn_end_timeout
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
        )

        # Verify turn events - should see Turn 1 interrupted
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: True)",  # First turn was interrupted
            "Turn 2 started",  # New turn started after interruption
            "Turn 2 ended (interrupted: True)",  # Second turn ends due to EndFrame
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 2)

    async def test_bot_starts_stops_multiple_times(self):
        """Test that multiple bot start/stop frames in the same turn work correctly."""
        # Create observer with a short timeout
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=0.2)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Start turn with user speaking
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            # Bot speaks, stops, speaks again (simulating HTTP TTS or function calls)
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),  # Bot speaks again, should not end turn
            BotStoppedSpeakingFrame(),
            # Add a sleep frame to allow turn timeout to occur
            SleepFrame(sleep=0.4),  # > 0.2 seconds turn_end_timeout
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
        )

        # Should only be one turn with a normal end
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: False)",
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 1)

    async def test_cancel_frame_flushes_active_turn(self):
        """Test that CancelFrame properly flushes an active turn."""
        # Create observer with a long timeout to ensure CancelFrame is what ends the turn
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=5.0)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Record start/end events with turn numbers
        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Start a turn but don't complete it naturally
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            # Send CancelFrame while bot is still speaking
            CancelFrame(),
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            CancelFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
            send_end_frame=False,  # Don't send EndFrame since we're testing CancelFrame
        )

        # Verify that the turn was ended due to CancelFrame (marked as interrupted)
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: True)",  # Should be interrupted due to CancelFrame
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 1)

    async def test_end_frame_with_no_active_turn(self):
        """Test that EndFrame doesn't cause issues when no turn is active."""
        # Create observer
        turn_observer = TurnTrackingObserver(turn_end_timeout_secs=0.2)

        # Create identity filter (passes all frames through)
        processor = IdentityFilter()

        # Record start/end events with turn numbers
        turn_events = []

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            turn_events.append(f"Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            turn_events.append(f"Turn {turn_number} ended (interrupted: {was_interrupted})")

        frames_to_send = [
            # Complete a turn normally
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.4),  # Let turn end naturally due to timeout
            # EndFrame will be sent by run_test when no turn is active
        ]

        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            observers=[turn_observer],
            send_end_frame=True,
        )

        # Should only see one turn that ends naturally, EndFrame shouldn't create additional events
        expected_events = [
            "Turn 1 started",
            "Turn 1 ended (interrupted: False)",  # Ends due to timeout, not EndFrame
        ]
        self.assertEqual(turn_events, expected_events)
        self.assertEqual(turn_observer._turn_count, 1)


if __name__ == "__main__":
    unittest.main()
