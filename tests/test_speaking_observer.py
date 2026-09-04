#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.speaking_observer import SpeakingObserver, SpeechEventKind
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager


class TestSpeakingObserver(unittest.IsolatedAsyncioTestCase):
    """The speaking lifecycle, reported moment by moment."""

    async def asyncSetUp(self):
        self.clock = 1_000_000.0
        self.observer = SpeakingObserver(time_source=lambda: self.clock)
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.events = []

        @self.observer.event_handler("on_speech_event")
        async def on_speech_event(observer, event):
            self.events.append(event)

    def _wait(self, seconds: float):
        """Advance the clock without sleeping."""
        self.clock += seconds

    async def _push(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Feed one frame to the observer, as a pipeline push would."""
        await self.observer.on_push_frame(
            FramePushed(
                source=IdentityFilter(name="source"),
                destination=IdentityFilter(name="destination"),
                frame=frame,
                direction=direction,
                timestamp=0,
            )
        )
        await self._settle()

    async def _settle(self):
        import asyncio

        await asyncio.sleep(0.01)

    async def test_speech_is_timed_to_speech_not_to_the_detector(self):
        """A bar drawn from confirmation times is fat at both ends."""
        await self._push(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=self.clock))
        self._wait(2.5)
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.8, timestamp=self.clock))

        started, stopped = self.events
        self.assertEqual(started.kind, SpeechEventKind.USER_SPEECH_STARTED)
        # The detector confirmed 0.2s after speech began, and 0.8s after it ended.
        self.assertAlmostEqual(started.timestamp, 1_000_000.0 - 0.2, places=6)
        self.assertAlmostEqual(stopped.timestamp, 1_000_002.5 - 0.8, places=6)
        self.assertAlmostEqual(stopped.timestamp - stopped.started_at, 2.5 - 0.8 + 0.2, places=6)

    async def test_a_closing_moment_names_the_stretch_it_ends(self):
        """So an interval reads whole from one record, without pairing."""
        await self._push(BotStartedSpeakingFrame())
        started_at = self.clock
        self._wait(6.3)
        await self._push(BotStoppedSpeakingFrame())

        stopped = self.events[-1]
        self.assertEqual(stopped.kind, SpeechEventKind.BOT_SPEECH_STOPPED)
        self.assertEqual(stopped.started_at, started_at)
        self.assertAlmostEqual(stopped.timestamp - stopped.started_at, 6.3, places=6)

    async def test_the_microphone_and_the_strategy_are_reported_apart(self):
        """The pipeline acts on one; only the other hears a false start."""
        await self._push(VADUserStartedSpeakingFrame(start_secs=0.0, timestamp=self.clock))
        await self._push(UserStartedSpeakingFrame())
        self._wait(1.0)
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.0, timestamp=self.clock))
        self._wait(0.3)  # the strategy deliberating
        await self._push(UserStoppedSpeakingFrame())

        kinds = [e.kind for e in self.events]
        self.assertEqual(
            kinds,
            [
                SpeechEventKind.USER_SPEECH_STARTED,
                SpeechEventKind.USER_TURN_STARTED,
                SpeechEventKind.USER_SPEECH_STOPPED,
                SpeechEventKind.USER_TURN_STOPPED,
            ],
        )
        vad = next(e for e in self.events if e.kind is SpeechEventKind.USER_SPEECH_STOPPED)
        turn = next(e for e in self.events if e.kind is SpeechEventKind.USER_TURN_STOPPED)
        self.assertAlmostEqual(turn.timestamp - vad.timestamp, 0.3, places=6)

    async def test_an_interruption_names_no_speech(self):
        """Any processor can call for one, and it ends no stretch."""
        await self._push(InterruptionFrame())
        event = self.events[0]
        self.assertEqual(event.kind, SpeechEventKind.INTERRUPTION)
        self.assertIsNone(event.started_at)

    async def test_a_broadcast_interruption_is_reported_once(self):
        """It arrives as two frames with two IDs, so an ID alone cannot tell."""
        downstream = InterruptionFrame()
        upstream = InterruptionFrame()
        downstream.broadcast_sibling_id = upstream.id
        upstream.broadcast_sibling_id = downstream.id

        await self._push(downstream, FrameDirection.DOWNSTREAM)
        await self._push(upstream, FrameDirection.UPSTREAM)

        self.assertEqual(len(self.events), 1)

    async def test_a_relayed_frame_is_reported_once(self):
        """A frame passed along the pipeline is one moment, not one per hop."""
        frame = BotStartedSpeakingFrame()
        for _ in range(4):
            await self._push(frame)
        self.assertEqual(len(self.events), 1)

    async def test_a_stretch_whose_start_was_missed_closes_without_one(self):
        """Rather than borrowing a start from another stretch."""
        await self._push(BotStoppedSpeakingFrame())
        self.assertIsNone(self.events[0].started_at)

    async def test_other_frames_are_left_alone(self):
        """Only the speaking lifecycle is reported."""
        await self._push(TextFrame("hello"))
        self.assertEqual(self.events, [])

    async def test_a_barge_in_reads_as_an_overlap(self):
        """Which is what makes a timeline show one voice cutting into another."""
        await self._push(BotStartedSpeakingFrame())
        self._wait(5.0)
        await self._push(VADUserStartedSpeakingFrame(start_secs=0.0, timestamp=self.clock))
        await self._push(InterruptionFrame())
        self._wait(0.2)
        await self._push(BotStoppedSpeakingFrame())

        bot_stopped = self.events[-1]
        user_started = next(e for e in self.events if e.kind is SpeechEventKind.USER_SPEECH_STARTED)
        # The user began while the bot still had the floor.
        self.assertLess(user_started.timestamp, bot_stopped.timestamp)
        self.assertGreater(user_started.timestamp, bot_stopped.started_at)

    async def test_speech_that_never_becomes_a_turn_is_still_reported(self):
        """A cough or a false start reaches the microphone and stops there."""
        await self._push(VADUserStartedSpeakingFrame(start_secs=0.0, timestamp=self.clock))
        self._wait(0.3)
        await self._push(VADUserStoppedSpeakingFrame(stop_secs=0.0, timestamp=self.clock))

        kinds = [e.kind for e in self.events]
        self.assertEqual(
            kinds, [SpeechEventKind.USER_SPEECH_STARTED, SpeechEventKind.USER_SPEECH_STOPPED]
        )
        # The strategy never ruled, so the pipeline never took a turn from it.
        self.assertNotIn(SpeechEventKind.USER_TURN_STARTED, kinds)
