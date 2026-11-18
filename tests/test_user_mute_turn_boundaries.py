#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mute strategies that act on the start and end of a user turn."""

import unittest

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import AlwaysUserMuteStrategy, BaseUserMuteStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


class MuteAfterUserTurnStrategy(BaseUserMuteStrategy):
    """Mutes from the end of the user's turn until the bot has spoken."""

    def __init__(self):
        super().__init__()
        self._muted = False

    async def process_frame(self, frame: Frame) -> bool:
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._muted = False
        return self._muted

    async def handle_user_turn_stopped(self) -> bool | None:
        self._muted = True
        return self._muted


class UnmuteAtUserTurnStartStrategy(BaseUserMuteStrategy):
    """Mutes the user, until the user's turn starts."""

    def __init__(self):
        super().__init__()
        self._muted = True

    async def process_frame(self, frame: Frame) -> bool:
        return self._muted

    async def handle_user_turn_started(self) -> bool | None:
        self._muted = False
        return self._muted


class MutedStrategy(BaseUserMuteStrategy):
    """Mutes the user on every frame."""

    async def process_frame(self, frame: Frame) -> bool:
        return True


class TestBaseUserMuteStrategyTurnBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_a_strategy_keeps_its_decision_at_a_turn_boundary_by_default(self):
        strategy = AlwaysUserMuteStrategy()

        self.assertIsNone(await strategy.handle_user_turn_started())
        self.assertIsNone(await strategy.handle_user_turn_stopped())


class TestUserAggregatorMuteAtTurnBoundaries(unittest.IsolatedAsyncioTestCase):
    def _aggregator(self, strategies):
        aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=ExternalUserTurnStrategies(),
                user_mute_strategies=strategies,
            ),
        )
        self.turns_started = 0

        @aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            self.turns_started += 1

        return aggregator

    async def _run(self, aggregator, frames):
        received, _ = await run_test(
            Pipeline([aggregator]), frames_to_send=frames, expected_down_frames=None
        )
        return received

    def _first_turn(self):
        return [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello.", user_id="", timestamp="now"),
            UserStoppedSpeakingFrame(),
            # The turn ends once the stop strategy has waited for more text.
            SleepFrame(sleep=0.8),
        ]

    async def test_the_frame_after_the_end_of_a_turn_is_muted(self):
        aggregator = self._aggregator([MuteAfterUserTurnStrategy()])

        received = await self._run(
            aggregator,
            [
                *self._first_turn(),
                # What the STT sends right behind the final transcript.
                UserStartedSpeakingFrame(),
                InterimTranscriptionFrame(text="m", user_id="", timestamp="now"),
                SleepFrame(sleep=0.2),
            ],
        )

        self.assertEqual(self.turns_started, 1)
        self.assertEqual(sum(isinstance(f, UserMuteStartedFrame) for f in received), 1)
        self.assertFalse(any(isinstance(f, InterimTranscriptionFrame) for f in received))

    async def test_a_strategy_without_the_hook_mutes_a_frame_too_late(self):
        class LateStrategy(MuteAfterUserTurnStrategy):
            async def handle_user_turn_stopped(self) -> bool | None:
                self._muted = True
                return None

        aggregator = self._aggregator([LateStrategy()])

        await self._run(
            aggregator,
            [*self._first_turn(), UserStartedSpeakingFrame(), SleepFrame(sleep=0.2)],
        )

        self.assertEqual(self.turns_started, 2)

    async def test_the_user_is_unmuted_when_the_bot_has_spoken(self):
        aggregator = self._aggregator([MuteAfterUserTurnStrategy()])

        received = await self._run(
            aggregator,
            [
                *self._first_turn(),
                BotStoppedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                SleepFrame(sleep=0.2),
            ],
        )

        self.assertEqual(self.turns_started, 2)
        self.assertEqual(sum(isinstance(f, UserMuteStoppedFrame) for f in received), 1)

    async def test_another_strategy_keeps_the_user_muted(self):
        aggregator = self._aggregator([UnmuteAtUserTurnStartStrategy(), MutedStrategy()])

        received = await self._run(aggregator, [*self._first_turn(), SleepFrame(sleep=0.2)])

        # The first strategy unmutes at the start of the turn, but the second still mutes.
        self.assertEqual(self.turns_started, 1)
        self.assertEqual(sum(isinstance(f, UserMuteStartedFrame) for f in received), 1)
        self.assertFalse(any(isinstance(f, UserMuteStoppedFrame) for f in received))

    async def test_the_start_of_a_turn_can_unmute_the_user(self):
        aggregator = self._aggregator([UnmuteAtUserTurnStartStrategy()])

        received = await self._run(aggregator, [*self._first_turn(), SleepFrame(sleep=0.2)])

        self.assertEqual(self.turns_started, 1)
        self.assertEqual(sum(isinstance(f, UserMuteStartedFrame) for f in received), 1)
        self.assertEqual(sum(isinstance(f, UserMuteStoppedFrame) for f in received), 1)


if __name__ == "__main__":
    unittest.main()
