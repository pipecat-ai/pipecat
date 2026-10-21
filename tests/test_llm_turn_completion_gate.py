#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import LLMTurnCompletionGate, SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    FilterIncompleteUserTurnStrategies,
    UserTurnStrategies,
    filter_incomplete,
)


class MarkerLLM(LLMService):
    """Answers with a turn-completion marker: ◐ for a turn trailing off on "to", ● otherwise."""

    def __init__(self, *, delay: float = 0, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self._delay = delay
        self.inferences: list[LLMContextFrame] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        self.inferences.append(frame)
        text = frame.context.messages[-1]["content"]
        marker = "◐" if text.endswith(" to") else "●"
        await self.push_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(self._delay)
        await self._push_llm_text(f"{marker} ANSWER[{text}]")
        await self.push_frame(LLMFullResponseEndFrame())


async def run_turn(
    frames,
    strategies: UserTurnStrategies,
    *,
    delay: float = 0,
    user_turn_stop_timeout: float = 5.0,
) -> tuple[list, LLMContext, MarkerLLM]:
    """Run the frames through a user aggregator and a marker LLM.

    Returns:
        The frames that reached the end of the pipeline, the context, and the LLM.
    """
    context = LLMContext()
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=strategies,
            user_turn_stop_timeout=user_turn_stop_timeout,
        ),
    )
    llm = MarkerLLM(delay=delay)
    down, _ = await run_test(Pipeline([aggregator, llm]), frames_to_send=frames)
    return down, context, llm


def eager(text: str) -> EagerTranscriptionFrame:
    return EagerTranscriptionFrame(text, "user", "2026-10-20T00:00:00Z")


def final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "2026-10-20T00:00:00Z")


def spoken(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


def speculative(llm: MarkerLLM) -> list[bool]:
    return [f.speculation for f in llm.inferences]


def stops(frames) -> int:
    return len([f for f in frames if isinstance(f, UserStoppedSpeakingFrame)])


class TestFilterIncomplete(unittest.TestCase):
    def test_gates_every_stop_strategy_and_keeps_the_start_strategies(self):
        detector = SpeechTimeoutUserTurnStopStrategy()
        strategies = UserTurnStrategies(stop=[detector])

        gated = filter_incomplete(strategies)

        assert gated.start == strategies.start
        assert len(gated.stop) == 1
        assert isinstance(gated.stop[0], LLMTurnCompletionGate)
        assert gated.stop[0].inner is detector

    def test_defaults_to_the_default_strategies(self):
        gated = filter_incomplete()

        assert len(gated.stop) == 1
        assert isinstance(gated.stop[0], LLMTurnCompletionGate)

    def test_the_bundle_gates_its_stop_strategies(self):
        detector = SpeechTimeoutUserTurnStopStrategy()

        strategies = FilterIncompleteUserTurnStrategies(stop=[detector])

        assert len(strategies.stop) == 1
        assert isinstance(strategies.stop[0], LLMTurnCompletionGate)
        assert strategies.stop[0].inner is detector


class TestGateOverALocalDetector(unittest.IsolatedAsyncioTestCase):
    async def test_the_user_going_on_lets_the_detector_answer_again(self):
        aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=filter_incomplete(
                    UserTurnStrategies(
                        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.1)]
                    )
                ),
            ),
        )
        triggered = []

        @aggregator.event_handler("on_user_turn_inference_triggered")
        async def on_inference_triggered(aggregator, strategy):
            triggered.append(strategy)

        down, _ = await run_test(
            Pipeline([aggregator, MarkerLLM()]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                final("I went to"),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.4),
                # The LLM said ◐: the user goes on.
                VADUserStartedSpeakingFrame(),
                final("Tokyo."),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.4),
            ],
        )

        # The detector answered both segments, and the second one's ● ended the turn.
        assert len(triggered) == 2
        assert spoken(down) == ["ANSWER[Tokyo.]"]
        assert stops(down) == 1


class TestGateOverEagerEndOfTurn(unittest.IsolatedAsyncioTestCase):
    async def test_a_complete_turn_is_spoken_at_the_end_of_turn(self):
        down, context, llm = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went to Tokyo."),
                SleepFrame(),
                final("I went to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
        )

        # One inference, the speculative one, released by the end of the turn.
        assert speculative(llm) == [True]
        assert spoken(down) == ["ANSWER[I went to Tokyo.]"]
        assert stops(down) == 1
        assert context.messages == [{"role": "user", "content": "I went to Tokyo."}]

    async def test_an_incomplete_turn_stays_open_and_unspoken(self):
        down, _, _ = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went to"),
                SleepFrame(),
                final("I went to"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(
                EagerUserTurnStrategies(),
                config=UserTurnCompletionConfig(incomplete_short_timeout=0.2),
            ),
        )

        # The turn never ends. (The session's end writes it to the context.)
        assert spoken(down) == []
        assert stops(down) == 0
        # The response is speculative, so its ◐ arms no re-prompt: the context
        # it would run against doesn't hold the turn yet.
        assert not any(isinstance(f, LLMMessagesAppendFrame) for f in down)

    async def test_the_user_going_on_withdraws_the_unspoken_response(self):
        down, context, _ = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went to"),
                SleepFrame(),
                final("I went to"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.2),
                # The user goes on after a pause.
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("Tokyo."),
                SleepFrame(),
                final("Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
        )

        assert len([f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]) == 1
        # The next end of turn is answered with everything the user said.
        assert spoken(down) == ["ANSWER[I went to Tokyo.]"]
        assert context.messages == [{"role": "user", "content": "I went to Tokyo."}]

    async def test_a_verdict_after_the_end_of_turn_ends_it_when_it_arrives(self):
        down, _, llm = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went to Tokyo."),
                SleepFrame(),
                final("I went to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.8),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
            delay=0.3,
        )

        assert speculative(llm) == [True]
        assert spoken(down) == ["ANSWER[I went to Tokyo.]"]

    async def test_a_turn_ended_without_a_prediction_is_answered_as_usual(self):
        down, _, llm = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                final("I went to"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
        )

        # An ordinary inference, whose ◐ still keeps the turn open.
        assert speculative(llm)[0] is False
        assert spoken(down) == []
        assert stops(down) == 0

    async def test_a_turn_left_incomplete_is_answered_once_the_user_stays_quiet(self):
        _, context, llm = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went to"),
                SleepFrame(),
                final("I went to"),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.8),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
            user_turn_stop_timeout=0.3,
        )

        # The turn ends, the unspoken response is withdrawn, and the LLM
        # answers the turn as usual.
        assert speculative(llm) == [True, False]
        assert context.messages[0] == {"role": "user", "content": "I went to"}

    async def test_a_withdrawn_complete_response_leaves_the_next_one_to_be_spoken(self):
        down, _, _ = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went"),
                SleepFrame(),
                # The service withdraws its prediction: the user went on.
                EagerEndOfTurnCancelFrame(),
                SleepFrame(),
                eager("I went to Tokyo."),
                SleepFrame(),
                final("I went to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
        )

        assert spoken(down) == ["ANSWER[I went to Tokyo.]"]

    async def test_a_differing_transcript_is_answered_again(self):
        down, context, llm = await run_turn(
            [
                ProposedUserStartedSpeakingFrame(),
                SleepFrame(),
                eager("I went"),
                SleepFrame(),
                final("I went to Tokyo."),
                SleepFrame(),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.5),
            ],
            filter_incomplete(EagerUserTurnStrategies()),
        )

        assert speculative(llm) == [True, False]
        assert spoken(down) == ["ANSWER[I went to Tokyo.]"]
        assert context.messages == [{"role": "user", "content": "I went to Tokyo."}]
