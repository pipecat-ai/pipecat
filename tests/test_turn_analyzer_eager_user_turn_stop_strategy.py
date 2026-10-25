#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for :class:`TurnAnalyzerEagerUserTurnStopStrategy`.

The analyzer is scripted: each audio frame pops the next verdict, so the tests
drive the eager end of turn, the commit and the resumption by hand and check
what the strategy emits.
"""

import asyncio
import unittest

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import (
    ExactMatch,
    TurnAnalyzerEagerUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# Verdicts a ScriptedTurnAnalyzer pops, one per audio frame.
INCOMPLETE = "incomplete"
EAGER = "eager"  # INCOMPLETE, with an eager end of turn to collect
COMPLETE = "complete"
COMPLETE_AND_EAGER = "complete+eager"  # COMPLETE, with a stale eager end of turn left behind


class ScriptedTurnAnalyzer(BaseTurnAnalyzer):
    """Turn analyzer whose verdicts are scripted.

    ``append_audio`` pops the next verdict from ``script`` (INCOMPLETE once the
    script runs out). With ``eager_at_vad_stop``, ``analyze_end_of_turn`` reaches
    an eager end of turn instead, the way an analyzer deciding at the VAD stop
    would.
    """

    def __init__(self, script: list[str] | None = None, *, eager_at_vad_stop: bool = False):
        super().__init__()
        self.script = list(script or [])
        self.eager_at_vad_stop = eager_at_vad_stop
        self.cleared = 0
        self._eager = False

    @property
    def speech_triggered(self) -> bool:
        return True

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        verdict = self.script.pop(0) if self.script else INCOMPLETE
        if verdict in (EAGER, COMPLETE_AND_EAGER):
            self._eager = True
        if verdict in (COMPLETE, COMPLETE_AND_EAGER):
            return EndOfTurnState.COMPLETE
        return EndOfTurnState.INCOMPLETE

    def pop_eager(self) -> bool:
        eager, self._eager = self._eager, False
        return eager

    async def analyze_end_of_turn(self):
        if self.eager_at_vad_stop:
            self._eager = True
        return EndOfTurnState.INCOMPLETE, None

    def clear(self):
        self.cleared += 1


class PlainTurnAnalyzer(ScriptedTurnAnalyzer):
    """Scripted analyzer that leaves ``pop_eager`` to the base class."""

    pop_eager = BaseTurnAnalyzer.pop_eager


class Events:
    """Records what a stop strategy emits."""

    def __init__(self, strategy: TurnAnalyzerUserTurnStopStrategy):
        self.inferences = []  # the speculation each inference carries, None when committed
        self.cancels = 0
        self.stops = []

        @strategy.event_handler("on_user_turn_inference_triggered")
        async def on_inference(strategy, speculation):
            self.inferences.append(speculation)

        @strategy.event_handler("on_user_turn_speculation_cancelled")
        async def on_cancel(strategy):
            self.cancels += 1

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            self.stops.append(params)
            # The controller closes the turn on every stop strategy when one of
            # them ends it.
            await strategy.handle_user_turn_stopped()

    @property
    def speculations(self):
        return [s for s in self.inferences if s is not None]


def audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\0" * 640, sample_rate=16000, num_channels=1)


def interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text, "user", "2026-10-24T00:00:00Z")


def final(text: str, *, finalized: bool = False) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "2026-10-24T00:00:00Z", finalized=finalized)


class TestTurnAnalyzerEagerUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def make(
        self,
        script: list[str],
        *,
        analyzer_cls: type[ScriptedTurnAnalyzer] = ScriptedTurnAnalyzer,
        wait_for_transcript: bool = False,
        **kwargs,
    ):
        """Build a strategy around a scripted analyzer, with the user just fallen silent."""
        analyzer = analyzer_cls(script)
        strategy = TurnAnalyzerEagerUserTurnStopStrategy(
            turn_analyzer=analyzer, wait_for_transcript=wait_for_transcript, **kwargs
        )
        await strategy.setup(frame_processor_setup(self.task_manager))
        events = Events(strategy)
        await strategy.handle_user_turn_started()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        return strategy, analyzer, events

    async def test_eager_then_commit_keeps_the_speculation(self):
        strategy, _, events = await self.make([INCOMPLETE, EAGER, INCOMPLETE, COMPLETE])
        await strategy.process_frame(final("book a flight"))

        for _ in range(4):
            await strategy.process_frame(audio())

        # One inference, the speculative one; the commit only finalizes, so the
        # same turn is not answered twice.
        assert len(events.inferences) == 1
        assert events.inferences[0].text == "book a flight"
        assert len(events.stops) == 1
        assert events.stops[0].confirms_speculation
        assert events.cancels == 0
        assert not strategy.speculating

    async def test_eager_is_answered_once_per_turn(self):
        strategy, _, events = await self.make([EAGER, EAGER, INCOMPLETE])
        await strategy.process_frame(final("hello"))

        for _ in range(3):
            await strategy.process_frame(audio())

        assert len(events.speculations) == 1
        assert strategy.speculating

    async def test_resumed_speech_withdraws_the_speculation(self):
        strategy, analyzer, events = await self.make([EAGER, INCOMPLETE])
        await strategy.process_frame(final("i think"))
        await strategy.process_frame(audio())
        assert strategy.speculating

        await strategy.process_frame(VADUserStartedSpeakingFrame())

        assert events.cancels == 1
        assert not strategy.speculating
        assert not events.stops

        # The next silence can speculate again.
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        analyzer.script = [EAGER]
        await strategy.process_frame(audio())
        assert len(events.speculations) == 2

    async def test_timeout_withdraws_the_speculation(self):
        strategy, _, events = await self.make([EAGER], speculation_timeout=0.05)
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(audio())
        assert strategy.speculating

        await asyncio.sleep(0.15)

        assert events.cancels == 1
        assert not strategy.speculating
        assert not events.stops

    async def test_differing_transcript_withdraws_the_speculation(self):
        strategy, _, events = await self.make([EAGER, COMPLETE])
        await strategy.process_frame(interim("hello"))
        await strategy.process_frame(audio())
        # The speculation ran on the interim transcript, the latest text there was.
        assert events.inferences[-1].text == "hello"

        await strategy.process_frame(final("hello there"))
        await strategy.process_frame(audio())

        # Withdrawn, and the committed transcript answered with a fresh inference.
        assert events.cancels == 1
        assert events.inferences[-1] is None
        assert len(events.stops) == 1
        assert not events.stops[0].confirms_speculation

    async def test_formatting_differences_are_tolerated_by_default(self):
        strategy, _, events = await self.make([EAGER, COMPLETE])
        await strategy.process_frame(interim("book a flight to tokyo"))
        await strategy.process_frame(audio())
        await strategy.process_frame(final("Book a flight to Tokyo."))
        await strategy.process_frame(audio())

        assert events.cancels == 0
        assert len(events.inferences) == 1
        assert events.stops[0].confirms_speculation

    async def test_exact_match_policy_withdraws_on_formatting_differences(self):
        strategy, _, events = await self.make([EAGER, COMPLETE], match_policy=ExactMatch())
        await strategy.process_frame(interim("book a flight to tokyo"))
        await strategy.process_frame(audio())
        await strategy.process_frame(final("Book a flight to Tokyo."))
        await strategy.process_frame(audio())

        assert events.cancels == 1
        assert not events.stops[0].confirms_speculation

    async def test_waiting_for_the_transcript_still_keeps_the_speculation(self):
        strategy, _, events = await self.make([EAGER, COMPLETE], wait_for_transcript=True)
        await strategy.process_frame(interim("hello"))
        await strategy.process_frame(audio())
        await strategy.process_frame(audio())
        # Committed, but the turn waits for the finalized transcript.
        assert not events.stops
        assert strategy.speculating

        await strategy.process_frame(final("hello", finalized=True))

        assert len(events.stops) == 1
        assert events.stops[0].confirms_speculation
        assert len(events.inferences) == 1

    async def test_unresolved_turn_end_withdraws_the_speculation(self):
        strategy, analyzer, events = await self.make([EAGER])
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(audio())
        assert strategy.speculating

        # The stop watchdog or an interruption ended the turn.
        await strategy.handle_user_turn_stopped()

        assert events.cancels == 1
        assert not strategy.speculating
        assert analyzer.cleared == 1

    async def test_eager_before_a_transcript_waits_for_one(self):
        strategy, _, events = await self.make([EAGER])
        await strategy.process_frame(audio())
        # Nothing to run a speculation on yet.
        assert not events.inferences
        assert not strategy.speculating

        await strategy.process_frame(interim("hello"))

        assert len(events.speculations) == 1
        assert events.speculations[0].text == "hello"

    async def test_held_eager_is_dropped_when_the_user_resumes(self):
        strategy, _, events = await self.make([EAGER])
        await strategy.process_frame(audio())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("hello"))

        assert not events.inferences
        assert events.cancels == 0

    async def test_held_eager_is_dropped_by_the_commit(self):
        strategy, _, events = await self.make([EAGER, COMPLETE])
        await strategy.process_frame(audio())
        await strategy.process_frame(audio())
        await strategy.process_frame(final("hello"))

        # The turn was answered by the commit, not speculatively.
        assert events.inferences == [None]
        assert not events.stops[0].confirms_speculation
        assert not strategy.speculating

    async def test_eager_reached_at_the_vad_stop_is_collected(self):
        analyzer = ScriptedTurnAnalyzer(eager_at_vad_stop=True)
        strategy = TurnAnalyzerEagerUserTurnStopStrategy(
            turn_analyzer=analyzer, wait_for_transcript=False
        )
        await strategy.setup(frame_processor_setup(self.task_manager))
        events = Events(strategy)
        await strategy.handle_user_turn_started()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(final("hello"))

        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))

        # No audio frame was needed to pick it up.
        assert len(events.speculations) == 1
        assert strategy.speculating

    async def test_eager_left_behind_by_the_commit_is_ignored(self):
        strategy, _, events = await self.make([COMPLETE_AND_EAGER])
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(audio())

        # The commit closed the turn; the eager end of turn collected on the
        # same frame belongs to it and must not speculate on the next turn.
        assert events.inferences == [None]
        assert len(events.stops) == 1
        assert not strategy.speculating
        assert events.cancels == 0

    async def test_no_speculation_while_the_user_is_speaking(self):
        strategy, _, events = await self.make([EAGER])
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(audio())

        assert not events.inferences

    async def test_analyzer_without_eager_ends_of_turn_behaves_as_before(self):
        strategy, _, events = await self.make(
            [INCOMPLETE, COMPLETE], analyzer_cls=PlainTurnAnalyzer
        )
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(audio())
        await strategy.process_frame(audio())

        assert events.inferences == [None]
        assert len(events.stops) == 1
        assert not events.stops[0].confirms_speculation
        assert events.cancels == 0

    async def test_cleanup_stops_the_speculation_clock(self):
        strategy, _, events = await self.make([EAGER], speculation_timeout=0.05)
        await strategy.process_frame(final("hello"))
        await strategy.process_frame(audio())

        await strategy.cleanup()
        await asyncio.sleep(0.15)

        # Cleanup is the end of the session: nothing is left to withdraw from.
        assert events.cancels == 0


class TestBaseTurnAnalyzerPopEager(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_to_no_eager_end_of_turn(self):
        analyzer = PlainTurnAnalyzer()
        assert analyzer.pop_eager() is False


class TestTurnAnalyzerEagerThroughTheAggregator(unittest.IsolatedAsyncioTestCase):
    """The strategy installed in a user aggregator, end to end."""

    def aggregator(self, context: LLMContext, script: list[str]) -> LLMUserAggregator:
        strategy = TurnAnalyzerEagerUserTurnStopStrategy(
            turn_analyzer=ScriptedTurnAnalyzer(script), wait_for_transcript=False
        )
        return LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(stop=[strategy]),
                user_turn_stop_timeout=5.0,
            ),
        )

    async def test_eager_end_of_turn_runs_a_speculative_inference_the_commit_confirms(self):
        context = LLMContext()
        context.set_messages([{"role": "system", "content": "Be brief."}])

        down, _ = await run_test(
            self.aggregator(context, [EAGER, COMPLETE]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                SleepFrame(),
                final("book a flight"),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
                SleepFrame(),
                audio(),  # the analyzer reaches an eager end of turn
                SleepFrame(),
                audio(),  # ... and commits
                SleepFrame(sleep=0.5),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        stops = [f for f in down if isinstance(f, UserStoppedSpeakingFrame)]

        # One inference, speculative, against a provisional context carrying
        # the transcript; the confirmed turn is not answered a second time.
        assert len(contexts) == 1
        assert contexts[0].speculation
        assert contexts[0].context is not context
        assert contexts[0].context.messages[-1] == {"role": "user", "content": "book a flight"}
        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in down)
        # The turn end is what releases the held response.
        assert len(stops) == 1
        assert context.messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "book a flight"},
        ]

    async def test_differing_committed_transcript_withdraws_and_answers_again(self):
        context = LLMContext()

        down, _ = await run_test(
            self.aggregator(context, [EAGER, COMPLETE]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                SleepFrame(),
                interim("i want to cancel"),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(stop_secs=0.2),
                SleepFrame(),
                audio(),  # eager, on the interim transcript
                SleepFrame(),
                final("i want to cancel, actually reschedule it"),
                SleepFrame(),
                audio(),  # commit
                SleepFrame(sleep=0.5),
            ],
        )

        contexts = [f for f in down if isinstance(f, LLMContextFrame)]
        cancels = [f for f in down if isinstance(f, EagerEndOfTurnCancelFrame)]

        # The speculative inference, then a second one on the committed transcript.
        assert [c.speculation for c in contexts] == [True, False]
        assert len(cancels) == 1
        assert context.messages == [
            {"role": "user", "content": "i want to cancel, actually reschedule it"}
        ]
