#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserTurnCompletionFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import (
    LLMTurnCompletionUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    FilterIncompleteUserTurnStrategies,
    MergeMode,
    StrategyMode,
    UserTurnStrategiesParams,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

SPEECH_TIMEOUT = 0.1


class TestLLMTurnCompletionUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    """Decides on the marker of the inference that answers the turn."""

    async def asyncSetUp(self):
        self.strategy = LLMTurnCompletionUserTurnStopStrategy(
            config=UserTurnCompletionConfig(incomplete_short_timeout=0.05)
        )
        await self.strategy.setup(frame_processor_setup(TaskManager()))
        self.decisions = 0
        self.withdrawals = 0
        self.pushed = []

        @self.strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.decisions += 1

        @self.strategy.event_handler("on_user_turn_speculation_cancelled")
        async def on_user_turn_speculation_cancelled(strategy):
            self.withdrawals += 1

        @self.strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            self.pushed.append(frame)

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def _infer(self, *, speculation: bool = False) -> LLMContextFrame:
        frame = LLMContextFrame(context=LLMContext(), speculation=speculation)
        await self.strategy.process_frame(frame)
        return frame

    async def _report(self, kind: str | None, inference: LLMContextFrame):
        await self.strategy.process_frame(
            UserTurnCompletionFrame(kind=kind, context_frame_id=inference.id)
        )

    async def test_it_turns_on_markers_in_the_llm_at_start(self):
        await self.strategy.process_frame(StartFrame())

        (update,) = self.pushed
        self.assertTrue(update.delta.filter_incomplete_user_turns)
        self.assertIs(update.delta.user_turn_completion_config, self.strategy.config)
        self.assertTrue(update.reach_inactive_services)

    async def test_each_marker_gives_its_decision(self):
        for kind, decisions in [("complete", 1), ("short", 0), ("long", 0), (None, 1)]:
            with self.subTest(kind=kind):
                self.decisions = 0
                await self._report(kind, await self._infer())
                self.assertEqual(self.decisions, decisions)
                await self.strategy.handle_user_turn_started()

    async def test_a_report_for_another_inference_is_ignored(self):
        earlier = await self._infer()
        await self._infer()

        await self._report("complete", earlier)

        self.assertEqual(self.decisions, 0)

    async def test_speaking_again_waits_for_the_next_inference(self):
        inference = await self._infer()
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())

        await self._report("complete", inference)

        self.assertEqual(self.decisions, 0)

    async def test_a_cancelled_answer_gets_no_decision(self):
        inference = await self._infer()
        await self.strategy.process_frame(UserTurnResponseCancelFrame())

        await self._report("complete", inference)

        self.assertEqual(self.decisions, 0)

    async def test_a_complete_speculative_answer_gets_a_decision(self):
        await self._report("complete", await self._infer(speculation=True))

        self.assertEqual(self.decisions, 1)

    async def test_an_incomplete_speculative_answer_is_withdrawn(self):
        """The committed turn brings an inference of its own, so nothing is re-prompted."""
        await self._report("short", await self._infer(speculation=True))
        await asyncio.sleep(0.15)

        self.assertEqual(self.withdrawals, 1)
        self.assertEqual(self.decisions, 0)
        self.assertEqual(self.pushed, [])

    async def test_a_quiet_user_is_re_prompted_after_an_incomplete_turn(self):
        await self._report("short", await self._infer())
        await asyncio.sleep(0.15)

        self.assertEqual(self.decisions, 1)
        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)
        self.assertTrue(reprompt.run_llm)
        self.assertEqual(reprompt.messages[0]["content"], self.strategy.config.short_prompt)

    async def test_speaking_again_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.15)

        self.assertEqual(self.decisions, 0)
        self.assertEqual(self.pushed, [])

    async def test_a_turn_ended_by_the_stop_timeout_is_still_re_prompted(self):
        await self._report("short", await self._infer())
        await self.strategy.handle_user_turn_stopped()
        await asyncio.sleep(0.15)

        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)

    async def test_another_inference_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self._infer()
        await asyncio.sleep(0.15)

        self.assertEqual(self.pushed, [])

    async def test_a_report_from_another_inference_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self._report(None, LLMContextFrame(context=LLMContext()))
        await asyncio.sleep(0.15)

        self.assertEqual(self.pushed, [])

    async def test_a_later_complete_marker_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self._report("complete", await self._infer())
        await asyncio.sleep(0.15)

        self.assertEqual(self.decisions, 1)
        self.assertEqual(self.pushed, [])


class ScriptedMarkerLLM(LLMService):
    """Answers each inference with the next scripted reply, read through the marker mixin."""

    def __init__(self, replies: list[str], *, delays: list[float] | None = None, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self._replies = list(replies)
        self._delays = list(delays or [])
        self.inferences = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        self.inferences += 1
        reply = self._replies.pop(0)
        await asyncio.sleep(self._delays.pop(0) if self._delays else 0)
        await self.push_frame(LLMFullResponseStartFrame())
        await self._push_llm_text(reply)
        await self.push_frame(LLMFullResponseEndFrame())


class OutputRecorder(FrameProcessor):
    """Records the spoken text and LLMMarkerResponseFrame kinds that reach the output."""

    def __init__(self):
        super().__init__()
        self.spoken: list[str] = []
        self.marker_kinds: list[str | None] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.spoken.append(frame.text)
        elif isinstance(frame, LLMMarkerResponseFrame):
            self.marker_kinds.append(frame.kind)
        await self.push_frame(frame, direction)


class TestFilteringIncompleteTurns(unittest.IsolatedAsyncioTestCase):
    """Filtering incomplete turns end to end: aggregators, a marker-reading LLM, the output."""

    async def _run(
        self,
        replies: list[str],
        frames_to_send: list,
        *,
        stop_mode: StrategyMode = StrategyMode.ALL,
        config: UserTurnCompletionConfig | None = None,
    ) -> tuple[OutputRecorder, LLMContext]:
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=FilterIncompleteUserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)],
                    config=config,
                ),
                user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=stop_mode),
            ),
        )
        recorder = OutputRecorder()

        await run_test(
            Pipeline(
                [
                    user_aggregator,
                    ScriptedMarkerLLM(replies),
                    recorder,
                    LLMAssistantAggregator(context),
                ]
            ),
            frames_to_send=frames_to_send,
        )
        return recorder, context

    def _utterance(self, text: str) -> list:
        return [
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame(text=text, user_id="", timestamp="now"),
            SleepFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=SPEECH_TIMEOUT + 0.2),
        ]

    async def test_a_complete_turn_is_answered(self):
        recorder, context = await self._run(
            ["● Sure, Boston it is."], self._utterance("Book a flight to Boston")
        )

        self.assertEqual(recorder.spoken, ["Sure, Boston it is."])
        self.assertEqual([m["role"] for m in context.get_messages()], ["user", "assistant"])
        self.assertTrue(context.get_messages()[1]["content"].startswith("●"))

    async def test_an_incomplete_turn_waits_for_the_rest(self):
        recorder, context = await self._run(
            ["◐", "● Sure, Boston it is."],
            self._utterance("Book a flight") + self._utterance("to Boston"),
        )

        self.assertEqual(recorder.spoken, ["Sure, Boston it is."])
        messages = context.get_messages()
        self.assertEqual(
            [(m["role"], m["content"][:1]) for m in messages],
            [("user", "B"), ("assistant", "◐"), ("user", "t"), ("assistant", "●")],
        )

    async def test_a_quiet_user_is_re_prompted_in_either_stop_mode(self):
        for stop_mode in (StrategyMode.ALL, StrategyMode.FIRST):
            with self.subTest(stop_mode=stop_mode):
                recorder, _ = await self._run(
                    ["◐", "● Are you still there?"],
                    self._utterance("Book a flight") + [SleepFrame(sleep=0.5)],
                    stop_mode=stop_mode,
                    config=UserTurnCompletionConfig(incomplete_short_timeout=0.2),
                )

                self.assertEqual(recorder.spoken, ["Are you still there?"])

    async def test_words_after_the_inference_are_answered_without_a_re_prompt(self):
        recorder, _ = await self._run(
            ["◐", "● Sure, Boston it is.", "● Are you still there?"],
            [
                VADUserStartedSpeakingFrame(),
                TranscriptionFrame(text="Book a flight", user_id="", timestamp="now"),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=SPEECH_TIMEOUT + 0.05),
                TranscriptionFrame(text="to Boston", user_id="", timestamp="now"),
                SleepFrame(sleep=0.6),
            ],
            config=UserTurnCompletionConfig(incomplete_short_timeout=0.3),
        )

        self.assertEqual(recorder.spoken, ["Sure, Boston it is."])

    async def test_every_response_reports_its_marker(self):
        recorder, _ = await self._run(
            ["◐", "● Sure, Boston it is."],
            self._utterance("Book a flight") + self._utterance("to Boston"),
        )

        self.assertEqual(recorder.marker_kinds, ["short", "complete"])


class TestFilteringIncompleteTurnsWithEagerEndOfTurn(unittest.IsolatedAsyncioTestCase):
    """The marker strategy decides on an eager answer while the turn commits, in either order."""

    async def _run(
        self, replies: list[str], frames_to_send: list, *, delays: list[float] | None = None
    ) -> tuple[OutputRecorder, LLMContext]:
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=FilterIncompleteUserTurnStrategies(),
                user_turn_strategies_params=UserTurnStrategiesParams(
                    start_merge_mode=MergeMode.EXTEND, stop_merge_mode=MergeMode.EXTEND
                ),
            ),
        )
        recorder = OutputRecorder()
        recommendation = STTMetadataFrame(
            service_name="flux",
            ttfs_p99_latency=0.0,
            user_turn_strategies=EagerUserTurnStrategies(),
        )
        self.llm = ScriptedMarkerLLM(replies, delays=delays)

        await run_test(
            Pipeline(
                [
                    user_aggregator,
                    self.llm,
                    recorder,
                    LLMAssistantAggregator(context),
                ]
            ),
            frames_to_send=[recommendation, SleepFrame(), *frames_to_send],
        )
        return recorder, context

    def _turn(self, text: str) -> list:
        return [
            ProposedUserStartedSpeakingFrame(),
            SleepFrame(),
            EagerTranscriptionFrame(text, "user", "now"),
            SleepFrame(sleep=0.1),
            TranscriptionFrame(text, "user", "now"),
            SleepFrame(),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.8),
        ]

    def _assert_answered_once(self, recorder: OutputRecorder, context: LLMContext):
        self.assertEqual(recorder.spoken, ["Sure, Boston it is."])
        self.assertEqual(
            [(m["role"], m["content"][:1]) for m in context.get_messages()],
            [("user", "B"), ("assistant", "●")],
        )

    async def test_a_complete_eager_answer_is_spoken_once_the_turn_commits(self):
        recorder, context = await self._run(
            ["● Sure, Boston it is."], self._turn("Book a flight to Boston")
        )

        self._assert_answered_once(recorder, context)

    async def test_a_slow_complete_eager_answer_is_generated_once(self):
        # The external stop strategy asks for the stop again while the turn
        # waits for the marker, and the eager strategy confirms it again.
        recorder, context = await self._run(
            ["● Sure, Boston it is."], self._turn("Book a flight to Boston"), delays=[0.8]
        )

        self._assert_answered_once(recorder, context)
        self.assertEqual(self.llm.inferences, 1)

    async def test_an_incomplete_eager_answer_is_dropped_and_the_committed_turn_answered(self):
        recorder, context = await self._run(
            ["◐", "● Sure, Boston it is."], self._turn("Book a flight to Boston")
        )

        self._assert_answered_once(recorder, context)

    async def test_an_eager_answer_marked_incomplete_after_the_commit_is_answered_again(self):
        recorder, context = await self._run(
            ["◐", "● Sure, Boston it is."],
            self._turn("Book a flight to Boston"),
            delays=[0.45, 0],
        )

        self._assert_answered_once(recorder, context)

    async def test_an_eager_answer_marked_incomplete_as_the_turn_commits_is_answered_again(
        self,
    ):
        # The marker arrives as the turn commits, so the speculation is
        # withdrawn while the eager strategy confirms it.
        recorder, context = await self._run(
            ["◐", "● Sure, Boston it is."],
            self._turn("Book a flight to Boston"),
            delays=[0.3, 0],
        )

        self._assert_answered_once(recorder, context)


if __name__ == "__main__":
    unittest.main()
