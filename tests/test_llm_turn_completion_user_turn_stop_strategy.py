#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserTurnCompletionFrame,
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
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

SPEECH_TIMEOUT = 0.1


class TestLLMTurnCompletionUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    """Votes on the marker of the inference that answers the turn."""

    async def asyncSetUp(self):
        self.strategy = LLMTurnCompletionUserTurnStopStrategy(
            config=UserTurnCompletionConfig(incomplete_short_timeout=0.05)
        )
        await self.strategy.setup(frame_processor_setup(TaskManager()))
        self.votes = 0
        self.pushed = []

        @self.strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.votes += 1

        @self.strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            self.pushed.append(frame)

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def _infer(self) -> LLMContextFrame:
        frame = LLMContextFrame(context=LLMContext())
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

    async def test_each_marker_gives_its_vote(self):
        for kind, votes in [("complete", 1), ("short", 0), ("long", 0), (None, 1)]:
            with self.subTest(kind=kind):
                self.votes = 0
                await self._report(kind, await self._infer())
                self.assertEqual(self.votes, votes)
                await self.strategy.handle_user_turn_started()

    async def test_a_report_for_another_inference_is_ignored(self):
        earlier = await self._infer()
        await self._infer()

        await self._report("complete", earlier)

        self.assertEqual(self.votes, 0)

    async def test_speaking_again_waits_for_the_next_inference(self):
        inference = await self._infer()
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())

        await self._report("complete", inference)

        self.assertEqual(self.votes, 0)

    async def test_a_quiet_user_is_re_prompted_after_an_incomplete_turn(self):
        await self._report("short", await self._infer())
        await asyncio.sleep(0.15)

        self.assertEqual(self.votes, 1)
        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)
        self.assertTrue(reprompt.run_llm)
        self.assertEqual(reprompt.messages[0]["content"], self.strategy.config.short_prompt)

    async def test_speaking_again_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.15)

        self.assertEqual(self.votes, 0)
        self.assertEqual(self.pushed, [])

    async def test_a_turn_ended_by_the_stop_timeout_is_still_re_prompted(self):
        await self._report("short", await self._infer())
        await self.strategy.handle_user_turn_stopped()
        await asyncio.sleep(0.15)

        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)

    async def test_a_later_complete_marker_cancels_the_re_prompt(self):
        await self._report("short", await self._infer())
        await self._report("complete", await self._infer())
        await asyncio.sleep(0.15)

        self.assertEqual(self.votes, 1)
        self.assertEqual(self.pushed, [])


class ScriptedMarkerLLM(LLMService):
    """Answers each inference with the next scripted reply, read through the marker mixin."""

    def __init__(self, replies: list[str], **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self._replies = list(replies)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self._push_llm_text(self._replies.pop(0))
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
        self, replies: list[str], frames_to_send: list
    ) -> tuple[OutputRecorder, LLMContext]:
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=FilterIncompleteUserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)]
                ),
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

    async def test_every_response_reports_its_marker(self):
        recorder, _ = await self._run(
            ["◐", "● Sure, Boston it is."],
            self._utterance("Book a flight") + self._utterance("to Boston"),
        )

        self.assertEqual(recorder.marker_kinds, ["short", "complete"])


if __name__ == "__main__":
    unittest.main()
