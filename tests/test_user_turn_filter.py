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
    UserStoppedSpeakingFrame,
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
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy, SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_filter import BaseUserTurnFilter, IncompleteUserTurnFilter
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

SPEECH_TIMEOUT = 0.1


class RecordingFilter(BaseUserTurnFilter):
    """Records what it is asked to judge; tests answer for it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proposals: list[int] = []
        self.frames = []
        self.turns: list[str] = []

    async def handle_user_turn_started(self):
        self.turns.append("started")

    async def handle_user_turn_stopped(self):
        self.turns.append("stopped")

    async def handle_turn_end_proposed(self, context_frame_id: int):
        self.proposals.append(context_frame_id)

    async def process_frame(self, frame):
        self.frames.append(frame)


class TestUserTurnFilterController(unittest.IsolatedAsyncioTestCase):
    """The controller ends a turn only once its filter accepts the proposal."""

    async def asyncSetUp(self):
        self.filter = RecordingFilter()
        self.stop = BaseUserTurnStopStrategy()
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()], stop=[self.stop]
            ),
            user_turn_stop_timeout=5.0,
            user_turn_filter=self.filter,
        )
        await self.controller.setup(frame_processor_setup(TaskManager()))

        self.stopped = 0
        self.rejected = 0
        # What the aggregator would generate for each inference; None runs nothing.
        self.next_inference: list[int | None] = []

        @self.controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            self.stopped += 1

        @self.controller.event_handler("on_user_turn_stop_rejected")
        async def on_user_turn_stop_rejected(controller):
            self.rejected += 1

        @self.controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            context_frame_id = self.next_inference.pop(0) if self.next_inference else None
            if context_frame_id is not None:
                await controller.handle_inference_started(context_frame_id)

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _speak(self):
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await self.controller.process_frame(VADUserStoppedSpeakingFrame())

    async def _propose(self, inference: int | None):
        self.next_inference.append(inference)
        await self.stop.trigger_user_turn_stopped()

    async def test_a_turn_end_waits_for_the_filter(self):
        await self._speak()
        await self._propose(101)
        self.assertEqual(self.filter.proposals, [101])
        self.assertEqual(self.stopped, 0)

        await self.filter.accept()
        self.assertEqual(self.stopped, 1)

    async def test_a_rejection_keeps_the_turn_open(self):
        await self._speak()
        await self._propose(101)
        await self.filter.reject()
        self.assertEqual((self.rejected, self.stopped), (1, 0))

        # The user goes on; the new inference is judged in turn.
        await self._speak()
        await self._propose(102)
        self.assertEqual(self.filter.proposals, [101, 102])
        await self.filter.accept()
        self.assertEqual(self.stopped, 1)

    async def test_proposing_again_without_a_new_inference_is_not_judged_again(self):
        await self._speak()
        await self._propose(101)
        await self.filter.reject()

        await self._propose(None)
        self.assertEqual(self.filter.proposals, [101])
        self.assertEqual(self.stopped, 0)

    async def test_a_turn_with_nothing_generated_ends_without_the_filter(self):
        await self._speak()
        await self._propose(None)
        self.assertEqual(self.filter.proposals, [])
        self.assertEqual(self.stopped, 1)

    async def test_an_accept_after_a_rejection_ends_the_turn(self):
        await self._speak()
        await self._propose(101)
        await self.filter.reject()
        await self.filter.accept()
        self.assertEqual(self.stopped, 1)

    async def test_a_filter_that_never_answers_leaves_the_turn_to_the_stop_timeout(self):
        self.controller._user_turn_stop_timeout = 0.05
        await self.controller.start()
        await self._speak()
        await self._propose(101)
        await asyncio.sleep(0.15)
        self.assertEqual(self.stopped, 1)

    async def test_the_filter_sees_frames_and_turn_boundaries(self):
        frame = TranscriptionFrame(text="hi", user_id="u", timestamp="")
        await self._speak()
        await self.controller.process_frame(frame)
        await self._propose(101)
        await self.filter.accept()

        self.assertIn(frame, self.filter.frames)
        self.assertEqual(self.filter.turns, ["started", "stopped"])


class ScriptedFilter(RecordingFilter):
    """Answers every proposal at once with the verdict it was given, if any."""

    def __init__(self, verdict: str | None, **kwargs):
        super().__init__(**kwargs)
        self._verdict = verdict

    async def handle_turn_end_proposed(self, context_frame_id: int):
        await super().handle_turn_end_proposed(context_frame_id)
        if self._verdict == "accept":
            await self.accept()
        elif self._verdict == "reject":
            await self.reject()


class TestUserAggregatorWithFilter(unittest.IsolatedAsyncioTestCase):
    """The user aggregator holds each turn's answer for the filter's verdict."""

    async def _run_turn(self, user_turn_filter: BaseUserTurnFilter | None):
        # Whether the turn ended is told by the UserStoppedSpeakingFrame: the
        # session ending also emits on_user_turn_stopped for an open turn.
        aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)]
                ),
                user_turn_filter=user_turn_filter,
            ),
        )
        down, _ = await run_test(
            aggregator,
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                TranscriptionFrame(text="Hi", user_id="", timestamp="now"),
                SleepFrame(),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=SPEECH_TIMEOUT + 0.1),
            ],
        )
        return down

    async def test_the_answer_is_held_and_judged_by_its_context_frame(self):
        user_turn_filter = ScriptedFilter(None)
        down = await self._run_turn(user_turn_filter)

        (context_frame,) = [f for f in down if isinstance(f, LLMContextFrame)]
        self.assertTrue(context_frame.held)
        self.assertEqual(user_turn_filter.proposals, [context_frame.id])
        self.assertFalse(any(isinstance(f, UserStoppedSpeakingFrame) for f in down))

    async def test_an_accepted_turn_ends(self):
        down = await self._run_turn(ScriptedFilter("accept"))

        self.assertTrue(any(isinstance(f, UserStoppedSpeakingFrame) for f in down))

    async def test_a_rejected_turn_drops_the_answer_and_stays_open(self):
        down = await self._run_turn(ScriptedFilter("reject"))

        self.assertTrue(any(isinstance(f, UserTurnResponseCancelFrame) for f in down))
        self.assertFalse(any(isinstance(f, UserStoppedSpeakingFrame) for f in down))

    async def test_without_a_filter_the_answer_is_not_held(self):
        down = await self._run_turn(None)

        (context_frame,) = [f for f in down if isinstance(f, LLMContextFrame)]
        self.assertFalse(context_frame.held)
        self.assertTrue(any(isinstance(f, UserStoppedSpeakingFrame) for f in down))


class TestIncompleteUserTurnFilter(unittest.IsolatedAsyncioTestCase):
    """Judges proposed turn ends by the LLM's turn-completion marker."""

    async def asyncSetUp(self):
        self.filter = IncompleteUserTurnFilter(
            config=UserTurnCompletionConfig(incomplete_short_timeout=0.05)
        )
        await self.filter.setup(frame_processor_setup(TaskManager()))
        self.verdicts: list[str] = []
        self.pushed = []

        @self.filter.event_handler("on_accept")
        async def on_accept(user_turn_filter):
            self.verdicts.append("accept")

        @self.filter.event_handler("on_reject")
        async def on_reject(user_turn_filter):
            self.verdicts.append("reject")

        @self.filter.event_handler("on_push_frame")
        async def on_push_frame(user_turn_filter, frame, direction):
            self.pushed.append(frame)

    async def asyncTearDown(self):
        await self.filter.cleanup()

    async def _report(self, kind: str | None, context_frame_id: int = 101):
        await self.filter.process_frame(
            UserTurnCompletionFrame(kind=kind, context_frame_id=context_frame_id)
        )

    async def test_it_turns_on_markers_in_the_llm_at_start(self):
        await self.filter.process_frame(StartFrame())

        (update,) = self.pushed
        self.assertTrue(update.delta.filter_incomplete_user_turns)
        self.assertIs(update.delta.user_turn_completion_config, self.filter.config)
        self.assertTrue(update.reach_inactive_services)

    async def test_each_marker_gives_its_verdict(self):
        for kind, verdict in [
            ("complete", "accept"),
            ("short", "reject"),
            ("long", "reject"),
            (None, "accept"),
        ]:
            with self.subTest(kind=kind):
                self.verdicts.clear()
                await self.filter.handle_turn_end_proposed(101)
                await self._report(kind)
                self.assertEqual(self.verdicts, [verdict])
                await self.filter.handle_user_turn_started()

    async def test_a_report_ahead_of_its_proposal_is_used(self):
        await self._report("complete")
        await self.filter.handle_turn_end_proposed(101)
        self.assertEqual(self.verdicts, ["accept"])

    async def test_a_report_for_another_inference_is_ignored(self):
        await self.filter.handle_turn_end_proposed(102)
        await self._report("complete", context_frame_id=101)
        self.assertEqual(self.verdicts, [])

    async def test_a_complete_marker_while_the_user_speaks_rejects(self):
        await self.filter.process_frame(VADUserStartedSpeakingFrame())
        await self.filter.handle_turn_end_proposed(101)
        await self._report("complete")
        self.assertEqual(self.verdicts, ["reject"])

    async def test_a_quiet_user_is_re_prompted_after_an_incomplete_turn(self):
        await self.filter.handle_turn_end_proposed(101)
        await self._report("short")
        await asyncio.sleep(0.15)

        self.assertEqual(self.verdicts, ["reject", "accept"])
        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)
        self.assertTrue(reprompt.run_llm)
        self.assertEqual(reprompt.messages[0]["content"], self.filter.config.short_prompt)

    async def test_speaking_again_cancels_the_re_prompt(self):
        await self.filter.handle_turn_end_proposed(101)
        await self._report("short")
        await self.filter.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.15)

        self.assertEqual(self.verdicts, ["reject"])
        self.assertEqual(self.pushed, [])

    async def test_a_turn_ended_by_the_stop_timeout_is_still_re_prompted(self):
        await self.filter.handle_turn_end_proposed(101)
        await self._report("short")
        await self.filter.handle_user_turn_stopped()
        await asyncio.sleep(0.15)

        (reprompt,) = self.pushed
        self.assertIsInstance(reprompt, LLMMessagesAppendFrame)

    async def test_a_later_complete_marker_cancels_the_re_prompt(self):
        await self.filter.handle_turn_end_proposed(101)
        await self._report("short")
        await self.filter.handle_turn_end_proposed(102)
        await self._report("complete", context_frame_id=102)
        await asyncio.sleep(0.15)

        self.assertEqual(self.verdicts, ["reject", "accept"])
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


class TestIncompleteUserTurnFilterPipeline(unittest.IsolatedAsyncioTestCase):
    """The marker filter end to end: aggregators, a marker-reading LLM, the output."""

    async def _run(
        self, replies: list[str], frames_to_send: list
    ) -> tuple[OutputRecorder, LLMContext]:
        context = LLMContext()
        user_aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)]
                ),
                user_turn_filter=IncompleteUserTurnFilter(),
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
