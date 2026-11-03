#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceResult, ClassifierError
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ClassifierUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_stop.classifier_user_turn_stop_strategy import TURN_COMPLETION_QUESTION
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ClassifierUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

SETTLE = 0.1


class _FakeClassifier(BaseClassifier):
    def __init__(self, *answers, delay: float = 0.0):
        super().__init__()
        self.answers = list(answers)
        self.delay = delay
        self.asked = []
        self.set_up_with = None
        self.cleaned_up = False

    async def setup(self, task_manager):
        await super().setup(task_manager)
        self.set_up_with = task_manager

    async def cleanup(self):
        await super().cleanup()
        self.cleaned_up = True

    async def _ask(self, state, questions, images=None):
        self.asked.append(state)
        if self.delay:
            await asyncio.sleep(self.delay)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        results = {}
        for name, question in questions.items():
            results[name] = ChoiceResult(
                choice=answer,
                probabilities={o: float(o == answer) for o in question.options},
                confidence=0.9,
            )
        return results, None


class _VADStopStrategy(BaseUserTurnStopStrategy):
    """Decides the user is done as soon as VAD stops, like a quick smart turn."""

    async def process_frame(self, frame: Frame):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.report_user_turn_stopped()


def _said(text: str, finalized: bool = False) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="", finalized=finalized)


def _heard(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=text, user_id="", timestamp="")


class TestClassifierUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.stopped = []

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def _start(self, *answers, context=None, delay=0.0, stt_latency=0.0, **kwargs):
        self.classifier = _FakeClassifier(*answers, delay=delay)
        self.strategy = ClassifierUserTurnStopStrategy(
            classifier=self.classifier, context=context, **kwargs
        )

        @self.strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.stopped.append(params)

        await self.strategy.setup(frame_processor_setup(self.task_manager))
        await self.strategy.process_frame(
            STTMetadataFrame(service_name="stt", ttfs_p99_latency=stt_latency)
        )
        await self.strategy.handle_user_turn_started()

    async def _say(self, *texts):
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        for text in texts:
            await self.strategy.process_frame(_said(text))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)

    async def test_complete_reports_that_the_user_is_done(self):
        await self._start("complete")
        await self._say("I'd like to", "book a table.")

        self.assertEqual(self.classifier.asked, [{"user": "I'd like to book a table."}])
        self.assertEqual(len(self.stopped), 1)

    async def test_nothing_is_asked_while_the_user_speaks(self):
        await self._start("complete")
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await self.strategy.process_frame(_said("I'd like to"))
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.classifier.asked, [])

        await self.strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)
        self.assertEqual(len(self.classifier.asked), 1)

    async def test_a_transcription_after_the_pause_is_classified(self):
        await self._start("complete")
        await self._say()
        self.assertEqual(self.classifier.asked, [])

        await self.strategy.process_frame(_said("book a table."))
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.classifier.asked, [{"user": "book a table."}])
        self.assertEqual(len(self.stopped), 1)

    async def test_a_pause_waits_for_the_stt_latency(self):
        await self._start("complete", stt_latency=0.3)
        await self._say("book a table.")
        self.assertEqual(self.classifier.asked, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)

    async def test_the_stt_latency_counts_from_when_the_user_stopped(self):
        await self._start("complete", stt_latency=0.3)
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await self.strategy.process_frame(_said("book a table."))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))
        await asyncio.sleep(SETTLE)

        self.assertEqual(len(self.stopped), 1)

    async def test_a_finalized_transcription_does_not_wait_for_the_stt_latency(self):
        await self._start("complete", stt_latency=1.0)
        await self._say()
        await self.strategy.process_frame(_said("book a table.", finalized=True))
        await asyncio.sleep(SETTLE)

        self.assertEqual(len(self.stopped), 1)

    async def test_speaking_again_waits_for_the_new_words(self):
        """The new words' transcription usually lands after the next pause."""
        await self._start("short", "complete", stt_latency=1.0, short_timeout=5.0)
        await self._say()
        await self.strategy.process_frame(_said("I'd like to", finalized=True))
        await asyncio.sleep(SETTLE)
        await self._say()
        self.assertEqual(len(self.classifier.asked), 1)

        await self.strategy.process_frame(_said("book a table.", finalized=True))
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.classifier.asked[1], {"user": "I'd like to book a table."})
        self.assertEqual(len(self.stopped), 1)

    async def test_a_transcription_finalized_before_the_stop_does_not_wait(self):
        """An STT can finalize the user's words before VAD reports the stop."""
        await self._start("complete", stt_latency=1.0)
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await self.strategy.process_frame(_said("book a table.", finalized=True))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)

        self.assertEqual(len(self.stopped), 1)

    async def test_an_interim_transcription_holds_the_pause(self):
        """An interim transcription means more words are on their way."""
        await self._start("short", "complete", stt_latency=1.0, short_timeout=5.0)
        await self._say()
        await self.strategy.process_frame(_said("book a", finalized=True))
        await self.strategy.process_frame(_heard("table"))
        await self.strategy.process_frame(_said("table"))
        await asyncio.sleep(SETTLE)

        self.assertEqual(self.classifier.asked, [{"user": "book a"}])

    async def test_a_proposed_stop_does_not_wait_for_the_stt_latency(self):
        """A service proposes the stop after pushing the transcription."""
        await self._start("complete", stt_latency=1.0)
        await self.strategy.process_frame(ProposedUserStartedSpeakingFrame())
        await self.strategy.process_frame(_said("book a table."))
        await self.strategy.process_frame(ProposedUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)

        self.assertEqual(len(self.stopped), 1)

    async def test_the_same_words_are_asked_about_once(self):
        await self._start("complete")
        await self._say("book a table.")
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)

        self.assertEqual(len(self.classifier.asked), 1)

    async def test_speech_without_new_words_asks_again(self):
        """The controller clears the stop decisions on any speech, such as a cough."""
        await self._start("complete", "complete")
        await self._say("book a table.")
        await self._say()

        self.assertEqual(len(self.classifier.asked), 2)
        self.assertEqual(len(self.stopped), 2)

    async def test_short_waits_before_reporting(self):
        await self._start("short", short_timeout=0.3)
        await self._say("I'd like to")
        self.assertEqual(self.stopped, [])
        self.assertTrue(self.strategy.decision_scheduled)

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)
        self.assertFalse(self.strategy.decision_scheduled)

    async def test_long_waits_before_reporting(self):
        await self._start("long", long_timeout=0.3)
        await self._say("let me think")
        self.assertEqual(self.stopped, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)

    async def test_speaking_again_ends_the_wait_and_classifies_the_whole_turn(self):
        await self._start("short", "complete", short_timeout=0.3)
        await self._say("I'd like to")
        await self._say("book a table.")
        await asyncio.sleep(0.3)

        self.assertEqual(self.classifier.asked[1], {"user": "I'd like to book a table."})
        self.assertEqual(len(self.stopped), 1)

    async def test_speaking_again_drops_a_verdict_on_its_way(self):
        await self._start("complete", "short", delay=0.2, short_timeout=5.0)
        await self.strategy.process_frame(_said("I'd like to"))
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.3)

        self.assertEqual(self.stopped, [])

    async def test_a_classifier_error_leaves_the_turn_to_the_others(self):
        await self._start(ClassifierError("down"))
        await self._say("hello")

        self.assertEqual(len(self.stopped), 1)

    async def test_a_slow_classifier_leaves_the_turn_to_the_others(self):
        await self._start("short", delay=0.5, classification_timeout=0.1)
        await self._say("hello")
        await asyncio.sleep(0.1)

        self.assertEqual(len(self.stopped), 1)

    async def test_nothing_is_asked_outside_a_turn(self):
        await self._start("complete")
        await self.strategy.handle_user_turn_stopped()
        await self._say("hello")

        self.assertEqual(self.classifier.asked, [])

    async def test_context_adds_the_assistants_last_message(self):
        context = LLMContext(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "assistant", "content": "Where would you go?"},
                {"role": "user", "content": "hmm"},
            ]
        )
        await self._start("long", context=context, long_timeout=5.0)
        await self._say("that's interesting")

        self.assertEqual(
            self.classifier.asked,
            [{"assistant": "Where would you go?", "user": "that's interesting"}],
        )

    async def test_each_classification_pushes_its_metrics(self):
        await self._start("complete")
        pushed = []

        @self.strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            pushed.append(frame)

        await self._say("book a table.")

        metrics = [f for f in pushed if isinstance(f, MetricsFrame)]
        self.assertEqual(len(metrics), 1)
        (processing,) = metrics[0].data
        self.assertIsInstance(processing, ProcessingMetricsData)
        self.assertEqual(processing.processor, self.classifier.name)

    async def test_the_classifier_is_set_up_and_cleaned_up(self):
        await self._start("complete")

        self.assertIs(self.classifier.set_up_with, self.task_manager)
        await self.strategy.cleanup()
        self.assertTrue(self.classifier.cleaned_up)


class TestClassifierInTheTurnController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.stops = []

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _start(self, *answers, vad_strategy=True, **kwargs):
        classifier = _FakeClassifier(*answers)
        stop: list[BaseUserTurnStopStrategy] = [_VADStopStrategy()] if vad_strategy else []
        stop.append(ClassifierUserTurnStopStrategy(classifier=classifier, **kwargs))
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy()], stop=stop),
            user_turn_stop_timeout=0.2,
        )
        await self.controller.setup(frame_processor_setup(self.task_manager))
        await self.controller.start()

        @self.controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            self.stops.append(strategy)

    async def _say(self, text: str):
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await self.controller.process_frame(_said(text))
        await self.controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(SETTLE)

    async def test_the_classifier_alone_ends_a_complete_turn(self):
        await self._start("complete", vad_strategy=False)
        await self._say("I think I would like to see Kyoto.")

        self.assertEqual(len(self.stops), 1)

    async def test_the_classifier_alone_holds_an_unfinished_turn(self):
        await self._start("short", vad_strategy=False, short_timeout=0.5)
        await self._say("I'd like to")
        await asyncio.sleep(0.3)
        self.assertEqual(self.stops, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stops), 1)

    async def test_the_turn_ends_once_the_vad_strategy_and_the_classifier_decide(self):
        await self._start("complete")
        await self._say("book a table.")

        self.assertEqual(len(self.stops), 1)
        self.assertIsInstance(self.stops[0], ClassifierUserTurnStopStrategy)

    async def test_short_keeps_the_turn_open_past_the_vad_strategy_and_the_stop_timeout(self):
        await self._start("short", short_timeout=0.5)
        await self._say("I'd like to")
        await asyncio.sleep(0.3)
        self.assertEqual(self.stops, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stops), 1)
        self.assertIsInstance(self.stops[0], ClassifierUserTurnStopStrategy)


class TestClassifierUserTurnStrategies(unittest.TestCase):
    def test_adds_the_classifier_strategy_after_the_default_stop_strategies(self):
        classifier = _FakeClassifier()
        strategies = ClassifierUserTurnStrategies(classifier=classifier)

        self.assertEqual(
            [type(s) for s in strategies.stop or []],
            [TurnAnalyzerUserTurnStopStrategy, ClassifierUserTurnStopStrategy],
        )

    def test_adds_the_classifier_strategy_to_the_stop_strategies_given(self):
        classifier = _FakeClassifier()
        vad_strategy = _VADStopStrategy()
        strategies = ClassifierUserTurnStrategies(classifier=classifier, stop=[vad_strategy])

        self.assertIs(strategies.stop[0], vad_strategy)
        self.assertIsInstance(strategies.stop[1], ClassifierUserTurnStopStrategy)
        self.assertIs(strategies.stop[1].classifier, classifier)

    def test_extending_a_service_adds_only_the_classifier_and_given_strategies(self):
        classifier = _FakeClassifier()
        strategies = ClassifierUserTurnStrategies(classifier=classifier)
        extension = strategies._stop_extension or []

        self.assertEqual(len(extension), 1)
        self.assertIsInstance(extension[0], ClassifierUserTurnStopStrategy)

    def test_needs_a_classifier(self):
        with self.assertRaises(ValueError):
            ClassifierUserTurnStrategies(stop=[_VADStopStrategy()])

    def test_the_question_has_the_three_verdicts(self):
        self.assertEqual(set(TURN_COMPLETION_QUESTION.options), {"complete", "short", "long"})


if __name__ == "__main__":
    unittest.main()
