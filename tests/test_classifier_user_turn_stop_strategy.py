#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceResult, ClassifierError
from pipecat.frames.frames import Frame, MetricsFrame, TranscriptionFrame
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ClassifierUserTurnCompletionStopStrategy,
)
from pipecat.turns.user_stop.classifier_user_turn_completion_stop_strategy import (
    TURN_COMPLETION_QUESTION,
)
from pipecat.turns.user_turn_strategies import ClassifierUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

SETTLE = 0.1


class _Detector(BaseUserTurnStopStrategy):
    """A detector the test fires by hand."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []
        self.started = 0
        self.stopped = 0

    async def handle_user_turn_started(self):
        self.started += 1

    async def handle_user_turn_stopped(self):
        self.stopped += 1

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        self.frames.append(frame)
        return ProcessFrameResult.CONTINUE


class _FakeClassifier(BaseClassifier):
    def __init__(self, *answers, delay: float = 0.0):
        super().__init__()
        self.answers = list(answers)
        self.delay = delay
        self.asked = []
        self.setup_called = False
        self.cleaned_up = False

    async def setup(self, worker):
        self.setup_called = True

    async def cleanup(self):
        self.cleaned_up = True

    async def _ask(self, state, questions):
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


def _said(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="")


class TestClassifierUserTurnCompletionStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.inference = []
        self.stopped = []

    async def _strategy(self, *answers, context=None, **kwargs):
        self.detector = _Detector()
        self.classifier = _FakeClassifier(*answers, delay=kwargs.pop("delay", 0.0))
        strategy = ClassifierUserTurnCompletionStopStrategy(
            self.detector, classifier=self.classifier, context=context, **kwargs
        )

        @strategy.event_handler("on_user_turn_inference_triggered")
        async def _inference(s, speculation):
            self.inference.append(speculation)

        @strategy.event_handler("on_user_turn_stopped")
        async def _stopped(s, params):
            self.stopped.append(params)

        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.handle_user_turn_started()
        return strategy

    async def test_complete_ends_the_turn_with_the_text(self):
        strategy = await self._strategy("complete")
        await strategy.process_frame(_said("I'd like to"))
        await strategy.process_frame(_said("book a table."))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        self.assertEqual(self.classifier.asked, [{"user": "I'd like to book a table."}])
        self.assertEqual(len(self.inference), 1)
        self.assertEqual(len(self.stopped), 1)
        self.assertEqual(self.detector.frames[0].text, "I'd like to")

    async def test_each_classification_pushes_its_metrics(self):
        strategy = await self._strategy("complete")
        pushed = []

        @strategy.event_handler("on_push_frame")
        async def _pushed(s, frame, direction):
            pushed.append(frame)

        await strategy.process_frame(_said("book a table."))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        metrics = [f for f in pushed if isinstance(f, MetricsFrame)]
        self.assertEqual(len(metrics), 1)
        (processing,) = metrics[0].data
        self.assertIsInstance(processing, ProcessingMetricsData)
        self.assertEqual(processing.processor, self.classifier.name)

    async def test_short_holds_the_turn_until_the_user_continues(self):
        strategy = await self._strategy("short", "complete", short_timeout=5.0)
        await strategy.process_frame(_said("I'd like to"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.stopped, [])

        await strategy.process_frame(_said("book a table."))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        self.assertEqual(self.classifier.asked[1], {"user": "I'd like to book a table."})
        self.assertEqual(len(self.stopped), 1)

    async def test_short_timeout_ends_the_turn(self):
        strategy = await self._strategy("short", short_timeout=0.2)
        await strategy.process_frame(_said("I'd like to"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.stopped, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)

    async def test_long_timeout_ends_the_turn(self):
        strategy = await self._strategy("long", long_timeout=0.2)
        await strategy.process_frame(_said("let me think"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)
        self.assertEqual(self.stopped, [])

        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)

    async def test_new_turn_cancels_a_pending_timeout(self):
        strategy = await self._strategy("short", short_timeout=0.2)
        await strategy.process_frame(_said("I'd like to"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        await strategy.handle_user_turn_started()
        await asyncio.sleep(0.3)
        self.assertEqual(self.stopped, [])
        self.assertEqual(self.detector.started, 2)

    async def test_classifier_error_ends_the_turn(self):
        strategy = await self._strategy(ClassifierError("down"))
        await strategy.process_frame(_said("hello"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)
        self.assertEqual(len(self.stopped), 1)

    async def test_slow_classifier_ends_the_turn(self):
        strategy = await self._strategy("short", delay=0.5, classification_timeout=0.1)
        await strategy.process_frame(_said("hello"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(0.3)
        self.assertEqual(len(self.stopped), 1)

    async def test_context_adds_the_assistants_last_message(self):
        context = LLMContext(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "assistant", "content": "Where would you go?"},
                {"role": "user", "content": "hmm"},
            ]
        )
        strategy = await self._strategy("long", context=context, long_timeout=5.0)
        await strategy.process_frame(_said("that's interesting"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        self.assertEqual(
            self.classifier.asked,
            [{"assistant": "Where would you go?", "user": "that's interesting"}],
        )

    async def test_forwards_lifecycle_to_inner_and_classifier(self):
        strategy = await self._strategy("complete")
        await strategy.handle_user_turn_stopped()
        await strategy.cleanup()

        self.assertTrue(self.classifier.setup_called)
        self.assertTrue(self.classifier.cleaned_up)
        self.assertEqual(self.detector.started, 1)
        self.assertEqual(self.detector.stopped, 1)

    async def test_detector_events_do_not_reach_the_controller(self):
        strategy = await self._strategy("short", short_timeout=5.0)
        await strategy.process_frame(_said("I'd like to"))
        await self.detector.trigger_user_turn_stopped()
        await asyncio.sleep(SETTLE)

        self.assertEqual(self.inference, [])
        self.assertEqual(self.stopped, [])

    def test_options_cover_the_three_verdicts(self):
        self.assertEqual(set(TURN_COMPLETION_QUESTION.options), {"complete", "short", "long"})


class TestClassifierUserTurnStrategies(unittest.TestCase):
    def test_wraps_every_stop_strategy(self):
        classifier = _FakeClassifier()
        strategies = ClassifierUserTurnStrategies(
            classifier=classifier, stop=[_Detector(), _Detector()], short_timeout=1.0
        )
        self.assertEqual(len(strategies.stop), 2)
        for s in strategies.stop:
            self.assertIsInstance(s, ClassifierUserTurnCompletionStopStrategy)
            self.assertIs(s.classifier, classifier)

    def test_needs_a_classifier(self):
        with self.assertRaises(ValueError):
            ClassifierUserTurnStrategies(stop=[_Detector()])
