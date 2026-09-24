#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings

from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceResult, ClassifierError
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.extensions.voicemail.voicemail_detector import (
    VoicemailDetector,
)
from pipecat.frames.frames import (
    EndWorkerFrame,
    Frame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

# Long enough for a verdict to reach the delayed voicemail handler, which
# fires VOICEMAIL_DELAY after the verdict.
VERDICT_SETTLE = 1.0
VOICEMAIL_DELAY = 0.1
DECISION_TIMEOUT = 0.2


class _FakeClassifier(BaseClassifier):
    """Answers the voicemail question from a scripted list of answers."""

    def __init__(self, *answers: tuple[str, float] | Exception):
        super().__init__()
        self.answers = list(answers)
        self.asked: list = []
        self.setup_task_manager = None
        self.cleaned_up = False

    async def setup(self, task_manager):
        self.setup_task_manager = task_manager

    async def cleanup(self):
        self.cleaned_up = True

    async def _ask(self, state, questions):
        self.asked.append(state)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        label, confidence = answer
        results = {
            name: ChoiceResult(
                choice=label,
                probabilities={o: float(o == label) for o in q.options},
                confidence=confidence,
            )
            for name, q in questions.items()
        }
        return results, None


class _Passthrough(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _names(frames) -> list[str]:
    return [type(f).__name__ for f in frames]


def _said(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="")


def _detector(*answers) -> tuple[VoicemailDetector, _FakeClassifier]:
    classifier = _FakeClassifier(*answers)
    detector = VoicemailDetector(
        classifier=classifier,
        voicemail_response_delay=VOICEMAIL_DELAY,
        decision_timeout=DECISION_TIMEOUT,
    )
    return detector, classifier


class TestVoicemailDetectorVerdicts(unittest.IsolatedAsyncioTestCase):
    async def test_voicemail_verdict_fires_handler_with_the_transcript(self):
        detector, classifier = _detector(("voicemail", 0.95))
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[_said("Hi, you've reached Sam."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])
        self.assertEqual(classifier.asked, ["Hi, you've reached Sam."])

    async def test_each_classification_pushes_its_metrics(self):
        detector, _ = _detector(("conversation", 0.9))

        down, _ = await run_test(
            detector,
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )

        metrics = [f for f in down if isinstance(f, MetricsFrame)]
        self.assertEqual(len(metrics), 1)
        (processing,) = metrics[0].data
        self.assertIsInstance(processing, ProcessingMetricsData)
        self.assertEqual(processing.processor, detector._classifier.name)

    async def test_conversation_verdict_fires_handler(self):
        detector, _ = _detector(("conversation", 0.9))
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])

    async def test_a_later_answer_replaces_an_earlier_one(self):
        detector, classifier = _detector(("voicemail", 0.4), ("voicemail", 0.95))
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                _said("Hi,"),
                SleepFrame(0.2),
                _said("you've reached Sam. Leave a message."),
                UserStoppedSpeakingFrame(),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(len(fired), 1)
        self.assertEqual(classifier.asked, ["Hi,", "Hi, you've reached Sam. Leave a message."])

    async def test_classifier_error_does_not_decide(self):
        detector, classifier = _detector(ClassifierError("down"), ("conversation", 0.9))
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                _said("Hello?"),
                SleepFrame(0.2),
                _said("Anyone there?"),
                UserStoppedSpeakingFrame(),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(len(fired), 1)
        self.assertEqual(len(classifier.asked), 2)

    async def test_setup_and_cleanup_reach_the_classifier(self):
        detector, classifier = _detector(("conversation", 0.9))
        await run_test(detector, frames_to_send=[_said("Hello?")], start_timeout=5.0)
        self.assertIsNotNone(classifier.setup_task_manager)
        self.assertTrue(classifier.cleaned_up)


class TestVoicemailDetectorFallback(unittest.IsolatedAsyncioTestCase):
    """Silence decides with the latest answer; more speech restarts the wait."""

    async def test_silence_decides_with_the_best_answer_so_far(self):
        classifier = _FakeClassifier(("voicemail", 0.3))
        detector = VoicemailDetector(
            classifier=classifier,
            decision_timeout=0.2,
            voicemail_response_delay=VOICEMAIL_DELAY,
        )
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                _said("Hi, this is Sam."),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.5 + VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])

    async def test_more_speech_cancels_the_silence_decision(self):
        classifier = _FakeClassifier(("voicemail", 0.3), ("conversation", 0.3))
        detector = VoicemailDetector(classifier=classifier, decision_timeout=0.3)
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                _said("Hi, this is Sam."),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.1),
                UserStartedSpeakingFrame(),
                SleepFrame(0.4),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [])
        self.assertIsNone(detector._decision)

    async def test_a_voicemail_verdict_waits_for_the_caller_to_stop(self):
        # The caller is still speaking: "Hi, this is Sam" looks like a person,
        # then the greeting goes on and turns out to be a voicemail. Neither
        # verdict acts until the caller stops.
        answers = (("conversation", 0.99), ("voicemail", 0.98))
        detector, _ = _detector(*answers)
        fired = {"conversation": [], "voicemail": []}

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired["conversation"].append(processor)

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired["voicemail"].append(processor)

        speaking = [
            UserStartedSpeakingFrame(),
            _said("Hi, this is Sam."),
            SleepFrame(0.2),
            _said("Sorry I missed your call, leave a message."),
            SleepFrame(VERDICT_SETTLE),
        ]
        await run_test(detector, frames_to_send=speaking, start_timeout=5.0)
        self.assertEqual(fired, {"conversation": [], "voicemail": []})

        detector, _ = _detector(*answers)

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation_later(processor):
            fired["conversation"].append(processor)

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail_later(processor):
            fired["voicemail"].append(processor)

        await run_test(
            detector,
            frames_to_send=[*speaking, UserStoppedSpeakingFrame(), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertEqual(fired, {"conversation": [], "voicemail": [detector]})

    async def test_a_confident_conversation_decides_after_the_silence(self):
        classifier = _FakeClassifier(("conversation", 0.99))
        detector = VoicemailDetector(classifier=classifier, decision_timeout=0.4)
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                _said("Hello?"),
                SleepFrame(0.1),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.2),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [], "a conversation verdict waits out the silence")

        fired.clear()
        classifier = _FakeClassifier(("conversation", 0.99))
        detector = VoicemailDetector(classifier=classifier, decision_timeout=0.2)

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation_later(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                _said("Hello?"),
                SleepFrame(0.1),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.5),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])

    async def test_silence_without_any_answer_assumes_a_conversation(self):
        classifier = _FakeClassifier(ClassifierError("down"))
        detector = VoicemailDetector(classifier=classifier, decision_timeout=0.2)
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                _said("Hello?"),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.5 + VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])


class TestVoicemailDetectorGating(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_releases_held_speech(self):
        detector, _ = _detector(("conversation", 0.9))
        down, _ = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSStartedFrame(),
                TTSTextFrame("Hi, this is Alex.", aggregated_by=AggregationType.SENTENCE),
                SleepFrame(0.2),
                _said("Hello?"),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        names = _names(down)
        self.assertIn("TTSTextFrame", names)
        self.assertLess(names.index("TTSStartedFrame"), names.index("TTSTextFrame"))

    async def test_voicemail_drops_held_speech_and_blocks_later_input(self):
        detector, _ = _detector(("voicemail", 0.95))
        down, _ = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSStartedFrame(),
                TTSTextFrame("Hi, this is Alex.", aggregated_by=AggregationType.SENTENCE),
                SleepFrame(0.2),
                _said("Please leave a message."),
                SleepFrame(VERDICT_SETTLE),
                LLMTextFrame(text="should not reach the conversation"),
                SleepFrame(0.2),
            ],
            start_timeout=5.0,
        )
        names = _names(down)
        self.assertNotIn("TTSTextFrame", names)
        self.assertNotIn("LLMTextFrame", names)

    async def test_transcriptions_pass_through_before_a_verdict(self):
        detector, _ = _detector(("conversation", 0.9))
        down, _ = await run_test(
            detector,
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertIn("TranscriptionFrame", _names(down))


class TestVoicemailDetectorEndWorkerFrame(unittest.IsolatedAsyncioTestCase):
    async def test_handler_pushing_upstream_ends_worker(self):
        detector, _ = _detector(("voicemail", 0.95))

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            detector,
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream: {_names(up)}",
        )

    async def test_handler_pushing_downstream_ends_worker(self):
        detector, _ = _detector(("voicemail", 0.95))

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.DOWNSTREAM
            )

        down, _up = await run_test(
            detector,
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in down),
            f"EndWorkerFrame did not escape downstream: {_names(down)}",
        )

    async def test_voicemail_verdict_lets_upstream_end_from_main_pipeline(self):
        detector, _ = _detector(("voicemail", 0.95))
        ender = _Passthrough()

        @detector.event_handler("on_voicemail_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="VOICEMAIL detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after VOICEMAIL: {_names(up)}",
        )

    async def test_conversation_verdict_lets_upstream_end_from_main_pipeline(self):
        detector, _ = _detector(("conversation", 0.9))
        ender = _Passthrough()

        @detector.event_handler("on_conversation_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="CONVERSATION detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after CONVERSATION: {_names(up)}",
        )


class _FakeLLM(LLMService):
    def __init__(self, reply: str):
        super().__init__()
        self.reply = reply
        self.seen = []


class TestDeprecatedLLMParameter(unittest.IsolatedAsyncioTestCase):
    def test_llm_parameter_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VoicemailDetector(llm=_FakeLLM("CONVERSATION"))
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_needs_a_classifier_or_an_llm(self):
        with self.assertRaises(ValueError):
            VoicemailDetector()

    def test_llm_builds_an_llm_classifier(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector = VoicemailDetector(llm=_FakeLLM("VOICEMAIL"), custom_system_prompt="Extra")
        classifier = detector._classifier
        self.assertIsInstance(classifier, LLMClassifier)
        instructions = classifier._instructions
        self.assertTrue(instructions.startswith("Extra"))
        self.assertIn("single JSON object", instructions)
