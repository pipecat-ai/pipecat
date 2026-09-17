#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import functools
import unittest

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import TypeSafeAPITimeoutError

from pipecat.extensions.voicemail.typesafe_classifier import (
    CONVERSATION,
    VOICEMAIL,
    TypeSafeVoicemailClassifier,
)
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.services.typesafe.choice_llm import (
    QUESTION_ID,
    TypeSafeChoiceLLMService,
    transcript_state_builder,
)
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as _run_test
from tests.typesafe_test_helpers import FakeTypeSafeClient, choice_response

# Long enough for the judgment and the detector's delayed voicemail handler.
SETTLE = 1.0
VOICEMAIL_DELAY = 0.1

run_test = functools.partial(_run_test, start_timeout=5.0)

CRITERIA = {"YES": "The user agrees", "NO": "The user declines"}


def verdict(choice: str, confidence: float, options=CRITERIA):
    other = next(k for k in options if k != choice)
    p = (1 + confidence) / 2
    return choice_response(choice, confidence, {choice: p, other: 1 - p}, question_id=QUESTION_ID)


def judgments(client: FakeTypeSafeClient) -> list:
    """The client's requests, without the connection warm-up."""
    return [(state, questions) for state, questions in client.requests if isinstance(state, dict)]


def context(*user_turns: str) -> LLMContext:
    messages: list = [{"role": "system", "content": "Answer YES or NO."}]
    for i, turn in enumerate(user_turns):
        if i:
            messages.append({"role": "assistant", "content": ""})
        messages.append({"role": "user", "content": turn})
    return LLMContext(messages=messages)


class TestTranscriptStateBuilder(unittest.TestCase):
    def test_joins_user_turns_only(self):
        ctx = LLMContext(
            messages=[
                {"role": "developer", "content": "prompt"},
                {"role": "user", "content": "Hi, you've reached Jamie."},
                {"role": "assistant", "content": "VOICEMAIL"},
                {"role": "user", "content": [{"type": "text", "text": "Leave a message."}]},
            ]
        )
        self.assertEqual(
            transcript_state_builder(ctx),
            {"transcript": "Hi, you've reached Jamie.\nLeave a message."},
        )


class TestTypeSafeChoiceLLMService(unittest.IsolatedAsyncioTestCase):
    def make_service(self, client, **kwargs) -> TypeSafeChoiceLLMService:
        return TypeSafeChoiceLLMService(
            judge=TypeSafeJudge(client=client),
            instructions="Did the user in `transcript` agree?",
            criteria=CRITERIA,
            warm_up=False,
            **kwargs,
        )

    async def test_answers_with_the_chosen_label(self):
        client = FakeTypeSafeClient(verdict("YES", 0.9))
        service = self.make_service(client)

        down, _ = await run_test(
            service,
            frames_to_send=[LLMContextFrame(context=context("Sure, go ahead")), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        text = next(f for f in down if isinstance(f, LLMTextFrame))
        self.assertEqual(text.text, "YES")
        state, questions = judgments(client)[0]
        self.assertEqual(state, {"transcript": "Sure, go ahead"})
        self.assertEqual(set(questions), {QUESTION_ID})

    async def test_low_confidence_answers_nothing(self):
        client = FakeTypeSafeClient(verdict("YES", 0.3))
        service = self.make_service(client, confidence_threshold=0.5)

        await run_test(
            service,
            frames_to_send=[LLMContextFrame(context=context("Hmm")), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMFullResponseEndFrame,
            ],
        )

    async def test_failed_request_reports_error_and_answers_nothing(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        service = self.make_service(client)

        _, up = await run_test(
            service,
            frames_to_send=[LLMContextFrame(context=context("Yes")), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMFullResponseEndFrame,
            ],
        )

        self.assertTrue(any(isinstance(f, ErrorFrame) for f in up))

    async def test_interruption_cancels_the_judgment(self):
        client = FakeTypeSafeClient(verdict("YES", 0.9), delay=0.5)
        service = self.make_service(client)

        await run_test(
            service,
            frames_to_send=[
                LLMContextFrame(context=context("Yes")),
                SleepFrame(0.1),
                InterruptionFrame(),
                SleepFrame(0.8),
            ],
            expected_down_frames=[LLMServiceMetadataFrame, InterruptionFrame],
        )

    async def test_reports_ttfb_processing_and_usage(self):
        client = FakeTypeSafeClient(verdict("NO", 0.9))
        service = self.make_service(client)

        down, _ = await run_test(
            service,
            frames_to_send=[LLMContextFrame(context=context("No thanks")), SleepFrame()],
            pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        reported = {
            type(data) for frame in down if isinstance(frame, MetricsFrame) for data in frame.data
        }
        self.assertLessEqual(
            {TTFBMetricsData, ProcessingMetricsData, LLMUsageMetricsData}, reported
        )


def caller_turn(text: str) -> list[Frame]:
    """The frames STT and VAD emit for one turn of the answering side."""
    return [
        UserStartedSpeakingFrame(),
        TranscriptionFrame(text=text, user_id="caller", timestamp="0"),
        UserStoppedSpeakingFrame(),
    ]


class TestVoicemailDetectorWithTypeSafe(unittest.IsolatedAsyncioTestCase):
    """The classifier plugs into VoicemailDetector as its LLM, unchanged."""

    def make_detector(self, client, **kwargs) -> VoicemailDetector:
        classifier = TypeSafeVoicemailClassifier(
            judge=TypeSafeJudge(client=client), warm_up=False, **kwargs
        )
        detector = VoicemailDetector(llm=classifier, voicemail_response_delay=VOICEMAIL_DELAY)
        self.events: list[str] = []

        @detector.event_handler("on_conversation_detected")
        async def _conversation(_processor):
            self.events.append("conversation")

        @detector.event_handler("on_voicemail_detected")
        async def _voicemail(_processor):
            self.events.append("voicemail")

        return detector

    def test_criteria_must_cover_both_labels(self):
        with self.assertRaises(ValueError):
            TypeSafeVoicemailClassifier(
                judge=TypeSafeJudge(client=FakeTypeSafeClient()),
                criteria={CONVERSATION: "a person"},
            )

    async def test_voicemail_verdict_fires_the_handler(self):
        client = FakeTypeSafeClient(verdict(VOICEMAIL, 0.9, {CONVERSATION: "", VOICEMAIL: ""}))
        detector = self.make_detector(client)

        await run_test(
            detector,
            frames_to_send=[
                *caller_turn("Hi, you've reached Jamie, leave a message."),
                SleepFrame(SETTLE),
            ],
        )

        self.assertEqual(self.events, ["voicemail"])
        state, _ = judgments(client)[0]
        self.assertEqual(state, {"transcript": "Hi, you've reached Jamie, leave a message."})

    async def test_conversation_verdict_fires_the_handler(self):
        client = FakeTypeSafeClient(verdict(CONVERSATION, 1.0, {CONVERSATION: "", VOICEMAIL: ""}))
        detector = self.make_detector(client)

        await run_test(
            detector,
            frames_to_send=[*caller_turn("Hello?"), SleepFrame(SETTLE)],
        )

        self.assertEqual(self.events, ["conversation"])

    async def test_low_confidence_waits_for_the_next_turn(self):
        client = FakeTypeSafeClient(verdict(VOICEMAIL, 0.3, {CONVERSATION: "", VOICEMAIL: ""}))
        detector = self.make_detector(client, confidence_threshold=0.5)

        await run_test(
            detector,
            frames_to_send=[
                *caller_turn("Hey, sorry, I can't come to the phone right now."),
                SleepFrame(0.3),
                *caller_turn("Leave a message after the tone."),
                SleepFrame(SETTLE),
            ],
        )

        # Neither weak verdict was acted on, and the second request saw the
        # whole transcript without the empty answers in between.
        self.assertEqual(self.events, [])
        self.assertEqual(len(judgments(client)), 2)
        self.assertEqual(
            judgments(client)[1][0]["transcript"],
            "Hey, sorry, I can't come to the phone right now.\nLeave a message after the tone.",
        )
