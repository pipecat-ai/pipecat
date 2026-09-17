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
    QUESTION_ID,
    VOICEMAIL,
    caller_transcript,
)
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as _run_test
from tests.typesafe_test_helpers import FakeTypeSafeClient, choice_response

# Long enough for the judgment and the delayed voicemail handler.
SETTLE = 1.0
VOICEMAIL_DELAY = 0.1

run_test = functools.partial(_run_test, start_timeout=5.0)


def verdict(choice: str, confidence: float):
    other = VOICEMAIL if choice == CONVERSATION else CONVERSATION
    p = (1 + confidence) / 2
    return choice_response(choice, confidence, {choice: p, other: 1 - p}, question_id=QUESTION_ID)


def caller_turn(text: str) -> list[Frame]:
    """The frames STT and VAD emit for one turn of the answering side."""
    return [
        UserStartedSpeakingFrame(),
        TranscriptionFrame(text=text, user_id="caller", timestamp="0"),
        UserStoppedSpeakingFrame(),
    ]


def judgments(client: FakeTypeSafeClient) -> list:
    """The client's requests, without the connection warm-up."""
    return [(state, questions) for state, questions in client.requests if isinstance(state, dict)]


class TestCallerTranscript(unittest.TestCase):
    def test_joins_user_turns_only(self):
        context = LLMContext(
            messages=[
                {"role": "developer", "content": "prompt"},
                {"role": "user", "content": "Hi, you've reached Jamie."},
                {"role": "assistant", "content": "VOICEMAIL"},
                {"role": "user", "content": [{"type": "text", "text": "Leave a message."}]},
            ]
        )
        self.assertEqual(caller_transcript(context), "Hi, you've reached Jamie.\nLeave a message.")


class TestVoicemailDetectorWithJudge(unittest.IsolatedAsyncioTestCase):
    def make_detector(self, client, **kwargs) -> VoicemailDetector:
        detector = VoicemailDetector(
            judge=TypeSafeJudge(client=client),
            voicemail_response_delay=VOICEMAIL_DELAY,
            **kwargs,
        )
        self.events: list[str] = []

        @detector.event_handler("on_conversation_detected")
        async def _conversation(_processor):
            self.events.append("conversation")

        @detector.event_handler("on_voicemail_detected")
        async def _voicemail(_processor):
            self.events.append("voicemail")

        return detector

    async def test_constructor_needs_exactly_one_classifier(self):
        with self.assertRaises(ValueError):
            VoicemailDetector()
        with self.assertRaises(ValueError):
            VoicemailDetector(
                judge=TypeSafeJudge(client=FakeTypeSafeClient()), custom_system_prompt="x"
            )

    async def test_voicemail_verdict_fires_handler_after_delay(self):
        client = FakeTypeSafeClient(verdict(VOICEMAIL, 0.9))
        detector = self.make_detector(client)

        down, _ = await run_test(
            detector,
            frames_to_send=[
                *caller_turn("Hi, you've reached Jamie, leave a message."),
                SleepFrame(SETTLE),
            ],
        )

        self.assertEqual(self.events, ["voicemail"])
        self.assertEqual(len(judgments(client)), 1)
        state, questions = judgments(client)[0]
        self.assertEqual(state, {"speech": "Hi, you've reached Jamie, leave a message."})
        self.assertEqual(set(questions), {QUESTION_ID})
        # The classifier's context frame stays inside the branch.
        self.assertFalse(any(isinstance(f, LLMContextFrame) for f in down))

    async def test_conversation_verdict_releases_gated_tts(self):
        client = FakeTypeSafeClient(verdict(CONVERSATION, 0.9))
        detector = self.make_detector(client)

        # The gate holds the TTS frame sent before the caller's turn and lets
        # it through once the verdict says a person answered.
        down, _ = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSTextFrame(text="Hi there", aggregated_by="sentence"),
                SleepFrame(0.2),
                *caller_turn("Hello?"),
                SleepFrame(SETTLE),
            ],
        )

        self.assertEqual(self.events, ["conversation"])
        names = [type(f).__name__ for f in down]
        self.assertIn("TTSTextFrame", names)
        self.assertGreater(names.index("TTSTextFrame"), names.index("UserStoppedSpeakingFrame"))

    async def test_low_confidence_waits_for_the_next_turn(self):
        client = FakeTypeSafeClient(verdict(VOICEMAIL, 0.3))
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

        # Both turns were judged and neither verdict was acted on; the second
        # request saw the whole transcript.
        self.assertEqual(self.events, [])
        self.assertEqual(len(judgments(client)), 2)
        self.assertEqual(
            judgments(client)[1][0]["speech"],
            "Hey, sorry, I can't come to the phone right now.\nLeave a message after the tone.",
        )

    async def test_failed_request_reports_error_and_waits(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        detector = self.make_detector(client)

        _, up = await run_test(
            detector,
            frames_to_send=[*caller_turn("Hello?"), SleepFrame(SETTLE)],
        )

        self.assertEqual(self.events, [])
        self.assertTrue(any(isinstance(f, ErrorFrame) for f in up))

    async def test_judges_only_until_a_verdict(self):
        client = FakeTypeSafeClient(verdict(CONVERSATION, 1.0))
        detector = self.make_detector(client)

        await run_test(
            detector,
            frames_to_send=[
                *caller_turn("Hello?"),
                SleepFrame(SETTLE),
                *caller_turn("Who is this?"),
                SleepFrame(0.3),
            ],
        )

        self.assertEqual(self.events, ["conversation"])
        self.assertEqual(len(judgments(client)), 1)
