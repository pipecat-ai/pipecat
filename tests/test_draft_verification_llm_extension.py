#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json

import pytest

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.draft_verification_llm_extension import (
    LATE_CORRECTION_TEMPLATE,
    DraftVerificationLLMExtension,
    LLMDraftVerifier,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test

DRAFT = "The capital of Australia is Sydney."
CORRECTED = "The capital of Australia is Canberra."


def _settings(extensions) -> LLMSettings:
    return LLMSettings(
        model="test-model",
        system_instruction=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        filter_incomplete_user_turns=None,
        user_turn_completion_config=None,
        extensions=extensions,
    )


class _FastLLM(LLMService):
    """Answers each context after ``latency`` seconds, streaming the next reply in ``replies``."""

    def __init__(self, *, replies: list[list[Frame]], latency: float = 0.01, extensions=()):
        super().__init__(settings=_settings(list(extensions)))
        self._replies = list(replies)
        self._latency = latency

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            reply = self._replies.pop(0)
            self.create_task(self._answer(reply), "answer")
        else:
            await self.push_frame(frame, direction)

    async def _answer(self, reply: list[Frame]):
        await asyncio.sleep(self._latency)
        for frame in reply:
            await self.push_frame(frame)


class _VerifierLLM(LLMService):
    """Answers ``run_inference`` after ``latency`` seconds with a fixed verdict."""

    def __init__(self, *, verdicts: list[str | None], latency: float):
        super().__init__(settings=_settings([]))
        self._verdicts = list(verdicts)
        self._latency = latency
        self.contexts: list[LLMContext] = []
        self.schemas: list[dict | None] = []

    async def run_inference(self, context, max_tokens=None, system_instruction=None, **kwargs):
        self.contexts.append(context)
        self.schemas.append(kwargs.get("response_schema"))
        verdict = self._verdicts.pop(0)
        await asyncio.sleep(self._latency)
        return verdict


def _agree() -> str:
    return json.dumps({"correct": True, "reply": DRAFT})


def _correct(text: str = CORRECTED) -> str:
    return json.dumps({"correct": False, "reply": text})


def _reply(text: str = DRAFT) -> list[Frame]:
    # Streamed in two chunks, as a model does.
    half = len(text) // 2
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text[:half]),
        LLMTextFrame(text[half:]),
        LLMFullResponseEndFrame(),
    ]


def _context(text: str = "What is the capital of Australia?") -> LLMContextFrame:
    return LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": text}]))


def _setup(*, verdicts, verifier_latency, replies=None, **kwargs):
    verifier = _VerifierLLM(verdicts=verdicts, latency=verifier_latency)
    extension = DraftVerificationLLMExtension(verifier=LLMDraftVerifier(llm=verifier), **kwargs)
    llm = _FastLLM(replies=replies or [_reply()], extensions=[extension])
    return llm, verifier, extension


def _spoken(frames) -> list[str]:
    """The text of each reply that went downstream, one string per reply."""
    replies, current = [], None
    for frame in frames:
        if isinstance(frame, LLMFullResponseStartFrame):
            current = ""
        elif isinstance(frame, LLMTextFrame) and current is not None:
            current += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame) and current is not None:
            replies.append(current)
            current = None
    return replies


@pytest.mark.asyncio
async def test_a_verified_reply_is_sent_unchanged():
    llm, verifier, _ = _setup(verdicts=[_agree()], verifier_latency=0.05)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.2)])

    assert _spoken(down) == [DRAFT]
    # The verifier saw the user's turn, the reply, and the question, in order.
    (context,) = verifier.contexts
    roles = [m["role"] for m in context.get_messages()]
    assert roles == ["user", "assistant", "developer"]
    assert context.get_messages()[1]["content"] == DRAFT


@pytest.mark.asyncio
async def test_a_held_reply_waits_for_its_verdict():
    llm, _, _ = _setup(verdicts=[_agree()], verifier_latency=0.1, hold_secs=1.0)

    down, _ = await run_test(
        llm,
        frames_to_send=[_context(), SleepFrame(sleep=0.05), SleepFrame(sleep=0.2)],
    )

    assert _spoken(down) == [DRAFT]


@pytest.mark.asyncio
async def test_a_correction_before_the_reply_is_spoken_replaces_it():
    llm, _, _ = _setup(verdicts=[_correct()], verifier_latency=0.05, hold_secs=0.5)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.2)])

    # The original never reaches the TTS or the context.
    assert _spoken(down) == [CORRECTED]


@pytest.mark.asyncio
async def test_a_correction_after_the_reply_was_sent_follows_it():
    llm, _, _ = _setup(verdicts=[_correct()], verifier_latency=0.1)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.3)])

    assert _spoken(down) == [DRAFT, LATE_CORRECTION_TEMPLATE.format(text=CORRECTED)]


@pytest.mark.asyncio
async def test_a_hold_that_runs_out_sends_the_reply_and_corrects_it_later():
    llm, _, _ = _setup(verdicts=[_correct()], verifier_latency=0.2, hold_secs=0.05)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.4)])

    assert _spoken(down) == [DRAFT, LATE_CORRECTION_TEMPLATE.format(text=CORRECTED)]


@pytest.mark.asyncio
async def test_late_corrections_can_be_turned_off():
    llm, _, _ = _setup(verdicts=[_correct()], verifier_latency=0.1, late_corrections=False)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.3)])

    assert _spoken(down) == [DRAFT]


@pytest.mark.asyncio
async def test_an_interruption_drops_the_held_reply_and_its_verdict():
    llm, _, extension = _setup(verdicts=[_correct()], verifier_latency=0.2, hold_secs=1.0)

    down, _ = await run_test(
        llm,
        frames_to_send=[
            _context(),
            SleepFrame(sleep=0.05),
            UserStartedSpeakingFrame(),
            InterruptionFrame(),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _spoken(down) == []
    assert not any(isinstance(f, LLMTextFrame) for f in down)
    assert extension.epoch >= 2


@pytest.mark.asyncio
async def test_a_late_verdict_after_the_user_moved_on_is_dropped():
    second = "Sure, it is sunny there today."
    llm, _, _ = _setup(
        verdicts=[_correct(), _agree()],
        verifier_latency=0.2,
        replies=[_reply(), _reply(second)],
    )

    down, _ = await run_test(
        llm,
        frames_to_send=[
            _context(),
            # The first reply is out and being checked when the user speaks
            # again and a new inference starts.
            SleepFrame(sleep=0.05),
            UserStartedSpeakingFrame(),
            _context("And the weather?"),
            SleepFrame(sleep=0.5),
        ],
    )

    # The correction of the first reply would answer a question the user has
    # moved past, so it is never spoken.
    assert _spoken(down) == [DRAFT, second]


@pytest.mark.asyncio
async def test_a_reply_that_calls_tools_is_not_checked():
    tool_reply = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Let me look that up."),
        FunctionCallInProgressFrame(function_name="lookup", tool_call_id="1", arguments={}),
        LLMFullResponseEndFrame(),
    ]
    llm, verifier, _ = _setup(
        verdicts=[_correct()], verifier_latency=0.01, hold_secs=1.0, replies=[tool_reply]
    )

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.2)])

    assert _spoken(down) == ["Let me look that up."]
    assert verifier.contexts == []


@pytest.mark.asyncio
async def test_an_unreadable_verdict_lets_the_reply_stand():
    llm, _, _ = _setup(verdicts=["not json"], verifier_latency=0.01, hold_secs=1.0)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.2)])

    assert _spoken(down) == [DRAFT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict",
    [
        f"```json\n{_correct()}\n```",
        f"Here is my verdict:\n{_correct()}\nThe reply named the wrong city.",
    ],
)
async def test_a_verdict_in_a_code_fence_or_prose_is_read(verdict):
    llm, _, _ = _setup(verdicts=[verdict], verifier_latency=0.01, hold_secs=1.0)

    down, _ = await run_test(llm, frames_to_send=[_context(), SleepFrame(sleep=0.2)])

    assert _spoken(down) == [CORRECTED]


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_schema", [False, True])
async def test_the_verdict_format_is_asked_for_with_or_without_a_schema(supports_schema):
    verifier_llm = _VerifierLLM(verdicts=[_agree()], latency=0.0)
    verifier_llm.supports_response_schema = supports_schema
    verifier = LLMDraftVerifier(llm=verifier_llm, prompt="Check the capital city.")

    await verifier.verify([{"role": "user", "content": "Capital of Australia?"}], DRAFT)

    assert (verifier_llm.schemas[0] is not None) == supports_schema
    instruction = verifier_llm.contexts[0].get_messages()[-1]["content"]
    assert instruction.startswith("Check the capital city.")
    assert '{"correct": true or false, "reply": "..."}' in instruction


@pytest.mark.asyncio
async def test_a_withdrawn_speculative_reply_is_dropped():
    llm, _, _ = _setup(verdicts=[_agree()], verifier_latency=0.2, hold_secs=1.0)

    down, _ = await run_test(
        llm,
        frames_to_send=[
            LLMContextFrame(context=_context().context, speculation=True),
            SleepFrame(sleep=0.05),
            UserTurnResponseCancelFrame(),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _spoken(down) == []


@pytest.mark.asyncio
async def test_a_speculative_reply_is_corrected_and_still_waits_for_its_turn():
    llm, _, _ = _setup(verdicts=[_correct()], verifier_latency=0.05, hold_secs=1.0)

    down, _ = await run_test(
        llm,
        frames_to_send=[
            LLMContextFrame(context=_context().context, speculation=True),
            # The verdict is in well before the turn is confirmed.
            SleepFrame(sleep=0.2),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ],
    )

    assert _spoken(down) == [CORRECTED]
    # The service's speculation gate still releases it only on confirmation.
    stopped = next(i for i, f in enumerate(down) if isinstance(f, UserStoppedSpeakingFrame))
    first_text = next(i for i, f in enumerate(down) if isinstance(f, LLMTextFrame))
    assert stopped < first_text
