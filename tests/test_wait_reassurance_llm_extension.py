#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserWaitingFrame,
    UserWaitingTimeoutUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import run_test
from pipecat.turns.user_stop.llm_turn_completion.llm_extension import LLMTurnCompletionLLMExtension
from pipecat.turns.user_turn_completion import USER_TURN_COMPLETE_MARKER
from pipecat.turns.wait_reassurance.llm_extension import (
    WAIT_REASSURANCE_PROMPT,
    WAIT_REASSURANCE_SILENT_MARK,
    WaitReassuranceLLMExtension,
)

INTERVAL = 7.0
MARK = WAIT_REASSURANCE_SILENT_MARK


class _MockLLMService(LLMService):
    """Minimal LLM service that pushes every frame onward, as a concrete service does."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=kwargs.pop("system_instruction", None),
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _service(**kwargs) -> _MockLLMService:
    return _MockLLMService(
        extensions=[WaitReassuranceLLMExtension(interval_secs=INTERVAL, **kwargs)]
    )


def _response(*texts: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(t) for t in texts),
        LLMFullResponseEndFrame(),
    ]


def _texts(frames: list) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


def _prompts(frames: list) -> list[str]:
    return [f.messages[0]["content"] for f in frames if isinstance(f, LLMMessagesAppendFrame)]


def _reports(frames: list) -> list[LLMMarkerResponseFrame]:
    return [f for f in frames if isinstance(f, LLMMarkerResponseFrame)]


# ---------------------------------------------------------------------------
# What the LLM is told, and how the wait is configured
# ---------------------------------------------------------------------------


def test_the_instruction_explains_the_prompt_and_the_mark():
    service = _service()
    service._compose_system_instruction()

    instruction = service._settings.system_instruction
    assert instruction is not None
    assert MARK in instruction
    assert "{mark}" not in instruction
    # The instruction names the message by its opening.
    opening = WAIT_REASSURANCE_PROMPT.split(",")[0]
    assert f'"{opening}' in instruction


@pytest.mark.asyncio
async def test_the_waiting_timeout_is_set_upstream_at_start():
    _, up = await run_test(_service(), frames_to_send=[])

    (update,) = [f for f in up if isinstance(f, UserWaitingTimeoutUpdateFrame)]
    assert update.timeout == INTERVAL


# ---------------------------------------------------------------------------
# When the model is asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_is_asked_when_the_user_has_waited():
    down, _ = await run_test(_service(), frames_to_send=[UserWaitingFrame(quiet_secs=12.4)])

    (append,) = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
    assert append.run_llm is True
    assert append.messages[0]["role"] == "developer"
    assert append.messages[0]["content"] == WAIT_REASSURANCE_PROMPT.format(quiet_secs=12)


@pytest.mark.asyncio
async def test_the_instruction_prompt_and_mark_are_the_apps():
    service = _service(
        instruction="Say {mark} to pass.",
        prompt="Quiet for {quiet_secs}s; say {mark} to pass.",
        silent_mark="PASS",
    )
    service._compose_system_instruction()
    assert service._settings.system_instruction == "Say PASS to pass."

    down, _ = await run_test(
        service, frames_to_send=[UserWaitingFrame(quiet_secs=3), *_response("PASS")]
    )

    assert _prompts(down) == ["Quiet for 3s; say PASS to pass."]
    assert _texts(down) == []
    (report,) = _reports(down)
    assert report.marker == "PASS"


# ---------------------------------------------------------------------------
# Reading the reply for the mark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_opening_with_the_mark_is_dropped_whole_and_reported_both_ways():
    down, up = await run_test(_service(), frames_to_send=_response(" ", MARK, " and more"))

    assert _texts(down) == []
    (report,) = _reports(down)
    assert report.marker == MARK
    assert report.kind == "silent"
    assert report.raw == f" {MARK}"
    assert report.markers == [MARK]
    # The user aggregator, upstream, starts its timer over on the same report.
    (upstream_report,) = _reports(up)
    assert upstream_report.kind == "silent"
    assert upstream_report.broadcast_sibling_id == report.id


@pytest.mark.asyncio
async def test_any_other_reply_goes_out_as_written():
    down, _ = await run_test(_service(), frames_to_send=_response("Still ", "on it."))

    assert _texts(down) == ["Still ", "on it."]
    assert _reports(down) == []


@pytest.mark.asyncio
async def test_held_whitespace_goes_out_with_the_text_that_decided_it():
    down, _ = await run_test(_service(), frames_to_send=_response(" ", "\n", "Nearly there."))

    assert _texts(down) == [" \nNearly there."]


@pytest.mark.asyncio
async def test_a_silent_reply_does_not_shadow_the_next():
    down, _ = await run_test(_service(), frames_to_send=[*_response(MARK), *_response("Hello.")])

    assert _texts(down) == ["Hello."]
    assert len(_reports(down)) == 1


@pytest.mark.asyncio
async def test_an_interruption_forgets_a_held_opening():
    down, _ = await run_test(
        _service(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame(" "),
            InterruptionFrame(),
            *_response("Hello."),
        ],
    )

    assert _texts(down) == ["Hello."]


@pytest.mark.asyncio
async def test_a_turn_marker_before_the_mark_is_read_as_silent_too():
    """With turn completion on, its marker is stripped first, so the mark after it still reads."""
    down, _ = await run_test(
        _service(),
        frames_to_send=[
            LLMUpdateSettingsFrame(delta=LLMSettings(extensions=[LLMTurnCompletionLLMExtension])),
            *_response(f"{USER_TURN_COMPLETE_MARKER} {MARK}"),
        ],
    )

    assert _texts(down) == []
    assert [r.kind for r in _reports(down)] == ["silent", "complete"]


@pytest.mark.asyncio
async def test_a_bare_mark_under_turn_completion_is_read_as_silent_when_the_reply_ends():
    down, _ = await run_test(
        _service(),
        frames_to_send=[
            LLMUpdateSettingsFrame(delta=LLMSettings(extensions=[LLMTurnCompletionLLMExtension])),
            *_response(MARK),
        ],
    )

    assert _texts(down) == []
    # Turn completion reports first: its "no marker" report and the held text come
    # out together at the end of the reply, and the mark is read from the text.
    assert [r.kind for r in _reports(down)] == [None, "silent"]
