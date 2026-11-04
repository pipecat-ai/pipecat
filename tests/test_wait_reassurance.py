#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import (
    BackgroundWorkFinishedFrame,
    BackgroundWorkStartedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMInstructorMetadataFrame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.wait_reassurance import (
    WAIT_REASSURANCE_PROMPT,
    WAIT_REASSURANCE_SILENT_MARK,
    WaitReassurance,
)
from pipecat.tests.utils import SleepFrame, run_test

INTERVAL = 0.2
MARGIN = 0.15


def _prompts(frames) -> list[str]:
    return [f.messages[0]["content"] for f in frames if isinstance(f, LLMMessagesAppendFrame)]


def _waiting() -> list[Frame]:
    """Work in flight and the bot's acknowledgement over: the conversation goes quiet."""
    return [
        BackgroundWorkStartedFrame(work_id="w1"),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
    ]


def _response(*texts: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(t) for t in texts),
        LLMFullResponseEndFrame(),
    ]


async def _run(frames: list[Frame], **kwargs):
    down, _ = await run_test(
        WaitReassurance(interval_secs=INTERVAL, **kwargs), frames_to_send=frames
    )
    return down


def _announcements(frames) -> list[LLMInstructorMetadataFrame]:
    return [f for f in frames if isinstance(f, LLMInstructorMetadataFrame)]


# ---------------------------------------------------------------------------
# What the LLM is told at start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_llm_is_told_at_start_how_it_will_be_prompted():
    down, up = await run_test(WaitReassurance(interval_secs=INTERVAL), frames_to_send=[])

    for frames in (down, up):
        (announcement,) = _announcements(frames)
        assert announcement.silent_mark == WAIT_REASSURANCE_SILENT_MARK
        assert announcement.instruction is not None
        assert WAIT_REASSURANCE_SILENT_MARK in announcement.instruction
        assert "{mark}" not in announcement.instruction
        # The instruction names the message by its opening.
        opening = WAIT_REASSURANCE_PROMPT.split(",")[0]
        assert f'"{opening}' in announcement.instruction


# ---------------------------------------------------------------------------
# When the model is asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_is_asked_once_the_wait_has_gone_quiet():
    down = await _run([*_waiting(), SleepFrame(INTERVAL + MARGIN)])

    (prompt,) = _prompts(down)
    assert prompt == WAIT_REASSURANCE_PROMPT.format(quiet_secs=0)
    (append,) = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
    assert append.run_llm is True
    assert append.messages[0]["role"] == "developer"


@pytest.mark.asyncio
async def test_nothing_is_asked_with_no_work_in_flight():
    down = await _run(
        [BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), SleepFrame(INTERVAL + MARGIN)]
    )

    assert _prompts(down) == []


@pytest.mark.asyncio
async def test_nothing_is_asked_while_someone_speaks():
    down = await _run(
        [
            BackgroundWorkStartedFrame(work_id="w1"),
            BotStartedSpeakingFrame(),
            SleepFrame(INTERVAL + MARGIN),
            BotStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            SleepFrame(INTERVAL + MARGIN),
        ]
    )

    assert _prompts(down) == []


@pytest.mark.asyncio
async def test_nothing_is_asked_once_the_work_is_over():
    down = await _run(
        [*_waiting(), BackgroundWorkFinishedFrame(work_id="w1"), SleepFrame(INTERVAL + MARGIN)]
    )

    assert _prompts(down) == []


@pytest.mark.asyncio
async def test_the_instruction_prompt_and_mark_are_the_apps():
    down = await _run(
        [*_waiting(), SleepFrame(INTERVAL + MARGIN)],
        instruction="Say {mark} to pass.",
        prompt="Quiet for {quiet_secs}s; say {mark} to pass.",
        silent_mark="PASS",
    )

    (announcement,) = _announcements(down)
    assert announcement.instruction == "Say PASS to pass."
    assert announcement.silent_mark == "PASS"
    assert _prompts(down) == ["Quiet for 0s; say PASS to pass."]


# ---------------------------------------------------------------------------
# After the decision
# ---------------------------------------------------------------------------


def _silent_decision() -> list[Frame]:
    """A reply the LLM service dropped as the silent mark, with its report."""
    return [
        LLMFullResponseStartFrame(),
        LLMMarkerResponseFrame(
            raw=WAIT_REASSURANCE_SILENT_MARK,
            marker=WAIT_REASSURANCE_SILENT_MARK,
            kind="silent",
            markers=[WAIT_REASSURANCE_SILENT_MARK],
        ),
        LLMFullResponseEndFrame(),
    ]


@pytest.mark.asyncio
async def test_a_silent_decision_starts_the_interval_over():
    down = await _run(
        [
            *_waiting(),
            SleepFrame(INTERVAL + MARGIN),
            *_silent_decision(),
            SleepFrame(INTERVAL + MARGIN),
        ]
    )

    assert len(_prompts(down)) == 2
    assert any(isinstance(f, LLMMarkerResponseFrame) for f in down)


@pytest.mark.asyncio
async def test_a_spoken_remark_starts_the_interval_over_through_the_bots_speech():
    down = await _run(
        [
            *_waiting(),
            SleepFrame(INTERVAL + MARGIN),
            *_response("Still on it."),
            SleepFrame(INTERVAL + MARGIN),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(INTERVAL + MARGIN),
        ]
    )

    assert len(_prompts(down)) == 2
