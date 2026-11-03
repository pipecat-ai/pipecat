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
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.wait_reassurance import (
    WAIT_REASSURANCE_SILENT_MARK,
    WaitReassurance,
)
from pipecat.tests.utils import SleepFrame, run_test

INTERVAL = 0.2
MARGIN = 0.15


def _prompts(frames) -> list[str]:
    return [f.messages[0]["content"] for f in frames if isinstance(f, LLMMessagesAppendFrame)]


def _texts(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


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


# ---------------------------------------------------------------------------
# When the model is asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_is_asked_once_the_wait_has_gone_quiet():
    down = await _run([*_waiting(), SleepFrame(INTERVAL + MARGIN)])

    (prompt,) = _prompts(down)
    assert WAIT_REASSURANCE_SILENT_MARK in prompt
    assert "seconds" in prompt
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
async def test_the_prompt_and_mark_are_the_apps():
    down = await _run(
        [*_waiting(), SleepFrame(INTERVAL + MARGIN)],
        prompt="Quiet for {quiet_secs}s; say {mark} to pass.",
        silent_mark="PASS",
    )

    assert _prompts(down) == ["Quiet for 0s; say PASS to pass."]


# ---------------------------------------------------------------------------
# What happens to the reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_opening_with_the_mark_is_dropped_whole():
    down = await _run(_response(" ", WAIT_REASSURANCE_SILENT_MARK, " and more"))

    assert _texts(down) == []
    assert [type(f) for f in down if not isinstance(f, LLMMessagesAppendFrame)][:2] == [
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_any_other_reply_passes_through_unchanged():
    down = await _run(_response("Still ", "on it."))

    assert _texts(down) == ["Still ", "on it."]


@pytest.mark.asyncio
async def test_held_whitespace_goes_out_with_the_text_that_decided_it():
    down = await _run(_response(" ", "\n", "Nearly there."))

    assert _texts(down) == [" \nNearly there."]


@pytest.mark.asyncio
async def test_a_silent_decision_starts_the_interval_over():
    down = await _run(
        [
            *_waiting(),
            SleepFrame(INTERVAL + MARGIN),
            *_response(WAIT_REASSURANCE_SILENT_MARK),
            SleepFrame(INTERVAL + MARGIN),
        ]
    )

    assert len(_prompts(down)) == 2


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


@pytest.mark.asyncio
async def test_an_interruption_forgets_a_held_opening():
    down = await _run(
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame(" "),
            InterruptionFrame(),
            *_response("Hello."),
        ]
    )

    assert _texts(down) == ["Hello."]
