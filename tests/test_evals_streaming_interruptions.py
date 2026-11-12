#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An untagged streaming transcript must not become evidence across an interruption."""

import asyncio

import pytest

from pipecat.evals.client import _BotSegmentTranscriptionFrame, _BotSpeechGate
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.script_session import EvalScriptSession
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import run_test


def _stream():
    return EvalEventStream(bot_audio=True, trace=EvalTrace())


def _untagged_final():
    return TranscriptionFrame(
        text="A late final from the interrupted previous answer.", user_id="", timestamp=""
    )


@pytest.mark.asyncio
async def test_ordinary_streaming_final_remains_usable():
    stream = _stream()
    stream.input_sent()
    stream.frame_to_event(LLMFullResponseStartFrame())
    stream.frame_to_event(BotStartedSpeakingFrame())
    final = _untagged_final()
    down, up = await run_test(_BotSpeechGate(stream, EvalTrace(), None), frames_to_send=[final])
    assert [frame for frame in down if isinstance(frame, TranscriptionFrame)] == [final]
    assert not any(isinstance(frame, ErrorFrame) for frame in up)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_reply_started", [False, True])
async def test_streaming_final_with_ambiguous_audio_origin_is_rejected(new_reply_started):
    stream = _stream()
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.input_sent()  # Caller deliberately interrupted a spoken answer.
    if new_reply_started:
        stream.frame_to_event(BotStoppedSpeakingFrame())
        stream.frame_to_event(LLMFullResponseStartFrame())
        stream.frame_to_event(BotStartedSpeakingFrame())
    assert stream.talked_over_until is not None

    down, up = await run_test(
        _BotSpeechGate(stream, EvalTrace(), None), frames_to_send=[_untagged_final()]
    )

    # Arrival after a new LLM/audio reply is not evidence that this transcript
    # belongs to that reply. A provider may finish recognizing older speech late.
    assert not any(isinstance(frame, TranscriptionFrame) for frame in down)
    assert any(isinstance(frame, ErrorFrame) for frame in up)


@pytest.mark.asyncio
async def test_repeated_ambiguous_finals_report_one_sanitized_error():
    stream = _stream()
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.input_sent()
    down, up = await run_test(
        _BotSpeechGate(stream, EvalTrace(), None),
        frames_to_send=[_untagged_final(), _untagged_final(), _untagged_final()],
    )
    assert not any(isinstance(frame, TranscriptionFrame) for frame in down)
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert errors[0].error == "Cannot attribute streaming transcription after interrupted speech"
    assert _untagged_final().text not in errors[0].error


@pytest.mark.asyncio
async def test_a_later_quiet_send_cannot_restore_unknown_audio_origin():
    stream = _stream()
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.input_sent()
    stream.frame_to_event(BotStoppedSpeakingFrame())
    stream.input_sent()
    assert stream.talked_over_until is None

    down, up = await run_test(
        _BotSpeechGate(stream, EvalTrace(), None), frames_to_send=[_untagged_final()]
    )
    assert not any(isinstance(frame, TranscriptionFrame) for frame in down)
    assert len([frame for frame in up if isinstance(frame, ErrorFrame)]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " \t\n"])
async def test_empty_transcripts_do_not_report_an_attribution_failure(text):
    stream = _stream()
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.input_sent()
    _, up = await run_test(
        _BotSpeechGate(stream, EvalTrace(), None),
        frames_to_send=[TranscriptionFrame(text=text, user_id="", timestamp="")],
    )
    assert not any(isinstance(frame, ErrorFrame) for frame in up)


@pytest.mark.asyncio
async def test_a_fully_drained_observing_turn_preserves_the_next_streaming_reply():
    session = EvalScriptSession(
        EvalScriptScenario(name="presence", turns=[], bot_audio=True), "ws://unused"
    )
    stream = session._stream
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.bot_turn_started()
    stream.bot_segment_ended()

    async def finish_reply():
        await asyncio.sleep(0)
        stream.frame_to_event(BotStoppedSpeakingFrame())
        stream.bot_segment_transcribed()
        await stream.bot_turn_stopped("The previous reply.")

    finishing = asyncio.create_task(finish_reply())
    await session._driver._await_previous_reply()
    await finishing
    stream.frame_to_event(BotStartedSpeakingFrame())
    fresh = TranscriptionFrame(text="Are you still there?", user_id="", timestamp="")
    down, up = await run_test(_BotSpeechGate(stream, EvalTrace(), None), frames_to_send=[fresh])
    assert [frame for frame in down if isinstance(frame, TranscriptionFrame)] == [fresh]
    assert not any(isinstance(frame, ErrorFrame) for frame in up)


@pytest.mark.asyncio
async def test_tagged_streaming_transcripts_are_filtered_by_their_audio_origin():
    stream = _stream()
    stream.frame_to_event(BotStartedSpeakingFrame())
    stream.input_sent()
    stream.frame_to_event(LLMFullResponseStartFrame())
    stream.frame_to_event(BotStartedSpeakingFrame())
    boundary = stream.talked_over_until
    old = _BotSegmentTranscriptionFrame(
        text="old", user_id="", timestamp="", segment_started_at=boundary - 1
    )
    fresh = _BotSegmentTranscriptionFrame(
        text="fresh", user_id="", timestamp="", segment_started_at=boundary + 1
    )
    down, up = await run_test(
        _BotSpeechGate(stream, EvalTrace(), None), frames_to_send=[old, fresh]
    )
    assert [frame for frame in down if isinstance(frame, TranscriptionFrame)] == [fresh]
    assert not any(isinstance(frame, ErrorFrame) for frame in up)
