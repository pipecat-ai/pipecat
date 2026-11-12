"""Optional bounds on caller audio assembled with pauses."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import pipecat.evals.client as client_module
from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import EvalScriptScenario, EvalScriptTurn
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.session import EvalSessionParams
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame


def caller(maximum=None, audio=(), sample_rate=16000):
    trace = EvalTrace()
    voice = SimpleNamespace(sample_rate=sample_rate, synthesize=AsyncMock(side_effect=audio))
    client = EvalClient(
        "ws://unused",
        stream=EvalEventStream(bot_audio=False, trace=trace),
        trace=trace,
        user_tts=voice,
        session_params=EvalSessionParams(max_user_audio_buffer_bytes=maximum),
    )
    client._sink = SimpleNamespace(inject=AsyncMock())
    return client, voice


@pytest.mark.parametrize("maximum", [0, -2, 1, True, "4", 4.0])
def test_caller_limit_requires_a_positive_even_integer(maximum):
    with pytest.raises(ValidationError):
        EvalSessionParams(max_user_audio_buffer_bytes=maximum)


def test_caller_limit_defaults_to_none_and_survives_serialization():
    assert EvalSessionParams().max_user_audio_buffer_bytes is None
    params = EvalSessionParams(max_user_audio_buffer_bytes=2)
    assert EvalSessionParams.model_validate_json(params.model_dump_json()) == params


@pytest.mark.asyncio
async def test_huge_pause_is_refused_before_allocating_silence():
    client, voice = caller(520)
    with pytest.raises(ValueError, match="PCM buffer limit"):
        await client.say_with_pauses([10**1000])
    voice.synthesize.assert_not_awaited()
    client._sink.inject.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts,audio", [([10, 10], []), (["first", 10, "second"], [bytes(100), bytes(100)])]
)
async def test_combined_audio_is_checked_before_extending_buffer(monkeypatch, parts, audio):
    maximum = 518

    class GuardedBuffer(bytearray):
        def extend(self, audio):
            assert len(self) + len(audio) <= maximum, "oversized PCM reached buffer.extend"
            super().extend(audio)

    monkeypatch.setattr(client_module, "bytearray", GuardedBuffer, raising=False)
    client, voice = caller(maximum, audio)
    with pytest.raises(ValueError, match="PCM buffer limit"):
        await client.say_with_pauses(parts)
    client._sink.inject.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum", [None, 520])
async def test_exact_boundary_and_default_preserve_audio_and_frame_order(maximum):
    client, voice = caller(maximum, [b"a\0" * 50, b"b\0" * 50])
    await client.say_with_pauses(["first", 10, "second"])
    frames = [call.args[0] for call in client._sink.inject.await_args_list]
    assert [type(frame) for frame in frames] == [TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame]
    assert frames[1].audio == b"a\0" * 50 + bytes(320) + b"b\0" * 50
    assert frames[1].sample_rate == 16000
    assert frames[1].num_channels == 1
    assert voice.synthesize.await_count == 2


@pytest.mark.asyncio
async def test_cancellation_discards_an_unfinished_aggregate():
    client, voice = caller(520)
    entered = asyncio.Event()

    async def synthesize(text):
        if text == "first":
            return bytes(100)
        entered.set()
        await asyncio.Future()

    voice.synthesize.side_effect = synthesize
    task = asyncio.create_task(client.say_with_pauses(["first", 10, "second"]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    client._sink.inject.assert_not_awaited()


@pytest.mark.asyncio
async def test_script_session_records_overflow_as_a_harness_error_and_stops():
    voice = SimpleNamespace(sample_rate=16000, synthesize=AsyncMock(return_value=bytes(100)))
    session = EvalScriptSession(
        EvalScriptScenario(
            name="bounded",
            user_audio=True,
            turns=[EvalScriptTurn(user="hello", parts=["hello", 10], expect=[])],
        ),
        "ws://unused",
        params=EvalSessionParams(max_user_audio_buffer_bytes=400),
        user_tts=voice,
    )
    session._client._sink = SimpleNamespace(inject=AsyncMock())
    for method in ("wait_for_bot", "start", "handshake", "stop"):
        setattr(session._client, method, AsyncMock())
    result = await session.run()
    assert not result.passed
    assert len(result.failures) == 1
    assert result.failures[0].kind == "error"
    assert "PCM buffer limit" in result.failures[0].reason
    session._client._sink.inject.assert_not_awaited()
    session._client.stop.assert_awaited_once()
