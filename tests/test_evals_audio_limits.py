#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval PCM admission bounds, including data queued behind a stalled consumer."""

import asyncio
import base64
import json
import time
import types
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from pipecat.evals.client import EvalClient
from pipecat.evals.client_transport import FRAME_S, EvalClientInputTransport, EvalClientTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import EvalExpectation, EvalScriptScenario, EvalScriptTurn
from pipecat.evals.serializer import EvalAudioLimitError, EvalClientSerializer
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import WebsocketClientInputTransport, WebsocketClientParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketOutputTransport

RATE = 16000
CHUNK = bytes([1, 0]) * int(RATE * FRAME_S)


def audio_message(audio=CHUNK, rate=RATE, **extra):
    return json.dumps(
        {
            "label": "rtvi-ai",
            "type": "eval-bot-audio",
            "data": {"audio": base64.b64encode(audio).decode(), "sampleRate": rate, **extra},
        }
    )


def input_transport(maximum, failures):
    async def fail(reason):
        failures.append(reason)

    transport = EvalClientInputTransport(
        None,
        types.SimpleNamespace(is_closing=False, is_connected=True),
        WebsocketClientParams(audio_in_enabled=True),
        max_bot_audio_bytes=maximum,
        audio_failure=fail,
    )
    transport._sample_rate = RATE
    return transport


@pytest.mark.parametrize("field", ["max_bot_audio_bytes", "max_user_audio_buffer_bytes"])
@pytest.mark.parametrize("value", [0, -2, True, False, 1, 2.0, float("inf"), float("nan"), "1280"])
def test_audio_byte_limits_require_positive_even_integers(field, value):
    with pytest.raises(ValidationError):
        EvalSessionParams(**{field: value})


def test_audio_byte_limits_are_optional_and_serializable():
    defaults = EvalSessionParams()
    assert defaults.max_bot_audio_bytes is None
    assert defaults.max_user_audio_buffer_bytes is None
    limits = EvalSessionParams(max_bot_audio_bytes=1280, max_user_audio_buffer_bytes=2560)
    assert EvalSessionParams.model_validate_json(limits.model_dump_json()) == limits


@pytest.mark.asyncio
async def test_serializer_enforces_cumulative_quota_before_decode_or_resample(monkeypatch):
    serializer = EvalClientSerializer(max_bot_audio_bytes=len(CHUNK))
    frame = await serializer.deserialize(audio_message())
    assert frame.audio == CHUNK and frame.sample_rate == RATE
    message = audio_message(b"\x01\x00")
    monkeypatch.setattr(
        base64, "b64decode", lambda *_args, **_kw: pytest.fail("decoded over quota")
    )
    serializer._resample = AsyncMock(side_effect=AssertionError("resampled over quota"))
    with pytest.raises(EvalAudioLimitError, match="PCM admission"):
        await serializer.deserialize(message)
    serializer._resample.assert_not_awaited()


@pytest.mark.asyncio
async def test_expansion_is_checked_before_native_resampling(monkeypatch):
    serializer = EvalClientSerializer(max_bot_audio_bytes=len(CHUNK))
    message = audio_message(b"\x01\x00", rate=1)
    monkeypatch.setattr(base64, "b64decode", lambda *_args, **_kw: pytest.fail("decoded expansion"))
    serializer._resample = AsyncMock(side_effect=AssertionError("expanded before limit"))
    with pytest.raises(EvalAudioLimitError, match="PCM admission"):
        await serializer.deserialize(message)
    serializer._resample.assert_not_awaited()


@pytest.mark.asyncio
async def test_normalized_wire_quota_allows_a_valid_higher_sample_rate():
    serializer = EvalClientSerializer(max_bot_audio_bytes=len(CHUNK))
    frame = await serializer.deserialize(audio_message(CHUNK * 3, rate=48000))
    assert len(frame.audio) == len(CHUNK)
    assert frame.sample_rate == RATE and frame.num_channels == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [0, -1, True, 16000.0, "16000", None, 2**31])
async def test_bounded_wire_rate_requires_a_positive_native_integer(rate):
    with pytest.raises(EvalAudioLimitError, match="invalid PCM"):
        await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(audio_message(rate=rate))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio,extra",
    [
        (b"", {}),
        (b"\x01", {}),
        (b"RIFF" + b"\x00" * 12, {}),
        (CHUNK, {"numChannels": 2}),
        (CHUNK, {"channels": True}),
    ],
)
async def test_invalid_bounded_wire_audio_fails_without_payload_text(audio, extra):
    with pytest.raises(EvalAudioLimitError, match="invalid PCM") as raised:
        await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(
            audio_message(audio, **extra)
        )
    assert "RIFF" not in str(raised.value)


@pytest.mark.asyncio
async def test_message_temporary_size_is_checked_before_json_decode(monkeypatch):
    serializer = EvalClientSerializer(max_bot_audio_bytes=1280)
    monkeypatch.setattr(json, "loads", lambda *_: pytest.fail("parsed oversized message"))
    with pytest.raises(EvalAudioLimitError, match="message exceeded"):
        await serializer.deserialize(" " * ((1 << 20) + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["serializer", "before_ready", "after_ready"])
async def test_deeply_nested_audio_message_is_a_static_unscored_failure(path):
    wire = (
        '{"label":"rtvi-ai","type":"eval-bot-audio","data":'
        + "[" * 20000
        + '"PRIVATE-CANARY"'
        + "]" * 20000
        + "}"
    )
    serializer = EvalClientSerializer(max_bot_audio_bytes=1280)
    if path == "serializer":
        with pytest.raises(EvalAudioLimitError) as raised:
            await serializer.deserialize(wire)
        reason = str(raised.value)
    else:
        failures = []

        async def fail(reason):
            failures.append(reason)

        transport = EvalClientTransport(
            "ws://unused",
            WebsocketClientParams(audio_in_enabled=True, serializer=serializer),
            max_bot_audio_bytes=1280,
            audio_failure=fail,
        )
        transport._bot_ready = path == "after_ready"
        transport.input()  # The real pipeline constructs its input before reading messages.
        await transport._on_message(None, wire)
        assert len(failures) == 1
        reason = failures[0]
    assert reason == "The eval target sent an invalid message"
    assert "PRIVATE-CANARY" not in reason


@pytest.mark.asyncio
async def test_raw_buffer_is_bounded_before_extend_and_recording(monkeypatch):
    failures = []
    transport = input_transport(len(CHUNK), failures)
    recorder = types.SimpleNamespace(add_bot=AsyncMock())
    # The recorder API is synchronous; record only accepted inputs.
    recorded = []
    recorder.add_bot = lambda audio, _rate: recorded.append(audio)
    transport._recorder = recorder
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1))
    assert bytes(transport._bot_pcm) == CHUNK
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK * 100, RATE, 1))
    assert not transport._bot_pcm
    assert recorded == [CHUNK]
    assert transport._raw_audio_bytes == len(CHUNK)
    assert len(failures) == 1
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1))
    assert len(failures) == 1 and recorded == [CHUNK]


@pytest.mark.asyncio
async def test_interruption_never_refunds_the_run_admission_quota(monkeypatch):
    failures = []
    transport = input_transport(len(CHUNK) * 2, failures)
    control = AsyncMock()
    monkeypatch.setattr(WebsocketClientInputTransport, "push_frame", control)
    for _ in range(2):
        await transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1))
        await transport.push_frame(InputTransportMessageFrame(message={"type": "bot-interrupted"}))
        assert not transport._bot_pcm
    await transport.push_audio_frame(InputAudioRawFrame(b"\x01\x00", RATE, 1))
    assert transport._raw_audio_bytes == 2 * len(CHUNK)
    assert len(failures) == 1
    assert control.await_count == 2


@pytest.mark.asyncio
async def test_gap_fill_padding_and_silence_all_consume_pipeline_quota(monkeypatch):
    failures, frames = [], []
    transport = input_transport(len(CHUNK) * 3, failures)
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK[:4], RATE, 1))

    async def retain(_self, frame):
        frames.append(frame)

    monkeypatch.setattr(WebsocketClientInputTransport, "push_audio_frame", retain)
    await asyncio.wait_for(transport._fill_task_handler(), timeout=0.5)
    assert len(frames) == 3
    assert frames[0].audio[:4] == CHUNK[:4]
    assert frames[0].audio[4:] == bytes(len(CHUNK) - 4)
    assert all(frame.audio == bytes(len(CHUNK)) for frame in frames[1:])
    assert transport._paced_audio_bytes == sum(len(frame.audio) for frame in frames)
    assert len(failures) == 1 and "pipeline limit" in failures[0]


@pytest.mark.asyncio
async def test_slow_input_queue_does_not_hold_raw_failure_notification(monkeypatch):
    failures = []
    transport = input_transport(len(CHUNK), failures)
    entered = asyncio.Event()

    async def blocked(_self, _frame):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(WebsocketClientInputTransport, "push_audio_frame", blocked)
    task = asyncio.create_task(transport._fill_task_handler())
    transport._fill_task = task
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await transport.push_audio_frame(InputAudioRawFrame(CHUNK * 2, RATE, 1))
        assert len(failures) == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not transport._bot_pcm
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_cannot_append_after_a_timing_callback_loses_the_failure_race():
    failures, recorded = [], []
    transport = input_transport(len(CHUNK), failures)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def timing(*_args):
        entered.set()
        await resume.wait()

    transport._timing = timing
    transport._recorder = types.SimpleNamespace(add_bot=lambda *args: recorded.append(args))
    admitting = asyncio.create_task(transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1)))
    await entered.wait()
    await transport.fail_audio("The eval target exceeded its PCM admission limit")
    resume.set()
    await admitting
    assert not transport._bot_pcm and not recorded
    assert len(failures) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "end", "cancel"])
async def test_audio_admission_is_closed_idempotently_without_holding_controls(monkeypatch, ending):
    failures = []
    transport = input_transport(len(CHUNK), failures)
    transport._session.disconnect = AsyncMock()
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1))
    if ending == "end":
        await transport.stop(EndFrame())
    elif ending == "cancel":
        await transport.cancel(CancelFrame())
    else:
        await transport.close_audio()
    await transport.close_audio()
    await transport.push_audio_frame(InputAudioRawFrame(CHUNK, RATE, 1))
    assert not transport._bot_pcm
    assert transport._raw_audio_bytes == len(CHUNK)
    assert not failures
    control = AsyncMock()
    monkeypatch.setattr(WebsocketClientInputTransport, "push_frame", control)
    message = InputTransportMessageFrame(message={"type": "bot-stopped-speaking"})
    await transport.push_frame(message)
    control.assert_awaited_once_with(message, FrameDirection.DOWNSTREAM)


def caller(maximum, audio):
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=False, trace=trace)
    voice = types.SimpleNamespace(sample_rate=RATE, synthesize=AsyncMock(side_effect=audio))
    client = EvalClient(
        "ws://unused",
        stream=stream,
        trace=trace,
        user_tts=voice,
        session_params=EvalSessionParams(max_user_audio_buffer_bytes=maximum),
    )
    client._sink = types.SimpleNamespace(inject=AsyncMock())
    return client, stream, voice


@pytest.mark.asyncio
async def test_caller_aggregate_counts_interleaved_tts_and_exact_pause_bytes():
    client, stream, voice = caller(520, [b"a\0" * 50, b"b\0" * 50])
    await client.say_with_pauses(["first", 10, "second"])
    frames = [call.args[0] for call in client._sink.inject.await_args_list]
    audio = next(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))
    assert audio == b"a\0" * 50 + bytes(320) + b"b\0" * 50
    assert len(audio) == 520
    assert voice.synthesize.await_count == 2
    assert stream.harness_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts,audio",
    [
        ([10**1000], []),
        ([10, 10], []),
        (["first", 10, "second"], [b"a\0" * 50, b"b\0" * 50]),
        (["first"], [CHUNK * 2]),
    ],
)
async def test_caller_aggregate_refuses_before_oversized_extend_or_pause_allocation(parts, audio):
    client, stream, voice = caller(518, audio)
    with pytest.raises(EvalAudioLimitError, match="PCM buffer limit"):
        await client.say_with_pauses(parts)
    client._sink.inject.assert_not_awaited()
    assert stream.harness_error is not None
    assert stream.harness_error["text"] == "The eval caller exceeded its PCM buffer limit"


@pytest.mark.asyncio
async def test_caller_cancellation_never_injects_an_incomplete_aggregate():
    entered = asyncio.Event()
    client, stream, voice = caller(2560, [])

    async def synthesize(text):
        if text == "first":
            return CHUNK[:20]
        entered.set()
        await asyncio.Future()

    voice.synthesize.side_effect = synthesize
    task = asyncio.create_task(client.say_with_pauses(["first", 10, "second"]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client._sink.inject.assert_not_awaited()
    assert stream.harness_error is None


class HeldRecognizer(STTService):
    """Hold the system-input task as a stalled provider send would."""

    def __init__(self):
        super().__init__(
            sample_rate=RATE, settings=STTSettings(model="held-provider", language=None)
        )
        self.entered = asyncio.Event()
        self.released = asyncio.Event()

    async def run_stt(self, audio):
        self.entered.set()
        try:
            await asyncio.Future()
        finally:
            self.released.set()
        yield None


@pytest.mark.asyncio
@pytest.mark.parametrize("target_flood", [True, False])
async def test_actual_accepted_target_and_held_stt_fail_closed_with_bounded_pcm(
    monkeypatch, target_flood
):
    from tests.test_evals_audio_contract import _bot

    # Keep the accepted transport, serializer and target pipeline real. Removing
    # only the target's pacing creates a valid burst rather than a fake message.
    monkeypatch.setattr(FastAPIWebsocketOutputTransport, "_write_audio_sleep", AsyncMock())
    maximum = len(CHUNK) * 4
    peaks = []
    original = EvalClientInputTransport.push_audio_frame

    async def observe(self, frame):
        await original(self, frame)
        peaks.append(len(self._bot_pcm))

    monkeypatch.setattr(EvalClientInputTransport, "push_audio_frame", observe)
    recognizer = HeldRecognizer()
    async with _bot(greeting=target_flood) as (bot, url):
        scenario = EvalScriptScenario(
            name="bounded-input",
            bot_audio=True,
            user_audio=False,
            turns=[
                EvalScriptTurn(
                    user=None,
                    expect=[EvalExpectation(event="response", text_contains="Berlin")],
                )
            ],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            bot_stt=recognizer,
            params=EvalSessionParams(
                max_bot_audio_bytes=maximum,
                require_audio_support=True,
                default_timeout_ms=5000,
                bot_ready_timeout_s=3,
            ),
        )
        started = time.monotonic()
        result = await asyncio.wait_for(session.run(), timeout=4)
        elapsed = time.monotonic() - started
        assert not result.passed
        assert result.failures and all(failure.kind == "error" for failure in result.failures)
        assert elapsed < 3.5
        assert recognizer.entered.is_set() and recognizer.released.is_set()
        transport = session._client._active_transport.input()
        assert transport._raw_audio_bytes <= maximum
        assert transport._paced_audio_bytes <= maximum
        assert not transport._bot_pcm and transport._fill_task is None
        assert all(peak <= maximum for peak in peaks)
        if target_flood:
            assert transport._raw_audio_bytes > 0
        else:
            assert transport._raw_audio_bytes == 0
            assert transport._paced_audio_bytes == maximum
        await asyncio.wait_for(bot.closed.wait(), timeout=2)
        assert any(message.get("data", {}).get("t") == "eval-cancel" for message in bot.messages)


@pytest.mark.asyncio
async def test_normal_bounded_audio_still_completes_against_the_accepted_transport():
    from tests.test_evals_audio_contract import SPOKEN_ANSWER, _bot, _StreamingRecognizer

    recognizer = _StreamingRecognizer([SPOKEN_ANSWER])
    async with _bot() as (bot, url):
        scenario = EvalScriptScenario(
            name="bounded-valid",
            bot_audio=True,
            user_audio=False,
            turns=[
                EvalScriptTurn(
                    user="Answer",
                    expect=[
                        EvalExpectation(
                            event="response",
                            text_contains=SPOKEN_ANSWER,
                        )
                    ],
                )
            ],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            bot_stt=recognizer,
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10, require_audio_support=True),
        )
        result = await asyncio.wait_for(session.run(), timeout=8)
        assert result.passed, result
        assert recognizer.finals == [SPOKEN_ANSWER]
        await asyncio.wait_for(bot.closed.wait(), timeout=2)


@pytest.mark.asyncio
async def test_websocket_oversize_close_cannot_be_scored_as_an_agent_failure():
    from tests.test_evals_audio_contract import _bot

    recognizer = HeldRecognizer()
    async with _bot() as (bot, url):
        scenario = EvalScriptScenario(
            name="oversized-wire",
            bot_audio=True,
            user_audio=False,
            turns=[EvalScriptTurn(user=None, expect=[EvalExpectation(event="response")])],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            bot_stt=recognizer,
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10, require_audio_support=True),
        )
        running = asyncio.create_task(session.run())
        try:
            await asyncio.wait_for(session._client._handshake_over.wait(), timeout=2)
            # Serialize an intentionally oversized audio report through the
            # accepted transport's real urgent-message path, bypassing pacing.
            await bot.agent.push_frame(
                OutputTransportMessageUrgentFrame(message=json.loads(audio_message(bytes(1 << 20))))
            )
            result = await asyncio.wait_for(running, timeout=4)
            assert not result.passed
            assert all(failure.kind == "error" for failure in result.failures)
            assert recognizer.released.is_set()
            await asyncio.wait_for(bot.closed.wait(), timeout=2)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_before_quota_exhaustion_releases_a_held_recognizer_promptly():
    from tests.test_evals_audio_contract import _bot

    recognizer = HeldRecognizer()
    async with _bot() as (bot, url):
        scenario = EvalScriptScenario(
            name="cancel-held-audio",
            bot_audio=True,
            user_audio=False,
            turns=[EvalScriptTurn(user=None, expect=[EvalExpectation(event="response")])],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            bot_stt=recognizer,
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10, require_audio_support=True),
        )
        running = asyncio.create_task(session.run())
        await asyncio.wait_for(recognizer.entered.wait(), timeout=2)
        started = time.monotonic()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=4)
        assert time.monotonic() - started < 3.5
        assert recognizer.released.is_set()
        assert session._stream.harness_error is None
        await asyncio.wait_for(bot.closed.wait(), timeout=2)
