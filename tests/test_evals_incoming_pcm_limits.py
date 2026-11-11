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
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import websockets
from pydantic import ValidationError
from starlette.websockets import WebSocket

from pipecat.evals.client import EvalClient
from pipecat.evals.client_transport import FRAME_S, EvalClientInputTransport, EvalClientTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import EvalExpectation, EvalScriptScenario, EvalScriptTurn
from pipecat.evals.serializer import EvalClientSerializer, EvalSerializer, parse_eval_connect
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import (
    WebsocketClientInputTransport,
    WebsocketClientParams,
    WebsocketClientSession,
)
from pipecat.transports.websocket.fastapi import FastAPIWebsocketOutputTransport
from pipecat.workers.runner import WorkerRunner

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


@pytest.mark.parametrize("field", ["max_bot_audio_bytes"])
@pytest.mark.parametrize("value", [0, -2, True, False, 1, 2.0, float("inf"), float("nan"), "1280"])
def test_audio_byte_limits_require_positive_even_integers(field, value):
    with pytest.raises(ValidationError):
        EvalSessionParams(**{field: value})


def test_audio_byte_limits_are_optional_and_serializable():
    defaults = EvalSessionParams()
    assert defaults.max_bot_audio_bytes is None
    limits = EvalSessionParams(max_bot_audio_bytes=1280)
    assert EvalSessionParams.model_validate_json(limits.model_dump_json()) == limits


@pytest.mark.parametrize("kind", ["serializer", "input", "transport"])
@pytest.mark.parametrize("value", [0, True, 1, 2.0])
def test_direct_constructors_cannot_ignore_invalid_byte_limits(kind, value):
    with pytest.raises(ValueError, match="positive even integers"):
        if kind == "serializer":
            EvalClientSerializer(max_bot_audio_bytes=value)
        elif kind == "input":
            EvalClientInputTransport(
                None, types.SimpleNamespace(), WebsocketClientParams(), max_bot_audio_bytes=value
            )
        else:
            EvalClientTransport("ws://unused", max_bot_audio_bytes=value)


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
    with pytest.raises(ValueError, match="PCM admission"):
        await serializer.deserialize(message)
    serializer._resample.assert_not_awaited()


@pytest.mark.asyncio
async def test_expansion_is_checked_before_native_resampling(monkeypatch):
    serializer = EvalClientSerializer(max_bot_audio_bytes=len(CHUNK))
    message = audio_message(b"\x01\x00", rate=1)
    monkeypatch.setattr(base64, "b64decode", lambda *_args, **_kw: pytest.fail("decoded expansion"))
    serializer._resample = AsyncMock(side_effect=AssertionError("expanded before limit"))
    with pytest.raises(ValueError, match="PCM admission"):
        await serializer.deserialize(message)
    serializer._resample.assert_not_awaited()


@pytest.mark.asyncio
async def test_normalized_wire_quota_allows_a_valid_higher_sample_rate():
    serializer = EvalClientSerializer(max_bot_audio_bytes=len(CHUNK))
    frame = await serializer.deserialize(audio_message(CHUNK * 3, rate=48000))
    assert len(frame.audio) == len(CHUNK)
    assert frame.sample_rate == RATE and frame.num_channels == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [b"RIFF", b"RIFX", b"OggS", b"fLaC", b"ID3\0"])
async def test_pcm_samples_are_not_rejected_for_resembling_a_container_header(prefix):
    pcm = prefix + bytes(12)
    frame = await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(audio_message(pcm))
    assert frame.audio == pcm


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 11025, 22050, 24000, 44100, 48000, 96000, 192000])
async def test_actual_stream_resampler_respects_each_chunk_projection(rate):
    from pipecat.audio.utils import create_stream_resampler

    resampler = create_stream_resampler()
    for samples in (1, 2, 3, 19, 640, 1, 1, 997):
        output = await resampler.resample(b"\x01\x00" * samples, rate, RATE)
        projected = ((samples * RATE + rate - 1) // rate) * 2
        assert len(output) <= projected
        assert len(output) % 2 == 0


@pytest.mark.asyncio
async def test_extreme_positive_native_rate_fails_with_a_static_reason():
    # The locked native filter rejects excessive tap counts before allocating
    # its coefficient arrays. Two bytes keep the PCM allocation small too.
    with pytest.raises(ValueError, match="^The eval target sent invalid PCM audio$"):
        await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(
            audio_message(b"\x01\x00", rate=2**31 - 1)
        )


@pytest.mark.asyncio
async def test_resampler_rate_change_cannot_allocate_another_filter_or_reset_the_quota():
    serializer = EvalClientSerializer(max_bot_audio_bytes=4096)
    await serializer.deserialize(audio_message(CHUNK, rate=48000))
    prior = serializer._bot_audio_bytes
    with pytest.raises(ValueError, match="^The eval target sent invalid PCM audio$"):
        await serializer.deserialize(audio_message(CHUNK, rate=44100))
    assert serializer._bot_audio_bytes == prior


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [None, "service", "target"])
async def test_pre_ready_limit_failure_is_prompt_and_preserves_the_first_cause(prior):
    from pipecat.evals.client import EvalBotUnavailableError

    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=True, trace=trace)
    client = EvalClient(
        "ws://unused",
        stream=stream,
        trace=trace,
        session_params=EvalSessionParams(max_bot_audio_bytes=1280, bot_ready_timeout_s=30),
    )
    if prior == "service":
        stream.harness_failed("first service error")
    elif prior == "target":
        client._refuse("closed_before_ready", "target closed first")
    transport = client._transport()
    transport.input()
    await transport._on_message(None, " " * ((1 << 20) + 1))
    with pytest.raises(EvalBotUnavailableError) as raised:
        await asyncio.wait_for(client.handshake(), 0.5)
    if prior == "target":
        assert raised.value.kind == "closed_before_ready"
        assert str(raised.value) == "target closed first"
    else:
        assert raised.value.kind == "error"
        expected = (
            "first service error"
            if prior == "service"
            else "The eval target audio message exceeded its byte limit"
        )
        assert str(raised.value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [0, -1, True, 16000.0, "16000", None, 2**31])
async def test_bounded_wire_rate_requires_a_positive_native_integer(rate):
    with pytest.raises(ValueError, match="invalid PCM"):
        await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(audio_message(rate=rate))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio,extra",
    [
        (b"", {}),
        (b"\x01", {}),
        (CHUNK, {"numChannels": 2}),
        (CHUNK, {"channels": True}),
    ],
)
async def test_invalid_bounded_wire_audio_fails_without_payload_text(audio, extra):
    with pytest.raises(ValueError, match="invalid PCM") as raised:
        await EvalClientSerializer(max_bot_audio_bytes=1280).deserialize(
            audio_message(audio, **extra)
        )
    assert "PRIVATE-CANARY" not in str(raised.value)


@pytest.mark.asyncio
async def test_message_temporary_size_is_checked_before_json_decode(monkeypatch):
    serializer = EvalClientSerializer(max_bot_audio_bytes=1280)
    monkeypatch.setattr(json, "loads", lambda *_: pytest.fail("parsed oversized message"))
    with pytest.raises(ValueError, match="message exceeded"):
        await serializer.deserialize(" " * ((1 << 20) + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["serializer", "before_ready", "after_ready"])
@pytest.mark.parametrize("invalid_unicode", [False, True])
async def test_message_byte_ceiling_and_unicode_validation_precede_json(
    monkeypatch, path, invalid_unicode
):
    wire = json.dumps(
        {"label": "rtvi-ai", "type": "unused", "data": "😀" * 270000}, ensure_ascii=False
    )
    assert len(wire) < 1 << 20 < len(wire.encode("utf-8"))
    if invalid_unicode:
        wire = '{"data":"\ud800PRIVATE-CANARY"}'
    reason = (
        "The eval target sent an invalid message"
        if invalid_unicode
        else "The eval target audio message exceeded its byte limit"
    )
    monkeypatch.setattr(json, "loads", lambda *_: pytest.fail("parsed invalid bounded message"))
    serializer = EvalClientSerializer(max_bot_audio_bytes=1280)
    if path == "serializer":
        with pytest.raises(ValueError, match=reason):
            await serializer.deserialize(wire)
    else:
        failures = []

        async def fail(message):
            failures.append(message)

        transport = EvalClientTransport(
            "ws://unused",
            WebsocketClientParams(serializer=serializer),
            max_bot_audio_bytes=1280,
            audio_failure=fail,
        )
        transport._bot_ready = path == "after_ready"
        await transport._on_message(None, wire)
        assert failures == [reason]


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
        with pytest.raises(ValueError) as raised:
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
@pytest.mark.parametrize(
    "rate,channels", [(0, 1), (1, 1), (10**100, 1), (16000.0, 1), (RATE, 0), (RATE, 2)]
)
async def test_bounded_gap_fill_rejects_invalid_format_before_silence_allocation(
    monkeypatch, rate, channels
):
    failures = []
    transport = input_transport(len(CHUNK), failures)
    transport._sample_rate = rate
    transport._params.audio_in_channels = channels
    frames = AsyncMock()
    monkeypatch.setattr(WebsocketClientInputTransport, "push_audio_frame", frames)
    monkeypatch.setattr(
        "pipecat.evals.client_transport._sleep_to_next_tick",
        AsyncMock(side_effect=AssertionError("paced invalid format")),
    )

    # The extreme rate exceeds Py_ssize_t, so an unguarded bytes multiplication
    # raises before allocation rather than asking the allocator for a large buffer.
    await transport._fill_task_handler()

    frames.assert_not_awaited()
    assert failures == ["The eval harness has an invalid PCM format"]
    assert transport._paced_audio_bytes == 0


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
    cancel_attempts, cancel_completed = [], []
    original_send = WebsocketClientSession.send

    async def delayed_cancel(self, message):
        parsed = json.loads(message)
        cancelling = parsed.get("data", {}).get("t") == "eval-cancel"
        if cancelling:
            cancel_attempts.append(parsed)
            await asyncio.sleep(0.05)
        sent = await original_send(self, message)
        if cancelling and sent:
            cancel_completed.append(parsed)
        return sent

    monkeypatch.setattr(WebsocketClientSession, "send", delayed_cancel)
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
        assert len(cancel_attempts) == len(cancel_completed) == 1
        assert (
            sum(message.get("data", {}).get("t") == "eval-cancel" for message in bot.messages) == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_send", [True, False])
async def test_stop_joins_one_cancel_send_after_its_origin_task_is_cancelled(complete_send):
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completed = []

    async def send(message):
        entered.set()
        try:
            await release.wait()
            completed.append(message)
        finally:
            finished.set()

    trace = EvalTrace()
    client = EvalClient(
        "ws://unused",
        stream=EvalEventStream(bot_audio=True, trace=trace),
        trace=trace,
        session_params=EvalSessionParams(max_bot_audio_bytes=1280),
    )
    output = types.SimpleNamespace(send_message=AsyncMock(side_effect=send))
    transport = types.SimpleNamespace(
        output=lambda: output, input=lambda: types.SimpleNamespace(close_audio=AsyncMock())
    )
    client._active_transport = transport
    origin = asyncio.create_task(
        client._audio_failure("The eval harness exceeded its PCM pipeline limit")
    )
    stopping = None
    started = time.monotonic()
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        origin.cancel()
        await asyncio.gather(origin, return_exceptions=True)
        stopping = asyncio.create_task(client.stop())
        await asyncio.sleep(0)
        assert not stopping.done()
        assert not finished.is_set()
        if complete_send:
            release.set()
        await asyncio.wait_for(stopping, 1.5)
        assert time.monotonic() - started < 1.5
        output.send_message.assert_awaited_once()
        assert len(completed) == int(complete_send)
        assert finished.is_set()
    finally:
        release.set()
        origin.cancel()
        if stopping is not None:
            stopping.cancel()
        await asyncio.gather(
            origin, *([stopping] if stopping is not None else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_normal_bounded_audio_still_completes_against_the_accepted_transport():

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
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10),
        )
        result = await asyncio.wait_for(session.run(), timeout=8)
        assert result.passed, result
        assert recognizer.finals == [SPOKEN_ANSWER]
        await asyncio.wait_for(bot.closed.wait(), timeout=2)


@pytest.mark.asyncio
async def test_websocket_oversize_close_cannot_be_scored_as_an_agent_failure():

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
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10),
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
async def test_real_oversized_websocket_message_before_ready_refuses_promptly():
    closed = asyncio.Event()

    async def target(connection):
        try:
            await connection.send(audio_message(bytes(1 << 20)))
            await connection.wait_closed()
        except websockets.ConnectionClosed:
            pass
        finally:
            closed.set()

    async with websockets.serve(target, "127.0.0.1", 0) as server:
        scenario = EvalScriptScenario(
            name="oversized-before-ready",
            turns=[EvalScriptTurn(user=None, expect=[EvalExpectation(event="llm_response")])],
        )
        session = EvalSession.from_scenario(
            scenario,
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            params=EvalSessionParams(max_bot_audio_bytes=1280, bot_ready_timeout_s=30),
        )
        started = time.monotonic()
        result = await asyncio.wait_for(session.run(), 4)
        assert time.monotonic() - started < 3.5
        assert len(result.failures) == 1
        assert result.failures[0].kind == "error"
        assert result.failures[0].event_name == "<bot-ready>"
        assert result.failures[0].reason == "The eval audio connection exceeded its message limit"
        assert session._stream.harness_error == result.failures[0].reason
        await asyncio.wait_for(closed.wait(), 1)


@pytest.mark.asyncio
async def test_cancellation_before_quota_exhaustion_releases_a_held_recognizer_promptly():

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
            params=EvalSessionParams(max_bot_audio_bytes=RATE * 2 * 10),
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


SAMPLE_RATE = RATE
TEXT_GREETING = "A greeting"
TEXT_ANSWER = "An answer"
SPOKEN_ANSWER = "The spoken answer is Berlin."


def _speech_pcm():
    with wave.open(str(Path(__file__).parent / "assets/continuous_speech_16k.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return wav.readframes(SAMPLE_RATE * 3 // 2)


class _StreamingRecognizer(STTService):
    """Emit a final transcript asynchronously after a complete incoming segment."""

    def __init__(self, replies, final_delay=0.05):
        super().__init__(settings=STTSettings(model="deterministic-stream"))
        self.replies = iter(replies)
        self.final_delay = final_delay
        self.segments = 0
        self.audio_bytes = 0
        self.active = False
        self.finals = []

    async def _final(self, text):
        await asyncio.sleep(self.final_delay)
        self.finals.append(text)
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="", timestamp="2026-11-10T00:00:00Z")
        )

    async def run_stt(self, audio):
        if any(audio):
            self.audio_bytes += len(audio)
            self.active = True
        elif self.active:
            self.active = False
            self.segments += 1
            self.create_task(self._final(next(self.replies)), name="test-final-transcript")
        yield None


class _Agent(FrameProcessor):
    """A deterministic agent that handles real text/audio input frames."""

    def __init__(self):
        super().__init__()
        self.skip_tts = False
        self.audio_bytes = 0
        self.audio_received = asyncio.Event()
        self.speaking = False
        self.replies = 0
        self.cancelled = asyncio.Event()

    async def reply(self, text):
        self.replies += 1
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(LLMFullResponseEndFrame())
        if not self.skip_tts:
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSAudioRawFrame(audio=_speech_pcm(), sample_rate=SAMPLE_RATE, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMConfigureOutputFrame):
            self.skip_tts = frame.skip_tts
        elif isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm:
            self.create_task(self.reply(TEXT_ANSWER), name="test-agent-text-reply")
        elif isinstance(frame, InputAudioRawFrame):
            if any(frame.audio):
                self.audio_bytes += len(frame.audio)
                self.audio_received.set()
                if not self.speaking:
                    self.speaking = True
                    await self.push_frame(UserStartedSpeakingFrame())
            elif self.speaking:
                self.speaking = False
                await self.push_frame(UserStoppedSpeakingFrame())
                self.create_task(self.reply(TEXT_ANSWER), name="test-agent-audio-reply")
        elif isinstance(frame, CancelFrame):
            self.cancelled.set()
        await self.push_frame(frame, direction)


class _AcceptedWebSocketBot:
    def __init__(self, *, greeting, audio_input=True):
        self.greeting = greeting
        self.audio_input = audio_input
        self.agent = None
        self.flags = None
        self.closed = asyncio.Event()
        self.connected = asyncio.Event()
        self.messages = []

    async def handle(self, connection):
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "websocket.connect"}
            try:
                payload = await connection.recv()
            except websockets.ConnectionClosed:
                return {"type": "websocket.disconnect", "code": connection.close_code or 1000}
            if isinstance(payload, str):
                self.messages.append(json.loads(payload))
                return {"type": "websocket.receive", "text": payload}
            return {"type": "websocket.receive", "bytes": payload}

        async def send(message):
            if message["type"] == "websocket.send":
                await connection.send(message.get("text", message.get("bytes")))
            elif message["type"] == "websocket.close":
                await connection.close(code=message.get("code", 1000))

        socket = WebSocket(
            {"type": "websocket", "path": "/", "query_string": b"", "headers": []},
            receive,
            send,
        )
        await socket.accept()
        self.flags = parse_eval_connect(await socket.receive_json())
        assert self.flags is not None
        transport = EvalFastAPIWebsocketTransport(
            socket,
            EvalTransportParams(
                serializer=EvalSerializer(),
                audio_in_enabled=self.audio_input,
                audio_out_enabled=True,
                audio_in_sample_rate=SAMPLE_RATE,
                audio_out_sample_rate=SAMPLE_RATE,
            ),
            self.flags,
        )
        self.agent = _Agent()
        self.connected.set()
        worker = PipelineWorker(
            Pipeline([transport.input(), self.agent, transport.output()]),
            params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            ),
        )
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)

        @worker.rtvi.event_handler("on_client_ready")
        async def ready(_rtvi):
            if self.greeting:
                await self.agent.reply(TEXT_GREETING)

        @transport.event_handler("on_client_disconnected")
        async def disconnected(_transport, _websocket):
            await worker.cancel()

        try:
            await runner.add_workers(worker)
            await runner.run()
        finally:
            self.closed.set()


@asynccontextmanager
async def _bot(*, greeting=False, audio_input=True):
    bot = _AcceptedWebSocketBot(greeting=greeting, audio_input=audio_input)
    async with websockets.serve(bot.handle, "127.0.0.1", 0) as server:
        yield bot, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
