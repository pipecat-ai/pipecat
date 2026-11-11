#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation latency uses completed input delivery and received reply timing."""

import asyncio
import base64
import inspect
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.evals import client_transport
from pipecat.evals.client_transport import EvalClientInputTransport, EvalClientOutputTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.serializer import EvalClientSerializer
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.evals.simulation_driver import EvalSimulationDriver
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.websocket.client import (
    WebsocketClientOutputTransport,
    WebsocketClientParams,
)

USER_TEXT_SENT_EVENT = "user_text_sent"
USER_AUDIO_SENT_EVENT = "user_audio_sent"
BOT_AUDIO_RECEIVED_EVENT = "bot_audio_received"


def _driver(*, user_audio=False, bot_audio=False):
    scenario = EvalSimulationScenario(
        name="timing",
        persona="A caller",
        goal="Get an answer",
        success="The agent answered",
        user_audio=user_audio,
        bot_audio=bot_audio,
    )
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=bot_audio, trace=trace)
    driver = EvalSimulationDriver(
        simulation=scenario,
        persona=SimpleNamespace(),
        client=SimpleNamespace(),
        stream=stream,
        judge=None,
        trace=trace,
        progress=AsyncMock(),
    )
    return driver, stream


@pytest.mark.parametrize(
    ("user_audio", "bot_audio", "expected"),
    [(False, False, 3.5), (True, False, 0.5), (False, True, 4), (True, True, 1)],
)
def test_both_modality_axes_select_transport_observed_timing(user_audio, bot_audio, expected):
    driver, stream = _driver(user_audio=user_audio, bot_audio=bot_audio)
    stream.events_seen.extend(
        [
            {"type": "persona_turn", "at": 0.2},
            {"type": USER_TEXT_SENT_EVENT, "at": 1},
            {"type": USER_AUDIO_SENT_EVENT, "at": 4},
            {"type": BOT_AUDIO_RECEIVED_EVENT, "at": 5},
            {"type": "llm_response", "at": 9, "started_at": 4.5},
            {"type": "response", "at": 10},
            {"type": "user_stopped_speaking", "at": 11},
            {"type": "tts_response", "at": 14},
        ]
    )
    assert driver._reply_latencies() == [expected]


def test_greeting_without_a_reply_has_no_latency():
    driver, stream = _driver(bot_audio=True)
    stream.events_seen.extend(
        [
            {"type": BOT_AUDIO_RECEIVED_EVENT, "at": 0.5},
            {"type": "response", "at": 1.5, "text": "A greeting"},
            {"type": USER_TEXT_SENT_EVENT, "at": 2},
        ]
    )
    assert driver._measurement("latency")[0] is None


@pytest.mark.asyncio
async def test_stream_preserves_transport_time_before_delayed_pipeline_work():
    _, stream = _driver()
    stream._t0 = 10
    with patch("pipecat.evals.events.time.monotonic", return_value=20):
        await stream.transport_timing(USER_TEXT_SENT_EVENT, 11.5)
    assert stream.events_seen[-1]["at"] == 1.5


def _output(*, serializer=None, send=None, timing=None):
    session = SimpleNamespace(
        is_closing=False, is_connected=True, send=send or AsyncMock(return_value=True)
    )
    params = WebsocketClientParams(audio_out_enabled=True, serializer=serializer)
    output = EvalClientOutputTransport(None, session, params, timing=timing)
    output._sample_rate = 16000
    return output


@pytest.mark.asyncio
async def test_text_timing_requires_a_successful_send():
    timing = AsyncMock()
    output = _output(timing=timing)
    output.push_error = AsyncMock()
    frame = OutputTransportMessageUrgentFrame(message={"type": "send-text"})
    output._write_frame = AsyncMock(return_value=False)
    await output.send_message(frame)
    timing.assert_not_called()
    output.push_error.assert_awaited_once()
    output._write_frame.return_value = True
    await output.send_message(frame)
    assert timing.await_args.args[0] == USER_TEXT_SENT_EVENT


@pytest.mark.asyncio
async def test_websocket_write_returns_actual_send_failure_and_keeps_serializer_drop_semantics():
    serializer = MagicMock(spec=FrameSerializer)
    serializer.serialize = AsyncMock(return_value="payload")
    output = _output(serializer=serializer, send=AsyncMock(return_value=False))
    assert (
        await WebsocketClientOutputTransport._write_frame(
            output, OutputAudioRawFrame(audio=b"\x01\x00", sample_rate=16000, num_channels=1)
        )
        is False
    )
    output._session.send.assert_awaited_once_with("payload")
    serializer.serialize.return_value = None
    assert (
        await WebsocketClientOutputTransport._write_frame(
            output, OutputAudioRawFrame(audio=b"\x01\x00", sample_rate=16000, num_channels=1)
        )
        is True
    )
    output._session.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_utterance_end_waits_for_paced_delivery_and_times_final_send():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_frame = AsyncMock(return_value=True)
    frame = OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
    assert await output.write_audio_frame(frame) is False
    assert output._pending == frame.audio
    with patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()):
        output._send_task = asyncio.create_task(output._send_task_handler())
        finished = asyncio.create_task(output.push_frame(BotStoppedSpeakingFrame()))
        await asyncio.sleep(0)
        assert not finished.done()
        await asyncio.wait_for(finished, 1)
        assert output._pending == b""
        assert timing.await_count == 1
        assert timing.await_args.args[0] == USER_AUDIO_SENT_EVENT
        assert output._utterance_last_sent_at is None
        output._send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await output._send_task


@pytest.mark.asyncio
async def test_interruption_releases_utterance_end_without_timing_discarded_audio():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_task = asyncio.create_task(asyncio.sleep(100))
    await output.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
    )
    with (
        patch.object(WebsocketClientOutputTransport, "process_frame", new=AsyncMock()),
        patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()),
    ):
        finished = asyncio.create_task(output.push_frame(BotStoppedSpeakingFrame()))
        await asyncio.sleep(0)
        assert not finished.done()
        output._utterance_last_sent_at = 123
        await output.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(finished, 1)
    timing.assert_not_called()
    output._send_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await output._send_task


@pytest.mark.asyncio
async def test_transport_stop_releases_utterance_end_without_a_completion_anchor():
    timing = AsyncMock()
    output = _output(timing=timing)
    sender = asyncio.create_task(asyncio.sleep(100))
    output._send_task = sender

    async def cancel(task):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    output.cancel_task = cancel
    await output.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
    )
    with patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()):
        finished = asyncio.create_task(output.push_frame(BotStoppedSpeakingFrame()))
        await asyncio.sleep(0)
        assert not finished.done()
        await output._cancel_send_task()
        await asyncio.wait_for(finished, 1)
    assert not output._pending
    assert output._utterance_last_sent_at is None
    timing.assert_not_called()


@pytest.mark.asyncio
async def test_interrupted_inflight_send_cannot_anchor_the_following_turn():
    timing = AsyncMock()
    output = _output(timing=timing)
    began = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def send(frame):
        if any(frame.audio) and not began.is_set():
            began.set()
            await release.wait()
            finished.set()
        return True

    output._send_frame = send
    with (
        patch.object(WebsocketClientOutputTransport, "process_frame", new=AsyncMock()),
        patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()),
    ):
        output._send_task = asyncio.create_task(output._send_task_handler())
        await output.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x01\x00" * 640, sample_rate=16000, num_channels=1)
        )
        acknowledgment = asyncio.create_task(output.push_frame(BotStoppedSpeakingFrame()))
        await asyncio.wait_for(began.wait(), 1)
        acknowledgment.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acknowledgment
        await output.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        await output.push_frame(BotStoppedSpeakingFrame())
        timing.assert_not_called()

        await output.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x02\x00" * 640, sample_rate=16000, num_channels=1)
        )
        await asyncio.wait_for(output.push_frame(BotStoppedSpeakingFrame()), 1)
        assert timing.await_count == 1
        assert timing.await_args.args[0] == USER_AUDIO_SENT_EVENT
        output._send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await output._send_task


@pytest.mark.asyncio
async def test_audio_send_failure_releases_utterance_end_and_never_records_delivery():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_frame = AsyncMock(return_value=False)
    output.push_error = AsyncMock()
    output._send_task = asyncio.create_task(asyncio.sleep(100))
    sleeper = output._send_task
    await output.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
    )
    with patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()):
        finished = asyncio.create_task(output.push_frame(BotStoppedSpeakingFrame()))
        await asyncio.sleep(0)
        assert not finished.done()
        await output._send_task_handler()
        await asyncio.wait_for(finished, 1)
    assert not output._pending
    assert output._utterance_last_sent_at is None
    output.push_error.assert_awaited_once()
    timing.assert_not_called()
    sleeper.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sleeper


@pytest.mark.asyncio
async def test_raw_bot_audio_is_timed_before_gap_fill_and_only_once_per_spoken_segment():
    timing = AsyncMock()
    session = SimpleNamespace(is_closing=False, is_connected=True)
    transport = EvalClientInputTransport(
        None, session, WebsocketClientParams(audio_in_enabled=True), timing=timing
    )
    await transport.push_audio_frame(
        InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1)
    )
    timing.assert_not_called()
    for _ in range(2):
        await transport.push_audio_frame(
            InputAudioRawFrame(audio=b"\x01\x00" * 640, sample_rate=16000, num_channels=1)
        )
    assert timing.await_count == 1
    assert timing.await_args.args[0] == BOT_AUDIO_RECEIVED_EVENT
    with patch.object(
        client_transport.WebsocketClientInputTransport, "push_frame", new=AsyncMock()
    ):
        await transport.push_frame(BotStoppedSpeakingFrame())
    await transport.push_audio_frame(
        InputAudioRawFrame(audio=b"\x01\x00" * 640, sample_rate=16000, num_channels=1)
    )
    assert timing.await_count == 2


@pytest.mark.parametrize("user_audio", [False, True])
@pytest.mark.parametrize("bot_audio", [False, True])
def test_partial_latency_coverage_remains_unavailable(user_audio, bot_audio):
    driver, stream = _driver(user_audio=user_audio, bot_audio=bot_audio)
    input_type = "user_audio_sent" if user_audio else "user_text_sent"
    if bot_audio:
        events = [
            {"type": input_type, "at": 1.0},
            {"type": "bot_audio_received", "at": 1.5},
            {"type": "response", "at": 2.0, "text": "First actual spoken reply"},
            {"type": input_type, "at": 3.0},
            # The second real transcript exists, but its raw-audio timing is absent.
            {"type": "response", "at": 4.0, "text": "Second actual spoken reply"},
        ]
    else:
        events = [
            {"type": input_type, "at": 1.0},
            {"type": "llm_response", "at": 2.0, "started_at": 1.5, "text": "First reply"},
            {"type": input_type, "at": 3.0},
            # Output crossing the input boundary has no valid response-latency anchor.
            {"type": "llm_response", "at": 4.0, "started_at": 2.5, "text": "Second reply"},
        ]
    stream.events_seen.extend(events)
    assert driver._reply_latencies() == [0.5, None]
    assert driver._measurement("latency") == (None, "one or more replies could not be timed")


def test_audio_boundary_timestamp_order_not_append_order():
    driver, stream = _driver(user_audio=True, bot_audio=True)
    stream.events_seen.extend(
        [
            # Bot raw PCM can be observed before BaseOutput forwards the caller's
            # BotStoppedSpeakingFrame and appends the already-observed last-send time.
            {"type": "bot_audio_received", "at": 2.0},
            {"type": "user_audio_sent", "at": 1.0},
            {"type": "response", "at": 3.0, "text": "An actual recognized reply"},
        ]
    )
    assert driver._measurement("latency")[0] == 1.0


@pytest.mark.asyncio
async def test_submillisecond_audio_boundaries_keep_deferred_delivery_order():
    driver, stream = _driver(user_audio=True, bot_audio=True)
    stream._t0 = 10
    await stream.transport_timing(BOT_AUDIO_RECEIVED_EVENT, 11.0004)
    await stream.transport_timing(USER_AUDIO_SENT_EVENT, 11.0001)
    await stream.append({"type": "response", "at": 2, "text": "A spoken reply"})

    assert stream.events_seen[0]["at"] > stream.events_seen[1]["at"]
    assert driver._reply_latencies() == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_token_at,expected", [(11.0004, 0.0), (11.00005, None)])
async def test_submillisecond_text_token_keeps_its_order_against_delivery(first_token_at, expected):
    driver, stream = _driver()
    stream._t0 = 10
    await stream.transport_timing(USER_TEXT_SENT_EVENT, 11.0001)
    stream.frame_to_event(LLMFullResponseStartFrame())
    with patch("pipecat.evals.events.time.monotonic", return_value=first_token_at):
        stream.frame_to_event(LLMTextFrame(text="A reply"))
    event = stream.frame_to_event(LLMFullResponseEndFrame())
    await stream.append(event | {"at": 2})

    assert event["started_at"] == first_token_at - stream._t0
    assert driver._reply_latencies() == [expected]


@pytest.mark.asyncio
async def test_submillisecond_reply_completion_does_not_sort_before_its_input():
    driver, stream = _driver()
    stream._t0 = 10
    await stream.transport_timing(USER_TEXT_SENT_EVENT, 11.0001)
    stream.frame_to_event(LLMFullResponseStartFrame())
    with patch("pipecat.evals.events.time.monotonic", return_value=11.0002):
        stream.frame_to_event(LLMTextFrame(text="A reply"))
    event = stream.frame_to_event(LLMFullResponseEndFrame())
    with patch("pipecat.evals.events.time.monotonic", return_value=11.0004):
        await stream.append(event)

    assert stream.events_seen[1]["at"] > stream.events_seen[0]["at"]
    assert driver._reply_latencies() == [0.0]


def test_unanswered_closing_input_does_not_invent_an_extra_reply():
    driver, stream = _driver(bot_audio=True)
    stream.events_seen.extend(
        [
            {"type": USER_TEXT_SENT_EVENT, "at": 1},
            {"type": BOT_AUDIO_RECEIVED_EVENT, "at": 1.5},
            {"type": "response", "at": 2, "text": "An actual spoken reply"},
            {"type": USER_TEXT_SENT_EVENT, "at": 3},
        ]
    )
    assert driver._measurement("latency")[0] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [4, 2])
async def test_real_base_output_pcm_delivery_and_last_anchor(chunks):
    sent = []
    timing = []

    async def send(payload):
        message = json.loads(payload)
        if message["type"] == "raw-audio":
            sent.append((time.monotonic(), base64.b64decode(message["data"]["base64Audio"])))
        return True

    async def observed(kind, at):
        timing.append((kind, at))

    session = SimpleNamespace(
        is_closing=False,
        is_connected=True,
        setup=AsyncMock(),
        connect=AsyncMock(),
        disconnect=AsyncMock(),
        send=send,
    )
    parent = SimpleNamespace(cleanup=AsyncMock())
    timing_kwargs = (
        {"timing": observed}
        if "timing" in inspect.signature(EvalClientOutputTransport.__init__).parameters
        else {}
    )
    output = EvalClientOutputTransport(
        parent,
        session,
        WebsocketClientParams(
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=chunks,
            audio_out_end_silence_secs=0,
            serializer=EvalClientSerializer(),
        ),
        **timing_kwargs,
    )
    pcm = b"\x01\x00" * 1920  # exactly 120ms of speech at 16kHz
    await run_test(
        output,
        frames_to_send=[
            TTSStartedFrame(),
            TTSAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1),
            TTSStoppedFrame(),
            SleepFrame(sleep=0.4),
        ],
        pipeline_params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
    )
    voiced = [(at, data) for at, data in sent if any(data)]
    assert b"".join(data for _, data in voiced) == pcm
    if timing_kwargs:
        assert len(timing) == 1
        assert timing[0][0] == "user_audio_sent"
        assert 0 <= timing[0][1] - voiced[-1][0] < 0.01
