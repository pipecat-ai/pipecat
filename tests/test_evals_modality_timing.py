#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation measurements use delivery timing and require measurable evidence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.evals import client_transport
from pipecat.evals.client_transport import EvalClientInputTransport, EvalClientOutputTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario
from pipecat.evals.simulation_driver import EvalSimulationDriver
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
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


@pytest.mark.parametrize("user_audio", [False, True])
@pytest.mark.parametrize("bot_audio", [False, True])
@pytest.mark.parametrize("measure", ["latency", "words"])
def test_missing_required_measurement_cannot_pass_the_run(user_audio, bot_audio, measure):
    driver, _ = _driver(user_audio=user_audio, bot_audio=bot_audio)
    metric = driver._measure(EvalSimulationMetric(measure, measure=measure, max_value=3))
    driver._succeeded = True
    driver._metrics = [metric]
    result = driver.result(failures=[], duration_ms=1, events_seen=[], debug_log=[])

    assert metric.value is None
    assert metric.score is None
    assert metric.passed is False
    assert metric.failure_kind == "error"
    assert result.passed is False


def test_measurable_zero_still_satisfies_a_range():
    driver, _ = _driver()
    metric = driver._measure(EvalSimulationMetric("turns", measure="turns", max_value=3))
    assert metric.value == 0
    assert metric.score == 1
    assert metric.passed is True
    assert metric.failure_kind is None


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


def test_greeting_and_overlapping_output_do_not_manufacture_a_zero_latency():
    driver, stream = _driver(bot_audio=True)
    stream.events_seen.extend(
        [
            {"type": BOT_AUDIO_RECEIVED_EVENT, "at": 0.5},
            {"type": USER_TEXT_SENT_EVENT, "at": 2},
            {"type": BOT_AUDIO_RECEIVED_EVENT, "at": 2.1, "started_at": 1.5},
        ]
    )
    metric = driver._measure(EvalSimulationMetric("latency", measure="latency", max_value=3))
    assert metric.value is None
    assert metric.passed is False
    assert metric.failure_kind == "error"


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
async def test_audio_write_waits_for_paced_delivery_and_times_final_send():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_frame = AsyncMock(return_value=True)
    with patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()):
        output._send_task = asyncio.create_task(output._send_task_handler())
        writer = asyncio.create_task(
            output.write_audio_frame(
                OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
            )
        )
        await asyncio.sleep(0)
        assert not writer.done()
        await asyncio.wait_for(writer, 1)
        assert output._pending == b""
        sent_at = output._utterance_last_sent_at
        assert sent_at is not None
        await output.push_frame(BotStoppedSpeakingFrame())
        timing.assert_awaited_once_with(USER_AUDIO_SENT_EVENT, sent_at)
        output._send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await output._send_task


@pytest.mark.asyncio
async def test_interruption_releases_audio_writer_without_timing_discarded_audio():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_task = asyncio.create_task(asyncio.sleep(100))
    writer = asyncio.create_task(
        output.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
        )
    )
    await asyncio.sleep(0)
    assert not writer.done()
    output._utterance_last_sent_at = 123
    with (
        patch.object(WebsocketClientOutputTransport, "process_frame", new=AsyncMock()),
        patch.object(WebsocketClientOutputTransport, "push_frame", new=AsyncMock()),
    ):
        await output.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(writer, 1)
        await output.push_frame(BotStoppedSpeakingFrame())
    timing.assert_not_called()
    output._send_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await output._send_task


@pytest.mark.asyncio
async def test_transport_stop_releases_audio_writer_without_a_completion_anchor():
    timing = AsyncMock()
    output = _output(timing=timing)
    sender = asyncio.create_task(asyncio.sleep(100))
    output._send_task = sender

    async def cancel(task):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    output.cancel_task = cancel
    writer = asyncio.create_task(
        output.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
        )
    )
    await asyncio.sleep(0)
    assert not writer.done()
    await output._cancel_send_task()
    await asyncio.wait_for(writer, 1)
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
        writer = asyncio.create_task(
            output.write_audio_frame(
                OutputAudioRawFrame(audio=b"\x01\x00" * 640, sample_rate=16000, num_channels=1)
            )
        )
        await asyncio.wait_for(began.wait(), 1)
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        await output.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        await output.push_frame(BotStoppedSpeakingFrame())
        timing.assert_not_called()

        await asyncio.wait_for(
            output.write_audio_frame(
                OutputAudioRawFrame(audio=b"\x02\x00" * 640, sample_rate=16000, num_channels=1)
            ),
            1,
        )
        await output.push_frame(BotStoppedSpeakingFrame())
        assert timing.await_count == 1
        assert timing.await_args.args[0] == USER_AUDIO_SENT_EVENT
        output._send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await output._send_task


@pytest.mark.asyncio
async def test_audio_send_failure_releases_writer_and_never_records_delivery():
    timing = AsyncMock()
    output = _output(timing=timing)
    output._send_frame = AsyncMock(return_value=False)
    output.push_error = AsyncMock()
    output._send_task = asyncio.create_task(asyncio.sleep(100))
    sleeper = output._send_task
    writer = asyncio.create_task(
        output.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x01\x00" * 1280, sample_rate=16000, num_channels=1)
        )
    )
    await asyncio.sleep(0)
    await output._send_task_handler()
    await asyncio.wait_for(writer, 1)
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
def test_partial_latency_coverage_does_not_pass_required_bound(user_audio, bot_audio):
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
    metric = driver._measure(EvalSimulationMetric("latency", measure="latency", max_value=3))
    driver._succeeded = True
    driver._metrics = [metric]
    assert metric.value is None
    assert metric.passed is False
    assert metric.failure_kind == "error"
    assert driver.result(failures=[], duration_ms=1, events_seen=[], debug_log=[]).passed is False


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
    metric = driver._measure(EvalSimulationMetric("latency", measure="latency", max_value=3))
    assert metric.value == 1.0
    assert metric.passed is True


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
    metric = driver._measure(EvalSimulationMetric("latency", measure="latency", max_value=3))
    assert metric.value == 0.5
    assert metric.passed
