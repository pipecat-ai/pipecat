#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise Dynamo text-realtime streaming and speculative input without a backend."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EagerEndOfTurnCancelFrame,
    EndFrame,
    ErrorFrame,
    HeartbeatFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.dynamo import llm as dynamo_llm
from pipecat.services.dynamo.llm import DynamoRealtimeLLMService, DynamoStreamingPrefillProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager


class _Socket:
    """Queue server events in response to the actual client wire messages."""

    def __init__(self):
        self.sent = []
        self.events = asyncio.Queue()
        self.events.put_nowait({"type": "session.created", "session": {"id": "session-1"}})
        self.response_events = [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": "."},
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15},
                },
            },
        ]
        self.closed = False

    async def send_json(self, event):
        self.sent.append(deepcopy(event))
        if event["type"] == "session.update":
            self.events.put_nowait({"type": "session.updated"})
        elif event["type"] == "response.create":
            for response in self.response_events:
                self.events.put_nowait(response)

    async def receive_json(self):
        return await self.events.get()

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_transport(monkeypatch):
    """Replace only the network transport, preserving Pipecat's runtime behavior."""
    sockets = []
    sessions = []

    class Session:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            sessions.append(self)

        async def ws_connect(self, url, **kwargs):
            socket = _Socket()
            sockets.append(socket)
            return socket

        async def close(self):
            self.closed = True

    monkeypatch.setattr(dynamo_llm.aiohttp, "ClientSession", Session)
    return sockets, sessions


@pytest.fixture
def blocked_response(monkeypatch):
    """Hold generated text until the test releases the response."""
    waiting = asyncio.Event()
    release = asyncio.Event()
    receive_json = _Socket.receive_json

    async def receive(self):
        event = await receive_json(self)
        if event["type"] == "response.output_text.delta":
            waiting.set()
            await release.wait()
        return event

    monkeypatch.setattr(_Socket, "receive_json", receive)
    return waiting, release


@pytest.fixture
def make_service(fake_transport):
    """Construct services with isolated fake connections and standard metric spies."""
    sockets, sessions = fake_transport

    def make(**kwargs):
        settings = kwargs.pop(
            "settings", DynamoRealtimeLLMService.Settings(model="dynamo-model", max_tokens=64)
        )
        service = DynamoRealtimeLLMService(
            base_url="http://localhost:8000/v1", settings=settings, **kwargs
        )
        service.create_task = lambda coroutine, **kwargs: asyncio.create_task(coroutine)

        async def cancel_task(task, **kwargs):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        service.cancel_task = cancel_task
        service._push_llm_text = AsyncMock()
        service.push_frame = AsyncMock()
        service.push_error = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.cancel_ttfb_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()
        return service, sockets, sessions

    return make


@pytest_asyncio.fixture
async def managed_service(fake_transport):
    """Run a service with real task management and lifecycle handling."""
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test"),
    )
    manager = TaskManager()
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    try:
        await service.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        yield service
    finally:
        await service.cleanup()
        assert not manager.current_tasks()


async def _wait_for(predicate):
    """Give background work a bounded opportunity to make progress."""
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


async def _prefill(service, sockets, context, text):
    """Wait for an actual speculative append, not merely task scheduling."""
    previous = sum(
        event["type"] == "input_text.append" for socket in sockets for event in socket.sent
    )
    service.prefill(context, text)
    await _wait_for(
        lambda: (
            sum(event["type"] == "input_text.append" for socket in sockets for event in socket.sent)
            > previous
        )
    )


def _items(socket):
    return [event["item"] for event in socket.sent if event["type"] == "conversation.item.create"]


@pytest.mark.asyncio
async def test_matching_prefill_commits_on_same_connection(make_service):
    """Only authoritative turn completion starts output generation."""
    service, sockets, sessions = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    await _prefill(service, sockets, context, "Hello")
    await _prefill(service, sockets, context, " there.")
    assert not any(event["type"] == "response.create" for event in sockets[0].sent)
    context.add_message({"role": "user", "content": "Hello there."})

    await service._process_context(context)

    assert len(sockets) == 1
    assert [event["type"] for event in sockets[0].sent][-2:] == [
        "input_text.commit",
        "response.create",
    ]
    assert [item["content"][0]["text"] for item in _items(sockets[0])] == ["Be helpful."]
    assert [call.args[0] for call in service._push_llm_text.await_args_list] == ["Hello", "."]
    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("final_text", ["Turn right.", "Turn left at the corner."])
async def test_correction_or_final_suffix_uses_authoritative_atomic_text(make_service, final_text):
    """A final correction never starts an additional speculative generation."""
    service, sockets, _ = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    await _prefill(service, sockets, context, "Turn left")
    context.add_message({"role": "user", "content": final_text})

    await service._process_context(context)

    events = sockets[0].sent
    assert [event["type"] for event in events][-3:] == [
        "input_text.clear",
        "conversation.item.create",
        "response.create",
    ]
    assert [event["text"] for event in events if event["type"] == "input_text.append"] == [
        "Turn left"
    ]
    assert _items(sockets[0])[-1]["content"][0]["text"] == final_text


@pytest.mark.asyncio
async def test_no_interim_input_uses_atomic_final_text(make_service):
    """Greetings and final-only ASR do not incur speculative work."""
    service, sockets, _ = make_service()
    await service._process_context(LLMContext([{"role": "user", "content": "Hello."}]))

    assert not any(event["type"].startswith("input_text.") for event in sockets[0].sent)
    assert _items(sockets[0])[-1]["content"][0]["text"] == "Hello."
    update = next(event for event in sockets[0].sent if event["type"] == "session.update")
    assert update["session"]["model"] == "dynamo-model"
    assert update["session"]["output_modalities"] == ["text"]
    assert update["session"]["max_output_tokens"] == 64


@pytest.mark.asyncio
async def test_changed_context_reopens_with_authoritative_history(make_service):
    """Summarization or interruption cannot leave a stale speculative prompt."""
    service, sockets, sessions = make_service()
    context = LLMContext([{"role": "system", "content": "Old instructions."}])
    await _prefill(service, sockets, context, "Hello")
    context.set_messages(
        [
            {"role": "system", "content": "New instructions."},
            {"role": "assistant", "content": "Previous answer."},
            {"role": "user", "content": "Hello."},
        ]
    )

    await service._process_context(context)

    assert len(sockets) == 2
    assert all(socket.closed for socket in sockets)
    assert all(session.closed for session in sessions)
    assert [item["content"][0]["text"] for item in _items(sockets[1])] == [
        "New instructions.",
        "Previous answer.",
        "Hello.",
    ]
    assert _items(sockets[1])[1]["content"][0]["type"] == "output_text"
    assert not any(event["type"].startswith("input_text.") for event in sockets[1].sent)


@pytest.mark.asyncio
async def test_prefill_processor_forwards_without_waiting_for_connection(make_service, monkeypatch):
    """Slow LLM setup does not hold ASR frames upstream of turn detection."""
    monkeypatch.setattr(FrameProcessor, "process_frame", AsyncMock())
    service, _, _ = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    connecting = asyncio.Event()
    release = asyncio.Event()

    async def stalled_open(history):
        connecting.set()
        await release.wait()

    service._open = stalled_open
    processor = DynamoStreamingPrefillProcessor(llm=service, context=context)
    processor.push_frame = AsyncMock()
    frame = InterimTranscriptionFrame("Hello", "user", "")
    try:
        await asyncio.wait_for(processor.process_frame(frame, FrameDirection.DOWNSTREAM), 0.1)
        processor.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(connecting.wait(), 1)
    finally:
        await service._stop_prefill()
        await service._close()


@pytest.mark.asyncio
async def test_prefill_processor_ignores_upstream_transcription(make_service, monkeypatch):
    """Upstream control traffic must not create another speculative append."""
    monkeypatch.setattr(FrameProcessor, "process_frame", AsyncMock())
    service, _, _ = make_service()
    service.prefill = Mock()
    processor = DynamoStreamingPrefillProcessor(llm=service, context=LLMContext())
    processor.push_frame = AsyncMock()
    frame = InterimTranscriptionFrame("Hello", "user", "")

    await processor.process_frame(frame, FrameDirection.UPSTREAM)

    service.prefill.assert_not_called()
    processor.push_frame.assert_awaited_once_with(frame, FrameDirection.UPSTREAM)


@pytest.mark.asyncio
async def test_prefill_coalesces_updates_before_sending(make_service):
    """A burst of partial deltas preserves exact text without one send per token."""
    service, sockets, _ = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    for delta in ["Hello", " ", "there", "."]:
        service.prefill(context, delta)
    try:
        await _wait_for(
            lambda: sockets and any(e["type"] == "input_text.append" for e in sockets[0].sent)
        )
        appends = [event for event in sockets[0].sent if event["type"] == "input_text.append"]
        assert [event["text"] for event in appends] == ["Hello there."]
    finally:
        await service._stop_prefill()
        await service._close()


@pytest.mark.asyncio
async def test_finalization_cancels_stalled_speculation(make_service):
    """Final text can generate even if the prefill connection is still opening."""
    service, sockets, _ = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    original_open = service._open
    connecting = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def first_open_stalls(history):
        nonlocal calls
        calls += 1
        if calls == 1:
            connecting.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        else:
            await original_open(history)

    service._open = first_open_stalls
    service.prefill(context, "Hello")
    await asyncio.wait_for(connecting.wait(), 1)
    context.add_message({"role": "user", "content": "Hello there."})
    try:
        await asyncio.wait_for(service._process_context(context), 1)
        assert cancelled.is_set()
        assert _items(sockets[0])[-1]["content"][0]["text"] == "Hello there."
        assert not any(event["type"].startswith("input_text.") for event in sockets[0].sent)
    finally:
        await service._stop_prefill()
        await service._close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,frame_type", [("stop", EndFrame), ("cancel", CancelFrame)])
async def test_pipeline_lifecycle_closes_speculative_connection(
    make_service, monkeypatch, method, frame_type
):
    """Abandoned speculative turns release both WebSocket and HTTP sessions."""
    monkeypatch.setattr(LLMService, method, AsyncMock())
    service, sockets, sessions = make_service()
    await _prefill(service, sockets, LLMContext(), "Hello")

    await getattr(service, method)(frame_type())

    assert sockets[0].closed and sessions[0].closed
    assert service._text == ""


@pytest.mark.asyncio
async def test_interruption_discards_prefill_before_next_turn(make_service, monkeypatch):
    """An interrupted partial is not included in the next user turn."""
    monkeypatch.setattr(LLMService, "_handle_interruptions", AsyncMock())
    service, sockets, sessions = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    await _prefill(service, sockets, context, "Abandoned request")
    await service._handle_interruptions(InterruptionFrame())
    assert sockets[0].closed and sessions[0].closed
    context.add_message({"role": "user", "content": "New request."})

    await service._process_context(context)

    assert len(sockets) == 2
    assert [item["content"][0]["text"] for item in _items(sockets[1])] == [
        "Be helpful.",
        "New request.",
    ]


@pytest.mark.asyncio
async def test_server_error_releases_connection(make_service):
    """A failed response cannot strand speculative resources."""
    service, sockets, sessions = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    await _prefill(service, sockets, context, "Hello")
    sockets[0].response_events = [{"type": "error", "error": {"message": "worker failed"}}]
    context.add_message({"role": "user", "content": "Hello"})

    with pytest.raises(Exception, match="worker failed"):
        await service._process_context(context)

    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
async def test_invalid_final_context_releases_prefill(make_service):
    """Validation failure after speculation still runs cleanup."""
    service, sockets, sessions = make_service()
    context = LLMContext([{"role": "system", "content": "Be helpful."}])
    await _prefill(service, sockets, context, "Hello")

    with pytest.raises(ValueError):
        await service._process_context(context)

    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
async def test_response_usage_is_reported_through_pipecat_metrics(make_service):
    """Realtime token usage reaches standard metrics, not just demo callbacks."""
    service, _, _ = make_service()
    await service._process_context(LLMContext([{"role": "user", "content": "Hello."}]))

    service.start_llm_usage_metrics.assert_awaited_once()
    usage = service.start_llm_usage_metrics.await_args.args[0]
    assert usage.prompt_tokens == 13
    assert usage.completion_tokens == 2
    assert usage.total_tokens == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_context_frame_always_finishes_response_lifecycle(make_service, monkeypatch, fail):
    """Success and errors preserve response framing and processing metrics."""
    monkeypatch.setattr(LLMService, "process_frame", AsyncMock())
    service, _, _ = make_service()
    if fail:
        service._process_context = AsyncMock(side_effect=RuntimeError("generation failed"))

    await service.process_frame(
        LLMContextFrame(LLMContext([{"role": "user", "content": "Hello."}])),
        FrameDirection.DOWNSTREAM,
    )
    await service._generation_task

    frames = [call.args[0] for call in service.push_frame.await_args_list]
    assert isinstance(frames[0], LLMFullResponseStartFrame)
    assert isinstance(frames[-1], LLMFullResponseEndFrame)
    service.start_processing_metrics.assert_awaited_once()
    service.stop_processing_metrics.assert_awaited_once()
    assert service.push_error.await_count == int(fail)


@pytest.mark.asyncio
async def test_system_instruction_is_included_without_mutating_context(make_service):
    """Service-owned instructions participate in both prefill and final history."""
    service, sockets, _ = make_service(
        settings=DynamoRealtimeLLMService.Settings(
            model="dynamo-model", system_instruction="Be concise."
        )
    )
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello.")
    assert context.get_messages() == []
    context.add_message({"role": "user", "content": "Hello."})

    await service._process_context(context)

    assert len(sockets) == 1
    assert [item["content"][0]["text"] for item in _items(sockets[0])] == ["Be concise."]
    assert context.get_messages() == [{"role": "user", "content": "Hello."}]
    assert any(event["type"] == "input_text.commit" for event in sockets[0].sent)


@pytest.mark.asyncio
async def test_disabled_prefill_sends_only_authoritative_context(make_service):
    """Baseline mode does not create a speculative connection or send ASR text."""
    service, sockets, _ = make_service(incremental_prefill=False)
    context = LLMContext()
    service.prefill(context, "Incorrect partial")
    assert not sockets
    assert service._prefill_task is None
    context.add_message({"role": "user", "content": "Correct final."})

    await service._process_context(context)

    assert not any(event["type"].startswith("input_text.") for event in sockets[0].sent)
    assert _items(sockets[0])[-1]["content"][0]["text"] == "Correct final."


@pytest.mark.parametrize("setting", ["temperature", "top_p", "top_k", "seed"])
def test_unsupported_sampling_settings_fail_explicitly(make_service, setting):
    """Unsupported options must not silently disappear from requests."""
    settings = DynamoRealtimeLLMService.Settings(model="dynamo-model", **{setting: 1})
    with pytest.raises(ValueError, match=setting):
        make_service(settings=settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_default", [False, True])
async def test_framework_thinking_preference_preserves_model_updates(
    make_service, provider_default
):
    """Framework thinking preferences allow model updates without changing wire options."""
    settings = DynamoRealtimeLLMService.Settings.from_mapping(
        {"model": "dynamo-model", "use_provider_thinking_default": provider_default}
    )
    service, sockets, sessions = make_service(settings=settings)

    await service._update_settings(DynamoRealtimeLLMService.Settings(model="updated-model"))
    await service._process_context(LLMContext([{"role": "user", "content": "Hello."}]))

    assert service._settings.given_fields()["use_provider_thinking_default"] is provider_default
    update = next(event for event in sockets[0].sent if event["type"] == "session.update")
    assert update["session"] == {
        "type": "realtime",
        "model": "updated-model",
        "output_modalities": ["text"],
    }
    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
async def test_tools_are_rejected_before_opening_connection(make_service):
    """Text-only transport cannot accept a tool-enabled context."""
    service, sockets, _ = make_service()
    context = LLMContext(
        [{"role": "user", "content": "What is the weather?"}],
        tools=[
            FunctionSchema(
                name="weather", description="Get the weather.", properties={}, required=[]
            )
        ],
    )

    with pytest.raises(ValueError, match="tools"):
        await service._process_context(context)

    assert not sockets


@pytest.mark.asyncio
async def test_multimodal_messages_are_rejected(make_service):
    """Image input is rejected rather than silently dropped from the prompt."""
    service, sockets, _ = make_service()
    context = LLMContext(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="text-only"):
        await service._process_context(context)

    assert not sockets


@pytest.mark.asyncio
async def test_failed_speculation_does_not_prevent_final_generation(make_service):
    """Optional prefill failure falls back to a fresh authoritative request."""
    service, sockets, sessions = make_service()
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello")
    sockets[0].send_json = AsyncMock(side_effect=ConnectionError("socket lost"))
    service.prefill(context, " there")
    await _wait_for(lambda: service.push_error.await_count == 1)
    assert sockets[0].closed and sessions[0].closed
    context.add_message({"role": "user", "content": "Hello there."})

    await service._process_context(context)

    assert len(sockets) == 2
    assert _items(sockets[1])[-1]["content"][0]["text"] == "Hello there."
    assert not any(event["type"].startswith("input_text.") for event in sockets[1].sent)
    assert sockets[1].closed and sessions[1].closed


@pytest.mark.asyncio
async def test_cancelled_generation_closes_connection(make_service):
    """Cancellation while waiting for output releases resources and propagates."""
    service, sockets, sessions = make_service()
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello.")
    sockets[0].response_events = []
    context.add_message({"role": "user", "content": "Hello."})
    generation = asyncio.create_task(service._process_context(context))
    try:
        await _wait_for(
            lambda: any(event["type"] == "response.create" for event in sockets[0].sent)
        )
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation
        assert sockets[0].closed and sessions[0].closed
        assert not service._generating
    finally:
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://host/proxy/v1", "wss://host/proxy/v1/realtime"),
        ("http://host:8000", "ws://host:8000/v1/realtime"),
        ("wss://host/proxy/v1/realtime?tenant=test", "wss://host/proxy/v1/realtime?tenant=test"),
    ],
)
def test_realtime_url_keeps_proxy_prefix_and_query(base_url, expected):
    """HTTP and explicit socket endpoints preserve reverse-proxy routing."""
    service = DynamoRealtimeLLMService(
        base_url=base_url, settings=DynamoRealtimeLLMService.Settings(model="test")
    )
    assert service._url == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_managed_pipeline_prefill_finalization_and_interruption(fake_transport, interrupt):
    """Real pipeline startup, managed tasks, and frame dispatch preserve the turn."""
    sockets, sessions = fake_transport
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test", system_instruction="Be concise."),
    )
    history = LLMContext()
    processor = DynamoStreamingPrefillProcessor(llm=service, context=history)
    frames = []
    if interrupt:
        frames.extend(
            [
                InterimTranscriptionFrame("Abandoned request", "user", ""),
                SleepFrame(0.05),
                InterruptionFrame(),
                SleepFrame(0.05),
            ]
        )
    frames.extend(
        [
            InterimTranscriptionFrame("Hello.", "user", ""),
            SleepFrame(0.05),
            LLMContextFrame(LLMContext([{"role": "user", "content": "Hello."}])),
        ]
    )

    async with asyncio.timeout(3):
        downstream, upstream = await run_test(Pipeline([processor, service]), frames_to_send=frames)

    assert len(sockets) == (2 if interrupt else 1)
    assert all(socket.closed for socket in sockets)
    assert all(session.closed for session in sessions)
    assert service._prefill_task is None
    assert service._websocket is None
    assert [event["type"] for event in sockets[-1].sent][-2:] == [
        "input_text.commit",
        "response.create",
    ]
    assert [
        event["text"] for event in sockets[-1].sent if event["type"] == "input_text.append"
    ] == ["Hello."]
    assert [
        type(frame)
        for frame in downstream
        if isinstance(frame, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))
    ] == [LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame, LLMFullResponseEndFrame]
    assert [frame.text for frame in downstream if isinstance(frame, LLMTextFrame)] == ["Hello", "."]
    assert not any(isinstance(frame, ErrorFrame) for frame in [*upstream, *downstream])


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_type", [EndFrame, CancelFrame])
async def test_managed_pipeline_shutdown_closes_idle_prefill(fake_transport, frame_type):
    """Ending a real pipeline releases prefill even without a final user turn."""
    sockets, sessions = fake_transport
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test"),
    )
    processor = DynamoStreamingPrefillProcessor(llm=service, context=LLMContext())

    async with asyncio.timeout(3):
        await run_test(
            Pipeline([processor, service]),
            frames_to_send=[
                InterimTranscriptionFrame("Unfinished request", "user", ""),
                SleepFrame(0.05),
                frame_type(),
            ],
            send_end_frame=False,
        )

    assert len(sockets) == 1
    assert sockets[0].closed and sessions[0].closed
    assert service._prefill_task is None
    assert service._websocket is None
    assert not any(event["type"] == "response.create" for event in sockets[0].sent)


@pytest.mark.asyncio
async def test_generation_forwards_frames_before_response_finishes(
    fake_transport, blocked_response
):
    """Ordinary frames and heartbeats flow while the backend is still generating."""
    sockets, sessions = fake_transport
    waiting, release = blocked_response
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test"),
    )
    heartbeat = HeartbeatFrame(timestamp=1)
    marker = TextFrame("ordinary frame")
    forwarded = []

    @service.event_handler("on_after_push_frame")
    async def on_push(service, frame):
        forwarded.append(frame)

    async def check_frame_flow():
        await waiting.wait()
        try:
            await service.pipeline_worker.queue_frames([heartbeat, marker])
            await _wait_for(lambda: heartbeat in forwarded and marker in forwarded)
            assert not any(isinstance(frame, LLMFullResponseEndFrame) for frame in forwarded)
            assert not sockets[0].closed
        finally:
            release.set()
            await service.pipeline_worker.queue_frame(EndFrame())

    async with asyncio.timeout(3):
        pipeline_result, check_result = await asyncio.gather(
            run_test(
                service,
                frames_to_send=[
                    LLMContextFrame(LLMContext([{"role": "user", "content": "Hello."}]))
                ],
                send_end_frame=False,
            ),
            check_frame_flow(),
            return_exceptions=True,
        )

    if isinstance(check_result, BaseException):
        raise check_result
    if isinstance(pipeline_result, BaseException):
        raise pipeline_result
    downstream, upstream = pipeline_result
    assert heartbeat in downstream and marker in downstream
    assert [frame.text for frame in downstream if isinstance(frame, LLMTextFrame)] == ["Hello", "."]
    assert isinstance(downstream[-1], EndFrame)
    assert sockets[0].closed and sessions[0].closed
    assert not any(isinstance(frame, ErrorFrame) for frame in upstream)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["interruption", "eager_cancel", "settings", "cancel", "cleanup"]
)
async def test_managed_generation_cancellation_discards_pending_responses(
    managed_service, fake_transport, blocked_response, action
):
    """Reset and teardown cancel active and queued work without leaking stale output."""
    service = managed_service
    sockets, sessions = fake_transport
    waiting, release = blocked_response
    pushed = []
    timeouts = []

    @service.event_handler("on_after_push_frame")
    async def on_push(service, frame):
        pushed.append(frame)

    @service.event_handler("on_completion_timeout")
    async def on_timeout(service):
        timeouts.append(service)

    await service.process_frame(
        LLMContextFrame(
            LLMContext([{"role": "user", "content": "Active request"}]),
            speculation=action == "eager_cancel",
        ),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.wait_for(waiting.wait(), 1)
    generation = service._generation_task
    assert generation in service.task_manager.current_tasks()
    await service.process_frame(
        LLMContextFrame(LLMContext([{"role": "user", "content": "Queued request"}])),
        FrameDirection.DOWNSTREAM,
    )

    async with asyncio.timeout(1):
        if action == "cleanup":
            await service.cleanup()
        else:
            frame = {
                "interruption": InterruptionFrame(),
                "eager_cancel": EagerEndOfTurnCancelFrame(),
                "settings": LLMUpdateSettingsFrame(
                    delta=DynamoRealtimeLLMService.Settings(model="updated-model")
                ),
                "cancel": CancelFrame(),
            }[action]
            await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert generation.done()
    assert service._generation_task is None
    assert not service._generation_queue
    assert sockets[0].closed and sessions[0].closed
    assert not any(isinstance(frame, (LLMTextFrame, ErrorFrame)) for frame in pushed)
    assert not timeouts

    release.set()
    await service.process_frame(
        LLMContextFrame(LLMContext([{"role": "user", "content": "Fresh request"}])),
        FrameDirection.DOWNSTREAM,
    )
    if action in ("cancel", "cleanup"):
        assert service._generation_task is None
        assert len(sockets) == 1
    else:
        await asyncio.wait_for(service._generation_task, 1)
        assert len(sockets) == 2
        assert sockets[1].closed and sessions[1].closed
        assert [frame.text for frame in pushed if isinstance(frame, LLMTextFrame)] == ["Hello", "."]
        assert _items(sockets[1])[-1]["content"][0]["text"] == "Fresh request"
        if action == "settings":
            update = next(event for event in sockets[1].sent if event["type"] == "session.update")
            assert update["session"]["model"] == "updated-model"


@pytest.mark.asyncio
async def test_graceful_shutdown_drains_context_snapshots_in_order(
    managed_service, fake_transport, blocked_response
):
    """Queued requests keep their text and finish their response frames before EndFrame."""
    service = managed_service
    sockets, sessions = fake_transport
    waiting, release = blocked_response
    pushed = []

    @service.event_handler("on_after_push_frame")
    async def on_push(service, frame):
        pushed.append(frame)

    context = LLMContext([{"role": "user", "content": "First request"}])
    await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    context.set_messages([{"role": "user", "content": "Second request"}])
    await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    context.set_messages([{"role": "user", "content": "Later context mutation"}])
    service.prefill(LLMContext(), "Late partial")
    assert service._prefill_task is None
    await asyncio.wait_for(waiting.wait(), 1)

    ending = service.create_task(service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM))
    try:
        await _wait_for(lambda: service._closing)
        assert not ending.done()
        assert len(sockets) == 1
        release.set()
        await asyncio.wait_for(ending, 1)
    finally:
        release.set()
        await service.cancel_task(ending)

    assert [_items(socket)[-1]["content"][0]["text"] for socket in sockets] == [
        "First request",
        "Second request",
    ]
    assert [
        type(frame)
        for frame in pushed
        if isinstance(frame, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))
    ] == [LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame, LLMFullResponseEndFrame] * 2
    assert isinstance(pushed[-1], EndFrame)
    assert all(socket.closed for socket in sockets)
    assert all(session.closed for session in sessions)
    assert service._generation_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_stage", ["handshake", "response"])
async def test_completion_timeout_notifies_handler_and_reports_error(
    fake_transport, monkeypatch, timeout_stage
):
    """Completion timeouts notify applications and preserve error and response frames."""
    sockets, sessions = fake_transport
    receive_json = _Socket.receive_json

    async def receive(self):
        event = await receive_json(self)
        if event["type"] == (
            "session.created" if timeout_stage == "handshake" else "response.output_text.delta"
        ):
            await asyncio.Event().wait()
        return event

    monkeypatch.setattr(_Socket, "receive_json", receive)
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test"),
        timeout=0.01,
    )
    notifications = []

    @service.event_handler("on_completion_timeout")
    async def on_timeout(sender):
        notifications.append(sender)

    async with asyncio.timeout(3):
        downstream, upstream = await run_test(
            service,
            frames_to_send=[LLMContextFrame(LLMContext([{"role": "user", "content": "Hello."}]))],
        )

    assert notifications == [service]
    errors = [frame for frame in upstream if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert isinstance(errors[0].exception, TimeoutError)
    assert [
        type(frame)
        for frame in downstream
        if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
    ] == [LLMFullResponseStartFrame, LLMFullResponseEndFrame]
    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
async def test_response_timeout_releases_connection(make_service):
    """A connected backend that never produces output cannot hold a turn forever."""
    service, sockets, sessions = make_service(timeout=0.01)
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello.")
    sockets[0].response_events = []
    context.add_message({"role": "user", "content": "Hello."})

    with pytest.raises(TimeoutError):
        await service._process_context(context)

    assert sockets[0].closed and sessions[0].closed
    assert not service._generating


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [None, {"delta": "missing type"}, {"type": 42}])
async def test_malformed_server_event_fails_and_releases_connection(make_service, event):
    """Invalid protocol envelopes are not treated as successful completions."""
    service, sockets, sessions = make_service()
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello.")
    sockets[0].response_events = [event]
    context.add_message({"role": "user", "content": "Hello."})

    with pytest.raises(ValueError, match="Invalid Dynamo realtime event"):
        await service._process_context(context)

    service._push_llm_text.assert_not_awaited()
    assert sockets[0].closed and sessions[0].closed


@pytest.mark.asyncio
async def test_cleanup_without_lifecycle_frame_releases_managed_prefill(fake_transport):
    """Direct teardown closes an idle socket and rejects later ASR deltas."""
    sockets, sessions = fake_transport
    service = DynamoRealtimeLLMService(
        base_url="http://localhost:8000/v1",
        settings=DynamoRealtimeLLMService.Settings(model="test"),
    )
    manager = TaskManager()
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    context = LLMContext()
    try:
        await _prefill(service, sockets, context, "Unfinished request")
        task = service._prefill_task
        assert task in manager.current_tasks()

        await service.cleanup()

        assert task.done()
        assert not manager.current_tasks()
        assert sockets[0].closed and sessions[0].closed
        assert service._prefill_task is None
        assert service._websocket is None
        service.prefill(context, "Late ASR delta")
        assert service._prefill_task is None
        assert not manager.current_tasks()
        assert len(sockets) == 1
    finally:
        await service.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_action", ["interruption", "settings"])
async def test_reset_rejects_prefill_while_old_socket_closes(make_service, reset_action):
    """A new partial cannot create work that an in-progress reset will corrupt."""
    service, sockets, sessions = make_service()
    context = LLMContext()
    await _prefill(service, sockets, context, "Old request")
    sending = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    close = sockets[0].close

    async def stalled_send(event):
        sending.set()
        await asyncio.Event().wait()

    async def stalled_close():
        closing.set()
        await release_close.wait()
        await close()

    sockets[0].send_json = stalled_send
    sockets[0].close = stalled_close
    service.prefill(context, " in progress")
    await asyncio.wait_for(sending.wait(), 1)
    if reset_action == "interruption":
        reset = asyncio.create_task(service._handle_interruptions(InterruptionFrame()))
    else:
        reset = asyncio.create_task(
            service._update_settings(DynamoRealtimeLLMService.Settings(model="updated-model"))
        )
    try:
        await asyncio.wait_for(closing.wait(), 1)
        service.prefill(context, "New partial during reset")
        assert service._prefill_task is None
        release_close.set()
        await asyncio.wait_for(reset, 1)
        assert len(sockets) == 1
        assert sockets[0].closed and sessions[0].closed
        assert service._prefill_task is None

        await _prefill(service, sockets, context, "Next request")
        assert len(sockets) == 2
        assert [
            event["text"] for event in sockets[1].sent if event["type"] == "input_text.append"
        ] == ["Next request"]
    finally:
        release_close.set()
        await asyncio.gather(reset, return_exceptions=True)
        await service._stop_prefill()
        await service._close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_action", ["interruption", "settings"])
async def test_final_generation_waits_for_connection_reset(make_service, reset_action):
    """Final input cannot reopen a connection until the previous reset finishes."""
    service, sockets, sessions = make_service()
    await _prefill(service, sockets, LLMContext(), "Old partial")
    closing = asyncio.Event()
    release_close = asyncio.Event()
    generation_started = asyncio.Event()
    close = sockets[0].close

    async def stalled_close():
        closing.set()
        await release_close.wait()
        await close()

    async def generate():
        generation_started.set()
        await service._process_context(LLMContext([{"role": "user", "content": "New final."}]))

    sockets[0].close = stalled_close
    if reset_action == "interruption":
        reset = asyncio.create_task(service._handle_interruptions(InterruptionFrame()))
    else:
        reset = asyncio.create_task(
            service._update_settings(DynamoRealtimeLLMService.Settings(model="updated-model"))
        )
    generation = None
    try:
        await asyncio.wait_for(closing.wait(), 1)
        generation = asyncio.create_task(generate())
        await asyncio.wait_for(generation_started.wait(), 1)
        assert not generation.done()
        assert len(sockets) == 1
        release_close.set()
        await asyncio.wait_for(asyncio.gather(reset, generation), 1)

        assert len(sockets) == 2
        assert all(socket.closed for socket in sockets)
        assert all(session.closed for session in sessions)
        assert _items(sockets[1])[-1]["content"][0]["text"] == "New final."
        assert not any(event["type"].startswith("input_text.") for event in sockets[1].sent)
        if reset_action == "settings":
            update = next(event for event in sockets[1].sent if event["type"] == "session.update")
            assert update["session"]["model"] == "updated-model"
    finally:
        release_close.set()
        if generation is not None and not generation.done():
            generation.cancel()
        await asyncio.gather(
            reset, *([generation] if generation is not None else []), return_exceptions=True
        )
        await service._stop_prefill()
        await service._close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mute_direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
async def test_prefill_processor_honors_mute_frames_and_forwards_transcriptions(
    make_service, monkeypatch, mute_direction
):
    """Muted ASR still flows through the pipeline without speculative LLM work."""
    monkeypatch.setattr(FrameProcessor, "process_frame", AsyncMock())
    service, _, _ = make_service()
    service.prefill = Mock()
    context = LLMContext()
    processor = DynamoStreamingPrefillProcessor(llm=service, context=context)
    processor.push_frame = AsyncMock()
    frames = [
        (UserMuteStartedFrame(), mute_direction),
        (InterimTranscriptionFrame("Muted partial", "user", ""), FrameDirection.DOWNSTREAM),
        (TranscriptionFrame("Muted final", "user", ""), FrameDirection.DOWNSTREAM),
        (UserMuteStoppedFrame(), mute_direction),
        (InterimTranscriptionFrame("Live partial", "user", ""), FrameDirection.DOWNSTREAM),
    ]

    for frame, direction in frames:
        await processor.process_frame(frame, direction)

    service.prefill.assert_called_once_with(context, "Live partial")
    assert [call.args for call in processor.push_frame.await_args_list] == frames


@pytest.mark.asyncio
async def test_stalled_close_is_bounded_without_failing_successful_response(make_service):
    """A peer that never closes cannot leak HTTP resources or fail completed output."""
    service, sockets, sessions = make_service(ws_close_timeout=0.01)
    context = LLMContext()
    await _prefill(service, sockets, context, "Hello.")
    close_cancelled = asyncio.Event()

    async def stalled_close():
        try:
            await asyncio.Event().wait()
        finally:
            close_cancelled.set()

    sockets[0].close = stalled_close
    context.add_message({"role": "user", "content": "Hello."})

    await asyncio.wait_for(service._process_context(context), 1)

    assert close_cancelled.is_set()
    assert sessions[0].closed
    assert service._websocket is None and service._http is None
    assert [call.args[0] for call in service._push_llm_text.await_args_list] == ["Hello", "."]
    service.push_error.assert_not_awaited()
