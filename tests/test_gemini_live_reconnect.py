#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for tool-result delivery in GeminiLiveLLMService across a reconnect.

A tool call the model is blocked on stays blocked until its result arrives, so a
result that doesn't reach the service has to survive the outage and go out once a
session can take it. Which shape that takes depends on how the session comes
back: a resumed session still holds the pending call, while a re-seeded one
receives the result as conversation history instead.
"""

import asyncio
from typing import Any

import pytest
from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, InputImageRawFrame
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _FakeSession:
    """Records what the service sends."""

    def __init__(self):
        self.tool_responses: list[Any] = []
        self.client_content: list[dict[str, Any]] = []

    async def send_tool_response(self, function_responses=None):
        self.tool_responses.append(function_responses)

    async def send_client_content(self, **kwargs):
        self.client_content.append(kwargs)

    async def send_realtime_input(self, **kwargs):
        pass

    async def close(self):
        pass


def _make_service() -> GeminiLiveLLMService:
    """Construct a service mid-conversation, with one completed tool call.

    ``__init__`` does no I/O. Metrics need a started pipeline, so they're stubbed
    out to let the handlers and the disconnect path run in isolation.
    """
    service = GeminiLiveLLMService(api_key="test-key")
    service._session = _FakeSession()
    service._tool_call_id_to_name["call-1"] = "book_flight"
    service._tool_calls_this_session.add("call-1")
    service._context = LLMContext(
        messages=[
            {"role": "user", "content": "book me a flight"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "book_flight", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "booked, ref XY123"},
        ]
    )

    async def _noop(*args, **kwargs):
        pass

    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]
    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service


def _seeded_text(session: "_FakeSession") -> str:
    """The text of the last conversation turn the service seeded."""
    return session.client_content[-1]["turns"][-1].parts[0].text


def _audio_frame() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=16000, num_channels=1)


def _video_frame() -> InputImageRawFrame:
    return InputImageRawFrame(image=b"\x00" * (8 * 8 * 3), size=(8, 8), format="RGB")


@pytest.fixture
def log_messages():
    """Collect the messages logged during a test."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    yield messages
    logger.remove(sink_id)


@pytest.mark.asyncio
async def test_result_dropped_across_a_disconnect_is_retried_under_its_own_name():
    """A result that missed its session is sent again, naming the function called."""
    service = _make_service()
    session = service._session

    await service._disconnect()
    await service._process_completed_function_calls(send_new_results=True)

    assert service._completed_tool_calls == set(), (
        "an unanswered call must not be recorded as answered"
    )

    service._session = session
    await service._process_completed_function_calls(send_new_results=True)

    assert [r.id for r in session.tool_responses] == ["call-1"]
    assert session.tool_responses[0].name == "book_flight", (
        "the model is blocked on book_flight, so the response has to name it"
    )


@pytest.mark.asyncio
async def test_delivered_result_is_not_resent_after_a_disconnect():
    """A result the service already sent isn't repeated when the session returns."""
    service = _make_service()
    session = service._session
    service._session_resumption_handle = "handle-abc"

    await service._process_completed_function_calls(send_new_results=True)
    assert len(session.tool_responses) == 1

    await service._disconnect()
    await service._handle_session_ready(session)

    assert len(session.tool_responses) == 1


@pytest.mark.asyncio
async def test_resumed_session_receives_the_pending_result_as_conversation():
    """A resumed session never issued the call, so the result arrives as text."""
    service = _make_service()
    session = service._session
    service._session_resumption_handle = "handle-abc"

    await service._disconnect()
    await service._process_completed_function_calls(send_new_results=True)
    assert session.tool_responses == []

    await service._handle_session_ready(session)

    assert session.tool_responses == [], "the restored session would ignore a tool response"
    assert service._completed_tool_calls == {"call-1"}
    assert _seeded_text(session) == "[Function book_flight returned booked, ref XY123]"


@pytest.mark.asyncio
async def test_reseeded_session_takes_the_result_as_history():
    """A re-seeded session never issued the call, so it gets no tool response."""
    service = _make_service()
    session = service._session

    await service._disconnect()
    await service._process_completed_function_calls(send_new_results=True)

    await service._handle_session_ready(session)

    assert session.client_content, "the reconnect re-seeds the conversation"
    assert session.tool_responses == []
    assert service._completed_tool_calls == {"call-1"}, (
        "the re-seed carries the result, so the call is answered"
    )


@pytest.mark.asyncio
async def test_send_failure_on_a_live_session_is_not_retried():
    """A send that fails with the session up is reported once, not on every update."""
    service = _make_service()
    errors: list[str | None] = []

    async def _raise(**kwargs):
        raise RuntimeError("send failed")

    async def _record_error(error_msg=None, **kwargs):
        errors.append(error_msg)

    service._session.send_tool_response = _raise  # type: ignore[method-assign]
    service.push_error = _record_error  # type: ignore[method-assign]

    for _ in range(3):
        await service._process_completed_function_calls(send_new_results=True)

    assert len(errors) == 1


@pytest.mark.asyncio
async def test_results_already_in_the_context_are_recorded_without_sending(log_messages):
    """The bootstrap pass adopts existing results rather than replaying them."""
    service = _make_service()
    session = service._session
    service._session = None

    await service._process_completed_function_calls(send_new_results=False)

    assert service._completed_tool_calls == {"call-1"}
    assert session.tool_responses == []
    assert not [m for m in log_messages if "dropping" in m], (
        "adopting a result is not a delivery attempt, so nothing is dropped"
    )


@pytest.mark.asyncio
async def test_dropped_tool_result_names_the_call_and_the_reason(log_messages):
    """A dropped result says which call it leaves unanswered, and why."""
    service = _make_service()
    service._session = None

    await service._process_completed_function_calls(send_new_results=True)

    assert any(
        "dropping tool result for tool_call_id=call-1 — no session" in m for m in log_messages
    )


@pytest.mark.asyncio
async def test_dropped_user_text_names_the_reason(log_messages):
    """Text that can't be sent says which connection state stopped it."""
    service = _make_service()
    service._disconnecting = True

    await service._send_user_text("are you there?")

    assert any("dropping user text — disconnecting" in m for m in log_messages)


@pytest.mark.asyncio
async def test_dropped_media_is_reported_once_per_outage(log_messages):
    """Media discarded while a session is away is reported once, not per frame."""
    service = _make_service()
    session = service._session
    service._session = None

    for _ in range(5):
        await service._send_user_audio(_audio_frame())

    assert len([m for m in log_messages if "dropping user audio" in m]) == 1

    await service._handle_session_ready(session)
    service._session = None
    await service._send_user_video(_video_frame())

    assert any("dropping user video" in m for m in log_messages)


@pytest.mark.asyncio
async def test_paused_media_is_not_reported_as_dropped(log_messages):
    """Pausing input is a deliberate choice, so nothing is reported as dropped."""
    service = _make_service()
    service._session = None
    service.set_audio_input_paused(True)
    service.set_video_input_paused(True)

    await service._send_user_audio(_audio_frame())
    await service._send_user_video(_video_frame())

    assert not [m for m in log_messages if "dropping" in m]


@pytest.mark.asyncio
async def test_concurrent_scans_send_the_result_once():
    """The reconnect flush and a context update can't both claim the same call."""
    service = _make_service()
    session = service._session
    service._session_resumption_handle = "handle-abc"
    delivered = session.send_tool_response

    async def _yield_mid_send(function_responses=None):
        await asyncio.sleep(0)
        await delivered(function_responses=function_responses)

    session.send_tool_response = _yield_mid_send  # type: ignore[method-assign]

    await service._disconnect()
    await asyncio.gather(
        service._handle_session_ready(session),
        service._process_completed_function_calls(send_new_results=True),
    )

    assert len(session.tool_responses) + len(session.client_content) == 1


@pytest.mark.asyncio
async def test_result_stays_pending_when_the_reseed_fails():
    """A re-seed that never landed can't be what delivered the result."""
    service = _make_service()
    session = service._session

    async def _fail(**kwargs):
        raise RuntimeError("re-seed failed")

    async def _swallow(**kwargs):
        pass

    session.send_client_content = _fail  # type: ignore[method-assign]
    service.push_error = _swallow  # type: ignore[method-assign]

    await service._disconnect()
    await service._process_completed_function_calls(send_new_results=True)
    await service._handle_session_ready(session)

    assert session.tool_responses == []
    assert service._completed_tool_calls == set()


@pytest.mark.asyncio
async def test_async_tool_result_without_a_session_is_retried():
    """An async tool's final result is held and resent like a synchronous one."""
    service = _make_service()
    session = service._session
    service._tool_call_id_to_name["call-2"] = "slow_job"
    service._tool_calls_this_session.add("call-2")
    service._context.add_message(async_tool_messages.build_started_message("call-2"))
    service._context.add_message(async_tool_messages.build_final_result_message("call-2", "done"))
    service._completed_tool_calls.add("call-1")

    service._session = None
    await service._process_completed_function_calls(send_new_results=True)

    assert "call-2" not in service._completed_tool_calls

    service._session = session
    await service._process_completed_function_calls(send_new_results=True)

    assert [(r.id, r.name) for r in session.tool_responses] == [("call-2", "slow_job")]


@pytest.mark.asyncio
async def test_a_session_lost_mid_scan_keeps_only_the_rest_pending():
    """Results already sent stay settled; the ones that missed the session don't."""
    service = _make_service()
    session = service._session
    service._tool_call_id_to_name["call-2"] = "check_seat"
    service._tool_calls_this_session.add("call-2")
    service._context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "check_seat", "arguments": "{}"},
                }
            ],
        }
    )
    service._context.add_message({"role": "tool", "tool_call_id": "call-2", "content": "12A"})
    delivered = session.send_tool_response

    async def _then_lose_the_session(function_responses=None):
        await delivered(function_responses=function_responses)
        service._session = None

    session.send_tool_response = _then_lose_the_session  # type: ignore[method-assign]

    await service._process_completed_function_calls(send_new_results=True)

    assert [r.id for r in session.tool_responses] == ["call-1"]
    assert service._completed_tool_calls == {"call-1"}


@pytest.mark.asyncio
async def test_call_issued_by_this_session_uses_the_tool_response_channel():
    """The ordinary path is unchanged: a live call gets a formal tool response."""
    service = _make_service()
    session = service._session

    await service._process_completed_function_calls(send_new_results=True)

    assert [r.id for r in session.tool_responses] == ["call-1"]
    assert session.client_content == []


@pytest.mark.asyncio
async def test_call_that_outlived_its_session_is_delivered_as_conversation():
    """A call the session never issued can't be answered with a tool response."""
    service = _make_service()
    session = service._session
    service._tool_calls_this_session.clear()  # as a new session leaves it

    await service._process_completed_function_calls(send_new_results=True)

    assert session.tool_responses == []
    assert service._completed_tool_calls == {"call-1"}
    assert _seeded_text(session) == "[Function book_flight returned booked, ref XY123]", (
        "the result reads as the adapter renders it, not as Gemini's response wrapper"
    )
