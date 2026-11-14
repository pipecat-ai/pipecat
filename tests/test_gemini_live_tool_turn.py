#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live turns that end on a tool call.

The Live API sends a ``toolCall`` in place of ``turn_complete`` and only
follows up with ``turn_complete`` after the tool result, if the model speaks
again. A turn that ends on the call (the result is withheld, or the model
stays silent) never gets that message. These tests drive
``_handle_server_message`` with a fabricated tool call and no
``turn_complete``.
"""

import asyncio
import contextlib

import pytest

from pipecat.frames.frames import EndFrame, LLMFullResponseEndFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from tests.test_gemini_live_interaction_status import (
    _FakeServerContent,
    _FakeServerMessage,
    _frame_types,
    _make_service,
    _setup_service,
    _start_bot_turn,
)


class _FunctionCall:
    def __init__(self, name: str = "end_call", call_id: str = "fc_end_call"):
        self.id = call_id
        self.name = name
        self.args = {}


def _tool_call_message(
    *,
    server_content: object = None,
    name: str = "end_call",
) -> _FakeServerMessage:
    message = _FakeServerMessage(server_content)
    message.tool_call = type("ToolCall", (), {"function_calls": [_FunctionCall(name=name)]})()
    return message


async def _prepare(service: GeminiLiveLLMService) -> None:
    """Give tool-call dispatch a context and skip actually running the tool."""
    service._context = object()  # type: ignore[assignment]

    async def _ignore_function_calls(function_calls):
        return None

    service.run_function_calls = _ignore_function_calls  # type: ignore[method-assign]


async def _cancel_timeout(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tool_call_without_turn_complete_finishes_the_turn_and_releases_end_frame():
    """A tool call with no turn_complete closes the bot turn.

    While the bot is still marked responding, an EndFrame is deferred and a
    30s watchdog is the only thing that would release it. The tool call has
    to finish the turn itself: push the end-of-turn frames and release the
    EndFrame without that wait.
    """
    service = _make_service(model="models/gemini-3.8-live")
    await _start_bot_turn(service)
    await _prepare(service)

    end_frame = EndFrame()
    await service.process_frame(end_frame, FrameDirection.DOWNSTREAM)
    assert service._end_frame_pending_bot_turn_finished is end_frame
    timeout_task = service._end_frame_deferral_timeout_task
    assert timeout_task is not None and not timeout_task.done()

    try:
        await service._handle_server_message(_tool_call_message())

        assert service._bot_is_responding is False
        assert service._end_frame_pending_bot_turn_finished is None
        assert timeout_task.cancelled() or service._end_frame_deferral_timeout_task is None
        assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    finally:
        await _cancel_timeout(timeout_task)
        await _cancel_timeout(service._deferred_turn_complete_timeout_task)


@pytest.mark.asyncio
async def test_tool_call_during_the_handler_still_releases_a_deferred_end_frame():
    """An EndFrame pushed from the tool handler is not left deferred.

    The handler runs before the server message returns, which is when an
    end-call tool pushes EndFrame. That frame has to be released once the
    tool call has been dispatched, still with no turn_complete following it.
    """
    service = _make_service(model="models/gemini-3.8-live")
    await _start_bot_turn(service)
    service._context = object()  # type: ignore[assignment]

    async def _end_the_call(function_calls):
        await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    service.run_function_calls = _end_the_call  # type: ignore[method-assign]

    await service._handle_server_message(_tool_call_message())

    assert service._bot_is_responding is False
    assert service._end_frame_pending_bot_turn_finished is None
    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    timeout_task = service._end_frame_deferral_timeout_task
    assert timeout_task is None or timeout_task.cancelled()
    await _cancel_timeout(timeout_task)


@pytest.mark.asyncio
async def test_tool_call_does_not_close_a_turn_held_for_background_reasoning():
    """A tool call is part of a turn the server is still working on.

    Thinking models hold turn_complete open while interaction_status stays
    IN_PROGRESS. The tool call in that window must not finish the turn; idle
    status still does.
    """
    service = _make_service()
    await _start_bot_turn(service)
    await _prepare(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    service.pushed_frames.clear()

    try:
        await service._handle_server_message(_tool_call_message())

        assert _frame_types(service) == []
        assert service._bot_is_responding is True
        assert service._turn_complete_pending_idle is not None
    finally:
        await _cancel_timeout(service._deferred_turn_complete_timeout_task)


@pytest.mark.asyncio
async def test_tool_call_bundled_with_turn_complete_closes_the_turn_once():
    """A message that already carries turn_complete is closed by that signal."""
    service = _make_service(model="models/gemini-3.8-live")
    await _start_bot_turn(service)
    await _prepare(service)

    await service._handle_server_message(
        _tool_call_message(server_content=_FakeServerContent(turn_complete=True))
    )

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_speech_after_a_tool_call_closes_on_turn_complete():
    """Speech the model produces after the tool result is its own turn.

    Closing the output on the tool call must not swallow the turn_complete
    that follows a later utterance.
    """
    service = _make_service(model="models/gemini-3.8-live")
    await _start_bot_turn(service)
    await _prepare(service)

    await service._handle_server_message(_tool_call_message())
    assert service._bot_is_responding is False
    service.pushed_frames.clear()

    await service._set_bot_is_responding(True)
    await service._handle_server_message(_FakeServerMessage(_FakeServerContent(turn_complete=True)))

    assert service._bot_is_responding is False
    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]


@pytest.mark.asyncio
async def test_tool_call_outside_a_bot_turn_pushes_no_end_frames():
    """A tool call with no bot audio or text in progress has nothing to close."""
    service = _make_service(model="models/gemini-3.8-live")
    await _setup_service(service)
    await _prepare(service)

    await service._handle_server_message(_tool_call_message())

    assert _frame_types(service) == []
    assert service._bot_is_responding is False
