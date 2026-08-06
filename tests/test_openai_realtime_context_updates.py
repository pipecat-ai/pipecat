#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mid-session context updates in OpenAIRealtimeLLMService.

A realtime session runs continuously: conversation history is held server-side
and seeded only once, so anything that changes mid-conversation has to reach
the model another way. These tests cover that machinery:

- The adapter folds every "system"/"developer" context message into the
  session instructions, combined with the service-level ``system_instruction``
  (so e.g. Pipecat Flows node task messages actually steer the model).
- An updated context frame refreshes the session instructions when — and only
  when — they changed.
- The ``run_llm`` intent on ``LLMContextFrame`` drives response creation: an
  explicit ``True`` (an ``LLMRunFrame``, e.g. a Flows node transition with
  ``respond_immediately=True``) creates a response, ``False`` suppresses one,
  and ``None`` preserves the legacy behavior of responding only to new tool
  results.
- The context aggregators mark explicit run requests with ``run_llm=True``.
"""

from typing import Any

import pytest

from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.frames.frames import LLMContextFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.tests.utils import run_test

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records the client events sent via ``send_client_event``."""

    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    def clear(self):
        self.events.clear()


def _make_service(system_instruction: str | None = None):
    """Construct a service wired to a fake send_client_event. ``__init__`` does no I/O."""
    settings = (
        OpenAIRealtimeLLMService.Settings(system_instruction=system_instruction)
        if system_instruction
        else None
    )
    service = OpenAIRealtimeLLMService(api_key="test-key", settings=settings)

    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]
    # Pretend the session finished its handshake so _create_response sends
    # immediately instead of deferring to session.updated.
    service._api_session_ready = True

    async def _noop(*args, **kwargs):
        pass

    # _create_response pushes frames and starts metrics, which need a started
    # pipeline; stub them out so the handlers can run in isolation.
    service.push_frame = _noop  # type: ignore[method-assign]
    service.start_processing_metrics = _noop  # type: ignore[method-assign]
    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]

    return service, recorder


# ---------------------------------------------------------------------------
# Adapter: system/developer messages fold into the session instructions
# ---------------------------------------------------------------------------


def test_adapter_folds_all_system_and_developer_messages_into_instructions():
    adapter = OpenAIRealtimeLLMAdapter()
    context = LLMContext(
        messages=[
            {"role": "system", "content": "Persona."},
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "Ask for a name."},
        ]
    )
    params = adapter.get_llm_invocation_params(context, system_instruction="Base.")

    assert params["system_instruction"] == "Base.\n\nPersona.\n\nAsk for a name."
    # The folded messages must not also be sent as conversation items.
    assert len(params["messages"]) == 1
    assert params["messages"][0].role == "user"


def test_adapter_without_any_system_content_yields_no_instructions():
    adapter = OpenAIRealtimeLLMAdapter()
    context = LLMContext(messages=[{"role": "user", "content": "hi"}])
    params = adapter.get_llm_invocation_params(context)
    assert params["system_instruction"] is None


def test_adapter_folds_list_content_system_messages():
    adapter = OpenAIRealtimeLLMAdapter()
    context = LLMContext(
        messages=[{"role": "system", "content": [{"type": "text", "text": "Persona."}]}]
    )
    params = adapter.get_llm_invocation_params(context)
    assert params["system_instruction"] == "Persona."
    assert params["messages"] == []


# ---------------------------------------------------------------------------
# Service: instructions refresh + run_llm intent on context frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_transition_refreshes_instructions_and_responds():
    service, recorder = _make_service(system_instruction="Persona.")
    context = LLMContext(messages=[{"role": "developer", "content": "Greet the user."}])

    # Initial context (e.g. a Flows initial node with respond_immediately=True):
    # the session is configured and a response created.
    await service._handle_context(context, run_llm=True)
    assert recorder.kinds() == ["SessionUpdateEvent", "ResponseCreateEvent"]
    assert "Greet the user." in recorder.events[0].session.instructions
    recorder.clear()

    # Node transition: new developer guidance appended, run requested. The new
    # instructions must reach the session before the response is created.
    context.add_message({"role": "developer", "content": "Now ask for the order."})
    await service._handle_context(context, run_llm=True)
    assert recorder.kinds() == ["SessionUpdateEvent", "ResponseCreateEvent"]
    instructions = recorder.events[0].session.instructions
    assert "Persona." in instructions
    assert "Greet the user." in instructions
    assert instructions.endswith("Now ask for the order.")


@pytest.mark.asyncio
async def test_unchanged_instructions_send_no_duplicate_session_update():
    service, recorder = _make_service(system_instruction="Persona.")
    context = LLMContext(messages=[{"role": "developer", "content": "Greet the user."}])
    await service._handle_context(context, run_llm=True)
    recorder.clear()

    # A context update with no new system-level guidance and no run intent
    # (e.g. a user turn transcription landing in the context).
    context.add_message({"role": "user", "content": "hello"})
    await service._handle_context(context)
    assert recorder.kinds() == []


@pytest.mark.asyncio
async def test_tool_result_triggers_response_without_explicit_intent():
    service, recorder = _make_service()
    context = LLMContext(messages=[{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    recorder.clear()

    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    )
    context.add_message({"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1"})
    await service._handle_context(context)

    kinds = recorder.kinds()
    assert "ConversationItemCreateEvent" in kinds  # the function_call_output
    assert kinds[-1] == "ResponseCreateEvent"


@pytest.mark.asyncio
async def test_run_llm_false_sends_tool_result_without_response():
    service, recorder = _make_service()
    context = LLMContext(messages=[{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    recorder.clear()

    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    )
    context.add_message({"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1"})
    await service._handle_context(context, run_llm=False)

    kinds = recorder.kinds()
    assert "ConversationItemCreateEvent" in kinds
    assert "ResponseCreateEvent" not in kinds


@pytest.mark.asyncio
async def test_run_llm_true_with_tool_result_creates_single_response():
    """A Flows edge transition delivers the tool result and the run intent on
    one context frame; only one response must be created."""
    service, recorder = _make_service()
    context = LLMContext(messages=[{"role": "user", "content": "hi"}])
    await service._handle_context(context)
    recorder.clear()

    context.add_message(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    )
    context.add_message({"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1"})
    await service._handle_context(context, run_llm=True)

    assert recorder.kinds().count("ResponseCreateEvent") == 1


# ---------------------------------------------------------------------------
# Aggregators: explicit run requests are marked on the context frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_aggregator_marks_llm_run_with_run_intent():
    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(context)

    received_down, _ = await run_test(
        user_aggregator,
        frames_to_send=[LLMRunFrame()],
    )
    context_frame = next(f for f in received_down if isinstance(f, LLMContextFrame))
    assert context_frame.run_llm is True


@pytest.mark.asyncio
async def test_user_aggregator_turn_context_frames_carry_no_explicit_intent():
    from pipecat.frames.frames import (
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.utils.time import time_now_iso8601

    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(context)

    received_down, _ = await run_test(
        user_aggregator,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="hello", user_id="user", timestamp=time_now_iso8601()),
            UserStoppedSpeakingFrame(),
        ],
    )
    context_frame = next(f for f in received_down if isinstance(f, LLMContextFrame))
    assert context_frame.run_llm is None
