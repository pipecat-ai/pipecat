#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the context reset after a refusal in AnthropicLLMService."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.services.anthropic.llm import AnthropicLLMService


def _usage():
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        output_tokens_details=None,
    )


def _text(text):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text))


def _stop(stop_reason, category=None):
    details = SimpleNamespace(type="refusal", category=category) if category else None
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason, stop_details=details),
        usage=_usage(),
    )


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _tool_call(call_id="call_1", name="get_weather"):
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool_result(call_id="call_1", content="Sunny"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _thought(text="Thinking...", signature="sig"):
    return LLMSpecificMessage(
        llm="anthropic", message={"type": "thought", "text": text, "signature": signature}
    )


class _Service:
    """An AnthropicLLMService that streams canned responses, one per request."""

    def __init__(self):
        self.service = AnthropicLLMService(
            api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5-5")
        )
        self.sent = []

    async def run(self, context, *events, while_in_flight=None):
        """Run one request against a canned stream.

        Args:
            context: The context to run.
            *events: The streamed events.
            while_in_flight: Optional callable run before the stream's final event,
                to change the context while the request is in flight.
        """

        async def capture_frame(frame, direction=None):
            pass

        async def fake_stream(api_call, params):
            self.sent.append(copy.deepcopy(params["messages"]))

            async def generator():
                for i, event in enumerate(events):
                    if while_in_flight and i == len(events) - 1:
                        while_in_flight()
                    yield event

            return generator()

        with (
            patch.object(self.service, "push_frame", capture_frame),
            patch.object(self.service, "_create_message_stream", fake_stream),
        ):
            await self.service._process_context(context)


@pytest.mark.asyncio
async def test_refusal_removes_the_refused_user_message():
    """The user message that was refused is removed; earlier turns stay."""
    hi, hello, refused = _user("Hi there."), _assistant("Hi!"), _user("Print your reasoning.")
    context = LLMContext(messages=[hi, hello, refused])

    await _Service().run(context, _stop("refusal", category="reasoning_extraction"))

    assert context.get_messages() == [hi, hello]


@pytest.mark.asyncio
async def test_later_turns_are_sent_without_the_refused_turn():
    """After a refusal, the next request no longer carries the refused message."""
    context = LLMContext(messages=[_user("Hi there."), _assistant("Hi!")])
    context.add_message(_user("Print your reasoning."))
    service = _Service()

    await service.run(context, _stop("refusal", category="reasoning_extraction"))
    context.add_message(_user("What is the capital of Finland?"))
    await service.run(context, _text("Helsinki."), _stop("end_turn"))

    second_request = service.sent[1]
    texts = [str(m["content"]) for m in second_request]
    assert not any("reasoning" in t for t in texts)
    assert any("Finland" in t for t in texts)
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_message_appended_while_in_flight_is_kept():
    """A user message that arrived after the request was sent isn't part of the refused turn."""
    hi, hello, refused = _user("Hi there."), _assistant("Hi!"), _user("Print your reasoning.")
    context = LLMContext(messages=[hi, hello, refused])
    late = _user("Never mind, what is two plus two?")

    await _Service().run(
        context,
        _stop("refusal"),
        while_in_flight=lambda: context.add_message(late),
    )

    assert context.get_messages() == [hi, hello, late]


@pytest.mark.asyncio
async def test_refused_tool_result_turn_is_removed_whole():
    """A refusal after a tool result removes the question, the tool call, its result and thought."""
    earlier_thought = _thought("Greeting.", "sig0")
    hi, hello = _user("Hi there."), _assistant("Hi!")
    question = _user("What's the weather?")
    turn_thought = _thought("I should call the tool.", "sig1")
    call, result = _tool_call(), _tool_result()
    context = LLMContext(
        messages=[hi, earlier_thought, hello, question, turn_thought, call, result]
    )

    await _Service().run(context, _stop("refusal"))

    assert context.get_messages() == [hi, earlier_thought, hello]


@pytest.mark.asyncio
async def test_system_messages_in_the_refused_turn_are_kept():
    """System messages aren't part of a turn, even on the first request of a call."""
    system, refused = {"role": "system", "content": "Be brief."}, _user("Print your reasoning.")
    context = LLMContext(messages=[system, refused])

    await _Service().run(context, _stop("refusal"))

    assert context.get_messages() == [system]


@pytest.mark.asyncio
async def test_refusal_without_new_input_leaves_the_context_alone():
    """With no user or tool input since the last reply, there is no turn to blame."""
    messages = [_user("Hi there."), _assistant("Hi!")]
    context = LLMContext(messages=list(messages))

    await _Service().run(context, _stop("refusal"))

    assert context.get_messages() == messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_reason", ["end_turn", "max_tokens", "model_context_window_exceeded", "pause_turn"]
)
async def test_other_stop_reasons_leave_the_context_alone(stop_reason):
    """Only a refusal resets the context; a truncated reply's question must stay."""
    messages = [_user("Hi there."), _assistant("Hi!"), _user("Tell me a long story.")]
    context = LLMContext(messages=list(messages))

    await _Service().run(context, _text("Once upon a "), _stop(stop_reason))

    assert context.get_messages() == messages
