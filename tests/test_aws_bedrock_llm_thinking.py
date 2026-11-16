#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AWSBedrockLLMService's handling of model reasoning.

Reasoning arrives either as ``reasoningContent`` deltas (Claude, Nova 2) or, for
Amazon Nova (v1) models, in the answer text wrapped in ``<thinking>`` tags before
a tool call. Either way it is emitted as thoughts and must not reach TTS.
"""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import (
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService

# Text deltas from a real us.amazon.nova-lite-v1:0 ConverseStream response to
# "Get the weather in Paris and in Tokyo." with a get_weather tool (trimmed in
# the middle). Nova sends one token per delta, so the tags are split.
NOVA_THINKING_DELTAS = [
    "<thinking",
    ">",
    "The",
    " User",
    " wants",
    " to",
    " know",
    " the",
    " weather",
    " in",
    " both",
    " cities",
    ".</",
    "thinking",
    ">\n",
]


def _text_events(deltas: list[str]) -> list[dict]:
    events: list[dict] = [{"messageStart": {"role": "assistant"}}]
    events += [
        {"contentBlockDelta": {"delta": {"text": d}, "contentBlockIndex": 0}} for d in deltas
    ]
    events.append({"contentBlockStop": {"contentBlockIndex": 0}})
    return events


def _tool_events() -> list[dict]:
    return [
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"city":"Paris"}'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 409, "outputTokens": 76, "totalTokens": 485}}},
    ]


async def _run(events: list[dict]):
    service = AWSBedrockLLMService(
        aws_access_key="test",
        aws_secret_key="test",
        aws_region="us-east-1",
        settings=AWSBedrockLLMService.Settings(model="us.amazon.nova-lite-v1:0"),
    )
    frames = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *args, **kwargs: frames.append(frame))
    service.run_function_calls = AsyncMock()

    async def stream():
        for event in events:
            yield event

    context = LLMContext()
    context.add_message({"role": "user", "content": "Get the weather in Paris and in Tokyo."})
    with patch.object(
        service, "_create_converse_stream", AsyncMock(return_value={"stream": stream()})
    ):
        await service._process_context(context)
    return service, frames


@pytest.mark.asyncio
async def test_nova_thinking_tags_are_emitted_as_thoughts_not_text():
    service, frames = await _run(_text_events(NOVA_THINKING_DELTAS) + _tool_events())

    spoken = "".join(f.text for f in frames if isinstance(f, LLMTextFrame))
    thought = "".join(f.text for f in frames if isinstance(f, LLMThoughtTextFrame))
    assert "thinking" not in spoken
    assert "User wants" not in spoken
    assert thought == "The User wants to know the weather in both cities."
    assert sum(isinstance(f, LLMThoughtStartFrame) for f in frames) == 1
    assert sum(isinstance(f, LLMThoughtEndFrame) for f in frames) == 1

    (calls,) = service.run_function_calls.call_args.args
    assert [(c.function_name, c.arguments) for c in calls] == [("get_weather", {"city": "Paris"})]


@pytest.mark.asyncio
async def test_text_after_thinking_block_is_still_spoken():
    deltas = NOVA_THINKING_DELTAS + ["Let", " me", " check", "."]
    _, frames = await _run(_text_events(deltas) + _tool_events())

    spoken = "".join(f.text for f in frames if isinstance(f, LLMTextFrame))
    assert spoken.strip() == "Let me check."


@pytest.mark.asyncio
async def test_text_without_thinking_tags_is_unchanged():
    deltas = ["Sure", "! I'll", " fetch", " the", " weather", "."]
    _, frames = await _run(_text_events(deltas) + _tool_events())

    assert (
        "".join(f.text for f in frames if isinstance(f, LLMTextFrame))
        == "Sure! I'll fetch the weather."
    )
    assert not any(isinstance(f, LLMThoughtTextFrame) for f in frames)


@pytest.mark.asyncio
async def test_short_text_that_is_a_tag_prefix_is_flushed_at_the_end():
    _, frames = await _run(_text_events(["<"]) + _tool_events())

    assert "".join(f.text for f in frames if isinstance(f, LLMTextFrame)) == "<"


# Reasoning deltas from a real us.anthropic.claude-haiku-4-5 ConverseStream
# response with extended thinking enabled (trimmed in the middle).
CLAUDE_REASONING_EVENTS = [
    {"messageStart": {"role": "assistant"}},
    *[
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": t}}, "contentBlockIndex": 0}}
        for t in ["The user is asking for", " the weather in Paris.", ""]
    ],
    {
        "contentBlockDelta": {
            "delta": {"reasoningContent": {"signature": "sig"}},
            "contentBlockIndex": 0,
        }
    },
    {"contentBlockStop": {"contentBlockIndex": 0}},
]


@pytest.mark.asyncio
async def test_reasoning_content_is_emitted_as_a_thought_for_the_context():
    service, frames = await _run(CLAUDE_REASONING_EVENTS + _tool_events())

    (start,) = [f for f in frames if isinstance(f, LLMThoughtStartFrame)]
    (end,) = [f for f in frames if isinstance(f, LLMThoughtEndFrame)]
    thought = "".join(f.text for f in frames if isinstance(f, LLMThoughtTextFrame))
    assert start.append_to_context
    assert start.llm == "aws"
    assert end.signature == "sig"
    assert thought == "The user is asking for the weather in Paris."
    assert not any(isinstance(f, LLMTextFrame) for f in frames)

    (calls,) = service.run_function_calls.call_args.args
    assert [c.function_name for c in calls] == ["get_weather"]


@pytest.mark.asyncio
async def test_reasoning_cut_short_is_closed_without_a_signature():
    async def failing_stream():
        for event in CLAUDE_REASONING_EVENTS[:2]:
            yield event
        raise RuntimeError("stream dropped")

    service = AWSBedrockLLMService(
        aws_access_key="test",
        aws_secret_key="test",
        aws_region="us-east-1",
        settings=AWSBedrockLLMService.Settings(model="us.anthropic.claude-haiku-4-5"),
    )
    frames = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *args, **kwargs: frames.append(frame))
    context = LLMContext()
    context.add_message({"role": "user", "content": "What's the weather in Paris?"})
    with patch.object(
        service, "_create_converse_stream", AsyncMock(return_value={"stream": failing_stream()})
    ):
        await service._process_context(context)

    (end,) = [f for f in frames if isinstance(f, LLMThoughtEndFrame)]
    assert end.signature is None


@pytest.mark.asyncio
async def test_each_text_block_can_open_with_a_thought():
    def text_block(index: int, deltas: list[str]) -> list[dict]:
        return [
            {"contentBlockDelta": {"delta": {"text": d}, "contentBlockIndex": index}}
            for d in deltas
        ] + [{"contentBlockStop": {"contentBlockIndex": index}}]

    def tool_block(index: int, city: str) -> list[dict]:
        return [
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": f"tooluse_{index}", "name": "get_weather"}},
                    "contentBlockIndex": index,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": f'{{"city":"{city}"}}'}},
                    "contentBlockIndex": index,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": index}},
        ]

    events = [
        {"messageStart": {"role": "assistant"}},
        *text_block(0, ["<thinking>", "Paris first.", "</thinking>"]),
        *tool_block(1, "Paris"),
        *text_block(2, ["<thinking>", "Now Tokyo.", "</thinking>\n"]),
        *tool_block(3, "Tokyo"),
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    service, frames = await _run(events)

    thoughts = []
    for f in frames:
        if isinstance(f, LLMThoughtStartFrame):
            thoughts.append("")
        elif isinstance(f, LLMThoughtTextFrame):
            thoughts[-1] += f.text
    assert thoughts == ["Paris first.", "Now Tokyo."]
    assert sum(isinstance(f, LLMThoughtEndFrame) for f in frames) == 2
    assert not any(isinstance(f, LLMTextFrame) for f in frames)

    (calls,) = service.run_function_calls.call_args.args
    assert [c.arguments["city"] for c in calls] == ["Paris", "Tokyo"]
