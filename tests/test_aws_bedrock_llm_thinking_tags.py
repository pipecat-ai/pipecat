#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AWSBedrockLLMService's handling of inline ``<thinking>`` tags.

Amazon Nova (v1) models put their chain of thought in the answer text, wrapped
in ``<thinking>`` tags, before a tool call. That text must not reach TTS.
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
