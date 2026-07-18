#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests: the reasoning gate inside the real ``_process_context``.

Replays the wf15 run-2002 leak shape (a textual ``<thought`` chain-of-thought
streamed per-token by vLLM) through the actual stream-consumption loop and
asserts on what the pipeline would actually speak/commit:

- a closed CoT block never reaches ``push_frame`` — only the reply does;
- a reasoning-only completion counts as empty and triggers the retry;
- a reasoning-only completion WITH a structured tool call runs the call
  immediately (nothing audible will fire BotStoppedSpeakingFrame to flush a
  deferred call).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _chunk(content=None, tool_call=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call)
    return SimpleNamespace(
        usage=None, model=None, choices=[SimpleNamespace(delta=delta)]
    )


def _tool_call_chunk(name=None, arguments=None, call_id=None, index=0):
    tc = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return _chunk(tool_call=[tc])


async def _stream(chunks):
    for c in chunks:
        yield c


class _FakeStream:
    """Mimics the openai AsyncStream: async-iterable + close()."""

    def __init__(self, chunks):
        self._it = _stream(chunks)

    def __aiter__(self):
        return self._it

    async def close(self):
        await self._it.aclose()


def _make_service(streams):
    """Real OpenAILLMService with a scripted sequence of completion streams."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(model="test-model")
        )
    calls = {"n": 0}

    async def fake_get_chat_completions(context):
        i = min(calls["n"], len(streams) - 1)
        calls["n"] += 1
        return _FakeStream(streams[i])

    service.get_chat_completions = fake_get_chat_completions
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame
    service.run_function_calls = AsyncMock()
    return service, pushed, calls


def _pushed_text(pushed):
    return "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))


# The wf15 run-2002 leak, per-token as vLLM streams a textual pseudo-tag.
RUN_2002_COT = [
    "<", "thought", "\n", "Thinking", " Process", ":", "\n\n", "1", ".",
    "  **", "Analyze", " the", " user", " input", ":**", " The", " user",
    " said", ' "כן"', " confirming", " his", " identity", ".",
]
RUN_2002_REPLY = ["\n", "מע", "ולה", ", ", "רשמ", "תי", ". ", "מתכ", "וון", " להצביע?"]


@pytest.mark.asyncio
async def test_run_2002_replay_speaks_only_the_reply():
    chunks = [_chunk(c) for c in RUN_2002_COT + ["\n</", "thought", ">"] + RUN_2002_REPLY]
    service, pushed, calls = _make_service([chunks])
    await service._process_context(LLMContext(messages=[{"role": "user", "content": "כן"}]))
    text = _pushed_text(pushed)
    assert text.strip() == "מעולה, רשמתי. מתכוון להצביע?"
    for marker in ("Thinking", "Analyze", "<thought", "thought>"):
        assert marker not in text
    assert calls["n"] == 1  # healthy turn: no retry


@pytest.mark.asyncio
async def test_reasoning_only_completion_retries_then_speaks():
    cot_only = [_chunk(c) for c in RUN_2002_COT]  # never closes, no reply
    healthy = [_chunk(c) for c in ["שלום", ", ", "מה שלומך?"]]
    service, pushed, calls = _make_service([cot_only, healthy])
    await service._process_context(LLMContext(messages=[{"role": "user", "content": "היי"}]))
    assert calls["n"] == 2  # dead turn regenerated exactly once
    assert _pushed_text(pushed) == "שלום, מה שלומך?"


@pytest.mark.asyncio
async def test_reasoning_only_with_tool_call_runs_immediately():
    chunks = [_chunk(c) for c in RUN_2002_COT]
    chunks += [
        _tool_call_chunk(name="transition_to_voting_intent", call_id="c1"),
        _tool_call_chunk(arguments='{"is_target_contact": true}'),
    ]
    service, pushed, calls = _make_service([chunks])
    ran = {}

    async def fake_run(function_calls):
        ran["calls"] = function_calls

    service.run_function_calls = fake_run
    await service._process_context(LLMContext(messages=[{"role": "user", "content": "כן"}]))
    assert calls["n"] == 1  # the tool call means the turn was NOT empty — no retry
    assert _pushed_text(pushed).strip() == ""  # CoT never spoken
    # Nothing audible → the call must run NOW, not wait for BotStoppedSpeaking.
    assert not service._pending_function_calls
    assert [c.function_name for c in ran["calls"]] == ["transition_to_voting_intent"]
    assert ran["calls"][0].arguments == {"is_target_contact": True}
