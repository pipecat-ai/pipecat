#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for BackendLLMWorker and run_backend_job.

A scripted LLM service stands in for the backend model: each LLMContextFrame
plays the next scripted response (text and/or function calls), so the tests
exercise the real aggregators, tool loop and job plumbing under a
WorkerRunner.
"""

import asyncio
from typing import Any

import pytest

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm import BackendLLMWorker, run_backend_job
from pipecat.workers.llm.backend_llm_worker import render_backend_request
from pipecat.workers.runner import WorkerRunner


class _ScriptedLLM(LLMService):
    """Plays one scripted response per LLMContextFrame.

    A script step is ``("text", str)`` or ``("call", name, call_id, args)``.
    ``settle_secs`` holds the response open after issuing its calls so a fast
    tool can return before the response ends.
    """

    def __init__(self, runs: list[list[tuple]], *, settle_secs: float = 0.0):
        super().__init__(
            settings=LLMSettings(
                model="scripted",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            )
        )
        self._runs = list(runs)
        self._settle_secs = settle_secs
        self.contexts_seen: list[list[Any]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        self.contexts_seen.append(list(frame.context.get_messages()))
        steps = self._runs.pop(0) if self._runs else []
        await self.push_frame(LLMFullResponseStartFrame())
        calls = []
        for step in steps:
            if step[0] == "text":
                await self.push_frame(LLMTextFrame(step[1]))
            else:
                _, name, call_id, args = step
                calls.append(
                    FunctionCallFromLLM(
                        context=frame.context,
                        tool_call_id=call_id,
                        function_name=name,
                        arguments=args,
                    )
                )
        if calls:
            await self.run_function_calls(calls)
            if self._settle_secs:
                await asyncio.sleep(self._settle_secs)
        await self.push_frame(LLMFullResponseEndFrame())


async def get_weather(params: FunctionCallParams, location: str):
    """Get the weather.

    Args:
        location: The city.
    """
    await params.result_callback({"temp": 62, "conditions": "rain"})


async def _run_backend(
    llm: _ScriptedLLM, *, task: str, messages: list[dict[str, Any]] | None = None
) -> tuple[str, list[tuple[str, str]], BackendLLMWorker]:
    """Run one delegated task against ``llm`` under a WorkerRunner."""
    backend = BackendLLMWorker(
        llm=llm,
        name="backend",
        context=LLMContext([{"role": "system", "content": "You are the backend."}], [get_weather]),
    )
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, backend)

    updates: list[tuple[str, str]] = []
    result: dict[str, str] = {}

    async def on_update(kind: str, text: str):
        updates.append((kind, text))

    async def body():
        try:
            result["text"] = await run_backend_job(
                requester,
                "backend",
                task=task,
                messages=messages,
                on_update=on_update,
                timeout_secs=10,
            )
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)
    return result["text"], updates, backend


@pytest.mark.asyncio
async def test_backend_runs_a_tool_loop_and_streams_each_response():
    llm = _ScriptedLLM(
        [
            [("text", "Let me check."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "It's 62 and raining in Seattle.")],
        ]
    )

    text, updates, backend = await _run_backend(
        llm,
        task="What's the weather in Seattle?",
        messages=[
            {"role": "user", "content": "what's the weather in seattle"},
            {"role": "assistant", "content": "Let me find out."},
        ],
    )

    assert text == "It's 62 and raining in Seattle."
    assert updates == [
        ("text", "Let me check."),
        ("text", "It's 62 and raining in Seattle."),
    ]

    # The backend saw the rendered request first, then the tool result.
    first_request = llm.contexts_seen[0][-1]
    assert first_request["role"] == "user"
    assert first_request["content"] == (
        "Voice conversation so far:\n"
        "USER: what's the weather in seattle\n"
        "ASSISTANT: Let me find out.\n"
        "\n"
        "Task from the voice assistant: What's the weather in Seattle?"
    )
    assert len(llm.contexts_seen) == 2
    roles = [m.get("role") for m in backend.context.get_messages()]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_fast_tool_result_before_response_end_does_not_finish_the_run_early():
    llm = _ScriptedLLM(
        [
            [("text", "Checking."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "Rain, 62 degrees.")],
        ],
        settle_secs=0.1,
    )

    text, updates, _ = await _run_backend(llm, task="Weather in Seattle?")

    assert text == "Rain, 62 degrees."
    assert updates == [("text", "Checking."), ("text", "Rain, 62 degrees.")]


@pytest.mark.asyncio
async def test_tool_only_response_sends_no_update_and_still_completes():
    llm = _ScriptedLLM(
        [
            [("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "It's raining.")],
        ]
    )

    text, updates, _ = await _run_backend(llm, task="Weather?")

    assert text == "It's raining."
    assert updates == [("text", "It's raining.")]


@pytest.mark.asyncio
async def test_follow_up_tasks_render_only_the_turns_since_the_last_one():
    llm = _ScriptedLLM([[("text", "First.")], [("text", "Second.")]])
    backend = BackendLLMWorker(llm=llm, name="backend", context=LLMContext())
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, backend)

    async def body():
        try:
            await run_backend_job(
                requester, "backend", task="One", messages=[{"role": "user", "content": "one"}]
            )
            await run_backend_job(
                requester, "backend", task="Two", messages=[{"role": "user", "content": "two"}]
            )
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)

    requests = [m["content"] for m in llm.contexts_seen[1] if m.get("role") == "user"]
    assert requests[0].startswith("Voice conversation so far:\nUSER: one")
    assert requests[1].startswith("Voice conversation since your last task:\nUSER: two")


def test_render_backend_request_omits_the_transcript_when_there_are_no_turns():
    assert render_backend_request("Do it", [], first=True) == "Task from the voice assistant: Do it"
