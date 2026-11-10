#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Explicit history edits preserve the final result of a tool that reported progress."""

import asyncio
import copy

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
)
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from tests.rime_realtime_peer import conversation, until
from tests.test_rime_realtime_parity import call
from tests.test_rime_realtime_service import sent


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_kind", ["update", "transform"])
@pytest.mark.parametrize("shared_first", [False, True])
async def test_replacement_preserves_final_tool_result(edit_kind, shared_first):
    finish = asyncio.Event()
    executions = []

    async def lookup(params):
        executions.append(True)
        await params.result_callback(
            "Working",
            properties=FunctionCallResultProperties(is_final=False, run_llm=False),
        )
        await finish.wait()
        await params.result_callback(
            "Finished", properties=FunctionCallResultProperties(run_llm=False)
        )

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        _,
        upstream,
        __,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        response = await peer.start()
        # Generated calls need a server record so deletion retires their results.
        peer.history.append(
            {
                "id": "item-lookup-1",
                "type": "function_call",
                "call_id": "lookup-1",
                "name": "lookup",
                "arguments": "{}",
            }
        )
        await call(peer, response)
        await peer.end(response)
        await until(lambda: len(sent(peer, "conversation.item.create")) == 1)
        finish.set()
        await until(lambda: len(sent(peer, "conversation.item.create")) == 3)
        await until(lambda: any("Finished" in str(m) for m in context.get_messages()))
        before = copy.deepcopy(peer.history)
        assert [item["output"] for item in before if "output" in item] == [
            '"Working"',
            '"Finished"',
        ]

        if shared_first:
            await worker.queue_frame(LLMContextFrame(context))
            response = await peer.start(await peer.next())
            await peer.end(response)
            assert peer.history == before

        if edit_kind == "update":
            frame = LLMMessagesUpdateFrame(
                messages=copy.deepcopy(context.get_messages()), run_llm=True
            )
        else:
            frame = LLMMessagesTransformFrame(transform=copy.deepcopy, run_llm=True)
        await worker.queue_frame(frame)
        response = await peer.start(await peer.next())
        await peer.end(response)

        assert [(item["call_id"], item["output"]) for item in peer.history if "output" in item] == [
            ("lookup-1", '"Finished"')
        ]
        assert executions == [True]
        assert service.is_usable
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
