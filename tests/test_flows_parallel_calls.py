#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parallel function calls with a transition among them, through a real pipeline.

The LLM answers the user's turn with an edge function (moves to another node) and a
node function (stays) in parallel. Whichever finishes first, the caller should get
one answer, from the next node.
"""

import asyncio
import unittest

from pipecat.flows import FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat.frames.frames import (
    EndFrame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner


class FakeLLM(LLMService):
    """Answers a user turn with ``calls`` in parallel, anything else with a reply."""

    def __init__(self, calls: list[str]):
        super().__init__()
        self.calls = calls
        self.flow_manager: FlowManager | None = None
        self.replies: list[str] = []  # the node each reply was given in

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        if frame.context.get_messages()[-1].get("role") == "user":
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(name, f"call-{i}", {}, frame.context)
                    for i, name in enumerate(self.calls)
                ]
            )
            return
        self.replies.append(self.flow_manager.current_node)
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("reply"))
        await self.push_frame(LLMFullResponseEndFrame())


async def run_turn(calls: list[str], delays: dict[str, float]) -> tuple[list[str], str]:
    """The replies to one user turn, and the node the flow ends up in."""
    llm = FakeLLM(calls)
    aggregator = LLMContextAggregatorPair(LLMContext())
    worker = PipelineWorker(
        Pipeline([aggregator.user(), llm, aggregator.assistant()]),
        params=PipelineParams(),
        enable_rtvi=False,
    )

    def function(name: str, next_node=None) -> FlowsFunctionSchema:
        async def handler(args, flow_manager):
            await asyncio.sleep(delays.get(name, 0))
            return {"status": "ok"}, next_node() if next_node else None

        return FlowsFunctionSchema(name, name, {}, [], handler=handler)

    send_email = function("send_email")
    add_note = function("add_note")
    booking: NodeConfig = {
        "name": "booking",
        "task_messages": [{"role": "developer", "content": "Book."}],
        "functions": [send_email],
    }
    check_availability = function("check_availability", lambda: booking)
    greeting: NodeConfig = {
        "name": "greeting",
        "task_messages": [{"role": "developer", "content": "Greet."}],
        "functions": [check_availability, send_email, add_note],
    }

    llm.flow_manager = FlowManager(worker=worker, llm=llm, context_aggregator=aggregator)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.sleep(0.2)
    await llm.flow_manager.initialize(greeting)
    await asyncio.sleep(0.3)
    llm.replies.clear()  # the greeting
    await worker.queue_frames(
        [LLMMessagesAppendFrame([{"role": "user", "content": "Tuesday please"}], run_llm=True)]
    )
    await asyncio.sleep(0.8)
    await worker.queue_frame(EndFrame())
    await running
    return llm.replies, llm.flow_manager.current_node


class TestParallelCallsWithTransition(unittest.IsolatedAsyncioTestCase):
    async def test_node_function_finishes_first(self):
        replies, node = await run_turn(
            ["check_availability", "send_email"], {"check_availability": 0.2}
        )
        self.assertEqual((replies, node), (["booking"], "booking"))

    async def test_edge_function_finishes_first(self):
        replies, node = await run_turn(["check_availability", "send_email"], {"send_email": 0.2})
        self.assertEqual((replies, node), (["booking"], "booking"))

    async def test_parallel_node_functions_answer_once(self):
        replies, node = await run_turn(["send_email", "add_note"], {"add_note": 0.2})
        self.assertEqual((replies, node), (["greeting"], "greeting"))

    async def test_a_single_node_function_still_gets_an_answer(self):
        replies, node = await run_turn(["send_email"], {})
        self.assertEqual((replies, node), (["greeting"], "greeting"))
